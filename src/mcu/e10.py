"""E-10: one AdamW grid + int8-val selection for every 8051 model in the matched-accuracy comparison.

Protocol: results/T1m/e10/PROTOCOL.md (written before any run).
  grid   lr {3e-4, 1e-3, 3e-3} (+1e-2 if 3e-3 is selected) x wd {0, 1e-2} x seeds 1-3, AdamW, batch 128, 25 epochs
  select 3-seed mean int8 val accuracy (deployed quantiser, bit-exact integer model); sensitivity: fp32 val_acc_last
Usage:
  python e10.py grid [--lrs ...] [--models ...] [--jobs 8]   # train + quantise every missing (model, lr, wd, seed)
  python e10.py select                                       # per-cell table, selection, extension list
  python e10.py verify                                       # seed-1 selected: gcc bit-exact + SDCC flash
  python e10.py report                                       # results CSV + matched-accuracy comparison
Outputs: results/T1m/e10/{runs/,cells.csv,results.csv,matched.csv,build/}
"""
import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, ensure, load_fmnist_u8  # noqa: E402

E10 = f"{EXP}/e10"
RUNS = os.environ.get("E10_RUNS", f"{E10}/runs")  # CNN variants (PROTOCOL_cnnvar.md): results/T1m/e10/cnnvar/runs
SFX = os.environ.get("E10_SFX", "")  # verify/ucsim output suffix for a second concurrent process (merged afterwards)
NONCNN_E10 = ["dann_d16_s64", "dann_prng_d16_s128", "dann_prng_d20_s128", "mlp_h48", "mlp_h64"]
CNNS_E10 = ["cnn_c2_4", "cnn_c3_6", "cnn_c4_8", "cnn_c6_12", "cnn_c8_16"]
# E-10b (2026-10-25): same protocol for every other deployable Table-3 row (T1m/results.csv fits=1)
NONCNN_E10B = ["dann_d8_s32", "dann_prng_d8_s32", "dann_prng_d8_s64", "dann_d8_s64", "mlp_h16", "dann_prng_d16_s64",
               "mlp_h32", "dann_d24_s64", "dann_prng_d24_s64", "dann_prng_d32_s96", "dann_prng_d24_s128", "mlp_h72"]
CNNS_E10B = ["cnn_c16_32", "cnn_c32_64"]
NONCNN = NONCNN_E10 + NONCNN_E10B
CNNS = CNNS_E10 + CNNS_E10B          # cycle order c2_4 < c3_6 < c4_8 < c6_12 < c8_16 < c16_32 < c32_64
MODELS = NONCNN + CNNS
DANNS = ["dann_d16_s64", "dann_prng_d16_s128", "dann_prng_d20_s128"]
BASE_LRS = [3e-4, 1e-3, 3e-3]
EXT_LR = 1e-2
WDS = [0.0, 1e-2]
SEEDS = [1, 2, 3]
EPOCHS = 25
T_RULE = 2.48
CNN_K = 0.854   # E-10b: board/ucsim of cnn_c8_16, as in cnn_results.csv / Table 3 (E-10 used 0.853 = mean of 3 CNNs)
# never-board-measured dANN / MLP rows (E-10b): board/ucsim measured on the same kind (board_summary.csv), labelled estimate
KIND_K = {"official": 0.890, "prng": 0.9115, "mlp": 0.811}
F_MHZ = 35.004

_DATA = None


def tag_of(model, lr, wd, seed):
    return f"{model}_lr{lr:g}_wd{wd:g}_t{seed}"


def parse(model):
    if model.startswith(("cnnpool_", "cnns4_", "cnnds_")):  # cnnvar (PROTOCOL_cnnvar.md)
        return "cnnvar", dict(name=model), ""
    if model.startswith("cnn_"):
        c1, c2 = model[5:].split("_")
        return "cnn", dict(c1=int(c1), c2=int(c2)), ""
    if model.startswith("mlp_h"):
        return "mlp", dict(h=int(model[5:])), ""
    mask = "prng" if model.startswith("dann_prng_") else "official"
    d, s = model.split("_d")[1].split("_s")
    return "dann", dict(d=int(d), s=int(s)), mask


def build_model(model, seed):
    import torch
    from train import Net, SYN
    from common import dann_masks
    from prng_rf import prng_masks
    kind, cfg, mask = parse(model)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m1 = m2 = None
    if kind == "cnnvar":
        from cnnvar import build_net
        net = build_net(model)
    elif kind == "cnn":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "t6"))
        from dann_g import SmallCNN
        net = SmallCNN(cfg["c1"], cfg["c2"])
    elif kind == "dann":
        m1, m2 = prng_masks(cfg["d"], cfg["s"], seed) if mask == "prng" else dann_masks(cfg["d"], cfg["s"], SYN, seed)
        net = Net(cfg["d"] * cfg["s"], cfg["s"], m1, m2)
    else:
        net = Net(cfg["h"], cfg["h"])
    return net.to("cpu"), kind, cfg, mask, m1, m2


def train_eval(job):
    """One (model, lr, wd, seed): AdamW training, fp32 val/test, int8 quantisation, int8 val/test."""
    import torch
    import torch.nn.functional as F
    global _DATA
    model, lr, wd, seed = job
    tag = tag_of(model, lr, wd, seed)
    if os.path.exists(f"{RUNS}/{tag}.json"):
        return tag, "skip"
    torch.set_num_threads(int(os.environ.get("E10_THREADS", "2")))
    if _DATA is None:
        _DATA = load_fmnist_u8()
    xtr_u8, ytr = _DATA["train"]
    xte_u8, yte = _DATA["test"]
    perm = np.random.default_rng(seed).permutation(len(xtr_u8))
    nval = int(0.1 * len(xtr_u8))
    tr_idx, va_idx = perm[:-nval], perm[-nval:]
    net, kind, cfg, mask, m1, m2 = build_model(model, seed)
    X = torch.tensor(xtr_u8, dtype=torch.float32) / 255.0
    Y = torch.tensor(ytr)
    xtr, ytr_t, xva, yva = X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx]
    xte = torch.tensor(xte_u8, dtype=torch.float32) / 255.0
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)

    def acc(x, y):
        net.eval()
        with torch.no_grad():
            return float((torch.cat([net(x[i:i + 5000]) for i in range(0, len(x), 5000)]).argmax(1) == y).float().mean())

    t0 = time.time()
    hist = []
    for ep in range(EPOCHS):
        net.train()
        p = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), 128):
            b = p[i:i + 128]
            loss = F.cross_entropy(net(xtr[b]), ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if kind == "dann":  # AdamW decoupled decay multiplies by (1 - lr*wd): masked zeros stay zero; check anyway
            pass
        hist.append(acc(xva, yva))
    te_fp32 = acc(xte, torch.tensor(yte))
    if kind == "dann":
        assert net.l1.weight.detach()[net.masks[0] == 0].abs().max().item() == 0.0
        assert net.l2.weight.detach()[net.masks[1] == 0].abs().max().item() == 0.0
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    xv, yv = xtr_u8[va_idx], ytr[va_idx]
    if kind == "cnnvar":
        from cnnvar import quantize_var, ref_forward_var
        q = quantize_var(model, sd, xv, yv)
        val_i8 = float((ref_forward_var(q, xv).argmax(1) == yv).mean())
        te_i8 = float((ref_forward_var(q, xte_u8).argmax(1) == yte).mean())
        np.savez_compressed(f"{RUNS}/{tag}.npz", **sd)
    elif kind == "cnn":
        from cnn_t1m import quantize_cnn, ref_forward_cnn
        q = quantize_cnn(sd, xv, yv)
        val_i8 = float((ref_forward_cnn(q, xv).argmax(1) == yv).mean())
        te_i8 = float((ref_forward_cnn(q, xte_u8).argmax(1) == yte).mean())
        np.savez_compressed(f"{RUNS}/{tag}.npz", **sd)
    else:
        from quant import quantize, ref_forward
        m = dict(W1=sd["l1.weight"].astype(np.float32), b1=sd["l1.bias"].astype(np.float32),
                 W2=sd["l2.weight"].astype(np.float32), b2=sd["l2.bias"].astype(np.float32),
                 W3=sd["out.weight"].astype(np.float32), b3=sd["out.bias"].astype(np.float32),
                 val_idx=va_idx.astype(np.int32))
        if m1 is not None:
            m.update(M1=m1, M2=m2)
        q = quantize(m, xv, yv)
        val_i8 = float((ref_forward(q, xv).argmax(1) == yv).mean())
        te_i8 = float((ref_forward(q, xte_u8).argmax(1) == yte).mean())
        np.savez_compressed(f"{RUNS}/{tag}.npz", **m)
    meta = dict(tag=tag, model=model, kind=kind, cfg=cfg, mask_source=mask, lr=lr, wd=wd, seed=seed, epochs=EPOCHS,
                val_fp32_last=hist[-1], val_hist=hist, test_fp32=te_fp32, val_int8=val_i8, test_int8=te_i8,
                sh1=int(q["sh1"]), sh2=int(q["sh2"]), sh_off=list(q["sh_off"]), train_s=round(time.time() - t0, 1),
                torch=torch.__version__, threads=torch.get_num_threads())
    if "shifts" in q:  # cnnvar: all requant shifts / offsets
        meta.update(shifts=[int(v) for v in q["shifts"]], offsets=[int(v) for v in q["offsets"]])
    json.dump(meta, open(f"{RUNS}/{tag}.json", "w"), indent=1)
    return tag, f"val_i8 {val_i8:.4f} test_i8 {te_i8:.4f} ({meta['train_s']} s)"


def cmd_grid(a):
    import multiprocessing as mp
    ensure(RUNS)
    jobs = [(m, lr, wd, s) for s in SEEDS for lr in a.lrs for wd in WDS for m in a.models
            if not os.path.exists(f"{RUNS}/{tag_of(m, lr, wd, s)}.json")]
    # big models first so the pool tail is short
    order = {m: i for i, m in enumerate(["cnn_c32_64", "dann_prng_d32_s96", "dann_prng_d24_s128", "dann_prng_d20_s128",
                                         "dann_d24_s64", "dann_prng_d24_s64", "dann_prng_d16_s128", "cnn_c16_32",
                                         "dann_d16_s64", "dann_prng_d16_s64", "dann_d8_s64", "dann_prng_d8_s64",
                                         "dann_d8_s32", "dann_prng_d8_s32", "mlp_h72", "mlp_h64", "mlp_h48", "mlp_h32",
                                         "mlp_h16", "cnn_c8_16", "cnn_c6_12", "cnn_c4_8", "cnn_c3_6", "cnn_c2_4"])}
    jobs.sort(key=lambda j: order.get(j[0], 99))
    print(f"{len(jobs)} jobs, {a.jobs} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.jobs) as pool:
        for i, (tag, msg) in enumerate(pool.imap_unordered(train_eval, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {tag}: {msg}", flush=True)


def load_runs():
    import glob
    return [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]


def cells(runs):
    by = {}
    for r in runs:
        by.setdefault((r["model"], r["lr"], r["wd"]), []).append(r)
    out = []
    for (m, lr, wd), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        f = lambda k: [r[k] for r in rs]
        out.append(dict(model=m, lr=lr, wd=wd, n=len(rs), seeds=[r["seed"] for r in rs],
                        val_int8=st.mean(f("val_int8")), val_fp32=st.mean(f("val_fp32_last")),
                        test_int8=st.mean(f("test_int8")), test_int8_sd=st.stdev(f("test_int8")) if len(rs) > 1 else float("nan"),
                        test_int8_sd0=float(np.std(f("test_int8"))), test_fp32=st.mean(f("test_fp32")),
                        test_int8_seeds=f("test_int8"), val_int8_seeds=f("val_int8")))
    return out


def select(cs, model, key):
    c = [x for x in cs if x["model"] == model and x["n"] == 3]
    if not c:
        return None
    k2 = "val_fp32" if key == "val_int8" else "val_int8"
    best = max(round(x[key], 4) for x in c)
    tied = [x for x in c if round(x[key], 4) == best]
    tied.sort(key=lambda x: (-x[k2], x["lr"]))
    return tied[0]


def cmd_select(a):
    cs = cells(load_runs())
    with open(f"{E10}/cells.csv", "w", newline="") as fh:
        keys = ["model", "lr", "wd", "n", "val_int8", "val_fp32", "test_int8", "test_int8_sd", "test_int8_sd0", "test_fp32",
                "val_int8_seeds", "test_int8_seeds"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for c in cs:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()})
    ext = []
    for m in MODELS:
        s = select(cs, m, "val_int8")
        sf = select(cs, m, "val_fp32")
        if s is None:
            print(m, "incomplete")
            continue
        lrs = sorted({c["lr"] for c in cs if c["model"] == m})
        top = max(lrs)
        flag = "AT TOP" if s["lr"] == top else ""
        print(f"{m:20s} int8-val pick lr {s['lr']:g} wd {s['wd']:g} val {s['val_int8']:.4f} test {s['test_int8']:.4f}"
              f" | fp32-val pick lr {sf['lr']:g} wd {sf['wd']:g} test {sf['test_int8']:.4f}  grid lrs {lrs} {flag}")
        if s["lr"] == 3e-3 and top == 3e-3:
            ext.append(m)
    print("needs extension to lr 1e-2:", ext)
    json.dump(ext, open(f"{E10}/extension_models.json", "w"))


def board_info():
    """timing / old-protocol values per model. Board ms: measured (board_summary.csv, seed 1 of the first runs) where
    available, else estimate = ucsim cycles of the E-10/E-10b selected seed-1 build (ucsim.json; first-run ucsim before
    2026-10-25, kept in board_ms_preboard_estimate) x board/ucsim factor of the same kind / 35.004 MHz."""
    bs = {r["model"].rsplit("_t", 1)[0]: r for r in csv.DictReader(open(f"{EXP}/board_summary.csv"))}
    cr = {r["model"]: r for r in csv.DictReader(open(f"{EXP}/cnn/cnn_results.csv"))}
    tr = {r["model"]: r for r in csv.DictReader(open(f"{EXP}/results.csv"))}
    out = {}
    # E-10b board runs (2026-10-25): rows with build "e10b:<cell>" timed the E-10b seed-1 build. The old-protocol fields
    # (ucsim_cycles, flash_old, acc_*_old) stay from the first-run sources; only board_ms / board_cycles / source change.
    e10b = {m: b for m, b in bs.items() if b.get("build", "first-run").startswith("e10b:")}
    bs = {m: b for m, b in bs.items() if m not in e10b}
    for m in MODELS:
        if m in bs:
            b = bs[m]
            out[m] = dict(board_ms=float(b["board_ms"]), board_cycles=float(b["board_cycles_median"]),
                          ucsim_cycles=float(b["ucsim_cycles_mc"]), flash_old=int(b["flash_bytes"]),
                          board_ms_source=f"measured_seed1_{b['f_mhz']}MHz", acc_int8_old=float(b["acc_int8_mean3"]),
                          acc_int8_sd_old=float(b["acc_int8_std3"]))
        elif m in cr:
            c = cr[m]
            uc = float(c["cycles_ucsim_mc"])
            out[m] = dict(board_ms=round(uc * CNN_K / (F_MHZ * 1e3), 2), board_cycles=round(uc * CNN_K),
                          ucsim_cycles=uc, flash_old=int(c["flash_bytes"]), board_ms_source=f"estimate_ucsim_x{CNN_K}_{F_MHZ}MHz",
                          acc_int8_old=float(c["acc_int8"]), acc_int8_sd_old=float(c["acc_int8_std"]))
        else:
            t = tr[m]
            uc = float(t["cycles_ucsim_mc"])
            k = KIND_K["mlp" if t["kind"] == "mlp" else t["mask_source"]]
            out[m] = dict(board_ms=round(uc * k / (F_MHZ * 1e3), 2), board_cycles=round(uc * k),
                          ucsim_cycles=uc, flash_old=int(t["flash_bytes"]), board_ms_source=f"estimate_ucsim_x{k}_{F_MHZ}MHz",
                          acc_int8_old=float(t["acc_int8"]), acc_int8_sd_old=float(t["acc_int8_std"]))
        # never-board-measured rows are estimated from the ucsim cycles of the E-10/E-10b selected
        # seed-1 build (ucsim.json cycles_e10) x the same per-kind factor; ucsim_cycles stays the first-run value.
        if m not in bs and m not in e10b:
            ue = json.load(open(f"{E10}/ucsim.json"))[m]["cycles_e10"]
            k = CNN_K if m in cr else KIND_K["mlp" if tr[m]["kind"] == "mlp" else tr[m]["mask_source"]]
            out[m].update(board_ms_preboard_estimate=out[m]["board_ms"], board_ms=round(ue * k / (F_MHZ * 1e3), 2),
                          board_cycles=round(ue * k), board_ms_source=f"estimate_ucsimE10build_x{k}_{F_MHZ}MHz")
        if m in e10b:
            b = e10b[m]
            # also keep the E-10-build ucsim estimate (what results.csv showed before the last 12 board runs)
            ue = json.load(open(f"{E10}/ucsim.json"))[m]["cycles_e10"]
            k = CNN_K if m in cr else KIND_K["mlp" if tr[m]["kind"] == "mlp" else tr[m]["mask_source"]]
            out[m].update(board_ms_preboard_estimate=out[m]["board_ms"], board_ms=float(b["board_ms"]),
                          board_ms_e10build_estimate=round(ue * k / (F_MHZ * 1e3), 2),
                          board_cycles=float(b["board_cycles_median"]),
                          board_ms_source=f"measured_seed1_e10bbuild_{b['f_mhz']}MHz")
        if m in cr:  # CNN old sd is ddof=1 already (statistics.stdev)
            out[m]["old_cfg"] = cr[m]["cfg"]
        else:
            out[m]["old_cfg"] = "Adam lr1e-3 wd0 (train.py)"
            seeds = [float(r["acc_int8"]) for r in csv.DictReader(open(f"{EXP}/results_all.csv")) if r["model"] == m]
            out[m]["acc_int8_sd_old_ddof1"] = st.stdev(seeds)
    return out


def matched(accs, x, cnns, ddof):
    """smallest CNN (cycle order) with mean >= mean_x - 2.48 * max(sd)."""
    mx, sx = accs[x]["mean"], accs[x][ddof]
    for c in cnns:
        mc, sc = accs[c]["mean"], accs[c][ddof]
        thr = T_RULE * max(sx, sc)
        if mc >= mx - thr:
            return c, mx - mc, thr
    return None, None, None


def cmd_verify(a):
    """seed-1 model of the selected cell: emit C, gcc bit-exact on 10 000 test images, SDCC board build (flash)."""
    import run_t1m as R
    from gen_c import emit
    from quant import quantize, ref_forward
    from cnn_t1m import quantize_cnn, ref_forward_cnn, emit_cnn, SIM_XDATA
    data = load_fmnist_u8()
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    cs = cells(load_runs())
    out = {}
    jobs = [(m, sd) for m in (a.models or MODELS) for sd in a.seeds]
    vs = json.load(open(f"{E10}/verify_seeds{SFX}.json")) if os.path.exists(f"{E10}/verify_seeds{SFX}.json") else {}
    for m, seed in jobs:
        s = select(cs, m, "val_int8")
        tag = tag_of(m, s["lr"], s["wd"], seed)
        meta = json.load(open(f"{RUNS}/{tag}.json"))
        w = dict(np.load(f"{RUNS}/{tag}.npz"))
        vi = np.random.default_rng(seed).permutation(60000)[-6000:]
        bdir = ensure(f"{E10}/build/{tag}")
        extra = ["prng_rf"] if meta["mask_source"] == "prng" else []
        if meta["kind"] == "cnn":
            q = quantize_cnn(w, xtr[vi], ytr[vi])
            L = ref_forward_cnn(q, xte)
            fused = meta["cfg"]["c1"] * 196 + meta["cfg"]["c2"] * 49 + 40 > SIM_XDATA
            emit_cnn(q, f"{bdir}/model.h", tag, fused)
        else:
            q = quantize(w, xtr[vi], ytr[vi])
            L = ref_forward(q, xte)
            emit(q, meta["kind"], meta["cfg"], f"{bdir}/model.h", tag, M1=w.get("M1"),
                 mask_source=meta["mask_source"] or "official", model_seed=seed)
        assert abs(float((L.argmax(1) == yte).mean()) - meta["test_int8"]) < 1e-9
        subprocess.run(["gcc", "-O2", "-Wall", "-Wno-unused-function", f"-I{bdir}", f"-I{R.C_DIR}", f"{R.C_DIR}/infer.c",
                        f"{R.C_DIR}/pc_main.c"] + [f"{R.C_DIR}/{e}.c" for e in extra] + ["-o", f"{bdir}/pc_infer"], check=True)
        xte.tofile(f"{bdir}/test_x.u8")
        subprocess.run([f"{bdir}/pc_infer", f"{bdir}/test_x.u8", f"{bdir}/pc_out.bin"], check=True)
        raw = np.fromfile(f"{bdir}/pc_out.bin", np.uint8).reshape(-1, 41)
        pcl = raw[:, :40].copy().view("<i4").astype(np.int64)
        os.remove(f"{bdir}/test_x.u8")
        os.remove(f"{bdir}/pc_out.bin")
        vs.setdefault(m, {})[str(seed)] = dict(tag=tag, pc_bitexact_logits=f"{int((pcl == L).all(1).sum())}/{len(xte)}",
                                               pc_bitexact_class=f"{int((raw[:, 40] == L.argmax(1)).sum())}/{len(xte)}",
                                               test_int8_ref=float((L.argmax(1) == yte).mean()))
        json.dump(vs, open(f"{E10}/verify_seeds{SFX}.json", "w"), indent=1)
        print(m, seed, vs[m][str(seed)], flush=True)
        if seed != 1:  # flash / SDCC build: seed-1 model only (as E-10)
            continue
        uart = R.sdcc_build(bdir, ["uart_main"], "uart_35M", ["-DSTC8", f"-DF_OSC={R.F_OSC}UL"], R.XRAM_MAX, extra)
        out[m] = dict(tag=tag, pc_bitexact_logits=f"{int((pcl == L).all(1).sum())}/{len(xte)}",
                      pc_bitexact_class=f"{int((raw[:, 40] == L.argmax(1)).sum())}/{len(xte)}",
                      flash_bytes=uart["flash"], xdata_bytes=uart["xdata"], uart_build_ok=uart["ok"],
                      fits=bool(uart["ok"] and uart["flash"] <= R.FLASH_MAX and uart["xdata"] <= R.XRAM_MAX))
        print(m, out[m], flush=True)
        old = json.load(open(f"{E10}/verify{SFX}.json")) if os.path.exists(f"{E10}/verify{SFX}.json") else {}
        old.update({m: out[m]})
        json.dump(old, open(f"{E10}/verify{SFX}.json", "w"), indent=1)


def stored_ucsim():
    """cycles stored for the first-run seed-1 builds: T1m/results.csv (dANN/MLP, 20 images) and cnn_results.csv (CNN, 5)."""
    out = {r["model"]: float(r["cycles_ucsim_mc"]) for r in csv.DictReader(open(f"{EXP}/results.csv")) if r["cycles_ucsim_mc"]}
    out.update({r["model"]: float(r["cycles_ucsim_mc"]) for r in csv.DictReader(open(f"{EXP}/cnn/cnn_results.csv"))})
    return out


def cmd_ucsim(a):
    """ucsim cycles of the E-10/E-10b seed-1 selected builds on the same first test images as the stored values
    (dANN/MLP 20 images, CNN 5 images), bit-exact check against the Python int8 reference."""
    import run_t1m as R
    from cnn_t1m import SIM_XDATA
    from quant import quantize, ref_forward
    from cnn_t1m import quantize_cnn, ref_forward_cnn
    assert R.S51, "set S51"
    data = load_fmnist_u8()
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    cs = cells(load_runs())
    old = stored_ucsim()
    path = f"{E10}/ucsim{SFX}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for m in (a.models or MODELS):
        s = select(cs, m, "val_int8")
        tag = tag_of(m, s["lr"], s["wd"], 1)
        bdir = f"{E10}/build/{tag}"
        assert os.path.exists(f"{bdir}/model.h"), f"run verify first: {tag}"
        meta = json.load(open(f"{RUNS}/{tag}.json"))
        w = dict(np.load(f"{RUNS}/{tag}.npz"))
        vi = np.random.default_rng(1).permutation(60000)[-6000:]
        extra = ["prng_rf"] if meta["mask_source"] == "prng" else []
        n = 5 if meta["kind"] == "cnn" else 20
        L = ref_forward_cnn(quantize_cnn(w, xtr[vi], ytr[vi]), xte[:n]) if meta["kind"] == "cnn" else \
            ref_forward(quantize(w, xtr[vi], ytr[vi]), xte[:n])
        sim = R.sdcc_build(bdir, ["sim_main"], "sim", [], SIM_XDATA, extra)
        assert sim["ok"], sim["link_log"]
        cyc, n_ok, n_cls = [], 0, 0
        for i in range(n):
            c, lg, cy, cy0, _ = R.ucsim_run(bdir, xte[i])
            n_ok += int((lg == L[i]).all())
            n_cls += int(c == L[i].argmax())
            cyc.append(cy - cy0)
        r = dict(tag=tag, n=n, cycles_e10=float(np.mean(cyc)), cycles_min=int(min(cyc)), cycles_max=int(max(cyc)),
                 cycles_stored_old=old.get(m), ucsim_bitexact_logits=f"{n_ok}/{n}", ucsim_bitexact_class=f"{n_cls}/{n}")
        if old.get(m):
            r["rel_change"] = r["cycles_e10"] / old[m] - 1
        res[m] = r
        json.dump(res, open(path, "w"), indent=1)
        print(m, r, flush=True)


def cmd_report(a):
    cs = cells(load_runs())
    bi = board_info()
    ver = json.load(open(f"{E10}/verify{SFX}.json")) if os.path.exists(f"{E10}/verify{SFX}.json") else {}
    vs = json.load(open(f"{E10}/verify_seeds{SFX}.json")) if os.path.exists(f"{E10}/verify_seeds{SFX}.json") else {}
    uc = json.load(open(f"{E10}/ucsim{SFX}.json")) if os.path.exists(f"{E10}/ucsim{SFX}.json") else {}
    rows, accs, accs_f = [], {}, {}
    for m in MODELS:
        s = select(cs, m, "val_int8")
        sf = select(cs, m, "val_fp32")
        lrs = sorted({c["lr"] for c in cs if c["model"] == m})
        b = bi[m]
        v = ver.get(m, {})
        rows.append(dict(model=m, sel_lr=s["lr"], sel_wd=s["wd"], lr_grid=" ".join(f"{x:g}" for x in lrs),
                         sel_at_grid_top=s["lr"] == max(lrs),
                         val_int8_mean=round(s["val_int8"], 4), val_int8_sd=round(st.stdev(s["val_int8_seeds"]), 4),
                         test_int8_mean=round(s["test_int8"], 4), test_int8_sd=round(s["test_int8_sd"], 4),
                         test_int8_sd_ddof0=round(s["test_int8_sd0"], 4), test_int8_seeds=" ".join(f"{x:.4f}" for x in s["test_int8_seeds"]),
                         test_fp32_mean=round(s["test_fp32"], 4),
                         fp32sel_lr=sf["lr"], fp32sel_wd=sf["wd"], fp32sel_test_int8_mean=round(sf["test_int8"], 4),
                         fp32sel_test_int8_sd=round(sf["test_int8_sd"], 4),
                         old_cfg=b["old_cfg"], old_test_int8_mean=b["acc_int8_old"],
                         old_test_int8_sd_as_published=b["acc_int8_sd_old"],
                         flash_bytes_e10=v.get("flash_bytes"), flash_bytes_old=b["flash_old"],
                         pc_bitexact_seed1=v.get("pc_bitexact_logits"),
                         pc_bitexact_seeds123=" ".join(vs.get(m, {}).get(str(k), {}).get("pc_bitexact_logits", v.get("pc_bitexact_logits", "na")
                                                                                         if k == 1 else "na") for k in SEEDS),
                         protocol="E-10" if m in NONCNN_E10 + CNNS_E10 else "E-10b",
                         ucsim_cycles=round(b["ucsim_cycles"]),
                         ucsim_cycles_e10build=round(uc[m]["cycles_e10"]) if m in uc else "",
                         ucsim_e10build_rel_change=round(uc[m]["rel_change"], 5) if m in uc else "",
                         ucsim_e10build_bitexact=uc[m]["ucsim_bitexact_logits"] if m in uc else "",
                         board_cycles=b["board_cycles"], board_ms=b["board_ms"],
                         board_ms_source=b["board_ms_source"],
                         board_ms_preboard_estimate=b.get("board_ms_preboard_estimate", ""),
                         board_ms_e10build_estimate=b.get("board_ms_e10build_estimate", "")))
        accs[m] = dict(mean=s["test_int8"], sd1=s["test_int8_sd"], sd0=s["test_int8_sd0"])
        accs_f[m] = dict(mean=sf["test_int8"], sd1=sf["test_int8_sd"], sd0=float(np.std(sf["test_int8_seeds"])))
    with open(f"{E10}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best_dann = max(DANNS, key=lambda m: accs[m]["mean"])
    mrows = []
    for variant, A, dd in [("primary_int8val_ddof1", accs, "sd1"), ("int8val_ddof0", accs, "sd0"),
                           ("fp32val_ddof1", accs_f, "sd1"), ("fp32val_ddof0", accs_f, "sd0")]:
        for x in NONCNN:
            c, gap, thr = matched(A, x, CNNS, dd)
            ms = bi[x]["board_ms"]
            mrows.append(dict(variant=variant, model=x, is_best_dann=(x == best_dann) if variant.startswith("primary") else "",
                              acc_mean=round(A[x]["mean"], 4), acc_sd=round(A[x][dd], 4), board_ms=ms,
                              smallest_matched_cnn=c or "none",
                              cnn_acc_mean=round(A[c]["mean"], 4) if c else "", cnn_acc_sd=round(A[c][dd], 4) if c else "",
                              gap_x_minus_cnn=round(gap, 4) if c else "", threshold=round(thr, 4) if c else "",
                              board_ms_source=bi[x]["board_ms_source"],
                              cnn_board_ms=bi[c]["board_ms"] if c else "", cnn_board_ms_source=bi[c]["board_ms_source"] if c else "",
                              time_ratio_cnn_over_x=round(bi[c]["board_ms"] / ms, 3) if c else "",
                              ucsim_ratio_cnn_over_x=round(bi[c]["ucsim_cycles"] / bi[x]["ucsim_cycles"], 3) if c else "",
                              flash_x=ver.get(x, {}).get("flash_bytes"), flash_cnn=ver.get(c, {}).get("flash_bytes") if c else "",
                              flash_share_cnn_over_x=round(ver[c]["flash_bytes"] / ver[x]["flash_bytes"], 3)
                              if c and c in ver and x in ver else "",
                              ucsim_e10build_ratio=round(uc[c]["cycles_e10"] / uc[x]["cycles_e10"], 3) if c and c in uc and x in uc else ""))
            # also: per-CNN gap table for the next-smaller CNN (how far it misses)
    with open(f"{E10}/matched.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrows[0]))
        w.writeheader()
        w.writerows(mrows)
    for r in rows:
        print({k: r[k] for k in ("model", "sel_lr", "sel_wd", "sel_at_grid_top", "test_int8_mean", "test_int8_sd",
                                 "old_test_int8_mean", "fp32sel_lr", "fp32sel_wd", "fp32sel_test_int8_mean", "flash_bytes_e10",
                                 "pc_bitexact_seed1", "board_ms")})
    print("best dANN:", best_dann)
    for r in mrows:
        print(r)
    # gap of every CNN to d16_s64 (primary)
    for c in CNNS:
        x = "dann_d16_s64"
        print(f"{c}: {accs[c]['mean']:.4f}±{accs[c]['sd1']:.4f}  gap to d16_s64 {accs[x]['mean'] - accs[c]['mean']:+.4f}"
              f"  thr {T_RULE * max(accs[x]['sd1'], accs[c]['sd1']):.4f}  ms {bi[c]['board_ms']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "select", "verify", "ucsim", "report", "one"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=BASE_LRS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--job", nargs=4)
    a = ap.parse_args()
    if a.cmd == "grid":
        a.models = a.models or MODELS
        cmd_grid(a)
    elif a.cmd == "one":
        ensure(RUNS)
        print(train_eval((a.job[0], float(a.job[1]), float(a.job[2]), int(a.job[3]))))
    elif a.cmd == "select":
        cmd_select(a)
    elif a.cmd == "verify":
        cmd_verify(a)
    elif a.cmd == "ucsim":
        cmd_ucsim(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
