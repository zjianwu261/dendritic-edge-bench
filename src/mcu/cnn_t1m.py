"""The T1m int8 / SDCC / ucsim flow for the small G1 CNNs.

Model: results/T6/models/g1/<cfg>_t<seed>.pt (src/t6/dann_g.py SmallCNN, trained with the G1 fair protocol):
  conv3x3 stride 2 pad 1 (1 -> C1) - ReLU - conv3x3 stride 2 pad 1 (C1 -> C2) - ReLU - fc (C2*49 -> 10)
The config per width is the one that fair_summary.py selects on validation (val_acc_last, 3-seed mean).

Integer scheme (same conventions as src/mcu/quant.py; must stay bit-exact with the MODEL_CNN branch of c/infer.c):
  input      raw uint8 pixels (scale 1/255)
  weights    per-layer symmetric int8, |w_q| <= 127; biases b_q = round(b / (s_in * s_w)) in int16 / int32
  MAC        uint8 x int8 -> int16 product, int32 accumulator; zero padding (top row / left column only)
  requant    t = (acc + 2^(sh-1)) >> sh;  ReLU: a = min(max(t, 0), 127)  (so activations are 0..127, stored uint8)
  output     int32 logits of the fc layer, class = first argmax
The two shifts are grid-searched on the training-set validation split of the model's own seed (the 10 % that
dann_g.py held out), never on the test set.
If C1*196 + C2*49 bytes of activations do not fit the ucsim xdata budget, conv2 is fused with the fc layer
(CNN_FUSED: the fc weights are stored transposed and logits are accumulated per conv2 output).

Per model: int8 test accuracy on all 10 000 test images for seeds 1-3; seed 1 also gets the gcc bit-exact check
(10 000 images), the SDCC board + ucsim builds (flash / xdata), and ucsim cycles on the first --n_sim test images.
Output: results/T1m/cnn/<cfg>_t<seed>.json, results/T1m/cnn/cnn_results.csv,
        build dirs results/T1m/build/<short>_t<seed>/ (short = cnn_c<C1>_<C2>).
Usage:  S51=<path to s51> python cnn_t1m.py [--cfgs cnn_c8_16_lr..._fair ...] [--n_sim 5]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_t1m as R  # noqa: E402  (sdcc_build, ucsim_run, constants)
from common import EXP, ensure, load_fmnist_u8  # noqa: E402
from gen_c import _arr, _bias_type  # noqa: E402
from quant import qweight, shift_round  # noqa: E402

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
G1 = f"{ROOT}/results/T6/models/g1"
OUT = f"{EXP}/cnn"
SIM_XDATA = 0x1B00          # ucsim build: xdata below res[] / img[] (sim_main.c)
UART_XDATA_FIXED = 784 + 40  # uart_main img[] + logits


def load_pt(cfg, seed):
    import torch
    sd = torch.load(f"{G1}/{cfg}_t{seed}.pt", map_location="cpu")
    m = {k: v.numpy().astype(np.float64) for k, v in sd.items()}
    meta = json.load(open(f"{G1}/{cfg}_t{seed}.json"))
    return m, meta


def val_idx(seed, n=60000):
    """dann_g.py: rng = default_rng(seed); idx = permutation(60000); last 10 % = validation."""
    return np.random.default_rng(seed).permutation(n)[-int(0.1 * n):]


def im2col(x, H):
    """x (N, C, H, H) -> (N, (H/2)^2, C*9) patches of a 3x3 stride-2 pad-1 conv, (c, ky, kx) order."""
    N, C = x.shape[:2]
    Ho = H // 2
    xp = np.zeros((N, C, H + 2, H + 2), x.dtype)
    xp[:, :, 1:-1, 1:-1] = x
    cols = np.empty((N, Ho, Ho, C, 3, 3), x.dtype)
    for ky in range(3):
        for kx in range(3):
            cols[:, :, :, :, ky, kx] = xp[:, :, ky:ky + 2 * Ho:2, kx:kx + 2 * Ho:2].transpose(0, 2, 3, 1)
    return cols.reshape(N, Ho * Ho, C * 9)


def conv_int(x, Wq, bq, H):
    """integer conv: x (N, C, H, H) int64, Wq (O, C, 3, 3) -> pre-activation (N, O, H/2, H/2)."""
    N, O = x.shape[0], Wq.shape[0]
    Ho = H // 2
    out = im2col(x, H) @ Wq.reshape(O, -1).T + bq
    return out.transpose(0, 2, 1).reshape(N, O, Ho, Ho)


def relu_int(t):
    return np.clip(t, 0, 127)


def ref_forward_cnn(q, x_u8, bs=1000):
    """Bit-exact integer forward pass of c/infer.c MODEL_CNN. x_u8 (N, 784) uint8 -> int64 logits (N, 10)."""
    out = []
    for i in range(0, len(x_u8), bs):
        x = x_u8[i:i + bs].astype(np.int64).reshape(-1, 1, 28, 28)
        a1 = relu_int(shift_round(conv_int(x, q["W1q"], q["b1q"], 28), q["sh1"]))
        a2 = relu_int(shift_round(conv_int(a1, q["W2q"], q["b2q"], 14), q["sh2"]))
        out.append(a2.reshape(len(x), -1) @ q["W3q"].T + q["b3q"])
    return np.concatenate(out)


def float_forward_cnn(m, x_u8, bs=2000):
    out = []
    for i in range(0, len(x_u8), bs):
        x = x_u8[i:i + bs].astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
        h = np.maximum(conv_float(x, m["c1.weight"], m["c1.bias"], 28), 0)
        h = np.maximum(conv_float(h, m["c2.weight"], m["c2.bias"], 14), 0)
        out.append(h.reshape(len(x), -1) @ m["fc.weight"].T + m["fc.bias"])
    return np.concatenate(out)


def conv_float(x, W, b, H):
    N, O = x.shape[0], W.shape[0]
    Ho = H // 2
    return (im2col(x, H) @ W.reshape(O, -1).T + b).transpose(0, 2, 1).reshape(N, O, Ho, Ho)


def _pre_max(pre):
    return max(float(np.percentile(pre[pre > 0], 99.99)) if (pre > 0).any() else 1.0, 1.0)


def build_q(m, o1, o2, xc):
    W1q, s1 = qweight(m["c1.weight"])
    W2q, s2 = qweight(m["c2.weight"])
    W3q, s3 = qweight(m["fc.weight"])
    A1 = s1 / 255.0
    b1q = np.round(m["c1.bias"] / A1).astype(np.int64)
    x = xc.astype(np.int64).reshape(-1, 1, 28, 28)
    pre1 = conv_int(x, W1q, b1q, 28)
    sh1 = max(int(np.ceil(np.log2(_pre_max(pre1) / 127.0))) - o1, 0)
    sa1 = A1 * 2 ** sh1
    A2 = sa1 * s2
    b2q = np.round(m["c2.bias"] / A2).astype(np.int64)
    a1 = relu_int(shift_round(pre1, sh1))
    pre2 = conv_int(a1, W2q, b2q, 14)
    sh2 = max(int(np.ceil(np.log2(_pre_max(pre2) / 127.0))) - o2, 0)
    A3 = A2 * 2 ** sh2 * s3
    b3q = np.round(m["fc.bias"] / A3).astype(np.int64)
    return dict(W1q=W1q, b1q=b1q, W2q=W2q, b2q=b2q, W3q=W3q, b3q=b3q, sh1=sh1, sh2=sh2, sh_off=(o1, o2),
                scales=dict(s_w1=s1, s_w2=s2, s_w3=s3, s_a1=sa1, s_logit=A3))


def quantize_cnn(m, xc, yc, offsets=(0, 1, 2, 3)):
    best = None
    sub = slice(0, 2000)  # shift search on the first 2000 validation images, then the full split is reported
    for o1 in offsets:
        for o2 in offsets:
            q = build_q(m, o1, o2, xc[sub])
            acc = float((ref_forward_cnn(q, xc[sub]).argmax(1) == yc[sub]).mean())
            if best is None or acc > best[0]:
                best = (acc, o1, o2)
    q = build_q(m, best[1], best[2], xc)
    q["cal_acc"] = float((ref_forward_cnn(q, xc).argmax(1) == yc).mean())
    # worst-case accumulator bounds (int32 in C); each product fits int16 (255 * 127)
    C1 = q["W1q"].shape[0]
    b1 = np.abs(q["W1q"]).reshape(C1, -1).sum(1) * 255 + np.abs(q["b1q"])
    b2 = np.abs(q["W2q"]).reshape(q["W2q"].shape[0], -1).sum(1) * 127 + np.abs(q["b2q"])
    b3 = np.abs(q["W3q"]).sum(1) * 127 + np.abs(q["b3q"])
    q["acc_bound"] = int(max(b1.max(), b2.max(), b3.max()))
    assert q["acc_bound"] < 2 ** 31
    return q


def emit_cnn(q, path, name, fused):
    C1, C2 = q["W1q"].shape[0], q["W2q"].shape[0]
    b1t, b2t, b3t = _bias_type(q["b1q"]), _bias_type(q["b2q"]), _bias_type(q["b3q"])
    lines = [f"/* generated by src/mcu/cnn_t1m.py: {name} */", "#ifndef T1M_MODEL_H", "#define T1M_MODEL_H",
             '#include "platform.h"', f'#define MODEL_NAME "{name}"', f"#define SH1 {q['sh1']}", f"#define SH2 {q['sh2']}",
             "#define MODEL_CNN 1", f"#define C1 {C1}", f"#define C2 {C2}"] + (["#define CNN_FUSED 1"] if fused else [])
    lines.append("#ifndef MODEL_DEFINES_ONLY")
    lines.append(_arr("int8_t", "w1", q["W1q"]))                         # (C1, 1, 3, 3)
    lines.append(_arr(b1t, "b1", q["b1q"]))
    lines.append(_arr("int8_t", "w2", q["W2q"]))                         # (C2, C1, 3, 3)
    lines.append(_arr(b2t, "b2", q["b2q"]))
    if fused:
        lines.append(_arr("int8_t", "w3t", q["W3q"].T))                  # (C2*49, 10)
    else:
        lines.append(_arr("int8_t", "w3", q["W3q"]))                     # (10, C2*49)
    lines.append(_arr(b3t, "b3", q["b3q"]))
    lines += ["#endif /* MODEL_DEFINES_ONLY */", "#endif"]
    open(path, "w").write("\n".join(lines) + "\n")
    sz = lambda t_, n: n * (2 if t_ == "int16_t" else 4)
    tb = dict(w1=q["W1q"].size, b1=sz(b1t, C1), w2=q["W2q"].size, b2=sz(b2t, C2), w3=q["W3q"].size, b3=sz(b3t, 10))
    tb["total"] = int(sum(tb.values()))
    return tb


def macs(C1, C2):
    """multiply-accumulates per image (padding taps skipped, as in the C code)."""
    def taps(H):  # valid taps over all outputs of a 3x3 s2 p1 conv on an HxH input
        Ho = H // 2
        n_row = sum(3 - (y == 0) for y in range(Ho))
        return n_row * n_row
    return dict(conv1=C1 * taps(28), conv2=C2 * C1 * taps(14), fc=10 * C2 * 49)


def process(cfg, seeds, data, n_sim):
    C1, C2 = [int(v) for v in cfg.split("_")[1][1:].split("_")[0:1] + cfg.split("_")[2:3]]
    short = f"cnn_c{C1}_{C2}"
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    fused = C1 * 196 + C2 * 49 + 40 > SIM_XDATA
    rows = []
    for s in seeds:
        m, meta = load_pt(cfg, s)
        vi = val_idx(s)
        acc_fp32_np = float((float_forward_cnn(m, xte).argmax(1) == yte).mean())
        q = quantize_cnn(m, xtr[vi], ytr[vi])
        L = ref_forward_cnn(q, xte)
        acc_int8 = float((L.argmax(1) == yte).mean())
        tag = f"{short}_t{s}"
        r = dict(tag=tag, cfg=cfg, seed=s, C1=C1, C2=C2, params=meta["params_stored"], val_acc_last_fp32=meta["val_acc_last"],
                 acc_fp32=meta["test_acc"], acc_fp32_numpy=acc_fp32_np, acc_int8=acc_int8, cal_acc_int8=q["cal_acc"],
                 sh1=q["sh1"], sh2=q["sh2"], acc_bound=q["acc_bound"], fused_fc=fused, macs=macs(C1, C2))
        if s == seeds[0]:
            bdir = ensure(f"{EXP}/build/{tag}")
            r["tables"] = emit_cnn(q, f"{bdir}/model.h", tag, fused)
            r["table_bytes"] = r["tables"]["total"]
            subprocess.run(["gcc", "-O2", "-Wall", "-Wno-unused-function", f"-I{bdir}", f"-I{R.C_DIR}", f"{R.C_DIR}/infer.c",
                            f"{R.C_DIR}/pc_main.c", "-o", f"{bdir}/pc_infer"], check=True)
            xte.tofile(f"{bdir}/test_x.u8")
            subprocess.run([f"{bdir}/pc_infer", f"{bdir}/test_x.u8", f"{bdir}/pc_out.bin"], check=True)
            raw = np.fromfile(f"{bdir}/pc_out.bin", np.uint8).reshape(-1, 41)
            pcl = raw[:, :40].copy().view("<i4").astype(np.int64)
            r["pc_bitexact_logits"] = f"{int((pcl == L).all(1).sum())}/{len(xte)}"
            r["pc_bitexact_class"] = f"{int((raw[:, 40] == L.argmax(1)).sum())}/{len(xte)}"
            os.remove(f"{bdir}/test_x.u8"); os.remove(f"{bdir}/pc_out.bin")
            uart = R.sdcc_build(bdir, ["uart_main"], "uart_35M", ["-DSTC8", f"-DF_OSC={R.F_OSC}UL"], R.XRAM_MAX)
            sim = R.sdcc_build(bdir, ["sim_main"], "sim", [], SIM_XDATA)
            r.update(flash_bytes=uart["flash"], xdata_bytes=uart["xdata"], uart_build_ok=uart["ok"],
                     fits=bool(uart["ok"] and uart["flash"] <= R.FLASH_MAX and uart["xdata"] <= R.XRAM_MAX),
                     uart_link_log=None if uart["ok"] else uart["link_log"], sim_build_ok=sim["ok"], sim_flash=sim["flash"],
                     sim_link_log=None if sim["ok"] else sim["link_log"])
            if sim["ok"] and R.S51 and n_sim > 0:
                cyc, n_ok, n_cls, inst = [], 0, 0, []
                for i in range(n_sim):
                    c, lg, cy, cy0, st = R.ucsim_run(bdir, xte[i])
                    n_ok += int((lg == L[i]).all()); n_cls += int(c == L[i].argmax())
                    cyc.append(cy - cy0); inst.append(st["inst"])
                r.update(ucsim_n=n_sim, ucsim_bitexact_logits=f"{n_ok}/{n_sim}", ucsim_bitexact_class=f"{n_cls}/{n_sim}",
                         cycles_ucsim_mc=float(np.mean(cyc)), cycles_ucsim_mc_min=int(min(cyc)), cycles_ucsim_mc_max=int(max(cyc)),
                         instr_ucsim=float(np.mean(inst)))
        json.dump(r, open(f"{ensure(OUT)}/{cfg}_t{s}.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k not in ("tables",)}), flush=True)
        rows.append(r)
    return rows


def summary():
    """cnn_results.csv: per CNN the 3-seed int8 accuracy, seed-1 build / ucsim numbers, and an ESTIMATED board time.
    The board estimate is analytic, not measured: ucsim machine cycles x the board/ucsim ratio measured on the
    STC8A8K64D4 for the dense-kernel MLPs (board_summary.csv, mlp rows), at 35.0 MHz."""
    import glob
    import statistics as st
    bs = [r for r in csv.DictReader(open(f"{EXP}/board_summary.csv")) if r["model"].startswith("mlp_")]
    k = st.mean(float(r["board_over_ucsim"]) for r in bs)
    by = {}
    for f in sorted(glob.glob(f"{OUT}/cnn_*_t*.json")):
        d = json.load(open(f))
        by.setdefault(d["cfg"], []).append(d)
    rows = []
    for cfg, ds in by.items():
        ds = sorted(ds, key=lambda d: d["seed"])
        s1 = ds[0]
        ai = [d["acc_int8"] for d in ds]
        af = [d["acc_fp32"] for d in ds]
        cyc = s1.get("cycles_ucsim_mc")
        rows.append({"model": f"cnn_c{s1['C1']}_{s1['C2']}", "cfg": cfg, "params": s1["params"], "n_seeds": len(ds),
                     "acc_fp32": round(st.mean(af), 4), "acc_fp32_std": round(st.stdev(af), 4) if len(af) > 1 else "",
                     "acc_int8": round(st.mean(ai), 4), "acc_int8_std": round(st.stdev(ai), 4) if len(ai) > 1 else "",
                     "acc_int8_seed1": s1["acc_int8"], "table_bytes": s1.get("table_bytes"), "flash_bytes": s1.get("flash_bytes"),
                     "xdata_bytes": s1.get("xdata_bytes"), "fits": s1.get("fits"), "fused_fc": s1["fused_fc"],
                     "macs": sum(s1["macs"].values()), "pc_bitexact_seed1": s1.get("pc_bitexact_logits"),
                     "ucsim_bitexact_seed1": s1.get("ucsim_bitexact_logits"), "cycles_ucsim_mc": round(cyc) if cyc else "",
                     "ucsim_mc_per_mac": round(cyc / sum(s1["macs"].values()), 1) if cyc else "",
                     "board_over_ucsim_assumed": round(k, 3),
                     "board_cycles_ESTIMATE": round(cyc * k) if cyc else "",
                     "board_ms_ESTIMATE_35MHz": round(cyc * k / 35e3, 1) if cyc else ""})
    keys = list(rows[0])
    with open(f"{OUT}/cnn_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--cfgs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_sim", type=int, default=5)
    a = ap.parse_args()
    if a.summary:
        summary(); return
    data = load_fmnist_u8()
    for cfg in a.cfgs:
        process(cfg, a.seeds, data, a.n_sim)


if __name__ == "__main__":
    main()
