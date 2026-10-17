"""Dendritic context-gated adapter v2 (dend2) on the T4a CL sequence.

Differences from v1 (cl_run.py --method dend):
  - DendriticAdapterV2 (cl_methods.py): one segment per unit per context, allocated when the context
    appears and initialised aligned with the context vector; kWTA default 5%; h = 192.
  - gradient mask: after each step the shared Down rows / Up columns of units that were NOT active
    (post-kWTA, any sample, any of the step's forward passes) for the current context are restored to
    their pre-step values (exact masking also under Adam momentum); segments of old contexts are
    always restored (frozen).  --gradmask 0 disables the unit mask (old segments still frozen).
  - context vector z = standardised [frozen goal-masked NoMaD feature (256), 16-bin brightness
    histogram of the current frame (x hist_w)]; standardisation (mean/std per dim) fixed from the first
    task's 512 prototype samples. Prototype of context k = mean z over 512 training samples of task k
    (stored: 272 floats + threshold); no raw data kept.
  - new-context detection: a sample is "new" if min_k ||z - mu_k|| / tau_k > 1, tau_k = 99th percentile
    of the distances of context k's own prototype samples. Reported: fraction of each new task's samples
    flagged new at its arrival (should be ~1), and false-new rate on seen tasks' test samples at the end.
  - context "id": z = mu_(true task); "infer": z = mu of the nearest prototype (per sample).
  - reports per-context winner units (active in > 50% of the context's test samples, output adapter and
    token adapter) and their pairwise Jaccard overlap, and the fraction of samples where the selected
    segment is the context's own segment.
Training loss, sequence, steps and evaluation are identical to cl_run.py.
Usage: python cl_dend2.py --context id --seq ... [--kwta 0.05] [--gradmask 1] [--seed 0]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_data import Task, evaluate, norm_deltas  # noqa: E402
from cl_methods import DendriticAdapterV2, encode  # noqa: E402
from common import EXP, ddpm_scheduler, load_model  # noqa: E402
from ctx_infer import ContextInferer  # noqa: E402


def cap_gpu_mem():
    mb = int(os.environ.get("DN_MEM_MB", "0"))
    if mb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(max(0.02, (mb - 450) / total))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="id", choices=["id", "infer", "random"])
    ap.add_argument("--mask_frac", type=float, default=0.0,
                    help=">0 -> grad mask keeps units active in >= this fraction of (sample, token, pass) instances; 0 = union (v2)")
    ap.add_argument("--holdout_val", type=int, default=0)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=50)
    ap.add_argument("--h", type=int, default=192)
    ap.add_argument("--kwta", type=float, default=0.05)
    ap.add_argument("--gradmask", type=int, default=1)
    ap.add_argument("--seg_gain", type=float, default=3.0)
    ap.add_argument("--hist_w", type=float, default=1.0)
    ap.add_argument("--ctx_feat", default="v2", choices=["v2", "v3"],
                    help="v3 = + Laplacian variance / occlusion ratio, group-wise standardisation (ctx_infer.py)")
    ap.add_argument("--novel_q", type=float, default=0.0,
                    help=">0 -> new-context threshold tau_k = this quantile of context k's validation distances "
                         "(needs --holdout_val 1); 0 = old rule (0.99 quantile of the prototype samples)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    # v3fix options; defaults reproduce the original (buggy) v3 runs
    ap.add_argument("--seg_eps", type=float, default=0.0, help="segment init z / (|z|^2 + eps)")
    ap.add_argument("--std_ref", default="first", choices=["first", "normal"],
                    help="standardisation reference: first = the first task's prototype samples (original; makes the first "
                         "prototype ~0); normal = the same samples drawn from the first task's NORMAL-light frames")
    ap.add_argument("--seg_ref", default="same", choices=["same", "normal"],
                    help="coordinates of the context vector fed to the dendritic segments. same = the inferer's "
                         "standardised prototype (original); normal = a SEPARATE standardisation fitted on the first task's "
                         "normal-light prototype samples, used only for the segments (the inferer / S2 keep --std_ref)")
    ap.add_argument("--train_mode", default="eval", choices=["eval", "train"],
                    help="model mode during the training steps (original: eval); prototypes / inference features / "
                         "evaluation are always computed in eval()")
    ap.add_argument("--seg_orth", type=int, default=0, help="1: orthogonalised segment init with adaptive gain (cl_methods.add_context)")
    # gate temperature / frozen-segment controls; both default off
    ap.add_argument("--seg_temp", type=int, default=0, help="1: temperature-normalised gate sigmoid(seg_gain * d / target_k) (needs --seg_orth 1)")
    ap.add_argument("--seg_freeze", type=int, default=0, help="1: XdG-style control, every segment stays at its initial value (only Down/Up train)")
    ap.add_argument("--assert_own", type=float, default=0.0,
                    help=">0 with --context id: after each task every seen task must have own_seg_frac_out >= this (test)")
    a = ap.parse_args()
    cap_gpu_mem()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed); gen = torch.Generator(device=a.device).manual_seed(a.seed)
    dev = a.device
    model, cfg = load_model("nomad", dev, verbose=False)
    sched = ddpm_scheduler(cfg); alpha = float(cfg["alpha"]); venc = model.vision_encoder
    for p in model.parameters():
        p.requires_grad_(False)
    n_backbone = sum(p.numel() for p in model.parameters())
    seq = [s.split(":") for s in a.seq.split(",")]
    tasks = [Task(n, l, device=dev, holdout_val=bool(a.holdout_val)) for n, l in seq]
    names = [f"{n}:{l}" for n, l in seq]
    for t in tasks:
        t.test = {k: v[: a.eval_n] for k, v in t.test.items()}
        t.test_sub = t.test
        t.val_sub = {k: v[: a.eval_n] for k, v in t.val.items()} if t.val is not None else None
    inferer = ContextInferer(venc, a.ctx_feat, a.novel_q if a.novel_q > 0 else 0.99, a.hist_w)
    zdim = inferer.dim
    ads = {"tok": DendriticAdapterV2(256, a.h, len(tasks), zdim, a.kwta, a.seg_gain).to(dev),
           "out": DendriticAdapterV2(256, a.h, len(tasks), zdim, a.kwta, a.seg_gain).to(dev)}
    if a.seg_temp:
        assert a.seg_orth, "--seg_temp needs the per-context gains of --seg_orth 1"
        for ad in ads.values():
            ad.temp_norm = True
    params = [p for ad in ads.values() for p in ad.parameters()]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    mus = inferer.mus
    segmus = []  # context vectors in segment coordinates (= mus unless --seg_ref normal)
    seg_inf = ContextInferer(venc, a.ctx_feat, a.novel_q if a.novel_q > 0 else 0.99, a.hist_w) if a.seg_ref == "normal" else None

    def zraw(t, ctx, obs, goal):  # (t, ctx kept for the call sites; the frame is recovered from obs)
        return inferer.raw(obs, goal)

    def zstd(zr):
        return inferer.std(zr)

    def proto_samples(t, i, n=512):
        b = t.sample_train(n, np.random.RandomState(1000 + i)); out = []
        for s in range(0, n, 32):
            obs, goal = t.images(b["ctx"][s:s + 32], b["goal"][s:s + 32])
            out.append(zraw(t, b["ctx"][s:s + 32], obs, goal))
        return torch.cat(out)

    def nearest(z):
        return inferer.nearest(z)

    def cond(obs, goal, mask, z):
        return encode(venc, obs, goal, mask, ads["tok"], ads["out"], z)

    stats = {"novel_frac_at_arrival": [], "infer_acc": {}, "own_seg_frac": {}, "false_new_final": {}}

    def eval_all(after):
        row = {}
        for j, t in enumerate(tasks):
            if j > after and a.context == "id":
                row[names[j]] = None; continue
            hits, own, nov = [], [], []

            def cfn(m, obs, goal, mk, j=j, t=t):
                z = zstd(zraw(t, None, obs, goal))
                k, novel = nearest(z)
                if a.context == "id":
                    k = torch.full_like(k, j)
                elif a.context == "random":  # control: a random seen context per sample
                    k = torch.randint(0, len(inferer.mus), k.shape, device=k.device)
                zz = torch.stack(segmus)[k]
                out = cond(obs, goal, mk, zz)
                if mk.sum() == 0:
                    hits.append((k == j).float()); nov.append(novel.float())
                    own.append((ads["out"].last_seg == j).float().mean(-1))
                return out
            r = evaluate(model, sched, t, n_action_samples=a.samples, seed=1, cond_fn=cfn, bs=a.eval_bs)
            if j <= after:
                r["ctx_infer_acc"] = float(torch.cat(hits).mean()); r["false_new"] = float(torch.cat(nov).mean())
                r["own_seg_frac_out"] = float(torch.cat(own).mean())
            row[names[j]] = r
        return row

    def eval_both(after):
        """val first (if held out; used only for model selection), then test (main numbers)."""
        out = {}
        if a.holdout_val:
            for t in tasks:
                t.test = t.val_sub
            out["row_val"] = eval_all(after)
        for t in tasks:
            t.test = t.test_sub
        out["row"] = eval_all(after)
        return out

    res = {"args": vars(a), "method": "dend2", "names": names, "n_backbone_params": n_backbone, "n_trainable": n_train,
           "trainable_frac": n_train / n_backbone, "R": [], "train_s": [], "peak_MB": []}
    mask_log = {k: [] for k in ads}
    QS = (0.8, 0.9, 0.95, 0.99)
    taus_val = {q: [] for q in QS}; tauw = []; calib = {"per_sample": {q: [] for q in QS}, "window32": []}

    def zval(t):
        """standardised z of the task's validation samples (threshold calibration)."""
        out = []
        for s0 in range(0, len(t.val["goal"]), 50):
            c = t.val["ctx"][s0:s0 + 50]; obs, goal = t.images(c, t.val["goal"][s0:s0 + 50])
            out.append(zstd(zraw(t, c, obs, goal)))
        return torch.cat(out)

    def windows(z, n=16, w=32, seed=0):
        g = np.random.RandomState(seed)
        return torch.stack([z[torch.from_numpy(g.choice(len(z), w, replace=False))].mean(0) for _ in range(n)])

    for i, t in enumerate(tasks):
        zr = proto_samples(t, i)
        if i == 0:
            if a.std_ref == "normal":  # same sample indices, normal-light frames of the first task's environment
                tref = Task(seq[0][0], "normal", device=dev, holdout_val=bool(a.holdout_val))
                inferer.fit_std(proto_samples(tref, 0)); del tref
            else:
                inferer.fit_std(zr)
            if seg_inf is not None:
                tref = Task(seq[0][0], "normal", device=dev, holdout_val=bool(a.holdout_val))
                seg_inf.fit_std(proto_samples(tref, 0)); del tref
        z = zstd(zr)
        if mus:
            _, novel = nearest(z); stats["novel_frac_at_arrival"].append(float(novel.float().mean()))
        if mus and a.holdout_val:  # detection of the arriving task with validation-calibrated thresholds
            M = torch.stack(mus); dz = torch.cdist(z, M)
            for q in QS:
                calib["per_sample"][q].append(float(((dz / torch.stack(taus_val[q])[None]).min(-1).values > 1).float().mean()))
            dw = torch.cdist(windows(z, seed=i), M)
            calib["window32"].append(float(((dw / torch.stack(tauw)[None]).min(-1).values > 1).float().mean()))
        zv = zval(t) if a.holdout_val else None
        _, mu = inferer.add_context(z, zv if a.novel_q > 0 else None)
        if a.holdout_val:
            dv = (zv - mu).norm(dim=-1)
            for q in QS:
                taus_val[q].append(torch.quantile(dv, q))
            tauw.append(torch.quantile((windows(zv, n=200, seed=100 + i) - mu).norm(dim=-1), 0.99))
        mu_seg = seg_inf.std(zr).mean(0) if seg_inf is not None else mu
        segmus.append(mu_seg)
        for ad in ads.values():
            ad.add_context(mu_seg, gen, a.seg_eps, orth=bool(a.seg_orth))
        res.setdefault("ctx_norm", []).append(float(mu_seg.norm()))
        if a.seg_orth:
            res["seg_gains"] = {k: list(ad.gains) for k, ad in ads.items()}
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None; t0 = time.time()
        if a.train_mode == "train":
            model.train()
        for step in range(a.steps):
            b = t.sample_train(a.bs, rng)
            obs, goal = t.images(b["ctx"], b["goal"])
            act = b["actions"].to(dev).float(); dist = b["dist"].to(dev).float(); amask = b["mask"].to(dev).float()
            zz = mu_seg.expand(a.bs, -1)
            gm = (torch.rand(a.bs, device=dev) < 0.5).long()
            active = {k: torch.zeros(a.h, dtype=torch.bool, device=dev) for k in ads}
            cnt = {k: torch.zeros(a.h, device=dev) for k in ads}; nin = {k: 0 for k in ads}
            c = cond(obs, goal, gm, zz)
            for k, ad in ads.items():
                active[k] |= ad.last_active; cnt[k] += ad.last_act_count; nin[k] += ad.last_act_n
            c_nm = cond(obs, goal, torch.zeros_like(gm), zz)
            for k, ad in ads.items():
                active[k] |= ad.last_active; cnt[k] += ad.last_act_count; nin[k] += ad.last_act_n
            if a.mask_frac > 0:
                active = {k: cnt[k] / nin[k] >= a.mask_frac for k in ads}
            for k in ads:
                mask_log[k].append(int(active[k].sum()))
            dist_loss = F.mse_loss(model("dist_pred_net", obsgoal_cond=c_nm).squeeze(-1), dist)
            na = norm_deltas(act); noise = torch.randn_like(na)
            ts = torch.randint(0, sched.config.num_train_timesteps, (a.bs,), device=dev)
            pred = model("noise_pred_net", sample=sched.add_noise(na, noise, ts), timestep=ts, global_cond=c)
            per = ((pred - noise) ** 2).mean(dim=(1, 2))
            diff_loss = (per * amask).mean() / (amask.mean() + 1e-2)
            loss = alpha * dist_loss + (1 - alpha) * diff_loss
            snap = {k: (ad.down.weight.detach().clone(), ad.down.bias.detach().clone(), ad.up.weight.detach().clone(),
                        ad.seg.detach().clone()) for k, ad in ads.items()}
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                for k, ad in ads.items():
                    dw, db, uw, sg = snap[k]
                    keep_seg = torch.ones(sg.shape[0], dtype=torch.bool, device=dev); keep_seg[i] = bool(a.seg_freeze)
                    ad.seg[keep_seg] = sg[keep_seg]                    # old (and unallocated) segments frozen
                    if a.gradmask:
                        off = ~active[k]
                        ad.down.weight[off] = dw[off]; ad.down.bias[off] = db[off]; ad.up.weight[:, off] = uw[:, off]
            if step % 200 == 0:
                print(f"task {i} {names[i]} step {step} loss {loss.item():.4f} active_out {int(active['out'].sum())} active_tok {int(active['tok'].sum())}", flush=True)
        model.eval()
        res.setdefault("mask_units_mean", []).append({k: float(np.mean(v)) for k, v in mask_log.items()})
        for v in mask_log.values():
            v.clear()
        res["train_s"].append(round(time.time() - t0, 1)); res["peak_MB"].append(round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0)
        res["R"].append({"after": names[i], **eval_both(i)})
        print(json.dumps({"after": names[i], "act_gc": {k: (v or {}).get("act_mse_gc") for k, v in res["R"][-1]["row"].items()},
                          "own_seg": {k: (v or {}).get("own_seg_frac_out") for k, v in res["R"][-1]["row"].items()}}), flush=True)
        if a.assert_own > 0 and a.context == "id":
            own_i = {names[j]: res["R"][-1]["row"][names[j]]["own_seg_frac_out"] for j in range(i + 1)}
            if min(own_i.values()) < a.assert_own:
                res["assert_failed"] = {"after": names[i], "own_seg_frac_out": own_i, "threshold": a.assert_own}
                out = os.path.join(EXP, "cl"); os.makedirs(out, exist_ok=True)
                json.dump(res, open(os.path.join(out, f"FAILED_dend2_{a.context}{a.tag}_s{a.seed}.json"), "w"), indent=1)
                raise SystemExit(f"ASSERT own_seg_frac_out < {a.assert_own}: {own_i}")
    if a.holdout_val:  # false-new rate on seen tasks' validation samples / windows with the calibrated thresholds
        M = torch.stack(mus); fn_s = {q: [] for q in QS}; fn_w = []
        for j, t in enumerate(tasks):
            zv = zval(t); dz = torch.cdist(zv, M)
            for q in QS:
                fn_s[q].append(float(((dz / torch.stack(taus_val[q])[None]).min(-1).values > 1).float().mean()))
            dw = torch.cdist(windows(zv, n=50, seed=500 + j), M)
            fn_w.append(float(((dw / torch.stack(tauw)[None]).min(-1).values > 1).float().mean()))
        res["novelty_calibration"] = {
            "note": "thresholds from each context's own validation samples; detection = fraction flagged new when a new task arrives (its 512 prototype samples / 16 windows of 32); false_new = on seen tasks' validation data at the end (in-sample for thresholds, so optimistic)",
            "per_sample_detect": {str(q): v for q, v in calib["per_sample"].items()},
            "per_sample_false_new": {str(q): v for q, v in fn_s.items()},
            "window32_detect": calib["window32"], "window32_false_new": fn_w}
    # winners per context and overlap
    win = {"tok": {}, "out": {}}
    with torch.no_grad():
        for j, t in enumerate(tasks):
            n = min(200, len(t.test["goal"])); c = t.test["ctx"][:n]
            obs, goal = t.images(c, t.test["goal"][:n])
            mk = torch.ones(n, dtype=torch.long, device=dev)
            zz = segmus[j].expand(n, -1)
            acts = {}
            freq = {kname: torch.zeros(a.h, device=dev) for kname in ads}
            for s0 in range(0, n, 50):
                cond(obs[s0:s0 + 50], goal[s0:s0 + 50], mk[s0:s0 + 50], zz[s0:s0 + 50])
                for kname, ad in ads.items():
                    freq[kname] += ad.last_act_samples.float().sum(0)
            for kname in ads:
                acts[kname] = set(torch.nonzero(freq[kname] / n > 0.5).squeeze(1).tolist())
            for kname in ads:
                win[kname][j] = acts[kname]
    res["winners"] = {k: {names[j]: sorted(v) for j, v in d.items()} for k, d in win.items()}
    res["winners_jaccard"] = {k: {f"{names[i]}|{names[j]}": round(jaccard(d[i], d[j]), 3) for i in d for j in d if i < j}
                              for k, d in win.items()}
    T = len(tasks)

    def summarize(key):
        rows = [r[key] for r in res["R"]]
        sm = {}
        for met in ("eps_mse_gc", "act_mse_gc", "final_err_m", "dist_mse"):
            fin = [rows[T - 1][n][met] for n in names]; diag = [rows[j][names[j]][met] for j in range(T)]
            sm[met] = {"final_avg": float(np.mean(fin)), "diag_avg": float(np.mean(diag)),
                       "bwt": float(np.mean([fin[j] - diag[j] for j in range(T - 1)])), "final": fin, "diag": diag}
        sm["ctx_infer_acc_final"] = float(np.mean([rows[T - 1][n]["ctx_infer_acc"] for n in names]))
        sm["false_new_final"] = float(np.mean([rows[T - 1][n]["false_new"] for n in names]))
        return sm
    if a.holdout_val:
        res["summary_val"] = summarize("row_val")
    rows = [r["row"] for r in res["R"]]
    summ = summarize("row")
    summ["own_seg_frac_final"] = float(np.mean([rows[T - 1][n]["own_seg_frac_out"] for n in names]))
    summ["novel_frac_at_arrival"] = stats["novel_frac_at_arrival"]
    summ["winners_jaccard_mean"] = {k: float(np.mean(list(v.values()))) if v else None for k, v in res["winners_jaccard"].items()}
    res["summary"] = summ
    res["inferer"] = {"feat_set": a.ctx_feat, "q": inferer.q, "taus": [float(x) for x in inferer.taus],
                      "tau_source": "validation" if a.novel_q > 0 else "prototype samples q0.99"}
    out = os.path.join(EXP, "cl"); os.makedirs(out, exist_ok=True)
    mtag = "" if a.gradmask else "_nomask"
    if a.gradmask and a.mask_frac > 0:
        mtag = f"_mf{int(round(a.mask_frac * 100)):02d}"
    fn = os.path.join(out, f"dend2_{a.context}_k{int(round(a.kwta * 100)):02d}{mtag}{a.tag}_s{a.seed}.json")
    json.dump(res, open(fn, "w"), indent=1)
    print("SUMMARY", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('final_avg', 'bwt', 'diag_avg', 'tok', 'out')}) for k, v in summ.items()}))
    print("saved", fn)


if __name__ == "__main__":
    main()
