"""One continual-learning run of NoMaD on a T4a task sequence.

Loss = official NoMaD objective: (1-alpha) * diffusion eps-MSE (random goal masking p=0.5, action-masked)
       + alpha * distance MSE (goal-conditioned), alpha = 1e-4.
After each task: evaluate on every task of the sequence (fixed test subset) -> matrices R[i][j].
Summary (errors, lower is better): final average, BWT = mean_{j<T} (R[T][j] - R[j][j]) (> 0 = forgetting),
also relative to zero-shot.

Usage: python cl_run.py --method dend --context infer --seq huron_bww8:normal,huron_soda3:dark,... --seed 0
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
from cl_methods import BiasAdapter, DendriticAdapter, add_lora, add_lora_bank, encode, set_bank_active  # noqa: E402
from ctx_infer import ContextInferer  # noqa: E402
from common import EXP, ddpm_scheduler, load_model  # noqa: E402


def cap_gpu_mem():
    mb = int(os.environ.get("DN_MEM_MB", "0"))
    if mb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(max(0.02, (mb - 450) / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ft", "ft_head", "lora", "replay", "dend", "ewc", "lora_bank", "bias"])
    ap.add_argument("--context", default="id", choices=["id", "infer"])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=50)
    ap.add_argument("--buffer", type=int, default=500)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--dend_h", type=int, default=128)
    ap.add_argument("--dend_seg", type=int, default=8)
    ap.add_argument("--kwta", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ewc_lambda", type=float, default=1e4)
    ap.add_argument("--ewc_batches", type=int, default=50, help="batches for the diagonal Fisher after each task")
    ap.add_argument("--ewc_fisher", default="batch", choices=["batch", "sample"])
    ap.add_argument("--ewc_samples", type=int, default=256)
    ap.add_argument("--holdout_val", type=int, default=0, help="train without the validation recordings and also evaluate on them")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-batches (effective batch = bs*accum)")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="activation checkpointing of the EfficientNet trunks")
    ap.add_argument("--ctx_feat", default="v3", choices=["v2", "v3"], help="lora_bank context inferer features (ctx_infer.py)")
    ap.add_argument("--novel_q", type=float, default=0.95, help="new-context threshold quantile on validation distances (calibrated on v2sup)")
    # v3fix options; defaults reproduce the original runs
    ap.add_argument("--std_ref", default="first", choices=["first", "normal"],
                    help="lora_bank inferer standardisation reference (same option as cl_dend2.py)")
    ap.add_argument("--proto_eval", type=int, default=0, help="1: lora_bank prototypes / thresholds computed in eval() (original: train())")
    a = ap.parse_args()
    cap_gpu_mem()
    torch.backends.mha.set_fastpath_enabled(False)
    import cl_methods
    cl_methods.GRAD_CKPT = bool(a.grad_ckpt)
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    dev = a.device
    lr = a.lr or {"ft": 3e-5, "ft_head": 1e-4, "replay": 3e-5, "ewc": 3e-5, "lora": 5e-4, "dend": 5e-4,
                    "lora_bank": 3e-4, "bias": 3e-4}[a.method]
    model, cfg = load_model("nomad", dev, verbose=False)
    sched = ddpm_scheduler(cfg)
    alpha = float(cfg["alpha"])
    venc = model.vision_encoder
    n_backbone = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(a.method in ("ft", "replay", "ewc"))
    extra, tok_ad, out_ad, banks = [], None, None, None
    n_ctx = len(a.seq.split(","))
    if a.method == "lora_bank":  # one LoRA per context (rank lora_r each), routed by context
        banks = add_lora_bank(venc, n_ctx, a.lora_r)
        extra = [p for b in banks for p in b.parameters()]
    if a.method == "bias":
        out_ad = BiasAdapter(256).to(dev)
        extra = list(out_ad.parameters())
    if a.method == "ft_head":
        for mod in (model.noise_pred_net, model.dist_pred_net):
            for p in mod.parameters():
                p.requires_grad_(True)
    if a.method == "lora":
        extra = add_lora(venc, a.lora_r)
    if a.method == "dend":
        tok_ad = DendriticAdapter(256, a.dend_h, a.dend_seg, 256, a.kwta).to(dev)
        out_ad = DendriticAdapter(256, a.dend_h, a.dend_seg, 256, a.kwta).to(dev)
        extra = list(tok_ad.parameters()) + list(out_ad.parameters())
    params = list({id(p): p for p in [p for p in model.parameters() if p.requires_grad] + extra}.values())
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if a.method in ("lora", "dend", "lora_bank", "bias") else 1e-2)

    seq = [s.split(":") for s in a.seq.split(",")]
    tasks = [Task(n, l, device=dev, holdout_val=bool(a.holdout_val)) for n, l in seq]
    names = [f"{n}:{l}" for n, l in seq]
    for t in tasks:  # fixed eval subset (first eval_n of the fixed test set)
        t.test = {k: v[: a.eval_n] for k, v in t.test.items()}
        t.test_sub = t.test
        t.val_sub = {k: v[: a.eval_n] for k, v in t.val.items()} if t.val is not None else None
    protos = {}

    def frozen_feat(t, obs, goal):
        with torch.no_grad():
            return F.normalize(encode(venc, obs, goal, torch.ones(obs.shape[0], dtype=torch.long, device=dev)), dim=-1)

    def ctx_vec(feat, true_i):
        if a.context == "id" or not protos:
            return protos[true_i].expand(feat.shape[0], -1), None
        P = torch.stack([protos[k] for k in sorted(protos)])
        pick = (feat @ P.T).argmax(-1)
        return P[pick], torch.tensor(sorted(protos), device=dev)[pick]

    def cond(obs, goal, mask, z):
        return encode(venc, obs, goal, mask, tok_ad, out_ad, z)

    inferer = ContextInferer(venc, a.ctx_feat, a.novel_q) if a.method == "lora_bank" else None
    bank_stats = {"novel_frac_at_arrival": [], "per_ctx_trainable": (n_train // n_ctx) if banks else None}

    def bank_raw(obs, goal):  # inferer features from the frozen backbone (no LoRA applied)
        set_bank_active(banks, -1)
        return inferer.raw(obs, goal)

    def bank_cond(obs, goal, mask, k):
        """per-sample routing: run the encoder once per routed context and scatter back."""
        out = None
        for kk in k.unique().tolist():
            sel = (k == kk)
            set_bank_active(banks, kk)
            c = encode(venc, obs[sel], goal[sel], mask[sel])
            if out is None:
                out = torch.empty(obs.shape[0], c.shape[-1], device=c.device, dtype=c.dtype)
            out[sel] = c
        return out

    infer_acc = {}

    def eval_all(after):
        model.eval()
        row = {}
        for j, t in enumerate(tasks):
            hits, nov = [], []
            if a.method == "lora_bank":
                if j > after and a.context == "id":
                    row[names[j]] = None; continue

                def cfn(m, obs, goal, mk, j=j):
                    k = torch.full((obs.shape[0],), j, dtype=torch.long, device=obs.device)
                    if a.context == "infer" or mk.sum() == 0:
                        k_inf, novel = inferer.nearest(inferer.std(bank_raw(obs, goal)))
                        if mk.sum() == 0 and j <= after:
                            hits.append((k_inf == j).float()); nov.append(novel.float())
                        if a.context == "infer":
                            k = k_inf
                    return bank_cond(obs, goal, mk, k)
            elif a.method == "dend":
                if j not in protos and a.context == "id":
                    row[names[j]] = None; continue  # adapter has no context for an unseen task in the id setting

                def cfn(m, obs, goal, mk, j=j, t=t):
                    z, pick = ctx_vec(frozen_feat(t, obs, goal), j)
                    if pick is not None and mk.sum() == 0:
                        hits.append((pick == j).float())
                    return cond(obs, goal, mk, z)
            else:
                cfn = lambda m, obs, goal, mk: cond(obs, goal, mk, None)  # noqa: E731
            r = evaluate(model, sched, t, n_action_samples=a.samples, seed=1, cond_fn=cfn, bs=a.eval_bs)
            if hits:
                r["ctx_infer_acc"] = float(torch.cat(hits).mean()); infer_acc[(after, j, t.test is t.test_sub)] = r["ctx_infer_acc"]
            if nov:
                r["false_new"] = float(torch.cat(nov).mean())
            row[names[j]] = r
        model.train()
        return row

    def eval_both(after):
        """val first (if held out), then test (test is what infer_acc / the main summary use)."""
        out = {}
        if a.holdout_val:
            for t in tasks:
                t.test = t.val_sub
            out["row_val"] = eval_all(after)
        for t in tasks:
            t.test = t.test_sub
        out["row"] = eval_all(after)
        return out

    res = {"args": vars(a), "lr": lr, "names": names, "n_backbone_params": n_backbone, "n_trainable": n_train,
           "trainable_frac": n_train / n_backbone, "R": [], "train_s": [], "peak_MB": []}
    if a.method not in ("dend", "lora_bank", "bias"):  # zero-shot reference: taken from the other runs of the same order/seed
        res["R"].append({"after": "zero_shot", **eval_both(-1)})
    buf, seen = [], 0
    ewc_F, ewc_theta, res["ewc_penalty_log"] = None, None, []

    def task_loss(t_, b_):
        obs_, goal_ = t_.images(b_["ctx"], b_["goal"])
        act_ = b_["actions"].to(dev).float(); dist_ = b_["dist"].to(dev).float(); am_ = b_["mask"].to(dev).float()
        gm_ = (torch.rand(len(act_), device=dev) < 0.5).long()
        c_ = cond(obs_, goal_, gm_, None); cn_ = cond(obs_, goal_, torch.zeros_like(gm_), None)
        dl_ = F.mse_loss(model("dist_pred_net", obsgoal_cond=cn_).squeeze(-1), dist_)
        na_ = norm_deltas(act_); nz_ = torch.randn_like(na_)
        ts_ = torch.randint(0, sched.config.num_train_timesteps, (len(act_),), device=dev)
        pr_ = model("noise_pred_net", sample=sched.add_noise(na_, nz_, ts_), timestep=ts_, global_cond=c_)
        per_ = ((pr_ - nz_) ** 2).mean(dim=(1, 2))
        return alpha * dl_ + (1 - alpha) * (per_ * am_).mean() / (am_.mean() + 1e-2)
    model.train()
    for i, t in enumerate(tasks):
        if a.method == "dend":  # prototype of the new task from frozen features (no raw data kept)
            b = t.sample_train(512, np.random.RandomState(1000 + i))
            feats = []
            for s in range(0, 512, 32):
                obs, goal = t.images(b["ctx"][s:s + 32], b["goal"][s:s + 32])
                feats.append(frozen_feat(t, obs, goal))
            protos[i] = F.normalize(torch.cat(feats).mean(0), dim=0)
        if a.method == "lora_bank":  # shared inferer (ctx_infer.py, same as dend2): prototype + validation threshold
            if a.proto_eval:
                model.eval()
            if i == 0 and a.std_ref == "normal":  # same sample indices, normal-light frames of the first environment
                tref = Task(seq[0][0], "normal", device=dev, holdout_val=bool(a.holdout_val))
                b = tref.sample_train(512, np.random.RandomState(1000)); zr = []
                for s in range(0, 512, 32):
                    obs, goal = tref.images(b["ctx"][s:s + 32], b["goal"][s:s + 32]); zr.append(bank_raw(obs, goal))
                inferer.fit_std(torch.cat(zr)); del tref
            b = t.sample_train(512, np.random.RandomState(1000 + i)); zr = []
            for s in range(0, 512, 32):
                obs, goal = t.images(b["ctx"][s:s + 32], b["goal"][s:s + 32]); zr.append(bank_raw(obs, goal))
            zr = torch.cat(zr)
            if i == 0 and a.std_ref == "first":
                inferer.fit_std(zr)
            zp = inferer.std(zr)
            if i > 0:
                bank_stats["novel_frac_at_arrival"].append(inferer.novel_frac(zp))
            zv = None
            if t.val is not None:
                zv = []
                for s in range(0, len(t.val["goal"]), 50):
                    obs, goal = t.images(t.val["ctx"][s:s + 50], t.val["goal"][s:s + 50]); zv.append(bank_raw(obs, goal))
                zv = inferer.std(torch.cat(zv))
            inferer.add_context(zp, zv)
            set_bank_active(banks, i)
            model.train()
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None; t0 = time.time()
        for step in range(a.steps):
            opt.zero_grad(set_to_none=True)
            for micro in range(a.accum):
                b = t.sample_train(a.bs, rng); tid = torch.full((a.bs,), i)
                if a.method == "replay" and buf:
                    nr = a.bs // 2
                    pick = [buf[k] for k in rng.randint(len(buf), size=nr)]
                    rb = {k: torch.stack([p[1][k] for p in pick]) for k in b}
                    b = {k: torch.cat([b[k][: a.bs - nr], rb[k]]) for k in b}
                    tid = torch.cat([tid[: a.bs - nr], torch.tensor([p[0] for p in pick])])
                # images per source task (replayed samples come from earlier tasks' frame stores)
                obs_l, goal_l = [], []
                for k in tid.unique().tolist():
                    sel = (tid == k)
                    o, g = tasks[k].images(b["ctx"][sel], b["goal"][sel]); obs_l.append((sel, o, g))
                obs = torch.empty(a.bs, 12, 96, 96, device=dev); goal = torch.empty(a.bs, 3, 96, 96, device=dev)
                for sel, o, g in obs_l:
                    obs[sel.to(dev)] = o; goal[sel.to(dev)] = g
                act = b["actions"].to(dev).float(); dist = b["dist"].to(dev).float(); amask = b["mask"].to(dev).float()
                z = None
                if a.method == "dend":
                    z = protos[i].expand(a.bs, -1)
                gmask = (torch.rand(a.bs, device=dev) < 0.5).long()
                c = cond(obs, goal, gmask, z)
                c_nomask = cond(obs, goal, torch.zeros_like(gmask), z)
                dist_pred = model("dist_pred_net", obsgoal_cond=c_nomask).squeeze(-1)
                dist_loss = F.mse_loss(dist_pred, dist)
                na = norm_deltas(act)
                noise = torch.randn_like(na)
                ts = torch.randint(0, sched.config.num_train_timesteps, (a.bs,), device=dev)
                pred = model("noise_pred_net", sample=sched.add_noise(na, noise, ts), timestep=ts, global_cond=c)
                per = ((pred - noise) ** 2).mean(dim=(1, 2))
                diff_loss = (per * amask).mean() / (amask.mean() + 1e-2)
                loss = alpha * dist_loss + (1 - alpha) * diff_loss
                if a.method == "ewc" and ewc_F is not None:
                    pen = sum((f * (p - p0) ** 2).sum() for p, f, p0 in zip(params, ewc_F, ewc_theta))
                    loss = loss + 0.5 * a.ewc_lambda * pen
                (loss / a.accum).backward()
                if a.method == "replay":  # reservoir sampling over this task's fresh samples
                    for k in range(a.bs - (a.bs // 2 if buf else 0)):
                        seen += 1
                        item = (i, {kk: b[kk][k] for kk in b})
                        if len(buf) < a.buffer:
                            buf.append(item)
                        elif rng.rand() < a.buffer / seen:
                            buf[rng.randint(a.buffer)] = item
            opt.step()
            if step % 200 == 0:
                if a.method == "ewc" and ewc_F is not None:
                    res["ewc_penalty_log"].append({"task": names[i], "step": step, "penalty_term": float(0.5 * a.ewc_lambda * pen)})
                print(f"task {i} {names[i]} step {step} loss {loss.item():.4f} diff {diff_loss.item():.4f} dist {dist_loss.item():.2f}", flush=True)
        if a.method == "ewc":  # online EWC: accumulate the diagonal (empirical, per-batch) Fisher, re-anchor
            Fi = [torch.zeros_like(p) for p in params]
            if a.ewc_fisher == "sample":  # per-sample empirical Fisher (squared per-sample gradients)
                fb = t.sample_train(a.ewc_samples, rng)
                for q in range(a.ewc_samples):
                    opt.zero_grad(set_to_none=True)
                    task_loss(t, {k: v[q:q + 1] for k, v in fb.items()}).backward()
                    for f, p in zip(Fi, params):
                        if p.grad is not None:
                            f += p.grad.detach() ** 2 / a.ewc_samples
            else:
                for _ in range(a.ewc_batches):
                    opt.zero_grad(set_to_none=True)
                    task_loss(t, t.sample_train(a.bs, rng)).backward()
                    for f, p in zip(Fi, params):
                        if p.grad is not None:
                            f += p.grad.detach() ** 2 / a.ewc_batches
            opt.zero_grad(set_to_none=True)
            ewc_F = Fi if ewc_F is None else [f0 + f for f0, f in zip(ewc_F, Fi)]
            ewc_theta = [p.detach().clone() for p in params]
            res["ewc_penalty_log"].append({"task": names[i], "fisher_mean": float(sum(f.sum() for f in Fi) / sum(f.numel() for f in Fi))})
        res["train_s"].append(round(time.time() - t0, 1)); res["peak_MB"].append(round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0)
        res["R"].append({"after": names[i], **eval_both(i)})
        print(json.dumps({"after": names[i], "eps_gc": {k: (v or {}).get("eps_mse_gc") for k, v in res["R"][-1]["row"].items()},
                          "act_gc": {k: (v or {}).get("act_mse_gc") for k, v in res["R"][-1]["row"].items()}}), flush=True)
    # summary
    T = len(tasks)

    def summarize(key):
        rows = [r[key] for r in res["R"] if r["after"] != "zero_shot"]
        zs = next((r[key] for r in res["R"] if r["after"] == "zero_shot"), None)
        sm = {}
        for met in ("eps_mse_gc", "act_mse_gc", "final_err_m", "dist_mse"):
            fin = [rows[T - 1][n][met] for n in names]
            diag = [rows[j][names[j]][met] for j in range(T)]
            sm[met] = {"final_avg": float(np.mean(fin)), "diag_avg": float(np.mean(diag)),
                       "bwt": float(np.mean([fin[j] - diag[j] for j in range(T - 1)])), "final": fin, "diag": diag}
            if zs:
                sm[met]["zero_shot"] = [zs[n][met] for n in names]
        return sm
    summ = summarize("row")
    if a.holdout_val:
        res["summary_val"] = summarize("row_val")
    if infer_acc:  # key = (after, task, evaluated on test); the main number is on the test split
        summ["ctx_infer_acc_final"] = float(np.mean([v for (aft, j, is_test), v in infer_acc.items() if aft == T - 1 and is_test]))
        if a.holdout_val:
            res["summary_val"]["ctx_infer_acc_final"] = float(np.mean([v for (aft, j, is_test), v in infer_acc.items() if aft == T - 1 and not is_test]))
    if a.method == "lora_bank":
        summ["novel_frac_at_arrival"] = bank_stats["novel_frac_at_arrival"]
        fn_ = [res["R"][-1]["row"][n].get("false_new") for n in names]
        summ["false_new_final"] = float(np.mean([x for x in fn_ if x is not None]))
        res["per_ctx_trainable"] = bank_stats["per_ctx_trainable"]; res["inferer"] = {"feat_set": a.ctx_feat, "q": a.novel_q,
                                                                                     "taus": [float(x) for x in inferer.taus]}
    res["summary"] = summ
    out = os.path.join(EXP, "cl"); os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"{a.method}{'_' + a.context if a.method in ('dend', 'lora_bank') else ''}{a.tag}_s{a.seed}.json")
    json.dump(res, open(fn, "w"), indent=1)
    if a.method == "dend":  # adapter weights + prototypes for diagnostics (dend_diag.py)
        os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
        torch.save({"tok": tok_ad.state_dict(), "out": out_ad.state_dict(), "protos": {k: v.cpu() for k, v in protos.items()},
                    "names": names, "args": vars(a)}, fn.replace(".json", ".pt").replace("/cl/", "/cl/ckpt/"))
    print("SUMMARY", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('final_avg', 'bwt', 'diag_avg')}) for k, v in summ.items()}))
    print("saved", fn)


if __name__ == "__main__":
    main()
