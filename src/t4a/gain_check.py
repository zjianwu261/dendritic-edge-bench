"""Single-task adaptation gain check: does adapting NoMaD on one task help on that task?

For each candidate task (env:light), and each learning rate, adapt a fresh NoMaD with LoRA (r=8, vision
encoder transformer; same loss as cl_run.py) for up to max(steps) steps, evaluating on the task's test
set (first eval_n samples, 2 DDPM samples) at every step count in --steps. Zero-shot is evaluated in the
same run with the same evaluation settings. One JSON per (task, lr) is written incrementally to
results/T4a/gain_check/, so an interrupted job can be resumed (finished (task, lr) are skipped).
--method ft runs full fine-tuning (bs 32 x accum 2, activation checkpointing) instead of LoRA.
make_gain_table() -> results/T4a/gain_check.csv.
Usage: python gain_check.py --tasks huron_bww8:normal huron_bww8:dark ... [--lrs 1e-4 3e-4] [--steps 1000 3000]
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cl_methods  # noqa: E402
from bench_data import Task, evaluate, norm_deltas  # noqa: E402
from cl_methods import add_lora, encode  # noqa: E402
from common import EXP, ddpm_scheduler, load_model  # noqa: E402
from zero_shot import mean_traj_ref  # noqa: E402

OUTD = os.path.join(EXP, "gain_check")


def cap_gpu_mem():
    mb = int(os.environ.get("DN_MEM_MB", "0"))
    if mb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(max(0.02, (mb - 450) / total))


def run_one(spec, lr, a, dev):
    tn, light = spec.split(":")
    sfx = "" if a.split == "test" else "_val"
    fn = os.path.join(OUTD, f"{a.method}_{tn}_{light}_lr{lr:g}_s{a.seed}{sfx}.json")
    if os.path.exists(fn):
        print("skip", fn, flush=True); return
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    model, cfg = load_model("nomad", dev, verbose=False)
    sched = ddpm_scheduler(cfg); alpha = float(cfg["alpha"]); venc = model.vision_encoder
    for p in model.parameters():
        p.requires_grad_(a.method == "ft")
    params = add_lora(venc, 8) if a.method == "lora" else list(model.parameters())
    bs, accum = (64, 1) if a.method == "lora" else (32, 2)
    cl_methods.GRAD_CKPT = a.method == "ft"
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if a.method == "lora" else 1e-2)
    t = Task(tn, light, device=dev, holdout_val=a.split == "val")
    if a.split == "val":  # select on validation recordings (training excludes them)
        t.test = t.val
    t.test = {k: v[: a.eval_n] for k, v in t.test.items()}
    ev = lambda: evaluate(model, sched, t, n_action_samples=a.samples, seed=1, bs=a.eval_bs,  # noqa: E731
                          cond_fn=lambda m, o, g, mk: encode(venc, o, g, mk))
    model.eval(); res = {"task": tn, "light": light, "lr": lr, "method": a.method, "seed": a.seed, "split": a.split,
                         "zero_shot": ev(), "ref_mean_traj": mean_traj_ref(t), "adapted": {}}
    print(spec, lr, "zero-shot act", round(res["zero_shot"]["act_mse_gc"], 4), flush=True)
    model.train(); t0 = time.time()
    for step in range(1, max(a.steps) + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            b = t.sample_train(bs, rng)
            obs, goal = t.images(b["ctx"], b["goal"])
            act = b["actions"].to(dev).float(); dist = b["dist"].to(dev).float(); amask = b["mask"].to(dev).float()
            gm = (torch.rand(bs, device=dev) < 0.5).long()
            c = encode(venc, obs, goal, gm); c_nm = encode(venc, obs, goal, torch.zeros_like(gm))
            dl = F.mse_loss(model("dist_pred_net", obsgoal_cond=c_nm).squeeze(-1), dist)
            na = norm_deltas(act); noise = torch.randn_like(na)
            ts = torch.randint(0, sched.config.num_train_timesteps, (bs,), device=dev)
            pred = model("noise_pred_net", sample=sched.add_noise(na, noise, ts), timestep=ts, global_cond=c)
            per = ((pred - noise) ** 2).mean(dim=(1, 2))
            loss = alpha * dl + (1 - alpha) * (per * amask).mean() / (amask.mean() + 1e-2)
            (loss / accum).backward()
        opt.step()
        if step in a.steps:
            model.eval(); r = ev(); model.train()
            r["train_s"] = round(time.time() - t0, 1)
            r["peak_MB"] = round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0
            res["adapted"][str(step)] = r
            print(spec, lr, step, "act", round(r["act_mse_gc"], 4), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)


def make_gain_table(split="test"):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTD, "*.json"))):
        d = json.load(open(f)); z = d["zero_shot"]
        if d.get("split", "test") != split:
            continue
        for st, r in d["adapted"].items():
            row = {"task": d["task"], "light": d["light"], "method": d["method"], "lr": d["lr"], "steps": int(st)}
            for met in ("act_mse_gc", "final_err_m", "eps_mse_gc", "dist_mse"):
                row[f"zs_{met}"] = round(z[met], 4); row[f"ad_{met}"] = round(r[met], 4)
                row[f"gain_pct_{met}"] = round(100 * (z[met] - r[met]) / z[met], 1)
            row["ref_mean_traj_act_mse"] = round(d["ref_mean_traj"]["act_mse"], 4)
            row["ad_over_ref_act_mse"] = round(r["act_mse_gc"] / d["ref_mean_traj"]["act_mse"], 3)
            row["zs_over_ref_act_mse"] = round(z["act_mse_gc"] / d["ref_mean_traj"]["act_mse"], 3)
            row["train_s"] = r["train_s"]
            rows.append(row)
    if rows:
        name = "gain_check.csv" if split == "test" else f"gain_check_{split}.csv"
        tmp = os.path.join(EXP, f"{name}.{os.getpid()}.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        os.replace(tmp, os.path.join(EXP, name))  # several queue jobs rebuild it
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    ap.add_argument("--steps", nargs="+", type=int, default=[1000, 3000])
    ap.add_argument("--method", default="lora", choices=["lora", "ft"])
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    a = ap.parse_args()
    cap_gpu_mem()
    torch.backends.mha.set_fastpath_enabled(False)
    for spec in a.tasks:
        for lr in a.lrs:
            run_one(spec, lr, a, a.device)
            make_gain_table(a.split)
    print("DONE")


if __name__ == "__main__":
    main()
