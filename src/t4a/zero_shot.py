"""Zero-shot NoMaD metrics on every T4a task x lighting, plus a trivial reference predictor.

Reference ("mean_traj"): predicts the mean training-set trajectory of the task for every sample
(no image input) -> action MSE / final error a model must beat to be using the images at all.
Output: results/T4a/zero_shot/<task>_<light>.json, and a summary CSV (make_zero_shot_table).
Usage: python zero_shot.py --tasks gostanford huron_cory1 ... [--lights normal dark] [--bs 100]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_data import Task, evaluate  # noqa: E402
from common import EXP, ddpm_scheduler, load_model  # noqa: E402


def cap_gpu_mem():
    mb = int(os.environ.get("DN_MEM_MB", "0"))
    if mb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(max(0.02, (mb - 450) / total))


def mean_traj_ref(task, ev=None):
    """ev: evaluation sample dict (default task.test)."""
    tr = task.sample_train(5000, np.random.RandomState(0))
    keep = tr["mask"]
    mu = tr["actions"][keep].mean(0)
    te = task.test if ev is None else ev
    m = te["mask"]
    err = ((te["actions"][m] - mu) ** 2).mean(dim=(1, 2))
    fe = (te["actions"][m][:, -1] - mu[-1]).norm(dim=-1) * task.meta["metric_spacing"]
    return {"act_mse": float(err.mean()), "final_err_m": float(fe.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--lights", nargs="+", default=["normal", "dark"])
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--device", default="cuda", help="cpu is fine for eval (~1-2 min per task x light)")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    cap_gpu_mem()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out_dir = os.path.join(EXP, "zero_shot"); os.makedirs(out_dir, exist_ok=True)
    model, cfg = load_model("nomad", a.device)
    sched = ddpm_scheduler(cfg)
    for tn in a.tasks:
        for light in a.lights:
            t0 = time.time()
            task = Task(tn, light, device=a.device)
            res = evaluate(model, sched, task, n_action_samples=a.samples, bs=a.bs)
            res.update(task=tn, light=light, ref_mean_traj=mean_traj_ref(task), seconds=round(time.time() - t0, 1),
                       device=a.device,
                       peak_gpu_MB=round(torch.cuda.max_memory_allocated() / 2**20) if a.device == "cuda" else 0, model="nomad (official ckpt)",
                       torch=torch.__version__, tf32=False)
            print(json.dumps(res), flush=True)
            json.dump(res, open(os.path.join(out_dir, f"{tn}_{light}.json"), "w"), indent=1)
            del task


if __name__ == "__main__":
    main()
