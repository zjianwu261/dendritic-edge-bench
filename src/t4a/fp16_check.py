"""Numerical check of the Orin TRT FP16 NoMaD explore pipeline against PyTorch FP32.
  prep   : N real test samples (default huron_bww8:normal, first N of the fixed test set) + fixed noise ->
           results/T4a/raw/orin1/fp16_check_in.npz, and the PyTorch reference (same DDPM update and
           noise, goal masked, n=8 samples) -> fp16_check_ref.npz
  compare: reads fp16_check_out.npz (from fp16_check_orin.py on Orin) -> fp16_check.json: encoder output
           error (abs / relative), waypoint error between TRT and PyTorch for identical noise, and the
           exploration action MSE vs labels (mean over the 8 samples, action-masked) for both.
Usage: python fp16_check.py prep [--n 200] ; python fp16_check.py compare
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_data import Task  # noqa: E402
from cl_methods import encode  # noqa: E402
from common import EXP, ddpm_scheduler, load_model  # noqa: E402
from export_onnx import DDPMUnrolled  # noqa: E402

RAW = os.path.join(EXP, "raw", "orin1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "compare"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--task", default="huron_bww8:normal")
    a = ap.parse_args()
    if a.mode == "prep":
        torch.backends.mha.set_fastpath_enabled(False)
        m, cfg = load_model("nomad", "cpu", verbose=False)
        t = Task(*a.task.split(":"), device="cpu")
        ctx, goal_i = t.test["ctx"][: a.n], t.test["goal"][: a.n]
        obs, goal = t.images(ctx, goal_i)
        g = torch.Generator().manual_seed(7)
        z0 = torch.randn(a.n, 8, 8, 2, generator=g); z = torch.randn(a.n, 10, 8, 8, 2, generator=g)
        un = DDPMUnrolled(m, ddpm_scheduler(cfg), 8).eval()
        conds, acts = [], []
        with torch.no_grad():
            for i in range(a.n):
                c = encode(m.vision_encoder, obs[i:i + 1], goal[i:i + 1], torch.ones(1, dtype=torch.long))
                conds.append(c[0].numpy())
                # DDPMUnrolled.forward recomputes the encoder; reuse its update with our cond
                x = z0[i]; cc = c.repeat(8, 1)
                for k, tt in enumerate(un.ts):
                    eps = m.noise_pred_net(sample=x, timestep=torch.full((8,), float(tt)), global_cond=cc)
                    co = un.coef[k]
                    x0 = torch.clamp((x - co["sb"] * eps) / co["sa"], -1, 1)
                    x = co["cx0"] * x0 + co["cxt"] * x
                    if tt > 0:
                        x = x + co["std"] * z[i, k]
                d = (x + 1) / 2 * (torch.tensor([7.5, 8.0])) + torch.tensor([-2.5, -4.0])
                acts.append(torch.cumsum(d, 1).numpy())
        np.savez(os.path.join(RAW, "fp16_check_in.npz"), obs=obs.numpy(), goal=goal.numpy(), z0=z0.numpy(), z=z.numpy())
        np.savez(os.path.join(RAW, "fp16_check_ref.npz"), cond=np.array(conds), actions=np.array(acts),
                 labels=t.test["actions"][: a.n].numpy(), mask=t.test["mask"][: a.n].numpy(), task=a.task)
        print("prep done", a.n)
    else:
        r = np.load(os.path.join(RAW, "fp16_check_ref.npz")); o = np.load(os.path.join(RAW, "fp16_check_out.npz"))
        cr, ct = r["cond"], o["cond"]
        ar, at = r["actions"], o["actions"]
        lab, mk = r["labels"], r["mask"].astype(bool)
        mse = lambda A: float((((A - lab[:, None]) ** 2).mean(axis=(2, 3)).mean(1))[mk].mean())  # noqa: E731
        res = {"task": str(r["task"]), "n": int(len(cr)), "engines": [str(x) for x in o["engines"]],
               "enc_abs_err_max": float(np.abs(ct - cr).max()), "enc_abs_err_mean": float(np.abs(ct - cr).mean()),
               "enc_rel_err_l2_mean": float((np.linalg.norm(ct - cr, axis=1) / np.linalg.norm(cr, axis=1)).mean()),
               "enc_cos_min": float(((ct * cr).sum(1) / np.linalg.norm(ct, axis=1) / np.linalg.norm(cr, axis=1)).min()),
               "wp_abs_err_mean_norm_units": float(np.abs(at - ar).mean()), "wp_abs_err_max_norm_units": float(np.abs(at - ar).max()),
               "final_wp_err_median_norm_units": float(np.median(np.linalg.norm(at[:, :, -1] - ar[:, :, -1], axis=-1))),
               "act_mse_uc_pytorch": mse(ar), "act_mse_uc_trt": mse(at), "n_masked": int(mk.sum())}
        json.dump(res, open(os.path.join(EXP, "fp16_check.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
