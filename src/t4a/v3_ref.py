"""Mean-trajectory reference for the v3 CL tasks.

For each task name (env:light) of the v3 sequence, the reference is zero_shot.mean_traj_ref: the mean training
trajectory (5000 training samples, RandomState(0), train split WITHOUT the validation recordings, as cl_run.py
--holdout_val 1 trains) scored on exactly the evaluation samples cl_run.py uses (first eval_n of the test set /
of the validation set).  The light condition does not change actions, so dark / occl tasks of one env share the
reference of that env.  CPU only, no frames are loaded.

Cached in results/T4a/v3_mean_traj_ref.json ({"<env>": {"test": {act_mse, final_err_m}, "val": {...}}, ...}).
Usage: python v3_ref.py [--eval_n 500]  (also imported by v3_verdict.py / make_tables.py)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bench import OUT, valid_currs  # noqa: E402
from common import EXP  # noqa: E402
from bench_data import Task  # noqa: E402
from zero_shot import mean_traj_ref  # noqa: E402

CACHE = os.path.join(EXP, "v3_mean_traj_ref.json")


class LabelTask(Task):
    """Task without frames (labels / sampling only)."""

    def __init__(self, name, holdout_val=True):
        d = os.path.join(OUT, name)
        self.name, self.light, self.device = name, "normal", "cpu"
        m = np.load(os.path.join(d, "meta.npz"))
        self.meta = {k: m[k] for k in m.files}
        self.meta["metric_spacing"] = float(self.meta["metric_spacing"])
        t = np.load(os.path.join(d, "test.npz"))
        self.test = {k: torch.from_numpy(t[k]) for k in t.files}
        st, sp = self.meta["traj_start"], self.meta["traj_split"]
        self.train_k = [k for k in range(len(sp)) if sp[k] == 0]
        self.val = None
        if holdout_val:
            v = np.load(os.path.join(d, "val.npz"))
            vk = set(int(k) for k in v["val_traj"])
            self.train_k = [k for k in self.train_k if k not in vk]
            self.val = {k: torch.from_numpy(v[k]) for k in ("ctx", "goal", "actions", "dist", "neg", "mask")}
        self.train_pool = np.concatenate([np.arange(st[k], st[k + 1]) for k in self.train_k])
        es = int(self.meta["end_slack"])
        self.train_cand = np.array([(k, c) for k in self.train_k for c in valid_currs(st[k + 1] - st[k], es)])


def compute(envs, eval_n=500):
    out = {}
    for e in envs:
        t = LabelTask(e, holdout_val=True)
        te = {k: v[:eval_n] for k, v in t.test.items()}
        va = {k: v[:eval_n] for k, v in t.val.items()}
        out[e] = {"test": mean_traj_ref(t, te), "val": mean_traj_ref(t, va), "eval_n": eval_n, "holdout_val": True}
    return out


def get_refs(names, eval_n=500):
    """names: task names like 'huron_bww8:dark' -> {name: {"test": {...}, "val": {...}}} (cached per env)."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    envs = sorted({n.split(":")[0] for n in names})
    miss = [e for e in envs if e not in cache or cache[e].get("eval_n") != eval_n]
    if miss:
        cache.update(compute(miss, eval_n))
        json.dump(cache, open(CACHE, "w"), indent=1)
    return {n: cache[n.split(":")[0]] for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_n", type=int, default=500)
    a = ap.parse_args()
    r = get_refs(["huron_bww8:dark", "recon:occl", "huron_cory1:dark"], a.eval_n)
    print(json.dumps(r, indent=1))
