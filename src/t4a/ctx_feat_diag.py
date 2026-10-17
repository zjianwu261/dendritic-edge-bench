"""Dump raw context features (v3 set) of the 5 v3 tasks for offline checks of the
context inferer on the VALIDATION split (prototype samples = the same 512 training samples cl_dend2 uses).
Output: results/T4a/diag/v3ctx/raw_<task>.pt  {proto: [512, 274], val: [n_val, 274], test: [500, 274]}"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_data import Task  # noqa
from common import EXP, load_model  # noqa
from ctx_infer import ContextInferer  # noqa

TASKS = ["huron_bww8:dark", "recon:occl", "huron_cory1:dark", "huron_bww8:occl", "recon:dark"]
ORDERS = {"main": TASKS, "shuf": ["recon:occl", "huron_bww8:occl", "huron_bww8:dark", "huron_cory1:dark", "recon:dark"]}


def main(dev="cpu"):
    torch.set_num_threads(16); torch.backends.mha.set_fastpath_enabled(False)
    model, _ = load_model("nomad", dev, verbose=False)
    inf = ContextInferer(model.vision_encoder, "v3")
    od = os.path.join(EXP, "diag", "v3ctx"); os.makedirs(od, exist_ok=True)
    for name in TASKS:
        fn = os.path.join(od, f"raw_{name.replace(':', '_')}.pt")
        if os.path.exists(fn):
            continue
        n, l = name.split(":"); t = Task(n, l, device=dev, holdout_val=True)
        out = {}
        for i_main in [ORDERS["main"].index(name), ORDERS["shuf"].index(name)]:
            b = t.sample_train(512, np.random.RandomState(1000 + i_main)); zs = []
            for s in range(0, 512, 64):
                obs, goal = t.images(b["ctx"][s:s + 64], b["goal"][s:s + 64]); zs.append(inf.raw(obs, goal))
            out[f"proto_i{i_main}"] = torch.cat(zs)
        for split, d in (("val", t.val), ("test", {k: v[:500] for k, v in t.test.items()})):
            zs = []
            for s in range(0, len(d["goal"]), 64):
                obs, goal = t.images(d["ctx"][s:s + 64], d["goal"][s:s + 64]); zs.append(inf.raw(obs, goal))
            out[split] = torch.cat(zs)
        torch.save(out, fn); print(name, {k: tuple(v.shape) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:]))
