"""Validation split for every T4a benchmark task.
From the TRAIN trajectories of each task, hold out ~15% of the training frames, chosen by whole
recordings (same recording regex as build_bench.py, seed 0), as a validation set; build 500 fixed
validation samples (same labelling as the test set). Saved as $DEB_DATA/t4a_bench/<task>/val.npz
(val_traj = trajectory indices; ctx/goal/actions/dist/neg/mask). The test split is untouched; the
light variants (normal / dark / occl) share it. Runs on CPU in seconds.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bench import OUT, make_sample, valid_currs  # noqa: E402

TASKS = ["huron_bww8", "huron_soda3", "huron_cory1", "huron_bww12", "gostanford", "recon"]


def main(frac=0.15, n_val=500, seed=0):
    for t in TASKS:
        d = os.path.join(OUT, t)
        m = dict(np.load(os.path.join(d, "meta.npz")))
        info = json.load(open(os.path.join(d, "info.json")))
        grp = info["args"]["group"]
        names, split, st = m["traj_names"], m["traj_split"], m["traj_start"]
        es = int(m["end_slack"])
        tr = [k for k in range(len(split)) if split[k] == 0]
        rec = {}
        for k in tr:
            rec.setdefault(re.match(grp, str(names[k])).group(1), []).append(k)
        keys = sorted(rec); rng = np.random.RandomState(seed); rng.shuffle(keys)
        total = sum(st[k + 1] - st[k] for k in tr); val, acc = [], 0
        for r in keys:
            if acc >= frac * total:
                break
            val += rec[r]; acc += sum(st[k + 1] - st[k] for k in rec[r])
        m["metric_spacing"] = float(m["metric_spacing"])
        pool = np.concatenate([np.arange(st[k], st[k + 1]) for k in val])
        cand = [(k, c) for k in val for c in valid_currs(st[k + 1] - st[k], es)]
        vr = np.random.RandomState(seed + 2)
        pick = vr.choice(len(cand), size=min(n_val, len(cand)), replace=False)
        rows = [make_sample(m, cand[i][0], cand[i][1], vr, pool) for i in pick]
        np.savez(os.path.join(d, "val.npz"), val_traj=np.array(sorted(val)), ctx=np.array([r[0] for r in rows]),
                 goal=np.array([r[1] for r in rows]), actions=np.stack([r[2] for r in rows]),
                 dist=np.array([r[3] for r in rows]), neg=np.array([r[4] for r in rows]), mask=np.array([r[5] for r in rows]))
        print(t, f"val recordings {len([r for r in keys if rec[r][0] in val])}/{len(keys)}, val frames {acc}/{total} "
              f"({100 * acc / total:.1f}%), val samples {len(rows)} (masked {sum(r[5] for r in rows)})")


if __name__ == "__main__":
    main()
