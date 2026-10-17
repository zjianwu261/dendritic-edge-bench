"""Build one T4a continual-learning task from a processed GNM-format dataset folder.

Output $DEB_DATA/t4a_bench/<task>/:
  frames.npy       uint8 [N,96,96,3]  official preprocessing (center-crop to 4:3, resize 96x96)
  frames_dark.npy  uint8 [N,96,96,3]  synthetic low light (lowlight.py; params per trajectory), darkened
                                      at stored resolution before crop/resize
  meta.npz         traj_id [N], pos [N,2], yaw [N], traj_start [T+1], traj_names, traj_split (0 train/1 test),
                   metric_spacing, end_slack
  test.npz         fixed test samples (seed 0): ctx [M,4] frame idx, goal [M], actions [M,8,2], dist [M],
                   neg [M], mask [M]
  scan.npy/depth.npy (HuRoN only) per frame, for multimodal experiments
Labels follow visualnav-transformer ViNT_Dataset (waypoint_spacing 1, context 3, len_traj_pred 8,
distance 0..20 with negative mining -> 20, action mask 3 < d < 20 and not negative, actions in local
frame divided by metric_waypoint_spacing). Train/test split is by recording (bag / source file), 80/20,
so sub-trajectories cut from the same recording never straddle the split.
"""
import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
from lowlight import darken, sample_params  # noqa: E402
from vint_train.data.data_utils import to_local_coords  # noqa: E402

CTX, H, DMAX, AMIN = 3, 8, 20, 3
OUT = os.path.join(common.DEB_DATA, "t4a_bench")


def crop_resize(img, size=(96, 96)):
    w, h = img.size
    if w > h:
        nw = int(h * 4 / 3); l = int(round((w - nw) / 2.0)); img = img.crop((l, 0, l + nw, h))
    else:
        nh = int(w / (4 / 3)); t = int(round((h - nh) / 2.0)); img = img.crop((0, t, w, t + nh))
    return np.asarray(img.resize(size))


def valid_currs(L, end_slack):
    return range(CTX, L - end_slack - H)


def make_sample(meta, k_traj, curr, rng, goals_pool):
    """One sample, same rules as ViNT_Dataset.__getitem__ / _compute_actions."""
    s = meta["traj_start"][k_traj]; L = meta["traj_start"][k_traj + 1] - s
    maxg = min(DMAX, L - curr - 1)
    off = rng.randint(0, maxg + 1)
    if off == 0:
        goal, dist, neg = int(goals_pool[rng.randint(len(goals_pool))]), DMAX, True
    else:
        goal, dist, neg = s + curr + off, off, False
    pos = meta["pos"][s + curr: s + curr + H + 1]; yaw = meta["yaw"][s + curr: s + curr + H + 1]
    wp = to_local_coords(pos, pos[0], yaw[0])[1:] / meta["metric_spacing"]
    ctx = [s + curr - CTX + i for i in range(CTX + 1)]
    mask = (dist < DMAX) and (dist > AMIN) and not neg
    return ctx, goal, wp.astype(np.float32), dist, neg, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--spacing", type=float, required=True, help="metric_waypoint_spacing (data_config.yaml)")
    ap.add_argument("--end_slack", type=int, default=0)
    ap.add_argument("--include", default=".*", help="regex on trajectory folder name")
    ap.add_argument("--exclude", default="^$")
    ap.add_argument("--env", default="", help="HuRoN: keep only meta.json env == this ('+'-separated list)")
    ap.add_argument("--group", default=r"^(.*)_\d+$", help="regex; group(1) = recording id for the split")
    ap.add_argument("--max_frames", type=int, default=40000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    names = sorted(n for n in os.listdir(a.src) if re.search(a.include, n) and not re.search(a.exclude, n)
                   and os.path.exists(os.path.join(a.src, n, "traj_data.pkl")))
    if a.env:
        envs = set(a.env.split("+"))
        names = [n for n in names if json.load(open(os.path.join(a.src, n, "meta.json")))["env"] in envs]
    # keep trajectories long enough for at least one sample
    trajs = {}
    for n in names:
        td = pickle.load(open(os.path.join(a.src, n, "traj_data.pkl"), "rb"))
        # GoStanford pickles hold object arrays of 1-element arrays; flatten to float
        L = len(td["position"])
        td = {"position": np.array([np.asarray(v, float).reshape(-1) for v in td["position"]]).reshape(L, 2),
              "yaw": np.array([float(np.asarray(v).reshape(-1)[0]) for v in td["yaw"]])}
        if len(valid_currs(L, a.end_slack)) > 0:
            trajs[n] = td
    groups = sorted({re.match(a.group, n).group(1) for n in trajs})
    rng.shuffle(groups)
    n_test_g = max(1, int(round(0.2 * len(groups))))
    test_groups = set(groups[:n_test_g])
    order = [n for n in sorted(trajs) if re.match(a.group, n).group(1) in test_groups] + \
            [n for n in sorted(trajs) if re.match(a.group, n).group(1) not in test_groups]
    # cap frames: keep all test trajectories (up to half the budget), random subset of train trajectories
    tr = [n for n in order if re.match(a.group, n).group(1) not in test_groups]
    te = [n for n in order if re.match(a.group, n).group(1) in test_groups]
    rng.shuffle(tr); rng.shuffle(te)
    sel, tot = [], 0
    for n in te:
        if tot + len(trajs[n]["position"]) > a.max_frames // 4:
            break
        sel.append((n, 1)); tot += len(trajs[n]["position"])
    for n in tr:
        if tot + len(trajs[n]["position"]) > a.max_frames:
            continue
        sel.append((n, 0)); tot += len(trajs[n]["position"])
    print(f"{a.task}: {len(trajs)} trajs in {len(groups)} recordings; selected {len(sel)} trajs, {tot} frames "
          f"({sum(s for _, s in sel)} test trajs)", flush=True)
    out = os.path.join(OUT, a.task); os.makedirs(out, exist_ok=True)
    frames = np.zeros((tot, 96, 96, 3), np.uint8); dark = np.zeros_like(frames)
    pos = np.zeros((tot, 2)); yaw = np.zeros(tot); tid = np.zeros(tot, np.int32)
    starts = [0]; extra = os.path.exists(os.path.join(a.src, sel[0][0], "scan.npy"))
    scans, depths, dark_params = [], [], []
    drng = np.random.RandomState(a.seed + 1000)
    for k, (n, _) in enumerate(sel):
        td = trajs[n]; L = len(td["position"]); s = starts[-1]
        p = sample_params(drng); dark_params.append(p)
        for i in range(L):
            img = Image.open(os.path.join(a.src, n, f"{i}.jpg")).convert("RGB")
            frames[s + i] = crop_resize(img)
            dark[s + i] = crop_resize(Image.fromarray(darken(np.asarray(img), p, drng)))
        pos[s:s + L] = td["position"]; yaw[s:s + L] = np.asarray(td["yaw"]).reshape(-1); tid[s:s + L] = k
        if extra:
            scans.append(np.load(os.path.join(a.src, n, "scan.npy"))); depths.append(np.load(os.path.join(a.src, n, "depth.npy")))
            assert len(scans[-1]) == L and len(depths[-1]) == L, n
        starts.append(s + L)
    meta = dict(traj_id=tid, pos=pos, yaw=yaw, traj_start=np.array(starts), traj_names=np.array([n for n, _ in sel]),
                traj_split=np.array([s for _, s in sel]), metric_spacing=a.spacing, end_slack=a.end_slack,
                dark_params=np.array(dark_params))
    np.save(os.path.join(out, "frames.npy"), frames); np.save(os.path.join(out, "frames_dark.npy"), dark)
    np.savez(os.path.join(out, "meta.npz"), **meta)
    modality = {}
    if extra:
        # Some HuRoN bags have no /laserscan messages (or empty ranges) -> scan width 0 for that trajectory.
        # Kept as a *missing modality*: NaN-filled scan + scan_valid=False (usable for multimodal experiments).
        nb = max(x.shape[1] for x in scans)
        sc_all = np.full((tot, nb), np.nan, np.float16)
        for k, x in enumerate(scans):
            if x.shape[1] == nb:
                sc_all[starts[k]:starts[k + 1]] = x
        scan_valid = np.isfinite(sc_all.astype(np.float32)).any(1)
        dp_all = np.concatenate(depths); depth_valid = dp_all.reshape(tot, -1).max(1) > 0
        np.save(os.path.join(out, "scan.npy"), sc_all); np.save(os.path.join(out, "depth.npy"), dp_all)
        np.savez(os.path.join(out, "modality_valid.npz"), scan=scan_valid, depth=depth_valid)
        modality = {"scan_width": int(nb), "trajs_without_scan": int(sum(x.shape[1] != nb for x in scans)),
                    "frames_scan_missing": int((~scan_valid).sum()), "frames_depth_missing": int((~depth_valid).sum())}
    # fixed test samples
    test_k = [k for k, (_, s) in enumerate(sel) if s == 1]
    pool = np.concatenate([np.arange(starts[k], starts[k + 1]) for k in test_k])
    cand = [(k, c) for k in test_k for c in valid_currs(starts[k + 1] - starts[k], a.end_slack)]
    trng = np.random.RandomState(a.seed + 1)
    pick = trng.choice(len(cand), size=min(a.n_test, len(cand)), replace=False)
    rows = [make_sample(meta, cand[i][0], cand[i][1], trng, pool) for i in pick]
    np.savez(os.path.join(out, "test.npz"), ctx=np.array([r[0] for r in rows]), goal=np.array([r[1] for r in rows]),
             actions=np.stack([r[2] for r in rows]), dist=np.array([r[3] for r in rows]),
             neg=np.array([r[4] for r in rows]), mask=np.array([r[5] for r in rows]))
    info = dict(task=a.task, src=a.src, n_frames=int(tot), n_trajs=len(sel), n_test_trajs=len(test_k),
                n_train_frames=int(sum(starts[k + 1] - starts[k] for k, (_, s) in enumerate(sel) if s == 0)),
                n_test_samples=len(rows), n_test_masked=int(sum(r[5] for r in rows)), recordings=len(groups),
                test_recordings=sorted(test_groups), args=vars(a),
                modality=modality, mean_brightness=float(frames.mean()), mean_brightness_dark=float(dark.mean()))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "test_recordings"}))


if __name__ == "__main__":
    main()
