"""Process SACSoN/HuRoN rosbags into the visualnav-transformer layout, plus extra modalities for T4a ②.

Same logic as visualnav-transformer train/process_bags.py (dataset "sacson" in process_bags_config.yaml):
  - topics /fisheye_image/compressed + /odometry, sampled at 4 Hz with the official "latest message
    once >= 1/rate s have passed" rule (get_images_and_odom), yaw from quaternion, ang_offset 0
  - filter_backwards() (official function, imported) cuts non-forward segments into sub-trajectories
  - images saved as <i>.jpg (full-res decoded fisheye, as the official process_sacson_img), traj_data.pkl
Differences: ROS1 `rosbag` is replaced by the pure-python `rosbags` reader. Extra per-sample files
(not used by the official loader): scan.npy [T, n_beams] float16 (/laserscan ranges, latest message at
the sample time), depth.npy [T, 64, 128] uint8 (/depth_spherical_image, estimated depth, resized),
meta.json (env, session, bag, timestamps).
Usage: python process_huron.py [--in $DEB_DATA/raw/huron --out $DEB_DATA/gnm/huron] [--workers 8]
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401  (sets sys.path for vint_train)
import types  # noqa: E402
sys.modules.setdefault("rosbag", types.SimpleNamespace(Bag=object))  # utils import ROS1 rosbag at top level; unused here
from vint_train.process_data.process_data_utils import filter_backwards, quat_to_yaw  # noqa: E402

IM, OD, SC, DP = "/fisheye_image/compressed", "/odometry", "/laserscan", "/depth_spherical_image/compressed"


def env_of(session):
    for e in ("bww8", "bww1", "bww2", "cory1", "soda3"):
        if e in session:
            return e
    return "unknown"


def process_bag(args):
    bag, out_root, rate = args
    from rosbags.highlevel import AnyReader
    sess = bag.parent.name
    traj_name = f"{sess}_{bag.stem}"
    if any(p.name.startswith(traj_name + "_") for p in Path(out_root).glob(traj_name + "_*")):
        return traj_name, "skip"
    try:
        with AnyReader([bag]) as r:
            conns = [c for c in r.connections if c.topic in (IM, OD, SC, DP)]
            if not {c.topic for c in conns} >= {IM, OD}:
                return traj_name, "missing topics"
            cur = {IM: None, OD: None, SC: None, DP: None}
            samples = []
            currtime = r.start_time / 1e9
            for c, t, raw in r.messages(connections=conns):
                cur[c.topic] = (c, raw)
                ts = t / 1e9
                if ts - currtime >= 1.0 / rate and cur[IM] is not None and cur[OD] is not None:
                    samples.append((dict(cur), ts))
                    currtime = ts
            imgs, pos, yaw, scans, depths, stamps = [], [], [], [], [], []
            for cur_s, ts in samples:
                im = r.deserialize(cur_s[IM][1], cur_s[IM][0].msgtype)
                od = r.deserialize(cur_s[OD][1], cur_s[OD][0].msgtype)
                img = cv2.imdecode(np.frombuffer(bytes(im.data), np.uint8), cv2.IMREAD_COLOR)
                imgs.append(img)
                p, q = od.pose.pose.position, od.pose.pose.orientation
                pos.append([p.x, p.y]); yaw.append(quat_to_yaw(q.x, q.y, q.z, q.w))
                if cur_s[SC] is not None:
                    sc = r.deserialize(cur_s[SC][1], cur_s[SC][0].msgtype)
                    scans.append(np.asarray(sc.ranges, np.float32))
                else:
                    scans.append(None)
                if cur_s[DP] is not None:
                    dm = r.deserialize(cur_s[DP][1], cur_s[DP][0].msgtype)
                    d = cv2.imdecode(np.frombuffer(bytes(dm.data), np.uint8), cv2.IMREAD_GRAYSCALE)
                    depths.append(cv2.resize(d, (128, 64), interpolation=cv2.INTER_AREA))
                else:
                    depths.append(None)
                stamps.append(ts)
    except Exception as e:  # noqa: BLE001
        return traj_name, f"error {e!r}"
    if len(imgs) < 2:
        return traj_name, "too short"
    traj = {"position": np.array(pos), "yaw": np.array(yaw)}
    cuts = filter_backwards(list(range(len(imgs))), traj)  # carry sample indices through the official cutter
    nb = max((len(s) for s in scans if s is not None), default=0)
    n_out = 0
    for k, (idx, td) in enumerate(cuts):
        idx = list(idx)
        folder = Path(out_root) / f"{traj_name}_{k}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "traj_data.pkl", "wb") as f:
            pickle.dump(td, f)
        for j, i in enumerate(idx):
            cv2.imwrite(str(folder / f"{j}.jpg"), imgs[i])
        sc = np.full((len(idx), nb), np.nan, np.float16)
        dp = np.zeros((len(idx), 64, 128), np.uint8)
        for j, i in enumerate(idx):
            if scans[i] is not None and len(scans[i]) == nb:
                sc[j] = scans[i]
            if depths[i] is not None:
                dp[j] = depths[i]
        np.save(folder / "scan.npy", sc); np.save(folder / "depth.npy", dp)
        json.dump({"env": env_of(sess), "session": sess, "bag": bag.name, "stamps": [stamps[i] for i in idx]},
                  open(folder / "meta.json", "w"))
        n_out += 1
    return traj_name, f"ok {n_out} trajs from {len(imgs)} samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(common.DEB_DATA, "raw/huron"))
    ap.add_argument("--out", default=os.path.join(common.DEB_DATA, "gnm/huron"))
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bags = sorted(Path(a.inp).glob("*/*.bag"))
    if a.limit > 0:
        bags = bags[: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        for name, st in ex.map(process_bag, [(b, a.out, a.rate) for b in bags]):
            print(name, st, flush=True)


if __name__ == "__main__":
    main()
