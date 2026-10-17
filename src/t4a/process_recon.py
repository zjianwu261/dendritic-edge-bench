"""Process RECON straight from recon_dataset.tar.gz (streamed, nothing extracted to disk besides output).

Same as visualnav-transformer train/process_recon.py (one trajectory per hdf5 file, images/rgb_left as
<i>.jpg, traj_data.pkl with jackal/position[:, :2] and jackal/yaw; no backwards filtering), plus extra
per-frame files for T4a: lidar.npy [T,360] float16, illum.npy [T] (android/illuminance, lux),
thermal.npy [T,32,32] float16, collision.npy [T] bool (collision/any).
Usage: python process_recon.py [--tar $DEB_DATA/raw/recon/recon_dataset.tar.gz --out $DEB_DATA/gnm/recon]
"""
import argparse
import io
import os
import pickle
import tarfile

import h5py
import numpy as np

DEB_DATA = os.environ.get("DEB_DATA", os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=os.path.join(DEB_DATA, "raw/recon/recon_dataset.tar.gz"))
    ap.add_argument("--out", default=os.path.join(DEB_DATA, "gnm/recon"))
    ap.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with tarfile.open(a.tar, mode="r|gz") as tf:
        for m in tf:
            if not m.name.endswith(".hdf5"):
                continue
            name = os.path.basename(m.name)[:-5]
            folder = os.path.join(a.out, name)
            if os.path.exists(os.path.join(folder, "traj_data.pkl")):
                n += 1; continue
            blob = tf.extractfile(m).read()
            try:
                with h5py.File(io.BytesIO(blob), "r") as h:
                    td = {"position": h["jackal"]["position"][:, :2], "yaw": h["jackal"]["yaw"][()]}
                    imgs = h["images"]["rgb_left"][()]
                    extra = {"lidar": h["lidar"][()].astype(np.float16), "illum": h["android"]["illuminance"][()],
                             "thermal": h["images"]["thermal"][()].astype(np.float16),
                             "collision": h["collision"]["any"][()]}
            except Exception as e:  # noqa: BLE001
                print("skip", name, repr(e), flush=True); continue
            os.makedirs(folder, exist_ok=True)
            for i, b in enumerate(imgs):
                with open(os.path.join(folder, f"{i}.jpg"), "wb") as f:
                    f.write(bytes(b))  # stored JPEG bytes; official code re-encodes via PIL, we keep the original
            for k, v in extra.items():
                np.save(os.path.join(folder, f"{k}.npy"), v)
            with open(os.path.join(folder, "traj_data.pkl"), "wb") as f:
                pickle.dump(td, f)
            n += 1
            if n % 500 == 0:
                print(n, "trajectories", flush=True)
            if 0 < a.limit <= n:
                break
    print("DONE", n, flush=True)


if __name__ == "__main__":
    main()
