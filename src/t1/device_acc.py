"""On-device accuracy: TRT FP16 on Orin #1 vs onnxruntime FP32 (workstation CPU) on the same inputs.

Usage: python device_acc.py {fmnist,pmnist,shd} model1 [model2 ...]   (model = ONNX basename)
Needs the engines built by src/t1/run_bench.sh. Engines are looked up in
~/dendritic-edge-bench/T1/$ORIN_WD (default trt_v07) as <model>_fp16_<build>_<onnx sha256[:12]>.engine
for build in $BUILDS (default "default nocublas"), so the engine always matches the current ONNX;
the evaluation runs under flock /tmp/dn_orin_bench.lock on the Orin (shared with other benchmark jobs).
Writes results/T1/raw/$ORIN_RAW/device_acc.jsonl (default orin1_v07).
Subsets: fmnist = full test set (10k); pmnist = first 1000 test images of each of the 10
tasks (context = nearest prototype, as in training); shd = full test set (2264).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, MODELS, T1, load_vision  # noqa: E402

J = os.environ.get("ORIN_HOST", "orin")
RD = "dendritic-edge-bench/T1"
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
WD = os.environ.get("ORIN_WD", "trt_v07")
RAWD = os.environ.get("ORIN_RAW", "orin1_v07")
BUILDS = os.environ.get("BUILDS", "default nocublas").split()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_npz(group):
    path = f"{T1}/data/eval_{group}.npz"
    if group == "fmnist":
        x, y = load_vision("fmnist", "cpu")["test"]
        arrs = {"x": x.numpy()[:, None, :]}
    elif group == "pmnist":
        from active_dendrites import MNIST_MEAN, MNIST_STD, N_TASKS
        d = load_vision("mnist", "cpu")
        xtr, (xte, yte) = d["train"][0], d["test"]
        xtr, xte = (xtr - MNIST_MEAN) / MNIST_STD, (xte - MNIST_MEAN) / MNIST_STD
        g = torch.Generator().manual_seed(1234)
        perms = [torch.randperm(784, generator=g) for _ in range(N_TASKS)]
        protos = torch.stack([xtr[:, p].mean(0) for p in perms])
        xs = torch.cat([xte[:1000, p] for p in perms])
        ctx = protos[torch.cdist(xs, protos).argmin(1)]
        y = yte[:1000].repeat(N_TASKS)
        arrs = {"x": xs.numpy()[:, None, :], "context": ctx.numpy()[:, None, :]}
    elif group == "shd":
        from shd import load_shd, make_batch
        flat, offs, y = load_shd("test", "cpu")
        packed = []
        for i in range(len(y)):
            xb = make_batch(flat, offs, torch.tensor([i])).numpy().astype(np.uint8).reshape(-1)
            packed.append(np.packbits(xb))
        arrs = {"x__packbits": np.stack(packed)}
    if not os.path.exists(path):
        np.savez(path, **arrs)
    return path, np.asarray(y)


def ort_outputs(model, npz):
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{MODELS}/{model}.onnx", providers=["CPUExecutionProvider"])
    data = np.load(npz)
    outs = []
    names = [i.name for i in sess.get_inputs()]
    for k in range(len(data[data.files[0]])):
        feed = {}
        for n, i in zip(names, sess.get_inputs()):
            if n in data.files:
                feed[n] = data[n][k].astype(np.float32)
            else:
                feed[n] = np.unpackbits(data[n + "__packbits"][k])[:int(np.prod(i.shape))] \
                    .reshape(i.shape).astype(np.float32)
        outs.append(sess.run(None, feed)[0].reshape(-1))
    return np.stack(outs)


def main():
    group, models = sys.argv[1], sys.argv[2:]
    npz, y = build_npz(group)
    subprocess.run(["ssh", J, f"mkdir -p {RD}/eval"], check=True)
    subprocess.run(["scp", "-q", f"{BENCH}/trt_eval.py",
                    f"{BENCH}/trt_sync_bench.py", f"{J}:{RD}/bin/"], check=True)
    subprocess.run(["rsync", "-a", npz, f"{J}:{RD}/eval/"], check=True)
    os.makedirs(f"{T1}/raw/{RAWD}/eval", exist_ok=True)
    npz_sha = sha256(npz)
    for m in models:
        sha = sha256(f"{MODELS}/{m}.onnx")[:12]
        ref = ort_outputs(m, npz)
        for b in BUILDS:
            t0 = time.time()
            eng = f"{RD}/{WD}/{m}_fp16_{b}_{sha}.engine"
            rn = f"{RD}/eval/{m}_fp16_{b}_out.npy"
            subprocess.run(["ssh", J, f"flock /tmp/dn_orin_bench.lock python3 {RD}/bin/trt_eval.py {eng} "
                            f"{RD}/eval/{os.path.basename(npz)} {rn}"], check=True)
            subprocess.run(["scp", "-q", f"{J}:{rn}", f"{T1}/raw/{RAWD}/eval/"], check=True)
            trt_out = np.load(f"{T1}/raw/{RAWD}/eval/{m}_fp16_{b}_out.npy")
            rec = {"model": m, "build": b, "engine": os.path.basename(eng), "group": group, "n": int(len(y)),
                   "eval_npz_sha256": npz_sha,
                   "acc_trt_fp16_orin": float((trt_out.argmax(1) == y).mean()),
                   "acc_ort_fp32_workstation": float((ref.argmax(1) == y).mean()),
                   "argmax_agreement": float((trt_out.argmax(1) == ref.argmax(1)).mean()),
                   "max_abs_out_diff": float(np.abs(trt_out - ref).max()),
                   "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            print(json.dumps(rec), flush=True)
            with open(f"{T1}/raw/{RAWD}/device_acc.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
