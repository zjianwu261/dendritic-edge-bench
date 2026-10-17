"""Shared helpers for T1 (train on 5080, export ONNX for TensorRT 8.5 / onnxruntime)."""
import json
import os
import time

import numpy as np
import torch

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T1 = f"{ROOT}/results/T1"
DATA = os.environ.get("DEB_DATA", f"{ROOT}/data")  # datasets: FashionMNIST/, MNIST/, SHD/
MODELS = f"{T1}/models"
OPSET = 13  # TensorRT 8.5 (Orin #1) parses opset 13 fine (see T0)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vision(name, device="cuda"):
    """MNIST / FashionMNIST as flattened float32 in [0, 1] (N, 784) on `device`."""
    import torchvision as tv
    cls = {"mnist": tv.datasets.MNIST, "fmnist": tv.datasets.FashionMNIST}[name]
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = cls(DATA, train=train, download=False)
        x = ds.data.reshape(len(ds.data), -1).float().div(255.0)
        out[split] = (x.to(device), ds.targets.long().to(device))
    return out


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def export_onnx(model, args, path, input_names, output_names):
    model = model.eval().cpu()
    args = tuple(a.cpu() for a in args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(model, args, path, opset_version=OPSET, input_names=input_names,
                          output_names=output_names, do_constant_folding=True, dynamo=False)
    # numerical check against onnxruntime (CPU)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(*args)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    got = sess.run(None, {n: a.numpy() for n, a in zip(input_names, args)})[0]
    err = float(np.abs(got - ref.numpy()).max())
    import onnx
    n_nodes = len(onnx.load(path).graph.node)
    info = {"onnx": path, "size_MB": round(os.path.getsize(path) / 2**20, 3),
            "nodes": n_nodes, "max_abs_err_vs_torch": err, "export_s": round(time.time() - t0, 1)}
    print(json.dumps(info))
    return info


def save_meta(name, meta):
    os.makedirs(MODELS, exist_ok=True)
    with open(f"{MODELS}/{name}.json", "w") as f:
        json.dump(meta, f, indent=1)
