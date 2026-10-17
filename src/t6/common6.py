"""Shared helpers for T6 (Orin-native dendritic networks). Adapted from src/t1/common.py."""
import json
import os
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init
import torch  # noqa: E402

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T6 = f"{ROOT}/results/T6"
DATA = os.environ.get("DEB_DATA", f"{ROOT}/data")  # datasets: FashionMNIST/, MNIST/, SHD/ (as T1)
MODELS = f"{T6}/models"
LOGS = f"{T6}/logs"
LUT = f"{T6}/lut"
OPSET = 13  # TensorRT 8.5 on Orin #1


def cap_gpu_mem():
    """Per-process cap from the queue's declared need (DN_MEM_MB); ~450 MB is CUDA context."""
    mb = int(os.environ.get("DN_MEM_MB", "0"))
    if mb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(max(0.02, (mb - 450) / total))


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vision(name, device="cuda"):
    """MNIST / FashionMNIST as flattened float32 in [0, 1] (N, 784) on `device` (as T1)."""
    import torchvision as tv
    cls = {"mnist": tv.datasets.MNIST, "fmnist": tv.datasets.FashionMNIST}[name]
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = cls(DATA, train=train, download=False)
        x = ds.data.reshape(len(ds.data), -1).float().div(255.0)
        out[split] = (x.to(device), ds.targets.long().to(device))
    return out


def mask_2to4(w, dim=-1):
    """Magnitude 2:4 mask: in every run of 4 consecutive elements along `dim` keep the 2 largest |w|.
    TensorRT's structured sparsity for Conv/FC groups the *input-channel* dimension (K)."""
    w = w.detach().transpose(dim, -1)
    shp = w.shape
    assert shp[-1] % 4 == 0, shp
    g = w.reshape(-1, 4).abs()
    idx = g.topk(2, dim=1).indices
    m = torch.zeros_like(g).scatter_(1, idx, 1.0)
    return m.reshape(shp).transpose(dim, -1).contiguous()


def is_2to4(w, dim=-1):
    w = w.detach().transpose(dim, -1)
    return bool(((w.reshape(-1, 4) != 0).sum(1) <= 2).all())


def export_onnx(model, args, path, input_names, output_names, check=True):
    model = model.eval().cpu()
    args = tuple(a.cpu() for a in args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(model, args, path, opset_version=OPSET, input_names=input_names,
                          output_names=output_names, do_constant_folding=True, dynamo=False)
    import onnx
    info = {"onnx": os.path.basename(path), "size_MB": round(os.path.getsize(path) / 2**20, 3),
            "nodes": len(onnx.load(path).graph.node), "export_s": round(time.time() - t0, 1)}
    if check:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(*args)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        got = sess.run(None, {n: a.numpy() for n, a in zip(input_names, args)})[0]
        info["max_abs_err_vs_torch"] = float(np.abs(got - ref.numpy()).max())
    return info


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
