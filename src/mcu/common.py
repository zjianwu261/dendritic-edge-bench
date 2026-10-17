"""T1m shared helpers: paths, Fashion-MNIST loading (raw uint8), dANN-LRF masks.

dANN-LRF masks come from the official Poirazi-Lab/dendritic_anns code (`opt.make_masks`,
model_type 2 = dendritic layer with local receptive fields, rfs_type="dendritic",
rfs_mode="random", 16 synapses / dendrite, structured dendrite->soma connectivity).
"""
import gzip
import os
import sys
import types

import numpy as np

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = f"{ROOT}/results/T1m"
MODELS = f"{EXP}/models"
DATA = os.environ.get("DEB_DATA", f"{ROOT}/data")  # same dataset folder as T1
FMNIST_RAW = f"{DATA}/FashionMNIST/raw"
OFFICIAL = os.environ.get("DENDRITIC_ANNS", f"{ROOT}/src/third_party/dendritic_anns/codes_")


def _read_idx(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_fmnist_u8():
    """Returns dict split -> (x uint8 (N,784), y int64 (N,))."""
    out = {}
    for split, pre in (("train", "train"), ("test", "t10k")):
        x = _read_idx(f"{FMNIST_RAW}/{pre}-images-idx3-ubyte").reshape(-1, 784)
        y = _read_idx(f"{FMNIST_RAW}/{pre}-labels-idx1-ubyte").astype(np.int64)
        out[split] = (x.copy(), y.copy())
    return out


def official_make_masks():
    """Import opt.make_masks without installing keras (make_masks itself is numpy-only)."""
    if "keras" not in sys.modules:
        k = types.ModuleType("keras")
        k.utils = types.SimpleNamespace(Progbar=object)
        sys.modules["keras"] = k
        sys.modules["keras.utils"] = k.utils
    if OFFICIAL not in sys.path:
        sys.path.insert(0, OFFICIAL)
    from opt import make_masks
    return make_masks


def dann_masks(d, s, synapses, seed):
    """Official masks, returned in (in, out) keras layout: [M1 (784, d*s), M2 (d*s, s)]."""
    mm = official_make_masks()
    m = mm([d], [s], synapses, 1, 28, 28, 10, 1, conventional=False, rfs=True,
           rfs_type="dendritic", rfs_mode="random", seed=seed)
    m1, m2 = m[0].astype(np.int8), m[2].astype(np.int8)
    assert m1.shape == (784, d * s) and (m1.sum(0) == synapses).all()
    # structured dendrite -> soma: soma j owns dendrites j*d .. j*d+d-1 (no index needed)
    ref = np.zeros((d * s, s), np.int8)
    for j in range(s):
        ref[j * d:(j + 1) * d, j] = 1
    assert (m2 == ref).all()
    return m1, m2


def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p
