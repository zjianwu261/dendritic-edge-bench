"""T1m step 1: parameter sweep of dANN-LRF and MLP on Fashion-MNIST (fp32 training).

Model (same as T1 / official `opt.get_model`): 784 -> H1 -> H2 -> 10, LeakyReLU(0.1).
  dANN-LRF: H1 = d*s dendrites (16 local-RF synapses each, official masks), H2 = s somata,
            dendrite->soma structured (d dendrites per soma), output dense.
  MLP:      H1 = H2 = h, dense.
Training follows official main_pytorch.py: Adam 1e-3, batch 128, 25 epochs, 10 % val split,
glorot-uniform init, zero bias, masked gradients; test accuracy of the final model.
Input = pixel/255 (the MCU receives raw uint8 pixels, so this scale is exact).

Step 5: --mask prng trains dANN models named dann_prng_d<d>_s<s> whose masks come from
the on-chip pseudo-random RF generator (prng_rf.py) instead of the official opt.make_masks.

Usage: python train.py [--seeds 1 2 3] [--only dann_d16_s64 ...] [--mask official|prng]
                       [--device cpu --threads 6]
Saves results/T1m/models/<name>_t<seed>.npz (+ .json).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MODELS, dann_masks, ensure, load_fmnist_u8
from prng_rf import prng_masks

DANN_CFGS = [(8, 32), (8, 64), (16, 64), (24, 64), (16, 128), (32, 128)]  # (dends/soma, soma)
MLP_CFGS = [16, 32, 48, 64, 72, 128]
# step 5: same sweep + points near the 63.5 KB line (no index tables -> ~19-21 B per dendrite)
PRNG_CFGS = DANN_CFGS + [(20, 128), (24, 128), (32, 96)]
SYN = 16


def cfg_list(mask="official"):
    if mask == "prng":
        return [(f"dann_prng_d{d}_s{s}", "dann", dict(d=d, s=s)) for d, s in PRNG_CFGS]
    out = [(f"dann_d{d}_s{s}", "dann", dict(d=d, s=s)) for d, s in DANN_CFGS]
    out += [(f"mlp_h{h}", "mlp", dict(h=h)) for h in MLP_CFGS]
    return out


class Net(nn.Module):
    def __init__(self, h1, h2, m1=None, m2=None):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(784, h1), nn.Linear(h1, h2), nn.Linear(h2, 10)
        for l in (self.l1, self.l2, self.out):
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
        self.masks = None
        if m1 is not None:
            self.masks = [torch.tensor(m1.T, dtype=torch.float32), torch.tensor(m2.T, dtype=torch.float32)]

    def to(self, dev):
        super().to(dev)
        if self.masks is not None:
            self.masks = [m.to(dev) for m in self.masks]
            with torch.no_grad():
                self.l1.weight.mul_(self.masks[0])
                self.l2.weight.mul_(self.masks[1])
            self.l1.weight.register_hook(lambda g: g * self.masks[0])
            self.l2.weight.register_hook(lambda g: g * self.masks[1])
        return self

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.1)
        x = F.leaky_relu(self.l2(x), 0.1)
        return self.out(x)


def evaluate(model, x, y, bs=5000):
    model.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            c += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return c / len(x)


def train_one(name, kind, cfg, seed, data, dev, epochs, mask="official"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr_u8, ytr = data["train"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(xtr_u8))
    nval = int(0.1 * len(xtr_u8))
    tr_idx, va_idx = perm[:-nval], perm[-nval:]
    X = torch.tensor(xtr_u8, dtype=torch.float32, device=dev) / 255.0
    Y = torch.tensor(ytr, device=dev)
    xte = torch.tensor(data["test"][0], dtype=torch.float32, device=dev) / 255.0
    yte = torch.tensor(data["test"][1], device=dev)
    xtr, ytr_t = X[tr_idx], Y[tr_idx]
    xva, yva = X[va_idx], Y[va_idx]
    if kind == "dann":
        if mask == "prng":
            m1, m2 = prng_masks(cfg["d"], cfg["s"], seed)
        else:
            m1, m2 = dann_masks(cfg["d"], cfg["s"], SYN, seed)
        model = Net(cfg["d"] * cfg["s"], cfg["s"], m1, m2).to(dev)
        params = int(m1.sum() + m2.sum()) + cfg["d"] * cfg["s"] + cfg["s"] + 10 * cfg["s"] + 10
    else:
        m1 = m2 = None
        model = Net(cfg["h"], cfg["h"]).to(dev)
        params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        p = torch.randperm(len(xtr), device=dev, generator=g)
        for i in range(0, len(xtr), 128):
            b = p[i:i + 128]
            loss = F.cross_entropy(model(xtr[b]), ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    va = evaluate(model, xva, yva)
    te = evaluate(model, xte, yte)
    if model.masks is not None:
        assert model.l1.weight.detach()[model.masks[0] == 0].abs().max().item() == 0.0
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    out = dict(W1=sd["l1.weight"], b1=sd["l1.bias"], W2=sd["l2.weight"], b2=sd["l2.bias"],
               W3=sd["out.weight"], b3=sd["out.bias"], val_idx=va_idx.astype(np.int32))
    if m1 is not None:
        out.update(M1=m1, M2=m2)
    tag = f"{name}_t{seed}"
    np.savez_compressed(f"{MODELS}/{tag}.npz", **out)
    meta = dict(name=name, kind=kind, cfg=cfg, seed=seed, epochs=epochs, params=params,
                mask_source=mask if kind == "dann" else "",
                val_acc=va, acc_fp32=te, train_s=round(time.time() - t0, 1),
                device=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
                torch=torch.__version__)
    with open(f"{MODELS}/{tag}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask", default="official", choices=["official", "prng"])
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    ensure(MODELS)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)  # 5080 is shared with the T1 run
    data = load_fmnist_u8()
    for seed in a.seeds:
        for name, kind, cfg in cfg_list(a.mask):
            if a.only and name not in a.only:
                continue
            train_one(name, kind, cfg, seed, data, a.device, a.epochs, a.mask)


if __name__ == "__main__":
    main()
