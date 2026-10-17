"""T1 group 1: dANN-LRF vs MLP on Fashion-MNIST.

dANN-LRF = official Poirazi-Lab/dendritic_anns model_type 2 (dendritic layer with local
receptive fields, dendrite->soma structured connectivity), masks built by the official
`opt.make_masks`; the network is the same Dense->LeakyReLU(0.1)->Dense->LeakyReLU(0.1)->Dense
stack as `opt.get_model`, re-written in plain PyTorch (the official one is a Keras-3 model on
the torch backend). Training follows `main_pytorch.py` / `custom_train_loop_torch`: Adam 1e-3,
batch 128, 25 epochs (fmnist), 10 % validation split, gradients multiplied by the masks.

Mask implementation (what we deploy): dense weight matrices with zeros; the masks are folded
into the weights at export time, so the ONNX graph is a plain dense MLP of shape
784 -> d*s -> s -> 10.

Usage: python dann_fmnist.py --model {dann,mlp,vann_same} [--dends 32 --soma 128]
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, export_onnx, load_vision, save_meta, set_seed  # noqa: E402

OFFICIAL = os.environ.get("DENDRITIC_ANNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party/dendritic_anns/codes_"))


def official_make_masks():
    """Import opt.make_masks without installing keras (make_masks itself is numpy-only)."""
    if "keras" not in sys.modules:
        k = types.ModuleType("keras")
        k.utils = types.SimpleNamespace(Progbar=object)
        sys.modules["keras"] = k
        sys.modules["keras.utils"] = k.utils
    sys.path.insert(0, OFFICIAL)
    from opt import make_masks
    return make_masks


class DendMLP(nn.Module):
    """784 -> H1 (dendrites or hidden) -> H2 (somata) -> 10, LeakyReLU(0.1)."""

    def __init__(self, h1, h2, n_in=784, n_out=10, masks=None):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(n_in, h1), nn.Linear(h1, h2), nn.Linear(h2, n_out)
        for l in (self.l1, self.l2, self.out):  # keras Dense default: glorot_uniform, zero bias
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
        self.masks = None
        if masks is not None:
            # keras masks are (in, out); torch weights are (out, in)
            self.masks = [torch.tensor(masks[0].T, dtype=torch.float32),
                          torch.tensor(masks[2].T, dtype=torch.float32)]
            with torch.no_grad():
                self.l1.weight.mul_(self.masks[0])
                self.l2.weight.mul_(self.masks[1])
            self.l1.weight.register_hook(lambda g: g * self.masks[0].to(g.device))
            self.l2.weight.register_hook(lambda g: g * self.masks[1].to(g.device))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.1)
        x = F.leaky_relu(self.l2(x), 0.1)
        return self.out(x)

    def effective_params(self):
        if self.masks is None:
            return sum(p.numel() for p in self.parameters())
        n = int(self.masks[0].sum() + self.masks[1].sum())
        n += self.l1.bias.numel() + self.l2.bias.numel() + sum(p.numel() for p in self.out.parameters())
        return n


def mlp_hidden_for(params, n_in=784, n_out=10):
    """h such that 784*h + h + h*h + h + 10*h + 10 ~= params (two equal hidden layers)."""
    a, b, c = 1, n_in + n_out + 2, n_out - params
    return int(round((-b + (b * b - 4 * a * c) ** 0.5) / 2))


def evaluate(model, x, y, bs=2000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dann", "mlp", "vann_same"], required=True)
    ap.add_argument("--dends", type=int, default=32)
    ap.add_argument("--soma", type=int, default=128)
    ap.add_argument("--synapses", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)  # official value for fmnist
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--hidden", type=int, default=0, help="MLP: explicit hidden width (0 = parameter-matched)")
    ap.add_argument("--no_export", action="store_true")
    args = ap.parse_args()
    dev = os.environ.get("T1_DEV", "cuda")  # T1_DEV=cpu trains on the CPU

    set_seed(args.trial)
    d, s = args.dends, args.soma
    data = load_vision("fmnist", dev)
    xtr, ytr = data["train"]
    rng = np.random.default_rng(args.trial)  # official: shuffled 10 % validation split
    idx = torch.tensor(rng.permutation(len(xtr)), device=xtr.device)
    nval = int(0.1 * len(xtr))
    xva, yva = xtr[idx[-nval:]], ytr[idx[-nval:]]
    xtr, ytr = xtr[idx[:-nval]], ytr[idx[:-nval]]
    xte, yte = data["test"]

    make_masks = official_make_masks()
    dann_masks = make_masks([d], [s], args.synapses, 1, 28, 28, 10, 1, conventional=False,
                            rfs=True, rfs_type="dendritic", rfs_mode="random", seed=args.trial)
    dann_eff = DendMLP(d * s, s, masks=dann_masks).effective_params()
    if args.model == "dann":
        model = DendMLP(d * s, s, masks=dann_masks)
        name = f"dann_lrf_d{d}_s{s}"
    elif args.model == "vann_same":  # official vANN (model_type 3) with the same layer sizes
        model = DendMLP(d * s, s)
        name = f"vann_same_d{d}_s{s}"
    else:  # MLP with the same number of (effective) parameters as the dANN-LRF
        h = args.hidden or mlp_hidden_for(dann_eff)
        model = DendMLP(h, h)
        name = f"mlp_h{h}_match_d{d}_s{s}" if not args.hidden else f"mlp_h{h}_fmnist"
    name += args.tag
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator(device=dev).manual_seed(args.trial)
    hist = []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev, generator=g)
        for i in range(0, len(xtr), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        va = evaluate(model, xva, yva)
        hist.append(va)
        print(f"{name} ep {ep + 1} val {va:.4f}", flush=True)
    # official protocol: no early stopping, report the test accuracy of the final model
    te = evaluate(model, xte, yte)
    meta = {"name": name, "group": "dann_fmnist", "dataset": "Fashion-MNIST", "model": args.model,
            "dends": d, "soma": s, "synapses": args.synapses, "epochs": args.epochs,
            "trial": args.trial, "test_acc": te, "val_acc_last": hist[-1],
            "params_effective": model.effective_params(),
            "params_stored": sum(p.numel() for p in model.parameters()),
            "dann_params_effective": dann_eff,
            "shape": [784, model.l1.out_features, model.l2.out_features, 10]}
    print(json.dumps(meta))
    if model.masks is not None:  # sanity: masked weights are still exactly zero
        z = (model.l1.weight.detach().cpu()[model.masks[0] == 0].abs().max().item())
        meta["masked_weight_absmax"] = z
    if not args.no_export:
        meta["export"] = export_onnx(model, (xte[:1].cpu(),)  # check on a real test image, not zeros
                                     , f"{MODELS}/{name}.onnx",
                                     ["x"], ["logits"])
    save_meta(name, meta)


if __name__ == "__main__":
    main()
