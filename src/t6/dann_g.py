"""T6 module 1 (G1, Fashion-MNIST): dANN-G and dANN-2:4, plus dense MLP baselines.

dANN-G  (no gather, Tensor-Core friendly):
  * input 28x28 is cut into G = (28/p)^2 local p x p blocks (p = 4 -> 49 blocks of 16 pixels);
  * dendritic layer = grouped GEMM / block-diagonal matrix: every block feeds m dendrites
    (group width in = p*p = 16, out = m, m in {8,16,32,64}); D = G*m dendrites, each dendrite
    sees one local receptive field of p*p pixels (the "local block" RF);
  * soma layer = grouped GEMM over *channel groups*: the dendrite tensor (G, m) is split along
    the channel axis into sg groups; soma group q (s/sg somata) integrates channels
    q*m/sg..(q+1)*m/sg-1 from ALL G locations (a channel shuffle, no gather). sg = 1 -> dense.
  * LeakyReLU(0.1) after both layers, dense s -> 10 readout (as the official dANN / T1).
  Optional sp24: dendritic synapses restricted to 2:4 inside each 16-pixel block (8 synapses
  per dendrite), soma weights 2:4 too.
dANN-2:4: dendritic layer 784 -> D with 2:4 synapse sparsity (dense storage shape, TensorRT
  sparse tactic), soma layer D -> s with 2:4 connectivity, dense readout. Masks = magnitude
  2:4 after `dense_epochs` dense epochs, then fixed (ASP-style), gradients / weights re-masked.
MLP: 784-h-h-10 LeakyReLU(0.1) (as T1 G1 baseline).
Training recipe = T1 / official dANN: Adam 1e-3, batch 128, 25 epochs, 10 % val split
(rng seed = trial), glorot init, report the final-epoch model (test) and its val accuracy.

Usage: python dann_g.py --arch {dannG,dann24,mlp} [--p 4 --m 32 --s 128 --sg 4 --sp24]
                        [--D 2048] [--h 256] [--seed 1] [--no_export]
       python dann_g.py --check      # equivalence vs the T1 mask implementation
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LOGS, MODELS, cap_gpu_mem, export_onnx, is_2to4, load_vision, mask_2to4, save_json, set_seed  # noqa: E402

LEAK = 0.1


def patch_perm(p):
    """perm[j] = raster pixel index of the j-th input of the grouped layer (patch-major order)."""
    n = 28 // p
    idx = torch.arange(784).view(n, p, n, p).permute(0, 2, 1, 3).reshape(-1)
    return idx


class GroupedLinear(nn.Module):
    """y[:, g*out:(g+1)*out] = x[:, g*in:(g+1)*in] @ W[g].T + b[g]  (block-diagonal weight)."""

    def __init__(self, groups, fin, fout):
        super().__init__()
        self.g, self.fin, self.fout = groups, fin, fout
        self.weight = nn.Parameter(torch.empty(groups, fout, fin))
        self.bias = nn.Parameter(torch.zeros(groups * fout))
        bound = math.sqrt(6.0 / (fin + fout))  # glorot uniform per group
        nn.init.uniform_(self.weight, -bound, bound)
        self.register_buffer("mask", torch.ones(groups, fout, fin))

    def forward(self, x, impl="einsum"):
        b = x.shape[0]
        if impl == "einsum":
            y = torch.einsum("bgi,goi->bgo", x.reshape(b, self.g, self.fin), self.weight)
            return y.reshape(b, -1) + self.bias
        if impl == "conv":  # x: (1, g*in, 1, 1)
            w = self.weight.reshape(self.g * self.fout, self.fin, 1, 1)
            return F.conv2d(x, w, self.bias, groups=self.g)
        if impl == "bmm":   # x: (g, 1, in)
            return torch.matmul(x, self.weight.transpose(1, 2)) + self.bias.view(self.g, 1, self.fout)
        raise ValueError(impl)

    def dense(self):
        """Equivalent dense (g*out, g*in) block-diagonal matrix."""
        return torch.block_diag(*[self.weight[i] for i in range(self.g)])


class DANNG(nn.Module):
    def __init__(self, p=4, m=32, s=128, sg=4, sp24=False):
        super().__init__()
        assert 28 % p == 0 and m % sg == 0 and s % sg == 0
        self.p, self.m, self.s, self.sg, self.sp24 = p, m, s, sg, sp24
        self.G = (28 // p) ** 2
        self.register_buffer("perm", patch_perm(p))
        self.l1 = GroupedLinear(self.G, p * p, m)
        self.l2 = GroupedLinear(sg, self.G * m // sg, s // sg)
        self.out = nn.Linear(s, 10)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.impl = "einsum"

    def layers24(self):
        return [self.l1, self.l2] if self.sp24 else []

    def shuffle(self, h):  # (b, G*m) -> (b, sg * (G*m/sg)) channel groups across all locations
        b = h.shape[0]
        return h.reshape(b, self.G, self.sg, self.m // self.sg).transpose(1, 2).reshape(b, -1)

    def forward(self, x):
        if self.impl == "einsum":
            h = F.leaky_relu(self.l1(x[:, self.perm]), LEAK)
            h = F.leaky_relu(self.l2(self.shuffle(h)), LEAK)
            return self.out(h)
        n, p = 28 // self.p, self.p
        xb = x.reshape(1, n, p, n, p).permute(0, 1, 3, 2, 4)            # (1, n, n, p, p) patch-major
        if self.impl == "conv":
            h = F.leaky_relu(self.l1(xb.reshape(1, 784, 1, 1), "conv"), LEAK)          # (1, G*m, 1, 1)
            h = h.reshape(1, self.G, self.sg, self.m // self.sg).transpose(1, 2).reshape(1, -1, 1, 1)
            h = F.leaky_relu(self.l2(h, "conv"), LEAK)
            return self.out(h.reshape(1, self.s))
        if self.impl == "bmm":
            h = F.leaky_relu(self.l1(xb.reshape(self.G, 1, p * p), "bmm"), LEAK)       # (G, 1, m)
            h = h.reshape(self.G, self.sg, self.m // self.sg).transpose(0, 1).reshape(self.sg, 1, -1)
            h = F.leaky_relu(self.l2(h, "bmm"), LEAK)                                   # (sg, 1, s/sg)
            return self.out(h.reshape(1, self.s))
        raise ValueError(self.impl)

    def dense_equivalent(self):
        """Dense (D x 784) and (s x D) matrices in the T1 layout (raster pixels, dendrite index
        g*m + c) that compute exactly the same function (for the T1 mask-implementation check)."""
        W1 = torch.zeros(self.G * self.m, 784)
        W1[:, self.perm] = self.l1.dense()
        mg = self.m // self.sg
        W2s = self.l2.dense()                                    # (s, D') in shuffled order
        order = torch.arange(self.G * self.m).view(self.G, self.sg, mg).transpose(0, 1).reshape(-1)
        W2 = torch.zeros(self.s, self.G * self.m)
        W2[:, order] = W2s                                       # shuffled position j <- dendrite order[j]
        return W1, self.l1.bias.detach().clone(), W2, self.l2.bias.detach().clone()

    def effective_params(self):
        n = int(self.l1.mask.sum() + self.l2.mask.sum()) + self.l1.bias.numel() + self.l2.bias.numel()
        return n + sum(p.numel() for p in self.out.parameters())


class Dense24MLP(nn.Module):
    """784 -> H1 -> H2 -> 10, LeakyReLU(0.1); optional 2:4 masks on the two hidden layers.
    arch 'mlp': H1 = H2 = h, no masks.  arch 'dann24': H1 = D dendrites, H2 = s somata, 2:4."""

    def __init__(self, h1, h2, sp24=False):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(784, h1), nn.Linear(h1, h2), nn.Linear(h2, 10)
        for l in (self.l1, self.l2, self.out):
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
        self.sp24 = sp24
        self.register_buffer("mask1", torch.ones_like(self.l1.weight))
        self.register_buffer("mask2", torch.ones_like(self.l2.weight))
        self.impl = "gemm"

    def layers24(self):
        return [(self.l1, "mask1"), (self.l2, "mask2")] if self.sp24 else []

    def forward(self, x):
        if self.impl == "conv":  # 1x1 convs on (1, C, 1, 1): TensorRT sparse tactics exist for Conv
            h = x.reshape(1, 784, 1, 1)
            h = F.leaky_relu(F.conv2d(h, self.l1.weight[:, :, None, None], self.l1.bias), LEAK)
            h = F.leaky_relu(F.conv2d(h, self.l2.weight[:, :, None, None], self.l2.bias), LEAK)
            return self.out(h.reshape(1, -1))
        x = F.leaky_relu(self.l1(x), LEAK)
        x = F.leaky_relu(self.l2(x), LEAK)
        return self.out(x)

    def effective_params(self):
        return int(self.mask1.sum() + self.mask2.sum()) + self.l1.bias.numel() + self.l2.bias.numel() \
            + sum(p.numel() for p in self.out.parameters())


class MLPReg(Dense24MLP):
    """MLP 784-h-h-10 with dropout after each hidden layer (weight decay via AdamW)."""

    def __init__(self, h, p):
        super().__init__(h, h)
        self.p = p

    def forward(self, x):
        x = F.dropout(F.leaky_relu(self.l1(x), LEAK), self.p, self.training)
        x = F.dropout(F.leaky_relu(self.l2(x), LEAK), self.p, self.training)
        return self.out(x)


class SmallCNN(nn.Module):
    """Small CNN baseline: conv3x3 s2 (1->c1) - ReLU - conv3x3 s2 (c1->c2) - ReLU - fc(c2*7*7 -> 10)."""

    def __init__(self, c1=16, c2=32):
        super().__init__()
        self.c1 = nn.Conv2d(1, c1, 3, 2, 1)
        self.c2 = nn.Conv2d(c1, c2, 3, 2, 1)
        self.fc = nn.Linear(c2 * 49, 10)
        self.impl = "conv"

    def forward(self, x):
        h = F.relu(self.c1(x.reshape(-1, 1, 28, 28)))
        h = F.relu(self.c2(h))
        return self.fc(h.flatten(1))

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


def build(arch, a):
    if arch == "mlpreg":
        return MLPReg(a.h, a.dropout), f"mlpreg_h{a.h}_do{a.dropout:g}_wd{a.wd:g}"
    if arch == "cnn":
        return SmallCNN(a.c1, a.c2), f"cnn_c{a.c1}_{a.c2}"
    if arch == "mlpsp":  # unstructured sparse MLP, random mask or magnitude pruning
        return Dense24MLP(a.h, a.h), f"mlpsp_h{a.h}_{a.prune}_s{a.sparsity:g}"
    if arch == "dannG":
        return DANNG(a.p, a.m, a.s, a.sg, a.sp24), f"dannG_p{a.p}_m{a.m}_s{a.s}_sg{a.sg}" + ("_sp24" if a.sp24 else "")
    if arch == "dann24":
        return Dense24MLP(a.D, a.s, sp24=True), f"dann24_D{a.D}_s{a.s}"
    return Dense24MLP(a.h, a.h), f"mlp_h{a.h}"


def apply_masks(model):
    with torch.no_grad():
        if isinstance(model, DANNG):
            for l in (model.l1, model.l2):
                l.weight.mul_(l.mask)
        else:
            model.l1.weight.mul_(model.mask1)
            model.l2.weight.mul_(model.mask2)


def set_sparse_masks(model, sparsity, how, gen=None):
    """Per-layer unstructured masks on l1 / l2 keeping round((1 - sparsity) * n) weights:
    how = random (uniform random positions) or magnitude (largest |w|)."""
    with torch.no_grad():
        for lin, mname in ((model.l1, "mask1"), (model.l2, "mask2")):
            n = lin.weight.numel()
            k = int(round((1 - sparsity) * n))
            if how == "random":
                score = torch.rand(lin.weight.shape, generator=gen, device=lin.weight.device) if gen is not None else \
                    torch.rand_like(lin.weight)
            else:
                score = lin.weight.abs()
            idx = score.flatten().topk(k).indices
            m = torch.zeros(n, device=lin.weight.device)
            m[idx] = 1.0
            getattr(model, mname).copy_(m.view_as(lin.weight))
    apply_masks(model)


def set_24_masks(model):
    if isinstance(model, DANNG):
        for l in model.layers24():
            l.mask.copy_(mask_2to4(l.weight, dim=-1))           # along the group input (K) axis
    else:
        for lin, mname in model.layers24():
            getattr(model, mname).copy_(mask_2to4(lin.weight, dim=1))
    apply_masks(model)


def check_24(model):
    if isinstance(model, DANNG):
        return all(is_2to4(l.weight, -1) for l in model.layers24())
    return all(is_2to4(lin.weight, 1) for lin, _ in model.layers24())


def evaluate(model, x, y, bs=2000):
    model.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            c += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return c / len(x)


def export_variants(model, name, x0, impls, outdir):
    infos = {}
    for impl in impls:
        model.impl = impl
        infos[impl] = export_onnx(model, (x0,), f"{outdir}/{name}_{impl}.onnx", ["x"], ["logits"])
    model.impl = "einsum" if isinstance(model, DANNG) else "gemm"
    return infos


def default_impls(model):
    if isinstance(model, SmallCNN):
        return ["conv"]
    return ["conv", "bmm"] if isinstance(model, DANNG) else ["gemm", "conv"]


def train(a):
    cap_gpu_mem()
    dev = os.environ.get("T6_DEV", "cuda")
    set_seed(a.seed)
    model, cfg = build(a.arch, a)
    if a.fair:  # fair protocol (T6-F): equal search space for dANN-G and MLP (lr x2, width x4, wd x2), AdamW
        cfg += f"_lr{a.lr:g}_wd{a.wd:g}_fair"
    name = f"{cfg}_t{a.seed}"
    if os.path.exists(f"{MODELS}/g1/{name}.json") and not a.force:
        print(f"{name}: done already, skipping")
        return
    data = load_vision("fmnist", dev)
    xtr, ytr = data["train"]
    rng = np.random.default_rng(a.seed)          # 10 % validation split as T1 / official
    idx = torch.tensor(rng.permutation(len(xtr)), device=dev)
    nval = int(0.1 * len(xtr))
    xva, yva = xtr[idx[-nval:]], ytr[idx[-nval:]]
    xtr, ytr = xtr[idx[:-nval]], ytr[idx[:-nval]]
    xte, yte = data["test"]
    model.to(dev)
    sparse = (a.arch == "dann24") or (a.arch == "dannG" and a.sp24)
    unstr = a.arch == "mlpsp"  # random mask from step 0, or magnitude pruning after dense_epochs
    if unstr and a.prune == "random":
        set_sparse_masks(model, a.sparsity, "random", torch.Generator(device=dev).manual_seed(10_000 + a.seed))
    opt = (torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd) if a.fair else
           torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=a.wd) if a.arch == "mlpreg"
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    g = torch.Generator(device=dev).manual_seed(a.seed)
    os.makedirs(LOGS, exist_ok=True)
    log = open(f"{LOGS}/g1_{name}.log", "w")
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        if sparse and ep == a.dense_epochs:
            set_24_masks(model)
        if unstr and a.prune == "magnitude" and ep == a.dense_epochs:
            set_sparse_masks(model, a.sparsity, "magnitude")
        model.train()
        perm = torch.randperm(len(xtr), device=dev, generator=g)
        for i in range(0, len(xtr), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (sparse and ep >= a.dense_epochs) or (unstr and (a.prune == "random" or ep >= a.dense_epochs)):
                apply_masks(model)
        va = evaluate(model, xva, yva)
        hist.append(va)
        s = f"{name} ep {ep + 1} val {va:.4f} ({time.time() - t0:.0f}s)"
        print(s, flush=True)
        log.write(s + "\n")
    te = evaluate(model, xte, yte)
    meta = {"name": name, "cfg": cfg, "task": "G1", "dataset": "Fashion-MNIST", "arch": a.arch, "args": vars(a),
            "test_acc": te, "val_acc_last": hist[-1], "val_hist": hist,
            "params_effective": model.effective_params(),
            "params_stored": sum(p.numel() for n, p in model.named_parameters()),
            "is_2to4": check_24(model) if sparse else None,
            "density_l1_l2": [float((model.l1.weight != 0).float().mean()), float((model.l2.weight != 0).float().mean())] if unstr else None, "train_minutes": round((time.time() - t0) / 60, 2)}
    os.makedirs(f"{MODELS}/g1", exist_ok=True)
    torch.save(model.state_dict(), f"{MODELS}/g1/{name}.pt")
    if not a.no_export:
        meta["export"] = export_variants(model, name, xte[:1].cpu(), default_impls(model), f"{MODELS}/g1")
    save_json(f"{MODELS}/g1/{name}.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("val_hist",)}), flush=True)
    log.write(json.dumps(meta) + "\n")


def check():
    """dANN-G / dANN-2:4 vs the T1 mask implementation (src/t1/dann_fmnist.DendMLP, read-only)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "t1"))
    from dann_fmnist import DendMLP  # T1 mask implementation (dense matrices with zeros)
    torch.manual_seed(0)
    x = torch.rand(256, 784)
    res = {}
    for (p, m, s, sg, sp) in [(4, 32, 128, 4, False), (4, 16, 64, 1, False), (4, 64, 256, 8, True), (7, 16, 64, 2, False)]:
        net = DANNG(p, m, s, sg, sp).eval()
        with torch.no_grad():
            for l in (net.l1, net.l2):
                l.bias.uniform_(-.1, .1)
        if sp:
            set_24_masks(net)
        W1, b1, W2, b2 = net.dense_equivalent()
        t1 = DendMLP(W1.shape[0], s).eval()     # masks=None; we load the dense-with-zeros weights
        with torch.no_grad():
            t1.l1.weight.copy_(W1); t1.l1.bias.copy_(b1)
            t1.l2.weight.copy_(W2); t1.l2.bias.copy_(b2)
            t1.out.load_state_dict(net.out.state_dict())
            ref = t1(x)
            out = {"einsum": float((net(x) - ref).abs().max())}
            for impl in ("conv", "bmm"):
                net.impl = impl
                out[impl] = float(max((net(x[i:i + 1]) - ref[i:i + 1]).abs().max() for i in range(16)))
            net.impl = "einsum"
        nz1 = (W1 != 0).sum(1).float()
        rows = (W1 != 0).nonzero()
        # receptive-field extent of each dendrite (pixels): bounding box of its synapses
        ext = []
        for d in range(0, W1.shape[0], max(1, W1.shape[0] // 64)):
            pix = rows[rows[:, 0] == d][:, 1]
            r, c = pix // 28, pix % 28
            ext.append(int(max(r.max() - r.min(), c.max() - c.min()) + 1))
        res[f"dannG_p{p}_m{m}_s{s}_sg{sg}" + ("_sp24" if sp else "")] = {
            "max_abs_diff_vs_T1_DendMLP": out, "synapses_per_dendrite": [float(nz1.min()), float(nz1.max())],
            "dendrites_per_soma": float((W2 != 0).sum(1).float().mean()),
            "rf_extent_px_max": max(ext), "dendrites": W1.shape[0]}
    # dANN-2:4 in both export forms vs the T1 class
    net = Dense24MLP(512, 64, sp24=True).eval()
    set_24_masks(net)
    t1 = DendMLP(512, 64).eval()
    t1.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith("mask")})
    with torch.no_grad():
        ref = t1(x)
        net.impl = "conv"
        dconv = float(max((net(x[i:i + 1]) - ref[i:i + 1]).abs().max() for i in range(16)))
        net.impl = "gemm"
        res["dann24_D512_s64"] = {"max_abs_diff_vs_T1_DendMLP": {"gemm": float((net(x) - ref).abs().max()),
                                                                 "conv": dconv}, "is_2to4": check_24(net)}
    # official dANN-LRF masks for comparison of the receptive fields
    from dann_fmnist import official_make_masks
    mk = official_make_masks()([32], [128], 16, 1, 28, 28, 10, 1, conventional=False, rfs=True,
                               rfs_type="dendritic", rfs_mode="random", seed=1)
    M1 = torch.tensor(mk[0].T).float()
    ext = []
    for d in range(0, M1.shape[0], 64):
        pix = M1[d].nonzero().squeeze(1)
        r, c = pix // 28, pix % 28
        ext.append(int(max(r.max() - r.min(), c.max() - c.min()) + 1))
    res["official_dANN_LRF_d32_s128"] = {"synapses_per_dendrite": float(M1.sum(1).mean()),
                                         "dendrites_per_soma": float(torch.tensor(mk[2].T).float().sum(1).mean()),
                                         "rf_extent_px_max": max(ext), "rf_extent_px_mean": float(np.mean(ext)),
                                         "max_synapses_in_any_4_consecutive_inputs": int(
                                             M1.reshape(M1.shape[0], -1, 4).sum(2).max())}
    print(json.dumps(res, indent=1))
    save_json(f"{LOGS}/check_g1_vs_t1.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["dannG", "dann24", "mlp", "mlpreg", "cnn", "mlpsp"])
    ap.add_argument("--sparsity", type=float, default=0.9, help="mlpsp: fraction of l1 / l2 weights set to zero")
    ap.add_argument("--prune", default="random", choices=["random", "magnitude"], help="mlpsp mask")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--s", type=int, default=128)
    ap.add_argument("--sg", type=int, default=4)
    ap.add_argument("--sp24", action="store_true")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--dense_epochs", type=int, default=5)
    ap.add_argument("--no_export", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fair", action="store_true", help="T6-F: AdamW(lr, wd), name suffix _lr*_wd*_fair; select on val_acc_last")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    if a.check:
        check()
    else:
        train(a)


if __name__ == "__main__":
    main()
