"""T6 module 3 (G3, permuted MNIST, 10 tasks): AD-cache = active-dendrites MLP whose dendritic gates
are cached per context; deployment = prototype lookup + feed-forward + one
element-wise multiply per layer (+ k-WTA).

Model / training are copied from src/t1/active_dendrites.py (Numenta DendriticMLP + PROTOTYPE_10,
src/t1 not modified), generalised to hidden width h, with two feed-forward sparsity patterns:
  ff = rand : fixed random 50 % mask per output unit (as T1 / nupic SparseWeights)
  ff = 24   : fixed random 2:4 mask (2 of every 4 consecutive inputs; 50 % as well), so the
              deployed layers are eligible for TensorRT structured-sparse tactics.
k-WTA 5 % of h, k segments (default 4), Adam 5e-4, 3 epochs / task, batch 256, prototype contexts.
Deployment graph (AD-cache): x -> argmin_t(|p_t|^2 - 2 x.p_t) -> gather cached gates g1[t], g2[t]
-> kWTA(W1 x * g1) -> kWTA(W2 h * g2) -> W3 h  (impl gemm | conv 1x1).
Baseline: dense MLP 784-h-h-10 (ReLU), same sequential protocol, no context (as T1).
Usage: python ad_cache.py --arch {ad,mlp} --h 1024 [--ff rand|24] [--k 4] [--seed 1]
       python ad_cache.py --check
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LOGS, MODELS, cap_gpu_mem, export_onnx, is_2to4, load_vision, save_json, set_seed  # noqa: E402

N_TASKS, KW, W_DENSITY, CTX_ON = 10, 0.05, 0.5, 0.1
MNIST_MEAN, MNIST_STD = 0.13062755, 0.30810780


def sparse_init_(lin, in_density, gen, ff="rand", density=W_DENSITY):
    fan_in = lin.weight.shape[1]
    bound = 1.0 / math.sqrt(in_density * density * fan_in)
    with torch.no_grad():
        lin.weight.uniform_(-bound, bound)
        mask = torch.zeros_like(lin.weight)
        if ff == "rand":   # exactly round(density*fan_in) non-zeros per output unit (T1 / nupic)
            n_on = int(round(density * fan_in))
            for i in range(mask.shape[0]):
                mask[i, torch.randperm(fan_in, generator=gen)[:n_on]] = 1
        else:              # random 2:4 pattern along the input dimension
            r = torch.rand(mask.shape[0], fan_in // 4, 4, generator=gen)
            idx = r.topk(2, dim=2).indices
            mask = torch.zeros(mask.shape[0], fan_in // 4, 4).scatter_(2, idx, 1.0).reshape(mask.shape)
        lin.weight.mul_(mask)
    lin.register_buffer("wmask", mask)


def kwta(x, k):
    thr = torch.topk(x, k, dim=1).values[:, -1:]
    return x * (x >= thr).to(x.dtype)


class ADLayer(nn.Module):
    def __init__(self, n_in, n_out, segments, in_density, gen, ff, dim_ctx=784):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        sparse_init_(self.lin, in_density, gen, ff)
        self.n_out, self.k = n_out, segments
        self.seg = nn.Linear(dim_ctx, n_out * segments, bias=False)
        bound = 1.0 / math.sqrt(CTX_ON * dim_ctx)
        nn.init.uniform_(self.seg.weight, -bound, bound)

    def gate(self, c):
        d = self.seg(c).view(-1, self.n_out, self.k)
        idx = d.abs().argmax(dim=2, keepdim=True)
        return torch.sigmoid(torch.gather(d, 2, idx).squeeze(2))

    def forward(self, x, c):
        return self.lin(x) * self.gate(c)


class ADNet(nn.Module):
    def __init__(self, h=2048, segments=4, seed=0, ff="rand"):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.l1 = ADLayer(784, h, segments, 1.0, gen, ff)
        self.l2 = ADLayer(h, h, segments, KW, gen, ff)
        self.out = nn.Linear(h, 10)
        sparse_init_(self.out, KW, gen, ff)
        self.kw = int(round(KW * h))

    def rezero(self):
        with torch.no_grad():
            for lin in (self.l1.lin, self.l2.lin, self.out):
                lin.weight.mul_(lin.wmask)

    def effective_params(self):
        n = sum(p.numel() for p in self.parameters())
        for lin in (self.l1.lin, self.l2.lin, self.out):
            n -= int((lin.wmask == 0).sum())
        return n

    def forward(self, x, c):
        h = kwta(self.l1(x, c), self.kw)
        h = kwta(self.l2(h, c), self.kw)
        return self.out(h)


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(784, h), nn.Linear(h, h), nn.Linear(h, 10)
        self.impl = "gemm"

    def forward(self, x, c=None):
        if self.impl == "conv":
            z = x.reshape(1, 784, 1, 1)
            z = F.relu(F.conv2d(z, self.l1.weight[:, :, None, None], self.l1.bias))
            z = F.relu(F.conv2d(z, self.l2.weight[:, :, None, None], self.l2.bias))
            return self.out(z.reshape(1, -1))
        return self.out(F.relu(self.l2(F.relu(self.l1(x)))))

    def rezero(self):
        pass

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPCtx(nn.Module):
    """Dense baseline WITH task context: 784-h-h-10 MLP whose input is x concatenated
    with the one-hot task id of the context. The task id is the index of the prototype that equals the
    given context (training: the current task's prototype; test: the nearest prototype of x) - i.e. the
    same information AD-cache receives."""

    def __init__(self, h, protos):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(784 + 10, h), nn.Linear(h, h), nn.Linear(h, 10)
        self.register_buffer("protos", protos.clone())
        self.register_buffer("pt", protos.t().contiguous())
        self.register_buffer("pn", (protos ** 2).sum(1, keepdim=True).t())
        self.register_buffer("eye", torch.eye(10))
        self.impl = "gemm"

    def task_of(self, x):   # nearest prototype (same rule as AD-cache deployment)
        return torch.argmin(self.pn - 2.0 * torch.matmul(x, self.pt), dim=1)

    def forward(self, x, c=None):
        t = self.task_of(x) if c is None else torch.cdist(c, self.protos).argmin(1)
        z = torch.cat([x, self.eye[t]], dim=1)
        return self.out(F.relu(self.l2(F.relu(self.l1(z)))))

    def rezero(self):
        pass

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


class ADCached(nn.Module):
    """Deployment graph (as src/t1/ad_cached_export.ADCached, plus a 1x1-conv variant)."""

    def __init__(self, m, protos):
        super().__init__()
        self.register_buffer("pt", protos.t().contiguous())
        self.register_buffer("pn", (protos ** 2).sum(1, keepdim=True).t())
        with torch.no_grad():
            self.register_buffer("g1", m.l1.gate(protos))   # (10, h)
            self.register_buffer("g2", m.l2.gate(protos))
        self.l1, self.l2, self.out, self.kw = m.l1.lin, m.l2.lin, m.out, m.kw
        self.impl = "gemm"

    def forward(self, x):
        t = torch.argmin(self.pn - 2.0 * torch.matmul(x, self.pt), dim=1)
        if self.impl == "conv":
            z = x.reshape(1, 784, 1, 1)
            z = F.conv2d(z, self.l1.weight[:, :, None, None], self.l1.bias).reshape(1, -1)
            z = kwta(z * self.g1[t], self.kw)
            z = F.conv2d(z.reshape(1, -1, 1, 1), self.l2.weight[:, :, None, None], self.l2.bias).reshape(1, -1)
            z = kwta(z * self.g2[t], self.kw)
            return self.out(z)
        z = kwta(self.l1(x) * self.g1[t], self.kw)
        z = kwta(self.l2(z) * self.g2[t], self.kw)
        return self.out(z)


def pmnist(dev):
    d = load_vision("mnist", dev)
    xtr, ytr = d["train"]
    xte, yte = d["test"]
    xtr, xte = (xtr - MNIST_MEAN) / MNIST_STD, (xte - MNIST_MEAN) / MNIST_STD
    g = torch.Generator().manual_seed(1234)  # task permutations shared by all models (as T1)
    perms = [torch.randperm(784, generator=g).to(dev) for _ in range(N_TASKS)]
    protos = torch.stack([xtr[:, p].mean(0) for p in perms])
    return xtr, ytr, xte, yte, perms, protos


def train(a):
    cap_gpu_mem()
    dev = os.environ.get("T6_DEV", "cuda")
    set_seed(a.seed)
    if a.arch == "ad":
        cfg = f"adc_h{a.h}_k{a.k}_ff{a.ff}"
        model = ADNet(a.h, a.k, seed=a.seed, ff=a.ff)
    elif a.arch == "mlpctx":
        cfg = f"mlpctx_h{a.h}"
        model = None   # needs the prototypes, built below
    else:
        cfg = f"mlp_h{a.h}"
        model = MLP(a.h)
    if a.fair:  # fair protocol (T6-F): 10 % validation split of the training set, equal search space (lr x2, h x4, wd x2)
        cfg += f"_lr{a.lr:g}_wd{a.wd:g}_fair"
    name = f"{cfg}_t{a.seed}"
    od = f"{MODELS}/g3"
    os.makedirs(od, exist_ok=True)
    if os.path.exists(f"{od}/{name}.json") and not a.force:
        print(f"{name}: done already, skipping")
        return
    t0 = time.time()
    xtr, ytr, xte, yte, perms, protos = pmnist(dev)
    xva = yva = None
    if a.fair:  # validation = 10 % of the training images (seeded by trial), same permutations; prototypes unchanged
        gv = torch.Generator().manual_seed(1000 + a.seed)
        idx = torch.randperm(len(xtr), generator=gv).to(dev)
        nval = len(xtr) // 10
        xva, yva = xtr[idx[:nval]], ytr[idx[:nval]]
        xtr, ytr = xtr[idx[nval:]], ytr[idx[nval:]]
    if model is None:
        model = MLPCtx(a.h, protos)
    model.to(dev)

    def eval_task(t, bs=2000, xs=None, ys=None):
        xs = xte if xs is None else xs; ys = yte if ys is None else ys
        model.eval()
        c = 0
        with torch.no_grad():
            for i in range(0, len(xs), bs):
                x = xs[i:i + bs][:, perms[t]]
                ctx = protos[torch.cdist(x, protos).argmin(1)]
                c += (model(x, ctx).argmax(1) == ys[i:i + bs]).sum().item()
        return c / len(xs)

    acc = []
    log = open(f"{LOGS}/g3_{name}.log", "w")
    for t in range(N_TASKS):
        opt = (torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd) if a.fair
               else torch.optim.Adam(model.parameters(), lr=5e-4))
        xt = xtr[:, perms[t]]
        ct = protos[t].expand(256, -1)
        model.train()
        for ep in range(3):
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(xt), 256):
                b = perm[i:i + 256]
                loss = F.cross_entropy(model(xt[b], ct[:len(b)]), ytr[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.rezero()
        row = [eval_task(j) for j in range(t + 1)]
        acc.append(row)
        s = f"{name} after task {t + 1}: mean {sum(row) / len(row):.4f}"
        print(s, flush=True)
        log.write(s + "\n")
    final = acc[-1]
    bwt = sum(final[j] - acc[j][j] for j in range(N_TASKS - 1)) / (N_TASKS - 1)
    meta = {"name": name, "cfg": cfg, "task": "G3", "dataset": "permuted MNIST (10 tasks)", "arch": a.arch,
            "args": vars(a), "test_acc": sum(final) / N_TASKS, "final_per_task": final, "bwt": bwt,
            "selection": ("config chosen on val_acc (final model, mean over tasks; T6-F)" if a.fair
                          else "none (fixed protocol, final model, as T1/nupic)"),
            "val_acc": (sum(eval_task(j, xs=xva, ys=yva) for j in range(N_TASKS)) / N_TASKS) if a.fair else None,
            "params_effective": model.effective_params(),
            "params_stored": sum(p.numel() for p in model.parameters()),
            "train_minutes": round((time.time() - t0) / 60, 2)}
    if a.arch == "ad":
        meta["is_2to4"] = all(is_2to4(l.weight, 1) for l in (model.l1.lin, model.l2.lin, model.out)) if a.ff == "24" else None
        model.eval().cpu()
        dep = ADCached(model, protos.cpu()).eval()
        x = xte[:2000, perms[3]].cpu()
        with torch.no_grad():
            ref = model(x, protos.cpu()[torch.cdist(x, protos.cpu()).argmin(1)])
            got = dep(x)
        meta["cached_vs_two_input_max_abs_diff"] = float((ref - got).abs().max())
        meta["cached_vs_two_input_argmax_agree"] = float((ref.argmax(1) == got.argmax(1)).float().mean())
    torch.save(model.state_dict(), f"{od}/{name}.pt")
    save_json(f"{od}/{name}.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "final_per_task"}), flush=True)


class _LutAD(nn.Module):
    def __init__(self, h, ff):
        super().__init__()
        m = ADNet(h, 4, seed=0, ff=ff).eval()
        self.dep = ADCached(m, torch.randn(10, 784)).eval()
        self.impl = "gemm"

    def forward(self, x):
        self.dep.impl = self.impl
        return self.dep(x)


def lut_configs():
    out = []
    for h in (256, 512, 1024, 2048, 4096):
        out.append((f"adc_h{h}_ffrand", lambda h=h: _LutAD(h, "rand"), ["gemm", "conv"], False))
        out.append((f"adc_h{h}_ff24", lambda h=h: _LutAD(h, "24"), ["gemm", "conv"], h >= 1024))
    for h in (128, 256, 512, 1024, 2048, 4096):
        out.append((f"mlp_h{h}", lambda h=h: MLP(h), ["gemm", "conv"], False))
    for h in (256, 512, 1024, 2048, 4096):   # dense baseline with task context (deployed: lookup + one-hot)
        out.append((f"mlpctx_h{h}", lambda h=h: _LutCtx(h), ["gemm"], False))
    return out


class _LutCtx(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.m = MLPCtx(h, torch.randn(10, 784)).eval()
        self.impl = "gemm"

    def forward(self, x):
        return self.m(x)


def check():
    """(1) this ADNet with h=2048, ff=rand reproduces the T1 ADNet (src/t1/active_dendrites.py) on
    the T1 checkpoint; (2) cached graph == two-input graph."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "t1"))
    from active_dendrites import ADNet as T1ADNet  # read-only
    res = {}
    sd = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results/T1/models/active_dend_k4.pt"), map_location="cpu")
    t1 = T1ADNet(4).eval()
    t1.load_state_dict(sd)
    mine = ADNet(2048, 4, seed=0, ff="rand").eval()
    mine.load_state_dict(sd)
    xtr, ytr, xte, yte, perms, protos = pmnist("cpu")
    x = xte[:2000, perms[5]]
    c = protos[torch.cdist(x, protos).argmin(1)]
    with torch.no_grad():
        r1, r2 = t1(x, c), mine(x, c)
        dep = ADCached(mine, protos).eval()
        r3 = dep(x)
        dep.impl = "conv"
        r4 = torch.cat([dep(x[i:i + 1]) for i in range(64)])
    res["mine_vs_T1_ADNet_max_abs_diff"] = float((r1 - r2).abs().max())
    res["init_masks_identical_to_T1_seed0"] = bool(all(torch.equal(a_.lin.wmask, b_.lin.wmask) for a_, b_ in
                                                       ((ADNet(2048, 4, 0).l1, T1ADNet(4, seed=0).l1),)))
    res["cached_gemm_vs_two_input_max_abs_diff"] = float((r3 - r1).abs().max())
    res["cached_gemm_argmax_agree"] = float((r3.argmax(1) == r1.argmax(1)).float().mean())
    res["cached_conv_vs_two_input_max_abs_diff_64"] = float((r4 - r1[:64]).abs().max())
    print(json.dumps(res, indent=1))
    save_json(f"{LOGS}/check_g3_vs_t1.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["ad", "mlp", "mlpctx"], default="ad")
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ff", choices=["rand", "24"], default="rand")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fair", action="store_true", help="T6-F: val split, AdamW(lr, wd), name suffix _lr*_wd*_fair")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    a = ap.parse_args()
    check() if a.check else train(a)


if __name__ == "__main__":
    main()
