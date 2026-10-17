"""T1 group 2: DH-SNN (DH-SRNN) vs LIF-SNN (vanilla SRNN) vs GRU on SHD.

Ported from the official DH-SNN code (src/third_party/DH-SNN/SHD, commit 79ab1fb):
  * data: `shd_generate_dataset.binary_image_readout`, dt = 1 ms, T = 1000 steps, 700 channels
    (spike at time t goes to bin ceil(t/dt); unit 0 dropped; channel = 700 - unit; binary).
  * DH-SRNN = `main_rnn_denri.py`: spike_rnn_test_denri_wotanh_R(700, 64, branch=8,
    tau_n ~ U(2,6), tau_m ~ U(0,4), vth=1) + readout_integrator_test(64, 20); output =
    sum_{t>10} softmax(readout mem); CrossEntropy on that sum; Adam lr 1e-2 (tau lr 2e-2),
    StepLR(20, 0.5), batch 100, 100 epochs; mask re-applied after every step; MG surrogate.
  * LIF-SNN = `main_rnn_origin.py`: spike_rnn_test_origin(700, 64), same readout / recipe.
  * GRU: nn.GRU(700, h) with h chosen to match the DH-SRNN parameter count, linear readout,
    CE on the time-averaged logits (t > 10).
Deviations from the official code: initial neuron states are zeros instead of torch.rand
(so the deployed model is deterministic); the input projection W_in x_t is computed for all
t at once (mathematically identical to dense(cat(x_t, s_{t-1})) because the layer is linear).

Exports: full sequence (x: 1x1000x700 -> 1x20, loop unrolled) and one streaming step
(x_t + states -> out + new states).
Usage: python shd.py --model {dhsrnn,srnn,gru} [--epochs 100]
"""
import argparse
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, MODELS, export_onnx, save_meta, set_seed  # noqa: E402

T_STEPS, N_IN, N_CLS = 1000, 700, 20


# ---------------------------------------------------------------- data
def load_shd(split, device="cuda"):
    """Returns flat event index (sample offsets + (bin*700+ch)) and labels, on device."""
    with h5py.File(f"{DATA}/SHD/shd_{split}.h5", "r") as f:
        times, units, labels = f["spikes/times"][:], f["spikes/units"][:], f["labels"][:]
    flat, offs = [], [0]
    for t, u in zip(times, units):
        t = t.astype(np.float64)
        b = np.ceil(t / 1e-3 - 1e-9).astype(np.int64)   # first i with t <= i*dt
        b[b < 0] = 0
        keep = (b < T_STEPS) & (u > 0)
        idx = np.unique(b[keep] * N_IN + (N_IN - u[keep].astype(np.int64)))  # binary
        flat.append(idx)
        offs.append(offs[-1] + len(idx))
    flat = torch.tensor(np.concatenate(flat), dtype=torch.int32, device=device)
    offs = torch.tensor(offs, dtype=torch.int64, device=device)
    return flat, offs, torch.tensor(labels.astype(np.int64), device=device)


def make_batch(flat, offs, ids):
    """Dense binary batch (B, T, 700). Vectorised: one scatter instead of a per-sample
    loop with host syncs; result identical to the v0.5 loop version."""
    dev = flat.device
    ids = ids.to(dev)
    start, end = offs[ids], offs[ids + 1]
    lens = end - start
    row = torch.repeat_interleave(torch.arange(len(ids), device=dev), lens)
    pos = torch.arange(int(lens.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
    idx = flat[(torch.repeat_interleave(start, lens) + pos)].long() + row * (T_STEPS * N_IN)
    x = torch.zeros(len(ids) * T_STEPS * N_IN, device=dev)
    x[idx] = 1.0
    return x.view(len(ids), T_STEPS, N_IN)


def make_batch_loop(flat, offs, ids):  # v0.5 reference implementation (used in a check only)
    x = torch.zeros(len(ids), T_STEPS * N_IN, device=flat.device)
    for j, i in enumerate(ids.tolist()):
        x[j, flat[offs[i]:offs[i + 1]].long()] = 1.0
    return x.view(len(ids), T_STEPS, N_IN)


# ---------------------------------------------------------------- spiking models
class MGSpike(torch.autograd.Function):
    """ActFun_adp with the multi-Gaussian surrogate (gamma=.5, lens=.5, scale=6, hight=.15)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        lens, scale, hight = 0.5, 6.0, 0.15

        def gauss(v, mu, s):
            return torch.exp(-((v - mu) ** 2) / (2 * s ** 2)) / math.sqrt(2 * math.pi) / s
        temp = gauss(x, 0., lens) * (1. + hight) - gauss(x, lens, scale * lens) * hight \
            - gauss(x, -lens, scale * lens) * hight
        return g * temp * 0.5


class SRNN(nn.Module):
    """DH-SRNN (branch > 0) or vanilla LIF SRNN (branch = 0), 1 layer + leaky readout."""

    def __init__(self, n=64, branch=8, seed=0, vth=1.0):
        super().__init__()
        self.n, self.branch, self.vth, self.export_mode = n, branch, vth, False
        nin = N_IN + n
        if branch:
            self.pad = (nin // branch * branch + branch - nin) % branch
            self.dense = nn.Linear(nin + self.pad, n * branch)
            self.tau_n = nn.Parameter(torch.empty(n, branch).uniform_(2, 6))
            size = nin + self.pad
            mask = torch.zeros(n * branch, size)
            g = torch.Generator().manual_seed(seed)
            for i in range(n):
                seq = torch.randperm(size, generator=g)
                for j in range(branch):
                    mask[i * branch + j, seq[j * size // branch:(j + 1) * size // branch]] = 1
            self.register_buffer("mask", mask)
        else:
            self.pad = 0
            self.dense = nn.Linear(nin, n)
            self.register_buffer("mask", torch.ones(n, nin))
        self.tau_m = nn.Parameter(torch.empty(n).uniform_(0, 4))
        self.readout = nn.Linear(n, N_CLS)
        nn.init.xavier_normal_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.ro_tau = nn.Parameter(torch.empty(N_CLS).uniform_(0, 4))
        self.apply_mask()

    def apply_mask(self):
        with torch.no_grad():
            self.dense.weight.mul_(self.mask)

    def spike(self, v):
        return (v > 0).float() if self.export_mode else MGSpike.apply(v)

    def init_state(self, b, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        d = z(b, self.n, self.branch) if self.branch else z(b, 1)
        return [d, z(b, self.n), z(b, self.n), z(b, N_CLS)]

    def in_proj(self, x):  # W_in x_t + b for all t (x: B,T,700 or B,700)
        return F.linear(x, self.dense.weight[:, :N_IN], self.dense.bias)

    def step(self, xp, state):
        d, mem, spk, ro = state
        cur = xp + F.linear(spk, self.dense.weight[:, N_IN:N_IN + self.n])
        if self.branch:
            beta = torch.sigmoid(self.tau_n)
            d = beta * d + (1 - beta) * cur.view(-1, self.n, self.branch)
            cur = d.sum(2)
        alpha = torch.sigmoid(self.tau_m)
        mem = mem * alpha + (1 - alpha) * cur - self.vth * spk
        spk = self.spike(mem - self.vth)
        a = torch.sigmoid(self.ro_tau)
        ro = ro * a + (1 - a) * self.readout(spk)
        return [d, mem, spk, ro], ro

    def forward(self, x):
        b, T, _ = x.shape
        xp = self.in_proj(x)
        st = self.init_state(b, x.device)
        out = 0
        for t in range(T):
            st, ro = self.step(xp[:, t], st)
            if t > 10:
                out = out + F.softmax(ro, dim=1)
        return out

    def effective_params(self):
        w = int(self.mask[:, :N_IN + self.n].sum())
        n = w + self.dense.bias.numel() + self.tau_m.numel() + self.readout.weight.numel() \
            + self.readout.bias.numel() + self.ro_tau.numel()
        return n + (self.tau_n.numel() if self.branch else 0)


class SNNStep(nn.Module):
    """Streaming export: one time step with explicit state tensors."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x_t, d, mem, spk, ro):
        st, out = self.m.step(self.m.in_proj(x_t), [d, mem, spk, ro])
        return (out,) + tuple(st)


# ---------------------------------------------------------------- GRU
class GRUNet(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gru = nn.GRU(N_IN, h, batch_first=True)
        self.readout = nn.Linear(h, N_CLS)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.readout(y[:, 11:]).mean(1)  # time-averaged logits, t > 10

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


class GRUStep(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x_t, h):
        y, h2 = self.m.gru(x_t.unsqueeze(1), h)
        return self.m.readout(y[:, 0]), h2


def gru_hidden_for(params):
    # 3*(700h + h^2 + 2h) + 20h + 20 = params
    a, b, c = 3, 3 * (N_IN + 2) + N_CLS, N_CLS - params
    return int(round((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)))


# ---------------------------------------------------------------- train / eval
def evaluate(model, data, bs=250):
    flat, offs, y = data
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            ids = torch.arange(i, min(i + bs, len(y)))
            correct += (model(make_batch(flat, offs, ids)).argmax(1) == y[ids.cuda()]).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dhsrnn", "srnn", "gru"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=0, help="debug: limit batches/epoch")
    ap.add_argument("--no_export", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    train, test = load_shd("train"), load_shd("test")
    print(f"data loaded in {time.time() - t0:.0f}s: train {len(train[2])} test {len(test[2])}", flush=True)

    ref_params = SRNN(64, 8, seed=args.seed).effective_params()
    if args.model == "dhsrnn":
        model, name = SRNN(64, 8, seed=args.seed), "dhsrnn_n64_b8"
    elif args.model == "srnn":
        model, name = SRNN(64, 0, seed=args.seed), "lif_srnn_n64"
    else:
        h = gru_hidden_for(ref_params)
        model, name = GRUNet(h), f"gru_h{h}"
    model.cuda()

    if args.model == "gru":
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        bs = 100
    else:  # official parameter groups
        base = [model.dense.weight, model.dense.bias, model.readout.weight, model.readout.bias]
        groups = [{"params": base, "lr": 1e-2},
                  {"params": [model.ro_tau], "lr": 2e-2}, {"params": [model.tau_m], "lr": 2e-2}]
        if args.model == "dhsrnn":
            groups.append({"params": [model.tau_n], "lr": 2e-2})
        opt = torch.optim.Adam(groups, lr=1e-2)
        bs = 100
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 5), gamma=0.5)

    flat, offs, y = train
    g = torch.Generator().manual_seed(args.seed)
    hist = []
    for ep in range(args.epochs):
        model.train()
        te0 = time.time()
        perm = torch.randperm(len(y), generator=g)
        tl, tc, nb = 0.0, 0, 0
        for i in range(0, len(y), bs):
            ids = perm[i:i + bs]
            xb, yb = make_batch(flat, offs, ids), y[ids.cuda()]
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.model == "gru":
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if args.model != "gru":
                model.apply_mask()
            tl += loss.item()
            tc += (out.argmax(1) == yb).sum().item()
            nb += 1
            if args.max_batches and nb >= args.max_batches:
                break
        sched.step()
        te = evaluate(model, test)
        hist.append(te)
        print(f"{name} ep {ep + 1} loss {tl / nb:.4f} train {tc / min(len(y), nb * bs):.4f} "
              f"test {te:.4f} ({time.time() - te0:.0f}s)", flush=True)

    meta = {"name": name, "group": "shd", "dataset": "SHD", "model": args.model, "T": T_STEPS,
            "dt_ms": 1, "epochs": args.epochs, "seed": args.seed,
            "test_acc": hist[-1], "test_acc_best_epoch": max(hist),
            "params_effective": model.effective_params(),
            "params_stored": sum(p.numel() for p in model.parameters()),
            "dhsrnn_params_effective": ref_params, "train_minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(meta), flush=True)
    torch.save(model.state_dict(), f"{MODELS}/{name}.pt")
    if not args.no_export:
        export(model, name, meta)
    save_meta(name, meta)


def export(model, name, meta):
    model.eval().cpu()
    if isinstance(model, SRNN):
        model.export_mode = True
        # sanity: export-mode forward == training-mode forward on a test sample
        meta["export_seq"] = export_onnx(model, (torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.02),),
                                         f"{MODELS}/{name}_seq.onnx", ["x"], ["out"])
        st = model.init_state(1, "cpu")
        meta["export_step"] = export_onnx(SNNStep(model), (torch.zeros(1, N_IN).bernoulli_(0.02), *st),
                                          f"{MODELS}/{name}_step.onnx",
                                          ["x_t", "d", "mem", "spk", "ro"],
                                          ["out", "d_o", "mem_o", "spk_o", "ro_o"])
    else:
        meta["export_seq"] = export_onnx(model, (torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.02),),
                                         f"{MODELS}/{name}_seq.onnx", ["x"], ["out"])
        h = torch.zeros(1, 1, model.gru.hidden_size)
        meta["export_step"] = export_onnx(GRUStep(model), (torch.zeros(1, N_IN).bernoulli_(0.02), h),
                                          f"{MODELS}/{name}_step.onnx", ["x_t", "h"], ["out", "h_o"])


if __name__ == "__main__":
    main()
