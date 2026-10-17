"""T6 module 2 (G2, SHD): DH-SSM = DH-SNN dendritic branches written as a diagonal linear recurrence,
trained with a chunked parallel scan and deployed in the same chunked form (no ONNX Loop).

One DH-SSM layer (n somata, nb branches per soma, input Cin):
  c_t   = grouped projection: branch b of every soma sees input-channel group b (contiguous
          Cin/nb channels; SHD channels are frequency bands, so a branch = a frequency band).
          Grouped GEMM / grouped 1x1 conv, no gather.
  d_t   = beta * d_{t-1} + (1 - beta) * c_t          beta = sigmoid(tau_n) per (branch, soma)
  I_t   = sum_b d_t[b]
  m_t   = alpha * m_{t-1} + (1 - alpha) * I_t        alpha = sigmoid(tau_m) per soma
  y_t   = H(m_t - vth)  (spk: spike, multi-Gaussian surrogate as T1/DH-SNN, NO reset)
        | relu(m_t)     (cont: continuous-valued version)
Readout (as DH-SNN): ro_t = a*ro_{t-1} + (1-a)*(W y_t + b); output = sum_{t>10} softmax(ro_t).
Differences to the T1 DH-SRNN (checked numerically in --check): no recurrent weights (the
dendritic input is feed-forward, which is what makes the recurrence linear), no reset
(m_t is a linear filter of the input), branch input partition is contiguous channel groups
instead of a per-neuron random partition, optional stacking of L layers, dropout between layers.

Chunked scan (training and deployment use the same code): for a diagonal recurrence
d_t = lam*d_{t-1} + u_t with T = nch*K:
  intra-chunk  y[i,k] = sum_{j<=k} lam^(k-j) u[i,j]         batched MatMul (C, nch, K) x (C, K, K)
  chunk states S[i]   = sum_{i'<=i} (lam^K)^(i-i') y[i',K-1]  batched MatMul (C, nch, nch)
  output       d[i,k] = y[i,k] + lam^(k+1) S[i-1]
Layout (C, T) throughout (conv style), so no transposes are needed around the scans.
K is a pure deployment parameter (the function does not depend on it); it is swept in the LUT.

Data / split: SHD, dt = 1 ms, T = 1000 (T1 binning); validation = the T1 split
(results/T1/data/shd_val_split.json: 10 % of train, stratified, seed 2026); model selection
= best validation epoch; test is only logged.
Usage: python dh_ssm.py --n 128 --nb 4 --L 2 --act spk [--seed 1] [--epochs 100]
       python dh_ssm.py --check
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
from common6 import DATA, LOGS, ROOT, MODELS, cap_gpu_mem, export_onnx, save_json, set_seed  # noqa: E402

T_STEPS, N_IN, N_CLS = 1000, 700, 20
SPLIT = f"{ROOT}/results/T1/data/shd_val_split.json"                  # T1 split (random 10 %, same speakers)
SPK_SPLIT = os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results/T6/data/shd_val_speaker_split.json")
VAL_SPEAKERS = (3, 6)   # the two training speakers with the fewest samples (722 + 447 = 14.3 %)


def speaker_val_ids():
    """Speaker-disjoint validation split (SHD train speakers 0,1,2,3,6,7,8,9,10,11)."""
    if os.path.exists(SPK_SPLIT):
        return json.load(open(SPK_SPLIT))["val_ids"]
    with h5py.File(f"{DATA}/SHD/shd_train.h5", "r") as f:
        spk, lab = f["extra/speaker"][:], f["labels"][:]
    val = np.where(np.isin(spk, VAL_SPEAKERS))[0]
    info = {"rule": "validation = all training samples of speakers 3 and 6 (the two smallest training speakers)",
            "val_speakers": list(VAL_SPEAKERS), "n_val": int(len(val)), "n_train": int(len(spk) - len(val)),
            "val_classes_present": int(len(np.unique(lab[val]))), "val_ids": val.tolist()}
    os.makedirs(os.path.dirname(SPK_SPLIT), exist_ok=True)
    json.dump(info, open(SPK_SPLIT, "w"))
    return info["val_ids"]


# ------------------------------------------------------------------ data (copied from src/t1/shd.py)
def load_shd(split, device="cuda"):
    with h5py.File(f"{DATA}/SHD/shd_{split}.h5", "r") as f:
        times, units, labels = f["spikes/times"][:], f["spikes/units"][:], f["labels"][:]
    flat, offs = [], [0]
    for t, u in zip(times, units):
        t = t.astype(np.float64)
        b = np.ceil(t / 1e-3 - 1e-9).astype(np.int64)
        b[b < 0] = 0
        keep = (b < T_STEPS) & (u > 0)
        idx = np.unique(b[keep] * N_IN + (N_IN - u[keep].astype(np.int64)))
        flat.append(idx)
        offs.append(offs[-1] + len(idx))
    flat = torch.tensor(np.concatenate(flat), dtype=torch.int32, device=device)
    offs = torch.tensor(offs, dtype=torch.int64, device=device)
    return flat, offs, torch.tensor(labels.astype(np.int64), device=device)


def make_batch(flat, offs, ids):
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


class MGSpike(torch.autograd.Function):
    """Heaviside with the DH-SNN multi-Gaussian surrogate (copied from src/t1/shd.py)."""

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


# ------------------------------------------------------------------ chunked diagonal scan
def scan_mats(lam, K, nch):
    """lam (C,) -> intra (C, K, K) with [k, j] = lam^(k-j) (k >= j); inter (C, nch, nch) with
    [i, i'] = lam^(K (i - i')) (i >= i'); carry (C, K) = lam^(k+1)."""
    ll = torch.log(lam.clamp_min(1e-12))[:, None, None]
    k = torch.arange(K, device=lam.device, dtype=lam.dtype)
    dk = (k[:, None] - k[None, :])
    intra = torch.where(dk >= 0, torch.exp(ll * dk.clamp_min(0)), torch.zeros((), device=lam.device))
    i = torch.arange(nch, device=lam.device, dtype=lam.dtype)
    di = (i[:, None] - i[None, :])
    inter = torch.where(di >= 0, torch.exp(ll * K * di.clamp_min(0)), torch.zeros((), device=lam.device))
    carry = torch.exp(ll[:, :, 0] * (k[None, :] + 1))
    return intra, inter, carry


def chunk_scan(u, lam, K, mats=None):
    """u (B, C, T) -> d (B, C, T) with d_t = lam * d_{t-1} + u_t, d_{-1} = 0."""
    B, C, T = u.shape
    nch = T // K
    assert nch * K == T, (T, K)
    intra, inter, carry = mats if mats is not None else scan_mats(lam, K, nch)
    uc = u.reshape(B, C, nch, K)
    y = torch.matmul(uc, intra.transpose(1, 2))                    # (B,C,nch,K) @ (C,K,K)
    e = y[..., K - 1:K]                                            # (B,C,nch,1) local chunk-end states
    S = torch.matmul(inter, e)                                     # (C,nch,nch) @ (B,C,nch,1)
    Sprev = F.pad(S[:, :, :-1], (0, 0, 1, 0))                      # state at the end of chunk i-1
    y = y + Sprev * carry[:, None, :]
    return y.reshape(B, C, T)


def seq_scan(u, lam):
    """Reference sequential recurrence (for the check)."""
    d = torch.zeros_like(u[..., 0])
    out = []
    for t in range(u.shape[-1]):
        d = lam * d + u[..., t]
        out.append(d)
    return torch.stack(out, -1)


# ------------------------------------------------------------------ model
class GroupedConv1x1(nn.Module):
    def __init__(self, cin, cout, groups):
        super().__init__()
        assert cin % groups == 0 and cout % groups == 0
        self.g = groups
        self.weight = nn.Parameter(torch.empty(cout, cin // groups))
        self.bias = nn.Parameter(torch.zeros(cout))
        bound = 1.0 / math.sqrt(cin // groups)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, scale=None):  # x (B, cin, T)
        w, b = self.weight, self.bias
        if scale is not None:           # fold a per-output-channel scale (1 - beta) into W, b
            w, b = w * scale[:, None], b * scale
        return F.conv1d(x, w[:, :, None], b, groups=self.g)


class DHSSMLayer(nn.Module):
    def __init__(self, cin, n, nb, act, vth=1.0, grouped=True):
        super().__init__()
        g = nb if grouped else 1
        # grouped = dendritic: branch b sees input-channel group b only.  grouped=False (
        # NON-dendritic multi-timescale diagonal SSM): every one of the nb filters of a soma sees ALL
        # inputs (dense projection), same state count / scans / deployment as DH-SSM.
        self.cin_pad = int(math.ceil(cin / (8 * g)) * 8 * g)    # group width multiple of 8
        self.cin, self.n, self.nb, self.act, self.vth = cin, n, nb, act, vth
        self.proj = GroupedConv1x1(self.cin_pad, n * nb, g)       # channel index = b * n + i
        self.tau_n = nn.Parameter(torch.empty(nb * n).uniform_(2, 6))
        self.tau_m = nn.Parameter(torch.empty(n).uniform_(0, 4))

    def forward(self, x, K, cache=None):  # x (B, cin, T)
        if self.cin_pad > self.cin:
            x = F.pad(x, (0, 0, 0, self.cin_pad - self.cin))
        beta, alpha = torch.sigmoid(self.tau_n), torch.sigmoid(self.tau_m)
        u = self.proj(x, 1 - beta)
        d = chunk_scan(u, beta, K, None if cache is None else cache[0])
        I = d.reshape(d.shape[0], self.nb, self.n, -1).sum(1)
        m = chunk_scan((1 - alpha)[:, None] * I, alpha, K, None if cache is None else cache[1])
        if self.act == "spk":
            return MGSpike.apply(m - self.vth) if self.training else (m > self.vth).to(m.dtype)
        return F.relu(m)


class DHSSM(nn.Module):
    def __init__(self, n=128, nb=4, L=1, act="spk", dropout=0.0, K=50, grouped=True):
        super().__init__()
        self.n, self.nb, self.L, self.act, self.K, self.p = n, nb, L, act, K, dropout
        self.layers = nn.ModuleList([DHSSMLayer(N_IN if l == 0 else n, n, nb, act, grouped=grouped)
                                     for l in range(L)])
        self.readout = nn.Conv1d(n, N_CLS, 1)
        nn.init.xavier_normal_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.ro_tau = nn.Parameter(torch.empty(N_CLS).uniform_(0, 4))
        self.impl = "train"  # "export": x (1, T, 700) input, precomputed scan matrices

    def forward(self, x):  # x (B, T, 700)
        h = x.transpose(1, 2)
        for l, layer in enumerate(self.layers):
            h = layer(h, self.K)
            if self.training and self.p > 0 and l < self.L - 1:
                h = F.dropout(h, self.p, True)
        a = torch.sigmoid(self.ro_tau)
        z = self.readout(h)
        ro = chunk_scan((1 - a)[:, None] * z, a, self.K)
        return F.softmax(ro[:, :, 11:], dim=1).sum(2)

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


class DHSSMExport(nn.Module):
    """Deployment graph: all scan matrices folded to constants for a given K; projections as grouped
    1x1 Conv (tensor-core friendly), (1-beta) folded into the projection weights."""

    def __init__(self, m, K):
        super().__init__()
        self.m, self.K = m, K
        nch = T_STEPS // K
        with torch.no_grad():
            self.mats = []
            for i, layer in enumerate(m.layers):
                beta, alpha = torch.sigmoid(layer.tau_n), torch.sigmoid(layer.tau_m)
                for j, (lam, nm) in enumerate(((beta, "b"), (alpha, "a"))):
                    for t, k in zip(scan_mats(lam, K, nch), ("intra", "inter", "carry")):
                        self.register_buffer(f"{k}_{nm}{i}", t.clone())
            for t, k in zip(scan_mats(torch.sigmoid(m.ro_tau), K, nch), ("intra", "inter", "carry")):
                self.register_buffer(f"{k}_ro", t.clone())

    def _mats(self, nm):
        return getattr(self, f"intra_{nm}"), getattr(self, f"inter_{nm}"), getattr(self, f"carry_{nm}")

    def forward(self, x):
        m = self.m
        h = x.transpose(1, 2)
        for i, layer in enumerate(m.layers):
            if layer.cin_pad > layer.cin:
                h = F.pad(h, (0, 0, 0, layer.cin_pad - layer.cin))
            beta, alpha = torch.sigmoid(layer.tau_n), torch.sigmoid(layer.tau_m)
            u = layer.proj(h, 1 - beta)
            d = chunk_scan(u, None, self.K, self._mats(f"b{i}"))
            I = d.reshape(1, layer.nb, layer.n, -1).sum(1)
            mm = chunk_scan((1 - alpha)[:, None] * I, None, self.K, self._mats(f"a{i}"))
            h = (mm > layer.vth).to(mm.dtype) if layer.act == "spk" else F.relu(mm)
        a = torch.sigmoid(m.ro_tau)
        ro = chunk_scan((1 - a)[:, None] * m.readout(h), None, self.K, self._mats("ro"))
        return F.softmax(ro[:, :, 11:], dim=1).sum(2)


# ------------------------------------------------------------------ GRU baseline (as T1)
class GRUNetD(nn.Module):
    """nn.GRU(700, h) + linear readout, CE on time-averaged logits (t > 10); dropout on the input
    spikes and GRU outputs (as src/t1/shd_val.GRUNetD)."""

    def __init__(self, h, p=0.0):
        super().__init__()
        self.gru = nn.GRU(N_IN, h, batch_first=True)
        self.readout = nn.Linear(h, N_CLS)
        self.p = p

    def forward(self, x):
        x = F.dropout(x, self.p, self.training)
        y, _ = self.gru(x)
        y = F.dropout(y[:, 11:], self.p, self.training)
        return self.readout(y).mean(1)

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMNetD(GRUNetD):
    """LSTM baseline (ONNX LSTM op at deployment), same head / loss / dropout as GRUNetD."""

    def __init__(self, h, p=0.0):
        super().__init__(h, p)
        self.gru = nn.LSTM(N_IN, h, batch_first=True)


# ------------------------------------------------------------------ LRU baseline
def scan_mats_c(r, th, K, nch):
    """Complex diagonal lam = r e^{i th}: real / imaginary parts of the intra / inter / carry matrices."""
    lr = torch.log(r.clamp_min(1e-12))[:, None, None]
    k = torch.arange(K, device=r.device, dtype=r.dtype)
    dk = (k[:, None] - k[None, :])
    ok = dk >= 0
    dkc = dk.clamp_min(0)
    mag = torch.where(ok, torch.exp(lr * dkc), torch.zeros((), device=r.device))
    ang = th[:, None, None] * dkc
    intra = (mag * torch.cos(ang), mag * torch.sin(ang))
    i = torch.arange(nch, device=r.device, dtype=r.dtype)
    di = (i[:, None] - i[None, :])
    oki = di >= 0
    dic = di.clamp_min(0) * K
    magi = torch.where(oki, torch.exp(lr * dic), torch.zeros((), device=r.device))
    angi = th[:, None, None] * dic
    inter = (magi * torch.cos(angi), magi * torch.sin(angi))
    kk = (k[None, :] + 1)
    cm = torch.exp(lr[:, :, 0] * kk)
    carry = (cm * torch.cos(th[:, None] * kk), cm * torch.sin(th[:, None] * kk))
    return intra, inter, carry


def chunk_scan_c(ure, uim, K, mats):
    """Complex d_t = lam d_{t-1} + u_t with real arithmetic only (TensorRT has no complex type)."""
    (Lr, Li), (Mr, Mi), (Cr, Ci) = mats
    B, C, T = ure.shape
    nch = T // K
    ur, ui = ure.reshape(B, C, nch, K), uim.reshape(B, C, nch, K)
    yr = torch.matmul(ur, Lr.transpose(1, 2)) - torch.matmul(ui, Li.transpose(1, 2))
    yi = torch.matmul(ui, Lr.transpose(1, 2)) + torch.matmul(ur, Li.transpose(1, 2))
    er, ei = yr[..., K - 1:K], yi[..., K - 1:K]
    Sr = torch.matmul(Mr, er) - torch.matmul(Mi, ei)
    Si = torch.matmul(Mr, ei) + torch.matmul(Mi, er)
    Spr, Spi = F.pad(Sr[:, :, :-1], (0, 0, 1, 0)), F.pad(Si[:, :, :-1], (0, 0, 1, 0))
    cr, ci = Cr[:, None, :], Ci[:, None, :]
    yr = yr + Spr * cr - Spi * ci
    yi = yi + Spr * ci + Spi * cr
    return yr.reshape(B, C, T), yi.reshape(B, C, T)


class LRULayer(nn.Module):
    """Linear Recurrent Unit (Orvieto et al. 2023): h_t = lam h_{t-1} + gamma (B x_t),
    y_t = Re(C h_t) + D x_t; then GELU -> 1x1 conv (H) -> dropout, residual, pre-LayerNorm.
    lam = exp(-exp(nu_log) + i exp(theta_log)), |lam| init U[r_min, r_max], phase U[0, max_phase]."""

    def __init__(self, H, N, r_min=0.9, r_max=0.999, max_phase=math.pi / 10, p=0.1):
        super().__init__()
        u1, u2 = torch.rand(N), torch.rand(N)
        nu = -0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)
        th = max_phase * u2
        self.nu_log, self.theta_log = nn.Parameter(torch.log(nu)), nn.Parameter(torch.log(th))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - torch.exp(-nu) ** 2)))
        self.B_re = nn.Parameter(torch.randn(N, H) / math.sqrt(2 * H))
        self.B_im = nn.Parameter(torch.randn(N, H) / math.sqrt(2 * H))
        self.C_re = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.C_im = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.randn(H))
        self.mix = nn.Conv1d(H, H, 1)
        self.p = p

    def lam(self):
        return torch.exp(-torch.exp(self.nu_log)), torch.exp(self.theta_log)

    def forward(self, x, K, mats=None):    # x (B, H, T)
        z = F.layer_norm(x.transpose(1, 2), (x.shape[1],)).transpose(1, 2)
        g = torch.exp(self.gamma_log)[:, None]
        ure = F.conv1d(z, (self.B_re * g)[:, :, None])
        uim = F.conv1d(z, (self.B_im * g)[:, :, None])
        if mats is None:
            r, th = self.lam()
            mats = scan_mats_c(r, th, K, z.shape[-1] // K)
        hr, hi = chunk_scan_c(ure, uim, K, mats)
        y = F.conv1d(hr, self.C_re[:, :, None]) - F.conv1d(hi, self.C_im[:, :, None]) + self.D[:, None] * z
        y = F.dropout(self.mix(F.gelu(y)), self.p, self.training)
        return x + y


class LRUNet(nn.Module):
    def __init__(self, H=128, N=256, L=2, dropout=0.1, K=50):
        super().__init__()
        self.K, self.H = K, H
        self.enc = nn.Conv1d(N_IN, H, 1)
        self.layers = nn.ModuleList([LRULayer(H, N, p=dropout) for _ in range(L)])
        self.readout = nn.Conv1d(H, N_CLS, 1)
        self.ro_tau = nn.Parameter(torch.empty(N_CLS).uniform_(0, 4))
        self.mats = None   # set by the export wrapper (constant scan matrices)

    def forward(self, x):
        h = self.enc(x.transpose(1, 2))
        for i, layer in enumerate(self.layers):
            h = layer(h, self.K, None if self.mats is None else self.mats[i])
        a = torch.sigmoid(self.ro_tau)
        ro = chunk_scan((1 - a)[:, None] * self.readout(h), a, self.K,
                        None if self.mats is None else self.mats[-1])
        return F.softmax(ro[:, :, 11:], dim=1).sum(2)

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


class LRUExport(nn.Module):
    def __init__(self, m, K):
        super().__init__()
        import copy
        self.m = copy.deepcopy(m).eval()
        self.m.K = K
        nch = T_STEPS // K
        with torch.no_grad():
            mats = [scan_mats_c(*l.lam(), K, nch) for l in self.m.layers]
            mats.append(scan_mats(torch.sigmoid(self.m.ro_tau), K, nch))
        self.m.mats = mats   # plain tensors -> constants in the traced graph

    def forward(self, x):
        return self.m(x)


# ------------------------------------------------------------------ LUT configurations
LUT_PRI = [(2, 128, 4), (2, 128, 1), (1, 64, 8), (1, 64, 1), (2, 64, 4), (2, 256, 4), (1, 128, 4),
           (2, 128, 8), (2, 128, 2), (1, 128, 1), (2, 64, 1), (2, 256, 1), (1, 256, 4), (1, 64, 4)]
LUT_KSWEEP = {(2, 128, 4), (1, 64, 8), (2, 256, 4), (2, 128, 1)}


def lut_configs():
    """Trained (L, n, nb) x {spk, cont}; chunk K = 50 everywhere, K in {25, 50, 100} for 4 configs."""
    out = []
    for L, n, nb in LUT_PRI:
        for act in ("spk", "cont"):
            def f(n=n, nb=nb, L=L, act=act):
                return _LutWrap(DHSSM(n, nb, L, act))
            ks = ["K25", "K50", "K100"] if (L, n, nb) in LUT_KSWEEP else ["K50"]
            out.append((f"dhssm_L{L}_n{n}_b{nb}_{act}", f, ks, False))
    for h in (16, 23, 32, 64, 128, 256):
        out.append((f"gru_h{h}", lambda h=h: _LutWrap(GRUNetD(h)), ["seq"], False))
    return out


class _LutWrap(nn.Module):
    """impl 'K<k>' -> DHSSMExport with chunk K; 'seq' -> GRU full sequence (ONNX GRU op)."""

    def __init__(self, m):
        super().__init__()
        self.m = m.eval()
        self.impl = None
        if isinstance(m, DHSSM):   # build the constant scan matrices OUTSIDE the tracer
            with torch.no_grad():
                self.__dict__["_exp_cache"] = {K: DHSSMExport(m, K).eval() for K in (25, 50, 100)}

    def example_inputs(self):
        return (torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.02),)

    def input_names(self):
        return ["x"]

    def forward(self, x):
        if self.impl.startswith("K"):
            cache = self.__dict__.setdefault("_exp_cache", {})   # not a registered submodule (tracer)
            K = int(self.impl[1:])
            if K not in cache:
                cache[K] = DHSSMExport(self.m, K).eval()
            return cache[K](x)
        return self.m(x)


# ------------------------------------------------------------------ train
def evaluate(model, data, ids, bs=100):
    flat, offs, y = data
    model.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs]
            c += (model(make_batch(flat, offs, b)).argmax(1) == y[b.to(y.device)]).sum().item()
    return c / len(ids)


def train(a):
    cap_gpu_mem()
    set_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.arch in ("dhssm", "dssmU"):
        cfg = f"{a.arch}_L{a.L}_n{a.n}_b{a.nb}_{a.act}"
        model = DHSSM(a.n, a.nb, a.L, a.act, a.dropout, K=50, grouped=(a.arch == "dhssm"))
    elif a.arch == "lru":
        cfg = f"lru_L{a.L}_H{a.H}_N{a.N}_lr{a.lr:g}"
        model = LRUNet(a.H, a.N, a.L, a.dropout, K=50)
    else:
        cfg = f"{a.arch}_h{a.h}" + (f"_lr{a.lr:g}" if a.split == "speaker" else "")
        model = (GRUNetD if a.arch == "gru" else LSTMNetD)(a.h, a.dropout)
    name = f"{cfg}{a.tag}" + ("_vs" if a.split == "speaker" else "") + f"_t{a.seed}"
    od = f"{MODELS}/g2"
    os.makedirs(od, exist_ok=True)
    if os.path.exists(f"{od}/{name}.json") and not a.force:
        print(f"{name}: done already, skipping")
        return
    train_d, test_d = load_shd("train"), load_shd("test")
    val_ids = torch.tensor(speaker_val_ids() if a.split == "speaker" else json.load(open(SPLIT))["val_ids"])
    isv = torch.zeros(len(train_d[2]), dtype=torch.bool)
    isv[val_ids] = True
    tr_ids = torch.where(~isv)[0]
    te_ids = torch.arange(len(test_d[2]))
    model.cuda()
    if a.arch in ("dhssm", "dssmU", "lru"):
        slow = lambda n_: "tau" in n_ or "_log" in n_   # noqa: E731  (v0.10 fix: LRU *_log params were in both groups)
        taus = [p for n_, p in model.named_parameters() if slow(n_)]
        rest = [p for n_, p in model.named_parameters() if not slow(n_)]
        opt = torch.optim.AdamW([{"params": rest, "lr": a.lr, "weight_decay": a.wd},
                                 {"params": taus, "lr": a.lr * 0.5 if a.arch == "lru" else a.lr_tau,
                                  "weight_decay": 0.0}])   # LRU: lambda params at 0.5 x lr (Orvieto et al.)
        lossf = lambda out, y: F.cross_entropy(out, y)  # noqa: E731  (DH-SNN: CE on summed softmax)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        lossf = lambda out, y: F.cross_entropy(out, y)  # noqa: E731
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if a.sched == "cos" else \
        torch.optim.lr_scheduler.StepLR(opt, max(1, a.epochs // 5), 0.5)
    flat, offs, y = train_d
    g = torch.Generator().manual_seed(a.seed)
    os.makedirs(LOGS, exist_ok=True)
    log = open(f"{LOGS}/g2_{name}.log", "w")

    def pr(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    pr(f"{name}: params {model.effective_params()} args {vars(a)}")
    hist, best, t0 = [], (-1.0, -1), time.time()
    bs, mb = 100, 100 // a.micro
    for ep in range(a.epochs):
        model.train()
        te0 = time.time()
        perm = tr_ids[torch.randperm(len(tr_ids), generator=g)]
        tl, tc, nb_ = 0.0, 0, 0
        for i in range(0, len(perm), bs):
            ids = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(ids), mb):
                sub = ids[j:j + mb]
                out = model(make_batch(flat, offs, sub))
                loss = lossf(out, y[sub.cuda()]) * len(sub) / len(ids)
                loss.backward()
                tl += loss.item()
                tc += (out.argmax(1) == y[sub.cuda()]).sum().item()
            nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            nb_ += 1
        sched.step()
        va = evaluate(model, train_d, val_ids, bs=mb)   # eval in micro-batches too (OOM fix)
        te = evaluate(model, test_d, te_ids, bs=mb)
        hist.append({"ep": ep + 1, "train": tc / len(perm), "val": va, "test": te})
        if va > best[0]:
            best = (va, ep + 1)
            torch.save(model.state_dict(), f"{od}/{name}.pt")
        pr(f"{name} ep {ep + 1} loss {tl / nb_:.4f} train {tc / len(perm):.4f} val {va:.4f} test {te:.4f} "
           f"({time.time() - te0:.0f}s) mem {torch.cuda.max_memory_allocated() / 2**20:.0f}MB")
    bh = hist[best[1] - 1]
    meta = {"name": name, "cfg": cfg, "task": "G2", "dataset": "SHD", "arch": a.arch, "args": vars(a),
            "split": ("speaker split: val = training speakers 3 and 6 / official test" if a.split == "speaker"
                      else "T1r split: train 90% / val 10% (stratified, seed 2026) / official test"),
            "selection": "best validation epoch (earliest among ties)",
            "best_epoch": best[1], "val_acc": bh["val"], "test_acc": bh["test"], "train_acc": bh["train"],
            "last_epoch_test_acc": hist[-1]["test"], "params_effective": model.effective_params(),
            "train_minutes": round((time.time() - t0) / 60, 1), "tf32": False,
            "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2**20), "history": hist}
    save_json(f"{od}/{name}.json", meta)
    pr(json.dumps({k: v for k, v in meta.items() if k != "history"}))


# ------------------------------------------------------------------ checks vs T1
def check():
    res = {}
    torch.manual_seed(0)
    # (1) chunked scan == sequential recurrence, all K used in deployment
    u = torch.randn(2, 64, T_STEPS, dtype=torch.float64)
    lam = torch.rand(64, dtype=torch.float64) * 0.5 + 0.5
    ref = seq_scan(u, lam[None, :])
    res["chunk_scan_vs_sequential_max_abs_err_fp64"] = {
        f"K{K}": float((chunk_scan(u, lam, K) - ref).abs().max()) for K in (10, 25, 50, 100, 200)}
    u32, lam32 = u.float(), lam.float()
    res["chunk_scan_vs_sequential_max_abs_err_fp32"] = {
        f"K{K}": float((chunk_scan(u32, lam32, K) - ref).abs().max()) for K in (25, 50, 100)}
    # (2) the T1 DH-SRNN step equations with the recurrent weights zeroed and the reset removed
    #     (reference loop re-implemented from src/t1/shd.py SRNN.step) == DH-SSM chunked model.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "t1"))
    from shd import SRNN  # T1 implementation (read-only)
    t1 = SRNN(64, 8, seed=0).eval()   # T1 state tensors are float32
    t1.export_mode = True
    x = torch.zeros(2, T_STEPS, N_IN).bernoulli_(0.03)

    def t1_like_loop(m, x, rec=True, reset=True, act="spk"):
        # exact copy of SRNN.step semantics with switches
        xp = m.in_proj(x)
        B = x.shape[0]
        d = torch.zeros(B, m.n, m.branch, dtype=x.dtype)
        mem = torch.zeros(B, m.n, dtype=x.dtype)
        spk = torch.zeros(B, m.n, dtype=x.dtype)
        ro = torch.zeros(B, N_CLS, dtype=x.dtype)
        out = 0
        mems = []
        for t in range(T_STEPS):
            cur = xp[:, t] + (F.linear(spk, m.dense.weight[:, N_IN:N_IN + m.n]) if rec else 0)
            beta = torch.sigmoid(m.tau_n)
            d = beta * d + (1 - beta) * cur.view(-1, m.n, m.branch)
            I = d.sum(2)
            alpha = torch.sigmoid(m.tau_m)
            mem = mem * alpha + (1 - alpha) * I - (m.vth * spk if reset else 0)
            mems.append(mem)
            spk = (mem - m.vth > 0).to(x.dtype) if act == "spk" else F.relu(mem)
            a = torch.sigmoid(m.ro_tau)
            ro = ro * a + (1 - a) * m.readout(spk)
            if t > 10:
                out = out + F.softmax(ro, dim=1)
        return out, torch.stack(mems, -1)

    with torch.no_grad():
        o_t1 = t1(x)
        o_loop, _ = t1_like_loop(t1, x)
    res["reference_loop_vs_T1_SRNN_forward_max_abs_err"] = float((o_t1 - o_loop).abs().max())

    # map the T1 weights (dense projection with a per-neuron random branch mask, recurrence dropped)
    # into a DH-SSM whose projection is dense-masked (to test the scan, not the partition)
    class DenseMaskedSSM(nn.Module):
        def __init__(s, m, act):
            super().__init__()
            s.m, s.act = m, act

        def forward(s, x, K=50):
            m = s.m
            beta, alpha, a = torch.sigmoid(m.tau_n).reshape(-1), torch.sigmoid(m.tau_m), torch.sigmoid(m.ro_tau)
            u = (m.in_proj(x) * (1 - beta)).transpose(1, 2)                 # (B, n*branch, T), index i*branch + b
            d = chunk_scan(u, beta, K)
            I = d.reshape(x.shape[0], m.n, m.branch, -1).sum(2)
            mem = chunk_scan((1 - alpha)[:, None] * I, alpha, K)
            h = (mem - m.vth > 0).to(x.dtype) if s.act == "spk" else F.relu(mem)
            z = m.readout(h.transpose(1, 2)).transpose(1, 2)
            ro = chunk_scan((1 - a)[:, None] * z, a, K)
            return F.softmax(ro[:, :, 11:], dim=1).sum(2), mem

    with torch.no_grad():   # scale the input weights so that the neurons actually spike in this test
        t1.dense.weight.mul_(20.0)
    for act in ("spk", "cont"):
        with torch.no_grad():
            o_ref, mem_ref = t1_like_loop(t1, x, rec=False, reset=False, act=act)
            o_ssm, mem_ssm = DenseMaskedSSM(t1, act)(x)
        res[f"dhssm_chunked_vs_T1_equations_no_rec_no_reset_{act}"] = {
            "membrane_max_abs_err": float((mem_ref - mem_ssm).abs().max()),
            "spike_flips": int(((mem_ref > t1.vth) != (mem_ssm > t1.vth)).sum()),
            "spikes_total": int((mem_ref > t1.vth).sum()),
            "output_max_abs_err": float((o_ref - o_ssm).abs().max())}
    # (3) DH-SSM training graph == export graph (K = 25/50/100), fp32
    for act in ("spk", "cont"):
        mdl = DHSSM(64, 4, 2, act).eval()
        xx = torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.03)
        with torch.no_grad():
            for layer in mdl.layers:
                layer.proj.weight.mul_(20.0)
            r = mdl(xx)
            res[f"export_vs_train_graph_{act}"] = {f"K{K}": float((DHSSMExport(mdl, K)(xx) - r).abs().max())
                                                   for K in (25, 50, 100)}
    print(json.dumps(res, indent=1))
    save_json(f"{LOGS}/check_g2_vs_t1.json", res)


def check_r029():
    """LRU complex chunked scan == sequential complex recurrence; LRU / ungrouped-SSM export == train
    graph; LSTM forward; speaker split stats."""
    res = {}
    torch.manual_seed(0)
    C, T = 32, T_STEPS
    r = torch.rand(C, dtype=torch.float64) * 0.1 + 0.9
    th = torch.rand(C, dtype=torch.float64) * 0.3
    ure, uim = torch.randn(2, C, T, dtype=torch.float64), torch.randn(2, C, T, dtype=torch.float64)
    lam = torch.polar(r, th)
    u = torch.complex(ure, uim)
    h = torch.zeros(2, C, dtype=torch.complex128)
    ref = []
    for t in range(T):
        h = lam * h + u[..., t]
        ref.append(h)
    ref = torch.stack(ref, -1)
    for K in (25, 50):
        yr, yi = chunk_scan_c(ure, uim, K, scan_mats_c(r, th, K, T // K))
        res[f"lru_chunk_scan_vs_sequential_fp64_K{K}"] = float(max((yr - ref.real).abs().max(), (yi - ref.imag).abs().max()))
    x = torch.zeros(1, T, N_IN).bernoulli_(0.03)
    lru = LRUNet(64, 64, 2).eval()
    dsu = DHSSM(64, 4, 2, "cont", grouped=False).eval()
    with torch.no_grad():
        res["lru_export_vs_train_K25"] = float((LRUExport(lru, 25)(x) - lru(x)).abs().max())
        res["dssmU_export_vs_train_K25"] = float((DHSSMExport(dsu, 25)(x) - dsu(x)).abs().max())
        res["lstm_forward_shape"] = list(LSTMNetD(32)(x).shape)
    res["dssmU_proj_groups"] = dsu.layers[0].proj.g
    ids = speaker_val_ids()
    info = json.load(open(SPK_SPLIT))
    res["speaker_split"] = {k: v for k, v in info.items() if k != "val_ids"}
    print(json.dumps(res, indent=1))
    save_json(f"{LOGS}/check_g2_r029.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["dhssm", "dssmU", "gru", "lstm", "lru"], default="dhssm")
    ap.add_argument("--split", choices=["t1r", "speaker"], default="speaker", help="speaker: speaker-disjoint val")
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--nb", type=int, default=4)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--act", choices=["spk", "cont"], default="spk")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_tau", type=float, default=2e-2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sched", choices=["cos", "step"], default="cos")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--micro", type=int, default=1, help="micro-batches per batch of 100")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check_r029", action="store_true")
    a = ap.parse_args()
    if a.check_r029:
        check_r029()
    elif a.check:
        check()
    else:
        train(a)


if __name__ == "__main__":
    main()
