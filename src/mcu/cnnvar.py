"""E-10 CNN-variant extension (cnnvar, 2026-10-26; exploratory). Protocol: results/T1m/e10/PROTOCOL_cnnvar.md.

Families (ReLU after every conv; activations 0..127 uint8 after requant):
  cnnpool_c{C1}_{C2}: conv3x3 s2 p1 (1->C1) - pool2 (14->7) - conv3x3 s1 p1 (C1->C2) - pool2 floor (7->3) - fc(C2*9)
  cnns4_c{C1}_{C2}  : conv4x4 s4 p0 (1->C1, 28->7) - conv3x3 s1 p1 (C1->C2) - fc(C2*49)
  cnnds_c{C}        : conv3x3 s2 p1 (1->C)+BN - [dw3x3 s2 p1 +BN - pw1x1 +BN] - [dw3x3 s1 p1 +BN - pw1x1 +BN] - GAP - fc(C)
Training: e10.py train_eval (identical AdamW protocol); this module supplies build_net / quantize_var / ref_forward_var.
The integer reference (ref_forward_var) is bit-exact with the MODEL_CNNVAR branches of c/infer.c.

Usage (S51 = ucsim s51 for the ucsim step):
  python cnnvar.py grid [--lrs ...] [--models ...] [--jobs 8]
  python cnnvar.py select
  python cnnvar.py verify [--models ...]        # seeds 1-3 gcc bit-exact on 10 000 test images; seed-1 SDCC flash build
  python cnnvar.py ucsim  [--models ...]        # seed-1 ucsim, first 5 test images, cycles
  python cnnvar.py report                       # cnnvar/{results,matched,fronts}.csv
"""
import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CV = f"{ROOT}/results/T1m/e10/cnnvar"
os.environ.setdefault("E10_RUNS", f"{CV}/runs")

FAMILIES = {
    "pool": ["cnnpool_c2_4", "cnnpool_c4_8", "cnnpool_c8_16", "cnnpool_c16_32", "cnnpool_c32_64"],
    "ds": ["cnnds_c4", "cnnds_c8", "cnnds_c16", "cnnds_c32", "cnnds_c64"],
    "s4": ["cnns4_c2_4", "cnns4_c4_8", "cnns4_c8_16", "cnns4_c16_32", "cnns4_c32_64"],
}
MODELS = [m for f in FAMILIES.values() for m in f]
GS = 12                    # GAP requant shift
GM_MAX = (2 ** 31 - 1) // 6223
OFFS = (0, 1, 2, 3)
BN_EPS = 1e-5


def arch(name):
    fam = name.split("_")[0][3:]
    ws = [int(v) for v in name.split("_c")[1].split("_")]
    return fam, ws


# ----------------------------------------------------------------------------------------------- torch models
def build_net(name):
    import torch.nn as nn
    import torch.nn.functional as F
    fam, ws = arch(name)

    class PoolCNN(nn.Module):
        def __init__(self, c1, c2):
            super().__init__()
            self.c1 = nn.Conv2d(1, c1, 3, 2, 1)
            self.c2 = nn.Conv2d(c1, c2, 3, 1, 1)
            self.fc = nn.Linear(c2 * 9, 10)

        def forward(self, x):
            h = F.max_pool2d(F.relu(self.c1(x.reshape(-1, 1, 28, 28))), 2)
            h = F.max_pool2d(F.relu(self.c2(h)), 2)
            return self.fc(h.flatten(1))

    class S4CNN(nn.Module):
        def __init__(self, c1, c2):
            super().__init__()
            self.c1 = nn.Conv2d(1, c1, 4, 4, 0)
            self.c2 = nn.Conv2d(c1, c2, 3, 1, 1)
            self.fc = nn.Linear(c2 * 49, 10)

        def forward(self, x):
            h = F.relu(self.c1(x.reshape(-1, 1, 28, 28)))
            h = F.relu(self.c2(h))
            return self.fc(h.flatten(1))

    class DSCNN(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1 = nn.Conv2d(1, c, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(c)
            self.dw1 = nn.Conv2d(c, c, 3, 2, 1, groups=c, bias=False)
            self.bn2 = nn.BatchNorm2d(c)
            self.pw1 = nn.Conv2d(c, c, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(c)
            self.dw2 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
            self.bn4 = nn.BatchNorm2d(c)
            self.pw2 = nn.Conv2d(c, c, 1, bias=False)
            self.bn5 = nn.BatchNorm2d(c)
            self.fc = nn.Linear(c, 10)

        def forward(self, x):
            h = F.relu(self.bn1(self.c1(x.reshape(-1, 1, 28, 28))))
            h = F.relu(self.bn2(self.dw1(h)))
            h = F.relu(self.bn3(self.pw1(h)))
            h = F.relu(self.bn4(self.dw2(h)))
            h = F.relu(self.bn5(self.pw2(h)))
            return self.fc(h.mean((2, 3)))

    if fam == "pool":
        return PoolCNN(*ws)
    if fam == "s4":
        return S4CNN(*ws)
    if fam == "ds":
        return DSCNN(ws[0])
    raise ValueError(name)


# ------------------------------------------------------------------------------- float layers (BN folded) -> spec
def fold(sd, conv, bn):
    W = sd[f"{conv}.weight"]
    b = sd.get(f"{conv}.bias", np.zeros(W.shape[0]))
    if bn is None:
        return W, b
    s = sd[f"{bn}.weight"] / np.sqrt(sd[f"{bn}.running_var"] + BN_EPS)
    return W * s[:, None, None, None], (b - sd[f"{bn}.running_mean"]) * s + sd[f"{bn}.bias"]


def float_layers(name, sd):
    """list of layer dicts (type, k, s, p, pool, W, b) + head ('flat' or 'gap') + fc."""
    fam, _ = arch(name)
    if fam == "pool":
        L = [dict(t="conv", k=3, s=2, p=1, pool=True), dict(t="conv", k=3, s=1, p=1, pool=True)]
        names = [("c1", None), ("c2", None)]
        head = "flat"
    elif fam == "s4":
        L = [dict(t="conv", k=4, s=4, p=0, pool=False), dict(t="conv", k=3, s=1, p=1, pool=False)]
        names = [("c1", None), ("c2", None)]
        head = "flat"
    else:
        L = [dict(t="conv", k=3, s=2, p=1, pool=False), dict(t="dw", k=3, s=2, p=1, pool=False),
             dict(t="pw", k=1, s=1, p=0, pool=False), dict(t="dw", k=3, s=1, p=1, pool=False),
             dict(t="pw", k=1, s=1, p=0, pool=False)]
        names = [("c1", "bn1"), ("dw1", "bn2"), ("pw1", "bn3"), ("dw2", "bn4"), ("pw2", "bn5")]
        head = "gap"
    for l, (c, bn) in zip(L, names):
        l["W"], l["b"] = fold(sd, c, bn)
    return L, head, sd["fc.weight"], sd["fc.bias"]


# ------------------------------------------------------------------------------------------- integer primitives
def patches(x, k, s, p):
    """x (N, C, H, H) -> (N, C, Ho*Ho, k*k) float64 patches, (ky, kx) order."""
    N, C, H, _ = x.shape
    Ho = (H + 2 * p - k) // s + 1
    xp = np.zeros((N, C, H + 2 * p, H + 2 * p), np.float64)
    xp[:, :, p:p + H, p:p + H] = x
    out = np.empty((N, C, Ho, Ho, k, k), np.float64)
    for ky in range(k):
        for kx in range(k):
            out[..., ky, kx] = xp[:, :, ky:ky + s * (Ho - 1) + 1:s, kx:kx + s * (Ho - 1) + 1:s]
    return out.reshape(N, C, Ho * Ho, k * k), Ho


def layer_pre(l, x, Wq, bq):
    """integer pre-activation (int64), exact via float64 BLAS (all partial sums < 2^53)."""
    N, C = x.shape[:2]
    O = Wq.shape[0]
    if l["t"] == "pw":
        H = x.shape[2]
        y = np.einsum("nchw,oc->nohw", x.astype(np.float64), Wq.reshape(O, C).astype(np.float64), optimize=True)
        return (np.rint(y).astype(np.int64) + bq[None, :, None, None])
    P, Ho = patches(x, l["k"], l["s"], l["p"])
    if l["t"] == "dw":
        y = np.einsum("ncpk,ck->ncp", P, Wq.reshape(C, -1).astype(np.float64))
    else:
        y = P.transpose(0, 2, 1, 3).reshape(N, Ho * Ho, C * l["k"] ** 2) @ Wq.reshape(O, -1).T.astype(np.float64)
        y = y.transpose(0, 2, 1)
    return np.rint(y).astype(np.int64).reshape(N, O, Ho, Ho) + bq[None, :, None, None]


def shift_round(a, sh):
    return (a + ((1 << (sh - 1)) if sh > 0 else 0)) >> sh


def maxpool2(a):
    N, C, H, _ = a.shape
    h = H // 2
    return a[:, :, :2 * h, :2 * h].reshape(N, C, h, 2, h, 2).max((3, 5))


def gap_int(a, GM):
    S = a.reshape(a.shape[0], a.shape[1], -1).sum(-1)
    return np.minimum((S * GM + (1 << (GS - 1))) >> GS, 127), S


def pre_max(pre):
    return max(float(np.percentile(pre[pre > 0], 99.99)) if (pre > 0).any() else 1.0, 1.0)


def qweight(w):
    s = float(np.abs(w).max()) / 127.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int64), s


# ------------------------------------------------------------------------------------------------ quantiser
def build_q(name, sd, offs, xc):
    """offs: one offset per requant layer (+1 for GAP in the DS family)."""
    L, head, Wf, bf = float_layers(name, sd)
    a = xc.astype(np.int64).reshape(-1, 1, 28, 28)
    s_in = 1.0 / 255.0
    layers = []
    for l, o in zip(L, offs):
        Wq, sw = qweight(l["W"])
        A = s_in * sw
        bq = np.round(l["b"] / A).astype(np.int64)
        pre = layer_pre(l, a, Wq, bq)
        sh = max(int(np.ceil(np.log2(pre_max(pre) / 127.0))) - o, 0)
        a = np.clip(shift_round(pre, sh), 0, 127)
        if l["pool"]:
            a = maxpool2(a)
        s_in = A * 2 ** sh
        layers.append(dict(t=l["t"], k=l["k"], s=l["s"], p=l["p"], pool=l["pool"], Wq=Wq, bq=bq, sh=sh, s_act=s_in))
    q = dict(name=name, layers=layers, head=head, offsets=list(offs))
    if head == "gap":
        S = a.reshape(a.shape[0], a.shape[1], -1).sum(-1)
        GM = int(np.clip(round(2 ** GS * 127 * 2 ** offs[len(L)] / pre_max(S)), 1, GM_MAX))
        q["GM"] = GM
        s_in = s_in * 2 ** GS / (49 * GM)
    Wfq, sf = qweight(Wf)
    q["Wfq"], q["bfq"] = Wfq, np.round(bf / (s_in * sf)).astype(np.int64)
    q["shifts"] = [l["sh"] for l in layers]
    q["sh1"], q["sh2"] = q["shifts"][0], q["shifts"][1]
    q["sh_off"] = list(offs)
    return q


def ref_forward_var(q, x_u8, bs=1000):
    """bit-exact integer forward of c/infer.c MODEL_CNNVAR. x_u8 (N, 784) uint8 -> int64 logits (N, 10)."""
    out = []
    for i in range(0, len(x_u8), bs):
        a = x_u8[i:i + bs].astype(np.int64).reshape(-1, 1, 28, 28)
        for l in q["layers"]:
            a = np.clip(shift_round(layer_pre(l, a, l["Wq"], l["bq"]), l["sh"]), 0, 127)
            if l["pool"]:
                a = maxpool2(a)
        f = gap_int(a, q["GM"])[0] if q["head"] == "gap" else a.reshape(len(a), -1)
        y = f.astype(np.float64) @ q["Wfq"].T.astype(np.float64)
        out.append(np.rint(y).astype(np.int64) + q["bfq"])
    return np.concatenate(out)


def acc_bound(q):
    """worst-case |accumulator| per layer (int32 in C)."""
    b, amax = [], 255
    for l in q["layers"]:
        W = np.abs(l["Wq"]).reshape(l["Wq"].shape[0], -1)
        b.append(int((W.sum(1) * amax + np.abs(l["bq"])).max()))
        amax = 127
    fin = 127
    b.append(int((np.abs(q["Wfq"]).sum(1) * fin + np.abs(q["bfq"])).max()))
    return max(b)


def quantize_var(name, sd, xc, yc):
    sub = slice(0, 2000)
    fam, _ = arch(name)
    acc = lambda offs: float((ref_forward_var(build_q(name, sd, offs, xc[sub]), xc[sub]).argmax(1) == yc[sub]).mean())
    if fam in ("pool", "s4"):   # full 4x4 grid, as cnn_t1m.quantize_cnn (first strictly best in (o1, o2) order)
        best = None
        for o1 in OFFS:
            for o2 in OFFS:
                a = acc((o1, o2))
                if best is None or a > best[0]:
                    best = (a, (o1, o2))
        offs = best[1]
    else:                        # coordinate ascent over 5 conv offsets + GAP, 2 sweeps, strict improvement only
        offs = [0] * 6
        cur = acc(offs)
        for _ in range(2):
            for j in range(6):
                for o in OFFS:
                    if o == offs[j]:
                        continue
                    t = list(offs)
                    t[j] = o
                    a = acc(t)
                    if a > cur:
                        cur, offs = a, t
        offs = tuple(offs)
    q = build_q(name, sd, offs, xc)
    q["cal_acc"] = float((ref_forward_var(q, xc).argmax(1) == yc).mean())
    q["acc_bound"] = acc_bound(q)
    assert q["acc_bound"] < 2 ** 31
    return q


# ------------------------------------------------------------------------------------------------- C emission
def _arr(ctype, nm, a, per_line=24):
    a = np.asarray(a).ravel()
    body = ["  " + ",".join(str(int(v)) for v in a[i:i + per_line]) + "," for i in range(0, len(a), per_line)]
    return f"static const CODE {ctype} {nm}[{len(a)}] = {{\n" + "\n".join(body) + "\n};\n"


def _bt(b):
    return "int16_t" if (b.min() >= -32768 and b.max() <= 32767) else "int32_t"


def emit_var(q, path, tag):
    fam, ws = arch(q["name"])
    lines = [f"/* generated by src/mcu/cnnvar.py: {tag} */", "#ifndef T1M_MODEL_H", "#define T1M_MODEL_H",
             '#include "platform.h"', f'#define MODEL_NAME "{tag}"', "#define MODEL_CNNVAR 1",
             f"#define MODEL_CNN{fam.upper()} 1"]
    if fam == "ds":
        lines += [f"#define CH {ws[0]}", f"#define GAP_M {q['GM']}UL", f"#define GAP_S {GS}"]
    else:
        lines += [f"#define C1 {ws[0]}", f"#define C2 {ws[1]}"]
    for i, l in enumerate(q["layers"]):
        lines.append(f"#define SH{i + 1} {l['sh']}")
    lines.append("#ifndef MODEL_DEFINES_ONLY")
    tb = {}
    for i, l in enumerate(q["layers"]):
        lines.append(_arr("int8_t", f"w{i + 1}", l["Wq"]))
        t = _bt(l["bq"])
        lines.append(_arr(t, f"b{i + 1}", l["bq"]))
        tb[f"w{i + 1}"] = int(l["Wq"].size)
        tb[f"b{i + 1}"] = int(l["bq"].size * (2 if t == "int16_t" else 4))
    lines.append(_arr("int8_t", "wf", q["Wfq"]))
    t = _bt(q["bfq"])
    lines.append(_arr(t, "bf", q["bfq"]))
    tb["wf"] = int(q["Wfq"].size)
    tb["bf"] = int(10 * (2 if t == "int16_t" else 4))
    lines += ["#endif /* MODEL_DEFINES_ONLY */", "#endif"]
    open(path, "w").write("\n".join(lines) + "\n")
    tb["total"] = int(sum(tb.values()))
    return tb


def macs(name):
    """valid (non-padding) multiply-accumulates per image, as executed by the C kernels."""
    fam, ws = arch(name)

    def taps_s2(H):              # 3x3 s2 p1 on even H: only top row / left column hit padding
        Ho = H // 2
        n = sum(3 - (y == 0) for y in range(Ho))
        return n * n

    def taps_s1(H, rows=None):   # 3x3 s1 p1 on HxH, outputs rows/cols < rows
        r = range(rows or H)
        n = sum(3 - (y == 0) - (y == H - 1) for y in r)
        return n * n
    if fam == "pool":
        c1, c2 = ws
        return dict(conv1=c1 * taps_s2(28), conv2=c2 * c1 * taps_s1(7, 6), fc=10 * c2 * 9)
    if fam == "s4":
        c1, c2 = ws
        return dict(conv1=c1 * 49 * 16, conv2=c2 * c1 * taps_s1(7), fc=10 * c2 * 49)
    c = ws[0]
    return dict(conv1=c * taps_s2(28), dw1=c * taps_s2(14), pw1=c * c * 49, dw2=c * taps_s1(7), pw2=c * c * 49,
                gap=c * 49, fc=10 * c)


# ------------------------------------------------------------------------------------------------- commands
def e10mod():
    import e10
    assert e10.RUNS == f"{CV}/runs", e10.RUNS
    return e10


def selected(cs, m, key="val_int8"):
    return e10mod().select(cs, m, key)


def cmd_grid(a):
    e10 = e10mod()
    a.models = a.models or MODELS
    e10.cmd_grid(a)


def cmd_select(a):
    e10 = e10mod()
    cs = e10.cells(e10.load_runs())
    with open(f"{CV}/cells.csv", "w", newline="") as fh:
        keys = ["model", "lr", "wd", "n", "val_int8", "val_fp32", "test_int8", "test_int8_sd", "test_int8_sd0", "test_fp32",
                "val_int8_seeds", "test_int8_seeds"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for c in cs:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()})
    ext = []
    for m in MODELS:
        s, sf = e10.select(cs, m, "val_int8"), e10.select(cs, m, "val_fp32")
        if s is None:
            print(m, "incomplete")
            continue
        lrs = sorted({c["lr"] for c in cs if c["model"] == m})
        top = max(lrs)
        print(f"{m:16s} int8-val pick lr {s['lr']:g} wd {s['wd']:g} val {s['val_int8']:.4f} test {s['test_int8']:.4f}"
              f"±{s['test_int8_sd']:.4f} | fp32-val pick lr {sf['lr']:g} wd {sf['wd']:g} test {sf['test_int8']:.4f}"
              f"  grid lrs {lrs} {'AT TOP' if s['lr'] == top else ''}")
        if s["lr"] == 3e-3 and top == 3e-3:
            ext.append(m)
    print("needs extension to lr 1e-2:", ext)
    json.dump(ext, open(f"{CV}/extension_models.json", "w"))


def load_q(m, seed, cs, data):
    e10 = e10mod()
    s = e10.select(cs, m, "val_int8")
    tag = e10.tag_of(m, s["lr"], s["wd"], seed)
    meta = json.load(open(f"{CV}/runs/{tag}.json"))
    w = dict(np.load(f"{CV}/runs/{tag}.npz"))
    xtr, ytr = data["train"]
    vi = np.random.default_rng(seed).permutation(60000)[-6000:]
    q = quantize_var(m, w, xtr[vi], ytr[vi])
    assert list(q["offsets"]) == meta["offsets"], (tag, q["offsets"], meta["offsets"])
    return tag, meta, q


def cmd_verify(a):
    import run_t1m as R
    from common import load_fmnist_u8
    e10 = e10mod()
    data = load_fmnist_u8()
    xte, yte = data["test"]
    cs = e10.cells(e10.load_runs())
    sfx = os.environ.get("CV_SFX", "")
    vpath = f"{CV}/verify{sfx}.json"
    res = json.load(open(vpath)) if os.path.exists(vpath) else {}
    for m in (a.models or MODELS):
        for seed in (1, 2, 3):
            tag, meta, q = load_q(m, seed, cs, data)
            L = ref_forward_var(q, xte)
            acc = float((L.argmax(1) == yte).mean())
            assert abs(acc - meta["test_int8"]) < 1e-9, (tag, acc, meta["test_int8"])
            bdir = f"{CV}/build/{tag}"
            os.makedirs(bdir, exist_ok=True)
            tb = emit_var(q, f"{bdir}/model.h", tag)
            subprocess.run(["gcc", "-O2", "-Wall", "-Wno-unused-function", f"-I{bdir}", f"-I{R.C_DIR}", f"{R.C_DIR}/infer.c",
                            f"{R.C_DIR}/pc_main.c", "-o", f"{bdir}/pc_infer"], check=True)
            xte.tofile(f"{bdir}/test_x.u8")
            subprocess.run([f"{bdir}/pc_infer", f"{bdir}/test_x.u8", f"{bdir}/pc_out.bin"], check=True)
            raw = np.fromfile(f"{bdir}/pc_out.bin", np.uint8).reshape(-1, 41)
            pcl = raw[:, :40].copy().view("<i4").astype(np.int64)
            os.remove(f"{bdir}/test_x.u8")
            os.remove(f"{bdir}/pc_out.bin")
            r = dict(tag=tag, test_int8_ref=acc, pc_bitexact_logits=f"{int((pcl == L).all(1).sum())}/{len(xte)}",
                     pc_bitexact_class=f"{int((raw[:, 40] == L.argmax(1)).sum())}/{len(xte)}",
                     offsets=list(q["offsets"]), shifts=q["shifts"], acc_bound=q["acc_bound"], table_bytes=tb["total"])
            if "GM" in q:
                r["GAP_M"] = q["GM"]
            if seed == 1:
                uart = R.sdcc_build(bdir, ["uart_main"], "uart_35M", ["-DSTC8", f"-DF_OSC={R.F_OSC}UL"], R.XRAM_MAX)
                r.update(flash_bytes=uart["flash"], xdata_bytes=uart["xdata"], uart_build_ok=uart["ok"],
                         fits=bool(uart["ok"] and uart["flash"] <= R.FLASH_MAX and uart["xdata"] <= R.XRAM_MAX),
                         uart_link_log=None if uart["ok"] else uart["link_log"], tables=tb)
            res.setdefault(m, {})[str(seed)] = r
            json.dump(res, open(vpath, "w"), indent=1)
            print(m, seed, {k: v for k, v in r.items() if k != "tables"}, flush=True)


def cmd_ucsim(a):
    import run_t1m as R
    from cnn_t1m import SIM_XDATA
    from common import load_fmnist_u8
    e10 = e10mod()
    assert R.S51, "set S51"
    data = load_fmnist_u8()
    xte = data["test"][0]
    cs = e10.cells(e10.load_runs())
    sfx = os.environ.get("CV_SFX", "")
    path = f"{CV}/ucsim{sfx}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    n = a.n
    for m in (a.models or MODELS):
        tag, meta, q = load_q(m, 1, cs, data)
        bdir = f"{CV}/build/{tag}"
        assert os.path.exists(f"{bdir}/model.h"), f"run verify first: {tag}"
        L = ref_forward_var(q, xte[:n])
        sim = R.sdcc_build(bdir, ["sim_main"], "sim", [], SIM_XDATA)
        assert sim["ok"], sim["link_log"]
        cyc, n_ok, n_cls, inst = [], 0, 0, []
        for i in range(n):
            c, lg, cy, cy0, s_ = R.ucsim_run(bdir, xte[i])
            n_ok += int((lg == L[i]).all())
            n_cls += int(c == L[i].argmax())
            cyc.append(cy - cy0)
            inst.append(s_["inst"])
        r = dict(tag=tag, n=n, cycles=float(np.mean(cyc)), cycles_min=int(min(cyc)), cycles_max=int(max(cyc)),
                 instr=float(np.mean(inst)), sim_flash=sim["flash"], sim_xdata=sim["xdata"],
                 ucsim_bitexact_logits=f"{n_ok}/{n}", ucsim_bitexact_class=f"{n_cls}/{n}",
                 macs=sum(macs(m).values()))
        res[m] = r
        json.dump(res, open(path, "w"), indent=1)
        print(m, r, flush=True)


# ------------------------------------------------------------------------------------------------- report
def cmd_report(a):
    import cnnvar_report
    cnnvar_report.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "select", "verify", "ucsim", "report"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(f"{CV}/runs", exist_ok=True)
    {"grid": cmd_grid, "select": cmd_select, "verify": cmd_verify, "ucsim": cmd_ucsim, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
