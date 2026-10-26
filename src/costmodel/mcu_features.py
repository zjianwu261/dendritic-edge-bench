"""E-11 (PROTOCOL.md P3): 8051 term counts per inference.

  F_fc     dense fully connected MACs (MLP layers, dANN dendrite->soma layer, output layers, CNN fc)
  F_conv   convolution MACs actually executed (padding taps are skipped by the C kernels)
  F_gather MACs of the stored-mask (index-table) dendrite layer: nonzero weights only
  F_prng   MACs of the PRNG dendrite layer (positions regenerated on chip)
  N_out    requantised outputs (each call of act() / relu8(); int32 logits are not requantised)

Two independent counts, asserted equal:
  1. python_counts(): instrumented walk of the Python int8 references (src/mcu/quant.py ref_forward
     for MLP / dANN, src/mcu/cnn_t1m.py im2col for plain CNNs, src/mcu/cnnvar.py patches for s4);
     conv MACs are counted from the reference's own patch extraction applied to an all-ones input,
     so only in-image (non-padding) taps are counted, as in the C kernels.
  2. c_counts(): a copy of src/mcu/c/infer.c (unchanged sources are copied to a temporary
     directory, never edited in place) is patched to increment one counter per executed MAC of
     each class and per requantised output, compiled with gcc together with the model.h of the
     board build, and run on one image.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MCU = f"{ROOT}/src/mcu"
C_DIR = f"{MCU}/c"
SCRATCH = os.environ.get("COSTMODEL_TMP", tempfile.gettempdir())
TERMS = ("F_fc", "F_conv", "F_gather", "F_prng", "N_out")


def parse_model_h(path):
    txt = open(path).read()
    defs = {}
    for k, v in re.findall(r"#define\s+(\w+)\s+([^\s/]+)", txt):
        defs[k] = v
    arrays = {nm: int(n) for nm, n in re.findall(r"static const CODE \w+ (\w+)\[(\d+)\]", txt)}
    if "MODEL_CNNS4" in defs:
        kind = "cnns4"
    elif "MODEL_CNNVAR" in defs:
        kind = "cnnvar_other"
    elif "MODEL_CNN" in defs:
        kind = "cnn"
    elif "MODEL_DANN_PRNG" in defs:
        kind = "prng"
    elif "MODEL_DANN" in defs:
        kind = "dann"
    elif "MODEL_MLP" in defs:
        kind = "mlp"
    else:
        raise ValueError(path)
    ints = {k: int(v) for k, v in defs.items() if re.fullmatch(r"\d+", v)}
    return kind, ints, arrays


def python_counts(kind, d):
    """Instrumented reference: counts per class, one image."""
    sys.path.insert(0, MCU)
    c = dict.fromkeys(TERMS, 0)
    if kind == "mlp":
        H1, H2 = d["H1"], d["H2"]
        # ref_forward: x @ W1q.T (784 x H1), a1 @ W2q.T (H1 x H2), a2 @ W3q.T (H2 x 10)
        c["F_fc"] = 784 * H1 + H1 * H2 + H2 * 10
        c["N_out"] = H1 + H2                      # act() on a1 and a2
    elif kind in ("dann", "prng"):
        D, S, SYN = d["D"], d["H2"], d["SYN"]
        H1 = d["H1"]
        assert H1 == D * S
        # ref_forward with the masked W1q: only the SYN nonzero (mask) positions per dendrite are MACs
        # of the sparse layer; W2q is block-structured (one weight per dendrite) -> H1 direct MACs.
        m1 = H1 * SYN
        c["F_gather" if kind == "dann" else "F_prng"] = m1
        c["F_fc"] = H1 + S * 10
        c["N_out"] = H1 + S
    elif kind == "cnn":
        import cnn_t1m
        C1, C2 = d["C1"], d["C2"]
        ones1 = np.ones((1, 1, 28, 28), np.int64)
        taps1 = int(cnn_t1m.im2col(ones1, 28).sum())          # non-padding taps of one output channel
        ones2 = np.ones((1, C1, 14, 14), np.int64)
        taps2 = int(cnn_t1m.im2col(ones2, 14).sum())
        c["F_conv"] = C1 * taps1 + C2 * taps2
        c["F_fc"] = 10 * C2 * 49
        c["N_out"] = C1 * 196 + C2 * 49
    elif kind == "cnns4":
        import cnnvar
        C1, C2 = d["C1"], d["C2"]
        P1, Ho1 = cnnvar.patches(np.ones((1, 1, 28, 28)), 4, 4, 0)
        P2, Ho2 = cnnvar.patches(np.ones((1, C1, 7, 7)), 3, 1, 1)
        c["F_conv"] = int(C1 * P1.sum() + C2 * P2.sum())
        c["F_fc"] = 10 * C2 * 49
        c["N_out"] = C1 * Ho1 * Ho1 + C2 * Ho2 * Ho2
    else:
        raise NotImplementedError(kind)
    return c


# ------------------------------------------------------------------ instrumented C kernel (gcc, on a copy)
_PATCHES = [
    # (old, new, expected number of occurrences, or None for >= 1)
    ("#define T(o, k) p = px[o] * pw[k]; acc += p",
     "#define T(o, k) p = px[o] * pw[k]; acc += p; cm_conv++", 1),
    ("        p = *px++ * *pw++;\n        a += p;",
     "        p = *px++ * *pw++;\n        a += p; cm_fc++;", 2),                     # dot_s8, dot_u8
    ("                p = px[*po++] * *pw++;\n                acc += p;",
     "                p = px[*po++] * *pw++;\n                acc += p; cm_gather++;", 1),
    ("                p = px[rf_off[t]] * *pw++;\n                acc += p;",
     "                p = px[rf_off[t]] * *pw++;\n                acc += p; cm_prng++;", 1),
    ("            acc2 += p;", "            acc2 += p; cm_fc++;", 2),               # dANN, PRNG soma
    ("                    logits[c] += p;", "                    logits[c] += p; cm_fc++;", 1),
    ("static int8_t act(int32_t t)\n{", "static int8_t act(int32_t t)\n{\n    cm_out++;", 1),
    ("static uint8_t relu8(int32_t t)\n{", "static uint8_t relu8(int32_t t)\n{\n    cm_out++;", 1),
]
_COUNTERS = "long cm_fc, cm_conv, cm_gather, cm_prng, cm_out;\n"
_MAIN = r'''
#include <stdio.h>
#include <stdlib.h>
#include "infer.h"
extern long cm_fc, cm_conv, cm_gather, cm_prng, cm_out;
int main(void)
{
    uint8_t img[N_IN];
    int i;
    srand(1);
    for (i = 0; i < N_IN; i++) img[i] = (uint8_t)(rand() & 0xFF);
    infer(img);
    printf("%ld %ld %ld %ld %ld\n", cm_fc, cm_conv, cm_gather, cm_prng, cm_out);
    return 0;
}
'''


def patched_infer_source():
    src = open(f"{C_DIR}/infer.c").read()
    for old, new, n in _PATCHES:
        k = src.count(old)
        if k != n:
            raise AssertionError(f"patch anchor found {k}x (expected {n}): {old[:60]!r}")
        src = src.replace(old, new)
    src = src.replace('#include "model.h"\n', '#include "model.h"\n' + _COUNTERS, 1)
    return src


def c_counts(model_h):
    tmp = tempfile.mkdtemp(prefix="cm_c_", dir=SCRATCH)
    try:
        for f in ("infer.h", "platform.h", "prng_rf.c", "prng_rf.h", "infer_cnnvar.inc",
                  "infer_cnnvar_defs.inc"):
            shutil.copy(f"{C_DIR}/{f}", tmp)
        shutil.copy(model_h, f"{tmp}/model.h")
        open(f"{tmp}/infer_count.c", "w").write(patched_infer_source())
        open(f"{tmp}/count_main.c", "w").write(_MAIN)
        kind, _, _ = parse_model_h(model_h)
        srcs = [f"{tmp}/infer_count.c", f"{tmp}/count_main.c"]
        if kind == "prng":
            srcs.append(f"{tmp}/prng_rf.c")
        subprocess.run(["gcc", "-O0", "-w", f"-I{tmp}", *srcs, "-o", f"{tmp}/cnt"], check=True)
        out = subprocess.run([f"{tmp}/cnt"], check=True, capture_output=True, text=True).stdout.split()
        fc, conv, g, p, o = (int(v) for v in out)
        return dict(F_fc=fc, F_conv=conv, F_gather=g, F_prng=p, N_out=o)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def effective_params(kind, arrays):
    """Nonzero-weight parameters as stored (index tables off1/base1 excluded; dANN w1 holds only the
    nonzero synapses). Counted as array elements (biases are elements regardless of their C type)."""
    names = [n for n in arrays if n not in ("off1", "base1")]
    return int(sum(arrays[n] for n in names))


def mcu_features(model_h):
    kind, d, arrays = parse_model_h(model_h)
    py = python_counts(kind, d)
    cc = c_counts(model_h)
    if py != cc:
        raise AssertionError(f"{model_h}: python {py} != C {cc}")
    f = dict(py)
    f["kind"] = kind
    f["params_eff"] = effective_params(kind, arrays)
    f["F_eff"] = f["F_fc"] + f["F_conv"] + f["F_gather"] + f["F_prng"]
    return f


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, mcu_features(p))
