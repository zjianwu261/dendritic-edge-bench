"""T1m step 2: int8 post-training quantization + bit-exact fixed-point reference.

Integer scheme (identical to src/mcu/c/infer.c; see `ref_forward`):
  input      x_q = raw pixel uint8 (scale 1/255, exact)
  weights    per-layer symmetric int8: w_q = clip(round(w / s_w), -127, 127), s_w = max|w| / 127
  bias       b_q = round(b / (s_in * s_w)), stored int16 if it fits, else int32
  MAC        int8/uint8 x int8 -> int16 product, accumulated in int32
  requant    t = (acc + 2^(sh-1)) >> sh   (arithmetic shift = floor; power-of-two only, no multiply)
  LeakyReLU  t >= 0: min(t, 127)
             t <  0: max(((max(t, -1270) * 13 + 64) >> 7), -128)   (slope 13/128 = 0.1016)
  output     logits = int32 accumulators of the last layer, class = first argmax
  activation scale of layer l = s_in * s_w * 2^sh  (feeds the next layer's bias scale)
Shifts are chosen by a small grid search on the training-set validation split (the 10 % held out
in train.py), never on the test set.
"""
import numpy as np

LEAK_NUM, LEAK_SH, NEG_CLAMP = 13, 7, -1270


def act_int(t):
    t = np.asarray(t, dtype=np.int64)
    pos = np.minimum(t, 127)
    neg = (np.maximum(t, NEG_CLAMP) * LEAK_NUM + (1 << (LEAK_SH - 1))) >> LEAK_SH
    neg = np.maximum(neg, -128)
    return np.where(t >= 0, pos, neg).astype(np.int64)


def shift_round(acc, sh):
    return (acc + ((1 << (sh - 1)) if sh > 0 else 0)) >> sh


def qweight(w):
    s = float(np.abs(w).max()) / 127.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int64), s


def ref_forward(q, x_u8, return_hidden=False):
    """Bit-exact integer forward pass. x_u8: (N, 784) uint8. Returns int64 logits (N, 10)."""
    x = x_u8.astype(np.int64)
    a1 = act_int(shift_round(x @ q["W1q"].T + q["b1q"], q["sh1"]))
    a2 = act_int(shift_round(a1 @ q["W2q"].T + q["b2q"], q["sh2"]))
    logits = a2 @ q["W3q"].T + q["b3q"]
    if return_hidden:
        return logits, a1, a2
    return logits


def _pre_max(pre):
    return max(float(np.percentile(pre[pre > 0], 99.99)) if (pre > 0).any() else 1.0, 1.0)


def build(m, sh_off1, sh_off2, xc):
    """Quantize float model `m` (npz dict) with shifts = max-rule shift - offset."""
    W1q, s1 = qweight(m["W1"])
    W2q, s2 = qweight(m["W2"])
    W3q, s3 = qweight(m["W3"])
    A1 = s1 / 255.0
    b1q = np.round(m["b1"] / A1).astype(np.int64)
    x = xc.astype(np.int64)
    pre1 = x @ W1q.T + b1q
    sh1 = max(int(np.ceil(np.log2(_pre_max(pre1) / 127.0))) - sh_off1, 0)
    sa1 = A1 * 2 ** sh1
    A2 = sa1 * s2
    b2q = np.round(m["b2"] / A2).astype(np.int64)
    a1 = act_int(shift_round(pre1, sh1))
    pre2 = a1 @ W2q.T + b2q
    sh2 = max(int(np.ceil(np.log2(_pre_max(pre2) / 127.0))) - sh_off2, 0)
    sa2 = A2 * 2 ** sh2
    A3 = sa2 * s3
    b3q = np.round(m["b3"] / A3).astype(np.int64)
    q = dict(W1q=W1q, b1q=b1q, W2q=W2q, b2q=b2q, W3q=W3q, b3q=b3q, sh1=sh1, sh2=sh2,
             scales=dict(s_w1=s1, s_w2=s2, s_w3=s3, s_a1=sa1, s_a2=sa2, s_logit=A3),
             sh_off=(sh_off1, sh_off2))
    if "M1" in m:  # masked weights must stay exactly zero after rounding
        assert (W1q[m["M1"].T == 0] == 0).all() and (W2q[m["M2"].T == 0] == 0).all()
    return q


def check_int32(q):
    """Worst-case |accumulator| bound, must fit int32 (C uses int32_t accumulators)."""
    b1 = np.abs(q["W1q"]).sum(1) * 255 + np.abs(q["b1q"])
    b2 = np.abs(q["W2q"]).sum(1) * 128 + np.abs(q["b2q"])
    b3 = np.abs(q["W3q"]).sum(1) * 128 + np.abs(q["b3q"])
    worst = int(max(b1.max(), b2.max(), b3.max()))
    assert worst < 2 ** 31, worst
    return worst


def quantize(m, x_cal, y_cal, offsets=(0, 1, 2, 3)):
    """Grid-search the two requant shifts on the calibration (validation) split."""
    best = None
    for o1 in offsets:
        for o2 in offsets:
            q = build(m, o1, o2, x_cal)
            acc = float((ref_forward(q, x_cal).argmax(1) == y_cal).mean())
            if best is None or acc > best[0]:
                best = (acc, q)
    q = best[1]
    q["cal_acc"] = best[0]
    q["acc_bound"] = check_int32(q)
    return q


def float_forward(m, x_u8):
    x = x_u8.astype(np.float64) / 255.0
    lk = lambda v: np.where(v >= 0, v, 0.1 * v)
    h = lk(x @ m["W1"].T.astype(np.float64) + m["b1"])
    h = lk(h @ m["W2"].T.astype(np.float64) + m["b2"])
    return h @ m["W3"].T.astype(np.float64) + m["b3"]
