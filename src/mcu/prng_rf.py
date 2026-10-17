"""T1m step 5: pseudo-random local receptive fields regenerated on chip (no index tables).

Bit-exact Python reference of src/mcu/c/prng_rf.c. The same generator produces the training
masks (train.py --mask prng) and, at inference time on the 8051, the synapse positions.

Generator: 16-bit xorshift (Marsaglia triple 7/9/8, period 2^16 - 1)
    x ^= x << 7; x ^= x >> 9; x ^= x << 8          (all mod 2^16)

Per dendrite n (0-based, in storage order: soma j owns dendrites j*D .. j*D+D-1):
    s   = (RF_K * (n + 1) + RF_C) mod 2^16;  if s == 0: s = 1      (seed = dendrite index)
    r   = xs(s); lo(r) += hi(hi(r) * hi(r)) (mod 256); r = xs(r)
                 seed mixing: one Feistel round (8x8 MUL AB), bijective. Without a non-linear
                 step the linearly spaced seeds give correlated neighbouring dendrites; a
                 non-bijective mix (e.g. r += lo*hi) produced ~7 % duplicate dendrites.
    r   = xs(r);  row = (hi(r) * 28) >> 8;  col = (lo(r) * 28) >> 8       (0..27 each)
    row = clamp(row, 2, 25); col = clamp(col, 2, 25)                   (centre of the RF)
    base = (row - 2) * 28 + (col - 2)        top-left pixel of the 5x5 window
    synapses 0..8  : the full 3x3 neighbourhood of the centre, fixed order
                     (window offsets 29,30,31,57,58,59,85,86,87)
    synapses 9..15 : 7 distinct pixels of the Chebyshev radius-2 ring (16 pixels, table RING),
                     drawn without replacement: r = xs(r), nibbles in the order
                     lo&15, lo>>4, hi&15, hi>>4; a nibble already used is skipped.
    pixel(t) = base + off[t]

RF_K = 0x9E37 (odd), RF_C = (0x3C6F * model_seed + 0x1D2B) mod 2^16.

Relation to the official `opt.make_masks` (rfs_type="dendritic", rfs_mode="random", 16 syn):
  official: centre uniform over all 28x28 pixels (numpy PCG64); synapses = full 3x3 around the
            centre + 7 random pixels of the radius-2 ring (without replacement). At the image
            border the windows are clipped, so the RF is filled from the radius-3 ring or further.
  here:     identical for centres in rows/cols 2..25 (73 % of the official centres);
            border centres are clamped inward to 2 / 25 instead of growing the ring, so every RF
            is a 3x3 + 7-of-16 pattern inside a full 5x5 window. The centre distribution is
            ~uniform (each row/col gets 9 or 10 of the 256 byte values) before clamping.
            Different random stream (xorshift16 vs PCG64), so masks differ from the official
            ones for the same seed.
"""
import numpy as np

W = 28
RF_K = 0x9E37
CORE = [29, 30, 31, 57, 58, 59, 85, 86, 87]
RING = [0, 1, 2, 3, 4, 28, 32, 56, 60, 84, 88, 112, 113, 114, 115, 116]
N_RING_PICK = 7
SYN = 16


def rf_c(model_seed):
    return (0x3C6F * int(model_seed) + 0x1D2B) & 0xFFFF


def xs16(x):
    x ^= (x << 7) & 0xFFFF
    x ^= x >> 9
    x ^= (x << 8) & 0xFFFF
    return x


def seed_of(n, C):
    s = (RF_K * (n + 1) + C) & 0xFFFF
    return s if s else 1


def rf_gen(s):
    """One dendrite from its seed state. Returns (base, offsets[16]) exactly like prng_rf.c."""
    r = xs16(s if s else 1)
    hi = r >> 8
    r = xs16((hi << 8) | (((r & 0xFF) + ((hi * hi) >> 8)) & 0xFF))
    r = xs16(r)
    row = ((r >> 8) * W) >> 8
    col = ((r & 0xFF) * W) >> 8
    row = min(max(row, 2), 25)
    col = min(max(col, 2), 25)
    base = (row - 2) * W + (col - 2)
    off = list(CORE)
    used = 0
    while len(off) < SYN:
        r = xs16(r)
        for nib in (r & 15, (r >> 4) & 15, (r >> 8) & 15, r >> 12):
            if len(off) < SYN and not (used >> nib) & 1:
                used |= 1 << nib
                off.append(RING[nib])
    return base, off


def rf_indices(h1, model_seed):
    """(h1, 16) pixel indices in synapse (storage) order + (h1,) bases."""
    C = rf_c(model_seed)
    idx = np.zeros((h1, SYN), np.int64)
    base = np.zeros(h1, np.int64)
    for n in range(h1):
        b, off = rf_gen(seed_of(n, C))
        base[n] = b
        idx[n] = b + np.array(off)
    assert (idx >= 0).all() and (idx < W * W).all()
    assert all(len(set(r)) == SYN for r in idx.tolist())
    return idx, base


def prng_masks(d, s, model_seed):
    """Same layout as common.dann_masks: M1 (784, d*s), M2 (d*s, s) (structured)."""
    h1 = d * s
    idx, _ = rf_indices(h1, model_seed)
    m1 = np.zeros((W * W, h1), np.int8)
    m1[idx.T, np.arange(h1)[None, :]] = 1
    assert (m1.sum(0) == SYN).all()
    m2 = np.zeros((h1, s), np.int8)
    for j in range(s):
        m2[j * d:(j + 1) * d, j] = 1
    return m1, m2


if __name__ == "__main__":
    # quick statistics vs the official masks
    import sys
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    for seed in (1, 2, 3):
        C = rf_c(seed)
        zero = [n for n in range(65536) if (RF_K * (n + 1) + C) & 0xFFFF == 0][0]
        print(f"seed {seed}: C=0x{C:04x}, first dendrite index with raw seed 0: {zero}")
    idx, base = rf_indices(4096, 1)
    rows = base // W + 2
    cols = base % W + 2
    print("centre row histogram", np.bincount(rows, minlength=28).tolist())
    print("centre col histogram", np.bincount(cols, minlength=28).tolist())
    print("pixel coverage (4096 dendrites):", len(np.unique(idx)), "of 784")
    print("first 3 dendrites:", [rf_gen(seed_of(n, rf_c(1))) for n in range(3)])
