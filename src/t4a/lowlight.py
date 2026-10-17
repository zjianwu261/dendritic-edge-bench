"""Synthetic low light for the T4a benchmark (lighting as a continual-learning / robustness axis).

Re-implementation of the physically based model in eval/lownoise.py of llie-robot-bench
(https://github.com/zjianwu261/llie-robot-bench, commit d44e383, MIT licence, same authors).
Nothing is imported from that repository.

dark = sRGB( PoissonGaussian( e * linear(img) ) ), linear(x) = x^2.2, 8-bit quantized
  e    ~ log-uniform[0.002, 0.1]   (exposure; calibrated there on LOLv2-real)
  shot ~ log-uniform[3e-6, 5e-3],  read = exp(2.18 log(shot) + 1.2 + N(0, 0.26))  (Brooks et al. CVPR'19)
T4a choice: one (e, shot, read) per *trajectory* (a camera in a dark place keeps its exposure),
independent noise per frame; applied at the stored image resolution, before the 96x96 crop/resize.
"""
import numpy as np

E_MIN, E_MAX = 0.002, 0.1
S_MIN, S_MAX = 3e-6, 5e-3


def sample_params(rng):
    logu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    e, shot = logu(E_MIN, E_MAX), logu(S_MIN, S_MAX)
    read = float(np.exp(2.18 * np.log(shot) + 1.2 + 0.26 * rng.standard_normal()))
    return e, shot, read


def darken(img_u8, params, rng):
    """img_u8: HxWx3 uint8 sRGB -> uint8 dark image."""
    e, shot, read = params
    sig = (img_u8.astype(np.float32) / 255.0) ** 2.2 * e
    var = shot * sig + read
    x = np.clip(sig + np.sqrt(var) * rng.standard_normal(img_u8.shape).astype(np.float32), 0, 1) ** (1 / 2.2)
    return np.round(x * 255).astype(np.uint8)
