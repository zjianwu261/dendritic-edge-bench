"""Runs ON Orin #1: NoMaD explore pipeline with the TRT FP16 engines on real inputs and fixed noise
Input npz (from fp16_check.py prep): obs [N,12,96,96], goal [N,3,96,96], z0 [N,n,8,2],
z [N,10,n,8,2]. Same DDPM update as trt_pipeline_bench.py (numpy, float32). Output npz: cond [N,256],
actions [N,n,8,2] (normalized deltas -> unnormalized cumulative waypoints).
Usage: python3 fp16_check_orin.py enc.engine unet_b8.engine in.npz out.npz
"""
import sys
import warnings

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from trt_pipeline_bench import Eng, alphas_cumprod  # noqa: E402
from trt_sync_bench import Cuda  # noqa: E402

AMIN, AMAX = np.array([-2.5, -4.0], np.float32), np.array([5.0, 4.0], np.float32)


def main():
    enc_p, unet_p, inp, outp = sys.argv[1:5]
    warnings.simplefilter("ignore")
    import tensorrt as trt
    lg = trt.Logger(trt.Logger.WARNING); trt.init_libnvinfer_plugins(lg, "")
    cu = Cuda(); rt = trt.Runtime(lg); st = cu.stream()
    enc, unet = Eng(enc_p, cu, trt, rt, st), Eng(unet_p, cu, trt, rt, st)
    d = np.load(inp)
    N, n = d["obs"].shape[0], d["z0"].shape[1]
    ac = alphas_cumprod(10); coefs = []
    for t in range(9, -1, -1):
        ap_, app = ac[t], (ac[t - 1] if t > 0 else 1.0)
        bp, bpp = 1 - ap_, 1 - app; ca = ap_ / app; cb = 1 - ca
        coefs.append((t, np.sqrt(ap_), np.sqrt(bp), np.sqrt(app) * cb / bp, np.sqrt(ca) * bpp / bp,
                      np.sqrt(max(bpp / bp * cb, 1e-20)) if t > 0 else 0.0))
    conds, acts = [], []
    for i in range(N):
        enc.host["obs"][...] = d["obs"][i:i + 1]; enc.host["goal"][...] = d["goal"][i:i + 1]
        enc.run(); c = enc.host["cond"].copy(); conds.append(c[0])
        unet.host["cond"][...] = np.repeat(c, n, 0)
        x = d["z0"][i].astype(np.float32)
        for k, (t, sa, sb, cx0, cxt, std) in enumerate(coefs):
            unet.host["sample"][...] = x; unet.host["t"][...] = t
            unet.run(); eps = unet.host["eps"].astype(np.float32)
            x0 = np.clip((x - sb * eps) / sa, -1, 1)
            x = (cx0 * x0 + cxt * x).astype(np.float32)
            if t > 0:
                x = x + std * d["z"][i, k]
        acts.append(np.cumsum((x + 1) / 2 * (AMAX - AMIN) + AMIN, axis=1))
    np.savez(outp, cond=np.array(conds), actions=np.array(acts), engines=np.array([enc_p, unet_p]))
    print("done", N)


if __name__ == "__main__":
    main()
