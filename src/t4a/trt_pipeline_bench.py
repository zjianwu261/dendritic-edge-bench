"""NoMaD full action generation on Orin with SEPARATE engines, the way the official explore.py runs it:
    encoder (1x) -> repeat cond to N samples -> 10 x [UNet engine -> DDPM step on the GPU-side host copy]
The DDPM update runs on the host in numpy (official code runs it in torch on GPU; it is ~10 tiny
elementwise ops on an [N,8,2] tensor).  Each timed iteration is synchronous end to end:
    H2D(obs, goal) -> enc -> D2H(cond) -> [H2D(x, t, cond) -> unet -> D2H(eps) -> numpy step] x 10
Reported: total p50/p95, plus per-stage medians (encoder e2e, per-UNet-step e2e, host step).
Runs on Orin #1 (python3.8, TRT 8.5) next to trt_sync_bench.py (imports its Cuda helper).
Usage: python3 trt_pipeline_bench.py enc.engine unet.engine --n 8 --label X --run 1 --out res.jsonl
"""
import argparse
import ctypes
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_sync_bench import Cuda, mem_available_mb, vm_hwm_mb, pct, read, GPU_FREQ  # noqa: E402

# squaredcos_cap_v2 betas for T=10 (diffusers betas_for_alpha_bar), computed here to avoid torch
def alphas_cumprod(T=10):
    import math
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    betas = [min(1 - f((i + 1) / T) / f(i / T), 0.999) for i in range(T)]
    return np.cumprod(1 - np.array(betas, dtype=np.float32))


class Eng:
    def __init__(self, path, cu, trt, runtime, stream):
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.cu, self.stream, self.bind, self.host, self.dev, self.nbytes = cu, stream, [], {}, {}, {}
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dt = np.dtype({trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32}[
                self.engine.get_binding_dtype(i)])
            n = int(np.prod(shape)) * dt.itemsize
            d, h = cu.malloc(n), cu.malloc_host(n)
            self.host[name] = np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(h.value)).view(dt).reshape(shape)
            self.dev[name], self.nbytes[name] = (d, h), n
            self.bind.append(d.value)
            (self.inputs if self.engine.binding_is_input(i) else self.outputs).append(name)

    def run(self):
        lib = self.cu.lib
        for k in self.inputs:
            d, h = self.dev[k]; lib.cudaMemcpyAsync(d, h, self.nbytes[k], 1, self.stream)
        self.ctx.execute_async_v2(self.bind, self.stream.value)
        for k in self.outputs:
            d, h = self.dev[k]; lib.cudaMemcpyAsync(h, d, self.nbytes[k], 2, self.stream)
        self.cu.check(lib.cudaStreamSynchronize(self.stream), "sync")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("enc"); ap.add_argument("unet")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--enc2", default="", help="optional second encoder stage (split encoder: encA tokens -> encB cond)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup_s", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--label", default=""); ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--build", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    avail0 = mem_available_mb()
    warnings.simplefilter("ignore")
    import tensorrt as trt
    lg = trt.Logger(trt.Logger.WARNING); trt.init_libnvinfer_plugins(lg, "")
    cu = Cuda(); runtime = trt.Runtime(lg); stream = cu.stream()
    enc, unet = Eng(a.enc, cu, trt, runtime, stream), Eng(a.unet, cu, trt, runtime, stream)
    enc2 = Eng(a.enc2, cu, trt, runtime, stream) if a.enc2 else None
    avail_alloc = mem_available_mb()
    rng = np.random.RandomState(0)
    enc.host["obs"][...] = rng.randn(*enc.host["obs"].shape); enc.host["goal"][...] = rng.randn(*enc.host["goal"].shape)
    ac = alphas_cumprod(a.steps)
    coefs = []
    for t in range(a.steps - 1, -1, -1):
        ap_, app = ac[t], (ac[t - 1] if t > 0 else 1.0)
        bp, bpp = 1 - ap_, 1 - app; ca = ap_ / app; cb = 1 - ca
        coefs.append((t, np.sqrt(ap_), np.sqrt(bp), np.sqrt(app) * cb / bp, np.sqrt(ca) * bpp / bp,
                      np.sqrt(max(bpp / bp * cb, 1e-20)) if t > 0 else 0.0))
    st = {"enc": [], "unet": [], "host": []}

    def once(record):
        t0 = time.perf_counter(); enc.run()
        if enc2 is not None:
            enc2.host["tokens"][...] = enc.host["tokens"]; enc2.run(); cond = enc2.host["cond"]
        else:
            cond = enc.host["cond"]
        t1 = time.perf_counter()
        unet.host["cond"][...] = np.repeat(cond, a.n, 0)
        x = rng.randn(a.n, 8, 2).astype(np.float32)
        tu = th = 0.0
        for (t, sa, sb, cx0, cxt, std) in coefs:
            unet.host["sample"][...] = x; unet.host["t"][...] = t
            s0 = time.perf_counter(); unet.run(); s1 = time.perf_counter()
            eps = unet.host["eps"]
            x0 = np.clip((x - sb * eps) / sa, -1, 1)
            x = cx0 * x0 + cxt * x
            if t > 0:
                x = x + std * rng.randn(*x.shape).astype(np.float32)
            tu += s1 - s0; th += time.perf_counter() - s1
        if record:
            st["enc"].append((t1 - t0) * 1e3); st["unet"].append(tu * 1e3 / a.steps); st["host"].append(th * 1e3 / a.steps)
        return (time.perf_counter() - t0) * 1e3

    tw = time.perf_counter(); nw = 0
    while time.perf_counter() - tw < a.warmup_s or nw < 10:
        once(False); nw += 1
    fw = read(GPU_FREQ)
    ts = np.array([once(True) for _ in range(a.iters)])
    res = {"label": a.label, "run": a.run, "build": a.build, "n_samples": a.n, "steps": a.steps,
           "timing_scope": "sync_e2e_pinned_pipeline", "warmup": nw, "iters": a.iters,
           "total_p50_ms": pct(ts, 50), "total_p95_ms": pct(ts, 95), "hz_p50": round(1000 / float(np.median(ts)), 2),
           "enc_p50_ms": pct(st["enc"], 50), "unet_step_p50_ms": pct(st["unet"], 50), "host_step_p50_ms": pct(st["host"], 50),
           "mem_avail_drop_MB": round(avail0 - min(avail_alloc, mem_available_mb()), 1), "vm_hwm_MB": round(vm_hwm_mb(), 1),
           "engine_device_mem_MB": round((enc.engine.device_memory_size + unet.engine.device_memory_size
                                           + (enc2.engine.device_memory_size if enc2 else 0)) / 2**20, 2),
           "enc_engines": [os.path.basename(a.enc)] + ([os.path.basename(a.enc2)] if a.enc2 else []),
           "gpu_freq_after_warmup": fw, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
