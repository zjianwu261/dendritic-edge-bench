"""Duration-based inference loop for energy measurement (T1 energy protocol).

Same synchronous batch=1 end-to-end call as trt_sync_bench.py (H2D all inputs -> execute_async_v2
-> D2H all outputs -> cudaStreamSynchronize), back to back, for >= --duration seconds after a
time-based warm-up (>= --warmup_s s and >= --warmup iterations). Reports the wall-clock
(time.time()) start/end of the measured window and the number of inferences in it, so the power
log (power_logger.py, same clock) can be integrated over exactly this window:
    E_inf = (mean P_window - P_idle) * duration / n_inf.
GPU devfreq cur_freq is sampled every ~0.25 s inside the window (outside the call, cheap sysfs read).
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
from trt_sync_bench import Cuda, GPU_FREQ, read, vm_hwm_mb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup_s", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--label", default="")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    t_load0 = time.time()
    import tensorrt as trt
    cu = Cuda()
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(a.engine, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    stream = cu.stream()
    bindings, ins, outs = [], [], []
    rng = np.random.RandomState(0)
    for i in range(eng.num_bindings):
        shape = tuple(eng.get_binding_shape(i))
        dt = np.dtype({trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
                       trt.int8: np.int8, trt.bool: np.bool_}[eng.get_binding_dtype(i)])
        n = int(np.prod(shape)) * dt.itemsize
        d, h = cu.malloc(n), cu.malloc_host(n)
        host = np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(h.value)).view(dt).reshape(shape)
        bindings.append(d.value)
        if eng.binding_is_input(i):
            host[...] = rng.randn(*shape).astype(dt) if dt.kind == "f" else 0
            ins.append((d, h, n))
        else:
            outs.append((d, h, n))
    lib = cu.lib

    def e2e():
        for d, h, n in ins:
            lib.cudaMemcpyAsync(d, h, n, 1, stream)
        ctx.execute_async_v2(bindings, stream.value)
        for d, h, n in outs:
            lib.cudaMemcpyAsync(h, d, n, 2, stream)
        cu.check(lib.cudaStreamSynchronize(stream), "sync")

    t_warm0 = time.time()
    tw_end = time.perf_counter() + a.warmup_s
    nw = 0
    while nw < a.warmup or time.perf_counter() < tw_end:
        e2e()
        nw += 1
    fr = []
    t0w = time.time()
    p0 = time.perf_counter()
    p_end = p0 + a.duration
    p_next_f = p0
    n = 0
    while True:
        e2e()
        n += 1
        p = time.perf_counter()
        if p >= p_next_f:
            fr.append(int(read(GPU_FREQ)))
            p_next_f = p + 0.25
        if p >= p_end:
            break
    p1 = time.perf_counter()
    t1w = t0w + (p1 - p0)
    res = {
        "label": a.label, "run": a.run, "engine": os.path.basename(a.engine),
        "t_load_start": round(t_load0, 3), "t_warmup_start": round(t_warm0, 3),
        "t_start": round(t0w, 3), "t_end": round(t1w, 3), "duration_s": round(p1 - p0, 4),
        "n_inf": n, "mean_ms": round((p1 - p0) * 1e3 / n, 5), "warmup_iters": nw,
        "gpu_freq_hz_samples_n": len(fr), "gpu_freq_hz_min_med_max": [min(fr), int(np.median(fr)), max(fr)],
        "gpu_freq_frac_at_max": round(float(np.mean(np.array(fr) == 624750000)), 3),
        "vm_hwm_MB": round(vm_hwm_mb(), 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    line = json.dumps(res)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
