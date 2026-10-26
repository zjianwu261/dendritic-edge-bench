"""Orin #2 (car, JetPack 6.2 / TensorRT 10.7) port of src/t6/orin/t6_sync_bench.py (= src/bench/trt_sync_bench.py,
T1r version).  Cross-device timing for the Paper B robustness check (2026-10-26).

Timing protocol is unchanged: each timed iteration is strictly serial
    perf_counter -> H2D (all inputs) -> execute_async_v3 -> D2H (all outputs) -> cudaStreamSynchronize -> perf_counter
pinned host buffers, warm-up >= --warmup_s seconds and >= --warmup iterations, 500 timed iterations, then a second
"infer_only" pass (execute + sync only).  GPU devfreq cur_freq is sampled every --freq_every iterations outside the
timed region.

Only API changes for TensorRT 10: the binding API (num_bindings / get_binding_* / execute_async_v2) was removed, so
I/O tensors are enumerated by name (num_io_tensors / get_tensor_*), addresses are set with set_tensor_address and
the engine runs with execute_async_v3(stream).  The stateful pass (state input X <-> output X_o) is kept for
completeness; none of the graphs timed here has state I/O.  GPU devfreq path is the Orin #2 one; frac_at_max is
measured against the devfreq max_freq read at start-up (the car is NOT clock-locked: DVFS).
Runs on the car's system Python 3.10 (numpy 1.26, TensorRT 10.7 python bindings); CUDA runtime through ctypes.
"""
import argparse
import ctypes
import json
import os
import platform
import time

import numpy as np

GDIR = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
GPU_FREQ = f"{GDIR}/cur_freq"


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0


def vm_hwm_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0


def pct(a, q):
    return round(float(np.percentile(a, q)), 4)


class Cuda:
    def __init__(self):
        lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        lib.cudaMalloc.argtypes = [ctypes.POINTER(vp), sz]
        lib.cudaMallocHost.argtypes = [ctypes.POINTER(vp), sz]
        lib.cudaMemcpyAsync.argtypes = [vp, vp, sz, ctypes.c_int, vp]
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(vp)]
        lib.cudaStreamSynchronize.argtypes = [vp]
        self.lib = lib

    def check(self, err, what):
        if err != 0:
            raise RuntimeError(f"{what} failed with cudaError {err}")

    def malloc(self, n):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), n), "cudaMalloc")
        return p

    def malloc_host(self, n):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMallocHost(ctypes.byref(p), n), "cudaMallocHost")
        return p

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--warmup_s", type=float, default=2.0)
    ap.add_argument("--freq_every", type=int, default=25)
    ap.add_argument("--build", default="")
    ap.add_argument("--onnx_sha256", default="")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--label", default="")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--device", default="orin2")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    avail0 = mem_available_mb()
    gmax = int(read(f"{GDIR}/max_freq"))
    import tensorrt as trt
    cu = Cuda()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = cu.stream()

    ins, outs, shapes, names, addr = [], [], {}, [], {}
    rng = np.random.RandomState(0)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        if any(d < 0 for d in shape):
            raise SystemExit(f"tensor {name} has dynamic shape {shape}; export static shapes")
        dt = np.dtype({trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
                       trt.int8: np.int8, trt.bool: np.bool_, trt.int64: np.int64}[engine.get_tensor_dtype(name)])
        is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        n = max(int(np.prod(shape)) * dt.itemsize, 1)
        dptr, hptr = cu.malloc(n), cu.malloc_host(n)
        host = np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(hptr.value))[: int(np.prod(shape)) * dt.itemsize]
        host = host.view(dt).reshape(shape)
        ctx.set_tensor_address(name, dptr.value)
        addr[name] = dptr.value
        names.append((name, is_in, n, hptr))
        shapes[name] = list(shape)
        if is_in:
            host[...] = rng.randn(*shape).astype(dt) if dt.kind == "f" else 0
            ins.append((dptr, hptr, n))
        else:
            outs.append((dptr, hptr, n))
    avail_alloc = mem_available_mb()
    lib = cu.lib

    def e2e():
        for d, h, n in ins:
            lib.cudaMemcpyAsync(d, h, n, 1, stream)
        ctx.execute_async_v3(stream.value)
        for d, h, n in outs:
            lib.cudaMemcpyAsync(h, d, n, 2, stream)
        cu.check(lib.cudaStreamSynchronize(stream), "sync")

    def infer_only():
        ctx.execute_async_v3(stream.value)
        cu.check(lib.cudaStreamSynchronize(stream), "sync")

    nm_set = {nm for nm, _, _, _ in names}
    pairs = [(nm, nm + "_o") for nm, is_in, _, _ in names if is_in and (nm + "_o") in nm_set]
    if pairs:
        st = {a for p in pairs for a in p}
        x_ins = [(ctypes.c_void_p(addr[nm]), h, n) for nm, i, n, h in names if i and nm not in st]
        y_outs = [(ctypes.c_void_p(addr[nm]), h, n) for nm, i, n, h in names if not i and nm not in st]
        cur = [0]

        def stateful():
            for a, b in pairs:  # ping: as allocated; pong: swapped
                if cur[0] == 0:
                    ctx.set_tensor_address(a, addr[a]); ctx.set_tensor_address(b, addr[b])
                else:
                    ctx.set_tensor_address(a, addr[b]); ctx.set_tensor_address(b, addr[a])
            for d, h, n in x_ins:
                lib.cudaMemcpyAsync(d, h, n, 1, stream)
            ctx.execute_async_v3(stream.value)
            for d, h, n in y_outs:
                lib.cudaMemcpyAsync(h, d, n, 2, stream)
            cu.check(lib.cudaStreamSynchronize(stream), "sync")
            cur[0] ^= 1

    def warm(fn):
        t_end = time.perf_counter() + args.warmup_s
        n = 0
        while n < args.warmup or time.perf_counter() < t_end:
            fn()
            n += 1
        return n

    def timed(fn):
        ts, fr = np.empty(args.iters), []
        for k in range(args.iters):
            if k % args.freq_every == 0:
                fr.append(int(read(GPU_FREQ)))
            t0 = time.perf_counter()
            fn()
            ts[k] = (time.perf_counter() - t0) * 1e3
        fr.append(int(read(GPU_FREQ)))
        return ts, fr

    f0 = read(GPU_FREQ)
    nw = warm(e2e)
    fw = read(GPU_FREQ)
    ts, fr = timed(e2e)
    f1 = read(GPU_FREQ)
    nwi = warm(infer_only)
    ti, fri = timed(infer_only)
    tsf = None
    if pairs:
        nws = warm(stateful)
        tsf, frs = timed(stateful)
    avail_end = mem_available_mb()

    res = {
        "label": args.label, "run": args.run, "engine": os.path.basename(args.engine),
        "device": args.device, "runtime": f"TensorRT {trt.__version__} python (ctypes cudart, execute_async_v3)",
        "precision": args.precision, "binding_shapes": shapes,
        "build": args.build, "onnx_sha256": args.onnx_sha256,
        "warmup": nw, "warmup_rule": f">= {args.warmup_s:g} s and >= {args.warmup} iters",
        "iters": args.iters, "timing_scope": "sync_e2e_pinned",
        "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95), "mean_ms": round(float(ts.mean()), 4),
        "min_ms": round(float(ts.min()), 4), "max_ms": round(float(ts.max()), 4),
        "infer_only_p50_ms": pct(ti, 50), "infer_only_p95_ms": pct(ti, 95),
        "mem_avail_drop_MB": round(avail0 - min(avail_alloc, avail_end), 1),
        "vm_hwm_MB": round(vm_hwm_mb(), 1),
        "engine_device_mem_MB": round(engine.device_memory_size_v2 / 2**20, 3)
        if hasattr(engine, "device_memory_size_v2") else round(engine.device_memory_size / 2**20, 3),
        "engine_file_MB": round(os.path.getsize(args.engine) / 2**20, 3),
        "gpu_freq_hz_before_warmup_after_warmup_after_timing": [f0, fw, f1],
        "gpu_freq_hz_warmup_end": int(fw),
        "gpu_freq_hz_timed_samples": fr,
        "gpu_freq_hz_timed_min_med_max": [min(fr), int(np.median(fr)), max(fr)],
        "gpu_devfreq_max_hz": gmax,
        "gpu_freq_timed_frac_at_max": round(float(np.mean(np.array(fr) == gmax)), 3),
        "infer_only_warmup": nwi, "infer_only_gpu_freq_hz_min_med_max": [min(fri), int(np.median(fri)), max(fri)],
        "python": platform.python_version(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if tsf is not None:
        res.update({"stateful_p50_ms": pct(tsf, 50), "stateful_p95_ms": pct(tsf, 95), "stateful_warmup": nws,
                    "stateful_state_pairs": [list(p) for p in pairs],
                    "stateful_gpu_freq_hz_min_med_max": [min(frs), int(np.median(frs)), max(frs)]})
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
