"""Synchronous batch=1 latency of a serialized TensorRT engine on Jetson.

Each timed iteration is strictly serial:
    perf_counter -> H2D (all inputs) -> execute_async_v2 -> D2H (all outputs)
    -> cudaStreamSynchronize -> perf_counter
so it includes host enqueue/launch overhead and nothing overlaps with the previous inference.
Host buffers are pinned (cudaMallocHost). A second pass times execute_async_v2 + sync only
("infer_only", no copies) as a supplementary column.

Warm-up is time based (>= --warmup_s seconds AND >= --warmup
iterations, same e2e call as the timed loop); the GPU devfreq cur_freq is read at the end of the
warm-up and sampled every --freq_every timed iterations (the sysfs read is outside the timed
region). The Orin is NOT clock-locked (DVFS, as deployed).
If the engine has recurrent-state bindings (input X with output X_o), a
third pass "stateful" keeps the state on the GPU (two device buffer sets, ping-pong), and
copies only the non-state inputs H2D and the non-state outputs D2H per step.

No pycuda / cuda-python on Orin #1, so the CUDA runtime is called through ctypes.
Runs on the Jetson system Python 3.8 with TensorRT 8.5 python bindings.

T4a copy: identical to src/bench/trt_sync_bench.py except init_libnvinfer_plugins() before deserializing
(NoMaD engines contain the InstanceNormalization plugin).

Usage: python3 trt_sync_bench.py model.engine --label X --run 0 --out results.jsonl
"""
import argparse
import warnings
import ctypes
import json
import os
import platform
import time

import numpy as np

GPU_FREQ = "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq"


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
        lib.cudaFree.argtypes = [vp]
        lib.cudaFreeHost.argtypes = [vp]
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
    ap.add_argument("--warmup", type=int, default=50, help="minimum warm-up iterations")
    ap.add_argument("--warmup_s", type=float, default=2.0, help="minimum warm-up seconds")
    ap.add_argument("--freq_every", type=int, default=25)
    ap.add_argument("--build", default="", help="TensorRT build variant tag (default / nocublas)")
    ap.add_argument("--onnx_sha256", default="")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--label", default="")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--device", default="orin1")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    avail0 = mem_available_mb()
    warnings.simplefilter("ignore")  # TRT 8.5 binding-API deprecation warnings
    import tensorrt as trt
    cu = Cuda()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")  # T4a: NoMaD GroupNorm -> InstanceNormalization plugin
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = cu.stream()

    bindings, ins, outs, shapes, names = [], [], [], {}, []
    rng = np.random.RandomState(0)
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(engine.get_binding_shape(i))
        if any(d < 0 for d in shape):
            raise SystemExit(f"binding {name} has dynamic shape {shape}; export static shapes")
        # trt.nptype() uses np.bool, which numpy >= 1.24 (pip-installed on Orin #1) removed
        dt = np.dtype({trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
                       trt.int8: np.int8, trt.bool: np.bool_}[engine.get_binding_dtype(i)])
        n = int(np.prod(shape)) * dt.itemsize
        dptr, hptr = cu.malloc(n), cu.malloc_host(n)
        host = np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(hptr.value)).view(dt).reshape(shape)
        bindings.append(dptr.value)
        names.append((name, engine.binding_is_input(i), n, hptr))
        shapes[name] = list(shape)
        if engine.binding_is_input(i):
            host[...] = rng.randn(*shape).astype(dt) if dt.kind == "f" else 0
            ins.append((dptr, hptr, n))
        else:
            outs.append((dptr, hptr, n))
    avail_alloc = mem_available_mb()

    lib = cu.lib

    def e2e():
        for d, h, n in ins:
            lib.cudaMemcpyAsync(d, h, n, 1, stream)
        ctx.execute_async_v2(bindings, stream.value)
        for d, h, n in outs:
            lib.cudaMemcpyAsync(h, d, n, 2, stream)
        cu.check(lib.cudaStreamSynchronize(stream), "sync")

    def infer_only():
        ctx.execute_async_v2(bindings, stream.value)
        cu.check(lib.cudaStreamSynchronize(stream), "sync")

    # ---- recurrent state pairs (input X <-> output X_o), for the stateful pass
    idx = {nm: i for i, (nm, _, _, _) in enumerate(names)}
    pairs = [(idx[nm], idx[nm + "_o"]) for nm, is_in, _, _ in names if is_in and (nm + "_o") in idx]
    state_in = {a for a, _ in pairs}
    state_out = {b for _, b in pairs}
    ping = pong = None
    if pairs:
        ping = list(bindings)
        pong = list(bindings)
        for a, b in pairs:  # pong: read state from the ping outputs, write into the ping inputs
            pong[a], pong[b] = ping[b], ping[a]
        # (device ptr, host ptr, bytes) of the non-state inputs / outputs (same buffers in ping and pong)
        x_ins = [(ctypes.c_void_p(bindings[i]), names[i][3], names[i][2]) for i in range(len(names))
                 if names[i][1] and i not in state_in]
        y_outs = [(ctypes.c_void_p(bindings[i]), names[i][3], names[i][2]) for i in range(len(names))
                  if not names[i][1] and i not in state_out]
        cur = [0]

        def stateful():
            bnd = ping if cur[0] == 0 else pong
            for d, h, n in x_ins:
                lib.cudaMemcpyAsync(d, h, n, 1, stream)
            ctx.execute_async_v2(bnd, stream.value)
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
        "device": args.device, "runtime": f"TensorRT {trt.__version__} python (ctypes cudart)",
        "precision": args.precision, "binding_shapes": shapes,
        "build": args.build, "onnx_sha256": args.onnx_sha256,
        "warmup": nw, "warmup_rule": f">= {args.warmup_s:g} s and >= {args.warmup} iters",
        "iters": args.iters, "timing_scope": "sync_e2e_pinned",
        "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95), "mean_ms": round(float(ts.mean()), 4),
        "min_ms": round(float(ts.min()), 4), "max_ms": round(float(ts.max()), 4),
        "infer_only_p50_ms": pct(ti, 50), "infer_only_p95_ms": pct(ti, 95),
        "mem_avail_drop_MB": round(avail0 - min(avail_alloc, avail_end), 1),
        "vm_hwm_MB": round(vm_hwm_mb(), 1),
        "engine_device_mem_MB": round(engine.device_memory_size / 2**20, 3),
        "engine_file_MB": round(os.path.getsize(args.engine) / 2**20, 3),
        "gpu_freq_hz_before_warmup_after_warmup_after_timing": [f0, fw, f1],
        "gpu_freq_hz_warmup_end": int(fw),
        "gpu_freq_hz_timed_samples": fr,
        "gpu_freq_hz_timed_min_med_max": [min(fr), int(np.median(fr)), max(fr)],
        "gpu_freq_timed_frac_at_max": round(float(np.mean(np.array(fr) == 624750000)), 3),
        "infer_only_warmup": nwi, "infer_only_gpu_freq_hz_min_med_max": [min(fri), int(np.median(fri)), max(fri)],
        "python": platform.python_version(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if tsf is not None:
        res.update({"stateful_p50_ms": pct(tsf, 50), "stateful_p95_ms": pct(tsf, 95), "stateful_warmup": nws,
                    "stateful_state_pairs": [[names[a][0], names[b][0]] for a, b in pairs],
                    "stateful_gpu_freq_hz_min_med_max": [min(frs), int(np.median(frs)), max(frs)]})
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
