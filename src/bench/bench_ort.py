"""onnxruntime batch=1 latency benchmark (warm-up >= 2 s and >= 50 iterations, timed 500 by default).

Synchronous timing: each timed call is one blocking `sess.run` (numpy in -> numpy out).
Outputs one JSON line (stdout, and appended to --out if given) with p50/p95/mean latency in
ms, the actual input shapes, and memory info. Works with Python 3.8+ (Raspberry Pi / 5080).

Feeds *all* model inputs; dynamic dims must be given explicitly with
  --shape name=1x1000x700 (repeatable); otherwise the script exits with an error.
--preload-torch (5080: load the cuDNN 9 / CUDA 12 libs shipped with the torch
  wheels before ORT); CPU-EP fallback disabled for non-CPU providers; --precision is a label
  checked against the model's input dtypes.
"""
import argparse
import json
import os
import platform
import resource
import time

import numpy as np

ORT_TYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
             "tensor(double)": np.float64, "tensor(int64)": np.int64,
             "tensor(int32)": np.int32, "tensor(bool)": np.bool_, "tensor(uint8)": np.uint8}


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def vm_hwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_shapes(items):
    out = {}
    for it in items or []:
        name, dims = it.split("=")
        out[name] = [int(d) for d in dims.lower().split("x")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads, 0 = ORT default")
    ap.add_argument("--warmup", type=int, default=50, help="minimum warm-up iterations")
    ap.add_argument("--warmup_s", type=float, default=2.0, help="minimum warm-up seconds")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default=platform.node())
    ap.add_argument("--shape", action="append", help="name=1x1000x700 for inputs with dynamic dims")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--preload-torch", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default="", help="append the JSON line to this file")
    args = ap.parse_args()

    if args.preload_torch:
        import torch  # noqa: F401
    rss_python = vm_hwm_mb()
    avail0 = mem_available_mb()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    if args.provider != "CPUExecutionProvider":
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(args.model, so, providers=[args.provider])
    if sess.get_providers()[0] != args.provider:
        raise SystemExit(f"requested {args.provider} but session uses {sess.get_providers()} (silent fallback)")

    given = parse_shapes(args.shape)
    rng = np.random.RandomState(0)
    feed, shapes = {}, {}
    for inp in sess.get_inputs():
        if inp.name in given:
            shape = given[inp.name]
            if len(shape) != len(inp.shape) or any(isinstance(d, int) and d != g for d, g in zip(inp.shape, shape)):
                raise SystemExit(f"--shape {inp.name}={shape} incompatible with model shape {inp.shape}")
        else:
            if not all(isinstance(d, int) for d in inp.shape):
                raise SystemExit(f"input {inp.name} has dynamic shape {inp.shape}; pass --shape {inp.name}=...")
            shape = list(inp.shape)
        dt = ORT_TYPES[inp.type]
        if args.precision == "fp32" and dt not in (np.float32, np.int64, np.int32, np.bool_, np.uint8):
            raise SystemExit(f"--precision fp32 but input {inp.name} is {inp.type}")
        feed[inp.name] = rng.randn(*shape).astype(dt) if np.issubdtype(dt, np.floating) else np.zeros(shape, dt)
        shapes[inp.name] = shape
    for k in given:
        if k not in feed:
            raise SystemExit(f"--shape given for unknown input {k}")

    t_end = time.perf_counter() + args.warmup_s
    nw = 0
    while nw < args.warmup or time.perf_counter() < t_end:
        sess.run(None, feed)
        nw += 1
    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    res = {
        "label": args.label, "run": args.run, "model": os.path.basename(args.model),
        "device": args.device, "runtime": f"onnxruntime {ort.__version__}",
        "provider": sess.get_providers()[0], "precision": args.precision,
        "input_shapes": shapes, "threads": args.threads,
        "affinity": sorted(os.sched_getaffinity(0)),
        "warmup": nw, "warmup_rule": f">= {args.warmup_s:g} s and >= {args.warmup} iters",
        "iters": args.iters, "timing_scope": "sync_e2e_python",
        "p50_ms": round(float(np.percentile(ts, 50)), 4),
        "p95_ms": round(float(np.percentile(ts, 95)), 4),
        "mean_ms": round(float(ts.mean()), 4),
        "min_ms": round(float(ts.min()), 4), "max_ms": round(float(ts.max()), 4),
        "vm_hwm_MB": round(vm_hwm_mb(), 1),
        "vm_hwm_before_ort_MB": round(rss_python, 1),
        "mem_avail_drop_MB": round(avail0 - mem_available_mb(), 1),
        "python": platform.python_version(), "machine": platform.machine(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
