"""Run a test set through a TensorRT engine on the Jetson, batch = 1, and save the outputs.

Input: an .npz with one array per engine input, first axis = sample. Arrays named
<input>__packbits (uint8, np.packbits over the flattened sample) are unpacked to float32 0/1
per sample (used for SHD spike rasters). Output: .npy with the first engine output per sample.
Accuracy is computed on the workstation (src/t1/make_results.py / device_acc.py).
Python 3.8 + TensorRT 8.5 python bindings; CUDA runtime via ctypes (same as trt_sync_bench).
"""
import argparse
import ctypes
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_sync_bench import Cuda  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("data")
    ap.add_argument("out")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    import tensorrt as trt
    cu = Cuda()
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(args.engine, "rb").read())
    ctx = engine.create_execution_context()
    stream = cu.stream()
    npmap = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
             trt.int8: np.int8, trt.bool: np.bool_}
    data = np.load(args.data)
    bindings, ins, outs = [], [], []
    for i in range(engine.num_bindings):
        name, shape = engine.get_binding_name(i), tuple(engine.get_binding_shape(i))
        dt = np.dtype(npmap[engine.get_binding_dtype(i)])
        n = int(np.prod(shape)) * dt.itemsize
        d, h = cu.malloc(n), cu.malloc_host(n)
        host = np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(h.value)).view(dt).reshape(shape)
        bindings.append(d.value)
        (ins if engine.binding_is_input(i) else outs).append((name, shape, d, h, n, host))
    src = {}
    for name, shape, *_ in ins:
        if name in data.files:
            src[name] = (data[name], False)
        elif name + "__packbits" in data.files:
            src[name] = (data[name + "__packbits"], True)
        else:
            raise SystemExit(f"no data for input {name}; have {data.files}")
    N = len(next(iter(src.values()))[0])
    res = []
    for k in range(N):
        for name, shape, d, h, n, host in ins:
            arr, packed = src[name]
            x = np.unpackbits(arr[k])[:int(np.prod(shape))] if packed else arr[k]
            host[...] = x.reshape(shape)
            cu.lib.cudaMemcpyAsync(d, h, n, 1, stream)
        ctx.execute_async_v2(bindings, stream.value)
        for name, shape, d, h, n, host in outs:
            cu.lib.cudaMemcpyAsync(h, d, n, 2, stream)
        cu.check(cu.lib.cudaStreamSynchronize(stream), "sync")
        res.append(np.array(outs[0][5], dtype=np.float32).reshape(-1))
    np.save(args.out, np.stack(res))
    print(f"{args.engine}: {N} samples -> {args.out}")


if __name__ == "__main__":
    main()
