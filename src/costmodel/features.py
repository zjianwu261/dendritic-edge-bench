"""E-11 (results/costmodel/PROTOCOL.md P2/P3): deployed-graph features.

GPU / CPU tiers (P2), from an exported ONNX graph:
  K  executed nodes of the graph that onnxruntime saves at ORT_ENABLE_BASIC (constant folding,
     redundant-node elimination). Constant / Identity / Shape / Size nodes are not counted.
     An ONNX Loop counts as 1 outer node + trip_count x (nodes of its body), recursively.
  F  multiply-accumulates executed by MatMul / Gemm / Conv (grouped and batched included), from
     tensor shapes; zeros of masked matrices are counted; Loop bodies x trip count; element-wise
     ops are not counted.
  B  elements of the floating-point constant tensors (initializers and Constant nodes of the
     main graph and of every subgraph) after folding, i.e. the stored weights.
Also returned (for the proxies of P7 only, never used by the model fit):
  F_eff  MACs of an ideal sparse execution: every MatMul / Gemm / Conv whose weight operand is a
         constant is scaled by nnz(W) / numel(W); fused GRU / LSTM ops are counted analytically
         (seq_len x dirs x gates*H x (I + H)).
  B_nnz  nonzero elements of the floating-point constants.

8051 tier (P3): src/costmodel/mcu_features.py.

Usage as a library: graph_features(path) -> dict. CLI: python features.py <onnx> ...
"""
import collections
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

SKIP_OPS = {"Constant", "Identity", "Shape", "Size"}
MAC_OPS = {"MatMul", "Gemm", "Conv"}
FUSED_RNN = {"GRU", "LSTM", "RNN"}
FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE,
               onnx.TensorProto.BFLOAT16}
SCRATCH = os.environ.get("COSTMODEL_TMP", tempfile.gettempdir())


def optimize_basic(path):
    """Run onnxruntime (CPU EP) at ORT_ENABLE_BASIC and return the saved optimized ModelProto."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.log_severity_level = 3
    fd, out = tempfile.mkstemp(suffix=".onnx", dir=SCRATCH)
    os.close(fd)
    try:
        so.optimized_model_filepath = out
        ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        m = onnx.load(out)
    finally:
        os.remove(out)
        if os.path.exists(out + ".data"):
            os.remove(out + ".data")
    return m


def _shape_of(vi):
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            return None
    return tuple(dims)


class _Graph:
    """Constant values and static shapes of one (sub)graph, with access to the enclosing scope."""

    def __init__(self, g, parent=None):
        self.g = g
        self.parent = parent
        self.const = {}
        self.shape = {}
        for init in g.initializer:
            self.const[init.name] = init
            self.shape[init.name] = tuple(init.dims)
        for n in g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.const[n.output[0]] = a.t
                        self.shape[n.output[0]] = tuple(a.t.dims)
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            s = _shape_of(vi)
            if s is not None:
                self.shape.setdefault(vi.name, s)

    def get_shape(self, name):
        if name in self.shape:
            return self.shape[name]
        return self.parent.get_shape(name) if self.parent else None

    def get_const(self, name):
        if name in self.const:
            return self.const[name]
        return self.parent.get_const(name) if self.parent else None


def _attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _nnz_frac(t):
    a = numpy_helper.to_array(t)
    return float(np.count_nonzero(a)) / a.size if a.size else 0.0


def node_macs(n, G):
    """Executed MACs of one MatMul / Gemm / Conv node (dense count) and the nnz fraction of its
    constant weight operand (1.0 if no operand is constant)."""
    if n.op_type == "Gemm":
        a, b = G.get_shape(n.input[0]), G.get_shape(n.input[1])
        if a is None or b is None:
            raise ValueError(f"Gemm {n.name}: unknown shape {a} {b}")
        ta, tb = _attr(n, "transA", 0), _attr(n, "transB", 0)
        M, K = (a[1], a[0]) if ta else (a[0], a[1])
        N = b[0] if tb else b[1]
        macs = M * N * K
        w = n.input[1] if G.get_const(n.input[1]) is not None else (
            n.input[0] if G.get_const(n.input[0]) is not None else None)
    elif n.op_type == "MatMul":
        a, b = G.get_shape(n.input[0]), G.get_shape(n.input[1])
        o = G.get_shape(n.output[0])
        if a is None or o is None:
            raise ValueError(f"MatMul {n.name}: unknown shape {a} {o}")
        K = a[-1]
        macs = int(np.prod(o)) * K
        w = n.input[1] if G.get_const(n.input[1]) is not None else (
            n.input[0] if G.get_const(n.input[0]) is not None else None)
    elif n.op_type == "Conv":
        wsh = G.get_shape(n.input[1])
        o = G.get_shape(n.output[0])
        if wsh is None or o is None:
            raise ValueError(f"Conv {n.name}: unknown shape {wsh} {o}")
        macs = int(np.prod(o)) * int(np.prod(wsh[1:]))  # wsh = (Cout, Cin/group, kH, kW...)
        w = n.input[1] if G.get_const(n.input[1]) is not None else None
    else:
        raise ValueError(n.op_type)
    frac = _nnz_frac(G.get_const(w)) if w is not None else 1.0
    return int(macs), frac


def fused_rnn_macs(n, G):
    """Analytic MACs of an ONNX GRU / LSTM / RNN node (for the proxies only)."""
    x = G.get_shape(n.input[0])
    W = G.get_shape(n.input[1])
    R = G.get_shape(n.input[2])
    if x is None or W is None or R is None:
        raise ValueError(f"{n.op_type} {n.name}: unknown shape")
    layout = _attr(n, "layout", 0)
    T = x[1] if layout else x[0]
    batch = x[0] if layout else x[1]
    dirs, gH, I = W
    H = R[2]
    return int(T * batch * dirs * gH * (I + H))


def _trip_count(n, G, default_trips):
    m = n.input[0] if len(n.input) else ""
    if m:
        t = G.get_const(m)
        if t is not None:
            return int(numpy_helper.to_array(t).reshape(-1)[0])
    if default_trips is None:
        raise ValueError(f"Loop {n.name}: trip count is not a constant")
    return int(default_trips)


def _walk(g, parent, default_trips, acc, mult, seen_const):
    G = _Graph(g, parent)
    for name, t in G.const.items():
        if name in seen_const:
            continue
        seen_const.add(name)
        if t.data_type in FLOAT_TYPES:
            a = numpy_helper.to_array(t)
            acc["B"] += int(a.size)
            acc["B_nnz"] += int(np.count_nonzero(a))
    for n in g.node:
        if n.op_type not in SKIP_OPS:
            acc["K"] += mult
            acc["ops"][n.op_type] += mult
        if n.op_type in MAC_OPS:
            macs, frac = node_macs(n, G)
            acc["F"] += mult * macs
            acc["F_eff"] += mult * macs * frac
        elif n.op_type in FUSED_RNN:
            macs = fused_rnn_macs(n, G)
            acc["F_eff"] += mult * macs
            acc["F_fusedrnn"] += mult * macs
            acc["fused_rnn"] = True
        elif n.op_type == "Loop":
            trips = _trip_count(n, G, default_trips)
            acc["loop_trips"].append(trips)
            body = _attr(n, "body")
            _walk(body, G, default_trips, acc, mult * trips, seen_const)
        elif n.op_type in ("Scan", "If"):
            raise NotImplementedError(f"{n.op_type} is not handled (none expected in the E-11 graphs)")


def features_from_model(m, default_trips=None):
    m = shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=True)
    acc = dict(K=0, F=0, F_eff=0.0, B=0, B_nnz=0, F_fusedrnn=0, fused_rnn=False,
               ops=collections.Counter(), loop_trips=[])
    _walk(m.graph, None, default_trips, acc, 1, set())
    acc["F_eff"] = int(round(acc["F_eff"]))
    return acc


def graph_features(path, default_trips=None):
    m = optimize_basic(path)
    f = features_from_model(m, default_trips)
    f["path"] = path
    return f


if __name__ == "__main__":
    for p in sys.argv[1:]:
        f = graph_features(p)
        print(os.path.basename(p), {k: f[k] for k in ("K", "F", "B", "F_eff", "B_nnz", "loop_trips",
                                                     "fused_rnn")}, dict(f["ops"]))
