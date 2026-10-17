"""Orin latency of the NoMaD + dend2 adapter chain (explore mode). Exports two encoder graphs with the
dend2 adapters (cl_methods.DendriticAdapterV2, d 256, h 192, kWTA 5 %, 5 allocated contexts, z dim 272 = v2 context
features [frozen NoMaD feature 256, 16-bin brightness histogram]) inserted exactly where cl_methods.encode puts them
(token adapter on the 5 tokens before the transformer, output adapter on the pooled conditioning vector):

  nomad_enc_dend2id_b1_gnd.onnx     inputs obs, goal, z[1,272]  (given context ID: z = stored prototype, fed by the host)
  nomad_enc_dend2inf_b1_gnd.onnx    inputs obs, goal            (inferred context, all inside the graph:
                                    frozen feature = transformer on the un-adapted tokens (2nd transformer pass,
                                    the EfficientNets are shared), 16-bin histogram of the current frame recovered from
                                    the normalised obs, per-dim standardisation, nearest of 5 prototypes -> z)
Deployment rewrites that do not change the function (up to ties): segment selection |d|-argmax and nearest-prototype
argmin as equality masks instead of ArgMax + Gather; round(x) as floor(x + 0.5); histogram bins as float equality.
Adapter / prototype / standardisation values are random (5 contexts allocated): TRT latency does not depend on the
values. The base graph is the existing results/T4a/models/nomad_enc_explore_b1_gnd.onnx (export_enc_v2.py).
Each graph is checked against the reference PyTorch path (cl_methods.encode with the adapters, ContextInferer-style
z) with onnxruntime CPU.  Output: results/T4a/dend2_orin/onnx/ + export_log.json
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cl_methods import DendriticAdapterV2, encode  # noqa: E402
from common import EXP, load_model  # noqa: E402
from ctx_infer import MEAN, STD, current_gray, low_level  # noqa: E402
from export_enc_v2 import EncA, EncB, swap_gn  # noqa: E402
from export_onnx import no_swish  # noqa: E402

OUT = os.path.join(EXP, "dend2_orin", "onnx")
NCTX, H, ZD = 5, 192, 272


def make_adapter(gen):
    a = DendriticAdapterV2(256, H, NCTX, ZD, 0.05, 3.0)
    with torch.no_grad():
        a.up.weight.normal_(0, 0.02, generator=gen)
        for _ in range(NCTX):
            a.add_context(torch.randn(ZD, generator=gen), gen=gen)
    return a.eval()


class AdDeploy(nn.Module):
    """DendriticAdapterV2.forward with the |d|-argmax segment selection as an equality mask."""

    def __init__(self, a):
        super().__init__()
        self.down, self.up = a.down, a.up
        self.register_buffer("seg", a.seg[:a.n_ctx].clone())
        self.k = max(1, int(round(a.kwta.frac * a.seg.shape[1])))

    def forward(self, x, z):
        dend = torch.einsum("shc,bc->bsh", self.seg, z)
        ad = dend.abs()
        m = (ad == ad.max(1, keepdim=True).values).to(dend.dtype)
        g = torch.sigmoid((dend * m).sum(1))
        a = F.relu(self.down(x)) * (g.unsqueeze(1) if x.dim() == 3 else g)
        thr = a.topk(self.k, dim=-1).values[..., -1:]
        a = a * (a >= thr).to(a.dtype)
        return x + self.up(a)


class EncDendId(nn.Module):
    def __init__(self, v, tok, out):
        super().__init__()
        self.a, self.b, self.tok, self.out = EncA(v), EncB(v), AdDeploy(tok), AdDeploy(out)

    def forward(self, obs, goal, z):
        return self.out(self.b(self.tok(self.a(obs, goal), z)), z)


class EncDendInf(EncDendId):
    def __init__(self, v, tok, out, mu0, sd0, protos):
        super().__init__(v, tok, out)
        self.register_buffer("mu0", mu0); self.register_buffer("sd0", sd0); self.register_buffer("protos", protos)
        self.register_buffer("mean", MEAN.clone()); self.register_buffer("std", STD.clone())
        self.register_buffer("bins", torch.arange(16, dtype=torch.float32))

    def forward(self, obs, goal):
        tok = self.a(obs, goal)
        f = self.b(tok)                                           # frozen feature (goal masked, explore)
        x = obs[:, -3:]
        rgb = torch.floor((x * self.std + self.mean) * 255 + 0.5).clamp(0, 255)
        gray = rgb.mean(1).flatten(1)
        b = torch.floor(gray / 16).clamp(0, 15)
        hist = (b.unsqueeze(-1) == self.bins).to(gray.dtype).mean(1)
        zs = (torch.cat([f, hist], -1) - self.mu0) / self.sd0
        d2 = ((zs.unsqueeze(1) - self.protos.unsqueeze(0)) ** 2).sum(-1)
        m = (d2 == d2.min(1, keepdim=True).values).to(zs.dtype)
        z = m @ self.protos
        return self.out(self.b(self.tok(tok, z)), z)


def exp(mod, args, ins, name, ref):
    p = os.path.join(OUT, name)
    torch.onnx.export(mod, args, p, input_names=ins, output_names=["cond"], opset_version=16, do_constant_folding=True, dynamo=False)
    s, ok = onnxsim.simplify(onnx.load(p)); assert ok
    onnx.save(s, p)
    import onnxruntime as ort
    o = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, {n: x.numpy() for n, x in zip(ins, args)})[0]
    ops = {}
    for n in s.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = {"onnx": name, "max_abs_err_vs_reference": float(np.abs(o - ref).max()), "n_nodes": len(s.graph.node),
            "ops": ops, "size_MB": round(os.path.getsize(p) / 2**20, 2)}
    print(json.dumps({k: info[k] for k in ("onnx", "max_abs_err_vs_reference", "n_nodes", "size_MB")}))
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    m, _ = load_model("nomad"); no_swish(m)
    v = m.vision_encoder.eval()
    tok_a, out_a = make_adapter(gen), make_adapter(gen)
    obs, goal = torch.randn(1, 12, 96, 96), torch.randn(1, 3, 96, 96)
    one = torch.ones(1, dtype=torch.long)
    with torch.no_grad():
        # reference context statistics: random-image prototypes through the real frozen path
        zr = []
        for _ in range(8):
            o2 = torch.randn(1, 12, 96, 96, generator=gen)
            hist, _ = low_level(current_gray(o2))
            zr.append(torch.cat([encode(v, o2, torch.randn(1, 3, 96, 96, generator=gen), one), hist], -1))
        zr = torch.cat(zr)
        mu0, sd0 = zr.mean(0), zr.std(0).clamp_min(1e-3)
        protos = (zr[:NCTX] - mu0) / sd0
        # reference outputs (original modules)
        z_id = protos[2:3]
        ref_id = encode(v, obs, goal, one, tok_a, out_a, z_id).numpy()
        f = encode(v, obs, goal, one)
        hist, _ = low_level(current_gray(obs))
        zs = (torch.cat([f, hist], -1) - mu0) / sd0
        z_inf = protos[torch.cdist(zs, protos).argmin(1)]
        ref_inf = encode(v, obs, goal, one, tok_a, out_a, z_inf).numpy()
        swap_gn(v); v.eval()
        log = {"nctx": NCTX, "h": H, "zdim": ZD, "kwta": 0.05,
               "nearest_proto_for_test_input": int(torch.cdist(zs, protos).argmin()),
               "graphs": [exp(EncDendId(v, tok_a, out_a).eval(), (obs, goal, z_id), ["obs", "goal", "z"], "nomad_enc_dend2id_b1_gnd.onnx", ref_id),
                          exp(EncDendInf(v, tok_a, out_a, mu0, sd0, protos).eval(), (obs, goal), ["obs", "goal"],
                              "nomad_enc_dend2inf_b1_gnd.onnx", ref_inf)]}
    json.dump(log, open(os.path.join(OUT, "export_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
