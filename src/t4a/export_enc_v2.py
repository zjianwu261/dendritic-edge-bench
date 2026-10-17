"""Re-export the NoMaD vision encoder for TensorRT 8.5 (the original export failed to build on Orin #1:
"Requested amount of GPU memory (512455680 bytes) could not be allocated" for every tactic).

Changes vs export_onnx.py:
  - GroupNorm is replaced by an explicit decomposition (reshape -> mean/var -> normalize -> affine),
    so the graph has no InstanceNormalization plugin layers (torch exports GroupNorm as Reshape +
    InstanceNormalization, which TRT 8.5 maps to a plugin);
  - every graph is simplified with onnxsim (constant folding);
  - besides the full encoder, a split version: encA = EfficientNet obs/goal encoders + compression
    -> tokens [1,5,256]; encB = positional encoding + transformer (goal masked) + masked mean pooling.
Outputs (results/T4a/models/): nomad_enc_explore_b1_gnd.onnx, nomad_encA_b1_gnd.onnx, nomad_encB_b1_gnd.onnx
Each checked against the official PyTorch encoder (onnxruntime CPU).
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cl_methods import encode  # noqa: E402
from common import EXP, load_model  # noqa: E402
from export_onnx import no_swish  # noqa: E402

OUT = os.path.join(EXP, "models")


class GNd(nn.Module):
    def __init__(self, gn):
        super().__init__()
        self.G, self.eps = gn.num_groups, gn.eps
        self.weight, self.bias = gn.weight, gn.bias

    def forward(self, x):
        N, C = x.shape[0], x.shape[1]
        xg = x.reshape(N, self.G, -1)
        mu = xg.mean(-1, keepdim=True)
        var = ((xg - mu) ** 2).mean(-1, keepdim=True)
        xg = (xg - mu) / torch.sqrt(var + self.eps)
        shp = [1, C] + [1] * (x.dim() - 2)
        return xg.reshape(x.shape) * self.weight.view(shp) + self.bias.view(shp)


def swap_gn(mod):
    for name, ch in mod.named_children():
        if isinstance(ch, nn.GroupNorm):
            setattr(mod, name, GNd(ch))
        else:
            swap_gn(ch)


class EncA(nn.Module):
    def __init__(self, v):
        super().__init__(); self.v = v

    def forward(self, obs, goal):
        v, cs = self.v, self.v.context_size
        g = v.goal_encoder._avg_pooling(v.goal_encoder.extract_features(torch.cat([obs[:, 3 * cs:], goal], 1)))
        g = v.compress_goal_enc(g.flatten(1)).unsqueeze(1)
        o = torch.cat(torch.split(obs, 3, dim=1), dim=0)
        o = v.compress_obs_enc(v.obs_encoder._avg_pooling(v.obs_encoder.extract_features(o)).flatten(1))
        o = o.reshape(cs + 1, -1, v.obs_encoding_size).transpose(0, 1)
        return torch.cat([o, g], 1)


class EncB(nn.Module):
    def __init__(self, v):
        super().__init__(); self.v = v

    def forward(self, tok):
        v = self.v
        mk = torch.ones(tok.shape[0], dtype=torch.long)
        kpm = torch.index_select(v.all_masks, 0, mk)
        x = v.sa_encoder(v.positional_encoding(tok), src_key_padding_mask=kpm)
        avg = torch.index_select(v.avg_pool_mask, 0, mk).unsqueeze(-1)
        return torch.mean(x * avg, dim=1)


class EncFull(nn.Module):
    def __init__(self, v):
        super().__init__(); self.a, self.b = EncA(v), EncB(v)

    def forward(self, obs, goal):
        return self.b(self.a(obs, goal))


def exp(mod, args, ins, outs, name, ref):
    p = os.path.join(OUT, name)
    torch.onnx.export(mod, args, p, input_names=ins, output_names=outs, opset_version=16, do_constant_folding=True, dynamo=False)
    s, ok = onnxsim.simplify(onnx.load(p)); assert ok
    onnx.save(s, p)
    import onnxruntime as ort
    o = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, {n: x.numpy() for n, x in zip(ins, args)})[0]
    ops = {}
    for n in s.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = {"onnx": name, "max_abs_err_vs_official": float(np.abs(o - ref).max()), "has_InstanceNorm": "InstanceNormalization" in ops,
            "n_nodes": len(s.graph.node)}
    print(json.dumps(info)); return info


def main():
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(0)
    m, _ = load_model("nomad"); no_swish(m)
    obs, goal = torch.randn(1, 12, 96, 96), torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        ref = encode(m.vision_encoder, obs, goal, torch.ones(1, dtype=torch.long)).numpy()
        v = m.vision_encoder; swap_gn(v); v.eval()
        tok = EncA(v)(obs, goal)
        log = [exp(EncFull(v).eval(), (obs, goal), ["obs", "goal"], ["cond"], "nomad_enc_explore_b1_gnd.onnx", ref),
               exp(EncA(v).eval(), (obs, goal), ["obs", "goal"], ["tokens"], "nomad_encA_b1_gnd.onnx", tok.numpy()),
               exp(EncB(v).eval(), (tok,), ["tokens"], ["cond"], "nomad_encB_b1_gnd.onnx", ref)]
    json.dump(log, open(os.path.join(OUT, "export_enc_v2_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
