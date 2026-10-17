"""GRU exported with the *same* TorchScript -> ONNX `Loop` construction as the
SNNs (shd_loop_export.py), so that "Loop implementation overhead" and "dendrite overhead" can
be separated:  GRU-op vs GRU-Loop = implementation;  GRU-Loop vs LIF-Loop vs DH-Loop = model.

Same structure as SNNLoop: the input projections W_i{r,z,n} x_t + b for all t are computed before
the loop (three T x 1 x h tensors, each indexed on axis 0 with one Gather), the loop body does the recurrent
MatMul, the gates, and the per-step readout accumulated over t > 10 with a constant 0/1 window
(the model output is the mean of the readout logits over t > 10, as in shd.GRUNet).
Gate order / formula = torch.nn.GRU (r, z, n; n = tanh(W_in x + b_in + r * (W_hn h + b_hn))).
Usage: python gru_loop_export.py <state_dict.pt> <hidden> <out_name>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, OPSET  # noqa: E402
from shd import N_CLS, N_IN, T_STEPS, GRUNet  # noqa: E402


class GRULoop(nn.Module):
    def __init__(self, m: GRUNet):
        super().__init__()
        g = m.gru
        h = g.hidden_size
        W, B = g.weight_ih_l0.detach(), g.bias_ih_l0.detach()
        U, C = g.weight_hh_l0.detach(), g.bias_hh_l0.detach()
        for i, k in enumerate("rzn"):  # separate gate matrices: no Slice ops inside the loop body
            self.register_buffer(f"w_i{k}", W[i * h:(i + 1) * h].t().contiguous())   # 700 x h
            self.register_buffer(f"b_i{k}", B[i * h:(i + 1) * h].clone())
            self.register_buffer(f"w_h{k}", U[i * h:(i + 1) * h].t().contiguous())   # h x h
            self.register_buffer(f"b_h{k}", C[i * h:(i + 1) * h].clone())
        self.register_buffer("w_ro", m.readout.weight.detach().t().contiguous())
        self.register_buffer("b_ro", m.readout.bias.detach().clone())
        self.register_buffer("win", (torch.arange(T_STEPS) > 10).float() / float(T_STEPS - 11))
        self.register_buffer("h0", torch.zeros(1, h))
        self.register_buffer("o0", torch.zeros(1, N_CLS))

    def forward(self, x):
        xr = (torch.matmul(x, self.w_ir) + self.b_ir).transpose(0, 1)   # T x 1 x h
        xz = (torch.matmul(x, self.w_iz) + self.b_iz).transpose(0, 1)
        xn = (torch.matmul(x, self.w_in) + self.b_in).transpose(0, 1)
        h = self.h0
        out = self.o0
        for t in range(xr.size(0)):
            # explicit (1, h) reshapes: TRT 8.5 otherwise sees a (1, 1, h) loop-carried value
            a_r, a_z, a_n = xr[t].reshape(1, -1), xz[t].reshape(1, -1), xn[t].reshape(1, -1)
            r = torch.sigmoid(a_r + torch.matmul(h, self.w_hr) + self.b_hr)
            z = torch.sigmoid(a_z + torch.matmul(h, self.w_hz) + self.b_hz)
            n = torch.tanh(a_n + r * (torch.matmul(h, self.w_hn) + self.b_hn))
            h = (1.0 - z) * n + z * h
            h = h.reshape(1, -1)
            out = (out + self.win[t] * (torch.matmul(h, self.w_ro) + self.b_ro)).reshape(1, -1)
        return out


def main():
    pt, hidden, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    m = GRUNet(hidden)
    m.load_state_dict(torch.load(pt, map_location="cpu"))
    m.eval()
    loop = torch.jit.script(GRULoop(m).eval())
    x = torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.02)
    path = f"{MODELS}/{name}_seqloop.onnx"
    torch.onnx.export(loop, (x,), path, opset_version=OPSET, input_names=["x"], output_names=["out"],
                      dynamo=False)
    import onnx
    import onnxruntime as ort
    with torch.no_grad():
        ref = m(x).numpy()
        ref_loop = loop(x).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    info = {"onnx": path, "nodes_top": len(onnx.load(path).graph.node), "source_state_dict": pt,
            "max_abs_err_ort_vs_torch_gru": float(np.abs(got - ref).max()),
            "max_abs_err_script_vs_torch_gru": float(np.abs(ref_loop - ref).max())}
    print(json.dumps(info))
    meta_p = f"{MODELS}/{name}.json"
    meta = json.load(open(meta_p))
    meta["export_seqloop"] = info
    json.dump(meta, open(meta_p, "w"), indent=1)


if __name__ == "__main__":
    main()
