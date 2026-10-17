"""Alternative full-sequence export of the SHD spiking models with an ONNX `Loop` (TorchScript).

Reason: the unrolled T=1000 graph (~28k nodes) is very slow to parse in TensorRT 8.5.
Same math as shd.SRNN in export mode (hard threshold); the vanilla LIF-SRNN is the
special case branch=1, beta=0. The "t > 10" readout window is a constant 0/1 weight vector.
Usage: python shd_loop_export.py {dhsrnn_n64_b8|lif_srnn_n64}
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, OPSET  # noqa: E402
from shd import N_IN, T_STEPS, SRNN  # noqa: E402


class SNNLoop(nn.Module):
    def __init__(self, m: SRNN):
        super().__init__()
        n = m.n
        br = m.branch if m.branch else 1
        W = m.dense.weight.detach()
        self.register_buffer("w_in", W[:, :N_IN].t().contiguous())            # 700 x n*br
        self.register_buffer("b_in", m.dense.bias.detach().clone())
        self.register_buffer("w_rec", W[:, N_IN:N_IN + n].t().contiguous())   # n x n*br
        beta = torch.sigmoid(m.tau_n.detach()) if m.branch else torch.zeros(n, 1)
        self.register_buffer("beta", beta.unsqueeze(0))                       # 1 x n x br
        self.register_buffer("omb", 1.0 - beta.unsqueeze(0))
        alpha = torch.sigmoid(m.tau_m.detach()).unsqueeze(0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("oma", 1.0 - alpha)
        self.register_buffer("w_ro", m.readout.weight.detach().t().contiguous())
        self.register_buffer("b_ro", m.readout.bias.detach().clone())
        a_ro = torch.sigmoid(m.ro_tau.detach()).unsqueeze(0)
        self.register_buffer("a_ro", a_ro)
        self.register_buffer("oma_ro", 1.0 - a_ro)
        self.register_buffer("win", (torch.arange(T_STEPS) > 10).float())
        self.n, self.br, self.vth = n, br, float(m.vth)

    def forward(self, x):
        # T x 1 x n*br, so that the loop body indexes axis 0 with a single Gather
        # (TRT 8.5 rejects the Slice with non-constant axes that xp[:, t] produces)
        xp = (torch.matmul(x, self.w_in) + self.b_in).transpose(0, 1)
        d = torch.zeros(1, self.n, self.br)
        mem = torch.zeros(1, self.n)
        spk = torch.zeros(1, self.n)
        ro = torch.zeros(1, 20)
        out = torch.zeros(1, 20)
        for t in range(xp.size(0)):
            cur = xp[t] + torch.matmul(spk, self.w_rec)
            d = self.beta * d + self.omb * cur.view(1, self.n, self.br)
            cur2 = d.sum(2)
            mem = mem * self.alpha + self.oma * cur2 - self.vth * spk
            spk = (mem > self.vth).float()
            ro = ro * self.a_ro + self.oma_ro * (torch.matmul(spk, self.w_ro) + self.b_ro)
            out = out + self.win[t] * torch.softmax(ro, 1)
        return out


def main():
    name = sys.argv[1]
    m = SRNN(64, 8 if name.startswith("dhsrnn") else 0)
    m.load_state_dict(torch.load(f"{MODELS}/{name}.pt", map_location="cpu"))
    m.eval().export_mode = True
    loop = torch.jit.script(SNNLoop(m).eval())
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
    info = {"onnx": path, "nodes_top": len(onnx.load(path).graph.node),
            "max_abs_err_ort_vs_unrolled_torch": float(np.abs(got - ref).max()),
            "max_abs_err_script_vs_torch": float(np.abs(ref_loop - ref).max())}
    print(json.dumps(info))
    meta_p = f"{MODELS}/{name}.json"
    meta = json.load(open(meta_p))
    meta["export_seqloop"] = info
    json.dump(meta, open(meta_p, "w"), indent=1)


if __name__ == "__main__":
    main()
