"""Deployment variants of the trained active-dendrites model (no training).

Both variants take only the image x (1x784) and include the nearest-prototype task inference
inside the graph (10 prototypes; argmin_t ||p_t||^2 - 2 x.p_t, which equals argmin of the
Euclidean distance used in training / evaluation because ||x||^2 is common to all t):
  * active_dend_k{k}_proto  : naive graph (same as active_dend_k{k}) + prototype lookup, i.e.
                              the dendritic segments are recomputed for every input.
  * active_dend_k{k}_cached : "gates cached per task": for each of the 10 prototypes the
                              selected-segment gate sigmoid(u_j*.c) of both layers is
                              precomputed (2 x 10 x 2048 values); at run time only the
                              prototype lookup, a Gather of the gate rows and the feed-forward
                              part (masked dense weights, k-WTA) are executed.
The gate of a layer depends only on the context vector, and the context is always one of the
10 prototypes, so both variants are exact re-writes of the trained model (checked below on the
pMNIST evaluation subset against the original two-input graph).
Usage: python ad_cached_export.py {4|8}
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from active_dendrites import ADNet, MNIST_MEAN, MNIST_STD, N_TASKS, kwta  # noqa: E402
from common import MODELS, T1, export_onnx, load_vision  # noqa: E402


class ProtoLookup(nn.Module):
    def __init__(self, protos):
        super().__init__()
        self.register_buffer("pt", protos.t().contiguous())                 # 784 x 10
        self.register_buffer("pn", (protos ** 2).sum(1, keepdim=True).t())  # 1 x 10

    def forward(self, x):  # -> task index (B,)
        return torch.argmin(self.pn - 2.0 * torch.matmul(x, self.pt), dim=1)


class ADProto(nn.Module):
    def __init__(self, m, protos):
        super().__init__()
        self.m, self.look = m, ProtoLookup(protos)
        self.register_buffer("protos", protos)

    def forward(self, x):
        return self.m(x, self.protos[self.look(x)])


class ADCached(nn.Module):
    def __init__(self, m, protos):
        super().__init__()
        self.look = ProtoLookup(protos)
        with torch.no_grad():
            g = []
            for layer in (m.l1, m.l2):
                d = layer.seg(protos).view(N_TASKS, layer.n_out, layer.k)
                idx = d.abs().argmax(dim=2, keepdim=True)
                g.append(torch.sigmoid(torch.gather(d, 2, idx).squeeze(2)))
        self.register_buffer("g1", g[0])   # 10 x 2048
        self.register_buffer("g2", g[1])
        self.l1, self.l2, self.out, self.kw = m.l1.lin, m.l2.lin, m.out, m.kw

    def forward(self, x):
        t = self.look(x)
        h = kwta(self.l1(x) * self.g1[t], self.kw)
        h = kwta(self.l2(h) * self.g2[t], self.kw)
        return self.out(h)


def main():
    k = int(sys.argv[1])
    name = f"active_dend_k{k}"
    m = ADNet(k)
    m.load_state_dict(torch.load(f"{MODELS}/{name}.pt", map_location="cpu"))
    m.eval()
    d = load_vision("mnist", "cpu")
    xtr = (d["train"][0] - MNIST_MEAN) / MNIST_STD
    g = torch.Generator().manual_seed(1234)
    perms = [torch.randperm(784, generator=g) for _ in range(N_TASKS)]
    protos = torch.stack([xtr[:, p].mean(0) for p in perms])
    ev = np.load(f"{T1}/data/eval_pmnist.npz")   # device_acc.py subset: 1000 test images x 10 tasks
    xs = torch.tensor(ev["x"][:, 0]); cs = torch.tensor(ev["context"][:, 0])
    base_meta = json.load(open(f"{MODELS}/{name}.json"))
    with torch.no_grad():
        ref = m(xs, cs).argmax(1)
    for tag, mod in (("proto", ADProto(m, protos)), ("cached", ADCached(m, protos))):
        mod.eval()
        with torch.no_grad():
            pred = mod(xs).argmax(1)
            max_diff = float((mod(xs[:2000]) - m(xs[:2000], cs[:2000])).abs().max())
        out = f"{name}_{tag}"
        info = export_onnx(mod, (xs[:1],), f"{MODELS}/{out}.onnx", ["x"], ["logits"])
        meta = {k2: base_meta[k2] for k2 in ("group", "dataset", "model", "segments", "seed", "test_acc",
                                             "bwt", "task_inference_acc", "params_effective", "params_stored")}
        meta.update({"name": out, "derived_from": name, "variant": tag,
                     "argmax_agreement_vs_two_input_model_eval_subset": float((pred == ref).float().mean()),
                     "max_abs_logit_diff_vs_two_input_model_2000": max_diff, "export": info})
        if tag == "cached":
            seg = sum(l.seg.weight.numel() for l in (m.l1, m.l2))
            meta["params_stored_deployed"] = base_meta["params_stored"] - seg + 2 * N_TASKS * 2048 + 10 * 785
            meta["note"] = ("segment weights replaced by 2 x 10 x 2048 cached gates (+ 10 prototypes); "
                            "per-inference MACs: prototype lookup 7.8k + feed-forward (dense-stored, 50% zeros)")
        else:
            meta["params_stored_deployed"] = base_meta["params_stored"] + 10 * 785
        json.dump(meta, open(f"{MODELS}/{out}.json", "w"), indent=1)
        print(out, json.dumps({k2: meta[k2] for k2 in ("argmax_agreement_vs_two_input_model_eval_subset",
                                                      "max_abs_logit_diff_vs_two_input_model_2000")}))


if __name__ == "__main__":
    main()
