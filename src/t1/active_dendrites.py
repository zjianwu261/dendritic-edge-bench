"""T1 group 3: active-dendrites MLP (k segments) vs same-parameter MLP, permuted MNIST, 10 tasks.

Re-implementation of Numenta's DendriticMLP + PROTOTYPE_10 config (Iyer et al. 2022), checked
against nupic.research (src/third_party/nupic.research, commit 7a287be):
  * 784 -> 2048 -> 2048 -> 10, single shared head; AbsoluteMaxGatingDendriticLayer:
    j* = argmax_j |u_j . c|, y = (W x + b) * sigmoid(u_j* . c); k-WTA 5 % (no boosting).
  * feed-forward weights 50 % sparse (fixed random mask incl. output layer, re-zeroed after
    every step = RezeroWeights), dense dendrite segments (dendrite_weight_sparsity = 0).
  * "modified" init: W ~ U(+-1/sqrt(w_density * in_density * fan_in)), in_density = 0.05 after
    k-WTA; segments ~ U(+-1/sqrt(context_percent_on(0.1) * 784)).
  * inputs normalised with MNIST mean/std; context = task prototype (mean of the task's
    training inputs); at test time the nearest prototype is used (task inferred).
  * Adam lr 5e-4, 3 epochs per task, batch 256, optimizer re-created per task.
  * num_segments = k (4 or 8) instead of 10 (= number of tasks) in PROTOTYPE_10.
Baseline: dense MLP 784 -> h -> h -> 10 (ReLU), h matched to the dendritic model's effective
(non-zero) parameter count, same sequential protocol, no context.

Export: dendritic model has two inputs (x 1x784, context 1x784) -> logits (mask
implementation: sparse weights exported as dense matrices with zeros); the prototype lookup
(10x784 distances) is not part of the exported graph.
Usage: python active_dendrites.py --model {ad,mlp} --segments {4,8}
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, export_onnx, load_vision, save_meta, set_seed  # noqa: E402

HID, N_TASKS, KW, W_DENSITY, CTX_ON = 2048, 10, 0.05, 0.5, 0.1
MNIST_MEAN, MNIST_STD = 0.13062755, 0.30810780


def sparse_init_(lin, in_density, gen, density=W_DENSITY):
    """nupic `_init_sparse_weights` + SparseWeights mask (kept as a buffer)."""
    fan_in = lin.weight.shape[1]
    bound = 1.0 / math.sqrt(in_density * density * fan_in)
    with torch.no_grad():
        lin.weight.uniform_(-bound, bound)
        # SparseWeights: exactly round(density*fan_in) non-zeros per output unit
        n_on = int(round(density * fan_in))
        mask = torch.zeros_like(lin.weight)
        for i in range(mask.shape[0]):
            mask[i, torch.randperm(fan_in, generator=gen)[:n_on]] = 1
        lin.weight.mul_(mask)
    lin.register_buffer("wmask", mask)


def kwta(x, k):
    thr = torch.topk(x, k, dim=1).values[:, -1:]
    return x * (x >= thr).to(x.dtype)


class ADLayer(nn.Module):
    def __init__(self, n_in, n_out, segments, in_density, gen, dim_ctx=784, seg_scale=1.0):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        sparse_init_(self.lin, in_density, gen)
        self.n_out, self.k = n_out, segments
        self.seg = nn.Linear(dim_ctx, n_out * segments, bias=False)
        bound = seg_scale / math.sqrt(CTX_ON * dim_ctx)
        nn.init.uniform_(self.seg.weight, -bound, bound)

    def forward(self, x, c):
        y = self.lin(x)
        d = self.seg(c).view(-1, self.n_out, self.k)
        idx = d.abs().argmax(dim=2, keepdim=True)
        dsel = torch.gather(d, 2, idx).squeeze(2)
        return y * torch.sigmoid(dsel)


class ADNet(nn.Module):
    def __init__(self, segments, seg_scale=1.0, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.l1 = ADLayer(784, HID, segments, 1.0, gen, seg_scale=seg_scale)
        self.l2 = ADLayer(HID, HID, segments, KW, gen, seg_scale=seg_scale)
        self.out = nn.Linear(HID, 10)
        sparse_init_(self.out, KW, gen)
        self.kw = int(round(KW * HID))

    def rezero(self):
        with torch.no_grad():
            for lin in (self.l1.lin, self.l2.lin, self.out):
                lin.weight.mul_(lin.wmask)

    def effective_params(self):
        n = sum(p.numel() for p in self.parameters())
        for lin in (self.l1.lin, self.l2.lin, self.out):
            n -= int((lin.wmask == 0).sum())
        return n

    def forward(self, x, c):
        h = kwta(self.l1(x, c), self.kw)
        h = kwta(self.l2(h, c), self.kw)
        return self.out(h)


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(784, h), nn.Linear(h, h), nn.Linear(h, 10)

    def forward(self, x, c=None):
        return self.out(F.relu(self.l2(F.relu(self.l1(x)))))

    def rezero(self):
        pass

    def effective_params(self):
        return sum(p.numel() for p in self.parameters())


def mlp_hidden_for(params):
    a, b, c = 1, 784 + 10 + 2, 10 - params
    return int(round((-b + math.sqrt(b * b - 4 * a * c)) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ad", "mlp"], required=True)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3, help="epochs per task")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seg_scale", type=float, default=1.0, help="x default init of segment weights")
    ap.add_argument("--tag", default="")
    ap.add_argument("--match", choices=["effective", "stored"], default="effective",
                    help="MLP baseline: match the AD model's effective (non-zero) or stored parameter count")
    ap.add_argument("--no_export", action="store_true")
    args = ap.parse_args()
    dev = os.environ.get("T1_DEV", "cuda")  # T1_DEV=cpu trains on the CPU
    set_seed(args.seed)
    t0 = time.time()

    data = load_vision("mnist", dev)
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    xtr, xte = (xtr - MNIST_MEAN) / MNIST_STD, (xte - MNIST_MEAN) / MNIST_STD
    g = torch.Generator().manual_seed(1234)  # task permutations shared by all models
    perms = [torch.randperm(784, generator=g).to(dev) for _ in range(N_TASKS)]
    protos = torch.stack([xtr[:, p].mean(0) for p in perms])  # (10, 784)

    ad_ref = ADNet(args.segments, seed=args.seed)
    ad_params = ad_ref.effective_params()
    ad_stored = sum(p.numel() for p in ad_ref.parameters())
    if args.model == "ad":
        model = ADNet(args.segments, args.seg_scale, seed=args.seed)
        name = f"active_dend_k{args.segments}"
    else:
        h = mlp_hidden_for(ad_params if args.match == "effective" else ad_stored)
        model = MLP(h)
        name = f"mlp_h{h}_match_k{args.segments}" if args.match == "effective" else f"mlp_h{h}_matchstored_k{args.segments}"
    name += args.tag
    model.to(dev)

    def infer_ctx(x):  # nearest prototype
        return protos[torch.cdist(x, protos).argmin(1)]

    def eval_task(t, bs=2000):
        model.eval()
        correct, task_hit = 0, 0
        with torch.no_grad():
            for i in range(0, len(xte), bs):
                x = xte[i:i + bs][:, perms[t]]
                c = infer_ctx(x)
                task_hit += (torch.cdist(x, protos).argmin(1) == t).sum().item()
                correct += (model(x, c).argmax(1) == yte[i:i + bs]).sum().item()
        return correct / len(xte), task_hit / len(xte)

    acc_matrix = []
    for t in range(N_TASKS):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        xt = xtr[:, perms[t]]
        ct = protos[t].expand(args.bs, -1)
        model.train()
        for ep in range(args.epochs):
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(xt), args.bs):
                b = perm[i:i + args.bs]
                loss = F.cross_entropy(model(xt[b], ct[:len(b)]), ytr[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.rezero()
        row = [eval_task(j)[0] for j in range(t + 1)]
        acc_matrix.append(row)
        print(f"{name} after task {t + 1}: mean acc {sum(row) / len(row):.4f} "
              f"per-task {[round(a, 3) for a in row]}", flush=True)
    final = acc_matrix[-1]
    task_inf = [eval_task(j)[1] for j in range(N_TASKS)]
    bwt = sum(final[j] - acc_matrix[j][j] for j in range(N_TASKS - 1)) / (N_TASKS - 1)
    meta = {"name": name, "group": "active_dend_pmnist", "dataset": "permuted MNIST (10 tasks)",
            "model": args.model, "segments": args.segments, "epochs_per_task": args.epochs,
            "lr": args.lr, "seg_scale": args.seg_scale, "seed": args.seed, "test_acc": sum(final) / N_TASKS,
            "final_per_task": final, "bwt": bwt, "task_inference_acc": sum(task_inf) / N_TASKS,
            "params_effective": model.effective_params(),
            "params_stored": sum(p.numel() for p in model.parameters()),
            "ad_params": ad_params, "ad_params_stored": ad_stored, "match": args.match, "train_minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(meta), flush=True)
    if not args.no_export:
        torch.save(model.state_dict(), f"{MODELS}/{name}.pt")
        x0, c0 = xte[:1, perms[0]].cpu(), protos[:1].cpu()
        if args.model == "ad":
            meta["export"] = export_onnx(model, (x0, c0), f"{MODELS}/{name}.onnx",
                                         ["x", "context"], ["logits"])
        else:
            meta["export"] = export_onnx(model, (x0,), f"{MODELS}/{name}.onnx", ["x"], ["logits"])
    save_meta(name, meta)


if __name__ == "__main__":
    main()
