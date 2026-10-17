"""SHD training with a held-out validation split; epoch / config chosen on
validation only (the test set is evaluated every epoch for logging, never used for selection).

* Validation = 10 % of the SHD *training* set, stratified by class, fixed seed 2026 (the
  split is saved to results/T1/data/shd_val_split.json). Note: SHD test speakers are
  disjoint from train speakers, the val split is not (same speakers as train), so val
  accuracy is optimistic w.r.t. test.
* DH-SRNN / LIF-SRNN: official recipe as in shd.py (Adam 1e-2 / tau 2e-2, StepLR(20, .5),
  batch 100, 100 epochs), trained on the 90 % split; reported model = best-val epoch.
* GRU: hyper-parameters exposed for a small grid (hidden, lr, dropout, weight decay);
  dropout is applied to the input spikes and to the GRU outputs before the readout
  (single-layer nn.GRU has no internal dropout); weight decay = AdamW (decoupled).
  Loss as in shd.py (CE on the time-averaged logits, t > 10); grad clip 1.0; StepLR(epochs/5, .5).
Every run writes results/T1/logs/shd_val/<name>.log and a meta json; the best-val state
dict is saved as results/T1/models/shd_val/<name>.pt.
Usage: python shd_val.py --model {dhsrnn,srnn,gru} [--hidden 64 --lr 3e-3 --dropout .3 --wd 1e-2]
       [--seed 0] [--epochs 100] [--mem_frac 0.12]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, MODELS, T1, set_seed  # noqa: E402
from shd import N_CLS, N_IN, T_STEPS, SRNN, GRUNet, gru_hidden_for, load_shd, make_batch  # noqa: E402

OUTM = f"{MODELS}/shd_val"
OUTL = f"{T1}/logs/shd_val"
SPLIT = f"{T1}/data/shd_val_split.json"


def val_split(y):
    if os.path.exists(SPLIT):
        return json.load(open(SPLIT))["val_ids"]
    rng = np.random.RandomState(2026)
    y = y.cpu().numpy()
    val = []
    for c in range(N_CLS):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val += idx[:int(round(0.1 * len(idx)))].tolist()
    val = sorted(val)
    json.dump({"seed": 2026, "fraction": 0.1, "stratified": True, "n_val": len(val), "val_ids": val},
              open(SPLIT, "w"))
    return val


class GRUNetD(GRUNet):
    """GRUNet with input / output dropout (state_dict identical to GRUNet)."""

    def __init__(self, h, p=0.0):
        super().__init__(h)
        self.p = p

    def forward(self, x):
        x = F.dropout(x, self.p, self.training)
        y, _ = self.gru(x)
        y = F.dropout(y[:, 11:], self.p, self.training)
        return self.readout(y).mean(1)


def evaluate(model, data, ids, bs=50):
    flat, offs, y = data
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs]
            correct += (model(make_batch(flat, offs, b)).argmax(1) == y[b.cuda()]).sum().item()
    return correct / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dhsrnn", "srnn", "gru"], required=True)
    ap.add_argument("--hidden", type=int, default=0, help="GRU hidden size (0 = parameter-matched)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem_frac", type=float, default=0.12)
    ap.add_argument("--micro", type=int, default=1,
                    help="split each batch of 100 into this many micro-batches (gradient accumulation; "
                         "same gradient up to dropout draws) to bound GPU memory")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    set_seed(args.seed)
    # FP32 matmuls (the v0.5 run used TF32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(OUTM, exist_ok=True)
    os.makedirs(OUTL, exist_ok=True)

    t0 = time.time()
    train, test = load_shd("train"), load_shd("test")
    val_ids = torch.tensor(val_split(train[2]))
    is_val = torch.zeros(len(train[2]), dtype=torch.bool)
    is_val[val_ids] = True
    tr_ids = torch.where(~is_val)[0]
    te_ids = torch.arange(len(test[2]))

    ref_params = SRNN(64, 8, seed=args.seed).effective_params()
    if args.model == "dhsrnn":
        model, name = SRNN(64, 8, seed=args.seed), "dhsrnn_n64_b8"
    elif args.model == "srnn":
        model, name = SRNN(64, 0, seed=args.seed), "lif_srnn_n64"
    else:
        h = args.hidden or gru_hidden_for(ref_params)
        model = GRUNetD(h, args.dropout)
        name = f"gru_h{h}_lr{args.lr:g}_do{args.dropout:g}_wd{args.wd:g}"
    name += f"_s{args.seed}{args.tag}"
    model.cuda()
    log = open(f"{OUTL}/{name}.log", "w")

    def pr(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    if args.model == "gru":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        base = [model.dense.weight, model.dense.bias, model.readout.weight, model.readout.bias]
        groups = [{"params": base, "lr": 1e-2},
                  {"params": [model.ro_tau], "lr": 2e-2}, {"params": [model.tau_m], "lr": 2e-2}]
        if args.model == "dhsrnn":
            groups.append({"params": [model.tau_n], "lr": 2e-2})
        opt = torch.optim.Adam(groups, lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 5), gamma=0.5)
    bs = 100
    flat, offs, y = train
    g = torch.Generator().manual_seed(args.seed)
    hist, best = [], (-1.0, -1)
    pr(f"{name}: train {len(tr_ids)} val {len(val_ids)} test {len(te_ids)}; params "
       f"{model.effective_params()}; args {vars(args)}")
    for ep in range(args.epochs):
        model.train()
        te0 = time.time()
        perm = tr_ids[torch.randperm(len(tr_ids), generator=g)]
        tl, tc, nb = 0.0, 0, 0
        for i in range(0, len(perm), bs):
            ids = perm[i:i + bs]
            yb = y[ids.cuda()]
            opt.zero_grad(set_to_none=True)
            outs, lsum = [], 0.0
            for sub in ids.chunk(args.micro):
                o = model(make_batch(flat, offs, sub))
                l = F.cross_entropy(o, y[sub.cuda()], reduction="sum") / len(ids)
                l.backward()
                outs.append(o.detach())
                lsum += l.item()
            out, loss = torch.cat(outs), torch.tensor(lsum)
            if args.model == "gru":
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if args.model != "gru":
                model.apply_mask()
            tl += loss.item()
            tc += (out.argmax(1) == yb).sum().item()
            nb += 1
        sched.step()
        va = evaluate(model, train, val_ids)
        te = evaluate(model, test, te_ids)
        hist.append({"ep": ep + 1, "train": tc / len(perm), "val": va, "test": te})
        if va > best[0]:  # strict: earliest epoch among ties
            best = (va, ep + 1)
            torch.save(model.state_dict(), f"{OUTM}/{name}.pt")
        pr(f"{name} ep {ep + 1} loss {tl / nb:.4f} train {tc / len(perm):.4f} val {va:.4f} test {te:.4f} "
           f"({time.time() - te0:.0f}s)")
    bh = hist[best[1] - 1]
    meta = {"name": name, "model": args.model, "args": vars(args), "dataset": "SHD",
            "split": "train 90% / val 10% (stratified, seed 2026) / official test",
            "selection": "best validation accuracy epoch (earliest among ties)",
            "best_epoch": best[1], "val_acc": bh["val"], "test_acc": bh["test"], "train_acc": bh["train"],
            "last_epoch_test_acc": hist[-1]["test"], "last_epoch_val_acc": hist[-1]["val"],
            "max_test_acc_any_epoch_NOT_FOR_SELECTION": max(h["test"] for h in hist),
            "params_effective": model.effective_params(),
            "params_stored": sum(p.numel() for p in model.parameters()),
            "tf32": False, "train_minutes": round((time.time() - t0) / 60, 1), "history": hist}
    json.dump(meta, open(f"{OUTM}/{name}.json", "w"), indent=1)
    pr(json.dumps({k: v for k, v in meta.items() if k != "history"}))


if __name__ == "__main__":
    main()
