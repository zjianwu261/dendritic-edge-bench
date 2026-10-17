"""Offline check of context-inferer variants on the dumped v3 features (ctx_feat_diag.py), VALIDATION split
only for the choice; test numbers are printed for information but not used for any decision.
Sequential protocol as in cl_dend2: standardisation from the first task's prototype samples, prototypes added
in task order, accuracy = nearest prototype of each seen task's samples after the last task."""
import json, math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP  # noqa
from ctx_feat_diag import ORDERS  # noqa
import ctx_infer as CI  # noqa

D = os.path.join(EXP, "diag", "v3ctx")
raw = {n: torch.load(os.path.join(D, f"raw_{n.replace(':', '_')}.pt")) for n in ORDERS["main"]}
GROUPS = {"feat": slice(0, 256), "hist": slice(256, 272), "low": slice(272, 274)}


def run(order, variant, q=0.95, split="val"):
    cols = {"v2": ["feat", "hist"], "v3": ["feat", "hist", "low"], "v3_nolow": ["feat", "hist"],
            "v3_feat": ["feat"], "v3_lowonly": ["hist", "low"]}[variant["set"]]
    idx = torch.cat([torch.arange(256 + 0)[GROUPS[c]] if False else torch.arange(274)[GROUPS[c]] for c in cols])
    def sel(z): return z[:, idx]
    first = sel(raw[order[0]][f"proto_i0"])
    mu0, sd = first.mean(0), first.std(0)
    if variant["group"]:
        fl = torch.cat([torch.full((GROUPS[c].stop - GROUPS[c].start,), CI.FLOOR[c]) for c in cols])
        sd0 = torch.maximum(sd, fl)
        gw = torch.cat([torch.full((GROUPS[c].stop - GROUPS[c].start,), 1 / math.sqrt(GROUPS[c].stop - GROUPS[c].start)) for c in cols])
    else:
        sd0, gw = sd + 1e-3, torch.ones_like(sd)
    st = lambda z: (sel(z) - mu0) / sd0 * gw  # noqa
    mus, taus, detect = [], [], []
    for i, n in enumerate(order):
        zp = st(raw[n][f"proto_i{i}"])
        if mus:
            d = torch.cdist(zp, torch.stack(mus)); detect.append(float(((d / torch.stack(taus)).min(-1).values > 1).float().mean()))
        mu = zp.mean(0); mus.append(mu)
        taus.append(torch.quantile((st(raw[n]["val"]) - mu).norm(dim=-1), q))
    M, T = torch.stack(mus), torch.stack(taus)
    acc, fn, conf = [], [], []
    for j, n in enumerate(order):
        z = st(raw[n][split]); d = torch.cdist(z, M)
        acc.append(float((d.argmin(-1) == j).float().mean())); fn.append(float(((d / T).min(-1).values > 1).float().mean()))
        conf.append(torch.bincount(d.argmin(-1), minlength=len(order)).tolist())
    return {"acc": round(sum(acc) / len(acc), 4), "acc_per_task": [round(a, 3) for a in acc], "detect": [round(x, 3) for x in detect],
            "detect_mean": round(sum(detect) / len(detect), 4), "false_new_mean": round(sum(fn) / len(fn), 4), "confusion": conf}


VARIANTS = {"A_v2_old": {"set": "v2", "group": False}, "B_v3_group": {"set": "v3", "group": True},
            "C_v3_nogroup": {"set": "v3", "group": False}, "D_v3_group_nolow": {"set": "v3_nolow", "group": True},
            "E_feat_only": {"set": "v3_feat", "group": False}, "F_lowlevel_only": {"set": "v3_lowonly", "group": True}}
if __name__ == "__main__":
    res = {}
    for split in ("val", "test"):
        for o, order in ORDERS.items():
            for vn, v in VARIANTS.items():
                r = run(order, v, split=split); res[f"{split}|{o}|{vn}"] = r
                print(f"{split:4s} {o:4s} {vn:18s} acc {r['acc']:.3f} {r['acc_per_task']} detect {r['detect_mean']:.3f} {r['detect']} false_new {r['false_new_mean']:.3f}")
    json.dump(res, open(os.path.join(D, "inferer_variants.json"), "w"), indent=1)
