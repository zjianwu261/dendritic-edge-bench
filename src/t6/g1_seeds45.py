"""PLAN §3g-16 item 3 / §3g-17 (b): add G1 seeds 4, 5 (queue g1s45_*, --no_export) to the 3-seed table.
Same rule as pareto.py accuracy(): final-epoch test_acc from models/g1/<cfg>_t<seed>.json, mean, std ddof=1.
Configs = the 42 g1s45_* queue lines (all G1 families except dann24); the queue name maps to cfg by
dannG_ -> dannG_p4_ and wd5e-4 -> wd0.0005 (dann_g.py formats wd with :g).  Writes experiments/T6/g1_seeds45.csv only (existing files untouched); prints checks.
  python src/t6/g1_seeds45.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import MODELS, T6  # noqa: E402

QUEUE = f"{T6}/queue/tasks.txt"
REF = f"{T6}/fmax/pareto_g1.csv"   # 3-seed G1 table written by pareto.py --lat fmax


def family(cfg):
    if cfg.startswith(("mlpreg", "mlp", "cnn")):
        return cfg.split("_")[0]
    return "dannG_sp24" if cfg.endswith("sp24") else "dannG"


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def main():
    names = [l.split("|")[0].strip() for l in open(QUEUE) if l.startswith("g1s45_")]
    cfgs = [n[len("g1s45_"):].replace("dannG_", "dannG_p4_", 1).replace("_wd5e-4", "_wd0.0005") for n in names]   # dann_g.py names wd with :g
    ref = {r["cfg"]: r for r in csv.DictReader(open(REF))}
    rows, missing = [], []
    for cfg in cfgs:
        acc, val = {}, {}
        for s in range(1, 6):
            p = f"{MODELS}/g1/{cfg}_t{s}.json"
            if not os.path.exists(p):
                missing.append(f"{cfg}_t{s}")
                continue
            m = json.load(open(p))
            assert m["cfg"] == cfg and m["args"]["seed"] == s, p
            acc[s], val[s] = m["test_acc"], m.get("val_acc_last")
        a3 = [acc[s] for s in (1, 2, 3) if s in acc]
        a5 = [acc[s] for s in range(1, 6) if s in acc]
        r = {"cfg": cfg, "family": family(cfg), "n_seeds": len(a5),
             "params_effective": json.load(open(f"{MODELS}/g1/{cfg}_t1.json"))["params_effective"],
             "acc3_mean": round(float(np.mean(a3)), 4), "acc3_std": round(sd(a3), 4),
             "acc_s4": acc.get(4, ""), "acc_s5": acc.get(5, ""),
             "acc5_mean": round(float(np.mean(a5)), 4), "acc5_std": round(sd(a5), 4),
             "delta5_3_pp": round(100 * (np.mean(a5) - np.mean(a3)), 3),
             "val3_mean": round(float(np.mean([val[s] for s in (1, 2, 3)])), 4),
             "val5_mean": round(float(np.mean([val[s] for s in range(1, 6) if s in val])), 4),
             "ref3_acc_mean": ref.get(cfg, {}).get("acc_mean", "")}
        rows.append(r)
    out = f"{T6}/g1_seeds45.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["family"], r["params_effective"])))
    print(f"{len(cfgs)} configs -> {out}; missing: {missing or 'none'}")
    bad = [r["cfg"] for r in rows if r["ref3_acc_mean"] != "" and abs(float(r["ref3_acc_mean"]) - r["acc3_mean"]) > 1e-4]
    print("seeds 1-3 mismatch vs fmax/pareto_g1.csv:", bad or "none",
          "| not in ref:", [r["cfg"] for r in rows if r["ref3_acc_mean"] == ""] or "none")
    d = max(rows, key=lambda r: abs(r["delta5_3_pp"]))
    print(f"max |delta| {d['delta5_3_pp']} pp at {d['cfg']}; mean |delta| "
          f"{np.mean([abs(r['delta5_3_pp']) for r in rows]):.3f} pp")
    for k in ("acc3_mean", "acc5_mean"):
        fam = {}
        for r in rows:
            fam.setdefault(r["family"], []).append((r[k], r["cfg"]))
        print(k, {f: (max(v)[0], max(v)[1], min(v)[0]) for f, v in sorted(fam.items())})
        best = sorted(rows, key=lambda r: -r[k])[:5]
        print("  top5", [(r["cfg"], r[k]) for r in best])
    # rank agreement
    from scipy.stats import spearmanr
    print("spearman acc3 vs acc5:", spearmanr([r["acc3_mean"] for r in rows], [r["acc5_mean"] for r in rows]))
    for r in sorted(rows, key=lambda r: -abs(r["delta5_3_pp"]))[:8]:
        print(r["cfg"], r["acc3_mean"], r["acc3_std"], r["acc_s4"], r["acc_s5"], r["acc5_mean"], r["delta5_3_pp"])


if __name__ == "__main__":
    main()
