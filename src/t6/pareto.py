"""T6 Pareto fronts and latency-matched comparison.
  python pareto.py g1|g2|g3
Accuracy: mean / std over the trained seeds of each config (G1: final-epoch test acc as T1;
G2: test acc at the best-validation epoch, random validation split only; G3: final mean acc over the 10 tasks).
Latency: Orin #1 LUT (lut/lut_<task>.csv, TensorRT 8.5 FP16 sync e2e p50, batch 1), per config the
lowest p50 over its implementations / builds (the chosen impl + build are written out).
Outputs: results/T6/pareto_<task>.csv (all configs, `pareto` flag per family and overall),
         results/T6/pareto_<task>.png, results/T6/latmatch_<task>.csv.
Latency matching: for every dendritic config, the best baseline whose p50 <= the dendritic p50 * (1+tol)
(tol = 5 %, about the run-to-run spread), and the reverse (fastest baseline reaching the same accuracy).
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, MODELS, T6  # noqa: E402

BASE = {"g1": ("mlp",), "g2": ("gru",), "g3": ("mlp",)}
TOL = 0.05


def family(task, cfg):
    if cfg.startswith("mlp") or cfg.startswith("gru"):
        return cfg.split("_")[0]
    if task == "g1":
        return "dannG_sp24" if cfg.endswith("sp24") else cfg.split("_")[0]
    if task == "g2":
        return "dhssm_" + cfg.split("_")[-1] + ("_nb1" if "_b1_" in cfg else "")
    return "adc_" + cfg.split("_")[-1]


def accuracy(task):
    accs = {}
    for f in glob.glob(f"{MODELS}/{task}/*.json"):
        m = json.load(open(f))
        if "_pilot" in m["name"] or m["args"].get("split") == "speaker":
            continue   # speaker-split (fair-protocol) G2 runs are summarised by g2_fair_summary.py
        cfg = m["cfg"] + (m["args"].get("tag", "") if task == "g2" else "")
        accs.setdefault(cfg, []).append((m["test_acc"], m.get("val_acc", m.get("val_acc_last")), m.get("params_effective")))
    return accs


LATCOL = "p50_ms"   # --lat fmax -> p50_fmax_ms (G1)


def latency(task):
    lat = {}
    p = f"{LUT}/lut_{task}.csv"
    if not os.path.exists(p):
        return lat
    for r in csv.DictReader(open(p)):
        if r.get(LATCOL, "") in ("", None):
            continue
        v = float(r[LATCOL])
        if r["cfg"] not in lat or v < lat[r["cfg"]][0]:
            lat[r["cfg"]] = (v, r["impl"], r["build"], r.get("engine_file_MB", ""))
    return lat


def pareto_flags(pts):
    """pts: list of (lat, acc); a point is on the front if no other point is faster-or-equal AND better."""
    out = []
    for i, (l, a) in enumerate(pts):
        dom = any((l2 <= l and a2 > a) or (l2 < l and a2 >= a) for j, (l2, a2) in enumerate(pts) if j != i)
        out.append(not dom)
    return out


def main(task):
    accs, lat = accuracy(task), latency(task)
    rows = []
    for cfg, v in sorted(accs.items()):
        lcfg = cfg.split("_r")[0] if task == "g2" and cfg.startswith("gru") else cfg
        if task == "g3":
            lcfg = lcfg.replace("_k4", "")   # segments do not exist in the cached deployment graph
        a = np.array([x[0] for x in v])
        va = [x[1] for x in v if x[1] is not None]
        r = {"cfg": cfg, "family": family(task, cfg), "n_seeds": len(v), "acc_mean": round(float(a.mean()), 4),
             "acc_std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
             "val_mean": round(float(np.mean(va)), 4) if va else "", "params_effective": v[0][2]}
        if lcfg in lat:
            r.update({"p50_ms": lat[lcfg][0], "impl": lat[lcfg][1], "build": lat[lcfg][2], "engine_MB": lat[lcfg][3]})
        rows.append(r)
    if task == "g2":   # GRU: keep the regularisation setting with the better mean validation acc per h
        best = {}
        for r in rows:
            if r["family"] == "gru":
                h = r["cfg"].split("_r")[0]
                if h not in best or (r["val_mean"] or 0) > (best[h]["val_mean"] or 0):
                    best[h] = r
        rows = [r for r in rows if r["family"] != "gru" or best[r["cfg"].split("_r")[0]] is r]
    have = [r for r in rows if "p50_ms" in r and r["n_seeds"] >= 3]
    for fam in sorted({r["family"] for r in have}) + ["ALL"]:
        sub = [r for r in have if fam == "ALL" or r["family"] == fam]
        for r, f in zip(sub, pareto_flags([(r["p50_ms"], r["acc_mean"]) for r in sub])):
            r["pareto_" + ("all" if fam == "ALL" else "family")] = f
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f"{T6}/pareto_{task}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["family"], r.get("p50_ms", 1e9))))
    # latency-matched comparison
    base = [r for r in have if r["family"] in BASE[task]]
    lm = []
    for r in have:
        if r["family"] in BASE[task] or not r.get("pareto_family"):
            continue
        cand = [b for b in base if b["p50_ms"] <= r["p50_ms"] * (1 + TOL)]
        bb = max(cand, key=lambda b: b["acc_mean"]) if cand else None
        reach = [b for b in base if b["acc_mean"] >= r["acc_mean"]]
        fb = min(reach, key=lambda b: b["p50_ms"]) if reach else None
        lm.append({"cfg": r["cfg"], "family": r["family"], "p50_ms": r["p50_ms"], "acc": r["acc_mean"], "acc_std": r["acc_std"],
                   "best_baseline_within_lat": bb["cfg"] if bb else "", "baseline_p50_ms": bb["p50_ms"] if bb else "",
                   "baseline_acc": bb["acc_mean"] if bb else "", "baseline_acc_std": bb["acc_std"] if bb else "",
                   "delta_acc_pp": round(100 * (r["acc_mean"] - bb["acc_mean"]), 2) if bb else "",
                   "fastest_baseline_same_acc": fb["cfg"] if fb else "none reaches",
                   "its_p50_ms": fb["p50_ms"] if fb else "",
                   "speedup_vs_it": round(fb["p50_ms"] / r["p50_ms"], 2) if fb else ""})
    if lm:
        with open(f"{T6}/latmatch_{task}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lm[0]))
            w.writeheader()
            w.writerows(sorted(lm, key=lambda r: r["p50_ms"]))
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fam in sorted({r["family"] for r in have}):
            sub = sorted([r for r in have if r["family"] == fam], key=lambda r: r["p50_ms"])
            ax.errorbar([r["p50_ms"] for r in sub], [100 * r["acc_mean"] for r in sub],
                        yerr=[100 * r["acc_std"] for r in sub], fmt="o", ms=4, label=fam, alpha=.8)
            fr = [r for r in sub if r.get("pareto_family")]
            ax.plot([r["p50_ms"] for r in fr], [100 * r["acc_mean"] for r in fr], "-", alpha=.5)
        ax.set_xscale("log")
        ax.set_xlabel("Orin #1 TensorRT FP16 sync e2e p50 (ms, batch 1)")
        ax.set_ylabel("test accuracy (%) mean +- std, 3 seeds")
        ax.set_title(f"T6 {task.upper()} accuracy vs measured Orin latency")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{T6}/pareto_{task}.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot failed", e)
    print(f"{len(rows)} configs ({len(have)} with latency and >= 3 seeds) -> pareto_{task}.csv; {len(lm)} latmatch rows")
    for r in sorted(lm, key=lambda r: r["p50_ms"]):
        print(r)


if __name__ == "__main__":
    if "--lat" in sys.argv and sys.argv[sys.argv.index("--lat") + 1] == "fmax":
        LATCOL = "p50_fmax_ms"
        T6 = T6 + "/fmax"
        os.makedirs(T6, exist_ok=True)
    main(sys.argv[1])
