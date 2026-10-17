"""Fair-protocol summary for G1 / G3 (T6-F), selected on VALIDATION only.

Inputs: results/T6/models/g1/*_fair_t{1,2,3}.json (G1 Fashion-MNIST: dANN-G vs MLP vs small CNN; selection key val_acc_last)
        results/T6/models/g3/*_fair_t{1,2,3}.json (G3 permuted MNIST: AD-cache vs MLPctx; selection key val_acc)
Per family: every config (size x lr x wd) is averaged over its 3 seeds; rows
  kind = per_size : for each size the (lr, wd) with the best mean val (one row per size -> matched-size comparison)
  kind = family_best : the config with the best mean val in the family (headline comparison)
Test accuracy is only reported, never used to choose.  Output results/T6/fair_summary.csv.
"""
import csv
import glob
import json
import os
import re
import statistics as st

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
M = f"{ROOT}/results/T6/models"
# the small CNN joins G1 (size = c1; c2 = 2 * c1)
FAM = {"G1": [("dannG", "val_acc_last", "m"), ("mlp", "val_acc_last", "h"), ("cnn", "val_acc_last", "c1"),
              # sparse MLP: h1024 with unstructured masks; size = sparsity
              ("mlpsp_h1024_random", "val_acc_last", "sparsity"), ("mlpsp_h1024_magnitude", "val_acc_last", "sparsity")], "G3": [("adc", "val_acc", "h"), ("mlpctx", "val_acc", "h")]}


def main():
    rows = []
    for g, fams in FAM.items():
        files = sorted(glob.glob(f"{M}/{g.lower()}/*_fair_t*.json"))
        for fam, vkey, skey in fams:
            cfgs = {}
            for f in files:
                d = json.load(open(f))
                cfg = d["cfg"]
                if not re.match(rf"^{fam}_", cfg) or (fam == "mlp" and cfg.startswith("mlpctx")):
                    continue
                cfgs.setdefault(cfg, []).append(d)
            summ = []
            for cfg, ds in cfgs.items():
                a = ds[0]["args"]
                size = a.get(skey)
                summ.append({"group": g, "family": fam, "cfg": cfg, "size_key": skey, "size": size, "lr": a.get("lr"), "wd": a.get("wd"),
                             "n_seeds": len(ds), "val_mean": st.mean(d[vkey] for d in ds),
                             "test_mean": st.mean(d["test_acc"] for d in ds),
                             "test_std": st.stdev(d["test_acc"] for d in ds) if len(ds) > 1 else 0.0,
                             "bwt_mean": st.mean(d["bwt"] for d in ds) if "bwt" in ds[0] else "",
                             "params_effective": ds[0].get("params_effective"), "params_stored": ds[0].get("params_stored"),
                             "selection_key": vkey})
            for size in sorted({s["size"] for s in summ}):
                best = max([s for s in summ if s["size"] == size], key=lambda s: s["val_mean"])
                rows.append(dict(best, kind="per_size"))
            if summ:
                rows.append(dict(max(summ, key=lambda s: s["val_mean"]), kind="family_best"))
    for r in rows:
        for k in ("val_mean", "test_mean", "test_std", "bwt_mean"):
            if isinstance(r[k], float):
                r[k] = round(r[k], 4)
    keys = ["group", "kind", "family", "cfg", "size_key", "size", "lr", "wd", "n_seeds", "selection_key", "val_mean", "test_mean",
            "test_std", "bwt_mean", "params_effective", "params_stored"]
    with open(f"{ROOT}/results/T6/fair_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["group"], r["kind"], r["cfg"], r["n_seeds"], "val", r["val_mean"], "test", r["test_mean"], "+-", r["test_std"], r["params_stored"])


if __name__ == "__main__":
    main()
