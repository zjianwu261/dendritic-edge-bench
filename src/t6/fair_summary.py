"""Fair-protocol summary for G1 / G3 (T6-F), selected on VALIDATION only.

Inputs: results/T6/models/g1/*_fair_t{1,2,3}.json (G1 Fashion-MNIST: dANN-G vs MLP vs small CNN; selection key val_acc_last)
        results/T6/models/g3/*_fair_t{1,2,3}.json (G3 permuted MNIST: AD-cache vs MLPctx; selection key val_acc)
Per family: every config (size x lr x wd) is averaged over its 3 seeds; rows
  kind = per_size : for each size the (lr, wd) with the best mean val (one row per size -> matched-size comparison)
  kind = family_best : the config with the best mean val in the family (headline comparison)
Test accuracy is only reported, never used to choose.  Output results/T6/fair_summary.csv.
G1 Orin latency comes from results/T6/lut/buildvar_cited.csv (GPU locked at 624.75 MHz, median over 5 fresh engine
builds, the values of the paper's G1 table): orin_p50_ms = median_p50_ms, orin_range_5builds = range_ms,
orin_impl_build, orin_n_builds.  orin_test_acc_t1 (FP16 test accuracy of the seed-1 engine) comes from the locked
single-build session (lut/lut_g1f.csv, raw/orin1_e7locked).  G1 models without a build-variance entry (sparse MLPs,
the narrower CNNs) have empty orin_* cells.  Energy was not measured for these models.
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


E7_KEYS = ["orin_p50_ms", "orin_impl", "orin_build", "orin_range", "orin_nruns", "orin_gpu_MHz_med", "orin_gpu_MHz_set",
           "orin_test_acc_t1", "orin_clock", "orin_energy"]


def e7_orin(lut, raw, prefix="orin_", t_from="", t_to="", locked_only=False):
    """Per-config columns from one lut_g1f table (lut_g1f.csv, written by e7_g1_orin.py table after the locked
    session; clock column 'locked_...')."""
    if not os.path.exists(lut):
        return {}
    tab = list(csv.DictReader(open(lut)))
    if locked_only and not (tab and tab[0].get("clock", "").startswith("locked_")):
        return {}
    res = [json.loads(l) for l in open(f"{raw}/results.jsonl")]
    out = {}
    for r in tab:
        if r["family"] == "anchor" or not r.get("p50_ms"):
            continue
        p50 = float(r["p50_ms"])
        if r["cfg"] in out and out[r["cfg"]][prefix + "p50_ms"] <= p50:
            continue
        b = r["build"]
        t0 = t_from or r.get("session_from", "")
        runs = [x for x in res if x["label"] == r["label"] and x["build"] == b and x["onnx_sha256"].startswith(r["onnx_sha256"])
                and x["time"] >= t0 and (not t_to or x["time"] < t_to)]
        if locked_only:
            runs = [x for x in runs if x["gpu_freq_timed_frac_at_max"] == 1.0]
        mhz = sorted(x["gpu_freq_hz_timed_min_med_max"][1] / 1e6 for x in runs)
        locked = bool(runs) and all(x["gpu_freq_timed_frac_at_max"] == 1.0 for x in runs)
        d = {"p50_ms": p50, "impl": r["impl"], "build": b, "range": r[f"range_{b}"],
             "nruns": r[f"nruns_{b}"], "gpu_MHz_med": round(st.median(mhz), 2) if mhz else "",
             "gpu_MHz_set": "/".join(f"{m:g}" for m in sorted(set(mhz))),
             "test_acc_t1": r["test_acc_t1"],
             "clock": ("locked_624.75MHz_E7_rerun" if locked else "UNLOCKED_DVFS_E7_2026-10-19_not_comparable_to_lut_g1"),
             "energy": "not_measured"}
        out[r["cfg"]] = {prefix + k: v for k, v in d.items()}
    return out


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
    L1 = f"{ROOT}/results/T6/lut"
    e7 = e7_orin(f"{L1}/lut_g1f.csv", f"{ROOT}/results/T6/raw/orin1_e7locked", locked_only=True)
    bv = {r["name"]: r for r in csv.DictReader(open(f"{L1}/buildvar_cited.csv")) if r["type"] == "config"}
    for r in rows:
        if r["group"] != "G1":
            continue
        b = bv.get("g1f_" + r["cfg"].split("_lr")[0])
        if b:
            assert b["graph"].startswith("g1f_" + r["cfg"] + "_t1__"), (r["cfg"], b["graph"])
            r.update({"orin_p50_ms": b["median_p50_ms"], "orin_range_5builds": b["range_ms"], "orin_impl_build": b["impl_build"],
                      "orin_n_builds": b["n_builds"], "orin_source": "lut/buildvar_cited.csv; locked 624.75 MHz; median of 5 builds",
                      "orin_test_acc_t1": e7.get(r["cfg"], {}).get("orin_test_acc_t1", "")})
    keys += ["orin_p50_ms", "orin_range_5builds", "orin_impl_build", "orin_n_builds", "orin_test_acc_t1", "orin_source"]
    with open(f"{ROOT}/results/T6/fair_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["group"], r["kind"], r["cfg"], r["n_seeds"], "val", r["val_mean"], "test", r["test_mean"], "+-", r["test_std"], r["params_stored"])


if __name__ == "__main__":
    main()
