"""G2 (SHD) fair-protocol summary generated from the per-run JSONs.

Inputs: results/T6/models/g2/*_vs_t{1,2,3}.json  (speaker split, val = training speakers 3 and 6, test acc at the
        best validation epoch; GRU / LSTM swept over lr x width).  Older *_t*.json without "_vs" are the earlier random-split protocol
        and are excluded.
        Orin p50 (TRT FP16, batch 1, best of default / nocublas build) by architecture, independent of weights / lr:
        results/T6/lut/lut_g2.csv (min over scan implementations) and lut_g2x.csv (configs measured later).
Output: results/T6/g2_fair_summary.csv, one row per config (mean over seeds) plus
        kind = family_best (best mean val within the family) and overall_best (best mean val overall).
Selection uses validation only; test is reported.
"""
import csv
import glob
import json
import os
import re
import statistics as st

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
G2 = f"{ROOT}/results/T6/models/g2"
LUT = f"{ROOT}/results/T6/lut"
OUT = f"{ROOT}/results/T6/g2_fair_summary.csv"


def family(cfg):
    if cfg.startswith("dhssm"):
        return "dhssm_nb1" if "_b1_" in cfg + "_" else "dhssm"
    return cfg.split("_")[0]


def arch_key(cfg):
    """cfg without lr / _vs suffix -> key used in the latency tables."""
    return re.sub(r"_lr[0-9.e-]+", "", cfg.replace("_vs", ""))


def load_lut():
    lat = {}
    p = f"{LUT}/lut_g2.csv"
    if os.path.exists(p):
        for x in csv.DictReader(open(p)):
            v = float(x["p50_ms"])
            k = x["cfg"]
            if k not in lat or v < lat[k][0]:
                lat[k] = (v, f"lut_g2.csv ({x['impl']}, {x['build']} build)")
    p = f"{LUT}/lut_g2x.csv"
    if os.path.exists(p):
        for x in csv.DictReader(open(p)):
            lab, impl = x["label"].split("__")
            k = arch_key(re.sub(r"_t\d+$", "", lab))
            if k not in lat:  # lut_g2 (scan sweep) has priority; g2x adds configs it lacks
                lat[k] = (float(x["p50_ms"]), f"lut_g2x.csv ({impl}, {x['build']} build, label {x['label']})")
    return lat


def main():
    files = sorted(glob.glob(f"{G2}/*_vs_t[0-9].json"))
    cfgs = {}
    for f in files:
        d = json.load(open(f))
        cfgs.setdefault(d["cfg"], []).append((os.path.basename(f), d))
    lat = load_lut()
    rows = []
    for cfg, ds in sorted(cfgs.items()):
        va = [d["val_acc"] for _, d in ds]; te = [d["test_acc"] for _, d in ds]
        ak = arch_key(cfg)
        L = lat.get(ak)
        a0 = ds[0][1]
        rows.append({"kind": "config", "family": family(cfg), "model": cfg, "arch_key": ak, "lr": json_arg(a0, "lr"),
                     "n_seeds": len(ds), "val_mean": round(st.mean(va), 4), "val_std": round(st.stdev(va), 4) if len(va) > 1 else "",
                     "test_mean": round(st.mean(te), 4), "test_std": round(st.stdev(te), 4) if len(te) > 1 else "",
                     "params_effective": a0.get("params_effective", ""),
                     "orin_p50_ms": L[0] if L else "", "orin_source": L[1] if L else "not measured",
                     "source_json": ";".join(f for f, _ in ds)})
    out = list(rows)
    for fam in sorted({r["family"] for r in rows}):
        b = max((r for r in rows if r["family"] == fam), key=lambda r: r["val_mean"])
        out.append({**b, "kind": "family_best"})
    b = max(rows, key=lambda r: r["val_mean"])
    out.append({**b, "kind": "overall_best"})
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"{len(files)} JSONs, {len(rows)} configs -> {OUT}")
    for r in out:
        if r["kind"] != "config":
            print(r["kind"], r["family"], r["model"], r["val_mean"], r["test_mean"], r["test_std"], r["params_effective"], r["orin_p50_ms"])


def json_arg(d, k):
    a = d.get("args")
    if isinstance(a, str):
        m = re.search(rf"'{k}': ([^,}}]+)", a)
        return m.group(1) if m else ""
    return (a or {}).get(k, "")


if __name__ == "__main__":
    main()
