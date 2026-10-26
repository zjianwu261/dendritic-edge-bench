"""E-11 verdict (PROTOCOL.md P7, P8) from results/costmodel/pairs_pred.csv -> verdict.json.

S1 direction: decisive ratios (|ln r| >= ln 1.25) with sign(ln rhat) == sign(ln r): >= 90 % over all four
   devices; >= 80 % on each of Orin #1, Pi, 8051 (checked only on devices with >= 4 decisive ratios).
S2 size: median |ln rhat - ln r| over the decisive ratios of Orin #1, Pi and 8051 (pooled) <= ln 1.5.
   Orin #2 (DVFS) is direction-only. Per-device medians are reported as secondary.
S3 beats proxies: the S2 statistic <= half of the same statistic for each of P-param, P-MAC, mapping-blind.
PASS = S1 & S2 & S3; DIRECTION = S1 & S3 & not S2; FAIL = not S1 or not S3.
"""
import csv
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit import OUT, check_protocol  # noqa: E402

LN125, LN15 = math.log(1.25), math.log(1.5)
DEV = ("orin1", "pi", "orin2", "mcu")
S2_DEV = ("orin1", "pi", "mcu")
TAGS = ("model", "P_param", "P_MAC", "blind")


def main():
    sha = check_protocol()
    R = list(csv.DictReader(open(f"{OUT}/pairs_pred.csv")))
    for r in R:
        for k in ("r", "ln_r", "rhat") + tuple(f"abs_lnerr_{t}" for t in TAGS):
            r[k] = float(r[k])
        r["decisive"] = int(r["decisive"])
    assert len(R) == 72
    dec = [r for r in R if r["decisive"]]
    per = {}
    for d in DEV:
        rr = [r for r in R if r["tier"] == d]
        dd = [r for r in rr if r["decisive"]]
        e = dict(n_ratios=len(rr), n_decisive=len(dd))
        for t in TAGS:
            ok = sum(int(r[f"dir_ok_{t}"]) for r in dd)
            e[f"dir_correct_{t}"] = ok
            e[f"dir_frac_{t}"] = ok / len(dd) if dd else None
            e[f"median_abs_lnerr_{t}"] = st.median(r[f"abs_lnerr_{t}"] for r in dd) if dd else None
        e["misses_model"] = [dict(pair=r["pair"], dendritic=r["dendritic"], dense=r["dense"], r=round(r["r"], 4),
                                  rhat=round(r["rhat"], 4)) for r in dd if not int(r["dir_ok_model"])]
        per[d] = e
    n_ok = sum(int(r["dir_ok_model"]) for r in dec)
    s1_pool = n_ok / len(dec)
    s1_dev = {d: (per[d]["dir_frac_model"] >= 0.8) for d in S2_DEV if per[d]["n_decisive"] >= 4}
    S1 = (s1_pool >= 0.9) and all(s1_dev.values())
    s2set = [r for r in dec if r["tier"] in S2_DEV]
    stat = {t: st.median(r[f"abs_lnerr_{t}"] for r in s2set) for t in TAGS}
    S2 = stat["model"] <= LN15
    s3 = {t: stat["model"] <= 0.5 * stat[t] for t in ("P_param", "P_MAC", "blind")}
    S3 = all(s3.values())
    if S1 and S2 and S3:
        reading = "PASS"
    elif S1 and S3:
        reading = "DIRECTION"
    else:
        reading = "FAIL"

    tie = [r for r in R if not r["decisive"]]
    tie_agree = sum(abs(math.log(r["rhat"])) < LN125 for r in tie)
    pdn = [r for r in dec if r["dir_ok_preddenom"] != ""]
    pdn_s2 = [float(r["abs_lnerr_preddenom"]) for r in pdn if r["tier"] in S2_DEV]
    sweep = list(csv.DictReader(open(f"{OUT}/sweep_pred.csv")))
    sw = {}
    for d in DEV:
        e = [float(r["abs_ln_err"]) for r in sweep if r["tier"] == d]
        if e:
            sw[d] = dict(n=len(e), median_abs_ln_err=st.median(e), median_factor=math.exp(st.median(e)),
                         max_factor=math.exp(max(e)))
    V = dict(
        protocol_sha256=sha,
        reading=reading,
        S1=dict(passed=S1, decisive_total=len(dec), direction_correct_total=n_ok, frac_pooled=s1_pool,
                threshold_pooled=0.9, per_device_frac={d: per[d]["dir_frac_model"] for d in DEV},
                per_device_checked_ge_0p8=s1_dev),
        S2=dict(passed=S2, statistic="median |ln rhat - ln r| over decisive ratios of orin1+pi+mcu (pooled)",
                n=len(s2set), median_abs_lnerr=stat["model"], median_factor=math.exp(stat["model"]),
                threshold=LN15, per_device_median_abs_lnerr={d: per[d]["median_abs_lnerr_model"] for d in S2_DEV},
                per_device_passed={d: (per[d]["median_abs_lnerr_model"] is not None
                                       and per[d]["median_abs_lnerr_model"] <= LN15) for d in S2_DEV}),
        S3=dict(passed=S3, proxy_statistic=stat, proxy_factor={t: math.exp(v) for t, v in stat.items()},
                model_le_half_of=s3),
        per_device=per,
        secondary=dict(
            tie_band=dict(n=len(tie), n_rhat_in_band=tie_agree, frac=tie_agree / len(tie) if tie else None,
                          pairs=[dict(pair=r["pair"], r=round(r["r"], 4), rhat=round(r["rhat"], 4)) for r in tie]),
            predicted_denominator=dict(
                n_decisive_with_prediction=len(pdn),
                dir_correct=sum(int(r["dir_ok_preddenom"]) for r in pdn),
                median_abs_lnerr_s2_devices=(st.median(pdn_s2) if pdn_s2 else None),
                note="fused GRU denominators have no prediction (outside the model's mapping class)"),
            sweep_absolute_error=sw,
            fit_constants={d: json.load(open(f"{OUT}/fit_{d}.json"))["constants"] for d in DEV},
            fit_ci95={d: json.load(open(f"{OUT}/fit_{d}.json"))["bootstrap"]["ci95"] for d in DEV}),
        placeholders=dict(X=n_ok, Y=len(dec), Z=round(math.exp(stat["model"]), 2),
                          Zp=round(math.exp(stat["P_param"]), 2), Zm=round(math.exp(stat["P_MAC"]), 2),
                          Zb=round(math.exp(stat["blind"]), 2),
                          n_calib={d: json.load(open(f"{OUT}/fit_{d}.json"))["n_calib"] for d in DEV}),
    )
    json.dump(V, open(f"{OUT}/verdict.json", "w"), indent=1)
    print(json.dumps(dict(reading=reading, S1=S1, S2=S2, S3=S3, pooled_dir=f"{n_ok}/{len(dec)}",
                          stat={t: round(math.exp(v), 3) for t, v in stat.items()}), indent=1))
    for d in DEV:
        e = per[d]
        print(d, e["n_ratios"], e["n_decisive"], {t: e[f"dir_correct_{t}"] for t in TAGS},
              {t: None if e[f"median_abs_lnerr_{t}"] is None else round(math.exp(e[f"median_abs_lnerr_{t}"]), 3)
               for t in TAGS}, [m["pair"] for m in e["misses_model"]])


if __name__ == "__main__":
    main()
