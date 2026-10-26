"""E-11b sensitivity analysis. Exploratory; the v1 verdict is unchanged.

No refit: constants are read from results/costmodel/fit_<tier>.json, features from features.csv,
measured dendritic times from predictions.csv / features.csv, Orin #2 build cells from orin2_builds.csv.

(a) S2/S3 pooled over Orin #1 + Pi only (without the 8051), from pairs_pred.csv.
(b) Alternative graphs for the five ratios the protocol did not pin down:
      O30 numerator  dhssm_L2_n128_b4_spk__K25  -> dhssm_L2_n128_b4_spk__K50
      O31 numerator  dhssm_L2_n256_b4_cont__K25 -> dhssm_L2_n256_b4_cont__K50
      O32 numerator  dhssm_L2_n128_b4_spk__K25  -> dhssm_L2_n128_b4_spk__K50
      O33 numerator  dhssm_L2_n128_b4_cont__K25 -> dhssm_L2_n128_b4_cont__K50
      Q1  denominator h256 conv|nocublas (P6)   -> faster h256 cell (primary: fastest of the four
          conv/gemm x default/nocublas cells; secondary: conv graph with its faster flag)
    Predictions use the existing constants and exactly the formulas of fit.py.
(c) Per-device table (CSV + LaTeX).

Outputs: results/costmodel/e11b/{sensitivity.json, per_device.csv, tab_model.tex, variants.csv}
"""
import csv
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
from fit import OUT, check_protocol, load_features, gx, mx, _cls  # noqa: E402

E11B = f"{OUT}/e11b"
LN125, LN15 = math.log(1.25), math.log(1.5)
TAGS = ("model", "P_param", "P_MAC", "blind")
PROX = ("P_param", "P_MAC", "blind")
DEV = ("orin1", "pi", "mcu", "orin2")
S2_DEV = ("orin1", "pi", "mcu")
HAND_A = dict(model=0.2253, blind=0.7227)  # values computed by hand beforehand from pairs_pred.csv, checked in (a)


def sgn(v):
    return (v > 0) - (v < 0)


def load_constants():
    TH = {}
    for t in ("orin1", "pi", "orin2"):
        c = json.load(open(f"{OUT}/fit_{t}.json"))["constants"]
        TH[t] = np.array([c["a_ms"], c["kappa_ms_per_node"], c["inv_phi_ms_per_MAC"], c["inv_beta_ms_per_weight"]])
    m = json.load(open(f"{OUT}/fit_mcu.json"))
    TH["mcu"] = np.array([m["constants"][k] for k in ("c0", "c_fc", "c_conv", "c_out")])
    return TH, m["loo_sparse_constant"]


def predict_ms(F, TH, loo, tier, g):
    r = F[(tier, g)]
    if r["role"] in ("fused_rnn", "excluded"):
        return None
    if tier == "mcu":
        c = float(np.dot(TH["mcu"], mx(r)))
        if r["role"] == "dendritic":
            c += loo[g] * r[_cls(r)]
        return c / r["mhz"] / 1000.0
    return float(np.dot(TH[tier], gx(r)))


def pair_row(F, TH, loo, pid, tier, num, den, T_d=None, den_src=None):
    """Same formulas as fit.py main() (pairs loop)."""
    n, d = F[(tier, num)], F[(tier, den)]
    T_n = n["target_ms"]
    if T_d is None:
        T_d, den_src = d["target_ms"], d["source"]
    r = T_n / T_d
    rhat = predict_ms(F, TH, loo, tier, num) / T_d
    p_param = n["params_eff"] / d["params_eff"]
    p_mac = n["F_eff"] / d["F_eff"]
    if tier == "mcu":
        fx = [1.0, n["F_fc"] + n["F_gather"] + n["F_prng"], n["F_conv"], n["N_out"]]
        blind = float(np.dot(TH["mcu"], fx)) / n["mhz"] / 1000.0
    else:
        blind = float(np.dot(TH[tier], [1.0, d["K"], n["F_eff"], n["params_eff"]]))
    row = dict(pair=pid, tier=tier, dendritic=num, dense=den, T_dendritic_ms=T_n, T_dense_ms=T_d,
               dense_source=den_src, r=r, ln_r=math.log(r), decisive=int(abs(math.log(r)) >= LN125),
               rhat=rhat, rhat_P_param=p_param, rhat_P_MAC=p_mac, rhat_blind=blind / T_d)
    for tag in TAGS:
        v = row["rhat" if tag == "model" else f"rhat_{tag}"]
        row[f"dir_ok_{tag}"] = int(sgn(math.log(v)) == sgn(math.log(r)))
        row[f"abs_lnerr_{tag}"] = abs(math.log(v) - math.log(r))
    return row


def stats(rows, devs):
    dd = [r for r in rows if r["tier"] in devs and r["decisive"]]
    e = dict(n_ratios=sum(r["tier"] in devs for r in rows), n_decisive=len(dd))
    for t in TAGS:
        e[f"dir_{t}"] = sum(int(r[f"dir_ok_{t}"]) for r in dd)
        med = st.median(float(r[f"abs_lnerr_{t}"]) for r in dd) if dd else None
        e[f"medln_{t}"] = med
        e[f"factor_{t}"] = None if med is None else math.exp(med)
    if dd:
        e["S3_model_le_half"] = {p: e["medln_model"] <= 0.5 * e[f"medln_{p}"] for p in PROX}
        e["S2_le_ln1p5"] = e["medln_model"] <= LN15
    return e


def verdict(rows):
    """Protocol S1/S2/S3 (pooled, as in verdict.py) plus per-device values."""
    dec = [r for r in rows if r["decisive"]]
    ok = sum(int(r["dir_ok_model"]) for r in dec)
    per = {d: stats(rows, (d,)) for d in DEV}
    s1_dev = {d: per[d]["dir_model"] / per[d]["n_decisive"] >= 0.8 for d in S2_DEV if per[d]["n_decisive"] >= 4}
    pooled = stats(rows, S2_DEV)
    S1 = ok / len(dec) >= 0.9 and all(s1_dev.values())
    S2 = pooled["S2_le_ln1p5"]
    S3 = all(pooled["S3_model_le_half"].values())
    reading = "PASS" if S1 and S2 and S3 else ("DIRECTION" if S1 and S3 else "FAIL")
    return dict(reading=reading, S1=S1, S2=S2, S3=S3, S1_pooled=f"{ok}/{len(dec)}", S1_per_device_ge_0p8=s1_dev,
                pooled_S2_devices=pooled, without_8051=stats(rows, ("orin1", "pi")), per_device=per)


def main():
    sha = check_protocol()
    os.makedirs(E11B, exist_ok=True)
    rows_feat = load_features()
    F = {(r["tier"], r["graph"]): r for r in rows_feat}
    TH, loo = load_constants()
    base = list(csv.DictReader(open(f"{OUT}/pairs_pred.csv")))
    for r in base:
        r["decisive"] = int(r["decisive"])
    assert len(base) == 72

    # ---- reproduction check: recompute all 72 rows from features + stored constants
    rep = []
    for pid, tier, num, den in data.pairs():
        ov = data.dense_measured_override(pid)
        rep.append(pair_row(F, TH, loo, pid, tier, num, den, *(ov if ov else (None, None))))
    maxdiff = 0.0
    for a, b in zip(base, rep):
        assert a["pair"] == b["pair"] and int(a["decisive"]) == b["decisive"]
        for k in ("r", "rhat", "rhat_P_param", "rhat_P_MAC", "rhat_blind"):
            maxdiff = max(maxdiff, abs(math.log(float(a[k])) - math.log(b[k])))
    assert maxdiff < 1e-9, maxdiff

    # ---- (a) without the 8051, straight from pairs_pred.csv
    a = stats(base, ("orin1", "pi"))
    a_out = dict(source="results/costmodel/pairs_pred.csv", devices=["orin1", "pi"],
                 n_ratios=a["n_ratios"], n_decisive=a["n_decisive"],
                 direction={t: f"{a[f'dir_{t}']}/{a['n_decisive']}" for t in TAGS},
                 median_abs_lnerr={t: a[f"medln_{t}"] for t in TAGS},
                 median_factor={t: a[f"factor_{t}"] for t in TAGS},
                 half_of_proxy_ln={p: 0.5 * a[f"medln_{p}"] for p in PROX},
                 model_le_half_of=a["S3_model_le_half"], model_le_ln1p5=a["S2_le_ln1p5"],
                 hand_values=HAND_A,
                 diff_vs_hand={k: a[f"medln_{k}"] - v for k, v in HAND_A.items()},
                 hand_check_lt_0p001=all(abs(a[f"medln_{k}"] - v) < 0.001 for k, v in HAND_A.items()))

    # ---- (b) alternatives
    med2 = data.orin2_medians()
    h256 = {k: v for k, v in med2.items() if k[0].startswith("g1f_mlp_h256_")}
    fastest = min(h256.items(), key=lambda kv: kv[1])
    conv = data.H256
    conv_fast = min(((b, med2[(conv, b)]) for b in ("default", "nocublas")), key=lambda x: x[1])
    ALT = {
        "O30": dict(num="dhssm_L2_n128_b4_spk__K50"),
        "O31": dict(num="dhssm_L2_n256_b4_cont__K50"),
        "O32": dict(num="dhssm_L2_n128_b4_spk__K50"),
        "O33": dict(num="dhssm_L2_n128_b4_cont__K50"),
        "Q1": dict(den=fastest[0][0], T_d=fastest[1],
                   src=f"orin2_builds.csv fastest h256 cell ({fastest[0][0].split('__')[1]}|{fastest[0][1]})"),
    }
    Q1_SECONDARY = dict(den=conv, T_d=conv_fast[1], src=f"orin2_builds.csv h256 conv, faster flag ({conv_fast[0]})")
    P = {p[0]: p for p in data.pairs()}
    by_id = {r["pair"]: r for r in rep}

    def alt_row(pid, spec):
        _, tier, num, den = P[pid]
        num2 = spec.get("num", num)
        den2 = spec.get("den", den)
        for g in (num2, den2):
            if (tier, g) not in F or F[(tier, g)]["target_ms"] is None:
                return None, f"not timed: {tier}:{g}"
        if "T_d" in spec:
            return pair_row(F, TH, loo, pid, tier, num2, den2, spec["T_d"], spec["src"]), None
        ov = data.dense_measured_override(pid)
        return pair_row(F, TH, loo, pid, tier, num2, den2, *(ov if ov else (None, None))), None

    var_rows, per_item = [], {}
    alt_all = dict(by_id)
    items = list(ALT.items()) + [("Q1_secondary", Q1_SECONDARY)]
    for name, spec in items:
        pid = name.split("_")[0]
        new, why = alt_row(pid, spec)
        old = by_id[pid]
        if new is None:
            per_item[name] = dict(status=why)
            continue
        if name in ALT:
            alt_all[pid] = new
        single = dict(by_id)
        single[pid] = new
        v = verdict([single[p[0]] for p in data.pairs()])
        e = dict(base=dict(dendritic=old["dendritic"], dense=old["dense"], T_dense_ms=old["T_dense_ms"],
                           r=old["r"], rhat=old["rhat"], decisive=old["decisive"],
                           dir_ok_model=old["dir_ok_model"], abs_lnerr_model=old["abs_lnerr_model"]),
                 alt=dict(dendritic=new["dendritic"], dense=new["dense"], T_dense_ms=new["T_dense_ms"],
                          dense_source=new["dense_source"], r=new["r"], rhat=new["rhat"], decisive=new["decisive"],
                          **{f"dir_ok_{t}": new[f"dir_ok_{t}"] for t in TAGS},
                          **{f"abs_lnerr_{t}": new[f"abs_lnerr_{t}"] for t in TAGS},
                          **{f"rhat_{t}": new[f"rhat_{t}"] for t in PROX}),
                 measured_direction_changed=sgn(math.log(new["r"])) != sgn(math.log(old["r"])),
                 predicted_direction_changed=sgn(math.log(new["rhat"])) != sgn(math.log(old["rhat"])),
                 decisive_changed=new["decisive"] != old["decisive"],
                 model_direction_correct_changed=(new["decisive"] and old["decisive"]
                                                  and new["dir_ok_model"] != old["dir_ok_model"]),
                 verdict_with_this_substitution=dict(reading=v["reading"], S1_pooled=v["S1_pooled"],
                                                     S2_pooled_factor=v["pooled_S2_devices"]["factor_model"]))
        per_item[name] = e
        var_rows.append(dict(item=name, pair=pid, tier=new["tier"], dendritic=new["dendritic"], dense=new["dense"],
                             T_dendritic_ms=new["T_dendritic_ms"], T_dense_ms=new["T_dense_ms"],
                             r_base=old["r"], rhat_base=old["rhat"], decisive_base=old["decisive"],
                             r_alt=new["r"], rhat_alt=new["rhat"], decisive_alt=new["decisive"],
                             dir_ok_model_alt=new["dir_ok_model"], abs_lnerr_model_alt=new["abs_lnerr_model"],
                             measured_dir_changed=int(e["measured_direction_changed"]),
                             predicted_dir_changed=int(e["predicted_direction_changed"]),
                             decisive_changed=int(e["decisive_changed"])))
    rows_alt = [alt_all[p[0]] for p in data.pairs()]
    v_base, v_alt = verdict(rep), verdict(rows_alt)
    with open(f"{E11B}/variants.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(var_rows[0].keys()))
        w.writeheader()
        w.writerows(var_rows)
    with open(f"{E11B}/pairs_pred_alt.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows_alt[0].keys()))
        w.writeheader()
        w.writerows(rows_alt)

    # ---- (c) per-device table
    groups = [("Orin #1", ("orin1",)), ("Raspberry Pi", ("pi",)), ("8051", ("mcu",)),
              ("Pooled (Orin #1 + Pi + 8051)", S2_DEV), ("Without 8051 (Orin #1 + Pi)", ("orin1", "pi")),
              ("Orin #2 (direction only)", ("orin2",)), ("All four", DEV)]
    tab = []
    for label, devs in groups:
        e = stats(base, devs)
        row = dict(group=label, devices="+".join(devs), n_ratios=e["n_ratios"], n_decisive=e["n_decisive"])
        for t in TAGS:
            row[f"dir_{t}"] = e[f"dir_{t}"]
            row[f"medln_{t}"] = e[f"medln_{t}"]
            row[f"factor_{t}"] = e[f"factor_{t}"]
        for p in PROX:
            row[f"S3_vs_{p}"] = ("" if "orin2" in devs else int(e["S3_model_le_half"][p]))
        row["S2_le_ln1p5"] = "" if "orin2" in devs else int(e["S2_le_ln1p5"])
        row["note"] = ("verdict row" if devs == S2_DEV else
                       "direction only; errors not part of S2/S3" if devs == ("orin2",) else
                       "errors include Orin #2 (not a protocol statistic)" if devs == DEV else "")
        tab.append(row)
    with open(f"{E11B}/per_device.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(tab[0].keys()))
        w.writeheader()
        w.writerows(tab)
    write_tex(tab)

    out = dict(protocol_sha256=sha, note="E-11b, exploratory; v1 verdict unchanged (no refit, constants from "
                                         "fit_<tier>.json)",
               reproduction_check=dict(max_abs_ln_diff_vs_pairs_pred=maxdiff, n=72),
               a_without_8051=a_out,
               b_alternatives=dict(items=per_item, base=strip(v_base), all_five_substituted=strip(v_alt),
                                   any_model_direction_change=any(
                                       isinstance(v, dict) and v.get("predicted_direction_changed")
                                       for k, v in per_item.items() if k in ALT),
                                   any_decisive_change=any(isinstance(v, dict) and v.get("decisive_changed")
                                                           for k, v in per_item.items() if k in ALT)),
               c_per_device=tab)
    json.dump(out, open(f"{E11B}/sensitivity.json", "w"), indent=1, default=float)
    print(json.dumps(dict(a=dict(n=a["n_decisive"], dir=a_out["direction"],
                                 factor={k: round(v, 3) for k, v in a_out["median_factor"].items()},
                                 medln={k: round(v, 4) for k, v in a_out["median_abs_lnerr"].items()},
                                 le_half=a_out["model_le_half_of"], hand_ok=a_out["hand_check_lt_0p001"]),
                          b_base=dict(reading=v_base["reading"], S1=v_base["S1_pooled"],
                                      S2=round(v_base["pooled_S2_devices"]["factor_model"], 4)),
                          b_alt=dict(reading=v_alt["reading"], S1=v_alt["S1_pooled"],
                                     S2=round(v_alt["pooled_S2_devices"]["factor_model"], 4))), indent=1))
    for r in var_rows:
        print(r)


def strip(v):
    keep = ("reading", "S1", "S2", "S3", "S1_pooled", "S1_per_device_ge_0p8")
    o = {k: v[k] for k in keep}
    o["pooled_S2_devices"] = v["pooled_S2_devices"]
    o["without_8051"] = v["without_8051"]
    o["per_device"] = v["per_device"]
    return o


def write_tex(tab):
    def fx(v, paren=False):
        s = f"{v:.2f}"
        return f"({s})" if paren else s

    lines = [r"% generated by src/costmodel/sensitivity.py (E-11b); numbers from results/costmodel/pairs_pred.csv",
             r"\begin{tabular}{lrrrrrrrrrrc}", r"\toprule",
             r"Device & Ratios & Decisive & \multicolumn{2}{c}{Model} & \multicolumn{2}{c}{P-param} & "
             r"\multicolumn{2}{c}{P-MAC} & \multicolumn{2}{c}{Blind} & S3 \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r" & & & dir. & err. & dir. & err. & dir. & err. & dir. & err. & P/M/B \\", r"\midrule"]
    names = {"Orin #1": r"Orin~\#1", "Raspberry Pi": r"Raspberry~Pi", "8051": "8051",
             "Pooled (Orin #1 + Pi + 8051)": r"\textbf{Pooled (Orin~\#1 + Pi + 8051)}",
             "Without 8051 (Orin #1 + Pi)": r"Without 8051 (Orin~\#1 + Pi)",
             "Orin #2 (direction only)": r"Orin~\#2 (direction only)", "All four": "All four"}
    for r in tab:
        o2 = r["devices"] == "orin2"
        allf = r["group"] == "All four"
        bold = r["devices"] == "+".join(S2_DEV)
        cells = [names[r["group"]], str(r["n_ratios"]), str(r["n_decisive"])]
        for t in TAGS:
            d = f"{r[f'dir_{t}']}/{r['n_decisive']}"
            if allf:
                err = "--"
            else:
                err = fx(r[f"factor_{t}"], paren=o2)
            if bold:
                d, err = rf"\textbf{{{d}}}", rf"\textbf{{{err}}}"
            cells += [d, err]
        if o2 or allf:
            cells.append("--")
        else:
            cells.append("/".join("yes" if r[f"S3_vs_{p}"] else "no" for p in PROX))
        lines.append(" & ".join(cells) + r" \\")
        if r["group"] in ("8051", "Without 8051 (Orin #1 + Pi)"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{E11B}/tab_model.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
