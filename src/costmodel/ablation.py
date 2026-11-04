"""E-14 (PLAN.md section 3g-16 item 1): post-hoc per-term ablation of the E-11 cost model. Exploratory
and post hoc (outside the pre-specified E-11 protocol); the section-6 / verdict.json reading is unchanged. No new measurement; reads features.csv, the 72 pairs of
data.pairs() and the stored fit_<tier>.json / pairs_pred.csv / verdict.json (only to check reproduction).
Writes only results/costmodel/e14_ablation/.

GPU / CPU tiers (orin1, pi, orin2): T = a + kappa K + F/phi + B/beta. Subsets (a always present):
  M1 F   M2 B   M3 F+B   M4 K+F   M5 K+B   M6 K+F+B (= full model, must reproduce verdict.json)
Same calibration set, same fit (fit.fit_log: log-space least squares, theta >= 0, NNLS init), same pairs,
same measured denominators (incl. the Q1 override), same decisive rule and error statistic as fit.py /
verdict.py.

8051 (cycle formula c0 + c_fc F_fc + c_conv F_conv + c_out N_out + c_g F_gather + c_p F_prng); the
correspondence used here (an interpretation, recorded in the project log for the planning role):
  F   <-> the MAC terms: c_fc F_fc + c_conv F_conv (+ the dendritic MAC terms c_g F_gather / c_p F_prng)
  K   <-> c_out N_out (per-output fixed overhead, the non-MAC term)
  B   <-> no counterpart in the cycle formula
so on the 8051 M3 == M1, M6 == M4, M2 == c0 only, M5 == c0 + c_out N_out.
Sparse-term variants (only meaningful where the MAC terms are present):
  loo   c_g / c_p leave-one-out within the mask class with the dense constants fixed (as fit.py)
  drop  c_g = c_p = 0, i.e. the gather / PRNG terms removed (those MACs cost nothing)
  fold  no c_g / c_p; gather / PRNG MACs charged at c_fc (same as the mapping-blind proxy on the 8051)
Pooled "stable" statistic = orin1 + pi + mcu decisive ratios (mcu variant named in the row).

Bootstrap: calibration graphs resampled with replacement, N_BOOT = 1000, index sequence generated exactly as
fit.py does (numpy default_rng(20261026), tiers in the order orin1, pi, orin2, mcu), and the same index sets are
used for every subset, so M6 reproduces the fit_<tier>.json intervals. Leave-one-calibration-graph-out: each
calibration graph removed in turn, constants and pair statistics recomputed.

Outputs: results/costmodel/e14_ablation/{ablation.csv, constants.csv, pairs_ablation.csv, summary.md,
reproduction.json}

Run with the dendritic conda env: python src/costmodel/ablation.py   (checks the SHA-256 of
results/costmodel/PROTOCOL.md first, as fit.py does)

Release notes. Paths are results/costmodel/ (the research repository used experiments/costmodel/). The output
summary.md is in Chinese, like the protocol records; its wording differs from the research copy only in that the
internal role name is replaced by "the planning role" (规划角色).
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
from fit import OUT, N_BOOT, check_protocol, fit_log, load_features, gx, mx, _cls  # noqa: E402

E14 = f"{OUT}/e14_ablation"
SEED = 20261026
LN125 = math.log(1.25)
GPU_TIERS = ("orin1", "pi", "orin2")
DEV = ("orin1", "pi", "orin2", "mcu")
STABLE = ("orin1", "pi", "mcu")
SUBSETS = dict(M1=("F",), M2=("B",), M3=("F", "B"), M4=("K", "F"), M5=("K", "B"), M6=("K", "F", "B"))
LABEL = dict(M1="a+F", M2="a+B", M3="a+F+B", M4="a+K+F", M5="a+K+B", M6="a+K+F+B (full)")
GPU_COL = dict(a=0, K=1, F=2, B=3)
GPU_NAMES = ["a_ms", "kappa_ms_per_node", "inv_phi_ms_per_MAC", "inv_beta_ms_per_weight"]
MCU_COL = dict(a=(0,), K=(3,), F=(1, 2), B=())
MCU_NAMES = ["c0", "c_fc", "c_conv", "c_out", "c_g", "c_p"]
MCU_VARIANTS = ("loo", "drop", "fold")
MCU_EQUIV = dict(M1="8051: c0 + MAC terms", M2="8051: c0 only (B has no counterpart)", M3="8051: == M1",
                 M4="8051: == M6", M5="8051: c0 + c_out N_out", M6="8051: full cycle formula")


def sgn(v):
    return (v > 0) - (v < 0)


def gpu_cols(subset):
    return sorted([0] + [GPU_COL[t] for t in SUBSETS[subset]])


def mcu_cols(subset):
    c = [0]
    for t in SUBSETS[subset]:
        c += list(MCU_COL[t])
    return sorted(set(c))


def has_mac(subset):
    return "F" in SUBSETS[subset]


def mcu_variants(subset):
    return MCU_VARIANTS if has_mac(subset) else ("drop",)


# ------------------------------------------------------------------------------------- fitting
def fit_cols(X, y, cols):
    """fit_log on the chosen columns; returns the full-length theta (zeros elsewhere) and fit info."""
    th = np.zeros(X.shape[1])
    Xc = X if len(cols) == X.shape[1] else np.ascontiguousarray(X[:, cols])
    t, info = fit_log(Xc, y)
    th[cols] = t
    return th, info


def at_bound(th, X, y):
    """Constant whose largest contribution on the calibration set is < 1e-9 of the smallest target."""
    return [bool(th[i] * max(X[:, i].max(), 1e-300) < 1e-9 * y.min()) for i in range(X.shape[1])]


def sparse_const(th_d, members):
    """fit.py sparse_const: c_g / c_p on dANNs of one class, dense constants fixed."""
    if not members:
        return None
    base = np.array([np.dot(th_d, mx(r)) for r in members])
    f = np.array([[r[_cls(r)]] for r in members])
    c, _ = fit_log(f, np.array([r["target"] for r in members]), fixed=base)
    return float(c[0])


class Tier:
    def __init__(self, rows, tier):
        self.tier = tier
        self.cal = [r for r in rows if r["tier"] == tier and r["role"] == "calib"]
        f = mx if tier == "mcu" else gx
        self.X = np.array([f(r) for r in self.cal], float)
        self.y = np.array([r["target"] for r in self.cal], float)
        self.den = [r for r in rows if r["tier"] == tier and r["role"] == "dendritic"]


def fit_model(T, subset, variant=None, idx=None):
    """Returns dict(theta, sparse (loo per graph), full_sparse (class fits), info) for one tier."""
    X, y = (T.X, T.y) if idx is None else (T.X[idx], T.y[idx])
    cols = mcu_cols(subset) if T.tier == "mcu" else gpu_cols(subset)
    th, info = fit_cols(X, y, cols)
    out = dict(theta=th, info=info, cols=cols, at_bound=at_bound(th, X, y))
    if T.tier == "mcu":
        loo, full = {}, {"F_gather": 0.0, "F_prng": 0.0}
        if variant == "loo":
            for r in T.den:
                others = [o for o in T.den if o is not r and _cls(o) == _cls(r)]
                loo[r["graph"]] = sparse_const(th, others)
            full = {k: sparse_const(th, [r for r in T.den if _cls(r) == k]) for k in ("F_gather", "F_prng")}
        out.update(loo=loo, full_sparse=full)
    return out


def predict_ms(T, m, r, variant=None):
    if r["role"] in ("fused_rnn", "excluded"):
        return None
    th = m["theta"]
    if T.tier != "mcu":
        return float(np.dot(th, gx(r)))
    if r["role"] == "dendritic" and variant == "fold":
        x = [1.0, r["F_fc"] + r["F_gather"] + r["F_prng"], r["F_conv"], r["N_out"]]
        c = float(np.dot(th, x))
    else:
        c = float(np.dot(th, mx(r)))
        if r["role"] == "dendritic" and variant == "loo":
            c += m["loo"][r["graph"]] * r[_cls(r)]
    return c / r["mhz"] / 1000.0


# ------------------------------------------------------------------------------------- pairs / stats
def pair_rows(F, tier, T, m, variant=None):
    out = []
    for pid, t, num, den in data.pairs():
        if t != tier:
            continue
        n, d = F[(t, num)], F[(t, den)]
        T_n, T_d = n["target_ms"], d["target_ms"]
        ov = data.dense_measured_override(pid)
        if ov is not None:
            T_d = ov[0]
        r = T_n / T_d
        rhat = predict_ms(T, m, n, variant) / T_d
        out.append(dict(pair=pid, tier=t, dendritic=num, dense=den, r=r, rhat=rhat,
                        decisive=int(abs(math.log(r)) >= LN125),
                        dir_ok=int(sgn(math.log(rhat)) == sgn(math.log(r))) if rhat > 0 else 0,
                        abs_lnerr=abs(math.log(rhat) - math.log(r)) if rhat > 0 else float("inf")))
    return out


def stats(rows):
    dd = [r for r in rows if r["decisive"]]
    med = st.median(r["abs_lnerr"] for r in dd) if dd else None
    return dict(n_ratios=len(rows), n_decisive=len(dd), dir_correct=sum(r["dir_ok"] for r in dd),
                median_abs_lnerr=med, median_factor=(math.exp(med) if med is not None else None))


# ------------------------------------------------------------------------------------- bootstrap indices
def boot_indices(tiers, n_boot=N_BOOT, seed=SEED):
    """Exactly the draw sequence of fit.py main(): one rng, tiers in order, n_boot draws each."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in ("orin1", "pi", "orin2", "mcu"):
        n = len(tiers[t].cal)
        out[t] = [rng.integers(0, n, n) for _ in range(n_boot)]
    return out


def boot_constants(T, subset, variant, idxs):
    vals = []
    for idx in idxs:
        m = fit_model(T, subset, variant, idx)
        v = list(m["theta"])
        if T.tier == "mcu":
            v += [m["full_sparse"]["F_gather"], m["full_sparse"]["F_prng"]]
        vals.append(v)
    return np.array(vals, float)


def const_vector(T, m):
    v = list(m["theta"])
    if T.tier == "mcu":
        v += [m["full_sparse"]["F_gather"], m["full_sparse"]["F_prng"]]
    return v


def active(T, subset, variant):
    cols = mcu_cols(subset) if T.tier == "mcu" else gpu_cols(subset)
    act = [i in cols for i in range(4)]
    if T.tier == "mcu":
        act += [variant == "loo"] * 2
    return act


# ------------------------------------------------------------------------------------- main analysis
def run(n_boot=N_BOOT, do_loo=True, subsets=tuple(SUBSETS)):
    rows = load_features()
    F = {(r["tier"], r["graph"]): r for r in rows}
    tiers = {t: Tier(rows, t) for t in DEV}
    bidx = boot_indices(tiers, n_boot) if n_boot else None
    res = {}  # (subset, tier, variant) -> dict
    for s in subsets:
        for t in DEV:
            T = tiers[t]
            for v in (mcu_variants(s) if t == "mcu" else (None,)):
                m = fit_model(T, s, v)
                pr = pair_rows(F, t, T, m, v)
                e = dict(model=m, pairs=pr, stats=stats(pr), consts=const_vector(T, m), active=active(T, s, v))
                if n_boot:
                    e["boot"] = boot_constants(T, s, v, bidx[t])
                if do_loo:
                    lc, ls = [], []
                    for i in range(len(T.cal)):
                        keep = np.array([j for j in range(len(T.cal)) if j != i])
                        mi = fit_model(T, s, v, keep)
                        lc.append(const_vector(T, mi))
                        ls.append(stats(pair_rows(F, t, T, mi, v)))
                    e["loo_consts"] = np.array(lc, float)
                    e["loo_stats"] = ls
                res[(s, t, v)] = e
    return dict(res=res, tiers=tiers, F=F)


def pooled(res, s, mcu_variant, devs=STABLE):
    rr = []
    for t in devs:
        v = mcu_variant if t == "mcu" else None
        rr += res[(s, t, v)]["pairs"]
    return stats(rr)


def verify_m6(out):
    """M6 must reproduce pairs_pred.csv (model column), fit_<tier>.json and verdict.json."""
    res = out["res"]
    pp = {r["pair"]: r for r in csv.DictReader(open(f"{OUT}/pairs_pred.csv"))}
    V = json.load(open(f"{OUT}/verdict.json"))
    rep = dict(max_abs_ln_diff_rhat=0.0, max_abs_ln_diff_rhat_fold_vs_blind_mcu=0.0, constants_max_rel_diff={},
               ci95_max_rel_diff={})
    for t in DEV:
        v = "loo" if t == "mcu" else None
        for r in res[("M6", t, v)]["pairs"]:
            q = pp[r["pair"]]
            assert int(q["decisive"]) == r["decisive"] and int(q["dir_ok_model"]) == r["dir_ok"], r["pair"]
            rep["max_abs_ln_diff_rhat"] = max(rep["max_abs_ln_diff_rhat"],
                                              abs(math.log(float(q["rhat"])) - math.log(r["rhat"])))
        if t == "mcu":
            for r in res[("M6", "mcu", "fold")]["pairs"]:
                rep["max_abs_ln_diff_rhat_fold_vs_blind_mcu"] = max(
                    rep["max_abs_ln_diff_rhat_fold_vs_blind_mcu"],
                    abs(math.log(float(pp[r["pair"]]["rhat_blind"])) - math.log(r["rhat"])))
        fj = json.load(open(f"{OUT}/fit_{t}.json"))
        names = MCU_NAMES if t == "mcu" else GPU_NAMES
        e = res[("M6", t, v)]
        rep["constants_max_rel_diff"][t] = max(
            abs(e["consts"][i] - fj["constants"][n]) / max(abs(fj["constants"][n]), 1e-300)
            for i, n in enumerate(names))
        if "boot" in e and len(e["boot"]) == fj["bootstrap"]["n"]:
            d = 0.0
            for i, n in enumerate(names):
                lo, hi = np.percentile(e["boot"][:, i], 2.5), np.percentile(e["boot"][:, i], 97.5)
                for a, b in ((lo, fj["bootstrap"]["ci95"][n][0]), (hi, fj["bootstrap"]["ci95"][n][1])):
                    d = max(d, abs(a - b) / max(abs(b), 1e-300))
            rep["ci95_max_rel_diff"][t] = d
    dec_total = sum(res[("M6", t, "loo" if t == "mcu" else None)]["stats"]["n_decisive"] for t in DEV)
    dir_total = sum(res[("M6", t, "loo" if t == "mcu" else None)]["stats"]["dir_correct"] for t in DEV)
    p = pooled(res, "M6", "loo")
    per = {t: res[("M6", t, "loo" if t == "mcu" else None)]["stats"]["median_abs_lnerr"] for t in STABLE}
    rep.update(direction=f"{dir_total}/{dec_total}",
               verdict_direction=f"{V['S1']['direction_correct_total']}/{V['S1']['decisive_total']}",
               pooled_median=p["median_abs_lnerr"], verdict_pooled_median=V["S2"]["median_abs_lnerr"],
               per_device_median=per, verdict_per_device_median=V["S2"]["per_device_median_abs_lnerr"])
    rep["ok"] = bool(rep["max_abs_ln_diff_rhat"] < 1e-9 and rep["direction"] == rep["verdict_direction"]
                     and abs(p["median_abs_lnerr"] - V["S2"]["median_abs_lnerr"]) < 1e-12
                     and all(abs(per[t] - V["S2"]["per_device_median_abs_lnerr"][t]) < 1e-12 for t in STABLE)
                     and all(x < 1e-9 for x in rep["constants_max_rel_diff"].values()))
    return rep


# ------------------------------------------------------------------------------------- outputs
def f6(x):
    return "" if x is None else (f"{x:.6g}" if isinstance(x, float) else x)


def write_outputs(out, rep):
    res, tiers = out["res"], out["tiers"]
    os.makedirs(E14, exist_ok=True)
    A = []
    for s in SUBSETS:
        for t in DEV:
            for v in (mcu_variants(s) if t == "mcu" else (None,)):
                e = res[(s, t, v)]
                ref = res[("M6", t, "loo" if t == "mcu" else None)]["stats"]
                S = e["stats"]
                ls = e.get("loo_stats", [])
                meds = [x["median_abs_lnerr"] for x in ls if x["median_abs_lnerr"] is not None]
                dirs = [x["dir_correct"] for x in ls]
                A.append(dict(subset=s, terms=LABEL[s], device=t, mcu_variant=v or "",
                              mcu_note=MCU_EQUIV[s] if t == "mcu" else "",
                              n_calib=len(tiers[t].cal), calib_rmse_log=e["model"]["info"]["rmse_log"],
                              fit_status=e["model"]["info"]["status"],
                              n_ratios=S["n_ratios"], n_decisive=S["n_decisive"], dir_correct=S["dir_correct"],
                              dir_frac=(S["dir_correct"] / S["n_decisive"] if S["n_decisive"] else None),
                              median_abs_lnerr=S["median_abs_lnerr"], median_factor=S["median_factor"],
                              d_median_vs_M6=(S["median_abs_lnerr"] - ref["median_abs_lnerr"]
                                              if S["median_abs_lnerr"] is not None else None),
                              d_dir_vs_M6=S["dir_correct"] - ref["dir_correct"],
                              loo_dir_min=min(dirs) if dirs else None, loo_dir_max=max(dirs) if dirs else None,
                              loo_median_min=min(meds) if meds else None, loo_median_max=max(meds) if meds else None,
                              consts_at_zero_bound=";".join(n for n, a, b in zip(
                                  MCU_NAMES if t == "mcu" else GPU_NAMES, e["model"]["at_bound"], e["active"])
                                  if a and b)))
        for v in MCU_VARIANTS if has_mac(s) else ("drop",):
            p = pooled(res, s, v)
            p4 = stats(sum((res[(s, t, (v if t == "mcu" else None))]["pairs"] for t in DEV), []))
            ref = pooled(res, "M6", "loo")
            ref4 = stats(sum((res[("M6", t, ("loo" if t == "mcu" else None))]["pairs"] for t in DEV), []))
            A.append(dict(subset=s, terms=LABEL[s], device="stable3(orin1+pi+mcu)", mcu_variant=v,
                          mcu_note=MCU_EQUIV[s], n_ratios=p["n_ratios"], n_decisive=p["n_decisive"],
                          dir_correct=p["dir_correct"], dir_frac=p["dir_correct"] / p["n_decisive"],
                          median_abs_lnerr=p["median_abs_lnerr"], median_factor=p["median_factor"],
                          d_median_vs_M6=p["median_abs_lnerr"] - ref["median_abs_lnerr"],
                          d_dir_vs_M6=p["dir_correct"] - ref["dir_correct"]))
            A.append(dict(subset=s, terms=LABEL[s], device="all4(direction)", mcu_variant=v,
                          mcu_note=MCU_EQUIV[s], n_ratios=p4["n_ratios"], n_decisive=p4["n_decisive"],
                          dir_correct=p4["dir_correct"], dir_frac=p4["dir_correct"] / p4["n_decisive"],
                          d_dir_vs_M6=p4["dir_correct"] - ref4["dir_correct"]))
    keys = list(A[0].keys())
    with open(f"{E14}/ablation.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        for r in A:
            w.writerow({k: f6(r.get(k)) for k in keys})

    C = []
    for (s, t, v), e in res.items():
        names = MCU_NAMES if t == "mcu" else GPU_NAMES
        for i, n in enumerate(names):
            if not e["active"][i]:
                continue
            b = e.get("boot")
            lc = e.get("loo_consts")
            val = e["consts"][i]
            row = dict(subset=s, terms=LABEL[s], device=t, mcu_variant=v or "", constant=n, value=val,
                       at_zero_bound=int(e["model"]["at_bound"][i]) if i < 4 else "")
            if b is not None:
                col = b[:, i]
                row.update(boot_n=len(col), boot_ci95_lo=float(np.percentile(col, 2.5)),
                           boot_ci95_hi=float(np.percentile(col, 97.5)), boot_median=float(np.median(col)),
                           boot_frac_lt_1e_6_of_value=(float(np.mean(col < 1e-6 * abs(val))) if val > 0 else ""))
            if lc is not None:
                col = lc[:, i]
                row.update(loo_min=float(col.min()), loo_max=float(col.max()),
                           loo_rel_range=(float((col.max() - col.min()) / val) if val > 0 else ""))
            C.append(row)
    keys = ["subset", "terms", "device", "mcu_variant", "constant", "value", "at_zero_bound", "boot_n",
            "boot_ci95_lo", "boot_ci95_hi", "boot_median", "boot_frac_lt_1e_6_of_value", "loo_min", "loo_max",
            "loo_rel_range"]
    with open(f"{E14}/constants.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        for r in C:
            w.writerow({k: f6(r.get(k)) for k in keys})

    P = []
    for (s, t, v), e in res.items():
        for r in e["pairs"]:
            P.append(dict(subset=s, mcu_variant=v or "", **r))
    with open(f"{E14}/pairs_ablation.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(P[0].keys()))
        w.writeheader()
        for r in P:
            w.writerow({k: f6(x) for k, x in r.items()})
    json.dump(rep, open(f"{E14}/reproduction.json", "w"), indent=1, default=float)
    return A, C


def write_summary(out, rep, A, C):
    res = out["res"]
    Ai = {(r["subset"], r["device"], r["mcu_variant"]): r for r in A}

    def cell(s, d, v=""):
        r = Ai.get((s, d, v))
        if r is None:
            return "—"
        if r.get("median_factor") is None or r.get("median_factor") == "":
            return f"{r['dir_correct']}/{r['n_decisive']}"
        return f"{r['dir_correct']}/{r['n_decisive']} · {r['median_factor']:.2f}×"

    L = ["# E-14 成本模型逐项消融（事后分析，不改 §6 判读）", "",
         "PLAN §3g-16 第 1 条。只用已有 `features.csv`、72 个比值和已有测量；同一非树突标定集、同一拟合"
         "（`fit.fit_log`，对数空间最小二乘、常数 ≥ 0、NNLS 初值）、同一实测分母（含 Q1 覆盖）、同一决定性"
         "判据（|ln r| ≥ ln 1.25）和误差口径（决定性比值上 |ln r̂ − ln r| 的中位数，表中写成倍数 exp(·)）。"
         "本文件只记数，不改协议、不改 verdict.json 的读法。代码 `src/costmodel/ablation.py`。", "",
         "## M6 复现", "",
         f"- 72 个 r̂ 与 `pairs_pred.csv` 的最大 |Δln r̂| = {rep['max_abs_ln_diff_rhat']:.1e}；方向 "
         f"{rep['direction']}（verdict.json {rep['verdict_direction']}）；三台稳定设备合并中位 "
         f"{math.exp(rep['pooled_median']):.4f}×（verdict.json {math.exp(rep['verdict_pooled_median']):.4f}×）；"
         "分设备 " + "、".join(f"{t} {math.exp(rep['per_device_median'][t]):.4f}×" for t in STABLE) + "。",
         "- 常数与 fit_<tier>.json 的最大相对差：" + "、".join(
             f"{t} {x:.1e}" for t, x in rep["constants_max_rel_diff"].items()) + "；bootstrap 95% 区间端点最大相对差："
         + "、".join(f"{t} {x:.1e}" for t, x in rep["ci95_max_rel_diff"].items()) + "。",
         f"- 8051 fold 变体（M6）与 `pairs_pred.csv` 的 mapping-blind 列最大 |Δln r̂| = "
         f"{rep['max_abs_ln_diff_rhat_fold_vs_blind_mcu']:.1e}（两者是同一公式）。",
         f"- 复现结论：{'通过' if rep['ok'] else '未通过（见 reproduction.json）'}。", "",
         "## 8051 的对应项（解释，交规划角色确认）", "",
         "- F ↔ 乘加项 c_fc F_fc + c_conv F_conv（以及树突的 c_g F_gather / c_p F_prng）；K（节点数）↔ c_out N_out"
         "（每个重量化输出的固定开销，即唯一的非乘加项）；B ↔ 周期式中没有对应项。",
         "- 因此 8051 上 M3 ≡ M1、M4 ≡ M6、M2 = 仅 c0、M5 = c0 + c_out N_out；M2 / M5 没有乘加项，也就没有 gather / PRNG 项。",
         "- gather / PRNG 变体：loo = 协议做法（留一法拟合 c_g / c_p）；drop = 去掉这两项（这些乘加不计成本）；"
         "fold = 去掉这两项、把这些乘加按 c_fc 计（与 mapping-blind 代理同式）。", "",
         "## 方向数与中位误差（决定性比值：方向正确数/决定性数 · 中位误差倍数）", "",
         "| 子集 | 项 | Orin #1 | Pi | Orin #2（只看方向） | 8051 (loo) | 三台稳定设备合并 | 四台方向合计 |",
         "|---|---|---|---|---|---|---|---|"]
    for s in SUBSETS:
        v = "loo" if has_mac(s) else "drop"
        L.append(f"| {s} | {LABEL[s]} | {cell(s, 'orin1')} | {cell(s, 'pi')} | "
                 f"{cell(s, 'orin2')} | {cell(s, 'mcu', v)}{'' if has_mac(s) else '†'} | "
                 f"{cell(s, 'stable3(orin1+pi+mcu)', v)} | {cell(s, 'all4(direction)', v)} |")
    L += ["", "† 无乘加项，8051 列与合并列用的是该子集本身（无 gather / PRNG 项）；8051 的 M2 只有常数 c0，"
          "r̂ = c0 / 实测稠密值，不是有意义的成本模型，只作对照。Orin #2 的中位误差只列出，按协议它不进幅度统计。"
          "M3 在 Orin #1 上 1/β 落在 0 界，因此与 M1 数值相同；8051 上 M3 ≡ M1、M4 ≡ M6 是按定义相同。", "",
          "### 8051 去掉 gather / PRNG 项", "",
          "| 子集 | loo（协议做法） | drop（两项去掉） | fold（按 c_fc 计） | 合并·loo | 合并·drop | 合并·fold |",
          "|---|---|---|---|---|---|---|"]
    for s in SUBSETS:
        if not has_mac(s):
            continue
        L.append(f"| {s} | {cell(s, 'mcu', 'loo')} | {cell(s, 'mcu', 'drop')} | {cell(s, 'mcu', 'fold')} | "
                 f"{cell(s, 'stable3(orin1+pi+mcu)', 'loo')} | {cell(s, 'stable3(orin1+pi+mcu)', 'drop')} | "
                 f"{cell(s, 'stable3(orin1+pi+mcu)', 'fold')} |")
    L += ["", "## 留一标定图：方向数与中位误差的范围（每台设备各自去掉一张标定图重拟合）", "",
          "| 子集 | Orin #1 | Pi | Orin #2 | 8051 (loo/drop) |", "|---|---|---|---|---|"]

    def lr(s, d, v=""):
        r = Ai[(s, d, v)]
        if r.get("loo_median_min") in (None, ""):
            return "—"
        dr = (f"{r['loo_dir_min']}" if r['loo_dir_min'] == r['loo_dir_max']
              else f"{r['loo_dir_min']}–{r['loo_dir_max']}")
        return (f"{dr}/{r['n_decisive']} · "
                f"{math.exp(r['loo_median_min']):.2f}–{math.exp(r['loo_median_max']):.2f}×")
    for s in SUBSETS:
        v = "loo" if has_mac(s) else "drop"
        L.append(f"| {s} | {lr(s, 'orin1')} | {lr(s, 'pi')} | {lr(s, 'orin2')} | {lr(s, 'mcu', v)} |")

    L += ["", "## 常数：点估计、bootstrap 95% 区间（按标定图重采样 1000 次，种子 20261026）、留一标定图范围", "",
          "“0 界”= 该常数在标定集上的最大贡献 < 最小目标值的 1e-9，即落在下界 0 上（常数有下界 0，不会出现负值）。", "",
          "“bootstrap 近 0 比例”= 重采样中该常数 < 点估计 × 1e-6 的比例（点估计本身在 0 界时不填）。"
          "8051 的 drop / fold 变体与 loo 的稠密常数相同，表中只列 loo（全部见 constants.csv）。", "",
          "| 子集 | 设备 | 常数 | 值 | 0 界 | bootstrap 95% | bootstrap 近 0 比例 | 留一范围 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in C:
        if r["device"] == "mcu" and r["mcu_variant"] != ("loo" if has_mac(r["subset"]) else "drop"):
            continue
        fz = r.get("boot_frac_lt_1e_6_of_value")
        L.append(f"| {r['subset']} | {r['device']} | {r['constant']} | {r['value']:.4g} | "
                 f"{'是' if r['at_zero_bound'] == 1 else ''} | "
                 f"[{r['boot_ci95_lo']:.3g}, {r['boot_ci95_hi']:.3g}] | {'' if fz in ('', None) else f'{fz:.1%}'} | "
                 f"[{r['loo_min']:.3g}, {r['loo_max']:.3g}] |")
    degen = [r for r in A if r.get("consts_at_zero_bound")]
    L += ["", "## 退化 / 落在 0 界的常数", ""]
    for r in degen:
        L.append(f"- {r['subset']} {r['device']} {r['mcu_variant']}：{r['consts_at_zero_bound']}")
    bad = [r for r in A if r.get("fit_status") not in (None, "") and int(r["fit_status"]) <= 0]
    L.append(f"- least_squares 状态 ≤ 0 的拟合：{len(bad)} 个。")
    L += ["", "## 文件", "",
          "- `ablation.csv`：子集 × 设备（× 8051 变体）的方向数、中位误差、与 M6 之差、留一范围；另有三台合并与四台方向合计行。",
          "- `constants.csv`：各常数点估计、bootstrap 区间、留一范围。",
          "- `pairs_ablation.csv`：每个子集 / 变体下 72 个比值的 r、r̂、方向、误差。",
          "- `reproduction.json`：M6 复现检查。", ""]
    open(f"{E14}/summary.md", "w").write("\n".join(L))


def main():
    check_protocol()
    out = run()
    rep = verify_m6(out)
    print(json.dumps(rep, indent=1, default=float))
    A, C = write_outputs(out, rep)
    write_summary(out, rep, A, C)
    for r in A:
        if r["device"] in STABLE + ("orin2", "stable3(orin1+pi+mcu)", "all4(direction)"):
            mf = r.get("median_factor")
            print(r["subset"], r["device"], r["mcu_variant"], f"{r['dir_correct']}/{r['n_decisive']}",
                  "" if mf in (None, "") else round(mf, 3), r.get("consts_at_zero_bound", ""))
    if not rep["ok"]:
        sys.exit("M6 does not reproduce verdict.json; see reproduction.json")


if __name__ == "__main__":
    main()
