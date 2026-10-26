"""E-11 fit (PROTOCOL.md P2-P6). Verifies the sha256 of results/costmodel/PROTOCOL.md against the value recorded
when the protocol was written (PROTOCOL_SHA256 below) before anything else.

GPU / CPU tiers: T = a + kappa*K + F/phi + B/beta, all >= 0, fitted per tier on the calibration graphs
  by minimising sum (ln T_hat - ln T)^2 (scipy least_squares, lower bound 0, x0 = linear-space NNLS).
  Parameters are fitted as theta = (a, kappa, 1/phi, 1/beta) (same model, linear in theta).
8051: cycles = c0 + c_fc F_fc + c_conv F_conv + c_out N_out + c_g F_gather + c_p F_prng;
  dense constants on the 14 non-dendritic board builds; c_g / c_p leave-one-out on the other dANNs of
  the same mask class with the dense constants fixed.
Outputs (results/costmodel/): fit_<tier>.json, predictions.csv (every graph), pairs_pred.csv (72 rows),
sweep_pred.csv (all dendritic configurations).
"""
import csv
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
from scipy.optimize import least_squares, nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

ROOT = data.ROOT
OUT = f"{ROOT}/results/costmodel"
PROTOCOL = f"{OUT}/PROTOCOL.md"
N_BOOT = 1000
GPU_TIERS = ("orin1", "pi", "orin2")
# sha256 of PROTOCOL.md, recorded in the project log when the protocol was written (2026-10-26), before any
# feature-extraction or fitting code existed; also cited in the paper
PROTOCOL_SHA256 = "bb3f0e1d660d3930f3ece6a135f03ba0d8220bbbde3f4fc2a49add465694c148"


# ------------------------------------------------------------------------------------- protocol check
def check_protocol():
    h = hashlib.sha256(open(PROTOCOL, "rb").read()).hexdigest()
    if h != PROTOCOL_SHA256:
        sys.exit(f"fit.py: PROTOCOL.md sha256 {h} != recorded {PROTOCOL_SHA256}; refusing to fit")
    print(f"PROTOCOL.md sha256 OK ({h})")
    return h


# ------------------------------------------------------------------------------------- generic log fit
def fit_log(X, y, fixed=None):
    """min sum (ln(X theta + fixed) - ln y)^2, theta >= 0. X columns are rescaled for conditioning."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if fixed is None else np.asarray(fixed, float)
    s = X.max(axis=0)
    s[s == 0] = 1.0
    Xs = X / s
    th0, _ = nnls(Xs, y - off)
    if not np.any(Xs @ th0 + off > 0):
        th0 = np.full(X.shape[1], 1e-6)
    th0 = np.maximum(th0, 0.0)

    def res(th):
        return np.log(np.maximum(Xs @ th + off, 1e-300)) - np.log(y)

    r = least_squares(res, th0, bounds=(0.0, np.inf), method="trf", x_scale="jac", max_nfev=20000,
                      ftol=1e-14, xtol=1e-14, gtol=1e-14)
    th = r.x / s
    return th, dict(nnls_init=(th0 / s).tolist(), cost=float(r.cost), status=int(r.status),
                    rmse_log=float(np.sqrt(np.mean(res(r.x) ** 2))))


def load_features():
    R = list(csv.DictReader(open(f"{OUT}/features.csv")))
    for r in R:
        for k in ("K", "F", "B", "F_eff", "B_nnz", "params_eff", "F_fc", "F_conv", "F_gather", "F_prng", "N_out",
                  "target", "target_ms", "mhz"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
    return R


def gx(r):
    return [1.0, r["K"], r["F"], r["B"]]


def mx(r):
    return [1.0, r["F_fc"], r["F_conv"], r["N_out"]]


# ------------------------------------------------------------------------------------- GPU / CPU tiers
def fit_gpu(rows, tier, rng):
    cal = [r for r in rows if r["tier"] == tier and r["role"] == "calib"]
    X = np.array([gx(r) for r in cal])
    y = np.array([r["target"] for r in cal])
    th, info = fit_log(X, y)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(cal), len(cal))
        b, _ = fit_log(X[idx], y[idx])
        boot.append(b)
    boot = np.array(boot)
    names = ["a_ms", "kappa_ms_per_node", "inv_phi_ms_per_MAC", "inv_beta_ms_per_weight"]
    out = dict(tier=tier, model="T = a + kappa*K + F/phi + B/beta", n_calib=len(cal),
               calib_graphs=[r["graph"] for r in cal], fit_info=info,
               constants={n: float(v) for n, v in zip(names, th)},
               derived=dict(phi_MAC_per_ms=(1 / th[2] if th[2] > 0 else None),
                            beta_weights_per_ms=(1 / th[3] if th[3] > 0 else None)),
               bootstrap=dict(n=N_BOOT, seed="numpy default_rng(20261026)",
                              ci95={n: [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]
                                    for i, n in enumerate(names)},
                              median={n: float(np.median(boot[:, i])) for i, n in enumerate(names)}))
    return th, out


# ------------------------------------------------------------------------------------- 8051
def _cls(r):
    return "F_gather" if r["F_gather"] > 0 else "F_prng"


def fit_mcu(rows, rng):
    cal = [r for r in rows if r["tier"] == "mcu" and r["role"] == "calib"]
    den = [r for r in rows if r["tier"] == "mcu" and r["role"] == "dendritic"]
    X = np.array([mx(r) for r in cal])
    y = np.array([r["target"] for r in cal])
    th, info = fit_log(X, y)

    def sparse_const(th_d, members):
        if not members:
            return None
        base = np.array([np.dot(th_d, mx(r)) for r in members])
        f = np.array([[r[_cls(r)]] for r in members])
        c, _ = fit_log(f, np.array([r["target"] for r in members]), fixed=base)
        return float(c[0])

    loo = {}
    for r in den:
        others = [o for o in den if o is not r and _cls(o) == _cls(r)]
        loo[r["graph"]] = sparse_const(th, others)
    full = {k: sparse_const(th, [r for r in den if _cls(r) == k]) for k in ("F_gather", "F_prng")}
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(cal), len(cal))
        b, _ = fit_log(X[idx], y[idx])
        boot.append(list(b) + [sparse_const(b, [r for r in den if _cls(r) == k]) for k in ("F_gather", "F_prng")])
    boot = np.array(boot, float)
    names = ["c0", "c_fc", "c_conv", "c_out", "c_g", "c_p"]
    vals = list(th) + [full["F_gather"], full["F_prng"]]
    out = dict(tier="mcu", model="cycles = c0 + c_fc F_fc + c_conv F_conv + c_out N_out + c_g F_gather + c_p F_prng",
               n_calib=len(cal), calib_graphs=[r["graph"] for r in cal], fit_info=info,
               constants={n: float(v) for n, v in zip(names, vals)},
               c_g_c_p_note="c_g / c_p here: fitted on all dANNs of the class (reporting only); predictions use the "
                            "leave-one-out values in loo_sparse_constant",
               loo_sparse_constant=loo,
               bootstrap=dict(n=N_BOOT, seed="numpy default_rng(20261026)",
                              note="calibration builds resampled; c_g / c_p refitted on all dANNs of the class for "
                                   "each resample",
                              ci95={n: [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]
                                    for i, n in enumerate(names)},
                              median={n: float(np.median(boot[:, i])) for i, n in enumerate(names)}))
    return th, loo, out


# ------------------------------------------------------------------------------------- predictions
def main():
    sha = check_protocol()
    rows = load_features()
    rng = np.random.default_rng(20261026)
    TH, FITS = {}, {}
    for t in GPU_TIERS:
        TH[t], FITS[t] = fit_gpu(rows, t, rng)
    th_m, loo, FITS["mcu"] = fit_mcu(rows, rng)
    for t, f in FITS.items():
        f["protocol_sha256"] = sha
        json.dump(f, open(f"{OUT}/fit_{t}.json", "w"), indent=1)
        print(t, json.dumps(f["constants"]), f["fit_info"]["rmse_log"])

    # predicted target (ms for GPU/CPU, cycles for 8051) and predicted ms, per graph
    pred, blind_num = {}, {}
    for r in rows:
        key = (r["tier"], r["graph"])
        if r["role"] in ("fused_rnn", "excluded"):
            pred[key] = None
            continue
        if r["tier"] == "mcu":
            c = float(np.dot(th_m, mx(r)))
            if r["role"] == "dendritic":
                c += loo[r["graph"]] * r[_cls(r)]
            pred[key] = (c, c / r["mhz"] / 1000.0)
        else:
            v = float(np.dot(TH[r["tier"]], gx(r)))
            pred[key] = (v, v)
    with open(f"{OUT}/predictions.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["tier", "graph", "role", "target", "target_ms", "pred_target", "pred_ms", "ln_pred_over_meas"])
        for r in rows:
            p = pred[(r["tier"], r["graph"])]
            w.writerow([r["tier"], r["graph"], r["role"], r["target"], r["target_ms"],
                        "" if p is None else p[0], "" if p is None else p[1],
                        "" if p is None or r["target"] is None else math.log(p[0] / r["target"])])

    F = {(r["tier"], r["graph"]): r for r in rows}
    out = []
    for pid, tier, num, den in data.pairs():
        n, d = F[(tier, num)], F[(tier, den)]
        T_n, T_d = n["target_ms"], d["target_ms"]
        ov = data.dense_measured_override(pid)
        den_src = d["source"]
        if ov is not None:
            T_d, den_src = ov
        r = T_n / T_d
        rhat = pred[(tier, num)][1] / T_d
        pd = pred[(tier, den)]
        rhat2 = pred[(tier, num)][1] / pd[1] if pd is not None else None
        p_param = n["params_eff"] / d["params_eff"]
        p_mac = n["F_eff"] / d["F_eff"]
        if tier == "mcu":
            fx = [1.0, n["F_fc"] + n["F_gather"] + n["F_prng"], n["F_conv"], n["N_out"]]
            blind = float(np.dot(th_m, fx)) / n["mhz"] / 1000.0
        else:
            blind = float(np.dot(TH[tier], [1.0, d["K"], n["F_eff"], n["params_eff"]]))
        p_blind = blind / T_d
        dec = abs(math.log(r)) >= math.log(1.25)
        sgn = lambda v: (v > 0) - (v < 0)
        row = dict(pair=pid, tier=tier, dendritic=num, dense=den, T_dendritic_ms=T_n, T_dense_ms=T_d,
                   dense_source=den_src, r=r, ln_r=math.log(r), decisive=int(dec),
                   rhat=rhat, rhat_preddenom=("" if rhat2 is None else rhat2),
                   rhat_P_param=p_param, rhat_P_MAC=p_mac, rhat_blind=p_blind)
        for tag, v in (("model", rhat), ("P_param", p_param), ("P_MAC", p_mac), ("blind", p_blind)):
            row[f"dir_ok_{tag}"] = int(sgn(math.log(v)) == sgn(math.log(r)))
            row[f"abs_lnerr_{tag}"] = abs(math.log(v) - math.log(r))
        row["dir_ok_preddenom"] = "" if rhat2 is None else int(sgn(math.log(rhat2)) == sgn(math.log(r)))
        row["abs_lnerr_preddenom"] = "" if rhat2 is None else abs(math.log(rhat2) - math.log(r))
        out.append(row)
    with open(f"{OUT}/pairs_pred.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("pairs_pred.csv", len(out))

    sw = []
    for r in rows:
        if r["role"] != "dendritic":
            continue
        p = pred[(r["tier"], r["graph"])]
        sw.append(dict(tier=r["tier"], graph=r["graph"], measured_ms=r["target_ms"], predicted_ms=p[1],
                       ratio_pred_over_meas=p[1] / r["target_ms"], abs_ln_err=abs(math.log(p[1] / r["target_ms"])),
                       in_72_pairs=int(any(r["graph"] == q[2] and r["tier"] == q[1] for q in data.pairs()))))
    with open(f"{OUT}/sweep_pred.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(sw[0].keys()))
        w.writeheader()
        w.writerows(sw)
    print("sweep_pred.csv", len(sw))


if __name__ == "__main__":
    main()
