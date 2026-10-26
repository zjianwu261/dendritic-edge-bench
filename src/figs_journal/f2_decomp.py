"""Fig. 1: dendritic / dense ratio of Orin latency and energy per matched pair, grouped diverging bars from the y = 1 baseline, log y.

Source CSVs:
  results/T1/results.csv               T1 pairs (Orin Nano #1, TRT FP16, batch 1): latency_p50_locked if present else
                                           latency_p50; energy_mJ (net of idle, locked clocks)
  results/T6/lut/buildvar_cited.csv    T6 G1 dANN-G / MLP latency ratios = median over 5 fresh builds (build-median);
                                           whisker = per-build-index range.  No energy for G1 (not measured).
  results/T6/energy_compare.csv        T6 G2 / G3 energy (E_net_mJ) and latency (lut_p50_ms)
  results/T6/g2_fair_summary.csv       latency of the G2 fair family-best pair DH-SSM n256 / GRU h768 (orin_p50_ms)
The pair list (PAIRS) is the only editorial choice; change it there.
"""
import csv

import numpy as np

from style import COL2, OI, ROOT, plt, save

T1 = {(r["model"], r["device"][:4]): r for r in csv.DictReader(open(f"{ROOT}/results/T1/results.csv"))}
BV = {r["name"]: r for r in csv.DictReader(open(f"{ROOT}/results/T6/lut/buildvar_cited.csv"))}
EC = {r["label"]: r for r in csv.DictReader(open(f"{ROOT}/results/T6/energy_compare.csv"))}
G2F = {r["model"]: r for r in csv.DictReader(open(f"{ROOT}/results/T6/g2_fair_summary.csv")) if r["kind"] == "config"}


def t1(m):
    r = T1[(m, "Orin")]
    return float(r["latency_p50_locked"] or r["latency_p50"]), float(r["energy_mJ"]) if r["energy_mJ"] else None


def t1pair(d, b):
    (ld, ed), (lb, eb) = t1(d), t1(b)
    return ld / lb, (ed / eb if ed and eb else None), None


def bv(name):
    r = BV[name]
    lo, hi = map(float, r["ratio_range_perk"].split("-"))
    return float(r["ratio_median"]), None, (lo, hi)


def ec(d, b, lat_d=None, lat_b=None):
    D, B = EC[d], EC[b]
    ld = lat_d if lat_d is not None else float(D["lut_p50_ms"])
    lb = lat_b if lat_b is not None else float(B["lut_p50_ms"])
    return ld / lb, float(D["E_net_mJ"]) / float(B["E_net_mJ"]), None


# (section, label, (lat_ratio, energy_ratio, lat_range))
PAIRS = [
    ("T1 G1", "dANN-LRF / MLP h85", t1pair("dann_lrf_d32_s128", "mlp_h85_match_d32_s128")),
    ("T1 G1", "dANN-LRF / vANN", t1pair("dann_lrf_d32_s128", "vann_same_d32_s128")),
    ("T1 G1", "dANN-LRF / MLP h1024", t1pair("dann_lrf_d32_s128", "mlp_h1024_fmnist")),
    ("T1 G2", "DH-SRNN / LIF-SRNN", t1pair("dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val_seqloop")),
    ("T1 G2", "DH-SRNN / GRU h23 (Loop)", t1pair("dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seqloop")),
    ("T1 G2", "DH-SRNN / GRU h256 (fused)", t1pair("dhsrnn_n64_b8_val_seqloop", "gru_acc_val_seq")),
    ("T1 G2 step", "DH-SRNN step / LIF-SRNN step", t1pair("dhsrnn_n64_b8_step", "lif_srnn_n64_step")),
    ("T1 G2 step", "DH-SRNN step / GRU h256 step", t1pair("dhsrnn_n64_b8_step", "gru_acc_val_step")),
    ("T1 G3", "AD k4 / MLP h3592", t1pair("active_dend_k4", "mlp_h3592_match_k4")),
    ("T1 G3", "AD k8 / MLP h4965", t1pair("active_dend_k8", "mlp_h4965_match_k8")),
    ("T6 G1", "dANN-G m8 / MLP h128", bv("g1f_dannG_p4_m8_s64_sg1 / g1f_mlp_h128")),
    ("T6 G1", "dANN-G m32 / MLP h512", bv("g1f_dannG_p4_m32_s64_sg1 / g1f_mlp_h512")),
    ("T6 G1", "dANN-G m32 / MLP h1024", bv("g1f_dannG_p4_m32_s64_sg1 / g1f_mlp_h1024")),
    ("T6 G2", "DH-SSM spk n128 / GRU h64", ec("dhssm_L2_n128_b4_spk__K25", "gru_h64__seq")),
    ("T6 G2", "DH-SSM cont n128 / GRU h256", ec("dhssm_L2_n128_b4_cont__K25", "gru_h256__seq")),
    ("T6 G2", "DH-SSM n256 / GRU h768", ec("dhssm_L2_n256_b4_cont__K25", "gru_h768_lr0.003_vs_t1__seq",
                                           float(G2F["dhssm_L2_n256_b4_cont"]["orin_p50_ms"]),
                                           float(G2F["gru_h768_lr0.003"]["orin_p50_ms"]))),
    ("T6 G3", "AD-cache h1024 / MLP h1024", ec("g3_adc_h1024_ffrand__gemm", "g3_mlp_h1024__gemm")),
    ("T6 G3", "AD-cache h2048 / MLP h2048", ec("g3_adc_h2048_ffrand__gemm", "g3_mlp_h2048__gemm")),
]


def main():
    n = len(PAIRS); x = np.arange(n); w = 0.38
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    lat = np.array([p[2][0] for p in PAIRS]); en = [p[2][1] for p in PAIRS]
    # diverging bars - each bar is drawn from the y = 1 baseline up (ratio > 1) or down (ratio < 1)
    ax.bar(x - w / 2, lat - 1.0, w, bottom=1.0, color=OI["blue"], label="latency ratio", zorder=2)
    for i, p in enumerate(PAIRS):
        if p[2][2]:
            lo, hi = p[2][2]
            ax.errorbar(x[i] - w / 2, lat[i], yerr=[[lat[i] - lo], [hi - lat[i]]], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5, zorder=3)
    ev = np.array([e if e is not None else np.nan for e in en])
    ok = ~np.isnan(ev)
    ax.bar((x + w / 2)[ok], ev[ok] - 1.0, w, bottom=1.0, color=OI["orange"], label="energy ratio", zorder=2)
    for i, e in enumerate(en):
        if e is None:
            ax.text(x[i] + w / 2, 1.12, "n/m", ha="center", va="bottom", fontsize=7, rotation=90, color="0.35")
    ax.axhline(1.0, color="k", lw=0.7, zorder=1)
    ax.set_yscale("log"); ax.set_ylim(0.03, 30)
    ax.set_ylabel("dendritic / dense (Orin)")
    ax.set_xticks(x); ax.set_xticklabels([p[1] for p in PAIRS], rotation=55, ha="right", fontsize=7)
    secs = [p[0] for p in PAIRS]
    starts = [i for i in range(n) if i == 0 or secs[i] != secs[i - 1]]
    for k, s in enumerate(starts):
        e = (starts[k + 1] if k + 1 < len(starts) else n) - 1
        if s > 0:
            ax.axvline(s - 0.5, color=OI["grey"], lw=0.5, ls=":")
        ax.text((s + e) / 2, 22, secs[s], ha="center", va="center", fontsize=7)
    ax.text(1.0, 1.02, "<1: dendritic cheaper; n/m: energy not measured; whiskers: 5-build range",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.3")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=2, borderaxespad=0.1)
    save(fig, "f2-decomp.pdf")
    for p in PAIRS:
        print(p[0], p[1], [None if v is None else (round(v, 3) if not isinstance(v, tuple) else v) for v in p[2]])


if __name__ == "__main__":
    main()
