"""E-11 figure f6 (Fig. 4 of the paper): predicted vs measured dendritic/dense latency ratio, 72 held-out
pairs, one marker per device, y = x, tie band [0.8, 1.25] shaded, the three proxies as small grey points.
Writes results/costmodel/f6-costmodel.{pdf,png}.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = f"{ROOT}/results/costmodel"
DEV = [("orin1", "Orin #1 (GPU)", "o", "#1f77b4"), ("orin2", "Orin #2 (GPU, DVFS)", "s", "#ff7f0e"),
       ("pi", "Raspberry Pi 4B (CPU)", "^", "#2ca02c"), ("mcu", "8051 MCU", "D", "#d62728")]
PROXY = [("rhat_P_param", "x", "P-param"), ("rhat_P_MAC", "+", "P-MAC"), ("rhat_blind", "1", "mapping-blind")]


def main():
    R = list(csv.DictReader(open(f"{OUT}/pairs_pred.csv")))
    V = json.load(open(f"{OUT}/verdict.json"))
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lo, hi = 0.008, 90.0
    ax.axvspan(0.8, 1.25, color="0.88", zorder=0, lw=0)
    ax.axhspan(0.8, 1.25, color="0.94", zorder=0, lw=0)
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, zorder=1)
    for col, mk, lab in PROXY:
        ax.scatter([float(r["r"]) for r in R], [float(r[col]) for r in R], marker=mk, s=12, c="0.6", lw=0.7,
                   zorder=2, label=f"{lab} (proxy)")
    for t, lab, mk, c in DEV:
        rr = [r for r in R if r["tier"] == t]
        ok = [r for r in rr if not (int(r["decisive"]) and not int(r["dir_ok_model"]))]
        bad = [r for r in rr if int(r["decisive"]) and not int(r["dir_ok_model"])]
        ax.scatter([float(r["r"]) for r in ok], [float(r["rhat"]) for r in ok], marker=mk, s=26, facecolor=c,
                   edgecolor="k", lw=0.4, zorder=3, label=lab)
        if bad:
            ax.scatter([float(r["r"]) for r in bad], [float(r["rhat"]) for r in bad], marker=mk, s=26,
                       facecolor="white", edgecolor=c, lw=1.2, zorder=3)
            for r in bad:
                ax.annotate(r["pair"], (float(r["r"]), float(r["rhat"])), fontsize=6, xytext=(3, -7),
                            textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured ratio  r = T(dendritic) / T(dense)")
    ax.set_ylabel(r"predicted ratio  $\hat r$")
    # No in-figure title: the protocol plan forbids headlining the pooled median error (per-device numbers go in
    # the caption/text). V is still loaded so the script fails loudly if verdict.json is missing.
    _ = V["S1"], V["S2"]
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.text(1.0, lo * 1.25, "tie band\n[0.8, 1.25]", fontsize=6, ha="center", va="bottom", color="0.35")
    fig.tight_layout()
    fig.savefig(f"{OUT}/f6-costmodel.pdf")
    fig.savefig(f"{OUT}/f6-costmodel.png", dpi=200)
    print("wrote f6-costmodel.pdf/.png")


if __name__ == "__main__":
    main()
