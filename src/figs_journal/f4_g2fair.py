"""Fig. 4: G2 SHD fair protocol - test accuracy (3 seeds, mean +- SD) vs Orin Nano p50 (log x).

Marker shape = dendritic (circle) vs non-dendritic (square / triangle); colour = scan model
(chunked parallel scan: DH-SSM, diagonal SSM, LRU) vs step RNN (GRU, sequential); filled = validation-selected points
(family best by mean val accuracy, kind = family_best); hollow = DH-SSM L2 n128 b4 (cont / spk), the configuration used in
the latency / energy pairs.  Other measured configurations (each at its val-selected lr) are shown as small grey context
points.  Families whose val-selected model has no Orin measurement (LSTM h512, DH-SSM n512 b1) cannot be plotted.

Latency: orin_p50_ms of g2_fair_summary.csv = best of default / nocublas build from lut_g2.csv / lut_g2x.csv
(GRU h768 = 88.45 ms, default build; its nocublas build is 91.33 ms - this figure uses 88.45 consistently).

Source CSVs: results/T6/g2_fair_summary.csv (test_mean, test_std, val_mean, orin_p50_ms, kind),
             results/T6/lut/lut_g2x.csv (cross-check of the GRU h768 / dssmU / LRU p50; asserted equal)
"""
import csv

from style import COL1, OI, ROOT, plt, save

SCAN, STEP = OI["blue"], OI["vermillion"]
FAM = {"dhssm": ("DH-SSM", SCAN, "o", True), "dssmU": ("diag. SSM", SCAN, "s", False), "lru": ("LRU", SCAN, "D", False),
       "gru": ("GRU", STEP, "^", False), "lstm": ("LSTM", STEP, "v", False)}
REF = {"dhssm_L2_n128_b4_cont", "dhssm_L2_n128_b4_spk"}


def main():
    g2 = list(csv.DictReader(open(f"{ROOT}/results/T6/g2_fair_summary.csv")))
    x2 = {r["label"]: float(r["p50_ms"]) for r in csv.DictReader(open(f"{ROOT}/results/T6/lut/lut_g2x.csv"))}
    cfg = [r for r in g2 if r["kind"] == "config"]
    fb = {r["model"] for r in g2 if r["kind"] == "family_best"}
    best = {}
    for r in cfg:  # per architecture: val-selected lr
        k = r["arch_key"]
        if k not in best or float(r["val_mean"]) > float(best[k]["val_mean"]):
            best[k] = r
    assert abs(float(best["gru_h768"]["orin_p50_ms"]) - x2["gru_h768_lr0.003_vs_t1__seq"]) < 1e-3
    assert abs(float(best["lru_L2_H64_N128"]["orin_p50_ms"]) - x2["lru_L2_H64_N128_lr0.001_vs_t1__K50"]) < 1e-3
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    pts = {}
    for k, r in sorted(best.items()):
        if not r["orin_p50_ms"]:
            print("no Orin p50, not plotted:", k, r["model"], "family_best" if r["model"] in fb else "")
            continue
        name, c, m, dend = FAM[r["family"]]
        x, y, e = float(r["orin_p50_ms"]), 100 * float(r["test_mean"]), 100 * float(r["test_std"])
        pts[k] = (x, y, e)
        if r["model"] in fb:
            kw = dict(mfc=c, mec=c, ecolor=c, ms=5, zorder=4)
        elif k in REF:
            kw = dict(mfc="white", mec=c, ecolor=c, ms=5, zorder=4)
        else:
            kw = dict(mfc="white", mec="0.65", ecolor="0.75", ms=3.2, zorder=2)
        ax.errorbar(x, y, yerr=e, fmt="o" if dend else m, elinewidth=0.7, capsize=1.5, mew=0.9, **kw)
    # n128 cont label placed up-left so it does not run into the LRU marker / error bar; its leader
    # starts at the label's lower-right corner (REL) so it passes above the grey context error bar at 1.95 ms
    # labels: text positions in data coords (ms, %), offset sideways from error bars; leader lines
    LAB = {"lru_L2_H64_N128": ("LRU", (5.0, 91.2)), "dssmU_L2_n128_b4_spk": ("diag. SSM", (7.0, 88.9)),
           "dhssm_L2_n256_b4_cont": ("DH-SSM n256\n(val-best)", (8.5, 82.3)),
           "dhssm_L2_n128_b4_cont": ("DH-SSM\nn128 cont", (1.36, 91.0)), "dhssm_L2_n128_b4_spk": ("n128 spk", (1.3, 84.9)),
           "gru_h768": ("GRU h768", (60, 89.6)), "gru_h256": ("h256", (31, 84.2)), "gru_h128": ("h128", (29, 78.0))}
    REL = {"dhssm_L2_n128_b4_cont": (1.0, 0.0)}
    for k, (t, (tx, ty)) in LAB.items():
        if k in pts:
            x, y, _ = pts[k]
            ax.annotate(t, (x, y), xytext=(tx, ty), textcoords="data", fontsize=7, ha="center", va="center",
                        color="0.2" if k not in ("gru_h256", "gru_h128") else "0.45",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5", shrinkA=1, shrinkB=3, relpos=REL.get(k, (0.5, 0.5))))
    ax.set_xscale("log"); ax.set_xlim(0.9, 150); ax.set_ylim(75.5, 92.5)
    ax.set_xlabel("Orin Nano p50 (ms, TRT FP16, batch 1)")
    ax.set_ylabel("SHD test accuracy (%)")
    H = [ax.errorbar([], [], fmt="o", mfc="0.3", mec="0.3", ms=4.5), ax.errorbar([], [], fmt="s", mfc="0.3", mec="0.3", ms=4.5),
         ax.scatter([], [], marker="s", color=SCAN, s=16), ax.scatter([], [], marker="s", color=STEP, s=16),
         ax.errorbar([], [], fmt="o", mfc="white", mec="0.3", ms=4.5), ax.errorbar([], [], fmt="o", mfc="white", mec="0.65", ms=3)]
    L = ["dendritic", "non-dendritic", "scan model", "step RNN", "DH-SSM n128 (latency pairs)", "other configs"]
    H.insert(4, ax.errorbar([], [], fmt="o", mfc="0.3", mec="0.3", ms=4.5)); L.insert(4, "filled: val-selected family best")
    ax.legend(H, L, loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False, handletextpad=0.2, labelspacing=0.2,
              fontsize=7, ncol=2, columnspacing=0.8)
    save(fig, "f4-g2fair.pdf")
    for k, v in pts.items():
        print(k, v)


if __name__ == "__main__":
    main()
