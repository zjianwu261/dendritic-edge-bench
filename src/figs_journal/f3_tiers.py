"""Fig. 3: hardware tiers.
 (a) Orin Nano vs RPi 4B p50 latency, log-log, y = x; each dense baseline is linked to its dendritic model (baseline_of).
 (b) 8051 (STC8A8K64D4): int8 test acc vs flash (63.5 KiB limit), marker area ~ ucsim cycles, filled = board-measured,
     hollow = ucsim / PC only, dANN / MLP / CNN shapes, Pareto fronts without and with CNNs.
 Optional separate file f3b-alt-latency.pdf: board latency per image vs flash (board-measured models).

Source CSVs:
  results/T1/results.csv          (a) main T1 groups only (excluded: "v0.5 weights" and "supplementary" groups, RPi-only rows);
                                      Orin = latency_p50_locked if present else latency_p50 (TRT FP16); RPi = latency_p50 (ORT FP32)
                                      (locked = latency_p50_locked present: G1 and the 1-ms step; DVFS: G2 sequence, G3)
  results/T1m/results.csv         (b) all T1m dANN / MLP models: flash_bytes, acc_int8 (3-seed mean), cycles_ucsim_mc
  results/T1m/cnn/cnn_results.csv (b) CNNs: same columns
  results/T1m/board_summary.csv   (b) which models are board-measured; f3b-alt-latency.pdf: board_ms vs flash_bytes
"""
import csv

from style import COL1, COL2, OI, ROOT, plt, save

GROUPS = {"G1 dANN / Fashion-MNIST": ("G1 Fashion-MNIST (locked)", OI["blue"]),
          "G2 SHD (full sequence T=1000)": ("G2 SHD sequence (DVFS)", OI["vermillion"]),
          "G2 SHD (streaming step, 1 ms)": ("G2 SHD 1-ms step (locked)", OI["green"]),
          "G3 active dendrites / pMNIST-10": ("G3 pMNIST-10 (DVFS)", OI["purple"])}


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/results/T1/results.csv")))
    pts = {}
    for r in rows:
        if r["group"] not in GROUPS:
            continue
        dev = "orin" if r["device"].startswith("Orin") else "rpi"
        v = float(r["latency_p50_locked"] or r["latency_p50"]) if dev == "orin" else float(r["latency_p50"])
        pts.setdefault(r["model"], {"group": r["group"], "base_of": r["baseline_of"]})[dev] = v
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 3.3), gridspec_kw={"width_ratios": [1, 1.2]})
    for m, p in pts.items():
        if "orin" not in p or "rpi" not in p:
            continue
        c = GROUPS[p["group"]][1]
        d = p["base_of"] == ""
        if not d and p["base_of"] in pts:
            q = pts[p["base_of"]]
            a.plot([q["orin"], p["orin"]], [q["rpi"], p["rpi"]], color=c, lw=0.5, alpha=0.6, zorder=1)
        a.scatter(p["orin"], p["rpi"], s=22 if d else 14, marker="o" if d else "s",
                  facecolor=c if d else "white", edgecolor=c, linewidth=0.9, zorder=3)
    lo, hi = 0.05, 400
    a.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls="--", zorder=0)
    a.text(3.0, 1.9, "y = x", fontsize=7, rotation=40, ha="center", va="center")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("Orin Nano p50 (ms, TRT FP16)"); a.set_ylabel("RPi 4B p50 (ms, ORT FP32)")
    hs = [a.scatter([], [], marker="o", color=c, s=18) for _, c in GROUPS.values()]
    hs += [a.scatter([], [], marker="o", facecolor="0.3", edgecolor="0.3", s=18),
           a.scatter([], [], marker="s", facecolor="white", edgecolor="0.3", s=14)]
    a.legend(hs, [n for n, _ in GROUPS.values()] + ["dendritic", "dense baseline (linked)"], loc="lower right",
             frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
    a.set_title("(a) Orin vs RPi", loc="left")
    a.text(0.01, 0.99, "Orin clock: locked 624.75 MHz or DVFS, as stored", transform=a.transAxes, fontsize=7, va="top",
           ha="left", color="0.3")

    panel_b(b)
    b.set_title("(b) 8051 MCU: int8 accuracy vs flash", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "f3-tiers.pdf")
    alt_latency()


def pareto(pts):
    """[(flash, acc)] -> front points (higher acc, smaller flash), sorted by flash."""
    front, best = [], -1
    for x, y in sorted(pts):
        if y > best:
            front.append((x, y)); best = y
    return front


def panel_b(b):
    """x = flash KiB with the 63.5 KiB limit, y = int8 test acc (3-seed mean); marker area ~ ucsim
    cycles per inference; filled = board-measured (board_summary.csv), hollow = simulator / PC only; dANN o, MLP s, CNN ^;
    Pareto fronts (acc vs flash) without and with CNNs; labels only on the board-measured points."""
    res = list(csv.DictReader(open(f"{ROOT}/results/T1m/results.csv")))
    cnn = list(csv.DictReader(open(f"{ROOT}/results/T1m/cnn/cnn_results.csv")))
    board = {r["model"].rsplit("_t", 1)[0] for r in csv.DictReader(open(f"{ROOT}/results/T1m/board_summary.csv"))}
    P = [(r["model"], r["kind"], float(r["flash_bytes"]) / 1024, float(r["acc_int8"]), r["cycles_ucsim_mc"]) for r in res]
    P += [(r["model"], "cnn", float(r["flash_bytes"]) / 1024, float(r["acc_int8"]), r["cycles_ucsim_mc"]) for r in cnn]
    STY = {"dann": (OI["blue"], "o", "dANN"), "mlp": (OI["orange"], "s", "MLP"), "cnn": (OI["green"], "^", "CNN")}
    size = lambda c: 8 + 12 * (float(c) / 1e6) ** 0.5 if c else 14  # noqa: E731  (no ucsim run: does not fit)
    for m, k, x, y, cyc in P:
        c, mk, _ = STY[k]
        on = m in board
        b.scatter(x, 100 * y, s=size(cyc), marker=mk, facecolor=c if on else "white", edgecolor=c,
                  linewidth=0.8, linestyle="-" if cyc else ":", zorder=4 if on else 3, alpha=1.0 if on else 0.9)
    LIM = 63.5  # fronts over deployable models only
    f1 = pareto([(x, y) for m, k, x, y, _ in P if k != "cnn" and x <= LIM]); f2 = pareto([(x, y) for _, _, x, y, _ in P if x <= LIM])
    for f, ls, lab in ((f1, "-", "Pareto front, dANN + MLP"), (f2, "--", "Pareto front, incl. CNN")):
        xs = [x for x, _ in f] + [LIM]; ys = [100 * y for _, y in f] + [100 * f[-1][1]]
        b.step(xs, ys, where="post", color="0.25", lw=0.8, ls=ls, zorder=2, label=lab)
    b.axvline(63.5, color=OI["vermillion"], lw=0.7, ls=":")
    b.text(66, 84.75, "63.5 KiB flash limit", fontsize=7, ha="left", va="bottom", color=OI["vermillion"])
    # label text positions in data coordinates (KiB, %), chosen by hand to avoid overlap; leader lines to the points
    # PRNG labels split onto two lines and placed clear of the dashed front's vertical step (x = 26.5 KiB) and of the
    # 63.5 KiB line; labels also for the board-measured CNNs c4-8 / c6-12.
    LAB = {"cnn_c8_16": ("CNN c8-16", (11.5, 89.5), "center"), "cnn_c6_12": ("CNN c6-12", (9.0, 88.65), "center"),
           "cnn_c4_8": ("CNN c4-8", (8.6, 87.6), "center"),
           "dann_prng_d16_s128": ("dANN-PRNG\nd16 s128", (28.5, 89.42), "left"),
           "dann_prng_d20_s128": ("dANN-PRNG\nd20 s128", (67, 89.95), "left"), "mlp_h64": ("MLP h64", (80, 88.25), "left"),
           "dann_d16_s64": ("dANN d16 s64", (80, 87.65), "left"), "mlp_h48": ("MLP h48", (30, 86.95), "center")}
    for m, k, x, y, _ in P:
        if m in LAB:
            t, (tx, ty), ha = LAB[m]
            b.annotate(t, (x, 100 * y), xytext=(tx, ty), textcoords="data", fontsize=7, ha=ha, va="center",
                       arrowprops=dict(arrowstyle="-", lw=0.4, color="0.45", shrinkA=1, shrinkB=3), zorder=5)
    b.set_xscale("log"); b.set_xlim(7, 170); b.set_ylim(84.6, 91.2)  # lower limit below cnn_c2_4 (84.93%)
    b.set_xticks([8, 16, 32, 63.5, 128]); b.set_xticklabels(["8", "16", "32", "63.5", "128"])
    b.minorticks_off()
    b.set_xlabel("8051 flash (KiB, log)"); b.set_ylabel("int8 test accuracy (%)")
    hs = [b.scatter([], [], marker=STY[k][1], facecolor="white", edgecolor=STY[k][0], s=20, label=STY[k][2]) for k in STY]
    hs += [b.scatter([], [], marker="o", facecolor="0.3", edgecolor="0.3", s=20, label="board-measured"),
           b.scatter([], [], marker="o", facecolor="white", edgecolor="0.3", s=20, label="ucsim / PC only")]
    hl, ll = b.get_legend_handles_labels()
    order = [2, 3, 4, 0, 1, 5, 6]  # dANN, MLP, CNN | fronts | board / ucsim
    b.legend([hl[i] for i in order], [ll[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, handletextpad=0.2, labelspacing=0.2,
             fontsize=7, ncol=3, columnspacing=0.8)
    b.text(0.01, 0.99, "marker area ~ ucsim cycles/inference\ndotted edge: > limit, no ucsim run", transform=b.transAxes,
           fontsize=7, va="top", ha="left", color="0.3")


def alt_latency():
    """Optional (not in the paper for now): board latency per image vs flash for the board-measured models."""
    fig, b = plt.subplots(figsize=(COL1, 2.5))
    br = list(csv.DictReader(open(f"{ROOT}/results/T1m/board_summary.csv")))
    LAB = {"cnn_c8_16_t1": "CNN c8-16", "cnn_c4_8_t1": "CNN c4-8", "cnn_c6_12_t1": "CNN c6-12", "dann_d16_s64_t1": "dANN d16 s64", "dann_prng_d16_s128_t1": "dANN-PRNG d16 s128",
           "dann_prng_d20_s128_t1": "dANN-PRNG d20 s128", "mlp_h48_t1": "MLP h48", "mlp_h64_t1": "MLP h64"}
    OFF = {"cnn_c8_16_t1": (6, 0), "cnn_c4_8_t1": (6, 0), "cnn_c6_12_t1": (6, 0), "dann_d16_s64_t1": (6, 0), "dann_prng_d16_s128_t1": (-6, 0),
           "dann_prng_d20_s128_t1": (-6, 0), "mlp_h48_t1": (-6, 0), "mlp_h64_t1": (0, -9)}
    for r in br:
        d = r["model"].startswith("dann")
        c = OI["blue"] if d else (OI["orange"] if r["model"].startswith("mlp") else OI["green"])
        x, y = float(r["flash_bytes"]) / 1024, float(r["board_ms"])
        b.scatter(x, y, s=24, marker="o" if d else ("s" if r["model"].startswith("mlp") else "^"),
                  facecolor=c, edgecolor=c, linewidth=1.0, zorder=3)
        dx, dy = OFF[r["model"]]
        b.annotate(f"{LAB[r['model']]}, {100 * float(r['acc_int8_mean3']):.1f}%", (x, y), xytext=(dx, dy),
                   textcoords="offset points", fontsize=7, ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                   va="center" if dy == 0 else "top")
    b.axvline(63.5, color=OI["vermillion"], lw=0.6, ls=":")
    b.text(62.8, 158, "63.5 KiB", fontsize=7, rotation=90, ha="right", va="top", color=OI["vermillion"])
    b.set_xlim(5, 68); b.set_ylim(20, 160)
    b.set_xlabel("flash (KiB)"); b.set_ylabel("board latency per image (ms, 35 MHz)")
    save(fig, "f3b-alt-latency.pdf")


if __name__ == "__main__":
    main()
