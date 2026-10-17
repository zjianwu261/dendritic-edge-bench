"""T1m: aggregate results/T1m/quant/*.json -> results_all.csv, results.csv, acc_vs_flash.png.

results.csv (one row per model config):
  acc_fp32 / acc_int8        mean over seeds (std in *_std), Fashion-MNIST test set (10 000)
  flash_bytes / xdata_bytes  SDCC 4.2.0 link of the board firmware (uart_main + infer + tables),
                             seed 1. If the image exceeds 64 KB the linker errors out but still
                             reports the required size in .mem (used as is, fits = 0); only if no
                             size is reported: table_bytes + code overhead, flash_is_estimate = 1
  fits                       flash <= 65024 (63.5 KB) and xdata <= 8192
  cycles_per_inf, ms_per_inf@MHz   STC8 SYSclk cycles on the board (empty here; board runs: make_board_summary.py)
  cycles_ucsim_mc / instr_ucsim    ucsim s51 (-t 52, classic 12T 8051): machine cycles (Timer0,
                             timing overhead subtracted) and executed instructions per inference
  mask_source (step 5)       official = step 1-3 pipeline (official opt.make_masks + index tables for
                             dANN; also filled in for the MLP rows, which have no mask);
                             prng = receptive fields regenerated on chip (prng_rf.py), no index tables
  gen_cycles_ucsim_mc        prng only: machine cycles of all H1 rf_gen calls alone (already
                             contained in cycles_ucsim_mc); gen_frac = gen / total
Step-5 summary (best int8 accuracy within 63.5 KB, cycle ratio at equal accuracy) ->
results/T1m/step5_summary.json.
"""
import csv
import glob
import json

import numpy as np

from common import EXP

FLASH_MAX = 65024
SERIES = (("dann", "official", "o", "dANN-LRF, official masks (sparse index tables)"),
          ("dann", "prng", "^", "dANN-LRF, PRNG masks (regenerated on chip, no index)"),
          ("mlp", "", "s", "MLP 784-h-h-10"))


def main():
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{EXP}/quant/*.json"))]
    for r in rows:
        r["mask_source"] = r.get("mask_source") or "official"
    keys = ["tag", "model", "kind", "mask_source", "seed", "params", "acc_fp32", "acc_int8", "cal_acc_int8", "sh1", "sh2",
            "table_bytes", "flash_bytes", "xdata_bytes", "fits", "pc_bitexact_logits", "pc_bitexact_class",
            "ucsim_n", "ucsim_bitexact_logits", "ucsim_bitexact_class", "cycles_ucsim_mc",
            "cycles_ucsim_mc_min", "cycles_ucsim_mc_max", "instr_ucsim", "fetch_bytes_ucsim",
            "gen_cycles_ucsim_mc"]
    with open(f"{EXP}/results_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # code overhead (flash - tables) per kind, from builds that linked
    ovh = {}
    for k in ("dann", "mlp"):
        for ms in ("official", "prng"):
            v = [r["flash_bytes"] - r["table_bytes"] for r in rows
                 if r["kind"] == k and r["mask_source"] == ms and r["uart_build_ok"]]
            ovh[(k, ms)] = int(np.median(v)) if v else None
    out = []
    for name in sorted({r["model"] for r in rows}, key=lambda n: [r for r in rows if r["model"] == n][0]["params"]):
        rs = [r for r in rows if r["model"] == name]
        r1 = [r for r in rs if r["seed"] == 1][0]
        est = r1["flash_bytes"] is None
        flash = r1["table_bytes"] + ovh[(r1["kind"], r1["mask_source"])] if est else r1["flash_bytes"]
        xdata = r1["xdata_bytes"] if not est else None
        fits = bool(r1["uart_build_ok"]) and flash <= FLASH_MAX and r1["xdata_bytes"] <= 8192
        pc_all = all(r["pc_bitexact_logits"].split("/")[0] == r["pc_bitexact_logits"].split("/")[1] for r in rs)
        out.append({
            "model": name, "kind": r1["kind"], "mask_source": r1["mask_source"], "params": r1["params"],
            "acc_fp32": round(float(np.mean([r["acc_fp32"] for r in rs])), 4),
            "acc_fp32_std": round(float(np.std([r["acc_fp32"] for r in rs])), 4),
            "acc_int8": round(float(np.mean([r["acc_int8"] for r in rs])), 4),
            "acc_int8_std": round(float(np.std([r["acc_int8"] for r in rs])), 4),
            "n_seeds": len(rs), "table_bytes": r1["table_bytes"], "flash_bytes": flash,
            "flash_is_estimate": int(est), "xdata_bytes": xdata, "fits": int(fits),
            "cycles_per_inf": "", "ms_per_inf@MHz": "",
            "cycles_ucsim_mc": r1.get("cycles_ucsim_mc", ""), "instr_ucsim": r1.get("instr_ucsim", ""),
            "pc_bitexact_all_seeds": "10000/10000 x %d" % len(rs) if pc_all else "MISMATCH",
            "ucsim_bitexact_seed1": r1.get("ucsim_bitexact_logits", ""),
            "gen_cycles_ucsim_mc": r1.get("gen_cycles_ucsim_mc", ""),
            "gen_frac": round(r1["gen_cycles_ucsim_mc"] / r1["cycles_ucsim_mc"], 3)
            if r1.get("gen_cycles_ucsim_mc") and r1.get("cycles_ucsim_mc") else "",
        })
    with open(f"{EXP}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for kind, ms, mk, lab in SERIES:
        o = [x for x in out if x["kind"] == kind and (kind == "mlp" or x["mask_source"] == ms)]
        fx = np.array([x["flash_bytes"] for x in o]) / 1024
        ay = np.array([x["acc_int8"] for x in o]) * 100
        ae = np.array([x["acc_int8_std"] for x in o]) * 100
        ax.errorbar(fx, ay, yerr=ae, marker=mk, ls="-", capsize=3, label=lab)
        for x, fxx, ayy in zip(o, fx, ay):
            ax.annotate(x["model"].replace("dann_prng_", "").replace("dann_", "").replace("mlp_", "")
                        + ("*" if x["flash_is_estimate"] else ""),
                        (fxx, ayy), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axvline(FLASH_MAX / 1024, color="r", ls="--", label="63.5 KB flash (STC8A8K64D4)")
    ax.set_xlabel("Flash (KB): SDCC image incl. UART/timer firmware  (* = estimate)")
    ax.set_ylabel("Fashion-MNIST test acc, int8 (%)  mean ± std, 3 seeds")
    ax.set_title("T1m: int8 accuracy vs 8051 flash footprint")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{EXP}/acc_vs_flash.png", dpi=150)



def plot_cycles():
    """Secondary figure: int8 accuracy vs ucsim machine cycles (configs that fit only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(f"{EXP}/results.csv")))
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, ms, mk, lab in SERIES:
        o = [r for r in rows if r["kind"] == kind and (kind == "mlp" or r["mask_source"] == ms)
             and r["cycles_ucsim_mc"]]
        cx = np.array([float(r["cycles_ucsim_mc"]) for r in o]) / 1e6
        ay = np.array([float(r["acc_int8"]) for r in o]) * 100
        ax.plot(cx, ay, marker=mk, label=lab)
        for r, x, y in zip(o, cx, ay):
            ax.annotate(r["model"].split("_", 1)[1], (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("ucsim s51 machine cycles / inference (M, classic 12T 8051)")
    ax.set_ylabel("int8 test acc (%)")
    ax.set_title("T1m: accuracy vs 8051 compute (configs that fit 63.5 KB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{EXP}/acc_vs_cycles_ucsim.png", dpi=150)


def _interp_cycles(o, acc):
    """cycles at accuracy `acc` on the (monotone part of the) acc-vs-cycles curve of configs o."""
    o = sorted(o, key=lambda r: r["cyc"])
    pts, best = [], -1
    for r in o:  # keep the Pareto front (accuracy increasing with cycles)
        if r["acc"] > best:
            pts.append(r)
            best = r["acc"]
    for a, b in zip(pts, pts[1:]):
        if a["acc"] <= acc <= b["acc"]:
            return a["cyc"] + (acc - a["acc"]) / (b["acc"] - a["acc"]) * (b["cyc"] - a["cyc"])
    return None


def summary():
    """Step-5 numbers: best int8 acc within 63.5 KB per series; cycle ratio at equal accuracy."""
    rows = list(csv.DictReader(open(f"{EXP}/results.csv")))
    ser = {}
    for kind, ms, _, _ in SERIES:
        o = [dict(model=r["model"], acc=float(r["acc_int8"]), std=float(r["acc_int8_std"]),
                  flash=int(r["flash_bytes"]), cyc=float(r["cycles_ucsim_mc"]))
             for r in rows if r["kind"] == kind and (kind == "mlp" or r["mask_source"] == ms)
             and r["fits"] == "1" and r["cycles_ucsim_mc"]]
        ser[ms or "mlp"] = o
    out = {}
    for k, o in ser.items():
        b = max(o, key=lambda r: r["acc"])
        out[f"best_fit_{k}"] = b
    bp, bm = out["best_fit_prng"], out["best_fit_mlp"]
    out["best_acc_diff_prng_minus_mlp_pp"] = round(100 * (bp["acc"] - bm["acc"]), 2)
    # cycle ratio at equal accuracy: at the lower of the two best accuracies, interpolate each
    # series' Pareto curve (accuracy vs ucsim machine cycles)
    for other in ("mlp", "official"):
        a = min(bp["acc"], out[f"best_fit_{other}"]["acc"])
        cp, co = _interp_cycles(ser["prng"], a), _interp_cycles(ser[other], a)
        out[f"cycles_at_equal_acc_prng_vs_{other}"] = dict(
            acc=a, cycles_prng=cp, cycles_other=co,
            ratio_other_over_prng=round(co / cp, 2) if cp and co else None)
    # matched pairs (closest accuracy) for the text
    pairs = []
    for p in ser["prng"]:
        m = min(ser["mlp"], key=lambda r: abs(r["acc"] - p["acc"]))
        pairs.append(dict(prng=p["model"], acc_prng=p["acc"], flash_prng=p["flash"], cyc_prng=p["cyc"],
                          mlp=m["model"], acc_mlp=m["acc"], flash_mlp=m["flash"], cyc_mlp=m["cyc"],
                          cyc_ratio_mlp_over_prng=round(m["cyc"] / p["cyc"], 2)))
    out["closest_acc_pairs"] = pairs
    with open(f"{EXP}/step5_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
    plot_cycles()
    summary()
