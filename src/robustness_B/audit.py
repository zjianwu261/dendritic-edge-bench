"""Robustness audit of the Paper B headline claims C-a .. C-f (2026-10-25).

Reads only existing experiment files (plus results/robustness_B/g3_buildvar_builds.csv from bv_g3.py, the new
G3 build-variance session); writes results/robustness_B/C{a..f}_*.csv and energy_coverage.csv.  Nothing else is
modified.  Long format: claim, section (seed | build | sensitivity | energy), item, n, value, lo, hi, holds, source, note.
  python src/robustness_B/audit.py
"""
import csv
import glob
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = f"{ROOT}/results"
T6, T1, OUT = f"{EXP}/T6", f"{EXP}/T1", f"{EXP}/robustness_B"
LUT = f"{T6}/lut"
sys.path.insert(0, os.path.join(HERE, "..", "t6"))
import buildvar_orin as BV  # noqa: E402

COLS = ["claim", "section", "item", "n", "value", "lo", "hi", "holds", "source", "note"]


def R(v, d=4):
    return "" if v is None else (round(float(v), d) if not isinstance(v, str) else v)


def row(rows, claim, section, item, value=None, lo=None, hi=None, n="", holds="", source="", note="", d=4):
    rows.append({"claim": claim, "section": section, "item": item, "n": n, "value": R(value, d), "lo": R(lo, d),
                 "hi": R(hi, d), "holds": holds, "source": source, "note": note})


def write(name, rows):
    p = f"{OUT}/{name}"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows ->", p)


def rcsv(p):
    return list(csv.DictReader(open(p)))


def seeds_json(pattern):
    fs = sorted(glob.glob(pattern))
    return [json.load(open(f)) for f in fs]


def msd(x):
    return float(np.mean(x)), (st.stdev(x) if len(x) > 1 else 0.0)


def welch_t(a, b):
    ma, sa = msd(a); mb, sb = msd(b)
    se = (sa ** 2 / len(a) + sb ** 2 / len(b)) ** 0.5
    return (ma - mb) / se if se > 0 else float("inf")


# ------------------------------------------------------------------ build-variance helpers (G1, phase 1/2 data)
BVB = []
for p in sorted(glob.glob(f"{LUT}/buildvar_builds_p*.csv")):
    BVB += rcsv(p)
BVS = rcsv(f"{LUT}/buildvar_summary.csv")


def cells(cfg):
    """{(graph, flag): [p50 per build k=1..5]} for a buildvar config."""
    out = {}
    for g in BV.CFG[cfg]:
        for b in BV.BUILDS:
            v = sorted(((int(r["k"]), float(r["p50"])) for r in BVB if r["graph"] == g and r["build"] == b and r.get("p50")))
            if v:
                out[(g, b)] = [x for _, x in v]
    return out


def cited_cell(c):
    return min(c, key=lambda k: np.median(c[k]))


def pair_build_stats(num, den):
    """Ratio num/den: median (cited cells), cited-cell worst / best pairing, all-cell worst / best pairing, best-of-5."""
    A, B = cells(num), cells(den)
    ca, cb = cited_cell(A), cited_cell(B)
    a, b = A[ca], B[cb]
    alla = [x for v in A.values() for x in v]; allb = [x for v in B.values() for x in v]
    perk_best = [min(v[k] for v in A.values()) / min(v[k] for v in B.values()) for k in range(5)]
    return {"median": np.median(a) / np.median(b), "cell_worst": max(a) / min(b), "cell_best": min(a) / max(b),
            "perk_lo": min(perk_best), "perk_hi": max(perk_best), "best5": min(alla) / min(allb),
            "num_cell": f"{ca[0]}|{ca[1]}", "den_cell": f"{cb[0]}|{cb[1]}", "num_builds": a, "den_builds": b,
            "num_cells": {f"{k[0].split('__')[-1]}|{k[1]}": float(np.median(v)) for k, v in A.items()},
            "den_cells": {f"{k[0].split('__')[-1]}|{k[1]}": float(np.median(v)) for k, v in B.items()}}


def bv_scope(cfg, field):
    """median over builds of the per-build median of `field` (infer_only_p50_ms / p95_ms), cited cell, bv1/bv2 raw."""
    c = cells(cfg); g, b = cited_cell(c)
    vals = []
    for ph in ("1", "2"):
        p = f"{T6}/raw/orin1_bv{ph}/results.jsonl"
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p)]
        for k in range(1, 6):
            v = [r[field] for r in rs if r["label"] == f"bv{k}_{g}" and r["build"] == b and r.get(field) is not None]
            if v:
                vals.append(float(np.median(v)))
    return float(np.median(vals)) if vals else None


# ================================================================== C-a
def claim_a():
    rows, C = [], "C-a"
    G1 = f"{T6}/models/g1"
    fs = {r["cfg"]: r for r in rcsv(f"{T6}/fair_summary.csv") if r["group"] == "G1" and r["kind"] == "per_size"}
    sel = {"m32": "dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair", "h128": "mlp_h128_lr0.001_wd0_fair",
           "h256": "mlp_h256_lr0.0003_wd0_fair", "h512": "mlp_h512_lr0.0003_wd0.01_fair", "h1024": "mlp_h1024_lr0.0003_wd0_fair",
           "c32_64": "cnn_c32_64_lr0.001_wd0.01_fair"}
    acc = {k: [d["test_acc"] for d in seeds_json(f"{G1}/{v}_t[123].json")] for k, v in sel.items()}
    for k, v in acc.items():
        m, s = msd(v)
        row(rows, C, "seed", f"test acc {k} ({sel[k]})", m, min(v), max(v), n=len(v), source=f"T6/models/g1/{sel[k]}_t*.json",
            note=f"SD(ddof1)={s:.4f}; per-seed {' '.join(f'{x:.4f}' for x in v)}; fair_summary test_mean {fs.get(sel[k], {}).get('test_mean', '')}")
    gaps = []
    for h in ("h128", "h256", "h512", "h1024"):
        a, b = acc["m32"], acc[h]
        g = np.mean(a) - np.mean(b); worst = min(a) - max(b); best = max(a) - min(b)
        gaps.append(g)
        row(rows, C, "seed", f"gap m32 - MLP {h} (pp)", 100 * g, 100 * worst, 100 * best, n="3 vs 3", holds=str(worst > 0),
            source="per-seed json", note=f"lo = worst seed pairing (min m32 - max MLP), hi = best pairing; Welch t = {welch_t(a, b):.1f}")
    row(rows, C, "seed", "claimed range 0.8-1.4 pp above every MLP width", 100 * min(gaps), 100 * min(gaps), 100 * max(gaps), holds="True",
        note="mean gaps; latency/ timing used seed-1 weights only (g1f_*_t1 graphs); no per-seed latency")
    # selection-rule sensitivity: test-oracle best grid cell per MLP width vs val-selected m32
    for h in ("128", "256", "512", "1024"):
        grid = {}
        for f in glob.glob(f"{G1}/mlp_h{h}_lr*_fair_t[123].json"):
            d = json.load(open(f)); grid.setdefault(d["cfg"], []).append(d["test_acc"])
        best = max(grid, key=lambda c: np.mean(grid[c]))
        row(rows, C, "sensitivity", f"selection rule: test-oracle best grid cell MLP h{h} vs val-selected m32 (pp)",
            100 * (np.mean(acc["m32"]) - np.mean(grid[best])), n=len(grid), holds=str(np.mean(acc["m32"]) > np.mean(grid[best])),
            source="T6/models/g1", note=f"oracle cell {best} test {np.mean(grid[best]):.4f}")
    # ---- builds
    for den, claimed in (("g1f_mlp_h256", 1.12), ("g1f_mlp_h128", 1.20)):
        s = pair_build_stats("g1f_dannG_p4_m32_s64_sg1", den)
        row(rows, C, "build", f"latency ratio m32 / {den[4:]} median of 5 builds (cited cells)", s["median"], n="5 x 5",
            source="T6/lut/buildvar_builds_p1.csv", note=f"claimed {claimed}; cells {s['num_cell']} / {s['den_cell']}", d=3)
        row(rows, C, "build", f"latency ratio m32 / {den[4:]} worst / best pairing, cited cells", s["cell_worst"], s["cell_best"], s["cell_worst"],
            n="5 x 5", holds=str(s["cell_best"] > 1), source="T6/lut/buildvar_builds_p1.csv",
            note=f"value = hi = worst case for dANN-G (slowest m32 build / fastest {den[4:]} build); lo = best case (fastest m32 / slowest {den[4:]}); "
                 f"m32 builds {' '.join(f'{x:.4f}' for x in s['num_builds'])}; {den[4:]} builds {' '.join(f'{x:.4f}' for x in s['den_builds'])}; "
                 "holds = dANN-G still slower (ratio > 1) in the best case", d=3)
        row(rows, C, "build", f"latency ratio m32 / {den[4:]} per build index, fastest cell each side", None, s["perk_lo"], s["perk_hi"], n=5,
            holds=str(s["perk_lo"] > 1), source="T6/lut/buildvar_builds_p1.csv", d=3)
        row(rows, C, "build", f"latency ratio m32 / {den[4:]} best-of-5 builds (deployer keeps fastest engine)", s["best5"], n="5 x 2-4 cells",
            holds=str(s["best5"] > 1), source="T6/lut/buildvar_builds_p1.csv", d=3)
        for ck, cv in s["den_cells"].items():
            row(rows, C, "sensitivity", f"impl|flag: m32 ({s['num_cell'].split('__')[-1]}) / {den[4:]} {ck} (medians)",
                np.median(s["num_builds"]) / cv, source="buildvar_builds_p1.csv", d=3)
        for ck, cv in s["num_cells"].items():
            row(rows, C, "sensitivity", f"impl|flag: m32 {ck} / {den[4:]} ({s['den_cell'].split('__')[-1]}) (medians)",
                cv / np.median(s["den_builds"]), source="buildvar_builds_p1.csv", d=3)
        # timing scope
        for fld in ("infer_only_p50_ms", "p95_ms"):
            a, b = bv_scope("g1f_dannG_p4_m32_s64_sg1", fld), bv_scope(den, fld)
            if a and b:
                row(rows, C, "sensitivity", f"timing scope {fld}: m32 / {den[4:]}", a / b, n="5 builds", source="raw/orin1_bv1/results.jsonl",
                    note=f"{a:.4f} / {b:.4f} ms (median over builds)", d=3)
    for den in ("g1f_mlp_h512", "g1f_mlp_h1024"):
        s = pair_build_stats("g1f_dannG_p4_m32_s64_sg1", den)
        row(rows, C, "build", f"context: m32 / {den[4:]} median, worst, best", s["median"], s["cell_best"], s["cell_worst"], n="5 x 5",
            holds=str(s["cell_worst"] < 1), source="buildvar_builds_p1.csv", d=3)
    # single-session E-7 locked (10 runs, one build), random-weight LUT, unlocked DVFS session
    f7 = {r["cfg"].split("_lr")[0]: r for r in rcsv(f"{LUT}/lut_g1f.csv") if r["family"] != "anchor"}
    fu = {}
    # the unlocked (void) session is not part of the release; its descriptive row is skipped when the file is absent
    for r in (rcsv(f"{LUT}/lut_g1f_unlocked.csv") if os.path.exists(f"{LUT}/lut_g1f_unlocked.csv") else []):
        k = r["cfg"].split("_lr")[0]
        fu[k] = min(fu.get(k, 9), float(r["p50_ms"]))
    e7 = {}
    for r in rcsv(f"{LUT}/lut_g1f.csv"):
        if r["family"] == "anchor":
            continue
        k = r["cfg"].split("_lr")[0]
        e7[k] = min(e7.get(k, 9), float(r["p50_ms"]))
    lg = {}
    for r in rcsv(f"{LUT}/lut_g1.csv"):
        lg[r["cfg"]] = min(lg.get(r["cfg"], 9), float(r["p50_ms"]))
    for h in ("h128", "h256"):
        row(rows, C, "sensitivity", f"E-7 single session (one build, 10 runs, locked): m32 / {h}", e7["dannG_p4_m32_s64_sg1"] / e7[f"mlp_{h}"],
            source="T6/lut/lut_g1f.csv", note=f"{e7['dannG_p4_m32_s64_sg1']} / {e7[f'mlp_{h}']} ms", d=3)
        row(rows, C, "sensitivity", f"random-weight LUT graphs (one build): dannG_p4_m32_s64_sg1 / mlp_{h} (g1v2)",
            lg["dannG_p4_m32_s64_sg1"] / lg[f"mlp_{h}"], source="T6/lut/lut_g1.csv (mlp rows = lut_g1v2)",
            note=f"{lg['dannG_p4_m32_s64_sg1']} / {lg[f'mlp_{h}']} ms; weights = random init, not seed-1", d=3)
        if fu:
            row(rows, C, "sensitivity", f"clock: unlocked DVFS session 10-19 (not comparable): m32 / {h}", fu["dannG_p4_m32_s64_sg1"] / fu[f"mlp_{h}"],
                source="T6/lut/lut_g1f_unlocked.csv", note="governor picked ~306 MHz for these graphs; confounded, descriptive only", d=3)
    row(rows, C, "sensitivity", "batch size / power mode / FP32 / Pi thread count", note="not in the data: batch 1, 15 W, FP16-allowed builds only; no Pi timing of the fair G1 models")
    row(rows, C, "sensitivity", "on-device FP16 accuracy of the fair G1 models", note="not measured (orin_test_acc_t1 is the PyTorch seed-1 accuracy); T1 dANN-LRF FP16 delta was -0.01 pp")
    row(rows, C, "energy", "energy dANN-G / MLP (fair G1)", note="not measured (fair_summary orin_energy = not_measured); claim states latency only")
    write("Ca_g1_dannG_vs_mlp.csv", rows)
    return rows


# ================================================================== C-b
def claim_b():
    rows, C = [], "C-b"
    G2 = f"{T6}/models/g2"
    fsum = {r["model"]: r for r in rcsv(f"{T6}/g2_fair_summary.csv") if r["kind"] == "config"}
    fam = {"LRU H64 N128": "lru_L2_H64_N128_lr0.001", "GRU h768": "gru_h768_lr0.003", "DH-SSM n256 b4 cont": "dhssm_L2_n256_b4_cont",
           "diag-SSM n128 spk": "dssmU_L2_n128_b4_spk", "DH-SSM n128 b4 cont": "dhssm_L2_n128_b4_cont", "DH-SSM n128 b4 spk": "dhssm_L2_n128_b4_spk",
           "LSTM h512": "lstm_h512_lr0.003"}
    per = {}
    for k, m in fam.items():
        ds = {int(d["name"][-1]): d for d in seeds_json(f"{G2}/{m}_vs_t[123].json")}
        per[k] = ds
        v = [ds[s]["val_acc"] for s in sorted(ds)]; t = [ds[s]["test_acc"] for s in sorted(ds)]
        mv, sv = msd(v); mt, stt = msd(t)
        row(rows, C, "seed", f"{k}: val / test", mt, min(t), max(t), n=len(t), source=f"T6/models/g2/{m}_vs_t*.json",
            note=f"val {mv:.4f}+-{sv:.4f} (per seed {' '.join(f'{x:.3f}' for x in v)}); test {mt:.4f}+-{stt:.4f} (per seed {' '.join(f'{x:.3f}' for x in t)})")
    allcfg = sorted({os.path.basename(f)[:-11] for f in glob.glob(f"{G2}/*_vs_t1.json")})
    wins = []
    for s in (1, 2, 3):
        best = None
        for c in allcfg:
            f = f"{G2}/{c}_vs_t{s}.json"
            if os.path.exists(f):
                d = json.load(open(f))
                if best is None or d["val_acc"] > best[1]:
                    best = (c, d["val_acc"])
        wins.append(best)
    row(rows, C, "seed", "val-best config in each seed (all speaker-split configs)", n=3,
        holds=str(all(w[0].startswith("lru_L2_H64_N128_lr0.001") for w in wins)),
        note="; ".join(f"seed {i + 1}: {w[0]} {w[1]:.4f}" for i, w in enumerate(wins)))
    lv = [per["LRU H64 N128"][s]["val_acc"] for s in (1, 2, 3)]; gv = [per["GRU h768"][s]["val_acc"] for s in (1, 2, 3)]
    row(rows, C, "seed", "val gap LRU - GRU h768 (pp), paired by seed", 100 * (np.mean(lv) - np.mean(gv)),
        100 * min(a - b for a, b in zip(lv, gv)), 100 * max(a - b for a, b in zip(lv, gv)), n=3,
        holds=str(min(a - b for a, b in zip(lv, gv)) > 0), note=f"Welch t = {welch_t(lv, gv):.1f}")
    lt = [per["LRU H64 N128"][s]["test_acc"] for s in (1, 2, 3)]; gt = [per["GRU h768"][s]["test_acc"] for s in (1, 2, 3)]
    row(rows, C, "seed", "test gap LRU - GRU h768 (pp), paired by seed", 100 * (np.mean(lt) - np.mean(gt)),
        100 * min(a - b for a, b in zip(lt, gt)), 100 * max(a - b for a, b in zip(lt, gt)), n=3, note=f"Welch t = {welch_t(lt, gt):.1f}")
    row(rows, C, "sensitivity", "lr: LRU H64 at lr 3e-3 instead of 1e-3", float(fsum["lru_L2_H64_N128_lr0.003"]["val_mean"]),
        note=f"val {fsum['lru_L2_H64_N128_lr0.003']['val_mean']} (test {fsum['lru_L2_H64_N128_lr0.003']['test_mean']}); LRU H128 N256 lr3e-3 collapses (val 0.387): LRU is val-best only at lr 1e-3; GRU h768 lr1e-3 val {fsum['gru_h768_lr0.001']['val_mean']}")
    # ---- latency per engine (one engine per flag; runs from raw/orin1, all sessions)
    rs = [json.loads(l) for l in open(f"{T6}/raw/orin1/results.jsonl")]
    lab = {"LRU H64 N128": "lru_L2_H64_N128_lr0.001_vs_t1__K50", "diag-SSM n128 spk": "dssmU_L2_n128_b4_spk_vs_t1__K50",
           "DH-SSM n128 b4 cont": "dhssm_L2_n128_b4_cont__K25", "DH-SSM n128 b4 spk": "dhssm_L2_n128_b4_spk__K25",
           "DH-SSM n256 b4 cont": "dhssm_L2_n256_b4_cont__K25", "GRU h768": "gru_h768_lr0.003_vs_t1__seq"}
    lat = {}
    for k, l in lab.items():
        for b in ("default", "nocublas"):
            v = [r["p50_ms"] for r in rs if r["label"] == l and r["build"] == b and r.get("gpu_freq_timed_frac_at_max") == 1.0]
            days = sorted({r["time"][:10] for r in rs if r["label"] == l and r["build"] == b})
            lat[(k, b)] = v
            row(rows, C, "build", f"{k} {b} engine p50 (ms)", np.median(v), min(v), max(v), n=f"1 build, {len(v)} runs",
                source="T6/raw/orin1/results.jsonl", note=f"label {l}; sessions {' '.join(days)}; all runs at 624.75 MHz")
    scan = [k for k in lab if k != "GRU h768"]
    fb = ["LRU H64 N128", "diag-SSM n128 spk", "DH-SSM n256 b4 cont"]
    best_eng = {k: min(np.median(lat[(k, b)]) for b in ("default", "nocublas")) for k in lab}
    row(rows, C, "build", "scan family bests (LRU, diag-SSM, DH-SSM n256) latency range, best flag (ms)", None,
        min(best_eng[k] for k in fb), max(best_eng[k] for k in fb), note="claimed 2.8-4.2 ms")
    g_all = lat[("GRU h768", "default")] + lat[("GRU h768", "nocublas")]
    s_all = [x for k in fb for b in ("default", "nocublas") for x in lat[(k, b)]]
    row(rows, C, "build", "GRU h768 / scan model latency ratio: worst (slowest scan run vs fastest GRU run) / best", best_eng["GRU h768"] / max(best_eng[k] for k in fb),
        min(g_all) / max(s_all), max(g_all) / min(s_all), holds=str(min(g_all) / max(s_all) > 1), note="value = GRU best flag / slowest family best", d=2)
    row(rows, C, "build", "TRT build-to-build spread not measured for G2", note="each G2 graph x flag has exactly one engine (the 10-08 re-runs reused the 10-02 engines); G1 data show up to 1.22x per rebuild, which cannot close a 20-33x gap but can move the LRU vs DH-SSM n128 comparison")
    lr = best_eng["LRU H64 N128"]
    for k in ("DH-SSM n128 b4 cont", "DH-SSM n128 b4 spk", "DH-SSM n256 b4 cont", "diag-SSM n128 spk"):
        row(rows, C, "build", f"LRU / {k} latency (best flag; run-range extremes)", lr / best_eng[k],
            min(lat[("LRU H64 N128", "default")] + lat[("LRU H64 N128", "nocublas")]) / max(lat[(k, "default")] + lat[(k, "nocublas")]),
            max(lat[("LRU H64 N128", "default")] + lat[("LRU H64 N128", "nocublas")]) / min(lat[(k, "default")] + lat[(k, "nocublas")]), d=3,
            note="LRU graph is K50; DH-SSM rows K25")
    # chunk length K
    g2 = {}
    for r in rcsv(f"{LUT}/lut_g2.csv"):
        g2[(r["cfg"], r["impl"])] = float(r["p50_ms"])
    for cfg in ("dhssm_L2_n128_b4_cont", "dhssm_L2_n128_b4_spk", "dhssm_L2_n256_b4_cont"):
        ks = {i: v for (c, i), v in g2.items() if c == cfg}
        row(rows, C, "sensitivity", f"chunk length K: {cfg}", min(ks.values()), min(ks.values()), max(ks.values()),
            note=" ".join(f"{i}={v}" for i, v in sorted(ks.items())) + f"; LRU K50 = {lr:.4f}", source="T6/lut/lut_g2.csv")
    row(rows, C, "sensitivity", "LRU / DH-SSM n128 cont at equal K=50", lr / g2[("dhssm_L2_n128_b4_cont", "K50")], d=3,
        note=f"{lr:.4f} / {g2[('dhssm_L2_n128_b4_cont', 'K50')]} ms")
    row(rows, C, "sensitivity", "scan range over all K (family bests + n128 DH-SSM)", None,
        min([lr, best_eng["diag-SSM n128 spk"]] + [v for (c, i), v in g2.items() if c.startswith("dhssm_L2_n128_b4") or c == "dhssm_L2_n256_b4_cont"]),
        max([lr, best_eng["diag-SSM n128 spk"]] + [v for (c, i), v in g2.items() if c.startswith("dhssm_L2_n128_b4") or c == "dhssm_L2_n256_b4_cont"]),
        note="K100 DH-SSM spk 3.98 ms; widens the claimed 2.8-4.2 to about 2.6-4.5 ms")
    row(rows, C, "sensitivity", "weights timed", note="LRU / diag-SSM / GRU h768: seed-1 speaker-split weights (g2x); DH-SSM rows: lut_g2 graphs (T1r-era weights, not the speaker-split models). Scan latency is weight-independent in structure, but not re-timed per seed")
    fp = {r["model"]: r for r in rcsv(f"{T6}/device_fp16_acc.csv")}
    for m, r in fp.items():
        row(rows, C, "sensitivity", f"on-device FP16 vs FP32 accuracy, seed 1: {m} (pp)", float(r["delta_fp16_vs_torch_pp"]), n=int(r["n_eval"]),
            source="T6/device_fp16_acc.csv", d=2)
    row(rows, C, "sensitivity", "GRU h768 flag: default 88.45 vs nocublas 91.33 ms", 91.3283 / 88.4487, d=3, note="ratio nocublas/default; energy run (nocublas) mean 91.4 ms")
    # energy
    ec = {r["label"]: r for r in rcsv(f"{T6}/energy_compare.csv")}
    el = {"LRU H64 N128": "lru_L2_H64_N128_lr0.001_vs_t1__K50", "diag-SSM n128 spk": "dssmU_L2_n128_b4_spk_vs_t1__K50",
          "DH-SSM n128 b4 cont": "dhssm_L2_n128_b4_cont__K25", "DH-SSM n128 b4 spk": "dhssm_L2_n128_b4_spk__K25",
          "DH-SSM n256 b4 cont": "dhssm_L2_n256_b4_cont__K25", "GRU h768": "gru_h768_lr0.003_vs_t1__seq"}
    E = {}
    for k, l in el.items():
        r = ec[l]; lo, hi = map(float, r["E_net_range"].split("-"))
        E[k] = (float(r["E_net_mJ"]), lo, hi)
        row(rows, C, "energy", f"{k} E_net (mJ/inference)", E[k][0], lo, hi, n=f"{r['n_runs']} runs, 1 engine (nocublas)",
            source="T6/energy_compare.csv", note="locked 624.75 MHz, net of idle")
    for k in ("DH-SSM n128 b4 cont", "DH-SSM n128 b4 spk", "DH-SSM n256 b4 cont", "diag-SSM n128 spk", "GRU h768"):
        row(rows, C, "energy", f"energy ratio LRU / {k}", E["LRU H64 N128"][0] / E[k][0], E["LRU H64 N128"][1] / E[k][2],
            E["LRU H64 N128"][2] / E[k][1], d=3, note="lo/hi = run-range extremes")
    write("Cb_g2_scan_vs_gru.csv", rows)
    return rows


# ================================================================== C-c
def claim_c():
    rows, C = [], "C-c"
    res = {(r["model"], r["device"][:4]): r for r in rcsv(f"{T1}/results.csv")}
    blds = {(r["model"], r["trt_build"]): r for r in rcsv(f"{T1}/results_orin_builds.csv")}
    en = {r["label"]: r for r in rcsv(f"{T1}/energy_orin1.csv")}
    T1P = [("dANN d32 / MLP h85 [param-matched]", "dann_lrf_d32_s128", "mlp_h85_match_d32_s128", True, True),
           ("dANN d32 / vANN", "dann_lrf_d32_s128", "vann_same_d32_s128", True, False),
           ("dANN d32 / MLP h1024", "dann_lrf_d32_s128", "mlp_h1024_fmnist", True, False),
           ("DH-SRNN / LIF-SRNN (loop)", "dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val_seqloop", True, False),
           ("DH-SRNN / GRU-pm (loop) [param-matched]", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seqloop", True, True),
           ("DH-SRNN / GRU-pm (fused op) [param-matched; not in F2]", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seq", False, True),
           ("DH-SRNN / GRU-acc (fused)", "dhsrnn_n64_b8_val_seqloop", "gru_acc_val_seq", True, False),
           ("DH-SRNN / LIF (step)", "dhsrnn_n64_b8_step", "lif_srnn_n64_step", True, False),
           ("DH-SRNN / GRU h23 (step) [param-matched; not in F2]", "dhsrnn_n64_b8_step", "gru_h23_step", False, True),
           ("DH-SRNN / GRU-acc (step)", "dhsrnn_n64_b8_step", "gru_acc_val_step", True, False),
           ("AD k4 / MLP h3592 [param-matched]", "active_dend_k4", "mlp_h3592_match_k4", True, True),
           ("AD k8 / MLP h4965 [param-matched]", "active_dend_k8", "mlp_h4965_match_k8", True, True),
           ("AD k4 / MLP h3941 [stored-matched; not in F2]", "active_dend_k4", "mlp_h3941_matchstored_k4", False, True),
           ("AD k8 / MLP h5230 [stored-matched; not in F2]", "active_dend_k8", "mlp_h5230_matchstored_k8", False, True)]
    naive_lat, naive_en = [], []
    for name, d, b, inf2, matched in T1P:
        o_d, o_b = res[(d, "Orin")], res[(b, "Orin")]
        f2lat = float(o_d["latency_p50_locked"] or o_d["latency_p50"]) / float(o_b["latency_p50_locked"] or o_b["latency_p50"])
        row(rows, C, "build", f"{name}: Orin latency ratio as in F2 (locked if present, else DVFS; nocublas)", f2lat, n="1 seed, 1 build/flag, 3 runs",
            source="T1/results.csv", note=f"in F2: {inf2}", d=3)
        rd = {fl: blds[(d, fl)] for fl in ("default", "nocublas")}; rb = {fl: blds[(b, fl)] for fl in ("default", "nocublas")}
        same = {fl: float(rd[fl]["latency_p50"]) / float(rb[fl]["latency_p50"]) for fl in rd}
        lo_d = [float(x) for fl in rd for x in rd[fl]["latency_p50_range"].split("-")]
        lo_b = [float(x) for fl in rb for x in rb[fl]["latency_p50_range"].split("-")]
        row(rows, C, "build", f"{name}: DVFS ratio default/default, nocublas/nocublas; worst/best over flags x runs", same["nocublas"],
            min(lo_d) / max(lo_b), max(lo_d) / min(lo_b), n="2 flags x 3 runs", holds=str(min(lo_d) / max(lo_b) > 1),
            source="T1/results_orin_builds.csv", note=f"default/default {same['default']:.3f}; lo = fastest dendritic / slowest dense", d=3)
        if o_d["latency_p50_locked"] and o_b["latency_p50_locked"]:
            row(rows, C, "sensitivity", f"{name}: clock locked vs DVFS (nocublas)", float(o_d["latency_p50_locked"]) / float(o_b["latency_p50_locked"]),
                n=3, note=f"DVFS {float(o_d['latency_p50']) / float(o_b['latency_p50']):.3f}", source="T1/results.csv", d=3)
        ed, eb = en.get(d), en.get(b)
        if ed and eb:
            mlock = float(ed["mean_ms"]) / float(eb["mean_ms"])
            row(rows, C, "sensitivity", f"{name}: locked-clock mean latency from the energy window", mlock, n="3 x 30 s",
                source="T1/energy_orin1.csv", note="mean not p50; same nocublas engine", d=3)
            e = float(ed["E_net_mJ"]) / float(eb["E_net_mJ"])
            dl, dh = map(float, ed["E_net_range"].split("-")); bl, bh = map(float, eb["E_net_range"].split("-"))
            row(rows, C, "energy", f"{name}: energy ratio", e, dl / bh, dh / bl, n="3 runs each, 1 engine", holds=str(dl / bh > 1),
                source="T1/energy_orin1.csv", note="locked 624.75 MHz, net of idle; nocublas engine only", d=3)
            if matched:
                naive_en.append((name, e, dl / bh, dh / bl))
        pd_, pb_ = res.get((d, "RPi ")), res.get((b, "RPi "))
        if pd_ and pb_:
            row(rows, C, "sensitivity", f"{name}: RPi 4B (ORT FP32, 3 threads) latency ratio", float(pd_["latency_p50"]) / float(pb_["latency_p50"]),
                n="3 runs", source="T1/results.csv", note="tier sensitivity; thread count fixed at 3, no thread sweep for these models", d=3)
        if matched:
            naive_lat.append((name, f2lat, min(lo_d) / max(lo_b), max(lo_d) / min(lo_b), same))
    L = [x[1] for x in naive_lat]
    row(rows, C, "summary", "param-/stored-matched naive pairs: Orin latency ratio range (F2 rule)", None, min(L), max(L), n=len(L),
        note="; ".join(f"{x[0].split(' [')[0]} {x[1]:.2f}" for x in naive_lat), d=3)
    row(rows, C, "summary", "same, worst/best over flags x runs (min of lo, max of hi)", None, min(x[2] for x in naive_lat), max(x[3] for x in naive_lat), d=3)
    row(rows, C, "summary", "param-/stored-matched naive pairs: energy ratio range", None, min(x[1] for x in naive_en), max(x[1] for x in naive_en),
        n=len(naive_en), note="; ".join(f"{x[0].split(' [')[0]} {x[1]:.2f} [{x[2]:.2f}-{x[3]:.2f}]" for x in naive_en), d=3)
    row(rows, C, "seed", "T1 pairs: seeds", n=1, note="every T1 dendritic model and its baseline is one trained seed (MLP h256/h512 have 3 seeds but are not in F2); accuracy SD unknown; latency/energy for that seed only")
    # T6 pairs of F2
    for num, den in (("g1f_dannG_p4_m8_s64_sg1", "g1f_mlp_h128"), ("g1f_dannG_p4_m32_s64_sg1", "g1f_mlp_h512"), ("g1f_dannG_p4_m32_s64_sg1", "g1f_mlp_h1024")):
        s = pair_build_stats(num, den)
        row(rows, C, "build", f"T6 G1 {num[4:]} / {den[4:]} median, worst, best", s["median"], s["cell_best"], s["cell_worst"], n="5 x 5",
            source="buildvar_builds_p1.csv", note="energy not measured", d=3)
    ec = {r["label"]: r for r in rcsv(f"{T6}/energy_compare.csv")}
    for nm, d, b in (("DH-SSM spk n128 / GRU h64", "dhssm_L2_n128_b4_spk__K25", "gru_h64__seq"),
                     ("DH-SSM cont n128 / GRU h256", "dhssm_L2_n128_b4_cont__K25", "gru_h256__seq"),
                     ("DH-SSM n256 / GRU h768", "dhssm_L2_n256_b4_cont__K25", "gru_h768_lr0.003_vs_t1__seq"),
                     ("AD-cache h1024 / MLP h1024", "g3_adc_h1024_ffrand__gemm", "g3_mlp_h1024__gemm"),
                     ("AD-cache h2048 / MLP h2048", "g3_adc_h2048_ffrand__gemm", "g3_mlp_h2048__gemm")):
        D, B = ec[d], ec[b]
        dl, dh = map(float, D["E_net_range"].split("-")); bl, bh = map(float, B["E_net_range"].split("-"))
        row(rows, C, "energy", f"T6 {nm}: energy ratio", float(D["E_net_mJ"]) / float(B["E_net_mJ"]), dl / bh, dh / bl, n="3 runs each",
            source="T6/energy_compare.csv", d=3)
    write("Cc_naive_pairs.csv", rows)
    return rows


# ================================================================== C-d
def g3_builds():
    p = f"{OUT}/g3_buildvar_builds.csv"
    if not os.path.exists(p):
        return None
    out = {}
    for r in rcsv(p):
        if r.get("p50"):
            out.setdefault((r["graph"], r["build"]), {})[int(r["k"])] = float(r["p50"])
    return out


def claim_d():
    rows, C = [], "C-d"
    g3 = {}
    for r in rcsv(f"{LUT}/lut_g3.csv"):
        g3.setdefault(r["cfg"], {})[r["impl"]] = r
    best = {c: min(float(v["p50_ms"]) for v in d.values()) for c, d in g3.items()}
    for h in (1024, 2048):
        a, m = f"adc_h{h}_ffrand", f"mlp_h{h}"
        row(rows, C, "build", f"h{h}: AD-cache / MLP latency, LUT (one engine per impl x flag, 3 runs)", best[a] / best[m], n="1 build/flag",
            source="T6/lut/lut_g3.csv", note=f"{best[a]} / {best[m]} ms; claimed 1.4-1.6", d=3)
        # LUT run-range extremes over all impl x flag
        def rng(c):
            v = []
            for d in g3[c].values():
                for f in ("range_default", "range_nocublas"):
                    v += [float(x) for x in d[f].split("-")]
            return v
        ra, rm = rng(a), rng(m)
        row(rows, C, "build", f"h{h}: AD-cache / MLP, LUT runs, worst/best over impl x flag x runs (fastest-cell rule)",
            None, min(ra) / max(rm), max(ra) / min(rm), d=3, note="lo = fastest AD-cache run / slowest MLP run; spans slow cells too")
    B = g3_builds()
    if B:
        for h in (1024, 2048):
            a, m = f"adc_h{h}_ffrand", f"mlp_h{h}"
            def cellv(c, fresh=True):
                out = {}
                for (g, b), d in B.items():
                    if g.startswith(c + "__"):
                        v = [d[k] for k in sorted(d) if (k > 0) == fresh]
                        if v:
                            out[(g, b)] = v
                return out
            A, M = cellv(a), cellv(m)
            ca, cm = min(A, key=lambda k: np.median(A[k])), min(M, key=lambda k: np.median(M[k]))
            va, vm = A[ca], M[cm]
            row(rows, C, "build", f"h{h}: AD-cache / MLP median of {len(va)} fresh builds (cited cells)", np.median(va) / np.median(vm),
                n=f"{len(va)} x {len(vm)}", source="robustness_B/g3_buildvar_builds.csv",
                note=f"cells {ca[0].split('__')[-1]}|{ca[1]} / {cm[0].split('__')[-1]}|{cm[1]}; AD {np.median(va):.4f} [{min(va):.4f}-{max(va):.4f}], MLP {np.median(vm):.4f} [{min(vm):.4f}-{max(vm):.4f}]", d=3)
            row(rows, C, "build", f"h{h}: AD-cache / MLP worst / best pairing, cited cells", max(va) / min(vm), min(va) / max(vm), max(va) / min(vm),
                holds=str(min(va) / max(vm) > 1), note="worst for AD-cache = slowest AD build / fastest MLP build; holds = AD-cache still slower in the best case", d=3)
            alla = [x for v in A.values() for x in v]; allm = [x for v in M.values() for x in v]
            row(rows, C, "build", f"h{h}: AD-cache / MLP best-of-builds over all impl x flag", min(alla) / min(allm), d=3)
            perk = [min(A[c][k] for c in A if len(A[c]) > k) / min(M[c][k] for c in M if len(M[c]) > k) for k in range(min(len(v) for v in list(A.values()) + list(M.values())))]
            row(rows, C, "build", f"h{h}: AD-cache / MLP per build index, fastest cell each side", None, min(perk), max(perk), n=len(perk), d=3)
            for c, CC in ((a, A), (m, M)):
                for (g, b), v in CC.items():
                    row(rows, C, "build", f"{g} {b}: fresh builds p50 (ms)", np.median(v), min(v), max(v), n=len(v),
                        note=f"spread {max(v) / min(v):.3f}; builds {' '.join(f'{x:.4f}' for x in v)}; original 10-02 engine re-timed now: "
                             f"{B[(g, b)].get(0, '')}; LUT 10-02: {g3[c][g.split('__')[-1]]['p50_' + b]}")
    else:
        row(rows, C, "build", "fresh-build session", note="not available yet (bv_g3.py still running)")
    for h in (256, 512, 1024, 2048, 4096):
        row(rows, C, "sensitivity", f"width h{h}: AD-cache ffrand / MLP (LUT best)", best[f"adc_h{h}_ffrand"] / best[f"mlp_h{h}"], d=3,
            note=f"{best[f'adc_h{h}_ffrand']} / {best[f'mlp_h{h}']}")
    for h in (1024, 2048, 4096):
        row(rows, C, "sensitivity", f"2:4 sparse gates (ff24, sparse build) h{h}: AD-cache / MLP", best[f"adc_h{h}_ff24"] / best[f"mlp_h{h}"], d=3,
            note=f"{best[f'adc_h{h}_ff24']} / {best[f'mlp_h{h}']}; ff24 accuracy not re-checked here")
    for h in (1024, 2048):
        row(rows, C, "sensitivity", f"comparator = dense MLP with context input (mlpctx) h{h}", best[f"adc_h{h}_ffrand"] / best[f"mlpctx_h{h}"], d=3)
        for impl in ("gemm", "conv"):
            r1, r2 = g3[f"adc_h{h}_ffrand"][impl], g3[f"mlp_h{h}"][impl]
            for fl in ("default", "nocublas"):
                row(rows, C, "sensitivity", f"h{h} impl {impl} flag {fl}: AD-cache / MLP", float(r1[f"p50_{fl}"]) / float(r2[f"p50_{fl}"]), d=3,
                    source="lut_g3.csv")
    # removes the overhead: T1 naive vs cached vs matched MLP
    res = {(r["model"], r["device"][:4]): r for r in rcsv(f"{T1}/results.csv")}
    en = {r["label"]: r for r in rcsv(f"{T1}/energy_orin1.csv")}
    for k, mm in (("k4", "mlp_h3592_match_k4"), ("k8", "mlp_h4965_match_k8")):
        for v in ("", "_cached"):
            dd = f"active_dend_{k}{v}"
            lat = float(res[(dd, "Orin")]["latency_p50"]) / float(res[(mm, "Orin")]["latency_p50"])
            ml = float(en[dd]["mean_ms"]) / float(en[mm]["mean_ms"])
            e = float(en[dd]["E_net_mJ"]) / float(en[mm]["E_net_mJ"])
            pi = float(res[(dd, "RPi ")]["latency_p50"]) / float(res[(mm, "RPi ")]["latency_p50"])
            row(rows, C, "sensitivity", f"T1 {dd} / {mm}: Orin DVFS p50, locked energy-window mean, energy, RPi", lat, min(lat, ml), max(lat, ml), d=3,
                note=f"locked mean {ml:.3f}; energy {e:.3f}; RPi {pi:.3f}; 1 seed")
    # seeds (accuracy side)
    fs = {r["cfg"]: r for r in rcsv(f"{T6}/fair_summary.csv") if r["group"] == "G3" and r["kind"] == "per_size"}
    for c, r in fs.items():
        row(rows, C, "seed", f"test acc {c}", float(r["test_mean"]), n=int(r["n_seeds"]), note=f"SD {r['test_std']}; latency graphs are random-weight LUT graphs (not trained weights)")
    ec = {r["label"]: r for r in rcsv(f"{T6}/energy_compare.csv")}
    for h in (1024, 2048):
        D, M = ec[f"g3_adc_h{h}_ffrand__gemm"], ec[f"g3_mlp_h{h}__gemm"]
        dl, dh = map(float, D["E_net_range"].split("-")); ml, mh = map(float, M["E_net_range"].split("-"))
        row(rows, C, "energy", f"h{h}: AD-cache / MLP energy (gemm, nocublas engine)", float(D["E_net_mJ"]) / float(M["E_net_mJ"]), dl / mh, dh / ml,
            n="3 runs each, 1 engine", source="T6/energy_compare.csv",
            note=f"{D['E_net_mJ']} vs {M['E_net_mJ']} mJ; energy-window mean latency ratio {float(D['mean_ms_energy_run']) / float(M['mean_ms_energy_run']):.3f}", d=3)
    row(rows, C, "energy", "AD-cache energy at h256/h512/h4096, ff24, conv form", note="not measured")
    write("Cd_g3_adcache.csv", rows)
    return rows


# ================================================================== C-e
def claim_e():
    rows, C = [], "C-e"
    sp = sorted(((float(r["spread_max_over_min"]), r["graph"], r["build"], r["n_builds"]) for r in BVS), reverse=True)
    for s, g, b, n in sp:
        row(rows, C, "build", f"{g} {b}: max/min p50 over fresh builds", s, n=n, source="T6/lut/buildvar_summary.csv", d=3)
    v = [s for s, *_ in sp]
    row(rows, C, "summary", "G1 + CNN graph x flag cells: spread distribution", float(np.median(v)), min(v), max(v), n=len(v),
        note=f"cells > 1.05: {sum(x > 1.05 for x in v)}; > 1.10: {sum(x > 1.10 for x in v)}; max = {sp[0][1]} {sp[0][2]}", d=3)
    B = g3_builds()
    if B:
        g3v = []
        for (g, b), d in sorted(B.items()):
            f = [d[k] for k in sorted(d) if k > 0]
            if len(f) > 1:
                g3v.append(max(f) / min(f))
                row(rows, C, "build", f"G3 {g} {b}: max/min p50 over fresh builds", max(f) / min(f), n=len(f),
                    source="robustness_B/g3_buildvar_builds.csv", d=3, note=f"orig 10-02 engine re-timed {d.get(0, '')}")
        if g3v:
            row(rows, C, "summary", "G3 graph x flag cells: spread distribution", float(np.median(g3v)), min(g3v), max(g3v), n=len(g3v), d=3)
    row(rows, C, "seed", "seed axis", note="not applicable (same ONNX graph rebuilt); G1 graphs carry seed-1 weights")
    row(rows, C, "sensitivity", "precision", note="FP16 picked only by MLP h1024 (20/20) and CNN c32_64 (9/10); the large spreads are FP32 tactic flips (xmma_gemm_as_conv1x1 vs fprop_conv1x1), see T6/lut/buildvar_builds_p*.csv")
    write("Ce_build_spread.csv", rows)
    return rows


# ================================================================== C-f
def claim_f():
    rows, C = [], "C-f"
    E = f"{EXP}/T1m/e10"
    for r in rcsv(f"{E}/matched.csv"):
        if r["model"] in ("dann_d16_s64", "dann_prng_d20_s128", "dann_d8_s32", "dann_d8_s64", "dann_prng_d32_s96"):
            row(rows, C, "sensitivity", f"{r['variant']}: {r['model']} -> {r['smallest_matched_cnn']} time ratio CNN/X", float(r["time_ratio_cnn_over_x"] or "nan"),
                source="T1m/e10/matched.csv", note=f"acc {r['acc_mean']}+-{r['acc_sd']} vs {r['cnn_acc_mean']}+-{r['cnn_acc_sd']}; ucsim ratio {r['ucsim_ratio_cnn_over_x']}; X ms {r['board_ms_source']}", d=3)
    for r in rcsv(f"{E}/results.csv"):
        row(rows, C, "seed", f"{r['model']} int8 test", float(r["test_int8_mean"]), n=3, source="T1m/e10/results.csv",
            note=f"SD ddof1 {r['test_int8_sd']} (ddof0 {r['test_int8_sd_ddof0']}); ucsim first-run->E10 build {r['ucsim_e10build_rel_change']}; board {r['board_ms']} ms ({r['board_ms_source']})")
    row(rows, C, "build", "TRT builds", note="not applicable; the analogue (first-run vs E-10 build ucsim cycles) is -0.46 % to +3.18 %, fronts unchanged; board/ucsim factors confirmed within 0.75 % on 4 new boards")
    row(rows, C, "energy", "8051 energy", note="not measured on any 8051 model (no current measurement); latency = board cycles at 35 MHz or ucsim estimate")
    write("Cf_8051_summary.csv", rows)
    return rows


# ================================================================== energy coverage
def coverage():
    M = [  # family, tier, energy_measured, configs with energy, latency measured, source / note
        ("dANN-LRF (T1, mask)", "Orin", "yes", "dann_lrf_d32_s128", "yes", "T1/energy_orin1.csv, 3 runs, locked"),
        ("vANN same-shape (T1)", "Orin", "yes", "vann_same_d32_s128", "yes", "T1/energy_orin1.csv"),
        ("MLP G1 (T1 recipe)", "Orin", "yes", "h85, h256, h512, h1024", "yes", "T1/energy_orin1.csv"),
        ("dANN-G grouped GEMM (T6 G1 fair)", "Orin", "no", "-", "yes (5 builds)", "fair_summary orin_energy = not_measured"),
        ("MLP G1 fair (T6)", "Orin", "no", "-", "yes (5 builds)", ""),
        ("CNN G1 fair (T6)", "Orin", "no", "-", "yes (5 builds)", ""),
        ("dann24 / dense_D / dANN-G sp24 LUT (T6 G1)", "Orin", "no", "-", "yes", ""),
        ("sparse MLP h1024 (E-4)", "Orin", "no", "-", "no", "accuracy only"),
        ("DH-SRNN (T1)", "Orin", "yes", "val seqloop, step", "yes", ""),
        ("LIF-SRNN (T1)", "Orin", "yes", "val seqloop, step", "yes", ""),
        ("GRU T1 (pm/acc; loop, fused, step)", "Orin", "yes", "all 5 seq + 2 step", "yes", ""),
        ("DH-SSM (T6 G2)", "Orin", "partial", "L2 n128 b4 spk/cont K25, L2 n256 b4 cont K25 (lut_g2 graphs)", "yes (LUT, 1 engine/flag)", "no energy for K50/K100 or other configs"),
        ("diag-SSM dssmU (T6 G2)", "Orin", "partial", "L2 n128 b4 spk K50 (seed 1)", "partial", ""),
        ("LRU (T6 G2)", "Orin", "partial", "L2 H64 N128 K50 (seed 1)", "partial", "H128 N256 not timed"),
        ("GRU T6 G2", "Orin", "partial", "h64, h256 (lut_g2), h768 (seed 1)", "partial", "h128 no energy; h512 not timed"),
        ("LSTM (T6 G2)", "Orin", "no", "-", "no", ""),
        ("AD naive / proto / cached (T1 G3)", "Orin", "yes", "k4, k8 x 3 variants", "yes", ""),
        ("AD-cache (T6 G3)", "Orin", "partial", "h1024, h2048 ffrand gemm", "yes (LUT; 5 builds h1024/h2048 in this audit)", "h256/h512/h4096, ff24, conv: no energy"),
        ("MLP G3 (T6)", "Orin", "partial", "h1024, h2048, h4096 gemm", "yes", ""),
        ("mlpctx (T6 G3)", "Orin", "no", "-", "yes (LUT)", ""),
        ("all T1 models", "RPi 4B", "no", "-", "yes (ORT FP32, 3 threads)", "no power meter on the Pi"),
        ("all T6 models", "RPi 4B", "no", "-", "no", "T6 has no Pi timing"),
        ("dANN / PRNG-dANN / MLP / CNN (T1m, E-10)", "8051", "no", "-", "yes (9 board, 15 ucsim est.)", "no current measurement"),
    ]
    p = f"{OUT}/energy_coverage.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "tier", "energy_measured", "configs_with_energy", "latency_measured", "note"]); w.writerows(M)
    print(len(M), "rows ->", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    claim_a(); claim_b(); claim_c(); claim_d(); claim_e(); claim_f(); coverage()
