"""cnnvar report (PROTOCOL_cnnvar.md §6): results / matched / fronts with every CNN family.

Inputs (read only): results/T1m/e10/{results.csv, matched.csv, runs/} (the 24 E-10/E-10b rows; plain CNNs),
                    results/T1m/e10/cnnvar/{runs/, verify.json, ucsim.json} (the 15 new CNNs).
Outputs: results/T1m/e10/cnnvar/{results.csv, matched.csv, fronts.csv}; summary printed.
New-CNN board ms = ucsim cycles x 0.854 / 35.004 MHz (ESTIMATE). Plain-CNN and dANN/MLP board ms = e10/results.csv.
cnns4_c2_4 / cnns4_c4_8 are board-timed (src/mcu/flash_e10_board.sh, seed-1 cnnvar build):
board_ms = median board cycles / trimmed MHz; the estimate is kept in board_ms_estimate. matched.csv also carries the
4 variants recomputed without cnns4_c2_4 (suffix _no_s4c2_4).
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnnvar as V  # noqa: E402

E10 = f"{V.ROOT}/results/T1m/e10"
CV = V.CV
CNN_K = 0.854
F_KHZ = 35004.0
T_RULE = 2.48
BOARD = {"cnns4_c2_4": "cnns4_c2_4_lr0.01_wd0_t1", "cnns4_c4_8": "cnns4_c4_8_lr0.003_wd0_t1"}


def board_meas(m):
    """Board result for a board-timed new CNN: (median cycles, trimmed MHz, ms, bit-exact n/N)."""
    import re
    rows = list(csv.DictReader(open(f"{V.ROOT}/results/T1m/board_{m}_t1.csv")))
    log = open(f"{V.ROOT}/results/T1m/logs/stcgal_{m}_t1.log").read()
    assert BOARD[m] in log, m   # flashed from the cnnvar seed-1 build
    f = float(re.findall(r"Adjusted frequency:\s*([0-9.]+)", log)[-1])
    cyc = float(np.median([int(r["cycles"]) for r in rows]))
    ok = sum(r["logits_match"] == "True" for r in rows)
    return cyc, f, round(cyc / (f * 1e3), 2), f"{ok}/{len(rows)}"
VARIANTS = [("primary_int8val_ddof1", "val_int8", "sd1"), ("int8val_ddof0", "val_int8", "sd0"),
            ("fp32val_ddof1", "val_fp32", "sd1"), ("fp32val_ddof0", "val_fp32", "sd0")]


def accs_from_runs(rundir, models):
    import e10
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{rundir}/*.json"))]
    cs = e10.cells(runs)
    out = {}
    for m in models:
        out[m] = {}
        for key in ("val_int8", "val_fp32"):
            s = e10.select(cs, m, key)
            seeds = s["test_int8_seeds"]
            out[m][key] = dict(mean=s["test_int8"], sd1=s["test_int8_sd"], sd0=float(np.std(seeds)), cell=s)
    return out, cs


def main():
    import e10
    R = {r["model"]: r for r in csv.DictReader(open(f"{E10}/results.csv"))}
    old_models = list(R)
    plain = [m for m in old_models if m.startswith("cnn_")]
    noncnn = [m for m in old_models if not m.startswith("cnn_")]
    A_old, _ = accs_from_runs(f"{E10}/runs", old_models)
    for m in old_models:  # consistency with e10/results.csv
        assert abs(A_old[m]["val_int8"]["mean"] - float(R[m]["test_int8_mean"])) < 6e-5, m
    A_new, cs_new = accs_from_runs(f"{CV}/runs", V.MODELS)
    ver = json.load(open(f"{CV}/verify.json"))
    uc = json.load(open(f"{CV}/ucsim.json"))

    # ---------------------------------------------------------------- results.csv (15 new CNNs)
    rows = []
    for m in V.MODELS:
        s = A_new[m]["val_int8"]["cell"]
        sf = A_new[m]["val_fp32"]["cell"]
        lrs = sorted({c["lr"] for c in cs_new if c["model"] == m})
        v1 = ver[m]["1"]
        u = uc[m]
        fam, ws = V.arch(m)
        rows.append(dict(model=m, family=fam, widths="_".join(map(str, ws)), sel_lr=s["lr"], sel_wd=s["wd"],
                         lr_grid=" ".join(f"{x:g}" for x in lrs), sel_at_grid_top=s["lr"] == max(lrs),
                         val_int8_mean=round(s["val_int8"], 4), test_int8_mean=round(s["test_int8"], 4),
                         test_int8_sd=round(s["test_int8_sd"], 4), test_int8_sd_ddof0=round(s["test_int8_sd0"], 4),
                         test_int8_seeds=" ".join(f"{x:.4f}" for x in s["test_int8_seeds"]),
                         test_fp32_mean=round(s["test_fp32"], 4), fp32sel_lr=sf["lr"], fp32sel_wd=sf["wd"],
                         fp32sel_test_int8_mean=round(sf["test_int8"], 4), fp32sel_test_int8_sd=round(sf["test_int8_sd"], 4),
                         flash_bytes=v1["flash_bytes"], xdata_bytes=v1["xdata_bytes"], fits=v1["fits"],
                         table_bytes=v1["table_bytes"], offsets=" ".join(map(str, v1["offsets"])),
                         shifts=" ".join(map(str, v1["shifts"])),
                         pc_bitexact_seeds123=" ".join(ver[m][k]["pc_bitexact_logits"] for k in ("1", "2", "3")),
                         ucsim_cycles=round(u["cycles"]), ucsim_n=u["n"], ucsim_bitexact=u["ucsim_bitexact_logits"],
                         macs=u["macs"], ucsim_cycles_per_mac=round(u["cycles"] / u["macs"], 1),
                         board_ms=round(u["cycles"] * CNN_K / F_KHZ, 2),
                         board_ms_source=f"ESTIMATE_ucsim_x{CNN_K}_{F_KHZ / 1e3:g}MHz",
                         board_ms_estimate=round(u["cycles"] * CNN_K / F_KHZ, 2),
                         board_cycles_median="", board_mhz="", board_bitexact="", board_over_ucsim="",
                         est_error_pct=""))
        if m in BOARD:
            cyc, f, ms, ok = board_meas(m)
            assert ok == "100/100", (m, ok)
            rows[-1].update(board_ms=ms, board_ms_source=f"measured_seed1_cnnvarbuild_{f:g}MHz", board_cycles_median=cyc,
                            board_mhz=f, board_bitexact=ok, board_over_ucsim=round(cyc / u["cycles"], 4),
                            est_error_pct=round(100 * (rows[-1]["board_ms_estimate"] - ms) / ms, 2))
    with open(f"{CV}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    RN = {r["model"]: r for r in rows}

    # ---------------------------------------------------------------- CNN pool, cycle order
    cnn = {}
    for m in plain:
        cnn[m] = dict(family="plain", cycles=float(R[m]["ucsim_cycles_e10build"]), ms=float(R[m]["board_ms"]),
                      ms_src=R[m]["board_ms_source"], flash=int(R[m]["flash_bytes_e10"]), A=A_old[m])
    for m in V.MODELS:
        if not RN[m]["fits"]:
            continue
        cnn[m] = dict(family=RN[m]["family"], cycles=float(uc[m]["cycles"]), ms=RN[m]["board_ms"],
                      ms_src=RN[m]["board_ms_source"], flash=int(RN[m]["flash_bytes"]), A=A_new[m])
    order_all = sorted(cnn, key=lambda m: cnn[m]["cycles"])
    order_plain = [m for m in order_all if m in plain]
    assert order_plain == sorted(plain, key=lambda m: ["cnn_c2_4", "cnn_c3_6", "cnn_c4_8", "cnn_c6_12", "cnn_c8_16",
                                                       "cnn_c16_32", "cnn_c32_64"].index(m))

    def match(x, key, dd, order):
        ax = A_old[x][key]
        for c in order:
            ac = cnn[c]["A"][key]
            thr = T_RULE * max(ax[dd], ac[dd])
            if ac["mean"] >= ax["mean"] - thr:
                return c, ax["mean"] - ac["mean"], thr
        return None, None, None

    old_matched = {(r["variant"], r["model"]): r for r in csv.DictReader(open(f"{E10}/matched.csv"))}
    mrows = []
    order_all_full = order_all
    for variant_full, key, dd in VARIANTS + [(v + "_no_s4c2_4", k, d) for v, k, d in VARIANTS]:
        variant = variant_full.replace("_no_s4c2_4", "")
        order_all = [m for m in order_all_full if not (variant_full.endswith("_no_s4c2_4") and m == "cnns4_c2_4")]
        for x in noncnn:
            ms_x, fl_x, cy_x = float(R[x]["board_ms"]), int(R[x]["flash_bytes_e10"]), float(R[x]["ucsim_cycles_e10build"])
            cp, _, _ = match(x, key, dd, order_plain)
            ca, gap, thr = match(x, key, dd, order_all)
            o = old_matched[(variant, x)]
            assert (cp or "none") == o["smallest_matched_cnn"], (variant, x, cp, o["smallest_matched_cnn"])
            r = dict(variant=variant_full, model=x, kind=R[x]["model"].split("_")[0] if not x.startswith("dann_prng") else "prng",
                     acc_mean=round(A_old[x][key]["mean"], 4), acc_sd=round(A_old[x][key][dd], 4),
                     board_ms=ms_x, board_ms_source=R[x]["board_ms_source"], flash_x=fl_x,
                     plain_matched_cnn=cp or "none",
                     plain_time_ratio=round(cnn[cp]["ms"] / ms_x, 3) if cp else "",
                     plain_time_ratio_e10matched_csv=o["time_ratio_cnn_over_x"],
                     plain_flash_share=round(cnn[cp]["flash"] / fl_x, 3) if cp else "",
                     all_matched_cnn=ca or "none", all_matched_family=cnn[ca]["family"] if ca else "",
                     all_cnn_acc_mean=round(cnn[ca]["A"][key]["mean"], 4) if ca else "",
                     all_cnn_acc_sd=round(cnn[ca]["A"][key][dd], 4) if ca else "",
                     gap_x_minus_cnn=round(gap, 4) if ca else "", threshold=round(thr, 4) if ca else "",
                     all_cnn_board_ms=cnn[ca]["ms"] if ca else "", all_cnn_board_ms_source=cnn[ca]["ms_src"] if ca else "",
                     all_time_ratio=round(cnn[ca]["ms"] / ms_x, 3) if ca else "",
                     all_ucsim_ratio=round(cnn[ca]["cycles"] / cy_x, 3) if ca else "",
                     all_flash_cnn=cnn[ca]["flash"] if ca else "",
                     all_flash_share=round(cnn[ca]["flash"] / fl_x, 3) if ca else "",
                     changed=(ca != cp))
            # smallest matched CNN per family (for the flashing list)
            for fam in ("plain", "pool", "ds", "s4"):
                cf, _, _ = match(x, key, dd, [m for m in order_all if cnn[m]["family"] == fam])
                r[f"match_{fam}"] = cf or "none"
                r[f"ratio_{fam}"] = round(cnn[cf]["ms"] / ms_x, 3) if cf else ""
            # exploratory diagnostic (not pre-specified; added after seeing the SD-driven matches): smallest CNN whose
            # mean is >= mean_X (no SD tolerance), plain-only and all families
            for nm, order in (("plain", order_plain), ("all", order_all)):
                cd = next((c for c in order if cnn[c]["A"][key]["mean"] >= A_old[x][key]["mean"]), None)
                r[f"diag_meange_{nm}"] = cd or "none"
                r[f"diag_meange_{nm}_ratio"] = round(cnn[cd]["ms"] / ms_x, 3) if cd else ""
                r[f"diag_meange_{nm}_flash_share"] = round(cnn[cd]["flash"] / fl_x, 3) if cd else ""
            mrows.append(r)
    with open(f"{CV}/matched.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrows[0]))
        w.writeheader()
        w.writerows(mrows)

    # ---------------------------------------------------------------- fronts
    import fronts_e10 as FR
    T = FR.table(True)
    for m in V.MODELS:
        T.append(dict(model=m, kind="cnn", mask_source=f"cnnvar_{RN[m]['family']}", acc=RN[m]["test_int8_mean"],
                      sd=RN[m]["test_int8_sd"], sd_kind="ddof1", flash=int(RN[m]["flash_bytes"]),
                      cycles=float(RN[m]["ucsim_cycles"]), src="T1m/e10/cnnvar/results.csv"))
    for o in T:
        o["fits"] = int(o["flash"] <= FR.FLASH_MAX)
    dep = [o for o in T if o["fits"]]
    pools = {"nocnn": [o for o in dep if o["kind"] != "cnn"],
             "cnnplain": [o for o in dep if o["kind"] != "cnn" or o["mask_source"] == "cnn"],
             "cnnall": dep}
    COST = {"F": ("flash",), "C": ("cycles",), "3": ("flash", "cycles")}
    for pn, pool in pools.items():
        for fn, c in COST.items():
            mem = FR.front(pool, c)
            for o in T:
                o[f"front_{fn}_{pn}"] = int(o["model"] in mem)
    old_fr = {r["model"]: r for r in csv.DictReader(open(f"{E10}/fronts_e10.csv"))}
    for o in T:   # the plain pool must reproduce e10/fronts_e10.csv
        if o["model"] in old_fr:
            for fn in COST:
                assert int(old_fr[o["model"]][f"front_{fn}_cnn"]) == o[f"front_{fn}_cnnplain"], (o["model"], fn)
                assert int(old_fr[o["model"]][f"front_{fn}_nocnn"]) == o[f"front_{fn}_nocnn"], (o["model"], fn)
    with open(f"{CV}/fronts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(T[0]))
        w.writeheader()
        for o in sorted(T, key=lambda o: o["flash"]):
            w.writerow(o)

    # ---------------------------------------------------------------- summary
    sh = lambda m: m.replace("dann_prng_", "PRNG ").replace("dann_", "").replace("mlp_", "mlp ").replace("cnn_", "")
    print("== new CNNs (int8 test, 3 seeds, sample SD; board ms = ESTIMATE ucsim x 0.854 / 35.004 MHz unless board-timed)")
    for m in BOARD:
        r = RN[m]
        print(f"  BOARD {m}: cycles {r['board_cycles_median']} @ {r['board_mhz']} MHz -> {r['board_ms']} ms "
              f"(est {r['board_ms_estimate']}, err {r['est_error_pct']}%), board/ucsim {r['board_over_ucsim']}, "
              f"bit-exact {r['board_bitexact']}")
    for r in rows:
        print(f"  {r['model']:16s} {r['sel_lr']:g}/{r['sel_wd']:g}{' TOP' if r['sel_at_grid_top'] else ''}"
              f" acc {100 * r['test_int8_mean']:.2f}±{100 * r['test_int8_sd']:.2f} flash {r['flash_bytes']} "
              f"cyc {r['ucsim_cycles']} ({r['ucsim_cycles_per_mac']}/MAC) est {r['board_ms']} ms pc {r['pc_bitexact_seeds123']}"
              f" ucsim {r['ucsim_bitexact']}")
    order_all = order_all_full
    for variant, _, _ in VARIANTS + [(v + "_no_s4c2_4", k, d) for v, k, d in VARIANTS]:
        print(f"== matched, {variant}")
        V_ = [r for r in mrows if r["variant"] == variant]
        for r in V_:
            print(f"  {sh(r['model']):14s} {100 * r['acc_mean']:.2f}±{100 * r['acc_sd']:.2f}  plain {sh(r['plain_matched_cnn'])}"
                  f" {r['plain_time_ratio']} (flash {r['plain_flash_share']})  ->  all {r['all_matched_cnn']} "
                  f"{r['all_time_ratio']} (ucsim {r['all_ucsim_ratio']}, flash {r['all_flash_share']}){'  CHANGED' if r['changed'] else ''}")
        for r in V_:
            print(f"    diag mean>=X: {sh(r['model']):14s} plain {sh(r['diag_meange_plain'])} {r['diag_meange_plain_ratio']}"
                  f" (flash {r['diag_meange_plain_flash_share']}) -> all {sh(r['diag_meange_all'])} {r['diag_meange_all_ratio']}"
                  f" (flash {r['diag_meange_all_flash_share']})")
        for grp, pred in (("MLP", lambda r: r["model"].startswith("mlp_")), ("dANN", lambda r: r["model"].startswith("dann_"))):
            g = [r for r in V_ if pred(r) and r["all_time_ratio"] != ""]
            gp = [r for r in V_ if pred(r) and r["plain_time_ratio"] != ""]
            print(f"  {grp}: plain ratio {min(r['plain_time_ratio'] for r in gp):.3f}-{max(r['plain_time_ratio'] for r in gp):.3f}"
                  f" | all-family ratio {min(r['all_time_ratio'] for r in g):.3f}-{max(r['all_time_ratio'] for r in g):.3f}"
                  f" | all-family flash share {min(r['all_flash_share'] for r in g):.3f}-{max(r['all_flash_share'] for r in g):.3f}"
                  f" | ratio>1: {[sh(r['model']) for r in g if r['all_time_ratio'] > 1]}")
    for pn in pools:
        for fn in COST:
            k = f"front_{fn}_{pn}"
            mem = sorted([o for o in T if o[k]], key=lambda o: o["flash" if fn != "C" else "cycles"])
            print(f"{k}: {len(mem)}: " + ", ".join(f"{sh(o['model'])} ({100 * o['acc']:.2f})" for o in mem))


if __name__ == "__main__":
    main()
