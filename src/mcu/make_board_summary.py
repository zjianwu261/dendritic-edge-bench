"""Regenerate results/T1m/board_summary.csv from the raw board measurements.

Inputs (read-only):
  results/T1m/board_<tag>.csv          host_uart.py output, one row per test image (cycles = Timer0 cycles
                                           minus the empty-timing overhead; its `ms` column used the --f_mhz given
                                           on the command line and is NOT used here)
  results/T1m/logs/stcgal_<tag>.log    the LAST "Adjusted frequency: X MHz" line = the trimmed SYSclk the
                                           firmware ran at (the "Target frequency" line near the top is the
                                           previous flash's reading)
  results/T1m/results_all.csv          per-seed int8 / fp32 accuracy (10k test images), flash, ucsim cycles
  results/T1m/results.csv              3-seed mean int8 accuracy
  results/T1m/cnn/<cfg>_t<seed>.json    per-seed CNN numbers (int8/fp32 acc, flash, ucsim cycles) for cnn_* tags
  results/T1m/cnn/cnn_results.csv       3-seed mean CNN int8 accuracy
Outputs:
  results/T1m/board_summary.csv        one row per flashed model
  results/T1m/board_per_image.csv      all images of all models, ms recomputed at the stcgal frequency
Usage: python make_board_summary.py
E-10b board runs (2026-10-25): the tags in E10B_BUILD were flashed from the E-10b selected-cell seed-1 build
(results/T1m/e10/build/<cell>_t1, src/mcu/flash_e10_board.sh), not the first-run build/<tag>. For them the per-seed
acc comes from e10/runs/<cell>_t1.json, mean/SD (ddof 1) from e10/results.csv, flash from e10/verify.json and ucsim from
e10/ucsim.json (cycles_e10), so board_over_ucsim pairs the same binary. Column `build` says which build was timed.
"""
import csv
import glob
import json
import os
import re
import statistics as st

EXP = os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results/T1m")


E10 = f"{EXP}/e10"
E10B_BUILD = {"dann_d8_s32_t1": "dann_d8_s32_lr0.01_wd0_t1", "dann_d8_s64_t1": "dann_d8_s64_lr0.003_wd0_t1",
              "cnn_c2_4_t1": "cnn_c2_4_lr0.003_wd0.01_t1", "cnn_c3_6_t1": "cnn_c3_6_lr0.003_wd0.01_t1"}


def e10b_info(tag, model):
    cell = E10B_BUILD[tag]
    j = json.load(open(f"{E10}/runs/{cell}.json"))
    r = {x["model"]: x for x in csv.DictReader(open(f"{E10}/results.csv"))}[model]
    ver, uc = json.load(open(f"{E10}/verify.json"))[model], json.load(open(f"{E10}/ucsim.json"))[model]
    assert ver["tag"] == cell and uc["tag"] == cell, (tag, ver["tag"], uc["tag"])
    ms = "cnn" if model.startswith("cnn_") else j["mask_source"]
    return ({"mask_source": ms, "acc_int8": j["test_int8"], "acc_fp32": round(j["test_fp32"], 4), "flash_bytes": ver["flash_bytes"],
             "cycles_ucsim_mc": uc["cycles_e10"]},
            {"acc_int8": r["test_int8_mean"], "acc_int8_std": r["test_int8_sd"], "n_seeds": len(r["test_int8_seeds"].split())},
            f"e10b:{cell}")


def adjusted_mhz(tag):
    txt = open(f"{EXP}/logs/stcgal_{tag}.log").read()
    m = re.findall(r"Adjusted frequency:\s*([0-9.]+)\s*MHz", txt)
    assert m, f"no 'Adjusted frequency' in stcgal_{tag}.log"
    return float(m[-1])


def main():
    allr = {r["tag"]: r for r in csv.DictReader(open(f"{EXP}/results_all.csv"))}
    mean = {r["model"]: r for r in csv.DictReader(open(f"{EXP}/results.csv"))}
    # CNN tags (cnn_c8_16_t1 ...) live in results/T1m/cnn/, not results_all.csv
    for r in csv.DictReader(open(f"{EXP}/cnn/cnn_results.csv")):
        mean[r["model"]] = {"acc_int8": r["acc_int8"], "acc_int8_std": r["acc_int8_std"], "n_seeds": r["n_seeds"]}
        for jf in glob.glob(f"{EXP}/cnn/{r['cfg']}_t*.json"):
            j = json.load(open(jf))
            allr.setdefault(j["tag"], {"mask_source": "cnn", "acc_int8": j["acc_int8"], "acc_fp32": j["acc_fp32"],
                                       "flash_bytes": j.get("flash_bytes", -1), "cycles_ucsim_mc": j.get("cycles_ucsim_mc", "nan")})
    rows, per_img = [], []
    for f in sorted(glob.glob(f"{EXP}/board_*_t*.csv")):
        tag = os.path.basename(f)[len("board_"):-len(".csv")]
        if not re.search(r"_t\d+$", tag):
            continue
        model, seed = re.match(r"(.+)_t(\d+)$", tag).groups()
        b = list(csv.DictReader(open(f)))
        fm = adjusted_mhz(tag)
        cyc = [int(r["cycles"]) for r in b]
        n = len(b)
        n_bit = sum(r["logits_match"] == "True" for r in b)
        n_cor = sum(int(r["cls"]) == int(r["label"]) for r in b)
        for r in b:
            per_img.append({"tag": tag, "i": r["i"], "label": r["label"], "cls": r["cls"], "logits_match": r["logits_match"],
                            "cycles": r["cycles"], "f_mhz_stcgal": fm, "ms": round(int(r["cycles"]) / (fm * 1e3), 4)})
        if tag in E10B_BUILD:
            a, m, build = e10b_info(tag, model)
        else:
            a, m, build = allr[tag], mean[model], "first-run"
        med = st.median(cyc)
        uc = float(a["cycles_ucsim_mc"])
        rows.append({
            "model": tag, "seed": int(seed), "mask_source": a["mask_source"],
            "acc_int8_seed": float(a["acc_int8"]),          # the timed model's own int8 accuracy, 10k test images
            "acc_fp32_seed": float(a["acc_fp32"]),
            "acc_int8_mean3": float(m["acc_int8"]), "acc_int8_std3": float(m["acc_int8_std"]), "n_seeds_mean": int(m["n_seeds"]),
            "board_acc_n": f"{n_cor}/{n}",                  # board class == label on the first n test images
            "flash_bytes": int(a["flash_bytes"]), "f_mhz": fm,
            "board_cycles_median": int(med) if med == int(med) else med,
            "board_cycles_min": min(cyc), "board_cycles_max": max(cyc),
            "board_ms": round(med / (fm * 1e3), 2),
            "ucsim_cycles_mc": int(round(uc)), "board_over_ucsim": round(med / uc, 3),
            "bitexact_n": f"{n_bit}/{n}", "build": build,
        })
    keys = list(rows[0])
    with open(f"{EXP}/board_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(f"{EXP}/board_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_img[0])); w.writeheader(); w.writerows(per_img)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
