"""C-c multi-seed tables (2026-10-26). Reads only files written by ms_train.py / ms_orin.py (+ the seed-1 T1
model metadata) and writes results/robustness_B/Cc_multiseed/*.csv:
  accuracy_seeds.csv     model x seed test accuracy (+ val acc / best epoch for the validation-selected SHD models)
  accuracy_summary.csv   model: mean, SD (n-1), min, max over seeds
  latency_engines.csv    session x seed x model x build: p50 (median of locked runs), min/max run p50, runs used
  energy_engines.csv     seed x model: E_net (median of 3 good runs), range, per-run flags (energy_parse.per_run)
  pairs_seed.csv         pair x seed: latency ratio (median over builds), build-extreme range, energy ratio, accuracies
  pairs_summary.csv      pair: per-seed ratios, across-seed range, accuracy mean +- SD, verdicts
A timed run is used only if trt_sync_bench saw 624.75 MHz on every sample and both env dumps show min=cur=max=624.75 MHz
(same rule as src/t6/e7_g1_orin.run_locked).
Usage: python ms_table.py
"""
import csv
import glob
import json
import os
import statistics as stt
import sys

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = f"{ROOT}/results/robustness_B/Cc_multiseed"
T1M = f"{ROOT}/results/T1/models"
LOCK_HZ = 624750000
sys.path.insert(0, f"{ROOT}/src/bench")

# graph label -> metadata json (the trained weights it was exported from)
META = {"dann_lrf_d32_s128": "dann_lrf_d32_s128", "mlp_h85_match_d32_s128": "mlp_h85_match_d32_s128",
        "vann_same_d32_s128": "vann_same_d32_s128", "mlp_h1024_fmnist": "mlp_h1024_fmnist",
        "dhsrnn_n64_b8_val_seqloop": "dhsrnn_n64_b8_val", "lif_srnn_n64_val_seqloop": "lif_srnn_n64_val",
        "gru_pm_val_seqloop": "gru_pm_val", "gru_acc_val_seqloop": "gru_acc_val", "gru_pm_val_seq": "gru_pm_val",
        "gru_acc_val_seq": "gru_acc_val", "dhsrnn_n64_b8_step": "dhsrnn_n64_b8", "lif_srnn_n64_step": "lif_srnn_n64",
        "gru_h23_step": "gru_h23", "gru_acc_val_step": "gru_acc_val",
        "active_dend_k4": "active_dend_k4", "active_dend_k8": "active_dend_k8", "mlp_h3592_match_k4": "mlp_h3592_match_k4",
        "mlp_h4965_match_k8": "mlp_h4965_match_k8", "mlp_h3941_matchstored_k4": "mlp_h3941_matchstored_k4",
        "mlp_h5230_matchstored_k8": "mlp_h5230_matchstored_k8", "active_dend_k4_cached": "active_dend_k4",
        "active_dend_k8_cached": "active_dend_k8"}
# (pair name, dendritic graph, baseline graph, kind, in C-c naive matched set, in F2)
PAIRS = [
    ("dANN d32 / MLP h85", "dann_lrf_d32_s128", "mlp_h85_match_d32_s128", "param-matched", True, True),
    ("dANN d32 / vANN", "dann_lrf_d32_s128", "vann_same_d32_s128", "same layer shape", False, True),
    ("dANN d32 / MLP h1024", "dann_lrf_d32_s128", "mlp_h1024_fmnist", "accuracy-oriented", False, True),
    ("DH-SRNN / LIF-SRNN (loop)", "dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val_seqloop", "same neurons", False, True),
    ("DH-SRNN / GRU-pm (loop)", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seqloop", "param-matched", True, True),
    ("DH-SRNN / GRU-pm (fused op)", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seq", "param-matched", True, False),
    ("DH-SRNN / GRU-acc (loop)", "dhsrnn_n64_b8_val_seqloop", "gru_acc_val_seqloop", "accuracy-oriented", False, False),
    ("DH-SRNN / GRU-acc (fused)", "dhsrnn_n64_b8_val_seqloop", "gru_acc_val_seq", "accuracy-oriented", False, True),
    ("DH-SRNN / LIF (step)", "dhsrnn_n64_b8_step", "lif_srnn_n64_step", "same neurons", False, True),
    ("DH-SRNN / GRU h23 (step)", "dhsrnn_n64_b8_step", "gru_h23_step", "param-matched", True, False),
    ("DH-SRNN / GRU-acc (step)", "dhsrnn_n64_b8_step", "gru_acc_val_step", "accuracy-oriented", False, True),
    ("AD k4 / MLP h3592", "active_dend_k4", "mlp_h3592_match_k4", "param-matched", True, True),
    ("AD k8 / MLP h4965", "active_dend_k8", "mlp_h4965_match_k8", "param-matched", True, True),
    ("AD k4 / MLP h3941", "active_dend_k4", "mlp_h3941_matchstored_k4", "stored-matched", True, False),
    ("AD k8 / MLP h5230", "active_dend_k8", "mlp_h5230_matchstored_k8", "stored-matched", True, False),
    ("AD k4 cached / MLP h3592", "active_dend_k4_cached", "mlp_h3592_match_k4", "param-matched, deployment variant", False, False),
    ("AD k8 cached / MLP h4965", "active_dend_k8_cached", "mlp_h4965_match_k8", "param-matched, deployment variant", False, False),
]
SEEDS = (1, 2, 3)


def meta(seed, name):
    p = f"{T1M}/{name}.json" if seed == 1 else f"{OUT}/models/s{seed}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", path)


def sd(v):
    return stt.stdev(v) if len(v) > 1 else float("nan")


def accuracy():
    rows, acc = [], {}
    for name in dict.fromkeys(META.values()):
        for s in SEEDS:
            m = meta(s, name)
            if not m:
                continue
            a = float(m["test_acc"])
            acc[(name, s)] = a
            rows.append({"model": name, "seed": s, "train_seed_arg": m.get("trial", m.get("seed", (m.get("args") or {}).get("seed"))),
                         "test_acc": round(a, 5), "val_acc": round(m["val_acc"], 5) if "val_acc" in m else "",
                         "best_epoch": m.get("best_epoch", ""), "train_minutes": m.get("train_minutes", ""),
                         "device": "T1 original" if s == 1 else "CPU (ms_train.py)",
                         "source": f"results/T1/models/{name}.json" if s == 1 else f"Cc_multiseed/models/s{s}/{name}.json"})
    write(f"{OUT}/accuracy_seeds.csv", rows)
    summ = []
    for name in dict.fromkeys(META.values()):
        v = [acc[(name, s)] for s in SEEDS if (name, s) in acc]
        if v:
            summ.append({"model": name, "n_seeds": len(v), "mean": round(stt.mean(v), 5), "sd": round(sd(v), 5),
                         "min": round(min(v), 5), "max": round(max(v), 5), "per_seed": " ".join(f"{x:.4f}" for x in v)})
    write(f"{OUT}/accuracy_summary.csv", summ)
    return acc


def gpu_states(envf):
    out = []
    if not os.path.exists(envf):
        return out
    L = open(envf).read().splitlines()
    for i, l in enumerate(L):
        if l.startswith("## gpu devfreq governor/cur/min/max"):
            try:
                out.append((int(L[i + 3]), int(L[i + 2]), int(L[i + 4])))  # min, cur, max
            except (ValueError, IndexError):
                out.append(None)
    return out


def run_ok(r, raw):
    base = r["engine"][:-len(".engine")]
    st = gpu_states(f"{raw}/env/{base}_r{r['run']}.txt")
    env_ok = len(st) >= 2 and all(x == (LOCK_HZ,) * 3 for x in st)
    return env_ok and r.get("gpu_freq_timed_frac_at_max") == 1.0 and r.get("gpu_freq_hz_timed_min_med_max") == [LOCK_HZ] * 3


def latency():
    rows, lat = [], {}
    for sess0 in ("A", "B", "C"):  # B (G1/G3) and C (G2) are the two halves of the seed-2/3 session: both -> "B"
        sess = "A" if sess0 == "A" else "B"
        raw = f"{OUT}/raw_orin1_{sess0}"
        if not os.path.exists(f"{raw}/results.jsonl"):
            continue
        info = json.load(open(f"{raw}/onnx_info.json"))
        res = [json.loads(l) for l in open(f"{raw}/results.jsonl") if l.strip()]
        for key, v in info.items():
            for k in v["builds"]:
                lab = f"s{v['seed']}b{k}_{v['model']}"
                rs0 = [r for r in res if r["label"] == lab and r["onnx_sha256"] == v["onnx_sha256"]]
                rs = [r for r in rs0 if run_ok(r, raw)]
                rec = {"session": sess0, "seed": v["seed"], "model": v["model"], "build": k,
                       "origin": "original T1 engine (trt_v07)" if k == 0 else "fresh build",
                       "nruns": len(rs), "n_excluded_unlocked": len(rs0) - len(rs)}
                if rs:
                    p = [r["p50_ms"] for r in rs]
                    rec.update(p50=round(stt.median(p), 4), p50_min=round(min(p), 4), p50_max=round(max(p), 4),
                               p95=round(stt.median([r["p95_ms"] for r in rs]), 4),
                               engine=rs[0]["engine"])
                    lat[(sess, v["seed"], v["model"], k)] = stt.median(p)
                rows.append(rec)
    if rows:
        write(f"{OUT}/latency_engines.csv", rows)
    return lat


def energy():
    import energy_parse as EP
    runs = []
    for e in ("EB", "EC"):
        raw = f"{OUT}/raw_orin1_{e}/out"
        p = f"{raw}/runs.jsonl"
        if os.path.exists(p):
            runs += [r for r in (EP.per_run(raw, json.loads(l)) for l in open(p) if l.strip()) if r]
    if not runs:
        return {}
    med_idle = stt.median(r["P_idle_mW"] for r in runs)
    for r in runs:  # the T1 per-set exclusion flags (energy_parse.main)
        extra = []
        if abs(r["P_idle_mW"] - med_idle) > 300:
            extra.append(f"idle_dev_{r['P_idle_mW'] - med_idle:+.0f}mW")
        if r["n_samples_active"] < 0.9 * r["duration_s"] / 0.1:
            extra.append(f"few_samples_{r['n_samples_active']}")
        r["flags"] = ";".join(x for x in [r["flags"]] + extra if x)
    write(f"{OUT}/energy_runs.csv", runs)
    out, rows = {}, []
    for lab in dict.fromkeys(r["label"] for r in runs):
        good = [r for r in runs if r["label"] == lab and not r["flags"]][-3:]
        s, rest = int(lab[1]), lab.split("_", 1)[1]
        k = int(lab.split("_", 1)[0].split("b")[1])
        rec = {"seed": s, "model": rest, "build": k, "n_good": len(good),
               "n_flagged": sum(1 for r in runs if r["label"] == lab and r["flags"])}
        if good:
            e = [r["E_net_mJ"] for r in good]
            rec.update(E_net_mJ=round(stt.median(e), 5), E_net_min=round(min(e), 5), E_net_max=round(max(e), 5),
                       mean_ms=round(stt.median(r["mean_ms"] for r in good), 4),
                       P_idle_mW=round(stt.median(r["P_idle_mW"] for r in good), 1),
                       dP_mW=round(stt.median(r["dP_mW"] for r in good), 1))
            out[(s, rest)] = (stt.median(e), min(e), max(e))
        rows.append(rec)
    write(f"{OUT}/energy_engines.csv", rows)
    return out


def main():
    acc = accuracy()
    lat = latency()
    en = energy()
    prow, srow = [], []
    for name, d, b, kind, cc, f2 in PAIRS:
        per = {}
        for s in SEEDS:
            rec = {"pair": name, "kind": kind, "in_Cc_matched_set": cc, "in_F2": f2, "seed": s}
            for sess in ("B", "A"):
                kd = sorted(k for (ss, se, m, k) in lat if ss == sess and se == s and m == d)
                kb = sorted(k for (ss, se, m, k) in lat if ss == sess and se == s and m == b)
                if not kd or not kb:
                    continue
                vd, vb = [lat[(sess, s, d, k)] for k in kd], [lat[(sess, s, b, k)] for k in kb]
                rec[f"lat_ratio_{sess}"] = round(stt.median(vd) / stt.median(vb), 4)
                rec[f"lat_ratio_{sess}_lo"] = round(min(vd) / max(vb), 4)
                rec[f"lat_ratio_{sess}_hi"] = round(max(vd) / min(vb), 4)
                rec[f"builds_{sess}"] = f"{len(kd)}x{len(kb)}"
                rec[f"lat_dend_ms_{sess}"] = round(stt.median(vd), 4)
                rec[f"lat_base_ms_{sess}"] = round(stt.median(vb), 4)
            if (s, d) in en and (s, b) in en:
                ed, eb = en[(s, d)], en[(s, b)]
                rec.update(energy_ratio=round(ed[0] / eb[0], 4), energy_ratio_lo=round(ed[1] / eb[2], 4),
                           energy_ratio_hi=round(ed[2] / eb[1], 4))
            md, mb = META[d], META[b]
            if (md, s) in acc and (mb, s) in acc:
                rec.update(acc_dend=acc[(md, s)], acc_base=acc[(mb, s)], acc_gap_pp=round(100 * (acc[(md, s)] - acc[(mb, s)]), 2))
            per[s] = rec
            prow.append(rec)
        # summary over seeds
        def col(c):
            return [per[s][c] for s in SEEDS if s in per and c in per[s]]
        r = {"pair": name, "kind": kind, "in_Cc_matched_set": cc, "in_F2": f2}
        for sess in ("B", "A"):
            v = col(f"lat_ratio_{sess}")
            if v:
                r[f"lat_ratio_{sess}_per_seed"] = " ".join(f"s{s}:{per[s][f'lat_ratio_{sess}']:.3f}" for s in SEEDS if f"lat_ratio_{sess}" in per.get(s, {}))
                r[f"lat_ratio_{sess}_min"], r[f"lat_ratio_{sess}_max"] = min(v), max(v)
                r[f"lat_ratio_{sess}_build_lo"] = min(col(f"lat_ratio_{sess}_lo"))
                r[f"lat_ratio_{sess}_build_hi"] = max(col(f"lat_ratio_{sess}_hi"))
        v = col("energy_ratio")
        if v:
            r["energy_ratio_per_seed"] = " ".join(f"s{s}:{per[s]['energy_ratio']:.3f}" for s in SEEDS if "energy_ratio" in per.get(s, {}))
            r["energy_ratio_min"], r["energy_ratio_max"] = min(v), max(v)
            r["energy_ratio_run_lo"], r["energy_ratio_run_hi"] = min(col("energy_ratio_lo")), max(col("energy_ratio_hi"))
        ad, ab = col("acc_dend"), col("acc_base")
        if ad and ab:
            r.update(acc_dend_mean=round(stt.mean(ad), 5), acc_dend_sd=round(sd(ad), 5), acc_base_mean=round(stt.mean(ab), 5),
                     acc_base_sd=round(sd(ab), 5), acc_gap_pp_mean=round(100 * (stt.mean(ad) - stt.mean(ab)), 2),
                     acc_gap_pp_per_seed=" ".join(f"{x:+.2f}" for x in col("acc_gap_pp")),
                     acc_gap_pp_worst_pairing=round(100 * (min(ad) - max(ab)), 2), n_seeds_acc=f"{len(ad)}x{len(ab)}",
                     dend_more_accurate_every_seed=all(x > 0 for x in col("acc_gap_pp")),
                     dend_more_accurate_every_pairing=min(ad) > max(ab))
        lb = col("lat_ratio_B")
        if lb:
            r["slower_every_seed_B"] = all(x > 1 for x in lb)
            r["slower_every_seed_and_build_B"] = r["lat_ratio_B_build_lo"] > 1
        if v:
            r["more_energy_every_seed"] = all(x > 1 for x in v)
        srow.append(r)
    write(f"{OUT}/pairs_seed.csv", prow)
    write(f"{OUT}/pairs_summary.csv", srow)
    # C-c headline: parameter-/stored-matched naive pairs (8), per seed and over seeds x builds / runs
    cc = [r for r in prow if r["in_Cc_matched_set"]]
    rows = []
    for s in list(SEEDS) + ["all"]:
        sel = [r for r in cc if s == "all" or r["seed"] == s]
        L = [r["lat_ratio_B"] for r in sel if "lat_ratio_B" in r]
        Llo = [r["lat_ratio_B_lo"] for r in sel if "lat_ratio_B_lo" in r]
        Lhi = [r["lat_ratio_B_hi"] for r in sel if "lat_ratio_B_hi" in r]
        E = [r["energy_ratio"] for r in sel if "energy_ratio" in r]
        Elo = [r["energy_ratio_lo"] for r in sel if "energy_ratio_lo" in r]
        Ehi = [r["energy_ratio_hi"] for r in sel if "energy_ratio_hi" in r]
        rec = {"seed": s, "n_pair_seed": len(sel)}
        if L:
            rec.update(lat_ratio_min=min(L), lat_ratio_max=max(L), lat_build_extreme_lo=min(Llo), lat_build_extreme_hi=max(Lhi),
                       every_pair_slower_median=all(x > 1 for x in L), every_pair_slower_every_build=min(Llo) > 1,
                       argmin=min(sel, key=lambda r: r.get("lat_ratio_B", 9e9))["pair"], argmax=max(sel, key=lambda r: r.get("lat_ratio_B", 0))["pair"])
        if E:
            rec.update(energy_ratio_min=min(E), energy_ratio_max=max(E), energy_run_extreme_lo=min(Elo), energy_run_extreme_hi=max(Ehi),
                       pairs_energy_below_1=" ; ".join(f"{r['pair']} s{r['seed']} {r['energy_ratio']:.3f}" for r in sel if r.get("energy_ratio", 9) < 1))
        rows.append(rec)
    write(f"{OUT}/Cc_multiseed_summary.csv", rows)


if __name__ == "__main__":
    main()
