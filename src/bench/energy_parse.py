"""Parse the Orin #1 energy runs (src/bench/energy_chain.sh) into per-run and per-config tables.

Per run (power log = jtop VDD_IN, 100 ms, wall-clock stamped; see power_logger.py):
  P_active = mean VDD_IN over [t_start, t_end] of the measured inference window (>= 30 s)
  P_idle   = mean of  idle_pre  = [t_idle_pre_start, t_load_start]   (~12 s, before the engine is loaded)
                and   idle_post = [t_idle_post_start + 2 s, t_end_run] (~12 s, after the process exited)
  E_net    = (P_active - P_idle) * duration / n_inf      [mJ]
  E_gross  =  P_active           * duration / n_inf      [mJ]
Runs with flags (trtexec in window, high loadavg, failed bench) are excluded. A run is also flagged "gpu_min_freq_not_max" when, in its before- or after-env dump,
the GPU devfreq min_freq differs from max_freq (i.e. the clocks were not locked during that run);
the CPU frequencies of the env dumps are reported (cpu_all_1510MHz) but do not flag a run.
Two further exclusion flags, applied per set (T1 / T6):
  "idle_dev_<x>mW": |P_idle - median P_idle of all runs of the set| > 300 mW;
  "few_samples_<n>": active-window power samples < 90 % of duration / 0.1 s.
Grouping into DVFS / locked must use these per-run archives, not timestamps (the Orin system
clock jumped between about 17:16 and 18:24 on 2026-10-01). Per config: the
engine is chosen as the one whose name contains "_val_" if it has >= 3 good runs, else the engine with
the most good runs; the last 3 good runs of that engine give median and min-max range.
Outputs:
  results/T1/energy_orin1.csv, results/T1/energy_orin1_runs.csv
  results/T6/energy.csv,       results/T6/energy_runs.csv
  --merge : also fill energy_mJ / energy_note of the Orin rows of results/T1/results.csv
            (re-run after src/t1/make_results.py, which regenerates results.csv)
  compare : summary of the jtop vs tegrastats window (results/T1/raw/orin1_energy/compare/)
Usage: python src/bench/energy_parse.py [--merge] [--only T1|T6] | python src/bench/energy_parse.py compare
"""
import csv
import json
import os
import statistics as st
import sys

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SETS = [("T1", f"{ROOT}/results/T1/raw/orin1_energy", f"{ROOT}/results/T1/energy_orin1"),
        ("T6", f"{ROOT}/results/T6/raw/energy", f"{ROOT}/results/T6/energy")]


def load_power(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            try:
                rows.append({k: float(v) if v not in ("", None) else None for k, v in r.items()})
            except ValueError:
                pass
    return rows


def win(rows, a, b, key="vdd_in_mW"):
    return [r[key] for r in rows if a <= r["t_unix"] <= b and r[key] is not None]


def env_clock_state(d, base):
    """(flags, info) from <base>_env.txt: GPU min/max of every dump, CPU cur freqs."""
    p = os.path.join(d, base + "_env.txt")
    if not os.path.exists(p):
        return [], {"gpu_min_max_MHz": "", "cpu_all_1510MHz": ""}
    L = open(p).read().splitlines()
    gpu, cpu_ok = [], True
    for i, l in enumerate(L):
        if l.startswith("## gpu devfreq"):
            try:
                gpu.append((int(L[i + 3]), int(L[i + 4])))
            except (ValueError, IndexError):
                pass
        if l.startswith("## cpu cur freq"):
            try:
                cpu_ok &= all(int(x) == 1510400 for x in L[i + 1].split())
            except (ValueError, IndexError):
                pass
    flags = ["gpu_min_freq_not_max"] if any(a != b for a, b in gpu) else []
    return flags, {"gpu_min_max_MHz": " ".join(f"{a // 1000000}/{b // 1000000}" for a, b in gpu),
                   "cpu_all_1510MHz": cpu_ok}


def per_run(d, m):
    b = m.get("bench")
    if not b:
        return None
    rows = load_power(os.path.join(d, m["base"] + "_power.csv"))
    act = win(rows, b["t_start"], b["t_end"])
    pre = win(rows, m["t_idle_pre_start"], b["t_load_start"])
    post = win(rows, m["t_idle_post_start"] + 2.0, m["t_end"])
    if not act or not pre or not post:
        return None
    pa, ppre, ppost = st.mean(act), st.mean(pre), st.mean(post)
    pidle = (ppre + ppost) / 2
    dur, n = b["duration_s"], b["n_inf"]
    ts = [r["t_unix"] for r in rows if b["t_start"] <= r["t_unix"] <= b["t_end"]]
    dts = [y - x for x, y in zip(ts, ts[1:])]
    dup = sum(1 for x, y in zip(act, act[1:]) if x == y) / max(1, len(act) - 1)
    gf = win(rows, b["t_start"], b["t_end"], "gpu_freq_hz")
    tj = win(rows, m["t_idle_pre_start"], m["t_end"], "temp_tj_C")
    ld = win(rows, m["t_idle_pre_start"], m["t_end"], "load1")
    cflags, cinfo = env_clock_state(d, m["base"])
    return {
        "label": m["label"], "tag": m["tag"], "engine": os.path.basename(m["engine"]),
        "flags": ";".join(list(m.get("flags", [])) + cflags), **cinfo, "n_inf": n, "duration_s": dur, "mean_ms": b["mean_ms"],
        "P_active_mW": round(pa, 1), "P_idle_pre_mW": round(ppre, 1), "P_idle_post_mW": round(ppost, 1),
        "P_idle_mW": round(pidle, 1), "dP_mW": round(pa - pidle, 1),
        "E_net_mJ": round((pa - pidle) * dur / n, 5), "E_gross_mJ": round(pa * dur / n, 5),
        "n_samples_active": len(act), "n_samples_idle": len(pre) + len(post),
        "dt_median_s": round(st.median(dts), 3) if dts else None,
        "dup_frac_active": round(dup, 3),
        "gpu_MHz_active_median": round(st.median(gf) / 1e6, 2) if gf else None,
        "gpu_frac_at_max_active": round(sum(1 for g in gf if g == 624750000) / len(gf), 3) if gf else None,
        "tj_max_C": max(tj) if tj else None, "load1_max": max(ld) if ld else None,
        "mem_avail_before_MB": m.get("mem_avail_before_MB"),
    }


def rng(v):
    return f"{min(v):.4g}-{max(v):.4g}"


def main():
    merge = "--merge" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    t1cfg = {}
    for name, d, outp in SETS:
        if only and name != only:
            continue
        p = os.path.join(d, "runs.jsonl")
        if not os.path.exists(p):
            print(f"{name}: no runs yet ({p})")
            continue
        runs = [r for r in (per_run(d, json.loads(l)) for l in open(p) if l.strip()) if r]
        if not runs:
            continue
        med_idle = st.median(r["P_idle_mW"] for r in runs)
        for r in runs:
            extra = []
            if abs(r["P_idle_mW"] - med_idle) > 300:
                extra.append(f"idle_dev_{r['P_idle_mW'] - med_idle:+.0f}mW")
            if r["n_samples_active"] < 0.9 * r["duration_s"] / 0.1:
                extra.append(f"few_samples_{r['n_samples_active']}")
            r["flags"] = ";".join(x for x in [r["flags"]] + extra if x)
            r["idle_median_of_set_mW"] = round(med_idle, 1)
        with open(outp + "_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)
        cfgs = []
        for lab in dict.fromkeys(r["label"] for r in runs):
            good = [r for r in runs if r["label"] == lab and not r["flags"]]
            if not good:
                continue
            by = {}
            for r in good:
                by.setdefault(r["engine"], []).append(r)
            val = [e for e in by if "_val_" in e and len(by[e]) >= 3]
            eng = val[0] if val else max(by, key=lambda e: len(by[e]))
            rs = by[eng][-3:]
            c = {"label": lab, "engine": eng, "n_runs": len(rs),
                 "E_net_mJ": round(st.median(r["E_net_mJ"] for r in rs), 4), "E_net_range": rng([r["E_net_mJ"] for r in rs]),
                 "E_gross_mJ": round(st.median(r["E_gross_mJ"] for r in rs), 4), "E_gross_range": rng([r["E_gross_mJ"] for r in rs]),
                 "P_idle_mW": round(st.median(r["P_idle_mW"] for r in rs), 1), "P_active_mW": round(st.median(r["P_active_mW"] for r in rs), 1),
                 "dP_mW": round(st.median(r["dP_mW"] for r in rs), 1), "dP_range": rng([r["dP_mW"] for r in rs]),
                 "mean_ms": round(st.median(r["mean_ms"] for r in rs), 4), "n_inf_median": int(st.median(r["n_inf"] for r in rs)),
                 "gpu_MHz_active_median": st.median(r["gpu_MHz_active_median"] for r in rs),
                 "tj_max_C": max(r["tj_max_C"] for r in rs), "load1_max": max(r["load1_max"] for r in rs),
                 "n_flagged_runs": sum(1 for r in runs if r["label"] == lab and r["flags"]),
                 "flagged_runs": " ".join(f"{r['tag']}:{r['flags']}" for r in runs if r["label"] == lab and r["flags"]),
                 "tags": " ".join(r["tag"] for r in rs)}
            cfgs.append(c)
            if name == "T1":
                t1cfg[lab] = c
        with open(outp + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cfgs[0]))
            w.writeheader()
            w.writerows(cfgs)
        print(f"{name}: {len(runs)} runs, {len(cfgs)} configs -> {outp}.csv")
        for c in cfgs:
            print(f"  {c['label']:34s} E_net {c['E_net_mJ']:>9} mJ ({c['E_net_range']})  dP {c['dP_mW']} mW  "
                  f"idle {c['P_idle_mW']} mW  {c['mean_ms']} ms/inf  n={c['n_runs']}  {c['engine']}")
    if merge and t1cfg:
        p = f"{ROOT}/results/T1/results.csv"
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            if "Orin" not in r["device"]:
                continue
            c = t1cfg.get(r["model"])
            if c:
                r["energy_mJ"] = c["E_net_mJ"]
                r["energy_note"] = (f"net of idle (idle {c['P_idle_mW']} mW, active {c['P_active_mW']} mW), median of {c['n_runs']} "
                                    f"runs x >=30 s, range {c['E_net_range']} mJ; jtop VDD_IN 100 ms; nocublas engine {c['engine']}; "
                                    f"gross {c['E_gross_mJ']} mJ; see results/T1/energy_orin1.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"merged {sum(1 for r in rows if r['model'] in t1cfg and 'Orin' in r['device'])} rows into {p}")


def compare():
    d = f"{ROOT}/results/T1/raw/orin1_energy/compare"
    rows = load_power(f"{d}/compare_jtop.csv")
    b = json.loads(open(f"{d}/compare_bench.json").read().strip().splitlines()[-1])
    t = [r["t_unix"] for r in rows]
    act = win(rows, b["t_start"], b["t_end"])
    idle = win(rows, t[0], b["t_load_start"]) + win(rows, b["t_end"] + 2, t[-1])
    dts = [y - x for x, y in zip(t, t[1:])]
    v = [r["vdd_in_mW"] for r in rows]
    dup = sum(1 for x, y in zip(v, v[1:]) if x == y) / (len(v) - 1)
    teg = open(f"{d}/compare_tegrastats.txt").read().splitlines()
    print(json.dumps({"window_s": round(t[-1] - t[0], 1), "jtop_samples": len(rows),
                      "jtop_dt_median_s": round(st.median(dts), 3), "jtop_dt_max_s": round(max(dts), 3),
                      "jtop_consecutive_identical_frac": round(dup, 3),
                      "jtop_idle_mean_mW": round(st.mean(idle), 1), "jtop_idle_sd_mW": round(st.pstdev(idle), 1),
                      "jtop_active_mean_mW": round(st.mean(act), 1), "jtop_active_sd_mW": round(st.pstdev(act), 1),
                      "bench": {k: b[k] for k in ("duration_s", "n_inf", "mean_ms")},
                      "tegrastats_lines": len(teg), "tegrastats_lines_with_VDD": sum("VDD_" in l for l in teg)}, indent=1))


if __name__ == "__main__":
    compare() if sys.argv[1:2] == ["compare"] else main()
