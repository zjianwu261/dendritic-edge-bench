"""Paper B cross-device timing on the car's Orin #2 (JetPack 6.2 / TensorRT 10.7), 2026-10-26.

Same ONNX graphs as the Orin #1 sessions (sha256 checked against the Orin #1 records), K=3 fresh trtexec builds per
graph x {default, nocublas} (labels b{k}_<graph>, k-outer manifest order), each engine timed NRUNS=3 times
(trt10_sync_bench.py, fresh process per run, rounds interleaved).  Car rules: inference only, everything under
/data/dn_orin2, no sudo / installs, jetson_clocks / nvpmodel never run (DVFS as found), no ROS / drivers / cmd_vel.
  python orin2.py manifest | push | start | status | pull | table | clean_engines
Outputs: results/robustness_B/orin2/{raw/, orin2_builds.csv, orin2_vs_orin1.csv, env_orin2.txt}
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

CAR = os.environ.get("ORIN2_HOST", "orin2")  # ssh host of Orin #2
RD = "/data/dn_orin2"
W = f"{RD}/work"
K, NRUNS = 3, 3
BUILDS = ("default", "nocublas")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP = f"{ROOT}/results"
OUTD = f"{EXP}/robustness_B/orin2"
RAW = f"{OUTD}/raw"
ONNX = f"{EXP}/T6/lut/onnx"

# (claim group, local onnx path, expected Orin #1 sha256 prefix or None when checked elsewhere)
GRAPHS = [
    ("C-a", f"{ONNX}/g1f/g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm.onnx"),
    ("C-a", f"{ONNX}/g1f/g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm.onnx"),
    ("C-a", f"{ONNX}/g1f/g1f_mlp_h128_lr0.001_wd0_fair_t1__conv.onnx"),
    ("C-a", f"{ONNX}/g1f/g1f_mlp_h256_lr0.0003_wd0_fair_t1__gemm.onnx"),
    ("C-a", f"{ONNX}/g1f/g1f_mlp_h256_lr0.0003_wd0_fair_t1__conv.onnx"),
    ("C-d", f"{ONNX}/g3/adc_h1024_ffrand__gemm.onnx"),
    ("C-d", f"{ONNX}/g3/adc_h2048_ffrand__gemm.onnx"),
    ("C-d", f"{ONNX}/g3/mlp_h1024__gemm.onnx"),
    ("C-d", f"{ONNX}/g3/mlp_h2048__gemm.onnx"),
    ("C-b", f"{ONNX}/g2x/lru_L2_H64_N128_lr0.001_vs_t1__K50.onnx"),
    ("C-b", f"{ONNX}/g2/dhssm_L2_n128_b4_cont__K25.onnx"),
    ("C-b", f"{ONNX}/g2/dhssm_L2_n128_b4_cont__K50.onnx"),
    ("C-b", f"{ONNX}/g2x/gru_h768_lr0.003_vs_t1__seq.onnx"),
    ("C-c", f"{EXP}/T1/models/dann_lrf_d32_s128.onnx"),
    ("C-c", f"{EXP}/T1/models/mlp_h85_match_d32_s128.onnx"),
    ("C-c", f"{EXP}/T1/models/dhsrnn_n64_b8_val_seqloop.onnx"),
    ("C-c", f"{EXP}/T1/models/gru_pm_val_seq.onnx"),
    ("C-c", f"{EXP}/T1/models/lif_srnn_n64_val_seqloop.onnx"),
]


def g(p):
    return os.path.basename(p)[:-5]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ssh(cmd, t=120, check=False):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", CAR, cmd],
                          capture_output=True, text=True, timeout=t, check=check)


def orin1_shas():
    """sha256 recorded by the Orin #1 sessions for every graph used here."""
    out = {}
    for f in (f"{EXP}/T6/raw/orin1/results.jsonl", f"{EXP}/T6/raw/orin1_bv1/results.jsonl",
              f"{EXP}/robustness_B/raw_orin1_bvg3/results.jsonl"):
        for line in open(f):
            r = json.loads(line)
            lab = r["label"].split("_", 1)[1] if r["label"].startswith("bv") else r["label"]
            out.setdefault(lab, set()).add(r.get("onnx_sha256", ""))
    for r in csv.DictReader(open(f"{EXP}/T1/results_orin_builds.csv")):
        out.setdefault(r["model"], set()).add(r["onnx_sha256_12"])
    return out


def manifest():
    o1 = orin1_shas()
    info, lines = {}, []
    for c, p in GRAPHS:
        s = sha(p)
        rec = o1.get(g(p), set())
        ok = any(x and s.startswith(x) for x in rec)
        if not ok:
            raise SystemExit(f"{g(p)}: local sha {s[:12]} not among Orin #1 records {rec}")
        info[g(p)] = {"claim": c, "local": os.path.relpath(p, ROOT), "onnx_sha256": s}
    for k in range(1, K + 1):
        for c, p in GRAPHS:
            lines.append(f"{RD}/onnx/{g(p)}.onnx b{k}_{g(p)} {','.join(BUILDS)}")
    os.makedirs(RAW, exist_ok=True)
    open(f"{RAW}/manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(info, open(f"{RAW}/graphs.json", "w"), indent=1)
    print(len(GRAPHS), "graphs (sha256 == Orin #1),", len(lines) * 2, "engines,", len(lines) * 2 * NRUNS, "runs")


def push():
    ssh(f"mkdir -p {RD}/bin {RD}/onnx", check=True)
    subprocess.run(["rsync", "-a", f"{HERE}/trt10_sync_bench.py", f"{HERE}/orin2_run.sh", f"{CAR}:{RD}/bin/"], check=True)
    subprocess.run(["rsync", "-a"] + [p for _, p in GRAPHS] + [f"{CAR}:{RD}/onnx/"], check=True)
    subprocess.run(["rsync", "-a", f"{RAW}/manifest.txt", f"{CAR}:{RD}/"], check=True)
    info = json.load(open(f"{RAW}/graphs.json"))
    r = ssh(f"cd {RD}/onnx && sha256sum *.onnx")
    car = {l.split()[1][:-5]: l.split()[0] for l in r.stdout.splitlines()}
    bad = [k for k, v in info.items() if car.get(k) != v["onnx_sha256"]]
    print("pushed; sha mismatch on car:", bad)


def start():
    if ssh(f"test -e {W} && echo yes").stdout.strip() == "yes":
        raise SystemExit(f"{W} exists; refusing to restart")
    cmd = (f"mkdir -p {W} && cd {RD} && setsid nohup bash bin/orin2_run.sh batch {RD}/manifest.txt {W} {NRUNS} "
           f"> {W}/batch.out 2>&1 < /dev/null & echo started")
    print(ssh(cmd, check=True).stdout)


def status():
    print(ssh(f"tail -3 {W}/progress.txt; cat {W}/failures.txt {W}/skipped.txt 2>/dev/null; "
              f"wc -l < {W}/results.jsonl 2>/dev/null; cat /proc/loadavg; "
              f"pgrep -af 'orin2_run|trtexec|trt10_sync' | cut -c1-120").stdout)


def pull():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{CAR}:{W}/", f"{RAW}/work/"], check=True)


def clean_engines():
    print(ssh(f"du -sh {W}/engines; rm -f {W}/engines/*.engine; rmdir {W}/engines; du -sh {RD}").stdout)


# ---------------------------------------------------------------- tables
def car_rows():
    info = json.load(open(f"{RAW}/graphs.json"))
    rows = [json.loads(l) for l in open(f"{RAW}/work/results.jsonl")]
    out = []
    for gr, v in info.items():
        for k in range(1, K + 1):
            for b in BUILDS:
                rs = [r for r in rows if r["label"] == f"b{k}_{gr}" and r["build"] == b and r["onnx_sha256"] == v["onnx_sha256"]]
                rec = {"claim": v["claim"], "graph": gr, "k": k, "build": b, "nruns": len(rs)}
                bl = f"{RAW}/work/logs/b{k}_{gr}_{b}_build.log"
                if os.path.exists(bl):
                    for line in open(bl):
                        if line.startswith("build_seconds"):
                            rec["build_seconds"] = int(line.split()[1])
                if rs:
                    p = [r["p50_ms"] for r in rs]
                    rec.update(p50=round(float(np.median(p)), 4), p50_min=min(p), p50_max=max(p),
                               p95=round(float(np.median([r["p95_ms"] for r in rs])), 4),
                               infer_only_p50=round(float(np.median([r["infer_only_p50_ms"] for r in rs])), 4),
                               gpu_mhz_timed_min_med_max="/".join(
                                   f"{x / 1e6:.0f}" for x in (min(r["gpu_freq_hz_timed_min_med_max"][0] for r in rs),
                                                              np.median([r["gpu_freq_hz_timed_min_med_max"][1] for r in rs]),
                                                              max(r["gpu_freq_hz_timed_min_med_max"][2] for r in rs))),
                               frac_at_max=round(float(np.mean([r["gpu_freq_timed_frac_at_max"] for r in rs])), 3),
                               engine_file_MB=rs[0]["engine_file_MB"])
                out.append(rec)
    return out


def cell(rows, gr, b, key="p50"):
    return [r[key] for r in rows if r["graph"] == gr and r["build"] == b and key in r]


def best_cell(rows, gr):
    """fastest build flag (median over builds)."""
    c = {b: float(np.median(cell(rows, gr, b))) for b in BUILDS if cell(rows, gr, b)}
    b = min(c, key=c.get)
    return b, c[b]


def ratio(rows, num, den):
    """num/den on the car: (num graph, num flag) / (den graph, den flag).  flag None = fastest flag.
    value = median-of-builds ratio; lo/hi = best/worst pairing for num (fastest num build / slowest den build ...)."""
    (gn, bn), (gd, bd) = num, den
    bn = bn or best_cell(rows, gn)[0]
    bd = bd or best_cell(rows, gd)[0]
    a, d = cell(rows, gn, bn), cell(rows, gd, bd)
    v = np.median(a) / np.median(d)
    return round(float(v), 3), round(min(a) / max(d), 3), round(max(a) / min(d), 3), bn, bd, a, d


def orin1_g1(gr, b):
    return [float(r["p50"]) for r in csv.DictReader(open(f"{EXP}/T6/lut/buildvar_builds_p1.csv"))
            if r["graph"] == gr and r["build"] == b and r["p50"]]


def orin1_g3(gr, b):
    return [float(r["p50"]) for r in csv.DictReader(open(f"{EXP}/robustness_B/g3_buildvar_builds.csv"))
            if r["graph"] == gr and r["build"] == b and r["origin"] == "fresh_build" and r["p50"]]


def orin1_t1(model, b):
    return [float(r["latency_p50"]) for r in csv.DictReader(open(f"{EXP}/T1/results_orin_builds.csv"))
            if r["model"] == model and r["trt_build"] == b and "Orin" in r["device"]]


def orin1_g2(lab, b):
    """single build, median of the locked runs in T6/raw/orin1/results.jsonl (as in the audit)."""
    p = [json.loads(l) for l in open(f"{EXP}/T6/raw/orin1/results.jsonl")]
    p = [r["p50_ms"] for r in p if r["label"] == lab and r["build"] == b and
         r.get("gpu_freq_hz_timed_min_med_max", [0])[0] == 624750000]
    return [float(np.median(p))] if p else []


def o1ratio(fn, num, den):
    a, d = fn(*num), fn(*den)
    return round(float(np.median(a) / np.median(d)), 3), round(min(a) / max(d), 3), round(max(a) / min(d), 3)


A_M32 = "g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm"
A_H128, A_H256 = "g1f_mlp_h128_lr0.001_wd0_fair_t1", "g1f_mlp_h256_lr0.0003_wd0_fair_t1"
LRU, DH25, DH50, GRU = ("lru_L2_H64_N128_lr0.001_vs_t1__K50", "dhssm_L2_n128_b4_cont__K25",
                        "dhssm_L2_n128_b4_cont__K50", "gru_h768_lr0.003_vs_t1__seq")

# claim id, text, car (num, den), Orin #1 function and (num, den), direction ("gt1": num slower, "tie": |r-1|<=0.10)
CLAIMS = [
    ("C-a1", "dANN-G m32 / MLP h256 (cited cells bmm|nocublas / conv|nocublas)",
     ((A_M32, "nocublas"), (A_H256 + "__conv", "nocublas")), orin1_g1, ((A_M32, "nocublas"), (A_H256 + "__conv", "nocublas")), "gt1"),
    ("C-a1f", "dANN-G m32 / MLP h256, fastest impl x flag each side",
     ((A_M32, None), ("best:" + A_H256, None)), None, None, "gt1"),
    ("C-a2", "dANN-G m32 / MLP h128 (cited cells bmm|nocublas / gemm|nocublas)",
     ((A_M32, "nocublas"), (A_H128 + "__gemm", "nocublas")), orin1_g1, ((A_M32, "nocublas"), (A_H128 + "__gemm", "nocublas")), "gt1"),
    ("C-a2f", "dANN-G m32 / MLP h128, fastest impl x flag each side",
     ((A_M32, None), ("best:" + A_H128, None)), None, None, "gt1"),
    ("C-d1", "AD-cache h1024 / MLP h1024 (gemm|nocublas both)",
     (("adc_h1024_ffrand__gemm", "nocublas"), ("mlp_h1024__gemm", "nocublas")), orin1_g3,
     (("adc_h1024_ffrand__gemm", "nocublas"), ("mlp_h1024__gemm", "nocublas")), "gt1"),
    ("C-d2", "AD-cache h2048 / MLP h2048 (gemm|default / gemm|nocublas)",
     (("adc_h2048_ffrand__gemm", "default"), ("mlp_h2048__gemm", "nocublas")), orin1_g3,
     (("adc_h2048_ffrand__gemm", "default"), ("mlp_h2048__gemm", "nocublas")), "gt1"),
    ("C-b1", "GRU h768 / LRU H64 N128 K50 (best flag each)",
     ((GRU, None), (LRU, None)), orin1_g2, ((GRU, "default"), (LRU, "nocublas")), "gt20"),
    ("C-b2", "GRU h768 / DH-SSM n128 cont K25 (best flag each)",
     ((GRU, None), (DH25, None)), orin1_g2, ((GRU, "default"), (DH25, "nocublas")), "gt20"),
    ("C-b3", "LRU K50 / DH-SSM n128 cont K25 (best flag each)",
     ((LRU, None), (DH25, None)), orin1_g2, ((LRU, "nocublas"), (DH25, "nocublas")), "tie"),
    ("C-b4", "LRU K50 / DH-SSM n128 cont K50 (equal K, best flag each)",
     ((LRU, None), (DH50, None)), orin1_g2, ((LRU, "nocublas"), (DH50, "nocublas")), "tie"),
    ("C-c1", "DH-SRNN (loop) / GRU-pm (fused op), nocublas, DVFS",
     (("dhsrnn_n64_b8_val_seqloop", "nocublas"), ("gru_pm_val_seq", "nocublas")), orin1_t1,
     (("dhsrnn_n64_b8_val_seqloop", "nocublas"), ("gru_pm_val_seq", "nocublas")), "gt1"),
    ("C-c2", "dANN d32 / MLP h85 (param-matched), nocublas, DVFS",
     (("dann_lrf_d32_s128", "nocublas"), ("mlp_h85_match_d32_s128", "nocublas")), orin1_t1,
     (("dann_lrf_d32_s128", "nocublas"), ("mlp_h85_match_d32_s128", "nocublas")), "gt1"),
    ("C-c3", "DH-SRNN / LIF-SRNN (both loop), nocublas, DVFS",
     (("dhsrnn_n64_b8_val_seqloop", "nocublas"), ("lif_srnn_n64_val_seqloop", "nocublas")), orin1_t1,
     (("dhsrnn_n64_b8_val_seqloop", "nocublas"), ("lif_srnn_n64_val_seqloop", "nocublas")), "gt1"),
]
ORIN1_NOTE = {"orin1_g1": "T6/lut/buildvar_builds_p1.csv, 5 builds, locked 624.75 MHz",
              "orin1_g3": "robustness_B/g3_buildvar_builds.csv, 5 fresh builds, locked 624.75 MHz",
              "orin1_g2": "T6/raw/orin1/results.jsonl, 1 build, locked 624.75 MHz",
              "orin1_t1": "T1/results_orin_builds.csv, 1 build, DVFS (15W)"}


def resolve(rows, gr):
    """'best:<mlp prefix>' -> fastest of __gemm/__conv by median-of-builds of its fastest flag."""
    if not gr.startswith("best:"):
        return gr
    c = {f"{gr[5:]}__{i}": best_cell(rows, f"{gr[5:]}__{i}")[1] for i in ("gemm", "conv")}
    return min(c, key=c.get)


def holds(d, v, lo, hi):
    if d == "gt1":
        return v > 1 and lo > 1, "slower in median and in the best pairing" if lo > 1 else ("median > 1 only" if v > 1 else "no")
    if d == "gt20":
        return lo >= 20, f"worst pairing >= 20x" if lo >= 20 else "no (< 20x)"
    if d == "tie":
        return abs(v - 1) <= 0.10, "within 10 %" if abs(v - 1) <= 0.10 else "no"


def table():
    rows = car_rows()
    keys = []
    for r in rows:
        keys += [x for x in r if x not in keys]
    with open(f"{OUTD}/orin2_builds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    out = []
    for cid, text, (num, den), f1, ref, d in CLAIMS:
        num = (resolve(rows, num[0]), num[1]); den = (resolve(rows, den[0]), den[1])
        v, lo, hi, bn, bd, a, dd = ratio(rows, num, den)
        h, why = holds(d, v, lo, hi)
        rec = {"claim": cid, "item": text, "orin2_num": f"{num[0]}|{bn}", "orin2_den": f"{den[0]}|{bd}",
               "orin2_ratio_median": v, "orin2_ratio_best_pair": lo, "orin2_ratio_worst_pair": hi,
               "orin2_num_builds_ms": " ".join(f"{x:.4f}" for x in a), "orin2_den_builds_ms": " ".join(f"{x:.4f}" for x in dd),
               "direction": d, "orin2_holds": h, "orin2_holds_rule": why}
        if f1:
            v1, lo1, hi1 = o1ratio(f1, *ref)
            h1, _ = holds(d, v1, lo1, hi1)
            rec.update(orin1_ratio_median=v1, orin1_ratio_best_pair=lo1, orin1_ratio_worst_pair=hi1, orin1_holds=h1,
                       orin1_source=ORIN1_NOTE[f1.__name__], orin1_cells=f"{ref[0][0]}|{ref[0][1]} / {ref[1][0]}|{ref[1][1]}")
        out.append(rec)
    # C-e: build-to-build spread per graph x flag (max/min of the per-build medians)
    sp = []
    for gr in sorted({r["graph"] for r in rows}):
        for b in BUILDS:
            c = cell(rows, gr, b)
            if len(c) >= 2:
                sp.append((gr, b, max(c) / min(c)))
    s = np.array([x[2] for x in sp])
    mx = max(sp, key=lambda x: x[2])
    out.append({"claim": "C-e", "item": f"rebuild spread max/min over {len(sp)} graph x flag cells, {K} builds each",
                "orin2_ratio_median": round(float(np.median(s)), 3), "orin2_ratio_worst_pair": round(float(s.max()), 3),
                "orin2_num": f"max at {mx[0]}|{mx[1]}", "orin2_holds": bool(s.max() <= 1.22),
                "orin2_holds_rule": "max <= Orin #1 max 1.22", "orin1_ratio_median": "1.04 (G1) / 1.02 (G3)",
                "orin1_ratio_worst_pair": 1.22, "orin1_source": "robustness_B/Ce_build_spread.csv (5 builds)"})
    keys = []
    for r in out:
        keys += [x for x in r if x not in keys]
    with open(f"{OUTD}/orin2_vs_orin1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    for r in out:
        print(r["claim"], r.get("orin1_ratio_median"), "|", r.get("orin2_ratio_median"),
              f"[{r.get('orin2_ratio_best_pair')}, {r.get('orin2_ratio_worst_pair')}]", r.get("orin2_holds"), r.get("orin2_num", ""))
    print("cells (graph|flag) spread > 1.10:", [(a, b, round(c, 3)) for a, b, c in sp if c > 1.10])


if __name__ == "__main__":
    {"manifest": manifest, "push": push, "start": start, "status": status, "pull": pull, "table": table,
     "clean_engines": clean_engines}[sys.argv[1]]()
