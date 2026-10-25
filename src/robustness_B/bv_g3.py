"""Robustness audit of Paper B (2026-10-25), claim C-d: TRT build-to-build variance of the G3 AD-cache vs
dense-MLP latency ratio (adc_h{1024,2048}_ffrand vs mlp_h{1024,2048}, Gemm and Conv forms, random-weight G3 LUT graphs).

Same protocol as src/t6/buildvar_orin.py (phase 1/2 for G1): each graph x {default, nocublas} x K=5 fresh trtexec builds
(labels bv{k}_<graph>, k-outer manifest order), each engine timed NRUNS=5 times (t6_sync_bench, fresh process per run,
rounds interleaved across engines).  In addition, label bv0_<graph> = the exact engine of the 10-02 G3 LUT session
(copied from ~/$RD/work/engines, not rebuilt) re-timed in the same session, to separate session drift from build variance.
Rules: Orin #1, flock /tmp/dn_orin_bench.lock (inside t6_orin.sh), REQUIRE_LOCK_HZ=624750000, stop if not locked at
the start; jetson_clocks / nvpmodel never run.  No energy measurement.  Workdir ~/$RD/work_bvg3, pulled to
results/robustness_B/raw_orin1_bvg3.  Outputs (new files only): results/robustness_B/g3_buildvar_builds.csv.
  python bv_g3.py run | wait | pull | table
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "t6"))
from common6 import LUT  # noqa: E402
import lut as L  # noqa: E402
import e7_g1_orin as E7  # noqa: E402
import buildvar_orin as BV  # noqa: E402

ORIN, RD = L.ORIN, L.RD
K, NRUNS = 5, 5
LOCK_HZ = E7.LOCK_HZ
BUILDS = ("default", "nocublas")
GRAPHS = [f"{m}_h{h}{s}__{i}" for m, s in (("adc", "_ffrand"), ("mlp", "")) for h in (1024, 2048) for i in ("gemm", "conv")]
W = "work_bvg3"
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = f"{ROOT}/results/robustness_B/raw_orin1_bvg3"
OUT = f"{ROOT}/results/robustness_B/g3_buildvar_builds.csv"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def manifest():
    info = {g: {"dir": "g3", "onnx_sha256": sha(f"{LUT}/onnx/g3/{g}.onnx")} for g in GRAPHS}
    lines = [f"~/{RD}/onnx/g3/{g}.onnx bv{k}_{g} {','.join(BUILDS)}" for k in range(0, K + 1) for g in GRAPHS]
    os.makedirs(RAW, exist_ok=True)
    open(f"{RAW}/bv_manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(info, open(f"{RAW}/bv_graphs.json", "w"), indent=1)
    print(len(GRAPHS), "graphs,", len(lines), "lines,", len(lines) * 2, "engines,", len(lines) * 2 * NRUNS, "runs")
    return info


def run():
    T = lambda: time.strftime("%F %T")
    info = manifest()
    st = BV.lock_state()
    print(T(), "GPU devfreq min/cur/max", st, flush=True)
    if st != (LOCK_HZ,) * 3:
        raise SystemExit(f"GPU NOT LOCKED {st}: stop (jetson_clocks / nvpmodel are not run by this script)")
    if BV.ssh(f"test -e ~/{RD}/{W} && echo yes").stdout.strip() == "yes":
        raise SystemExit(f"~/{RD}/{W} already exists on the Orin; refusing to restart")
    for g, v in info.items():
        r = BV.ssh(f"sha256sum ~/{RD}/onnx/g3/{g}.onnx")
        if not r.stdout.startswith(v["onnx_sha256"]):
            raise SystemExit(f"Orin graph {g} missing or sha differs: {r.stdout} {r.stderr}")
        for b in BUILDS:
            s12 = v["onnx_sha256"][:12]
            if BV.ssh(f"test -s ~/{RD}/work/engines/{g}_{b}_{s12}.engine && echo yes").stdout.strip() != "yes":
                raise SystemExit(f"original LUT engine {g}_{b}_{s12} not found")
    cp = " && ".join(f"cp -n work/engines/{g}_{b}_{v['onnx_sha256'][:12]}.engine {W}/engines/bv0_{g}_{b}_{v['onnx_sha256'][:12]}.engine"
                     for g, v in info.items() for b in BUILDS)
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/{W}/engines && cd ~/{RD} && {cp}"], check=True, timeout=120)
    subprocess.run(["rsync", "-a", f"{RAW}/bv_manifest.txt", f"{ORIN}:{RD}/{W}/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" {W}/bv_manifest.txt > {W}/bv_manifest_abs.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/t6_orin.sh batch {W}/bv_manifest_abs.txt $HOME/{RD}/{W} {NRUNS} "
           f"> {W}/bv.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)
    print(T(), "started (REQUIRE_LOCK_HZ set)", flush=True)
    wait()


def wait(poll_s=300, run_max_h=8):
    T = lambda: time.strftime("%F %T")
    t_end, lost = time.time() + float(run_max_h) * 3600, 0
    while True:
        try:
            done = BV.ssh(f"grep -c 'batch done' ~/{RD}/{W}/progress.txt || true").stdout.strip()
            st = BV.lock_state()
            if st != (LOCK_HZ,) * 3:
                lost += 1
                print(T(), "WARNING LOCK LOST", st, flush=True)
            if done not in ("", "0"):
                break
            print(T(), "running:", BV.ssh(f"tail -1 ~/{RD}/{W}/progress.txt").stdout.strip(), "| lock", st, flush=True)
        except subprocess.TimeoutExpired:
            print(T(), "ssh timeout", flush=True)
        if time.time() > t_end:
            raise SystemExit("batch not done in time")
        time.sleep(int(poll_s))
    pull()
    print(T(), "pulled; lock-lost polls", lost, flush=True)
    table()
    print(T(), "RUN DONE", flush=True)


def pull():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/{W}/", f"{RAW}/"], check=True)


def table():
    info = json.load(open(f"{RAW}/bv_graphs.json"))
    start = max(l.split()[0][:19] for l in open(f"{RAW}/progress.txt") if "batch start" in l)
    rows = [json.loads(l) for l in open(f"{RAW}/results.jsonl")]
    rows = [r for r in rows if r.get("time", "") >= start]
    out = []
    for g, v in info.items():
        for k in range(0, K + 1):
            lab = f"bv{k}_{g}"
            for b in BUILDS:
                rs0 = [r for r in rows if r["label"] == lab and r["build"] == b and r["onnx_sha256"] == v["onnx_sha256"]]
                rs = [r for r in rs0 if E7.run_locked(r, RAW)]
                rec = {"graph": g, "k": k, "origin": "orig_10-02_engine" if k == 0 else "fresh_build", "build": b,
                       "label": lab, "nruns": len(rs), "n_excluded_unlocked": len(rs0) - len(rs)}
                lj = f"{RAW}/logs/{lab}_{b}.layers.json"
                if os.path.exists(lj):
                    rec.update(BV.tactics(lj))
                if rs:
                    p = [r["p50_ms"] for r in rs]
                    rec.update(p50=round(float(np.median(p)), 4), p50_min=round(min(p), 4), p50_max=round(max(p), 4))
                out.append(rec)
    keys = []
    for r in out:
        keys += [x for x in r if x not in keys]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(len(out), "rows ->", OUT, "| locked runs", sum(r["nruns"] for r in out),
          "excluded", sum(r["n_excluded_unlocked"] for r in out))


if __name__ == "__main__":
    {"manifest": manifest, "run": run, "wait": wait, "pull": pull, "table": table}[sys.argv[1]](*sys.argv[2:])
