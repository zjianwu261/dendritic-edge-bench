#!/usr/bin/env bash
# One energy measurement of one TensorRT engine on Orin #1 (T1 energy protocol).
# Holds flock /tmp/dn_orin_bench.lock for the whole measurement (idle-pre, inference, idle-post), so
# no other build / timing job runs on the board meanwhile (the whole-board VDD_IN is measured).
#   1. archive env (nvpmodel, jetson_clocks --show, GPU devfreq, thermal, loadavg, top CPU procs, trtexec)
#   2. start power_logger.py (jtop VDD_IN, 100 ms, wall-clock stamped) + tegrastats --interval 100 (raw)
#   3. idle IDLE_PRE s  ->  trt_energy_bench.py (warm-up >= 2 s, then >= DUR s back-to-back e2e)
#      ->  idle IDLE_POST s ; stop loggers; archive env again
#   4. quick flag check (trtexec seen in window, high loadavg, low MemAvailable) -> exit 3 if flagged
# Exit codes: 0 ok, 3 flagged (caller re-measures), 4 not run (MemAvailable too low), other = error.
# Usage: energy_one.sh <label> <engine> <outdir> <tag>
set -uo pipefail
LABEL=$1; ENG=$2; OUT=$3; TAG=$4
IDLE_PRE=${IDLE_PRE:-12}; IDLE_POST=${IDLE_POST:-14}; DUR=${DUR:-30}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-1000}; MAX_LOAD1=${MAX_LOAD1:-2.5}
LOCK=/tmp/dn_orin_bench.lock
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
B="$OUT/${LABEL}_${TAG}"
avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
env_dump() {
  echo "## $(date -Iseconds) label=$LABEL tag=$TAG engine=$(basename "$ENG") $1"
  echo "## nvpmodel -q"; nvpmodel -q 2>&1 || true
  echo "## jetson_clocks --show"; jetson_clocks --show 2>&1 || true
  G=/sys/devices/gpu.0/devfreq/17000000.ga10b
  echo "## gpu devfreq governor/cur/min/max"; cat $G/governor $G/cur_freq $G/min_freq $G/max_freq 2>&1 || true
  echo "## cpu cur freq (kHz)"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 | tr '\n' ' '; echo
  echo "## thermal (type temp_mC)"; for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(cat $z/temp)"; done
  echo "## loadavg"; cat /proc/loadavg
  echo "## MemAvailable_MB $(avail_mb)"
  echo "## top CPU processes"; ps -eo pid,pcpu,rss,etime,args --sort=-pcpu | head -8 | cut -c1-200
  echo "## trtexec processes"; ps -eo pid,etime,rss,args | grep -E "[t]rtexec" | cut -c1-200 || true
}
sleep "${PRE_LOCK_SLEEP:-5}"  # let other jobs waiting on the lock get it between our runs
exec 9>"$LOCK"
flock 9
am=$(avail_mb)
if [ "$am" -lt "$MIN_AVAIL_MB" ]; then
  echo "$(date -Iseconds) $LABEL $TAG NOT RUN: MemAvailable $am MB < $MIN_AVAIL_MB" | tee -a "$OUT/skipped.txt"
  exit 4
fi
env_dump "before" > "${B}_env.txt"
python3 "$HERE/power_logger.py" "${B}_power.csv" --interval 0.1 9>&- &
PL=$!
tegrastats --interval 100 > "${B}_tegra.txt" 2>&1 9>&- &
TG=$!
sleep 1
T_IDLE_PRE=$(date +%s.%N)
sleep "$IDLE_PRE"
python3 "$HERE/trt_energy_bench.py" "$ENG" --duration "$DUR" --label "$LABEL" --run 0 > "${B}_bench.json" 2> "${B}_bench.err"
RC=$?
T_IDLE_POST=$(date +%s.%N)
sleep "$IDLE_POST"
T_END=$(date +%s.%N)
kill $PL $TG 2>/dev/null; wait $PL 2>/dev/null
env_dump "after" >> "${B}_env.txt"
python3 - "$B" "$LABEL" "$TAG" "$ENG" "$T_IDLE_PRE" "$T_IDLE_POST" "$T_END" "$RC" "$MAX_LOAD1" "$am" "$OUT/runs.jsonl" <<'EOF'
import json, sys, csv
b, label, tag, eng, tip, tpo, tend, rc, maxload, am, runs = sys.argv[1:]
meta = {"label": label, "tag": tag, "engine": eng, "base": b.split("/")[-1], "t_idle_pre_start": float(tip),
        "t_idle_post_start": float(tpo), "t_end": float(tend), "bench_rc": int(rc), "mem_avail_before_MB": int(am)}
try:
    meta["bench"] = json.loads(open(b + "_bench.json").read().strip().splitlines()[-1])
except Exception as e:
    meta["bench"] = None; meta["bench_error"] = str(e)
rows = [r for r in csv.reader(l for l in open(b + "_power.csv") if not l.startswith("#"))][1:]
flags = []
if not rows: flags.append("no_power_samples")
else:
    if max(int(r[9] or 0) for r in rows) > 0: flags.append("trtexec_in_window")
    l1 = [float(r[7]) for r in rows if r[7]]
    if l1 and l1[0] > float(maxload): flags.append(f"load1_start_{l1[0]}")
    meta["n_samples"] = len(rows); meta["load1_start_max"] = [l1[0], max(l1)] if l1 else None
if meta["bench"] is None or int(rc) != 0: flags.append("bench_failed")
# clocks not locked in the before/after env dump -> flag
try:
    L = open(b + "_env.txt").read().splitlines()
    mm = [(L[i + 3].strip(), L[i + 4].strip()) for i, l in enumerate(L) if l.startswith("## gpu devfreq")]
    meta["gpu_min_max_hz"] = mm
    if any(a != c for a, c in mm): flags.append("gpu_min_freq_not_max")
except Exception as e:
    meta["env_parse_error"] = str(e)
meta["flags"] = flags
open(runs, "a").write(json.dumps(meta) + "\n")
print(json.dumps({k: meta.get(k) for k in ("label", "tag", "flags", "n_samples")}))
sys.exit(3 if flags else 0)
EOF
