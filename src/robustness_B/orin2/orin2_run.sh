#!/usr/bin/env bash
# Orin #2 (car) batch runner for the Paper B cross-device timing check (2026-10-26).
# Port of src/t6/orin/t6_orin.sh to the car: same build flags (--fp16 --memPoolSize=workspace:1024
# [--tacticSources=-CUBLAS,-CUBLAS_LT] --profilingVerbosity=detailed --exportLayerInfo), same phase structure
# (phase 1 builds every engine, phase 2 times every engine NRUNS times, rounds interleaved, fresh process per run).
# Differences: everything lives under /data/dn_orin2 (lock file too); GPU devfreq path of JetPack 6; no clock-lock
# check (the car is DVFS; jetson_clocks / nvpmodel are never run); per-run timeout; tegrastats per run (user-level,
# no extra permissions) is killed on exit.  Inference only: no ROS, no /cmd_vel, no drivers.
#   orin2_run.sh batch <manifest> <workdir> [nruns=3]
# manifest lines: <onnx path> <label> <builds, comma separated: default|nocublas>
set -uo pipefail
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ROOT=/data/dn_orin2
LOCK=$ROOT/bench.lock
MIN_AVAIL_MB=${MIN_AVAIL_MB:-2048}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-900}
BUILD_TIMEOUT_S=${BUILD_TIMEOUT_S:-3600}
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
HERE=$(cd "$(dirname "$0")" && pwd)
avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
TP=""
trap '[ -n "$TP" ] && kill $TP 2>/dev/null' EXIT INT TERM

build_flags() {
  case $1 in
    default) echo "" ;;
    nocublas) echo "--tacticSources=-CUBLAS,-CUBLAS_LT" ;;
    *) echo "BAD" ;;
  esac
}

env_dump() {
  echo "## $(date -Iseconds) $1"
  echo "## nvpmodel -q"; nvpmodel -q 2>&1 || true
  echo "## gpu devfreq governor/cur/min/max"; cat $G/governor $G/cur_freq $G/min_freq $G/max_freq 2>&1 || true
  echo "## emc (cur/min/max)"; cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || echo "n/a (debugfs not readable)"
  echo "## thermal (type temp_mC)"; for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done
  echo "## loadavg"; cat /proc/loadavg
  echo "## cpu cur freq (kHz)"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 | tr '\n' ' '; echo
  echo "## cpu governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1
  echo "## MemAvailable MB $(avail_mb)"
  echo "## other GPU/bench processes"; ps -eo pid,comm,args | grep -E "trtexec|trt10_sync|python3" | grep -v grep | cut -c1-160 || true
}

one_run() {  # $1 engine $2 label $3 build $4 run $5 sha $6 wd  (under the lock)
  local eng=$1 lab=$2 b=$3 r=$4 sha=$5 wd=$6
  local base; base=$(basename "$eng" .engine)
  env_dump "before $lab $b run $r" > "$wd/env/${base}_r$r.txt"
  tegrastats --interval 100 > "$wd/tegra/${base}_r$r.txt" 2>&1 &
  TP=$!
  timeout "$RUN_TIMEOUT_S" python3 "$HERE/trt10_sync_bench.py" "$eng" --label "$lab" --precision fp16 --build "$b" \
    --onnx_sha256 "$sha" --run "$r" --out "$wd/results.jsonl" > /dev/null 2>> "$wd/logs/${base}_run.err"
  local rc=$?
  kill $TP 2>/dev/null; wait $TP 2>/dev/null; TP=""
  env_dump "after run $r rc=$rc" >> "$wd/env/${base}_r$r.txt"
  [ $rc -eq 0 ] || echo "$(date -Iseconds) RUN FAILED rc=$rc $eng r$r" >> "$wd/failures.txt"
}

batch() {
  local man=$1 wd=$2 nruns=${3:-3}
  mkdir -p "$wd/engines" "$wd/logs" "$wd/env" "$wd/tegra"
  local list="$wd/engines_$(basename "$man").txt"
  : > "$list"
  echo "$(date -Iseconds) batch start $man" >> "$wd/progress.txt"
  env_dump "batch start" > "$wd/env/batch_start.txt"
  while read -r onnx lab builds; do
    [ -z "$onnx" ] && continue
    case $onnx in \#*) continue ;; esac
    local sha; sha=$(sha256sum "$onnx" | cut -c1-64)
    for b in ${builds//,/ }; do
      local flags; flags=$(build_flags "$b")
      local base="${lab}_${b}"
      local eng="$wd/engines/${base}_${sha:0:12}.engine"
      if [ ! -s "$eng" ]; then
        local am; am=$(avail_mb)
        if [ "$am" -lt "$MIN_AVAIL_MB" ]; then
          echo "$(date -Iseconds) SKIP build $base: MemAvailable ${am} MB < ${MIN_AVAIL_MB} MB" >> "$wd/skipped.txt"; continue
        fi
        local s; s=$(date +%s)
        echo "## onnx $onnx sha256 $sha build $b flags $flags" > "$wd/logs/${base}_build.log"
        # shellcheck disable=SC2086
        if ! flock "$LOCK" timeout "$BUILD_TIMEOUT_S" $TRTEXEC --onnx=$onnx --fp16 --memPoolSize=workspace:1024 $flags \
            --profilingVerbosity=detailed --exportLayerInfo=$wd/logs/${base}.layers.json --saveEngine=$eng \
            >> "$wd/logs/${base}_build.log" 2>&1; then
          echo "$(date -Iseconds) BUILD FAILED $base, see ${base}_build.log" >> "$wd/failures.txt"; rm -f "$eng"; continue
        fi
        echo "build_seconds $(( $(date +%s) - s ))" >> "$wd/logs/${base}_build.log"
      fi
      echo "$eng $lab $b $sha" >> "$list"
    done
    echo "$(date -Iseconds) built $lab" >> "$wd/progress.txt"
  done < "$man"
  for r in $(seq 1 "$nruns"); do
    while read -r eng lab b sha; do
      exec 9>"$LOCK"; flock 9
      one_run "$eng" "$lab" "$b" "$r" "$sha" "$wd"
      flock -u 9; exec 9>&-
    done < "$list"
    echo "$(date -Iseconds) round $r done" >> "$wd/progress.txt"
  done
  env_dump "batch end" > "$wd/env/batch_end.txt"
  echo "$(date -Iseconds) batch done $man" >> "$wd/progress.txt"
}

case ${1:-} in
  batch) shift; batch "$@" ;;
  *) echo "usage: orin2_run.sh batch <manifest> <workdir> [nruns]"; exit 1 ;;
esac
