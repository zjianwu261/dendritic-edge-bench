#!/usr/bin/env bash
# T6 Orin #1 batch runner (adapted from src/bench/run_trt_sync_jetson.sh).
# Runs ON the Jetson.
#   t6_orin.sh batch <manifest> <workdir> [nruns=3]
# manifest lines: <onnx path> <label> <builds, comma separated>
#   builds: default | nocublas (--tacticSources=-CUBLAS,-CUBLAS_LT) |
#           sp_default | sp_nocublas (same + --sparsity=enable, TensorRT 2:4 structured sparsity)
# Phase 1 builds every (onnx, build) engine (skipped if the engine for this ONNX sha256 exists);
# phase 2 times every engine NRUNS times, rounds interleaved across engines, each run a fresh
# process (t6_sync_bench.py = T1 sync e2e bench, warm-up >= 2 s and >= 50 iters).
# EVERY trtexec build and EVERY timed run holds  flock /tmp/dn_orin_bench.lock  (shared with
# other benchmark jobs). Per build: log (INFO level, contains the "(Sparsity)" lines) and the
# per-layer tactic json (--profilingVerbosity=detailed --exportLayerInfo).
# To protect the 8 GB board: builder workspace limited to 1 GB (--memPoolSize=workspace:1024);
# before every build MemAvailable is checked and the build is SKIPPED (logged to skipped.txt) if it
# is < MIN_AVAIL_MB (2048). Engines built before 2026-10-01 16:10 had no workspace limit.
set -uo pipefail
TRTEXEC=/usr/src/tensorrt/bin/trtexec
LOCK=/tmp/dn_orin_bench.lock
MIN_AVAIL_MB=${MIN_AVAIL_MB:-2048}
avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
HERE=$(cd "$(dirname "$0")" && pwd)

build_flags() {
  case $1 in
    default) echo "" ;;
    nocublas) echo "--tacticSources=-CUBLAS,-CUBLAS_LT" ;;
    sp_default) echo "--sparsity=enable" ;;
    sp_nocublas) echo "--sparsity=enable --tacticSources=-CUBLAS,-CUBLAS_LT" ;;
    *) echo "BAD"; ;;
  esac
}

env_dump() {
  echo "## $(date -Iseconds) $1"
  echo "## nvpmodel -q"; nvpmodel -q 2>&1 || true
  G=/sys/devices/gpu.0/devfreq/17000000.ga10b
  echo "## gpu devfreq governor/cur/min/max"; cat $G/governor $G/cur_freq $G/min_freq $G/max_freq 2>&1 || true
  echo "## thermal (type temp_mC)"; for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(cat $z/temp)"; done
  echo "## loadavg"; cat /proc/loadavg
  echo "## cpu cur freq (kHz)"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 | tr '\n' ' '; echo
  echo "## gpu min_freq (locked if == max)"; cat /sys/devices/gpu.0/devfreq/17000000.ga10b/min_freq 2>&1
  echo "## other GPU/bench processes"; ps -eo pid,comm,args | grep -E "trtexec|trt_sync|t6_sync|python3" | grep -v grep | cut -c1-160 || true
}

one_run() {  # $1 engine $2 label $3 build $4 run $5 sha $6 wd  (called under the lock)
  local eng=$1 lab=$2 b=$3 r=$4 sha=$5 wd=$6
  local base; base=$(basename "$eng" .engine)
  env_dump "before $lab $b run $r" > "$wd/env/${base}_r$r.txt"
  tegrastats --interval 100 > "$wd/tegra/${base}_r$r.txt" 2>&1 &
  local tp=$!
  python3 "$HERE/t6_sync_bench.py" "$eng" --label "$lab" --precision fp16 --build "$b" \
    --onnx_sha256 "$sha" --run "$r" --out "$wd/results.jsonl" > /dev/null
  local rc=$?
  kill $tp 2>/dev/null || true
  env_dump "after run $r rc=$rc" >> "$wd/env/${base}_r$r.txt"
  [ $rc -eq 0 ] || echo "$(date -Iseconds) RUN FAILED $eng r$r" >> "$wd/failures.txt"
}
export -f one_run env_dump
export HERE

batch() {
  local man=$1 wd=$2 nruns=${3:-3}
  mkdir -p "$wd/engines" "$wd/logs" "$wd/env" "$wd/tegra"
  local list="$wd/engines_$(basename "$man").txt"
  : > "$list"
  echo "$(date -Iseconds) batch start $man" >> "$wd/progress.txt"
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
        if ! flock "$LOCK" bash -c "[ \$(awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo) -ge $MIN_AVAIL_MB ] || { echo 'SKIP: low memory after acquiring the lock'; exit 3; }; exec $TRTEXEC --onnx=$onnx --fp16 --memPoolSize=workspace:1024 $flags --profilingVerbosity=detailed \
            --exportLayerInfo=$wd/logs/${base}.layers.json --saveEngine=$eng" >> "$wd/logs/${base}_build.log" 2>&1; then
          echo "$(date -Iseconds) BUILD FAILED or SKIPPED (low memory) $base, see ${base}_build.log" >> "$wd/failures.txt"; rm -f "$eng"; continue
        fi
        echo "build_seconds $(( $(date +%s) - s ))" >> "$wd/logs/${base}_build.log"
      fi
      echo "$eng $lab $b $sha" >> "$list"
    done
    echo "$(date -Iseconds) built $lab" >> "$wd/progress.txt"
  done < "$man"
  for r in $(seq 1 "$nruns"); do
    while read -r eng lab b sha; do
      flock "$LOCK" bash -c "one_run '$eng' '$lab' '$b' '$r' '$sha' '$wd'"
    done < "$list"
    echo "$(date -Iseconds) round $r done" >> "$wd/progress.txt"
  done
  echo "$(date -Iseconds) batch done $man" >> "$wd/progress.txt"
}

evalall() {  # device accuracy. $1 engines list (from batch), $2 eval-npz dir, $3 workdir
  local list=$1 ev=$2 wd=$3
  mkdir -p "$wd/devout"
  while read -r eng lab b sha; do
    local npz
    case $lab in
      dev_g1_*) npz=$ev/eval_fmnist.npz ;;
      dev_g2_*) npz=$ev/eval_shd.npz ;;
      dev_g3_*) npz=$ev/eval_pmnist.npz ;;
      *) continue ;;
    esac
    [ -s "$wd/devout/${lab}_${b}.npy" ] && continue
    flock "$LOCK" python3 "$HERE/t6_trt_eval.py" "$eng" "$npz" "$wd/devout/${lab}_${b}.npy" \
      >> "$wd/devout/eval.log" 2>&1 || echo "$(date -Iseconds) EVAL FAILED $lab" >> "$wd/failures.txt"
    echo "$(date -Iseconds) evaluated $lab $b" >> "$wd/progress.txt"
  done < "$list"
}

case $1 in
  batch) shift; batch "$@" ;;
  evalall) shift; evalall "$@" ;;
  *) echo "usage: t6_orin.sh batch <manifest> <workdir> [nruns] | evalall <engines list> <npz dir> <workdir>"; exit 1 ;;
esac
