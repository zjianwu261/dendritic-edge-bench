#!/usr/bin/env bash
# C-c multi-seed timing on Orin #1 (2026-10-26). Orin-side batch runner.
# T1 timing protocol (src/bench/run_trt_sync_jetson.sh + trt_sync_bench.py): TensorRT 8.5 FP16, trtexec build with the
# T1 flags (--fp16 [--tacticSources=-CUBLAS,-CUBLAS_LT] --profilingVerbosity=detailed --exportLayerInfo, no workspace
# limit), batch 1, synchronous e2e timing, warm-up >= 2 s and >= 50 iterations, 500 timed iterations, fresh process per
# run, env (nvpmodel, jetson_clocks --show, devfreq, thermal, loadavg) archived before/after every run, tegrastats 100 ms.
# Differences to run_trt_sync_jetson.sh (taken from src/t6/orin/t6_orin.sh, the E-7 / buildvar rules):
#   * several builds per ONNX: the label (s<seed>b<k>_<model>) is in the engine name, so every manifest line is a
#     fresh trtexec build; MemAvailable >= MIN_AVAIL_MB checked before every build (inside the lock);
#   * timed runs in rounds interleaved across engines (round r = run r of every engine);
#   * REQUIRE_LOCK_HZ: a run whose GPU devfreq min/cur/max is not all == REQUIRE_LOCK_HZ is skipped (unlocked.txt).
#   * jetson_clocks / nvpmodel are only queried (--show / -q), never set.
# Every build and every timed run holds flock /tmp/dn_orin_bench.lock.
# Usage: ms_orin.sh batch <manifest: onnx label build[,build]> <workdir> [nruns=3]
#        ms_orin.sh energy <list: label engine> <outdir> <nreps=3>   (src/bench/energy_one.sh, T1 energy protocol)
set -uo pipefail
TRTEXEC=/usr/src/tensorrt/bin/trtexec
LOCK=/tmp/dn_orin_bench.lock
MIN_AVAIL_MB=${MIN_AVAIL_MB:-1500}
G=/sys/devices/gpu.0/devfreq/17000000.ga10b
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$HOME/dendritic-edge-bench/T1/bin/trt_sync_bench.py
ENERGY_ONE=$HOME/dendritic-edge-bench/energy/bin/energy_one.sh

env_dump() {
  echo "## $(date -Iseconds) $1"
  echo "## nvpmodel -q"; nvpmodel -q 2>&1 || true
  echo "## jetson_clocks --show"; jetson_clocks --show 2>&1 || true
  echo "## gpu devfreq governor/cur/min/max"; cat $G/governor $G/cur_freq $G/min_freq $G/max_freq 2>&1 || true
  echo "## cpu cur freq (kHz)"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 | tr '\n' ' '; echo
  echo "## cpu governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1 || true
  echo "## thermal (type temp_mC)"; for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(cat $z/temp)"; done
  echo "## loadavg"; cat /proc/loadavg
  echo "## MemAvailable_MB $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
  echo "## other GPU/bench processes"; ps -eo pid,comm,args | grep -E "trtexec|trt_sync|t6_sync|python3" | grep -v grep | cut -c1-160 || true
}

lock_ok() {
  [ -z "${REQUIRE_LOCK_HZ:-}" ] && return 0
  local mn cu mx; mn=$(cat $G/min_freq); cu=$(cat $G/cur_freq); mx=$(cat $G/max_freq)
  [ "$mn" = "$REQUIRE_LOCK_HZ" ] && [ "$cu" = "$REQUIRE_LOCK_HZ" ] && [ "$mx" = "$REQUIRE_LOCK_HZ" ] && return 0
  echo "min=$mn cur=$cu max=$mx"; return 1
}

one_run() {  # $1 engine $2 label $3 build $4 run $5 sha $6 wd   (called under the lock)
  local eng=$1 lab=$2 b=$3 r=$4 sha=$5 wd=$6
  local base; base=$(basename "$eng" .engine)
  env_dump "before $lab $b run $r" > "$wd/env/${base}_r$r.txt"
  local why
  if ! why=$(lock_ok); then
    echo "$(date -Iseconds) UNLOCKED_SKIP $base r$r $why" >> "$wd/unlocked.txt"
    echo "## LOCK_CHECK FAIL $why -> run skipped" >> "$wd/env/${base}_r$r.txt"
    return 0
  fi
  echo "## LOCK_CHECK OK ${REQUIRE_LOCK_HZ:-unset}" >> "$wd/env/${base}_r$r.txt"
  tegrastats --interval 100 > "$wd/tegra/${base}_r$r.txt" 2>&1 &
  local tp=$!
  python3 "$BENCH" "$eng" --label "$lab" --precision fp16 --build "$b" --onnx_sha256 "$sha" --run "$r" \
    --out "$wd/results.jsonl" > /dev/null
  local rc=$?
  kill $tp 2>/dev/null || true
  env_dump "after run $r rc=$rc" >> "$wd/env/${base}_r$r.txt"
  [ $rc -eq 0 ] || echo "$(date -Iseconds) RUN FAILED $eng r$r" >> "$wd/failures.txt"
}
export -f one_run env_dump lock_ok
export BENCH G REQUIRE_LOCK_HZ

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
      local flags=""
      case $b in default) flags="" ;; nocublas) flags="--tacticSources=-CUBLAS,-CUBLAS_LT" ;; *) echo "bad build $b"; exit 2 ;; esac
      local base="${lab}_fp16_${b}"
      local eng="$wd/engines/${base}_${sha:0:12}.engine"
      if [ ! -s "$eng" ]; then
        local s; s=$(date +%s)
        echo "## onnx $onnx sha256 $sha build $b flags $flags" > "$wd/logs/${base}_build.log"
        # shellcheck disable=SC2086
        if ! flock "$LOCK" bash -c "[ \$(awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo) -ge $MIN_AVAIL_MB ] || { echo 'SKIP: low memory after acquiring the lock'; exit 3; }; exec $TRTEXEC --onnx=$onnx --fp16 $flags --profilingVerbosity=detailed \
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

energy() {  # rounds interleaved across engines; energy_one.sh holds the lock for the whole measurement
  local list=$1 out=$2 nrep=${3:-3}
  mkdir -p "$out"
  echo "$(date -Iseconds) energy start $list" >> "$out/progress.txt"
  for r in $(seq 1 "$nrep"); do
    while read -r lab eng; do
      [ -z "$lab" ] && continue
      grep -q "^$lab r$r " "$out/done.txt" 2>/dev/null && continue
      local ok=0
      for att in 1 2 3; do
        local why
        if ! why=$(lock_ok); then
          echo "$(date -Iseconds) UNLOCKED_SKIP energy $lab r$r $why" >> "$out/unlocked.txt"; sleep 60; continue
        fi
        local tag="r${r}_$(date +%m%d%H%M%S)"
        bash "$ENERGY_ONE" "$lab" "$eng" "$out" "$tag" >> "$out/energy_stdout.txt" 2>&1
        local rc=$?
        echo "$(date -Iseconds) $lab $tag rc=$rc" >> "$out/progress.txt"
        if [ $rc -eq 0 ]; then echo "$lab r$r $tag" >> "$out/done.txt"; ok=1; break; fi
        sleep 30
      done
      [ $ok -eq 1 ] || echo "$(date -Iseconds) ENERGY FAILED $lab r$r" >> "$out/failures.txt"
    done < "$list"
    echo "$(date -Iseconds) energy round $r done" >> "$out/progress.txt"
  done
  echo "$(date -Iseconds) energy done $list" >> "$out/progress.txt"
}

case $1 in
  batch) shift; batch "$@" ;;
  energy) shift; energy "$@" ;;
  *) echo "usage: ms_orin.sh batch <manifest> <workdir> [nruns] | energy <list> <outdir> [nreps]"; exit 1 ;;
esac
