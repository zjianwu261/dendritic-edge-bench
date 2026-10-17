#!/usr/bin/env bash
# Energy measurement chain on Orin #1 (runs ON the Orin; started detached by energy_bench.sh start).
# For every config list (T1 main table, then T6 key comparisons) it does 3 good runs per config,
# rounds interleaved across configs (round r measures every config that has < r good runs), each run
# via energy_one.sh (holds the shared bench lock). A flagged run (trtexec in window, high load,
# failed bench) is re-measured up to 2 more times right away; a "not run" (MemAvailable low) waits.
# Final pass: configs whose engine did not exist yet (T6 engines still being built by the T6 chain,
# validation-selected G2 engines) are retried every 10 min until DEADLINE_H hours after start.
# Config list lines:   label | spec
#   spec = t1:<name>[,<preferred name>...]  -> newest ~/dendritic-edge-bench/T1/trt_v07/<name>_fp16_nocublas_*.engine,
#                                             trying names right-to-left (last = preferred, e.g. *_val_*)
#          t6:<onnx path>                   -> ~/dendritic-edge-bench/T6/work/engines/<onnx base>_nocublas_<sha12>.engine
# Good runs are recorded in <outdir>/done.txt as "label engine tag"; resumable (just start again).
# Usage: energy_chain.sh run  <list1> <outdir1> [<list2> <outdir2> ...]
#        energy_chain.sh compare <outdir> <engine>      (jtop vs tegrastats, same 40 s window)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NRUNS=${NRUNS:-3}; DEADLINE_H=${DEADLINE_H:-14}
T1E=$HOME/dendritic-edge-bench/T1/trt_v07
T6E=$HOME/dendritic-edge-bench/T6/work/engines
START=$(date +%s)
log() { echo "$(date -Iseconds) $*"; }

resolve() {  # spec -> engine path (or empty)
  local spec=$1
  case $spec in
    t1:*)
      local names=${spec#t1:} e="" n
      for n in ${names//,/ }; do
        local c; c=$(ls -t "$T1E/${n}_fp16_nocublas_"*.engine 2>/dev/null | head -1)
        [ -n "$c" ] && [ -s "$c" ] && e=$c
      done
      echo "$e" ;;
    t6:*)
      local onnx=${spec#t6:}; onnx=${onnx/#\~/$HOME}
      [ -f "$onnx" ] || { echo ""; return; }
      local sha; sha=$(sha256sum "$onnx" | cut -c1-12)
      local e="$T6E/$(basename "$onnx" .onnx)_nocublas_${sha}.engine"
      [ -s "$e" ] && echo "$e" || echo "" ;;
    *) [ -s "$spec" ] && echo "$spec" || echo "" ;;
  esac
}

ngood() { local c=0; [ -f "$2/done.txt" ] && c=$(grep -c "^$1 $3 " "$2/done.txt"); echo "${c:-0}"; }

measure() {  # label engine outdir k ; up to 3 attempts; waits on low memory
  local lab=$1 eng=$2 out=$3 k=$4 att=0 rc waits=0
  while [ $att -lt 3 ]; do
    local tag="r${k}_$(date +%m%d%H%M%S)"
    bash "$HERE/energy_one.sh" "$lab" "$eng" "$out" "$tag"
    rc=$?
    if [ $rc -eq 0 ]; then echo "$lab $eng $tag" >> "$out/done.txt"; log "OK $lab $tag"; return 0; fi
    if [ $rc -eq 4 ]; then
      waits=$((waits + 1)); log "WAIT mem $lab ($waits)"; [ $waits -gt 30 ] && return 1; sleep 120; continue
    fi
    att=$((att + 1)); log "FLAGGED/FAILED rc=$rc $lab $tag attempt $att"; sleep 60
  done
  return 1
}

one_pass() {  # list outdir round  -> prints number of configs still missing an engine
  local list=$1 out=$2 r=$3 missing=0
  while IFS='|' read -r lab spec; do
    lab=$(echo "$lab" | xargs); spec=$(echo "$spec" | xargs)
    [ -z "$lab" ] && continue; case $lab in \#*) continue ;; esac
    local eng; eng=$(resolve "$spec")
    if [ -z "$eng" ]; then missing=$((missing + 1)); log "no engine yet: $lab ($spec)" >&2; continue; fi
    local g; g=$(ngood "$lab" "$out" "$eng")
    [ "$g" -ge "$r" ] && continue
    measure "$lab" "$eng" "$out" $((g + 1)) >&2
  done < "$list"
  echo $missing
}

case $1 in
  run)
    shift; args=("$@")
    log "chain start pid $$ lists: $*"
    for r in $(seq 1 "$NRUNS"); do
      for ((i = 0; i < ${#args[@]}; i += 2)); do one_pass "${args[i]}" "${args[i+1]}" "$r" > /dev/null; done
    done
    while true; do  # final pass: missing engines / makeup runs (e.g. *_val_* engine appeared later)
      miss=0
      for ((i = 0; i < ${#args[@]}; i += 2)); do
        m=$(one_pass "${args[i]}" "${args[i+1]}" "$NRUNS"); miss=$((miss + m))
      done
      [ "$miss" -eq 0 ] && break
      if [ $(( $(date +%s) - START )) -gt $((DEADLINE_H * 3600)) ]; then log "deadline reached, $miss configs without engine"; break; fi
      log "final pass: $miss configs without engine, sleeping 600 s"; sleep 600
    done
    log "chain done"; date -Iseconds > "${args[1]}/../chain.done" ;;
  compare)
    out=$2; eng=$3; mkdir -p "$out"
    exec 9>/tmp/dn_orin_bench.lock; flock 9
    log "compare start (lock held)"
    python3 "$HERE/power_logger.py" "$out/compare_jtop.csv" --interval 0.1 9>&- &
    PL=$!
    tegrastats --interval 100 > "$out/compare_tegrastats.txt" 2>&1 9>&- &
    TG=$!
    sleep 15
    python3 "$HERE/trt_energy_bench.py" "$eng" --duration 15 --label compare > "$out/compare_bench.json"
    sleep 12
    kill $PL $TG; wait $PL
    { echo "tegrastats lines: $(wc -l < "$out/compare_tegrastats.txt")";
      echo "tegrastats lines containing VDD_: $(grep -c VDD_ "$out/compare_tegrastats.txt")";
      echo "jtop samples: $(grep -vc '^[#t]' "$out/compare_jtop.csv")";
      ls -l /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in1_input /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr1_input;
      id; } > "$out/compare_summary.txt" 2>&1
    log "compare done" ;;
esac
