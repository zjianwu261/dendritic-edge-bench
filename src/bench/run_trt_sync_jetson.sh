#!/usr/bin/env bash
# Build a TensorRT engine on the Jetson and time it with synchronous per-inference timing
# (trt_sync_bench.py). Each of NRUNS runs is a fresh process (engine reloaded).
# The power mode / clock / thermal state is archived before and after every run;
# tegrastats (100 ms, no root needed) runs during every timed run to log EMC / GR3D.
#   BUILD=default  : trtexec defaults
#   BUILD=nocublas : --tacticSources=-CUBLAS,-CUBLAS_LT
#   Engine file name carries the build variant and the first 12 hex of the ONNX sha256, so a
#   re-exported ONNX is never timed with a stale engine; the per-layer info of
#   the engine (tactic / kernel names) is exported to <engine>.layers.json.
#   Every build and every timed run is serialised with other benchmark jobs via
#   flock /tmp/dn_orin_bench.lock.
# Usage: run_trt_sync_jetson.sh <model.onnx> <fp16|fp32> <workdir> <label> [nruns=3] [build=default]
set -euo pipefail
ONNX=$1; PREC=$2; WD=$3; LABEL=$4; NRUNS=${5:-3}; BUILD=${6:-default}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
LOCK=/tmp/dn_orin_bench.lock
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WD"
SHA=$(sha256sum "$ONNX" | cut -c1-64)
case $BUILD in
  default) EXTRA="" ;;
  nocublas) EXTRA="--tacticSources=-CUBLAS,-CUBLAS_LT" ;;
  *) echo "unknown build $BUILD" >&2; exit 2 ;;
esac
base=$(basename "$ONNX" .onnx)_${PREC}_${BUILD}
eng="$WD/${base}_${SHA:0:12}.engine"
flag=""; [ "$PREC" = fp16 ] && flag="--fp16"
if [ ! -s "$eng" ]; then
  s=$(date +%s)
  { echo "## onnx $ONNX sha256 $SHA build $BUILD"; } > "$WD/${base}_build.log"
  flock "$LOCK" $TRTEXEC --onnx="$ONNX" $flag $EXTRA --profilingVerbosity=detailed \
      --exportLayerInfo="$WD/${base}.layers.json" --saveEngine="$eng" >> "$WD/${base}_build.log" 2>&1 \
    || { echo "{\"label\": \"$LABEL\", \"build\": \"$BUILD\", \"error\": \"trtexec build failed, see ${base}_build.log\"}"; exit 1; }
  echo "build_seconds $(( $(date +%s) - s ))" >> "$WD/${base}_build.log"
fi
env_dump() {
  echo "## $(date -Iseconds) label=$LABEL build=$BUILD $1"
  echo "## nvpmodel -q"; nvpmodel -q 2>&1 || true
  echo "## jetson_clocks --show"; jetson_clocks --show 2>&1 || true
  G=/sys/devices/gpu.0/devfreq/17000000.ga10b
  echo "## gpu devfreq governor/cur/min/max"; cat $G/governor $G/cur_freq $G/min_freq $G/max_freq 2>&1 || true
  echo "## cpu cur freq (kHz)"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 | tr '\n' ' '; echo
  echo "## cpu governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1 || true
  echo "## thermal (type temp_mC)"; for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(cat $z/temp)"; done
  echo "## loadavg"; cat /proc/loadavg
  echo "## other GPU/bench processes"; ps -eo pid,comm,args | grep -E "trtexec|trt_sync|python3" | grep -v grep | cut -c1-160 || true
}
one_run() {  # executed under the lock
  r=$1
  env_dump "before run $r" > "$WD/${base}_env_r$r.txt"
  tegrastats --interval 100 > "$WD/${base}_tegra_r$r.txt" 2>&1 &
  TP=$!
  python3 "$HERE/trt_sync_bench.py" "$eng" --label "$LABEL" --precision "$PREC" --build "$BUILD" \
    --onnx_sha256 "$SHA" --run "$r" --out "$WD/results.jsonl"
  kill $TP 2>/dev/null || true
  env_dump "after run $r" >> "$WD/${base}_env_r$r.txt"
}
export -f one_run env_dump
export WD LABEL BUILD base eng PREC SHA HERE
for r in $(seq 1 "$NRUNS"); do
  flock "$LOCK" bash -c "one_run $r"
done
