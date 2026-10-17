#!/usr/bin/env bash
# Runs ON Orin #1. Build TRT FP16 engines (default tactics and no-cuBLAS) and time them with
# trt_sync_bench.py (copied unchanged from src/bench). Builds AND timings are both done while holding
# flock /tmp/dn_orin_bench.lock (other benchmark jobs share this board). 3 fresh processes per engine.
# Usage: orin_run.sh <model_dir> <out_dir> <name1> [name2 ...]   (names without .onnx)
set -uo pipefail
MD=$1; WD=$2; shift 2
TRTEXEC=/usr/src/tensorrt/bin/trtexec
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WD"
env_dump() {
  echo "## $(date -Iseconds) $1"; nvpmodel -q 2>&1 | head -2
  G=/sys/devices/gpu.0/devfreq/17000000.ga10b; cat $G/governor $G/cur_freq $G/min_freq $G/max_freq 2>&1
  for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(cat $z/temp)"; done; cat /proc/loadavg; free -m
}
for name in "$@"; do
  for variant in default nocublas; do
    extra=""; [ $variant = nocublas ] && extra="--tacticSources=-CUBLAS,-CUBLAS_LT"
    eng=$WD/${name}_fp16_${variant}.engine
    if [ ! -s "$eng" ]; then
      flock /tmp/dn_orin_bench.lock bash -c "s=\$(date +%s); $TRTEXEC --onnx=$MD/$name.onnx --fp16 $extra --saveEngine=$eng > $WD/${name}_${variant}_build.log 2>&1; echo build_seconds \$(( \$(date +%s) - s )) >> $WD/${name}_${variant}_build.log" \
        || { echo "{\"label\": \"$name/$variant\", \"error\": \"build failed\"}" >> $WD/results.jsonl; continue; }
    fi
    for r in 1 2 3; do
      flock /tmp/dn_orin_bench.lock bash -c "$(declare -f env_dump); env_dump '$name/$variant r$r' > $WD/${name}_${variant}_env_r$r.txt; python3 $HERE/trt_sync_bench.py $eng --label $name/$variant --run $r --out $WD/results.jsonl --build $variant --warmup 50 --iters 300"
    done
  done
done
echo DONE
