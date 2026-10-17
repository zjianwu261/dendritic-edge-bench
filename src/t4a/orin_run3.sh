#!/usr/bin/env bash
# Orin #1 headless (~5.5 GB free), clocks locked externally (this script never touches
# jetson_clocks / nvpmodel). Build the NoMaD encoder (workspace <= 1 GB; skipped if MemAvailable < 2 GB),
# then time the full explore pipeline (encoder 1x + UNet 10x + host DDPM step) with trt_pipeline_bench.py.
set -uo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=/usr/src/tensorrt/bin/trtexec; WD=trt
build() { # name variant extra
  local eng=$WD/$1_fp16_$2.engine
  [ -s "$eng" ] && return 0
  flock /tmp/dn_orin_bench.lock bash -c "
    avail=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo); echo \"MemAvailable_MB \$avail\" > $WD/$1_$2_build3.log
    if [ \$avail -lt 2048 ]; then echo SKIPPED_LOW_MEM >> $WD/$1_$2_build3.log; exit 0; fi
    s=\$(date +%s); $TRTEXEC --onnx=models/$1.onnx --fp16 --memPoolSize=workspace:1024 $3 --saveEngine=$eng >> $WD/$1_$2_build3.log 2>&1
    echo build_seconds \$(( \$(date +%s) - s )) >> $WD/$1_$2_build3.log"
}
build nomad_enc_explore_b1 default ""
build nomad_enc_explore_b1 nocublas "--tacticSources=-CUBLAS,-CUBLAS_LT"
for variant in default nocublas; do
  e=$WD/nomad_enc_explore_b1_fp16_${variant}.engine; [ -s "$e" ] || continue
  for r in 1 2 3; do
    flock /tmp/dn_orin_bench.lock python3 bin/trt_sync_bench.py $e --label nomad_enc_explore_b1/$variant --build $variant --run $r --out $WD/results3.jsonl --warmup 50 --iters 300
  done
  for n in 8 1; do
    u=$WD/nomad_unet_b${n}_fp16_${variant}.engine; [ -s "$u" ] || continue
    for r in 1 2 3; do
      flock /tmp/dn_orin_bench.lock python3 bin/trt_pipeline_bench.py $e $u --n $n --label nomad_explore_pipeline_n$n/$variant --build $variant --run $r --out $WD/pipeline.jsonl
    done
  done
done
echo DONE3
