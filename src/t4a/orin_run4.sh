#!/usr/bin/env bash
# NoMaD encoder without InstanceNormalization plugins (GroupNorm decomposed, onnxsim), full and
# split (encA EfficientNets + encB transformer); also the onnxsim'd original. Builds and timings under
# flock /tmp/dn_orin_bench.lock; build skipped if MemAvailable < 2 GB. Clocks are never touched.
set -uo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=/usr/src/tensorrt/bin/trtexec; WD=trt
build() { # onnx_base
  local eng=$WD/$1_fp16.engine
  [ -s "$eng" ] && return 0
  flock /tmp/dn_orin_bench.lock bash -c "
    avail=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo); echo \"MemAvailable_MB \$avail\" > $WD/$1_build4.log
    if [ \$avail -lt 2048 ]; then echo SKIPPED_LOW_MEM >> $WD/$1_build4.log; exit 0; fi
    s=\$(date +%s); $TRTEXEC --onnx=models/$1.onnx --fp16 --memPoolSize=workspace:1024 --saveEngine=$eng >> $WD/$1_build4.log 2>&1
    echo build_seconds \$(( \$(date +%s) - s )) >> $WD/$1_build4.log"
}
for b in nomad_enc_explore_b1_gnd nomad_encA_b1_gnd nomad_encB_b1_gnd nomad_enc_explore_b1_sim; do build $b; done
for b in nomad_enc_explore_b1_gnd nomad_encA_b1_gnd nomad_encB_b1_gnd nomad_enc_explore_b1_sim; do
  e=$WD/${b}_fp16.engine; [ -s "$e" ] || continue
  for r in 1 2 3; do flock /tmp/dn_orin_bench.lock python3 bin/trt_sync_bench.py $e --label $b --build default --run $r --out $WD/results4.jsonl --warmup 50 --iters 300; done
done
U8=$WD/nomad_unet_b8_fp16_default.engine; U1=$WD/nomad_unet_b1_fp16_default.engine
for n in 8 1; do
  U=$WD/nomad_unet_b${n}_fp16_default.engine
  if [ -s $WD/nomad_enc_explore_b1_gnd_fp16.engine ]; then
    for r in 1 2 3; do flock /tmp/dn_orin_bench.lock python3 bin/trt_pipeline_bench.py $WD/nomad_enc_explore_b1_gnd_fp16.engine $U --n $n --label pipeline_gnd_full_n$n --build default --run $r --out $WD/pipeline.jsonl; done
  fi
  if [ -s $WD/nomad_encA_b1_gnd_fp16.engine ] && [ -s $WD/nomad_encB_b1_gnd_fp16.engine ]; then
    for r in 1 2 3; do flock /tmp/dn_orin_bench.lock python3 bin/trt_pipeline_bench.py $WD/nomad_encA_b1_gnd_fp16.engine $U --enc2 $WD/nomad_encB_b1_gnd_fp16.engine --n $n --label pipeline_gnd_split_n$n --build default --run $r --out $WD/pipeline.jsonl; done
  fi
done
echo DONE4
