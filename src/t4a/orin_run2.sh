#!/usr/bin/env bash
# Orin #1, NoMaD engines, second pass: the bench loads the TRT plugin library (needed to deserialize the UNet
# engines); fused 10-step graphs are not built (builder exceeds ~6 GB RSS on the 8 GB board).
# All builds and timings under flock /tmp/dn_orin_bench.lock.
set -uo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=/usr/src/tensorrt/bin/trtexec; WD=trt; MD=models
for name in nomad_enc_explore_b1 nomad_enc_nav_b10; do
  for variant in default nocublas; do
    extra="--memPoolSize=workspace:1024"; [ $variant = nocublas ] && extra="$extra --tacticSources=-CUBLAS,-CUBLAS_LT"
    eng=$WD/${name}_fp16_${variant}.engine
    [ -s "$eng" ] || flock /tmp/dn_orin_bench.lock bash -c "s=\$(date +%s); $TRTEXEC --onnx=$MD/$name.onnx --fp16 $extra --saveEngine=$eng > $WD/${name}_${variant}_build2.log 2>&1; echo build_seconds \$(( \$(date +%s) - s )) >> $WD/${name}_${variant}_build2.log"
  done
done
for name in nomad_enc_explore_b1 nomad_enc_nav_b10 nomad_unet_b8 nomad_unet_b1; do
  for variant in default nocublas; do
    eng=$WD/${name}_fp16_${variant}.engine; [ -s "$eng" ] || continue
    for r in 1 2 3; do
      flock /tmp/dn_orin_bench.lock python3 bin/trt_sync_bench.py $eng --label $name/$variant --build $variant --run $r --out $WD/results2.jsonl --warmup 50 --iters 300
    done
  done
done
for variant in default nocublas; do
  for n in 8 1; do
    e=$WD/nomad_enc_explore_b1_fp16_${variant}.engine; u=$WD/nomad_unet_b${n}_fp16_${variant}.engine
    [ -s "$e" ] && [ -s "$u" ] || continue
    for r in 1 2 3; do
      flock /tmp/dn_orin_bench.lock python3 bin/trt_pipeline_bench.py $e $u --n $n --label nomad_explore_pipeline_n$n/$variant --build $variant --run $r --out $WD/pipeline.jsonl
    done
  done
done
echo DONE2
