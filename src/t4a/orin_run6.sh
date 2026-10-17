#!/usr/bin/env bash
# dend2 deployment latency (runs ON Orin #1, from ~/dendritic-edge-bench/T4a): NoMaD explore chain with the dend2 adapters.
# Builds nomad_enc_dend2{id,inf}_b1_gnd (same flags as orin_run4.sh: --fp16, 1 GB workspace, skipped if
# MemAvailable < 2 GB), then times, interleaved in the same session, 3 fresh processes each:
#   encoder alone (trt_sync_bench, warm-up 50, 300 iters): base gnd encoder, dend2id, dend2inf
#   full chain (trt_pipeline_bench, encoder + 10 x UNet, n = 8 and n = 1): base, dend2id, dend2inf
# Every build and every timed run holds flock /tmp/dn_orin_bench.lock; clocks are never touched.
set -uo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=/usr/src/tensorrt/bin/trtexec; WD=trt; M=dend2_onnx; OUTD=$WD/dend2
mkdir -p $OUTD/env
env_dump() {
  echo "## $(date -Iseconds) $1"; nvpmodel -q 2>&1 | head -2
  G=/sys/devices/gpu.0/devfreq/17000000.ga10b; echo "## gpu cur/min/max"; cat $G/cur_freq $G/min_freq $G/max_freq
  echo "## cpu kHz"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tr '\n' ' '; echo
  echo "## loadavg"; cat /proc/loadavg
}
export -f env_dump
build() { # onnx_base
  local eng=$WD/$1_fp16.engine
  [ -s "$eng" ] && return 0
  flock /tmp/dn_orin_bench.lock bash -c "
    avail=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo); echo \"MemAvailable_MB \$avail\" > $OUTD/$1_build.log
    if [ \$avail -lt 2048 ]; then echo SKIPPED_LOW_MEM >> $OUTD/$1_build.log; exit 0; fi
    s=\$(date +%s); $TRTEXEC --onnx=$M/$1.onnx --fp16 --memPoolSize=workspace:1024 --saveEngine=$eng >> $OUTD/$1_build.log 2>&1
    echo build_seconds \$(( \$(date +%s) - s )) >> $OUTD/$1_build.log"
}
build nomad_enc_dend2id_b1_gnd
build nomad_enc_dend2inf_b1_gnd
ENCS="nomad_enc_explore_b1_gnd nomad_enc_dend2id_b1_gnd nomad_enc_dend2inf_b1_gnd"
for r in 1 2 3; do
  for b in $ENCS; do
    e=$WD/${b}_fp16.engine; [ -s "$e" ] || continue
    flock /tmp/dn_orin_bench.lock bash -c "env_dump 'before $b r$r' > $OUTD/env/enc_${b}_r$r.txt; python3 bin/trt_sync_bench.py $e --label $b --build default --run $r --out $OUTD/enc.jsonl --warmup 50 --iters 300 > /dev/null; env_dump after >> $OUTD/env/enc_${b}_r$r.txt"
  done
  for n in 8 1; do
    U=$WD/nomad_unet_b${n}_fp16_default.engine
    for b in $ENCS; do
      e=$WD/${b}_fp16.engine; [ -s "$e" ] || continue
      flock /tmp/dn_orin_bench.lock bash -c "env_dump 'before chain $b n$n r$r' > $OUTD/env/chain_${b}_n${n}_r$r.txt; python3 bin/trt_pipeline_bench.py $e $U --n $n --label chain_${b}_n$n --build default --run $r --out $OUTD/pipeline.jsonl > /dev/null; env_dump after >> $OUTD/env/chain_${b}_n${n}_r$r.txt"
    done
  done
  echo "$(date -Iseconds) round $r done" >> $OUTD/progress.txt
done
echo "$(date -Iseconds) DONE6" >> $OUTD/progress.txt
