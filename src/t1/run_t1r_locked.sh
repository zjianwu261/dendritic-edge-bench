#!/usr/bin/env bash
# Locked-clock supplementary device runs. Run from the workstation.
#  (a) one replacement energy run of mlp_h5230_matchstored_k8 (its r1 was taken with unlocked clocks),
#      via src/bench/energy_one.sh on the Orin (holds flock /tmp/dn_orin_bench.lock; up to 2 attempts);
#  (b) appendix column: locked-clock latency of the small models (G1 all + G2 single-step),
#      nocublas engines (hard-linked from trt_v07, same ONNX sha), 3 runs each, raw -> raw/orin1_v07_locked.
#      Clocks are NOT touched by this script (they were locked externally during these runs);
#      every run archives GPU min/max so make_results can check the lock.
#  (c) pull, re-parse energy (T1 only), rebuild results.csv.
# Log: results/T1/logs/t1r_locked.out ; done marker results/T1/logs/t1r_locked.done
set -uo pipefail
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
J=${ORIN_HOST:-orin}
ED=dendritic-edge-bench/energy
cd $ROOT
echo "## start $(date -Iseconds)"
# ---- (a) energy replacement run
scp -q src/bench/energy_one.sh $J:$ED/bin/
ENG=$(ssh $J "ls -t ~/dendritic-edge-bench/T1/trt_v07/mlp_h5230_matchstored_k8_fp16_nocublas_*.engine | head -1")
for att in 1 2; do
  TAG="r4_$(date +%m%d%H%M%S)"
  ssh $J "cd ~/$ED && bash bin/energy_one.sh mlp_h5230_matchstored_k8 $ENG \$HOME/$ED/t1 $TAG"
  rc=$?
  echo "## energy mlp_h5230 $TAG rc=$rc"
  if [ $rc -eq 0 ]; then ssh $J "echo 'mlp_h5230_matchstored_k8 $ENG $TAG' >> ~/$ED/t1/done.txt"; break; fi
  sleep 60
done
rsync -a $J:$ED/t1/ results/T1/raw/orin1_energy/
# ---- (b) locked-clock latency of the small models
SMALL="dann_lrf_d32_s128 mlp_h85_match_d32_s128 vann_same_d32_s128 mlp_h256_fmnist mlp_h512_fmnist mlp_h1024_fmnist dhsrnn_n64_b8_step lif_srnn_n64_step gru_h23_step gru_acc_val_step"
ssh $J "mkdir -p ~/dendritic-edge-bench/T1/trt_v07_locked && cd ~/dendritic-edge-bench/T1 && for m in $SMALL; do cp -l trt_v07/\${m}_fp16_nocublas_*.engine trt_v07_locked/ 2>/dev/null; done; ls trt_v07_locked | wc -l"
ARGS=""; for m in $SMALL; do ARGS="$ARGS $m.onnx"; done
BUILDS=nocublas ORIN_RAW=orin1_v07_locked ORIN_WD=trt_v07_locked bash src/t1/run_bench.sh orin $ARGS
# ---- (c) rebuild
python3 src/bench/energy_parse.py --only T1
python3 src/t1/make_results.py > results/T1/logs/make_results_locked.out
date > results/T1/logs/t1r_locked.done
echo "## done $(date -Iseconds)"
