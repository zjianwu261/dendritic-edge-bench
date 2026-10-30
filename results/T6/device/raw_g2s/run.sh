set -u
cd $HOME/dendritic-edge-bench/T6
W=$HOME/dendritic-edge-bench/T6/work_g2s
G=$HOME/dendritic-edge-bench/T6/device/g2s
mkdir -p $W
printf '%s' "$HOME/dendritic-edge-bench/T6/onnx/g2s/dhssm_L2_n128_b4_cont_vs_t2__K50.onnx dhssm_L2_n128_b4_cont_vs_t2__K50 default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/lru_L2_H64_N128_lr0.001_vs_t2__K50.onnx lru_L2_H64_N128_lr0.001_vs_t2__K50 default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/dssmU_L2_n128_b4_spk_vs_t2__K50.onnx dssmU_L2_n128_b4_spk_vs_t2__K50 default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/gru_h768_lr0.003_vs_t2__seq.onnx gru_h768_lr0.003_vs_t2__seq default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/dhssm_L2_n128_b4_cont_vs_t3__K50.onnx dhssm_L2_n128_b4_cont_vs_t3__K50 default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/lru_L2_H64_N128_lr0.001_vs_t3__K50.onnx lru_L2_H64_N128_lr0.001_vs_t3__K50 default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/dssmU_L2_n128_b4_spk_vs_t3__K50.onnx dssmU_L2_n128_b4_spk_vs_t3__K50 default,nocublas
$HOME/dendritic-edge-bench/T6/onnx/g2s/gru_h768_lr0.003_vs_t3__seq.onnx gru_h768_lr0.003_vs_t3__seq default,nocublas
" > $G/manifest.txt
echo "$(date -Iseconds) start" >> $G/progress.txt
bash bin/t6_orin.sh batch $G/manifest.txt $W 0
cp $W/engines_manifest.txt.txt $G/engines.txt
cp $W/failures.txt $G/build_failures.txt 2>/dev/null
mkdir -p $G/logs && cp $W/logs/*_build.log $G/logs/ 2>/dev/null
while read -r eng lab b sha; do
  [ -s $G/devout/${lab}_${b}.npy ] && continue
  flock /tmp/dn_orin_bench.lock python3 bin/t6_trt_eval.py $eng device/eval/eval_shd.npz $G/devout/${lab}_${b}.npy >> $G/eval.log 2>&1 \
    || echo "$(date -Iseconds) EVAL FAILED $lab $b" >> $G/failures.txt
  echo "$(date -Iseconds) evaluated $lab $b $(basename $eng)" >> $G/progress.txt
done < $G/engines.txt
echo "$(date -Iseconds) done" >> $G/progress.txt
