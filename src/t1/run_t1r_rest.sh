#!/usr/bin/env bash
# Device runs of the G2 models with validation-selected weights, then rebuild the tables.
# Safe to re-run: make_results.py always takes the last 3 runs per label/build.
# Log: results/T1/logs/t1r_rest.out ; done-marker: results/T1/logs/t1r_rest.done
set -uo pipefail
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PYTHON:-python}
cd $ROOT
G2="dhsrnn_n64_b8_val_seqloop.onnx lif_srnn_n64_val_seqloop.onnx gru_pm_val_seqloop.onnx gru_pm_val_seq.onnx gru_acc_val_seqloop.onnx gru_acc_val_seq.onnx gru_acc_val_step.onnx"
bash src/t1/run_bench.sh both $G2
cd $ROOT/src/t1
OMP_NUM_THREADS=4 nice $PY device_acc.py fmnist mlp_h1024_fmnist
OMP_NUM_THREADS=4 nice $PY device_acc.py pmnist mlp_h3941_matchstored_k4 mlp_h5230_matchstored_k8
OMP_NUM_THREADS=4 nice $PY device_acc.py shd dhsrnn_n64_b8_val_seqloop lif_srnn_n64_val_seqloop gru_pm_val_seqloop gru_pm_val_seq gru_acc_val_seqloop gru_acc_val_seq
python3 make_results.py > $ROOT/results/T1/logs/make_results_v07.out
date > $ROOT/results/T1/logs/t1r_rest.done
