#!/usr/bin/env bash
# C-c multi-seed retraining driver (CPU; see ms_train.py). Longest jobs first, P parallel workers,
# T threads each. A job is skipped if logs/<job>_s<si>.done exists.
# Usage: ms_run_all.sh [P=6] [T=2]
set -uo pipefail
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PYTHON:-python}
L=$ROOT/results/robustness_B/Cc_multiseed/logs
P=${1:-6}; T=${2:-2}
mkdir -p "$L"
JOBS="g2v_dhsrnn g2o_dhsrnn g2v_gruacc g2v_lif g2o_lif g2o_gru23 g3_mlp5230 g3_mlp3941 g2v_grupm g3_mlp4965 g3_mlp3592 g3_ad8 g3_ad4 g1_dann g1_vann g1_mlp1024 g1_mlp85"
one() {
  job=$1; si=$2
  [ -e "$L/${job}_s${si}.done" ] && return 0
  # v2: skip a job that another driver instance is already running
  pgrep -f "ms_train.py --cpu $job $si\$" > /dev/null && { echo "$(date -Iseconds) SKIP running elsewhere $job s$si" >> "$L/driver.log"; return 0; }
  echo "$(date -Iseconds) START $job s$si" >> "$L/driver.log"
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T nice -n 10 $PY -u $ROOT/src/robustness_B/ms_train.py --cpu "$job" "$si" > "$L/${job}_s${si}.log" 2>&1
  rc=$?
  echo "$(date -Iseconds) END $job s$si rc=$rc" >> "$L/driver.log"
  [ $rc -eq 0 ] && date -Iseconds > "$L/${job}_s${si}.done"
}
export -f one
export L PY ROOT T
for j in $JOBS; do for si in 2 3; do echo "$j $si"; done; done | xargs -P "$P" -n 2 bash -c 'one "$0" "$1"'
for si in 2 3; do OMP_NUM_THREADS=$T $PY -u $ROOT/src/robustness_B/ms_train.py --cpu export $si > "$L/export_s${si}.log" 2>&1; echo "$(date -Iseconds) export s$si rc=$?" >> "$L/driver.log"; done
echo "$(date -Iseconds) ALL DONE" >> "$L/driver.log"
