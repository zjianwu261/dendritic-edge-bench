#!/usr/bin/env bash
# T4a data pipeline (CPU only): process HuRoN + RECON -> build benchmark tasks -> zero-shot NoMaD eval (CPU).
# Idempotent: finished steps are skipped.
# Usage: setsid nohup src/t4a/run_data_pipeline.sh > results/T4a/logs/data_pipeline.log 2>&1 &
set -euo pipefail
trap 'echo "== $(date) FAILED at line $LINENO"; exit 1' ERR
R=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; PY=${PYTHON:-python}; S=$R/src/t4a
D=${DEB_DATA:-$R/data}; B=$D/t4a_bench; L=$R/results/T4a/logs; mkdir -p $L
echo "== $(date) process huron"; $PY $S/process_huron.py --workers 8 > $L/process_huron.log 2>&1
echo "== $(date) process recon"; grep -q "^DONE" $L/process_recon.log 2>/dev/null || $PY $S/process_recon.py --limit 2500 > $L/process_recon.log 2>&1
build() { t=$1; shift
  if [ ! -s $B/$t/info.json ] || [ ! -s $B/$t/test.npz ]; then $PY $S/build_bench.py --task $t "$@" > $L/build_$t.log 2>&1; fi
  for f in frames.npy frames_dark.npy meta.npz test.npz info.json; do [ -s $B/$t/$f ] || { echo "missing $B/$t/$f"; exit 1; }; done
  tail -1 $L/build_$t.log | cut -c1-300; }
H=$D/gnm/huron
build huron_bww8  --src $H --spacing 0.255 --end_slack 3 --env bww8
build huron_soda3 --src $H --spacing 0.255 --end_slack 3 --env soda3
build huron_cory1 --src $H --spacing 0.255 --end_slack 3 --env cory1
build huron_bww12 --src $H --spacing 0.255 --end_slack 3 --env bww1+bww2
build recon --src $D/gnm/recon --spacing 0.25 --end_slack 3 --group '^(jackal_[0-9-]+)_'
echo "== $(date) zero-shot (CPU)"
CUDA_VISIBLE_DEVICES= $PY $S/zero_shot.py --device cpu --threads 6 --tasks huron_bww8 huron_soda3 huron_cory1 huron_bww12 recon > $L/zero_shot.log 2>&1
$PY $S/make_tables.py > /dev/null
echo "== $(date) DONE"
