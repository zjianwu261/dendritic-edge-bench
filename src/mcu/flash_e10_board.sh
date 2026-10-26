#!/bin/bash
# Board timing of the equal-budget seed-1 builds: the 4 E-10b board models (dann_d8_s32, dann_d8_s64, cnn_c2_4,
# cnn_c3_6), the 12 remaining E-10 / E-10b rows that were first reported as estimates, and the two CNN variants
# cnns4_c2_4 / cnns4_c4_8 (PROTOCOL_cnnvar.md). Flashing ERASES the program currently on the board.
# Same procedure as src/mcu/step4_flash_and_measure.sh (stcgal -t 35000, then host_uart.py, 100 images,
# logit bit-exact vs pc_infer of the same build), but the hex / pc_infer come from the selected-cell seed-1
# build in results/T1m/e10/build/<cell>_t1 (cnnvar: results/T1m/e10/cnnvar/build/<cell>_t1; written by
# `e10.py verify` / `cnnvar.py verify`; not included in the release), and stcgal power-cycles the board itself
# (-a, DTR autoreset).
#   bash src/mcu/flash_e10_board.sh dann_d8_s32
set -e
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
STCGAL=${STCGAL:-stcgal}
PYTHON=${PYTHON:-python}
PORT=${PORT:-/dev/ttyUSB0}
declare -A CELL=([dann_d8_s32]=dann_d8_s32_lr0.01_wd0 [dann_d8_s64]=dann_d8_s64_lr0.003_wd0
                 [cnn_c2_4]=cnn_c2_4_lr0.003_wd0.01 [cnn_c3_6]=cnn_c3_6_lr0.003_wd0.01
                 [dann_prng_d8_s32]=dann_prng_d8_s32_lr0.003_wd0.01 [dann_prng_d8_s64]=dann_prng_d8_s64_lr0.003_wd0.01
                 [mlp_h16]=mlp_h16_lr0.003_wd0 [mlp_h32]=mlp_h32_lr0.003_wd0.01
                 [dann_prng_d16_s64]=dann_prng_d16_s64_lr0.003_wd0 [dann_d24_s64]=dann_d24_s64_lr0.003_wd0.01
                 [dann_prng_d24_s64]=dann_prng_d24_s64_lr0.003_wd0 [dann_prng_d32_s96]=dann_prng_d32_s96_lr0.003_wd0
                 [dann_prng_d24_s128]=dann_prng_d24_s128_lr0.003_wd0 [mlp_h72]=mlp_h72_lr0.001_wd0.01
                 [cnn_c16_32]=cnn_c16_32_lr0.001_wd0 [cnn_c32_64]=cnn_c32_64_lr0.001_wd0.01
                 [cnns4_c2_4]=cnns4_c2_4_lr0.01_wd0 [cnns4_c4_8]=cnns4_c4_8_lr0.003_wd0)
M=${1:?usage: $0 <one of the 18 board models listed in CELL>}
[ -n "${CELL[$M]}" ] || { echo "refusing: $M is not one of the 18 board models"; exit 1; }
BDIR=$ROOT/results/T1m/e10/build/${CELL[$M]}_t1
case $M in cnns4_*) BDIR=$ROOT/results/T1m/e10/cnnvar/build/${CELL[$M]}_t1;; esac
TAG=${M}_t1
HEX=$BDIR/uart_35M.hex
LOG=$ROOT/results/T1m/logs/stcgal_$TAG.log
[ -f "$HEX" ] || { echo "missing $HEX"; exit 1; }
mkdir -p "$(dirname "$LOG")"
{ echo "# E-10b build: $HEX"; "$STCGAL" -P stc8d -p $PORT -a -t 35000 "$HEX" 2>&1; } | tee "$LOG"
F=$(grep -oP 'Adjusted frequency:\s*\K[0-9.]+' "$LOG" | tail -1)
[ -n "$F" ] || { echo "no trimmed frequency in $LOG"; exit 1; }
sleep 1
"$PYTHON" $ROOT/src/mcu/host_uart.py --tag $TAG --bdir $BDIR --n 100 --f_mhz $F --port $PORT
