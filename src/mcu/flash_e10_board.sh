#!/bin/bash
# E-10b board timing of the four equal-budget seed-1 builds (dann_d8_s32, dann_d8_s64, cnn_c2_4, cnn_c3_6).
# Flashing ERASES the program currently on the board.
# Same procedure as src/mcu/step4_flash_and_measure.sh (stcgal -t 35000, then host_uart.py, 100 images,
# logit bit-exact vs pc_infer of the same build), but the hex / pc_infer come from the E-10b selected-cell seed-1
# build in results/T1m/e10/build/<cell>_t1 (written by `e10.py verify`; not included in the release),
# and stcgal power-cycles the board itself (-a, DTR autoreset).
#   bash src/mcu/flash_e10_board.sh dann_d8_s32
set -e
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
STCGAL=${STCGAL:-stcgal}
PYTHON=${PYTHON:-python}
PORT=${PORT:-/dev/ttyUSB0}
declare -A CELL=([dann_d8_s32]=dann_d8_s32_lr0.01_wd0 [dann_d8_s64]=dann_d8_s64_lr0.003_wd0
                 [cnn_c2_4]=cnn_c2_4_lr0.003_wd0.01 [cnn_c3_6]=cnn_c3_6_lr0.003_wd0.01)
M=${1:?usage: $0 <dann_d8_s32|dann_d8_s64|cnn_c2_4|cnn_c3_6>}
[ -n "${CELL[$M]}" ] || { echo "refusing: $M is not one of the 4 E-10b board models"; exit 1; }
BDIR=$ROOT/results/T1m/e10/build/${CELL[$M]}_t1
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
