#!/bin/bash
# T1m board measurement: flash one model onto the STC8A8K64D4.
# Flashing ERASES the program currently on the board, hence the explicit flag, e.g.:
#   bash step4_flash_and_measure.sh dann_d16_s64_t1 --yes-overwrite-board
# Procedure: the script starts stcgal, then power-cycle the board (STC ISP needs a cold start).
# -t 35000 asks stcgal to trim the IRC to 35.000 MHz; note the frequency stcgal reports and pass
# it to host_uart.py (--f_mhz). Firmware is built for F_OSC = 35 MHz, UART 115200 8N1.
set -e
TAG=${1:?usage: $0 <tag e.g. dann_d16_s64_t1> --yes-overwrite-board}
[ "$2" = "--yes-overwrite-board" ] || { echo "refusing: add --yes-overwrite-board"; exit 1; }
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
STCGAL=${STCGAL:-stcgal}
HEX=$ROOT/results/T1m/build/$TAG/uart_35M.hex
LOG=$ROOT/results/T1m/logs/stcgal_$TAG.log
[ -f "$HEX" ] || { echo "missing $HEX"; exit 1; }
mkdir -p "$(dirname "$LOG")"
"$STCGAL" -P auto -p "${PORT:-/dev/ttyUSB0}" -t 35000 "$HEX" 2>&1 | tee "$LOG"
echo ">>> read the trimmed frequency from $LOG, then run e.g.:"
echo "python $ROOT/src/mcu/host_uart.py --tag $TAG --n 100 --f_mhz 35.0"
