#!/usr/bin/env bash
# Workstation side of the Orin #1 energy measurements (T1 main table + T6).
#   deploy  : copy scripts + config lists to Orin:~/dendritic-edge-bench/energy/
#   compare : jtop vs tegrastats in the same 40 s window, pulls the result
#   start   : launch energy_chain.sh detached on the Orin (setsid nohup); log energy/chain.out
#   status  : tail of the chain log + count of good runs
#   pull    : rsync raw logs to results/T1/raw/orin1_energy/ and results/T6/raw/energy/
# Parse afterwards with:  python src/bench/energy_parse.py
set -euo pipefail
ROOT=${DEB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
J=${ORIN_HOST:-orin}
RD=dendritic-edge-bench/energy
case $1 in
  deploy)
    ssh $J "mkdir -p ~/$RD/bin ~/$RD/t1 ~/$RD/t6 ~/$RD/compare"
    scp -q $ROOT/src/bench/{power_logger.py,trt_energy_bench.py,trt_sync_bench.py,energy_one.sh,energy_chain.sh} $J:$RD/bin/
    scp -q $ROOT/results/T1/energy_configs.txt $J:$RD/t1_configs.txt
    scp -q $ROOT/results/T6/energy_configs.txt $J:$RD/t6_configs.txt ;;
  compare)
    ENG=${2:?engine path on the Orin}
    ssh $J "cd ~/$RD && bash bin/energy_chain.sh compare \$HOME/$RD/compare $ENG"
    mkdir -p $ROOT/results/T1/raw/orin1_energy/compare
    rsync -a $J:$RD/compare/ $ROOT/results/T1/raw/orin1_energy/compare/ ;;
  start)
    ssh $J "cd ~/$RD && setsid nohup bash bin/energy_chain.sh run \$HOME/$RD/t1_configs.txt \$HOME/$RD/t1 \$HOME/$RD/t6_configs.txt \$HOME/$RD/t6 >> chain.out 2>&1 < /dev/null & echo started" ;;
  status)
    ssh $J "cd ~/$RD && tail -n 15 chain.out; echo good runs: t1 \$(cat t1/done.txt 2>/dev/null | wc -l) t6 \$(cat t6/done.txt 2>/dev/null | wc -l); ls chain.done 2>/dev/null; pgrep -fa energy_chain.sh | head -3" ;;
  pull)
    mkdir -p $ROOT/results/T1/raw/orin1_energy $ROOT/results/T6/raw/energy
    rsync -a $J:$RD/t1/ $ROOT/results/T1/raw/orin1_energy/
    rsync -a $J:$RD/t6/ $ROOT/results/T6/raw/energy/
    rsync -a $J:$RD/compare/ $ROOT/results/T1/raw/orin1_energy/compare/
    rsync -a $J:$RD/chain.out $ROOT/results/T1/raw/orin1_energy/chain.out ;;
esac
