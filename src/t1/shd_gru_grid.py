"""Small GRU grid on SHD, selected on the validation split (shd_val.py).

Grid: hidden {23 (= DH-SRNN parameter count), 64, 128, 256} x lr {1e-3, 3e-3}
      x regularisation {(dropout 0, wd 0), (dropout 0.3, AdamW wd 1e-2)}; 60 epochs, seed 0.
One sequential worker, micro-batches of 25 (4 per batch of 100, gradient accumulation) and a
0.1 memory fraction (fits next to other jobs on a shared 16 GB GPU):
  python shd_gru_grid.py grid
  python shd_gru_grid.py seeds <hidden> <lr> <dropout> <wd>   # re-run a chosen config, seeds 1, 2
Skips runs whose meta json already exists.
"""
import itertools
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUTM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results/T1/models/shd_val")
EPOCHS = 60


def run(h, lr, do, wd, seed=0):
    name = f"gru_h{h}_lr{lr:g}_do{do:g}_wd{wd:g}_s{seed}"
    if os.path.exists(f"{OUTM}/{name}.json"):
        return
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([PY, f"{HERE}/shd_val.py", "--model", "gru", "--hidden", str(h), "--lr", str(lr),
                    "--dropout", str(do), "--wd", str(wd), "--epochs", str(EPOCHS), "--seed", str(seed),
                    "--mem_frac", "0.1", "--micro", "4"], check=False, env=env)


def main():
    which = sys.argv[1]
    if which == "seeds":
        h, lr, do, wd = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        for s in (1, 2):
            run(h, lr, do, wd, s)
        return
    for (do, wd), lr, h in itertools.product([(0.0, 0.0), (0.3, 1e-2)], [3e-3, 1e-3], [23, 64, 128, 256]):
        run(h, lr, do, wd)


if __name__ == "__main__":
    main()
