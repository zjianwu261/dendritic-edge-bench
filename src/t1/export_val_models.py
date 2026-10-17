"""Turn a validation-selected SHD checkpoint (shd_val.py) into deployable
ONNX models under results/T1/models/<alias>{_seq,_step,_seqloop}.onnx + <alias>.json.

SNN: full-sequence unrolled graph + step graph (shd.export) and the ONNX Loop graph
     (shd_loop_export.SNNLoop). GRU: ONNX GRU-op sequence graph + step graph (shd.export) and the
     TorchScript Loop graph (gru_loop_export.GRULoop).
Usage: python export_val_models.py <shd_val run name> <alias>
  e.g. python export_val_models.py lif_srnn_n64_s0 lif_srnn_n64_val
"""
import json
import shutil
import subprocess
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, save_meta  # noqa: E402
from shd import SRNN, GRUNet, export  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    run, alias = sys.argv[1], sys.argv[2]
    vm = json.load(open(f"{MODELS}/shd_val/{run}.json"))
    sd = torch.load(f"{MODELS}/shd_val/{run}.pt", map_location="cpu")
    if vm["model"] == "gru":
        h = vm["args"]["hidden"] or sd["gru.weight_hh_l0"].shape[1]
        m = GRUNet(h)
    else:
        h = 0
        m = SRNN(64, 8 if vm["model"] == "dhsrnn" else 0, seed=vm["args"]["seed"])
    m.load_state_dict(sd)
    meta = {k: v for k, v in vm.items() if k != "history"}
    meta.update({"name": alias, "group": "shd", "source_run": run, "T": 1000, "dt_ms": 1,
                 "selection_note": "epoch chosen on the 10% validation split; test_acc = test accuracy of that epoch"})
    shutil.copy(f"{MODELS}/shd_val/{run}.pt", f"{MODELS}/{alias}.pt")
    export(m, alias, meta)
    save_meta(alias, meta)
    py = sys.executable
    if vm["model"] == "gru":
        subprocess.run([py, f"{HERE}/gru_loop_export.py", f"{MODELS}/{alias}.pt", str(h), alias], check=True)
    else:
        subprocess.run([py, f"{HERE}/shd_loop_export.py", alias], check=True)


if __name__ == "__main__":
    main()
