"""E-11: compute results/costmodel/features.csv (one row per timed graph per tier).

Columns: tier, graph, role (calib / dendritic / fused_rnn / excluded), calib (0/1), K, F, B, F_eff, B_nnz,
params_eff (+ source), 8051 terms, target (ms for GPU/CPU, board cycles for the 8051), target_ms, mhz,
source, file, sha256.
"""
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

OUT = f"{data.EXP}/costmodel/features.csv"
COLS = ["tier", "graph", "role", "calib", "K", "F", "B", "F_eff", "B_nnz", "params_eff", "params_eff_source",
        "F_fc", "F_conv", "F_gather", "F_prng", "N_out", "loop_trips", "fused_rnn", "target", "target_ms",
        "mhz", "source", "file", "sha256", "ops"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _onnx_job(path):
    import features
    f = features.graph_features(path)
    ops = ";".join(f"{k}:{v}" for k, v in sorted(f["ops"].items()))
    return path, dict(K=f["K"], F=f["F"], B=f["B"], F_eff=f["F_eff"], B_nnz=f["B_nnz"],
                      loop_trips="|".join(map(str, f["loop_trips"])), fused_rnn=int(f["fused_rnn"]), ops=ops,
                      sha256=sha(path))


def _mcu_job(path):
    import mcu_features
    return path, dict(mcu_features.mcu_features(path), sha256=sha(path))


def main():
    G = data.all_graphs()
    paths = sorted({v["onnx"] for t in ("orin1", "pi", "orin2") for v in G[t].values()})
    mh = sorted({v["model_h"] for v in G["mcu"].values()})
    res = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("CM_JOBS", "6"))) as ex:
        for p, r in ex.map(_onnx_job, paths):
            res[p] = r
            print("onnx", os.path.basename(p), r["K"], r["F"], r["B"], flush=True)
        for p, r in ex.map(_mcu_job, mh):
            res[p] = r
            print("mcu", p.split("/")[-2], r, flush=True)
    rows = []
    for tier in data.TIERS:
        for g, v in sorted(G[tier].items()):
            if tier == "mcu":
                f = res[v["model_h"]]
                dend = data.is_dendritic(g)
                role = "dendritic" if dend else ("calib" if g in data.MCU_CALIB else "excluded")
                rows.append(dict(tier=tier, graph=g, role=role, calib=int(role == "calib"), K="", F="", B="",
                                 F_eff=f["F_eff"], B_nnz="", params_eff=f["params_eff"],
                                 params_eff_source="model.h arrays (no index tables)",
                                 **{k: f[k] for k in ("F_fc", "F_conv", "F_gather", "F_prng", "N_out")},
                                 loop_trips="", fused_rnn=0, target=v["target"], target_ms=v["ms"],
                                 mhz=round(v["mhz"], 4), source=v["source"],
                                 file=os.path.relpath(v["model_h"], data.ROOT), sha256=f["sha256"], ops=f["kind"]))
                continue
            f = res[v["onnx"]]
            if v["target"] is None:
                role = "excluded"
            elif f["fused_rnn"]:
                role = "fused_rnn"
            elif data.is_dendritic(g):
                role = "dendritic"
            else:
                role = "calib"
            pe = data.params_eff_record(g)
            pes = "training record (T1/results.csv params or model json params_effective)"
            if pe is None:
                pe, pes = f["B_nnz"], "B_nnz (nonzero float constants of the graph)"
            rows.append(dict(tier=tier, graph=g, role=role, calib=int(role == "calib"), **{
                k: f[k] for k in ("K", "F", "B", "F_eff", "B_nnz", "loop_trips", "fused_rnn", "ops", "sha256")},
                params_eff=pe, params_eff_source=pes, F_fc="", F_conv="", F_gather="", F_prng="", N_out="",
                target=v["target"] if v["target"] is not None else "", target_ms=v["target"] if v["target"] else "",
                mhz="", source=v["source"], file=os.path.relpath(v["onnx"], data.ROOT)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
