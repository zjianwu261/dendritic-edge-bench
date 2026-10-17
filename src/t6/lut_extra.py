"""Orin #1 TRT FP16 latency of the validation-best G2 models of the
fair protocol (speaker split) that have no LUT entry yet:
    GRU h768 (lr 3e-3)                 -> ONNX GRU op, full sequence ("seq", as the GRU LUT entries)
    dssmU L2 n128 b4 spk (non-dendritic diagonal SSM, ungrouped projection) -> DHSSMExport, chunk K = 50
    LRU L2 H64 N128 (lr 1e-3)          -> LRUExport, chunk K = 50
(validation means over 3 seeds: LRU 87.1, GRU h768 85.4, dssmU 82.2; trained seed-1 weights are exported --
TRT latency does not depend on the weight values, the LUT used random weights).
Anchors re-timed in the same session (engines already on the Orin): dhssm_L2_n128_b4_cont__K50, gru_h256__seq.
Protocol = lut.py / t6_orin.sh (T1 sync e2e, builds default + nocublas, 3 fresh processes per engine, every
build / run under flock /tmp/dn_orin_bench.lock, clocks never touched).
  python lut_extra.py make     export ONNX (+ ORT check) -> results/T6/lut/onnx/g2x/, manifest
  python lut_extra.py push     rsync + start t6_orin.sh batch on the Orin (setsid)
  python lut_extra.py table    after lut.py pull: results/T6/lut/lut_g2x.csv
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, MODELS, T6, export_onnx  # noqa: E402
from dh_ssm import DHSSM, DHSSMExport, GRUNetD, LRUExport, LRUNet, N_IN, T_STEPS  # noqa: E402

ORIN = os.environ.get("ORIN_HOST", "orin")
RD = "dendritic-edge-bench/T6"
OD = f"{LUT}/onnx/g2x"
RAW = f"{T6}/raw/orin1"
ANCHORS = ["dhssm_L2_n128_b4_cont__K50", "gru_h256__seq"]


def models():
    g = f"{MODELS}/g2"
    gru = GRUNetD(768); gru.load_state_dict(torch.load(f"{g}/gru_h768_lr0.003_vs_t1.pt", map_location="cpu"))
    ds = DHSSM(128, 4, 2, "spk", grouped=False); ds.load_state_dict(torch.load(f"{g}/dssmU_L2_n128_b4_spk_vs_t1.pt", map_location="cpu"))
    lru = LRUNet(64, 128, 2, 0.1, K=50); lru.load_state_dict(torch.load(f"{g}/lru_L2_H64_N128_lr0.001_vs_t1.pt", map_location="cpu"))
    return [("gru_h768_lr0.003_vs_t1__seq", gru.eval()), ("dssmU_L2_n128_b4_spk_vs_t1__K50", DHSSMExport(ds.eval(), 50).eval()),
            ("lru_L2_H64_N128_lr0.001_vs_t1__K50", LRUExport(lru.eval(), 50).eval())], {"gru": gru, "dssmU": ds, "lru": lru}


def make():
    os.makedirs(OD, exist_ok=True)
    torch.manual_seed(0)
    x = torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.02)
    exp, train_m = models()
    lines, info = [], {}
    for lab, m in exp:
        path = f"{OD}/{lab}.onnx"
        info[lab] = export_onnx(m, (x,), path, ["x"], ["y"], check=True)
        print(lab, info[lab], flush=True)
        lines.append(f"~/{RD}/onnx/g2x/{lab}.onnx {lab} default,nocublas")
    with torch.no_grad():  # export graph == training graph (eval mode)
        info["export_vs_train_max_abs"] = {
            "dssmU": float((exp[1][1](x) - train_m["dssmU"](x)).abs().max()),
            "lru": float((exp[2][1](x) - train_m["lru"](x)).abs().max())}
    for a in ANCHORS:
        lines.append(f"~/{RD}/onnx/g2/{a}.onnx {a} default,nocublas")
    open(f"{OD}/manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(info, open(f"{OD}/export_info.json", "w"), indent=1)
    print(json.dumps(info["export_vs_train_max_abs"]))


def push():
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/onnx/g2x ~/{RD}/bin ~/{RD}/work"], check=True)
    subprocess.run(["rsync", "-a", f"{OD}/", f"{ORIN}:{RD}/onnx/g2x/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" onnx/g2x/manifest.txt > work/g2x_manifest.txt && "
           f"setsid nohup bash bin/t6_orin.sh batch work/g2x_manifest.txt $HOME/{RD}/work 3 "
           f"> work/g2x_manifest.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)


def table():
    rows = [json.loads(l) for l in open(f"{RAW}/results.jsonl")]
    labs = [l.split()[1] for l in open(f"{OD}/manifest.txt") if l.strip()]
    out = []
    for lab in labs:
        rec = {"label": lab}
        best = None
        for b in ("default", "nocublas"):
            rs = [r for r in rows if r["label"] == lab and r["build"] == b]
            if lab in ANCHORS:  # only the runs of this session (after the g2x batch started)
                rs = [r for r in rs if r.get("time", "") >= "2026-10-08"]
            if not rs:
                continue
            p50 = float(np.median([r["p50_ms"] for r in rs]))
            rec[f"p50_{b}"] = round(p50, 4); rec[f"nruns_{b}"] = len(rs)
            rec[f"range_{b}"] = f"{min(r['p50_ms'] for r in rs):.4f}-{max(r['p50_ms'] for r in rs):.4f}"
            rec[f"gpu_fmax_frac_{b}"] = round(float(np.median([r.get("gpu_freq_timed_frac_at_max", np.nan) for r in rs])), 2)
            if best is None or p50 < best[0]:
                best = (p50, b, rs)
        if best:
            rec.update({"p50_ms": round(best[0], 4), "build": best[1],
                        "p95_ms": round(float(np.median([r["p95_ms"] for r in best[2]])), 4),
                        "engine_file_MB": best[2][0].get("engine_file_MB"), "vm_hwm_MB": best[2][0].get("vm_hwm_MB")})
        out.append(rec)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(f"{LUT}/lut_g2x.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    {"make": make, "push": push, "table": table}[sys.argv[1]]()
