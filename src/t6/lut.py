"""T6 Orin #1 latency look-up table (LUT).

Latency on the Orin depends only on the graph (layer shapes, implementation, sparsity pattern),
not on the trained weight values, so the LUT is built from random-weight ONNX files with exactly
the deployed graph of every candidate configuration (2:4 masks applied where the config is
sparse). Trained models are then mapped to the LUT by their config key; a subset of trained
Pareto models is re-measured directly as a check.

  python lut.py make <task> [--only substr]   export random-weight ONNX + manifest (lut/onnx/<task>/)
  python lut.py push <task> [--manifest name] copy to Orin #1 and start the batch runner there
                                              (setsid nohup; every build/run holds the flock)
  python lut.py pull                          rsync results back (engines excluded) -> raw/orin1/
  python lut.py table <task>                  aggregate -> lut/lut_<task>.csv
Measurement protocol = T1: TensorRT 8.5 FP16, batch 1, sync e2e, warm-up >= 2 s,
500 timed iters, 3 fresh processes per engine; builds default / nocublas (and, for 2:4 models,
the same two with --sparsity=enable); the LUT latency of a (cfg, impl) = lowest p50 (median of
the runs) over its builds, the build is recorded.
G1 (sub-ms models, Orin not clock-locked): >= 10 independent runs per engine; two
columns: p50_ms = median over all runs (best build), p50_fmax_ms = median over the runs in which
>= 90 % of the GPU-frequency samples taken during timing were at the 15 W maximum (624.75 MHz)
(best build).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, T6, export_onnx  # noqa: E402

ORIN = os.environ.get("ORIN_HOST", "orin")
RD = "dendritic-edge-bench/T6"   # relative to ~ on the Orin
RAW = f"{T6}/raw/orin1"
DENSE_B, SPARSE_B = "default,nocublas", "default,nocublas,sp_default,sp_nocublas"


# ------------------------------------------------------------------ candidate configurations
def g1_configs():
    from dann_g import DANNG, Dense24MLP, set_24_masks
    out = []
    for h in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        out.append((f"mlp_h{h}", lambda h=h: Dense24MLP(h, h), ["gemm", "conv"], False))
    for m in (8, 16, 32, 64):
        for s in (64, 128, 256):
            for sg in (1, 4):
                if (m // sg) % 4:
                    continue
                out.append((f"dannG_p4_m{m}_s{s}_sg{sg}", lambda m=m, s=s, sg=sg: DANNG(4, m, s, sg), ["bmm"], False))
    for m in (16, 32, 64):
        def f(m=m):
            n = DANNG(4, m, 128, 4, True)
            set_24_masks(n)
            return n
        # pilot (2026-10-01): grouped layers are never eligible for sparse math -> dense builds only
        out.append((f"dannG_p4_m{m}_s128_sg4_sp24", f, ["bmm"], False))
    for D in (512, 1024, 2048, 4096):
        for s in (128, 256):
            def f(D=D, s=s):
                n = Dense24MLP(D, s, sp24=True)
                set_24_masks(n)
                return n
            imp = ["gemm", "conv"] if D >= 2048 else ["conv"]
            out.append((f"dann24_D{D}_s{s}", f, imp, True))
            # the same shape without sparsity = what the 2:4 speed-up is measured against
            out.append((f"dense_D{D}_s{s}", lambda D=D, s=s: Dense24MLP(D, s), imp, False))
    return out


def configs(task):
    if task == "g1":
        return g1_configs()
    if task == "g2":
        from dh_ssm import lut_configs
        return lut_configs()
    if task == "g3":
        from ad_cache import lut_configs
        return lut_configs()
    raise SystemExit(task)


def make(task, only=None, manifest="manifest.txt"):
    od = f"{LUT}/onnx/{task}"
    os.makedirs(od, exist_ok=True)
    lines = []
    for cfg, ctor, impls, sparse in configs(task):
        if only and not any(o in cfg for o in only.split(",")):
            continue
        torch.manual_seed(0)
        net = ctor().eval()
        for impl in impls:
            lab = f"{cfg}__{impl}"
            path = f"{od}/{lab}.onnx"
            if not os.path.exists(path):
                net.impl = impl
                inputs = net.example_inputs() if hasattr(net, "example_inputs") else (torch.rand(1, 784),)
                names = net.input_names() if hasattr(net, "input_names") else ["x"]
                info = export_onnx(net, inputs, path, names, ["y"], check=True)
                print(lab, info, flush=True)
            lines.append(f"~/{RD}/onnx/{task}/{lab}.onnx {lab} {SPARSE_B if sparse else DENSE_B}")
    with open(f"{od}/{manifest}", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} entries -> {od}/{manifest}")


def push(task, manifest="manifest.txt"):
    od = f"{LUT}/onnx/{task}"
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/onnx/{task} ~/{RD}/bin ~/{RD}/work"], check=True)
    subprocess.run(["rsync", "-a", f"{od}/", f"{ORIN}:{RD}/onnx/{task}/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/t6_orin.sh", f"{here}/orin/t6_sync_bench.py", f"{ORIN}:{RD}/bin/"], check=True)
    # the manifest uses ~ ; expand it on the Orin
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" onnx/{task}/{manifest} > work/{task}_{manifest} && "
           f"setsid nohup bash bin/t6_orin.sh batch work/{task}_{manifest} $HOME/{RD}/work 3 "
           f"> work/{task}_{manifest}.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)


def pull():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/work/", f"{RAW}/"], check=True)


def sparsity_info(log):
    """Parse the '(Sparsity)' INFO lines of a trtexec build log."""
    txt = open(log, errors="ignore").read()
    elig = re.findall(r"\(Sparsity\) Layers eligible for sparse math:(.*)", txt)
    picks = re.findall(r"\(Sparsity\) TRT inference plan pick(?:s|ed) sparse implementation for layers:(.*)", txt)
    return {"eligible": elig[-1].strip() if elig else "", "picked": picks[-1].strip() if picks else ""}


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# The LUT sessions are 2026-10-01..02; later runs of the same labels (g2x anchors 10-08, G1 anchors re-timed
# on 10-19) are other sessions and must not enter the LUT tables.
LUT_UNTIL = "2026-10-03"
G1_MLP_V2 = f"{LUT}/lut_g1v2.csv"


def table(task, until=LUT_UNTIL):
    """Grouping key = (label, onnx_sha256, build).  Labels are not unique across tasks (e.g. mlp_h256__gemm exists
    in g1 (784-in) and g3 (permuted-MNIST graph) with different ONNX); grouping by (label, build) alone would pool
    both graphs into lut_g1 and lut_g3.  The sha of lut/onnx/<task>/<label>.onnx selects the runs;
    if that file is missing and the label has >1 sha, every sha becomes its own row (impl suffix '@sha12')."""
    rows = [json.loads(l) for l in open(f"{RAW}/results.jsonl")] if os.path.exists(f"{RAW}/results.jsonl") else []
    rows = [r for r in rows if not until or r.get("time", "") < until]
    by = {}
    for r in rows:
        by.setdefault((r["label"], r["onnx_sha256"], r["build"]), []).append(r)
    shas = {}
    for l, s, _ in by:
        shas.setdefault(l, set()).add(s)
    cfgs = {c[0] for c in configs(task)}
    groups = []   # (label, sha, row-suffix)
    for lab in sorted(shas):
        cfg = lab.split("__")[0]
        if cfg not in cfgs:
            continue
        onnx = f"{LUT}/onnx/{task}/{lab}.onnx"
        want = _sha(onnx) if os.path.exists(onnx) else None
        if want:
            if want not in shas[lab]:
                print(f"WARNING {task} {lab}: no runs with the sha of {onnx} ({want[:12]}); have "
                      f"{sorted(s[:12] for s in shas[lab])} -> skipped", flush=True)
                continue
            if len(shas[lab]) > 1:
                print(f"WARNING {task} {lab}: {len(shas[lab])} shas under one label; using {want[:12]} "
                      f"(= {onnx}), dropping {sorted(s[:12] for s in shas[lab] if s != want)}", flush=True)
            groups.append((lab, want, ""))
        else:
            if len(shas[lab]) > 1:
                print(f"WARNING {task} {lab}: {len(shas[lab])} shas and no task ONNX -> separate rows", flush=True)
            for s in sorted(shas[lab]):
                groups.append((lab, s, f"@{s[:12]}" if len(shas[lab]) > 1 else ""))
    out = []
    for lab, sha, suf in groups:
        cfg, impl = lab.split("__")
        rec = {"cfg": cfg, "impl": impl + suf}
        rec_by = {(lab, b): v for (l, s, b), v in by.items() if l == lab and s == sha}
        best = None
        for b in ("default", "nocublas", "sp_default", "sp_nocublas"):
            rs = rec_by.get((lab, b), [])[-10:]   # G1 has >= 10 runs per engine; others 3
            if not rs:
                continue
            p50 = float(np.median([r["p50_ms"] for r in rs]))
            rec[f"p50_{b}"] = round(p50, 4)
            rec[f"range_{b}"] = f"{min(r['p50_ms'] for r in rs):.4f}-{max(r['p50_ms'] for r in rs):.4f}"
            rec[f"nruns_{b}"] = len(rs)
            hi = [r["p50_ms"] for r in rs if r.get("gpu_freq_timed_frac_at_max", 0) >= 0.9]
            rec[f"p50_fmax_{b}"] = round(float(np.median(hi)), 4) if hi else ""
            rec[f"fmax_frac_{b}"] = round(float(np.median([r.get("gpu_freq_timed_frac_at_max", np.nan) for r in rs])), 2)
            if b.startswith("sp"):
                logf = f"{RAW}/logs/{lab}_{b}_build.log"
                if os.path.exists(logf):
                    si = sparsity_info(logf)
                    rec[f"sparse_eligible_{b}"] = si["eligible"]
                    rec[f"sparse_picked_{b}"] = si["picked"]
            if best is None or p50 < best[0]:
                best = (p50, b, rs)
        if best is None:
            continue
        fm = [(rec[f"p50_fmax_{b}"], b) for b in ("default", "nocublas", "sp_default", "sp_nocublas")
              if rec.get(f"p50_fmax_{b}") not in (None, "")]
        rec["p50_fmax_ms"], rec["build_fmax"] = min(fm) if fm else ("", "")
        rec["n_runs_fmax"] = sum(1 for b in ("default", "nocublas", "sp_default", "sp_nocublas")
                                 for r in rec_by.get((lab, b), [])[-10:] if r.get("gpu_freq_timed_frac_at_max", 0) >= 0.9)
        rs = best[2]
        rec.update({"p50_ms": round(best[0], 4), "build": best[1],
                    "p95_ms": round(float(np.median([r["p95_ms"] for r in rs])), 4),
                    "infer_only_p50_ms": round(float(np.median([r["infer_only_p50_ms"] for r in rs])), 4),
                    "engine_file_MB": rs[0]["engine_file_MB"], "onnx_sha256": sha[:12]})
        out.append(rec)
    if task == "g1" and os.path.exists(G1_MLP_V2):
        # The lut/onnx/g1 mlp_h* graphs carry the shared-zero-bias Identity export artifact; their rows
        # are replaced by the re-exported graphs (distinct non-zero biases, LeakyReLU) of the locked g1v2 session
        # (g1v2_orin.py; lut/lut_g1v2.csv).
        v2 = [dict(r) for r in csv.DictReader(open(G1_MLP_V2))]
        for r in v2:
            for k in ("label", "n_runs_excluded_unlocked"):
                r.pop(k, None)
        out = [r for r in out if not r["cfg"].startswith("mlp_h")] + v2
        out.sort(key=lambda r: f"{r['cfg']}__{r['impl']}")
        print(f"g1: {len(v2)} mlp_h* rows from {G1_MLP_V2}", flush=True)
    keys = []
    for r in out:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(LUT, exist_ok=True)
    with open(f"{LUT}/lut_{task}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {LUT}/lut_{task}.csv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "push", "pull", "table"])
    ap.add_argument("task", nargs="?", default="g1")
    ap.add_argument("--only", default=None)
    ap.add_argument("--manifest", default="manifest.txt")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.task, a.only, a.manifest)
    elif a.cmd == "push":
        push(a.task, a.manifest)
    elif a.cmd == "pull":
        pull()
    else:
        table(a.task)


if __name__ == "__main__":
    main()
