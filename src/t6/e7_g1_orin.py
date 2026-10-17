"""Orin #1 latency of the G1 fair-protocol models, trained seed-1 weights,
one session, engines interleaved, results keyed by ONNX sha256 (not by label).

Models (configs = the per-size rows of results/T6/fair_summary.csv, i.e. selected on validation):
  dANN-G frontier  m8 / m16 / m32 / m64 (p4, s64, sg1)    impl bmm (the deployed grouped-GEMM graph)
  MLP              h128 / h256 / h512 / h1024 (same width steps as in the paper)   impls gemm and conv
  small CNN        c8_16 / c16_32 / c32_64 (same session) impl conv
  anchors          the random-weight LUT graphs mlp_h256__gemm and dannG_p4_m64_s64_sg1__bmm from
                   results/T6/lut/onnx/g1 (engines already on the Orin), to show the session-to-session drift
Labels are prefixed "g1f_" so that they cannot collide with G3 graphs.
Protocol = lut.py / t6_orin.sh: TRT FP16, batch 1, sync e2e, builds default + nocublas, NRUNS fresh processes per
engine with rounds interleaved across engines, every build / timed run under flock /tmp/dn_orin_bench.lock;
jetson_clocks / nvpmodel are never touched (only read by env_dump).
  python e7_g1_orin.py make     export ONNX (+ ORT check) -> results/T6/lut/onnx/g1f/, manifest
  python e7_g1_orin.py push     rsync + start t6_orin.sh batch on the Orin (setsid; NRUNS = 10)
  python e7_g1_orin.py pull     rsync results back (lut.pull)
  python e7_g1_orin.py table    -> results/T6/lut/lut_g1f.csv (grouped by onnx_sha256 + build, this session only)
  python e7_g1_orin.py auto     background: wait until the Orin answers, push, wait for "batch done" of this
                                manifest, pull, table
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, MODELS, T6, export_onnx  # noqa: E402
from dann_g import build  # noqa: E402
import lut as L  # noqa: E402

ORIN, RD, RAW = L.ORIN, L.RD, L.RAW
OD = f"{LUT}/onnx/g1f"
NRUNS = 10
ANCHORS = ["mlp_h256__gemm", "dannG_p4_m64_s64_sg1__bmm"]


def selected():
    rows = [r for r in csv.DictReader(open(f"{T6}/fair_summary.csv")) if r["group"] == "G1" and r["kind"] == "per_size"]
    return rows


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make():
    os.makedirs(OD, exist_ok=True)
    lines, info = [], {}
    x = torch.rand(1, 784, generator=torch.Generator().manual_seed(0))
    for r in selected():
        cfg = r["cfg"]
        meta = json.load(open(f"{MODELS}/g1/{cfg}_t1.json"))
        a = SimpleNamespace(**meta["args"])
        net, _ = build(a.arch, a)
        net.load_state_dict(torch.load(f"{MODELS}/g1/{cfg}_t1.pt", map_location="cpu"))
        net.eval()
        impls = {"dannG": ["bmm"], "mlp": ["gemm", "conv"], "cnn": ["conv"]}[r["family"]]
        for impl in impls:
            if hasattr(net, "impl"):
                net.impl = impl
            lab = f"g1f_{cfg}_t1__{impl}"
            path = f"{OD}/{lab}.onnx"
            if not os.path.exists(path):
                info[lab] = export_onnx(net, (x,), path, ["x"], ["y"], check=True)
            else:
                info[lab] = {"onnx": os.path.basename(path)}
            with torch.no_grad():  # the exported graph computes the trained model
                net_ref = net(x)
            info[lab].update(cfg=cfg, family=r["family"], impl=impl, test_acc_t1=meta["test_acc"], onnx_sha256=sha(path),
                             logit0=float(net_ref[0, 0]))
            print(lab, info[lab], flush=True)
            lines.append(f"~/{RD}/onnx/g1f/{lab}.onnx {lab} default,nocublas")
    for an in ANCHORS:
        p = f"{LUT}/onnx/g1/{an}.onnx"
        info[an] = {"anchor": True, "onnx_sha256": sha(p)}
        lines.append(f"~/{RD}/onnx/g1/{an}.onnx {an} default,nocublas")
    open(f"{OD}/manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(info, open(f"{OD}/export_info.json", "w"), indent=1)
    print(len(lines), "manifest lines")


def push():
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/onnx/g1f ~/{RD}/bin ~/{RD}/work"], check=True, timeout=60)
    subprocess.run(["rsync", "-a", f"{OD}/", f"{ORIN}:{RD}/onnx/g1f/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/t6_orin.sh", f"{here}/orin/t6_sync_bench.py", f"{ORIN}:{RD}/bin/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" onnx/g1f/manifest.txt > work/g1f_manifest.txt && "
           f"setsid nohup bash bin/t6_orin.sh batch work/g1f_manifest.txt $HOME/{RD}/work {NRUNS} "
           f"> work/g1f_manifest.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)


def table():
    info = json.load(open(f"{OD}/export_info.json"))
    labs = [l.split()[1] for l in open(f"{OD}/manifest.txt") if l.strip()]
    rows = [json.loads(l) for l in open(f"{RAW}/results.jsonl")]
    t0 = min((r.get("time", "") for r in rows if r["label"].startswith("g1f_")), default="")
    out = []
    for lab in labs:
        sh = info[lab]["onnx_sha256"]
        rec = {"label": lab, "family": info[lab].get("family", "anchor"), "cfg": info[lab].get("cfg", lab),
               "impl": info[lab].get("impl", lab.split("__")[1]), "test_acc_t1": info[lab].get("test_acc_t1", ""), "onnx_sha256": sh[:12]}
        best = None
        for b in ("default", "nocublas"):
            rs = [r for r in rows if r["label"] == lab and r["build"] == b and r.get("onnx_sha256") == sh
                  and (not t0 or r.get("time", "") >= t0)]
            if not rs:
                continue
            p = [r["p50_ms"] for r in rs]
            rec[f"p50_{b}"] = round(float(np.median(p)), 4)
            rec[f"range_{b}"] = f"{min(p):.4f}-{max(p):.4f}"
            rec[f"nruns_{b}"] = len(rs)
            rec[f"fmax_frac_{b}"] = round(float(np.median([r.get("gpu_freq_timed_frac_at_max", np.nan) for r in rs])), 2)
            if best is None or np.median(p) < best[0]:
                best = (float(np.median(p)), b)
        if best:
            rec["p50_ms"], rec["build"] = round(best[0], 4), best[1]
        out.append(rec)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(f"{LUT}/lut_g1f.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)


def auto(poll_s=600, max_h=96):
    import time
    t_end = time.time() + max_h * 3600
    def ssh(cmd, t=40):
        return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ORIN, cmd], capture_output=True, text=True, timeout=t)
    while True:
        try:
            if ssh("true").returncode == 0:
                break
        except subprocess.TimeoutExpired:
            pass
        print(time.strftime("%F %T"), "Orin #1 not reachable", flush=True)
        if time.time() > t_end:
            raise SystemExit("gave up")
        time.sleep(poll_s)
    started = ssh(f"grep -c 'batch start .*g1f_manifest' ~/{RD}/work/progress.txt || true").stdout.strip()
    if started in ("", "0"):
        push(); print(time.strftime("%F %T"), "pushed + started", flush=True)
    while True:
        try:
            r = ssh(f"grep -c 'batch done .*g1f_manifest' ~/{RD}/work/progress.txt || true").stdout.strip()
            if r not in ("", "0"):
                break
            print(time.strftime("%F %T"), "running:", ssh(f"tail -1 ~/{RD}/work/progress.txt").stdout.strip(), flush=True)
        except subprocess.TimeoutExpired:
            print(time.strftime("%F %T"), "ssh timeout", flush=True)
        time.sleep(poll_s)
    L.pull(); table(); print("AUTO DONE", flush=True)


if __name__ == "__main__":
    {"make": make, "push": push, "pull": L.pull, "table": table, "auto": auto}[sys.argv[1]]()
