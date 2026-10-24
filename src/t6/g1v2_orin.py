"""G1 MLP LUT graph fix: re-export of the random-weight MLP graphs of lut/onnx/g1 and a locked re-measurement.

The 16 lut/onnx/g1/mlp_h* graphs carry an export artifact: Dense24MLP zero-inits all biases, h1 == h2 makes l1.bias and
l2.bias identical tensors, torch.onnx de-duplicates them into Identity(l1.bias) -> l2.bias, and TRT turns that into an
extra Constant + Reformat + ElementWise bias add (only the MLP rows carry it).  This script re-exports them without the
artifact and re-measures them in one clock-locked Orin #1 session, together with
  * a 2x2 check {LeakyReLU(0.1), ReLU} x {shared zero bias (Identity), distinct non-zero bias}, h256 / h512, gemm;
  * the two anchors of e7_g1_orin.py (mlp_h256__gemm, dannG_p4_m64_s64_sg1__bmm of lut/onnx/g1, the 10-01 locked engines).
Nothing on disk is overwritten: new task dir lut/onnx/g1v2, labels g1v2_* / g1x2_* (unique build-log names), Orin
workdir ~/RD/work_g1v2 (pulled to raw/orin1_g1v2).
Graph = Dense24MLP(h, h) (same xavier weights, torch.manual_seed(0), as lut.py make), activation LeakyReLU(0.1) (the G1
arm); biases of l1 / l2 / out replaced by U(-1/sqrt(fan_in), 1/sqrt(fan_in)) from Generator(seed 1), one draw per layer
(= PyTorch default bias init; distinct, non-zero, like the g3 LUT / trained g1f graphs).
  python g1v2_orin.py make      export + onnx checks (no Identity, no initializer shared by two nodes, op list) + manifest
  python g1v2_orin.py run       check lock (min=cur=max=624.75 MHz, else stop), copy anchor engines, push, start with
                                REQUIRE_LOCK_HZ, wait, pull, table, anchors   (jetson_clocks / nvpmodel never run)
  python g1v2_orin.py table | anchors | merge
merge: lut/lut_g1v2.csv (cfg mlp_h*, lut.table columns) is what lut.py table g1 now substitutes for the mlp_h* rows.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, T6, export_onnx  # noqa: E402
from dann_g import LEAK, Dense24MLP  # noqa: E402
import lut as L  # noqa: E402
import e7_g1_orin as E7  # noqa: E402

ORIN, RD = L.ORIN, L.RD
OD = f"{LUT}/onnx/g1v2"
WD = "work_g1v2"
RAWV = f"{T6}/raw/orin1_g1v2"
NRUNS = 10
LOCK_HZ = E7.LOCK_HZ
ANCHORS = E7.ANCHORS
G = "/sys/devices/gpu.0/devfreq/17000000.ga10b"


class MLPAct(Dense24MLP):
    """Dense24MLP with selectable activation (lrelu = LeakyReLU(LEAK), relu)."""

    def __init__(self, h, act="lrelu"):
        super().__init__(h, h)
        self.act = act

    def a(self, x):
        return F.leaky_relu(x, LEAK) if self.act == "lrelu" else F.relu(x)

    def forward(self, x):
        if self.impl == "conv":
            h = x.reshape(1, 784, 1, 1)
            h = self.a(F.conv2d(h, self.l1.weight[:, :, None, None], self.l1.bias))
            h = self.a(F.conv2d(h, self.l2.weight[:, :, None, None], self.l2.bias))
            return self.out(h.reshape(1, -1))
        return self.out(self.a(self.l2(self.a(self.l1(x)))))


def net(h, act="lrelu", bias="dsb"):
    torch.manual_seed(0)                    # same weights as lut.py make (Dense24MLP xavier, seed 0)
    n = MLPAct(h, act)
    if bias == "dsb":
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for l in (n.l1, n.l2, n.out):
                b = 1 / np.sqrt(l.in_features)
                l.bias.copy_((torch.rand(l.bias.shape, generator=g) * 2 - 1) * b)
    return n.eval()


def graphs():
    out = []   # (label, ctor, impl)
    for h in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        for impl in ("gemm", "conv"):
            out.append((f"g1v2_mlp_h{h}__{impl}", lambda h=h: net(h), impl))
    for h in (256, 512):
        for act in ("lrelu", "relu"):
            for bias in ("shz", "dsb"):
                out.append((f"g1x2_{act}_{bias}_h{h}__gemm", lambda h=h, act=act, bias=bias: net(h, act, bias), "gemm"))
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def onnx_check(path):
    import onnx
    m = onnx.load(path)
    ops = [n.op_type for n in m.graph.node]
    inits = {i.name for i in m.graph.initializer}
    use = {}
    for n in m.graph.node:
        for i in n.input:
            if i in inits:
                use[i] = use.get(i, 0) + 1
    shared = sorted(k for k, v in use.items() if v > 1)
    return {"ops": ops, "identity": "Identity" in ops, "shared_init": shared,
            "bias_init": sorted(k for k in inits if k.endswith("bias"))}


OPS = {("lrelu", "gemm"): ["Gemm", "LeakyRelu", "Gemm", "LeakyRelu", "Gemm"],
       ("lrelu", "conv"): ["Constant", "Reshape", "Conv", "LeakyRelu", "Conv", "LeakyRelu", "Constant", "Reshape", "Gemm"],
       ("relu", "gemm"): ["Gemm", "Relu", "Gemm", "Relu", "Gemm"]}


def make():
    os.makedirs(OD, exist_ok=True)
    info, lines = {}, []
    x = torch.rand(1, 784, generator=torch.Generator().manual_seed(0))
    for lab, ctor, impl in graphs():
        p = f"{OD}/{lab}.onnx"
        n = ctor()
        n.impl = impl
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")
        d = export_onnx(n, (x,), p, ["x"], ["y"], check=True)
        c = onnx_check(p)
        act = "relu" if "_relu_" in lab else "lrelu"
        shz = "_shz_" in lab
        exp_ops = (["Identity"] if shz else []) + OPS[(act, impl)]
        ok = c["ops"] == exp_ops and (c["identity"] == shz) and (bool(c["shared_init"]) == shz)
        if not shz:
            ok = ok and c["bias_init"] == ["l1.bias", "l2.bias", "out.bias"]
        d.update(c, onnx_sha256=sha(p), act=act, bias="shz" if shz else "dsb", impl=impl, check_ok=ok,
                 bias_absmax=[float(t.detach().abs().max()) for t in (n.l1.bias, n.l2.bias, n.out.bias)])
        info[lab] = d
        print(lab, "ok" if ok else "CHECK FAILED", c["ops"], c["shared_init"], d["max_abs_err_vs_torch"], flush=True)
        if not ok:
            raise SystemExit(f"onnx check failed for {lab}")
        lines.append(f"~/{RD}/onnx/g1v2/{lab}.onnx {lab} default,nocublas")
    for an in ANCHORS:
        p = f"{LUT}/onnx/g1/{an}.onnx"
        info[an] = {"anchor": True, "onnx_sha256": sha(p), **onnx_check(p)}
        lines.append(f"~/{RD}/onnx/g1/{an}.onnx {an} default,nocublas")
    # old g1 graphs for reference (Identity + shared-zero bias)
    old = {}
    for lab, _, impl in graphs():
        if lab.startswith("g1v2_"):
            q = f"{LUT}/onnx/g1/{lab[5:]}.onnx"
            old[lab[5:]] = {"onnx_sha256": sha(q), **onnx_check(q)}
    json.dump({"graphs": info, "old_g1": old}, open(f"{OD}/export_info.json", "w"), indent=1)
    open(f"{OD}/manifest.txt", "w").write("\n".join(lines) + "\n")
    for k, v in info.items():   # same-sha notes
        for k2, v2 in old.items():
            if v["onnx_sha256"] == v2["onnx_sha256"]:
                print(f"note: {k} has the same sha as the old g1 graph {k2}")
    print(len(lines), "manifest lines ->", f"{OD}/manifest.txt")


def ssh(cmd, t=60):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ORIN, cmd], capture_output=True, text=True, timeout=t)


def lock_state():
    r = ssh(f"cat {G}/min_freq {G}/cur_freq {G}/max_freq")
    try:
        return tuple(int(v) for v in r.stdout.split()[:3])
    except ValueError:
        return None


def run(poll_s=300, run_max_h=6):
    T = lambda: time.strftime("%F %T")
    st = lock_state()
    print(T(), "GPU devfreq min/cur/max", st, flush=True)
    if st != (LOCK_HZ,) * 3:
        raise SystemExit(f"{T()} GPU NOT LOCKED {st}: stop (jetson_clocks / nvpmodel are not run by this script)")
    if ssh(f"test -e ~/{RD}/{WD}/progress.txt && echo yes").stdout.strip() == "yes":
        raise SystemExit(f"~/{RD}/{WD} already exists on the Orin; refusing to restart")
    info = json.load(open(f"{OD}/export_info.json"))["graphs"]
    cp = " && ".join(f"cp -n work/engines/{an}_{b}_{info[an]['onnx_sha256'][:12]}.engine {WD}/engines/"
                     for an in ANCHORS for b in ("default", "nocublas"))
    r = subprocess.run(["ssh", ORIN, f"cd ~/{RD} && mkdir -p {WD}/engines onnx/g1v2 && {cp} && ls {WD}/engines"],
                       capture_output=True, text=True, timeout=60)
    print(r.stdout, r.stderr, flush=True)
    if r.returncode:
        raise SystemExit("anchor engine copy failed")
    subprocess.run(["rsync", "-a", f"{OD}/", f"{ORIN}:{RD}/onnx/g1v2/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/t6_orin.sh", f"{here}/orin/t6_sync_bench.py", f"{ORIN}:{RD}/bin/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" onnx/g1v2/manifest.txt > {WD}/g1v2_manifest.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/t6_orin.sh batch {WD}/g1v2_manifest.txt $HOME/{RD}/{WD} {NRUNS} "
           f"> {WD}/g1v2_manifest.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)
    print(T(), "started (REQUIRE_LOCK_HZ set)", flush=True)
    t_end, lost = time.time() + run_max_h * 3600, 0
    while True:
        time.sleep(poll_s)
        try:
            done = ssh(f"grep -c 'batch done' ~/{RD}/{WD}/progress.txt || true").stdout.strip()
            st = lock_state()
            if st != (LOCK_HZ,) * 3:
                lost += 1
                print(T(), "WARNING LOCK LOST", st, flush=True)
            if done not in ("", "0"):
                break
            print(T(), "running:", ssh(f"tail -1 ~/{RD}/{WD}/progress.txt").stdout.strip(), "| lock", st, flush=True)
        except subprocess.TimeoutExpired:
            print(T(), "ssh timeout", flush=True)
        if time.time() > t_end:
            raise SystemExit("batch not done in time")
    pull()
    print(T(), "pulled; lock-lost polls", lost, flush=True)
    table()
    anchors()
    print(T(), "RUN DONE", flush=True)


def pull():
    os.makedirs(RAWV, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/{WD}/", f"{RAWV}/"], check=True)


def session_rows():
    p = f"{RAWV}/progress.txt"
    start = max(l.split()[0][:19] for l in open(p) if "batch start" in l)
    rows = [json.loads(l) for l in open(f"{RAWV}/results.jsonl")]
    return [r for r in rows if r.get("time", "") >= start], start


def table():
    """Every label of the manifest: lut.table-style record from lock-checked runs of this session (sha-keyed).
    -> lut/lut_g1v2_session.csv (all labels, with label column), lut/lut_g1v2.csv (g1v2 mlp rows as cfg mlp_h*)."""
    info = json.load(open(f"{OD}/export_info.json"))["graphs"]
    labs = [l.split()[1] for l in open(f"{OD}/manifest.txt") if l.strip()]
    rows, start = session_rows()
    out = []
    for lab in labs:
        sh = info[lab]["onnx_sha256"]
        cfg, impl = lab.split("__")
        rec = {"cfg": cfg[5:] if cfg.startswith("g1v2_") else cfg, "impl": impl}
        best, nall, nlock = None, 0, 0
        for b in ("default", "nocublas"):
            rs0 = [r for r in rows if r["label"] == lab and r["build"] == b and r["onnx_sha256"] == sh]
            rs = [r for r in rs0 if E7.run_locked(r, RAWV)]
            nall += len(rs0); nlock += len(rs)
            if not rs:
                continue
            p = [r["p50_ms"] for r in rs]
            p50 = float(np.median(p))
            rec[f"p50_{b}"] = round(p50, 4)
            rec[f"range_{b}"] = f"{min(p):.4f}-{max(p):.4f}"
            rec[f"nruns_{b}"] = len(rs)
            rec[f"p50_fmax_{b}"] = round(p50, 4)   # every lock-checked run is 100 % at 624.75 MHz
            rec[f"fmax_frac_{b}"] = round(float(np.median([r["gpu_freq_timed_frac_at_max"] for r in rs])), 2)
            if best is None or p50 < best[0]:
                best = (p50, b, rs)
        if best is None:
            print("NO LOCKED RUNS", lab, flush=True)
            continue
        rs = best[2]
        rec.update({"p50_fmax_ms": round(best[0], 4), "build_fmax": best[1], "n_runs_fmax": nlock,
                    "p50_ms": round(best[0], 4), "build": best[1],
                    "p95_ms": round(float(np.median([r["p95_ms"] for r in rs])), 4),
                    "infer_only_p50_ms": round(float(np.median([r["infer_only_p50_ms"] for r in rs])), 4),
                    "engine_file_MB": rs[0]["engine_file_MB"], "onnx_sha256": sh[:12],
                    "label": lab, "n_runs_excluded_unlocked": nall - nlock,
                    "clock": f"locked_{LOCK_HZ / 1e6:g}MHz_checked", "session": f"g1v2_{start}"})
        out.append(rec)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    for path, sub in ((f"{LUT}/lut_g1v2_session.csv", out), (f"{LUT}/lut_g1v2.csv", [r for r in out if r["label"].startswith("g1v2_")])):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(sub)
        print(len(sub), "rows ->", path, flush=True)
    for r in out:
        print(r["label"], r.get("p50_default"), r.get("p50_nocublas"), r["p50_ms"], r["build"], r.get("range_" + r["build"]),
              "excl", r["n_runs_excluded_unlocked"], flush=True)
    return out


def anchors():
    s = {r["label"]: r for r in csv.DictReader(open(f"{LUT}/lut_g1v2_session.csv"))}
    e7 = {(r["anchor"], r["build"]): r for r in csv.DictReader(open(f"{LUT}/e7_locked_anchors.csv"))}
    res = []
    for an in ANCHORS:
        for b in ("nocublas", "default"):
            v = float(s[an][f"p50_{b}"]) if an in s and s[an].get(f"p50_{b}") else None
            e = e7[(an, b)]
            d = {"anchor": an, "build": b, "g1v2_p50": v, "g1v2_range": s.get(an, {}).get(f"range_{b}", ""),
                 "e7_p50": float(e["new_p50"]), "e7_range": e["new_range"],
                 "ref_1001_median": float(e["ref_locked_median"]), "ref_1001_range": e["ref_locked_range"]}
            if v:
                d["ratio_vs_e7"] = round(v / d["e7_p50"], 3)
                d["ratio_vs_1001"] = round(v / d["ref_1001_median"], 3)
                lo, hi = (float(x) for x in e["ref_locked_range"].split("-"))
                d["within_1001_range"] = lo <= v <= hi
            res.append(d)
            print(d, flush=True)
    with open(f"{LUT}/g1v2_anchors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    return res


if __name__ == "__main__":
    {"make": make, "run": run, "pull": pull, "table": table, "anchors": anchors}[sys.argv[1]](*sys.argv[2:])
