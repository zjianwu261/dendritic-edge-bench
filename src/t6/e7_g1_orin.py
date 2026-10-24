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
  python e7_g1_orin.py auto     obsolete (first session, did not check the clock)
A first session of these graphs (2026-10-19 morning) ran with the GPU clock not locked (DVFS 306-510 MHz) and is not
used.  The locked session uses its own Orin workdir ~/RD/work_e7l (pulled to raw/orin1_e7locked), so the two sessions
can never be pooled.
  python e7_g1_orin.py auto_locked   poll Orin #1 every 5 min (read-only sysfs GPU devfreq min/cur/max, up to 96 h);
                                when min = cur = max = 624.75 MHz: copy the anchor engines of the locked G1 session into
                                work_e7l, push, start the batch with REQUIRE_LOCK_HZ (every timed run re-checks the lock
                                and is skipped if not locked), wait, pull, table (lock-checked), anchor check, fair_summary,
                                comparisons, summary printout.  jetson_clocks / nvpmodel are never run.
  python e7_g1_orin.py table [cutoff]   locked table -> lut_g1f.csv; only runs with time >= cutoff (default = the
                                latest "batch start" of this manifest in work_e7l/progress.txt) and lock-checked
  python e7_g1_orin.py table_unlocked   the unlocked first session -> lut_g1f_unlocked.csv (time window 10-19 09:00-11:00;
                                raw data not included in this release)
  python e7_g1_orin.py anchors | compare
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
LOCK_HZ = 624750000
WDL = "work_e7l"                        # Orin workdir of the locked re-run (relative to ~/RD)
RAWL = f"{T6}/raw/orin1_e7locked"
UNLOCKED_WIN = ("2026-10-19T09:00", "2026-10-19T11:00")
# lut_g1.csv nocublas p50 (locked); mlp_h256__gemm = the g1 graph alone (sha 2cb85cdb41c1)
LUT_G1 = {"mlp_h256__gemm": 0.136, "dannG_p4_m64_s64_sg1__bmm": 0.1229}


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


def gpu_states(envf):
    """(cur, min, max) GPU devfreq of every env_dump in a run's env file (before and after the run)."""
    if not os.path.exists(envf):
        return []
    L_ = open(envf).read().splitlines()
    out = []
    for k, l in enumerate(L_):
        if l.startswith("## gpu devfreq governor/cur/min/max"):
            try:
                out.append(tuple(int(x) for x in L_[k + 2:k + 5]))
            except ValueError:
                out.append(None)
    return out


def run_locked(r, src):
    base = r["engine"][:-len(".engine")]
    st = gpu_states(f"{src}/env/{base}_r{r['run']}.txt")
    env_ok = len(st) >= 2 and all(x == (LOCK_HZ,) * 3 for x in st)
    return env_ok and r.get("gpu_freq_timed_frac_at_max") == 1.0 and r.get("gpu_freq_hz_timed_min_med_max") == [LOCK_HZ] * 3


def session_start(src=RAWL):
    p = f"{src}/progress.txt"
    st = [l.split()[0][:19] for l in open(p) if "batch start" in l and "g1f_manifest" in l] if os.path.exists(p) else []
    return max(st) if st else ""


def table(cutoff=None, src=RAWL, out_csv=f"{LUT}/lut_g1f.csv", t_to="", require_lock=True):
    """cutoff: runs with time >= cutoff only (default: start of the locked re-run). Lock-checked runs only."""
    if cutoff is None:
        cutoff = session_start(src)
    if not cutoff:
        raise SystemExit("no session start found (locked re-run not started?)")
    info = json.load(open(f"{OD}/export_info.json"))
    labs = [l.split()[1] for l in open(f"{OD}/manifest.txt") if l.strip()]
    rows = [json.loads(l) for l in open(f"{src}/results.jsonl")]
    print("table: src", src, "cutoff", cutoff, "t_to", t_to or "-", "require_lock", require_lock, flush=True)
    out = []
    for lab in labs:
        sh = info[lab]["onnx_sha256"]
        rec = {"label": lab, "family": info[lab].get("family", "anchor"), "cfg": info[lab].get("cfg", lab),
               "impl": info[lab].get("impl", lab.split("__")[1]), "test_acc_t1": info[lab].get("test_acc_t1", ""), "onnx_sha256": sh[:12]}
        best = None
        for b in ("default", "nocublas"):
            rs = [r for r in rows if r["label"] == lab and r["build"] == b and r.get("onnx_sha256") == sh
                  and r.get("time", "") >= cutoff and (not t_to or r.get("time", "") < t_to)]
            n_all = len(rs)
            if require_lock:
                rs = [r for r in rs if run_locked(r, src)]
            if not rs:
                continue
            p = [r["p50_ms"] for r in rs]
            rec[f"p50_{b}"] = round(float(np.median(p)), 4)
            rec[f"range_{b}"] = f"{min(p):.4f}-{max(p):.4f}"
            rec[f"nruns_{b}"] = len(rs)
            rec[f"n_unlocked_excluded_{b}"] = n_all - len(rs)
            rec[f"fmax_frac_{b}"] = round(float(np.median([r.get("gpu_freq_timed_frac_at_max", np.nan) for r in rs])), 2)
            if best is None or np.median(p) < best[0]:
                best = (float(np.median(p)), b)
        if best:
            rec["p50_ms"], rec["build"] = round(best[0], 4), best[1]
        rec["clock"] = f"locked_{LOCK_HZ / 1e6:g}MHz_checked" if require_lock else "UNLOCKED_DVFS_2026-10-19"
        rec["session_from"] = cutoff
        out.append(rec)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)
    print("->", out_csv, flush=True)
    return out


def table_unlocked():
    return table(cutoff=UNLOCKED_WIN[0], t_to=UNLOCKED_WIN[1], src=RAW, out_csv=f"{LUT}/lut_g1f_unlocked.csv", require_lock=False)


def anchors():
    """Locked re-run anchors vs the locked G1 session (same ONNX sha, runs >= 0.9 of the time at 624.75 MHz, before 10-19)."""
    info = json.load(open(f"{OD}/export_info.json"))
    old = [json.loads(l) for l in open(f"{RAW}/results.jsonl")]
    new = {r["label"]: r for r in csv.DictReader(open(f"{LUT}/lut_g1f.csv"))}
    res = []
    for an in ANCHORS:
        sh = info[an]["onnx_sha256"]
        for b in ("nocublas", "default"):
            ref = [r["p50_ms"] for r in old if r["label"] == an and r["build"] == b and r.get("onnx_sha256") == sh
                   and r["time"] < "2026-10-19" and r.get("gpu_freq_timed_frac_at_max", 0) >= 0.9]
            v = new.get(an, {}).get(f"p50_{b}", "")
            d = {"anchor": an, "build": b, "new_p50": v, "new_range": new.get(an, {}).get(f"range_{b}", ""),
                 "ref_locked_median": round(float(np.median(ref)), 4) if ref else "", "ref_locked_runs": len(ref),
                 "ref_locked_range": f"{min(ref):.4f}-{max(ref):.4f}" if ref else "",
                 "lut_g1_p50": LUT_G1[an] if b == "nocublas" else ""}
            if v != "" and ref:
                v = float(v)
                d["ratio_vs_ref_median"] = round(v / float(np.median(ref)), 3)
                if b == "nocublas":
                    d["ratio_vs_lut_g1"] = round(v / LUT_G1[an], 3)
                d["within_ref_range"] = bool(min(ref) <= v <= max(ref))
            res.append(d)
            print(d, flush=True)
    keys = []
    for r in res:
        keys += [k for k in r if k not in keys]
    with open(f"{LUT}/e7_locked_anchors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(res)
    return res


def compare():
    """Descriptive comparisons (locked session): per cfg best (impl, build); dANN-G vs MLP / CNN latency ratios and acc."""
    rows = [r for r in csv.DictReader(open(f"{LUT}/lut_g1f.csv")) if r["family"] != "anchor" and r.get("p50_ms")]
    best = {}
    for r in rows:
        if r["cfg"] not in best or float(r["p50_ms"]) < float(best[r["cfg"]]["p50_ms"]):
            best[r["cfg"]] = r
    fam = lambda f: sorted([r for r in best.values() if r["family"] == f], key=lambda r: float(r["p50_ms"]))
    out = []
    for d in fam("dannG"):
        for o in fam("mlp") + fam("cnn"):
            b = o["build"]
            rd, ro = [float(x) for x in d[f"range_{d['build']}"].split("-")], [float(x) for x in o[f"range_{b}"].split("-")]
            out.append({"dannG": d["cfg"], "other": o["cfg"], "other_family": o["family"],
                        "dannG_p50": float(d["p50_ms"]), "other_p50": float(o["p50_ms"]),
                        "ratio_dannG_over_other": round(float(d["p50_ms"]) / float(o["p50_ms"]), 3),
                        "ranges_overlap": not (rd[1] < ro[0] or ro[1] < rd[0]),
                        "dannG_acc_t1": d["test_acc_t1"], "other_acc_t1": o["test_acc_t1"],
                        "dannG_impl_build": f"{d['impl']}/{d['build']}", "other_impl_build": f"{o['impl']}/{b}"})
    keys = list(out[0]) if out else []
    with open(f"{LUT}/e7_locked_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    lines = []
    for f in ("dannG", "mlp", "cnn"):
        lines.append(f"  {f}: " + "; ".join(f"{r['cfg'].split('_lr')[0]} {float(r['p50_ms']):.4f} ms [{r['range_' + r['build']]}] "
                                            f"{r['impl']}/{r['build']} acc_t1 {r['test_acc_t1']}" for r in fam(f)))
    print("\n".join(lines), flush=True)
    return out, lines


def progress_entry(anc, lines, note):
    """Print a short summary of the locked session."""
    import time
    s = [f"\n## {time.strftime('%Y-%m-%d %H:%M')} e7_g1_orin.py auto_locked: locked G1 session",
         f"- {note}",
         f"- Table `results/T6/lut/lut_g1f.csv` (locked, lock-checked runs only); raw `results/T6/raw/orin1_e7locked/`; "
         f"comparisons `results/T6/lut/e7_locked_compare.csv`; anchors `results/T6/lut/e7_locked_anchors.csv`.",
         "- Anchors (new p50 vs same-sha locked G1 runs median [range]; vs lut_g1):"]
    for a in anc:
        s.append(f"  - {a['anchor']} {a['build']}: {a['new_p50']} [{a['new_range']}] vs {a['ref_locked_median']} "
                 f"[{a['ref_locked_range']}, n={a['ref_locked_runs']}], ratio {a.get('ratio_vs_ref_median', '')}, "
                 f"vs lut_g1 {a.get('ratio_vs_lut_g1', '-')}, within range {a.get('within_ref_range', '')}")
    s.append("- p50 by family (best impl/build per cfg):")
    s += lines
    s.append("- Pairwise dANN-G / MLP / CNN ratios: e7_locked_compare.csv.")
    print("\n".join(s), flush=True)


def auto(poll_s=600, max_h=96):
    """OBSOLETE (first session; does not check the clock). Use auto_locked."""
    raise SystemExit("auto is obsolete: use auto_locked")
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


def auto_locked(poll_s=300, max_h=96, run_max_h=12):
    import shutil
    import time
    T = lambda: time.strftime("%F %T")
    def ssh(cmd, t=40):
        return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ORIN, cmd], capture_output=True, text=True, timeout=t)
    G = "/sys/devices/gpu.0/devfreq/17000000.ga10b"
    def lock_state():
        try:
            r = ssh(f"cat {G}/min_freq {G}/cur_freq {G}/max_freq {G}/governor")
            v = r.stdout.split()
            return (tuple(int(x) for x in v[:3]), v[3] if len(v) > 3 else "") if r.returncode == 0 and len(v) >= 3 else (None, r.stderr.strip()[:80])
        except (subprocess.TimeoutExpired, ValueError) as e:
            return None, str(e)[:80]
    t_end = time.time() + max_h * 3600
    while True:
        st, gov = lock_state()
        print(T(), "GPU devfreq min/cur/max", st, gov, flush=True)
        if st == (LOCK_HZ,) * 3:
            time.sleep(20)
            st2, _ = lock_state()   # confirm (cur can jump if only min was raised)
            if st2 == (LOCK_HZ,) * 3:
                break
        if time.time() > t_end:
            raise SystemExit(f"{T()} gave up: GPU clock not locked within {max_h} h")
        time.sleep(poll_s)
    print(T(), "LOCKED at 624.75 MHz -> push", flush=True)
    # anchors: the exact engines of the locked G1 session; g1f engines are rebuilt under the locked clock
    info = json.load(open(f"{OD}/export_info.json"))
    cp = " && ".join(f"cp -n work/engines/{an}_{b}_{info[an]['onnx_sha256'][:12]}.engine {WDL}/engines/"
                     for an in ANCHORS for b in ("default", "nocublas"))
    subprocess.run(["ssh", ORIN, f"cd ~/{RD} && mkdir -p {WDL}/engines && {cp} && ls {WDL}/engines"], check=True, timeout=60)
    subprocess.run(["rsync", "-a", f"{OD}/", f"{ORIN}:{RD}/onnx/g1f/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/t6_orin.sh", f"{here}/orin/t6_sync_bench.py", f"{ORIN}:{RD}/bin/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" onnx/g1f/manifest.txt > {WDL}/g1f_manifest.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/t6_orin.sh batch {WDL}/g1f_manifest.txt $HOME/{RD}/{WDL} {NRUNS} "
           f"> {WDL}/g1f_manifest.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)
    print(T(), "pushed + started (REQUIRE_LOCK_HZ set)", flush=True)
    t_run = time.time() + run_max_h * 3600
    lost = 0
    while True:
        time.sleep(poll_s)
        try:
            r = ssh(f"grep -c 'batch done .*g1f_manifest' ~/{RD}/{WDL}/progress.txt || true").stdout.strip()
            st, _ = lock_state()
            if st is not None and st != (LOCK_HZ,) * 3:
                lost += 1
                print(T(), "WARNING LOCK LOST", st, "(runs are skipped on the Orin and excluded in table)", flush=True)
            if r not in ("", "0"):
                break
            print(T(), "running:", ssh(f"tail -1 ~/{RD}/{WDL}/progress.txt").stdout.strip(), "| lock", st, flush=True)
        except subprocess.TimeoutExpired:
            print(T(), "ssh timeout", flush=True)
        if time.time() > t_run:
            raise SystemExit(f"{T()} batch not done within {run_max_h} h")
    os.makedirs(RAWL, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/{WDL}/", f"{RAWL}/"], check=True)
    nskip = sum(1 for _ in open(f"{RAWL}/unlocked.txt")) if os.path.exists(f"{RAWL}/unlocked.txt") else 0
    print(T(), "pulled; unlocked-skipped runs:", nskip, "lock-lost polls:", lost, flush=True)
    table()
    anc = anchors()
    import fair_summary
    shutil.copy(f"{T6}/fair_summary.csv", f"{T6}/fair_summary.csv.bak_e7l")
    fair_summary.main()
    _, lines = compare()
    progress_entry(anc, lines, f"Watcher saw min=cur=max=624.75 MHz, pushed and ran NRUNS={NRUNS}; unlocked-skipped runs {nskip}, "
                               f"lock-lost polls {lost}; fair_summary.csv re-generated (backup fair_summary.csv.bak_e7l).")
    print(T(), "AUTO_LOCKED DONE", flush=True)


if __name__ == "__main__":
    fn = {"make": make, "push": push, "pull": L.pull, "table": table, "table_unlocked": table_unlocked, "anchors": anchors,
          "compare": compare, "auto": auto, "auto_locked": auto_locked}[sys.argv[1]]
    fn(*sys.argv[2:])
