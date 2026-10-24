"""TRT build-to-build variance check (rebuilding the same ONNX graph with identical flags can give different tactics).

Engines behind numbers the paper cites, each x {default, nocublas}, built K=5 independent times (fresh trtexec builds,
same flags as every LUT build: --fp16 --memPoolSize=workspace:1024 [+ --tacticSources=-CUBLAS,-CUBLAS_LT]
--profilingVerbosity=detailed --exportLayerInfo), each build timed NRUNS=5 times (t6_sync_bench, fresh process per run).
Independent builds = distinct labels bv{k}_<graph> (so distinct engine/log/layers names); manifest is ordered k-outer
so the 5 builds of one graph are spread over the build phase.
  Set A  (G1 latmatch rows with a speed-up): dann24_D512_s128__conv, dannG_p4_m16_s128_sg4_sp24__bmm (onnx/g1),
         g1v2_mlp_h256/h512 __gemm/__conv (onnx/g1v2)
  Set B  (G1 fair-protocol models, onnx/g1f): dANN-G m8..m64 bmm, MLP h128..h1024 gemm+conv | CNN c8_16/c16_32/c32_64 conv (phase 2)
  anchors: mlp_h256__gemm, dannG_p4_m64_s64_sg1__bmm (onnx/g1)
Rules: Orin #1, flock /tmp/dn_orin_bench.lock (inside t6_orin.sh), REQUIRE_LOCK_HZ=624750000, stop if not locked at the
start; jetson_clocks / nvpmodel never run.  Workdir ~/RD/work_bv<phase>, pulled to raw/orin1_bv<phase>.
  python buildvar_orin.py manifest <phase 1|2> | run <phase> | table <phase> | ratios
Outputs (new files only): lut/buildvar_builds_p<phase>.csv (one row per engine build), lut/buildvar_summary.csv
(per graph x build), lut/buildvar_ratios.csv.
"""
import csv
import fnmatch
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, T6  # noqa: E402
import lut as L  # noqa: E402
import e7_g1_orin as E7  # noqa: E402

ORIN, RD = L.ORIN, L.RD
K, NRUNS = 5, 5
LOCK_HZ = E7.LOCK_HZ
G = "/sys/devices/gpu.0/devfreq/17000000.ga10b"
BUILDS = ("default", "nocublas")

SET_A = [("g1", "dann24_D512_s128__conv"), ("g1", "dannG_p4_m16_s128_sg4_sp24__bmm"),
         ("g1v2", "g1v2_mlp_h256__gemm"), ("g1v2", "g1v2_mlp_h256__conv"),
         ("g1v2", "g1v2_mlp_h512__gemm"), ("g1v2", "g1v2_mlp_h512__conv")]
ANCH = [("g1", a) for a in E7.ANCHORS]


def _g1f(pattern):
    """Paths of the exported g1f graphs; if the ONNX files are not present (they are not distributed), the same
    sorted list is rebuilt from the labels in lut/onnx/g1f/manifest.txt."""
    ps = sorted(glob.glob(f"{LUT}/onnx/g1f/{pattern}"))
    man = f"{LUT}/onnx/g1f/manifest.txt"
    if not ps and os.path.exists(man):
        labs = [l.split()[1] for l in open(man) if l.strip()]
        ps = sorted(f"{LUT}/onnx/g1f/{l}.onnx" for l in labs if fnmatch.fnmatch(f"{l}.onnx", pattern))
    return ps


SET_B_DM = [("g1f", os.path.basename(p)[:-5]) for p in _g1f("g1f_*.onnx")
            if "_dannG_" in p or "_mlp_" in p]
SET_B_CNN = [("g1f", os.path.basename(p)[:-5]) for p in _g1f("g1f_cnn_*.onnx")]
PHASE = {"1": SET_A + ANCH + SET_B_DM, "2": SET_B_CNN}


def wd(ph):
    return f"work_bv{ph}"


def raw(ph):
    return f"{T6}/raw/orin1_bv{ph}"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ssh(cmd, t=60):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ORIN, cmd],
                          capture_output=True, text=True, timeout=t)


def lock_state():
    r = ssh(f"cat {G}/min_freq {G}/cur_freq {G}/max_freq")
    try:
        return tuple(int(v) for v in r.stdout.split()[:3])
    except ValueError:
        return None


def manifest(ph):
    gs = PHASE[ph]
    lines, info = [], {}
    for d, g in gs:
        p = f"{LUT}/onnx/{d}/{g}.onnx"
        info[g] = {"dir": d, "onnx_sha256": sha(p)}
    for k in range(1, K + 1):
        for d, g in gs:
            lines.append(f"~/{RD}/onnx/{d}/{g}.onnx bv{k}_{g} {','.join(BUILDS)}")
    os.makedirs(raw(ph), exist_ok=True)
    open(f"{raw(ph)}/bv_manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(info, open(f"{raw(ph)}/bv_graphs.json", "w"), indent=1)
    print(len(gs), "graphs,", len(lines), "manifest lines,", len(lines) * len(BUILDS), "engines,",
          len(lines) * len(BUILDS) * NRUNS, "timed runs")
    return info


def run(ph, poll_s=300, run_max_h=8):
    T = lambda: time.strftime("%F %T")
    info = manifest(ph)
    st = lock_state()
    print(T(), "GPU devfreq min/cur/max", st, flush=True)
    if st != (LOCK_HZ,) * 3:
        raise SystemExit(f"{T()} GPU NOT LOCKED {st}: stop (jetson_clocks / nvpmodel are not run by this script)")
    W = wd(ph)
    if ssh(f"test -e ~/{RD}/{W} && echo yes").stdout.strip() == "yes":
        raise SystemExit(f"~/{RD}/{W} already exists on the Orin; refusing to restart")
    # the Orin copies of the graphs must be the local files (no push of graphs: they are already there)
    for g, v in info.items():
        r = ssh(f"sha256sum ~/{RD}/onnx/{v['dir']}/{g}.onnx")
        if not r.stdout.startswith(v["onnx_sha256"]):
            raise SystemExit(f"Orin graph {g} missing or sha differs: {r.stdout} {r.stderr}")
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/{W}"], check=True, timeout=60)
    subprocess.run(["rsync", "-a", f"{raw(ph)}/bv_manifest.txt", f"{ORIN}:{RD}/{W}/"], check=True)
    subprocess.run(["rsync", "-a", f"{here}/orin/t6_orin.sh", f"{here}/orin/t6_sync_bench.py", f"{ORIN}:{RD}/bin/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" {W}/bv_manifest.txt > {W}/bv_manifest_abs.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/t6_orin.sh batch {W}/bv_manifest_abs.txt $HOME/{RD}/{W} {NRUNS} "
           f"> {W}/bv.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)
    print(T(), "started (REQUIRE_LOCK_HZ set)", flush=True)
    wait(ph, poll_s, run_max_h)


def wait(ph, poll_s=300, run_max_h=8):
    T = lambda: time.strftime("%F %T")
    W = wd(ph)
    t_end, lost = time.time() + float(run_max_h) * 3600, 0
    while True:
        try:
            done = ssh(f"grep -c 'batch done' ~/{RD}/{W}/progress.txt || true").stdout.strip()
            st = lock_state()
            if st != (LOCK_HZ,) * 3:
                lost += 1
                print(T(), "WARNING LOCK LOST", st, flush=True)
            if done not in ("", "0"):
                break
            print(T(), "running:", ssh(f"tail -1 ~/{RD}/{W}/progress.txt").stdout.strip(), "| lock", st, flush=True)
        except subprocess.TimeoutExpired:
            print(T(), "ssh timeout", flush=True)
        if time.time() > t_end:
            raise SystemExit("batch not done in time")
        time.sleep(int(poll_s))
    pull(ph)
    print(T(), "pulled; lock-lost polls", lost, flush=True)
    table(ph)
    print(T(), "RUN DONE", flush=True)


def pull(ph):
    os.makedirs(raw(ph), exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/{wd(ph)}/", f"{raw(ph)}/"], check=True)


def tactics(path):
    """Per-layer summary of an exportLayerInfo json: compute layers (non-NoOp, non-Reformat) with tactic + output dtype."""
    d = json.load(open(path))
    comp, n16, n32, ntc = [], 0, 0, 0
    for Lr in d["Layers"]:
        if Lr.get("LayerType") in ("NoOp", "Reformat"):
            continue
        out = " ".join(o.get("Format/Datatype", "") for o in Lr.get("Outputs", []))
        tn = Lr.get("TacticName") or Lr.get("LayerType", "?")   # cuBLAS layers carry no TacticName
        is16 = "FP16" in out or "f16f16" in tn or "_h884" in tn or "_h1688" in tn or "fp16" in tn.lower()
        n16 += is16
        n32 += not is16
        ntc += "tensor" in tn or "h16816" in tn or "h1688" in tn or "h884" in tn or "hmma" in tn
        comp.append(f"{Lr.get('Name')}|{tn}|{out}")
    prec = "FP16" if n16 and not n32 else ("FP32" if n32 and not n16 else "mixed")
    return {"n_layers": len(d["Layers"]), "n_compute": len(comp), "n_fp16_compute": n16, "n_fp32_compute": n32,
            "n_tensorcore": ntc, "precision": prec, "tactics": " ;; ".join(comp)}


def table(ph):
    R = raw(ph)
    info = json.load(open(f"{R}/bv_graphs.json"))
    start = max(l.split()[0][:19] for l in open(f"{R}/progress.txt") if "batch start" in l)
    rows = [json.loads(l) for l in open(f"{R}/results.jsonl")]
    rows = [r for r in rows if r.get("time", "") >= start]
    out = []
    for g, v in info.items():
        for k in range(1, K + 1):
            lab = f"bv{k}_{g}"
            for b in BUILDS:
                rs0 = [r for r in rows if r["label"] == lab and r["build"] == b and r["onnx_sha256"] == v["onnx_sha256"]]
                rs = [r for r in rs0 if E7.run_locked(r, R)]
                rec = {"graph": g, "k": k, "build": b, "label": lab, "nruns": len(rs), "n_excluded_unlocked": len(rs0) - len(rs)}
                lj = f"{R}/logs/{lab}_{b}.layers.json"
                if os.path.exists(lj):
                    rec.update(tactics(lj))
                bl = f"{R}/logs/{lab}_{b}_build.log"
                if os.path.exists(bl):
                    txt = open(bl).read()
                    rec["precision_flag"] = next((l.split("Precision:")[1].strip() for l in txt.splitlines() if "] Precision:" in l), "")
                    rec["build_seconds"] = next((l.split()[1] for l in txt.splitlines() if l.startswith("build_seconds")), "")
                if rs:
                    p = [r["p50_ms"] for r in rs]
                    rec.update(p50=round(float(np.median(p)), 4), p50_min=round(min(p), 4), p50_max=round(max(p), 4),
                               engine_file_MB=rs[0]["engine_file_MB"])
                out.append(rec)
    keys = []
    for r in out:
        keys += [x for x in r if x not in keys]
    p = f"{LUT}/buildvar_builds_p{ph}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(len(out), "rows ->", p, "| locked runs", sum(r["nruns"] for r in out), "excluded", sum(r["n_excluded_unlocked"] for r in out))
    summary()
    return out


def summary():
    rows = []
    for p in sorted(glob.glob(f"{LUT}/buildvar_builds_p*.csv")):
        rows += list(csv.DictReader(open(p)))
    ref = {}
    for r in csv.DictReader(open(f"{LUT}/lut_g1f.csv")):
        ref[r["label"]] = r
    for r in csv.DictReader(open(f"{LUT}/lut_g1v2_session.csv")):
        ref[r["label"]] = r
    for r in csv.DictReader(open(f"{LUT}/lut_g1.csv")):
        ref.setdefault(f"{r['cfg']}__{r['impl']}", r)
    out = []
    for g in dict.fromkeys(r["graph"] for r in rows):
        for b in BUILDS:
            rs = [r for r in rows if r["graph"] == g and r["build"] == b and r.get("p50")]
            if not rs:
                continue
            v = [float(r["p50"]) for r in rs]
            pr = [r.get("precision", "") for r in rs]
            old = ref.get(g, {}).get(f"p50_{b}", "")
            out.append({"graph": g, "build": b, "n_builds": len(rs), "median_p50": round(float(np.median(v)), 4),
                        "min_p50": round(min(v), 4), "max_p50": round(max(v), 4),
                        "spread_max_over_min": round(max(v) / min(v), 3),
                        "per_build_p50": " ".join(f"{x:.4f}" for x in v),
                        "per_build_precision": " ".join(pr),
                        "n_builds_any_fp16_compute": sum(int(r.get("n_fp16_compute") or 0) > 0 for r in rs),
                        "n_builds_all_fp16_compute": sum(x == "FP16" for x in pr),
                        "n_distinct_tactic_sets": len({r.get("tactics", "") for r in rs}),
                        "precision_flag": rs[0].get("precision_flag", ""), "lut_p50_cited": old})
    for g in dict.fromkeys(r["graph"] for r in out):
        ss = [r for r in out if r["graph"] == g]
        best = min(ss, key=lambda r: r["median_p50"])
        for r in ss:
            r["is_best_build_for_graph"] = r is best
    keys = list(out[0])
    p = f"{LUT}/buildvar_summary.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(len(out), "rows ->", p)
    for r in out:
        print(r["graph"][:48].ljust(48), r["build"].ljust(8), r["median_p50"], f"{r['min_p50']}-{r['max_p50']}",
              r["spread_max_over_min"], r["per_build_precision"], "| cited", r["lut_p50_cited"])
    return out


# config -> list of graphs (impls); cited value = old LUT best (min over impl x build)
CFG = {"dann24_D512_s128": ["dann24_D512_s128__conv"],
       "dannG_sp24_m16": ["dannG_p4_m16_s128_sg4_sp24__bmm"],
       "lut_mlp_h256": ["g1v2_mlp_h256__gemm", "g1v2_mlp_h256__conv"],
       "lut_mlp_h512": ["g1v2_mlp_h512__gemm", "g1v2_mlp_h512__conv"]}
for _d, _g in SET_B_DM + SET_B_CNN:
    _c = _g.split("_lr")[0][4:]
    CFG.setdefault("g1f_" + _c, []).append(_g)
RATIOS = [("dann24_D512_s128", "lut_mlp_h512"), ("dannG_sp24_m16", "lut_mlp_h256"),
          ("g1f_dannG_p4_m16_s64_sg1", "g1f_mlp_h1024"), ("g1f_dannG_p4_m32_s64_sg1", "g1f_mlp_h1024"),
          ("g1f_dannG_p4_m64_s64_sg1", "g1f_mlp_h1024"), ("g1f_dannG_p4_m16_s64_sg1", "g1f_mlp_h512"),
          ("g1f_dannG_p4_m32_s64_sg1", "g1f_mlp_h512"), ("g1f_dannG_p4_m64_s64_sg1", "g1f_mlp_h512"),
          ("g1f_dannG_p4_m8_s64_sg1", "g1f_mlp_h128"), ("g1f_dannG_p4_m16_s64_sg1", "g1f_mlp_h256"),
          ("g1f_cnn_c32_64", "g1f_dannG_p4_m32_s64_sg1"), ("g1f_cnn_c32_64", "g1f_dannG_p4_m64_s64_sg1"),
          ("g1f_cnn_c8_16", "g1f_dannG_p4_m8_s64_sg1"), ("g1f_cnn_c8_16", "g1f_mlp_h128")]


def cfgval(c, S, B):
    """median-over-builds value of a config = min over (impl, build flag) of the per-graph median; plus that cell's
    min/max over builds, best-of-5 (min over everything), and per-k best (min over impl x flag of build k)."""
    cells = [r for r in S if r["graph"] in CFG[c]]
    if len(cells) < 2 * len(CFG[c]):
        return None
    best = min(cells, key=lambda r: float(r["median_p50"]))
    allp = [float(r["p50"]) for r in B if r["graph"] in CFG[c] and r.get("p50")]
    perk = []
    for k in range(1, K + 1):
        v = [float(r["p50"]) for r in B if r["graph"] in CFG[c] and int(r["k"]) == k and r.get("p50")]
        perk.append(min(v) if v else float("nan"))
    cellk = [float(x) for x in best["per_build_p50"].split()]
    return {"cellk": cellk, "med": float(best["median_p50"]), "lo": float(best["min_p50"]), "hi": float(best["max_p50"]),
            "cell": f"{best['graph']}|{best['build']}", "best5": min(allp), "perk": perk,
            "worst_cell_med": max(float(r["median_p50"]) for r in cells)}


def cited():
    v = {}
    for r in csv.DictReader(open(f"{LUT}/lut_g1.csv")):
        v.setdefault(r["cfg"], []).append(float(r["p50_ms"]))
    f = {}
    for r in csv.DictReader(open(f"{LUT}/lut_g1f.csv")):
        if r["family"] != "anchor":
            f.setdefault("g1f_" + r["cfg"].split("_lr")[0], []).append(min(float(r["p50_default"]), float(r["p50_nocublas"])))
    out = {"dann24_D512_s128": min(v["dann24_D512_s128"]), "dannG_sp24_m16": min(v["dannG_p4_m16_s128_sg4_sp24"]),
           "lut_mlp_h256": min(v["mlp_h256"]), "lut_mlp_h512": min(v["mlp_h512"])}
    out.update({k: min(x) for k, x in f.items()})
    return out


def ratios():
    S = list(csv.DictReader(open(f"{LUT}/buildvar_summary.csv")))
    B = []
    for p in sorted(glob.glob(f"{LUT}/buildvar_builds_p*.csv")):
        B += list(csv.DictReader(open(p)))
    C = cited()
    out = []
    for a, b in RATIOS:
        A, Bv = cfgval(a, S, B), cfgval(b, S, B)
        if A is None or Bv is None:
            print("skip (not measured yet)", a, b)
            continue
        pk = [x / y for x, y in zip(A["perk"], Bv["perk"])]
        out.append({"num": a, "den": b, "num_med": A["med"], "num_range": f"{A['lo']:.4f}-{A['hi']:.4f}", "num_cell": A["cell"],
                    "den_med": Bv["med"], "den_range": f"{Bv['lo']:.4f}-{Bv['hi']:.4f}", "den_cell": Bv["cell"],
                    "ratio_med": round(A["med"] / Bv["med"], 3),
                    "ratio_bound_lo": round(A["lo"] / Bv["hi"], 3), "ratio_bound_hi": round(A["hi"] / Bv["lo"], 3),
                    "ratio_perk_min": round(min(pk), 3), "ratio_perk_max": round(max(pk), 3),
                    "ratio_best_of_5": round(A["best5"] / Bv["best5"], 3),
                    "ratio_cited": round(C[a] / C[b], 3), "num_cited": C[a], "den_cited": C[b]})
    # CNN c8_16 fastest of all Set-B configs?
    vals = {c: cfgval(c, S, B) for c in CFG if c.startswith("g1f_")}
    vals = {c: v for c, v in vals.items() if v}
    if "g1f_cnn_c8_16" in vals:
        order = sorted(vals, key=lambda c: vals[c]["med"])
        print("Set B order by median-over-builds:", [(c, vals[c]["med"]) for c in order])
        out.append({"num": "g1f_cnn_c8_16", "den": order[1] if order[0] == "g1f_cnn_c8_16" else order[0],
                    "num_med": vals["g1f_cnn_c8_16"]["med"], "den_med": vals[order[1] if order[0] == "g1f_cnn_c8_16" else order[0]]["med"],
                    "ratio_med": round(vals["g1f_cnn_c8_16"]["med"] / vals[order[1] if order[0] == "g1f_cnn_c8_16" else order[0]]["med"], 3),
                    "num_cell": "fastest_setB=" + order[0]})
    keys = []
    for r in out:
        keys += [x for x in r if x not in keys]
    p = f"{LUT}/buildvar_ratios.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)
    print(len(out), "rows ->", p)
    return out


def cited_csv():
    """Final table cited in the paper: lut/buildvar_cited.csv (config rows + ratio rows + fastest-check row)."""
    S = list(csv.DictReader(open(f"{LUT}/buildvar_summary.csv")))
    B = []
    for p in sorted(glob.glob(f"{LUT}/buildvar_builds_p*.csv")):
        B += list(csv.DictReader(open(p)))
    C = cited()
    out = []
    vals = {}
    for c in CFG:
        v = cfgval(c, S, B)
        if not v:
            continue
        vals[c] = v
        g, b = v["cell"].split("|")
        cellb = [r for r in B if r["graph"] == g and r["build"] == b]
        allb = [r for r in B if r["graph"] in CFG[c]]
        out.append({"type": "config", "name": c, "median_p50_ms": v["med"], "range_ms": f"{v['lo']:.4f}-{v['hi']:.4f}",
                    "impl_build": f"{g.split('__')[-1]}|{b}", "graph": g, "n_builds": len(cellb),
                    "fp16_chosen_builds_cell": f"{sum(int(r.get('n_fp16_compute') or 0) > 0 for r in cellb)}/{len(cellb)}",
                    "fp16_chosen_builds_all_cells": f"{sum(int(r.get('n_fp16_compute') or 0) > 0 for r in allb)}/{len(allb)}",
                    "best_of_5_ms": v["best5"], "old_cited": C.get(c, "")})
    for a, b in RATIOS:
        A, Bv = vals.get(a), vals.get(b)
        if not A or not Bv:
            continue
        pk = [x / y for x, y in zip(A["cellk"], Bv["cellk"])]
        pb = [x / y for x, y in zip(A["perk"], Bv["perk"])]
        out.append({"type": "ratio", "name": f"{a} / {b}", "ratio_median": round(A["med"] / Bv["med"], 3),
                    "ratio_range_perk": f"{min(pk):.3f}-{max(pk):.3f}",
                    "ratio_range_perk_bestcell": f"{min(pb):.3f}-{max(pb):.3f}",
                    "ratio_bound_extremes": f"{A['lo'] / Bv['hi']:.3f}-{A['hi'] / Bv['lo']:.3f}",
                    "speedup_median": round(Bv["med"] / A["med"], 3),
                    "speedup_range_perk": f"{1 / max(pk):.3f}-{1 / min(pk):.3f}",
                    "old_cited": round(C[a] / C[b], 3)})
    setb = [c for c in vals if c.startswith("g1f_")]
    if "g1f_cnn_c8_16" in setb:
        order = sorted(setb, key=lambda c: vals[c]["med"])
        nk = sum(min(setb, key=lambda c: vals[c]["perk"][k]) == "g1f_cnn_c8_16" for k in range(K))
        second = order[1] if order[0] == "g1f_cnn_c8_16" else order[0]
        out.append({"type": "check", "name": "c8_16 fastest of Set B (median-over-builds)",
                    "result": f"fastest={order[0]} ({vals[order[0]]['med']}); next={order[1]} ({vals[order[1]]['med']})",
                    "ratio_median": round(vals["g1f_cnn_c8_16"]["med"] / vals[second]["med"], 3),
                    "c8_16_fastest_in_k_of_5": nk,
                    "old_cited": f"c8_16 {C['g1f_cnn_c8_16']} vs next {sorted((x for x in C.items() if x[0].startswith('g1f_') and x[0] != 'g1f_cnn_c8_16'), key=lambda x: x[1])[0]}"})
    keys = []
    for r in out:
        keys += [x for x in r if x not in keys]
    p = f"{LUT}/buildvar_cited.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    for r in out:
        print({k: v for k, v in r.items() if v != ""})
    print(len(out), "rows ->", p)


if __name__ == "__main__":
    {"manifest": manifest, "run": run, "wait": wait, "pull": pull, "table": table, "summary": summary, "ratios": ratios, "cited_csv": cited_csv}[sys.argv[1]](*sys.argv[2:])
