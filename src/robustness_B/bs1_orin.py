"""PLAN §3g-12 BS-1 (B-g1, B-g3) on Orin #1, run after the E-13 timing batch in the same locked session.

G2 rebuilds (B-g1): the 9 graphs of table 4 / fig. 3, each at its E-11 Orin #1 target flag (features.csv `source`):
  build 1 = the original engine (hard link of ~/RD/work/engines/<graph>_<flag>_<sha12>.engine, not rebuilt),
  builds 2, 3 = fresh trtexec builds (t6_orin.sh: --fp16 --memPoolSize=workspace:1024 [+ nocublas tactic flag]).
  All 3 engines of every graph are timed in THIS session: 3 runs (fresh process each) x 500 iterations,
  rounds interleaved, REQUIRE_LOCK_HZ=624750000 (a run not at the lock is skipped and logged), flock inside t6_orin.sh.
  Workdir ~/RD/work_g2b -> results/robustness_B/g2_builds/ (results.jsonl, logs/, env/, builds.csv, summary.csv,
  pairs.csv).
G1 energy (B-g3): table 5 graphs (dANN-G m8/16/32/64 bmm, MLP h128 gemm / h256 conv / h512 conv / h1024 gemm = the
  cited cells of src/costmodel/data.py, CNN c32_64), each at its E-11 target flag; engine = the median-p50 build of
  the 5 buildvar builds at that flag (T6/lut/buildvar_builds_p{1,2}.csv). 3 reps per engine, interleaved, through
  ~/dendritic-edge-bench/T1/ms_cc/bin/ms_orin.sh energy (= src/bench/energy_one.sh protocol: jtop VDD_IN 100 ms,
  idle 12 s / >= 30 s inference / idle 14 s, holds the bench lock; REQUIRE_LOCK_HZ checked before every rep).
  -> results/T6/g1_energy.csv (+ raw in results/T6/raw/energy_g1/).
jetson_clocks / nvpmodel are never set (env dumps only query them); no sudo.

  python bs1_orin.py g2_start | g2_status | g2_pull | g2_table
  python bs1_orin.py e_start  | e_status  | e_pull  | e_table

Release notes. Device-side driver (needs the Orin #1 and its workdirs; not runnable from the release alone). Paths are
results/<exp>/ (the research repository used experiments/<exp>/); ROOT = this checkout.
"""
import csv
import json
import os
import statistics as st
import subprocess
import sys

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, f"{ROOT}/src/costmodel")
sys.path.insert(0, f"{ROOT}/src/bench")
import orin1_oos_time as OT  # noqa: E402  (ssh, check, lock_state)

ORIN, RD, LOCK_HZ = OT.ORIN, OT.RD, OT.LOCK_HZ
FEAT = f"{ROOT}/results/costmodel/features.csv"
G2OUT = f"{ROOT}/results/robustness_B/g2_builds"
G2WD = "work_g2b"
G2 = ["dhssm_L2_n128_b4_cont__K50", "dhssm_L2_n128_b4_cont__K25", "dhssm_L2_n128_b4_spk__K25",
      "dhssm_L2_n256_b4_cont__K25", "lru_L2_H64_N128_lr0.001_vs_t1__K50", "dssmU_L2_n128_b4_spk_vs_t1__K50",
      "gru_h768_lr0.003_vs_t1__seq", "gru_h64__seq", "gru_h256__seq"]
G1 = ["g1f_dannG_p4_m8_s64_sg1_lr0.001_wd0_fair_t1__bmm", "g1f_dannG_p4_m16_s64_sg1_lr0.001_wd0.01_fair_t1__bmm",
      "g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm", "g1f_dannG_p4_m64_s64_sg1_lr0.001_wd0.01_fair_t1__bmm",
      "g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm", "g1f_mlp_h256_lr0.0003_wd0_fair_t1__conv",
      "g1f_mlp_h512_lr0.0003_wd0.01_fair_t1__conv", "g1f_mlp_h1024_lr0.0003_wd0_fair_t1__gemm",
      "g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv"]
EWD = "dendritic-edge-bench/T6/energy_g1"
MS_SH = "dendritic-edge-bench/T1/ms_cc/bin/ms_orin.sh"
EOUT = f"{ROOT}/results/T6/raw/energy_g1"
LUT = f"{ROOT}/results/T6/lut"


def feat():
    return {r["graph"]: r for r in csv.DictReader(open(FEAT)) if r["tier"] == "orin1"}


def target_flag(r):
    s = r["source"]
    for b in ("nocublas", "default"):
        if f"p50_{b}" in s or f"({b})" in s:
            return b
    raise ValueError(s)


def g2_spec():
    F = feat()
    out = []
    for g in G2:
        r = F[g]
        onnx = r["file"]
        sub = onnx.split("/")[-2]   # g2 | g2x
        out.append(dict(graph=g, flag=target_flag(r), sha=r["sha256"], onnx_local=f"{ROOT}/{onnx.replace('experiments/', 'results/', 1)}",
                        onnx_orin=f"onnx/{sub}/{g}.onnx", e11_target_ms=float(r["target"]), e11_source=r["source"]))
    return out


def g2_start():
    OT.check()
    if OT.ssh(f"test -e ~/{RD}/{G2WD} && echo EXISTS").stdout.strip() == "EXISTS":
        raise SystemExit(f"~/{RD}/{G2WD} exists: refusing to start twice")
    if "batch done" not in OT.ssh(f"cat ~/{RD}/work_e13/progress.txt").stdout:
        raise SystemExit("E-13 batch not finished: BS-1 runs after BS-2 timing")
    spec = g2_spec()
    os.makedirs(G2OUT, exist_ok=True)
    prep, lines = [f"mkdir -p {G2WD}/engines {G2WD}/logs/orig"], []
    for s in spec:
        got = OT.ssh(f"sha256sum ~/{RD}/{s['onnx_orin']}").stdout.split()
        assert got and got[0] == s["sha"], (s["graph"], "Orin ONNX sha256 != features.csv", got)
        base = f"{s['graph']}_{s['flag']}"
        orig = f"work/engines/{base}_{s['sha'][:12]}.engine"
        assert OT.ssh(f"test -s ~/{RD}/{orig} && echo OK").stdout.strip() == "OK", (orig, "original engine missing")
        prep.append(f"ln {orig} {G2WD}/engines/g2b1_{base}_{s['sha'][:12]}.engine")
        prep.append(f"cp -p work/logs/{base}_build.log {G2WD}/logs/orig/ 2>/dev/null || true")
        prep.append(f"cp -p work/logs/{base}.layers.json {G2WD}/logs/orig/ 2>/dev/null || true")
    for k in (1, 2, 3):   # build-index outer: the fresh builds of one graph are spread over the build phase
        for s in spec:
            lines.append(f"$HOME/{RD}/{s['onnx_orin']} g2b{k}_{s['graph']} {s['flag']}")
    open(f"{G2OUT}/manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(spec, open(f"{G2OUT}/spec.json", "w"), indent=1)
    r = OT.ssh(f"cd ~/{RD} && " + " && ".join(prep), t=120)
    if r.returncode != 0:
        raise SystemExit(f"prep failed: {r.stderr}")
    subprocess.run(["rsync", "-a", f"{ROOT}/src/t6/orin/t6_orin.sh", f"{ROOT}/src/t6/orin/t6_sync_bench.py",
                    f"{ORIN}:{RD}/bin/"], check=True)
    subprocess.run(["rsync", "-a", f"{G2OUT}/manifest.txt", f"{ORIN}:{RD}/{G2WD}/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^\\$HOME#$HOME#\" {G2WD}/manifest.txt > {G2WD}/manifest_abs.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/t6_orin.sh batch {G2WD}/manifest_abs.txt $HOME/{RD}/{G2WD} 3 "
           f"> {G2WD}/batch.out 2>&1 < /dev/null &")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(len(lines), "engines (9 linked originals + 18 fresh builds), 3 runs each; started")


def g2_status():
    print("lock", OT.lock_state())
    print(OT.ssh(f"tail -3 ~/{RD}/{G2WD}/progress.txt; ls ~/{RD}/{G2WD}/engines | wc -l; "
                 f"wc -l < ~/{RD}/{G2WD}/results.jsonl 2>/dev/null; cat ~/{RD}/{G2WD}/failures.txt "
                 f"~/{RD}/{G2WD}/skipped.txt ~/{RD}/{G2WD}/unlocked.txt 2>/dev/null | tail -5; df -h / | tail -1").stdout)


def g2_pull():
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", "--exclude", "tegra", f"{ORIN}:{RD}/{G2WD}/", f"{G2OUT}/"],
                   check=True)
    subprocess.run(["rsync", "-a", f"{ORIN}:{RD}/{G2WD}/tegra/", f"{G2OUT}/tegra/"], check=True)
    print("pulled ->", G2OUT)


def _layers(path):
    """(n_layers, precision summary, tactic signature) from a --exportLayerInfo json."""
    if not os.path.exists(path):
        return "", "", ""
    try:
        j = json.load(open(path))
    except Exception:  # noqa: BLE001
        return "", "", ""
    L = j.get("Layers", [])
    tac = []
    prec = set()
    for l in L:
        if isinstance(l, dict):
            tac.append(f"{l.get('Name', '')}|{l.get('TacticName', l.get('TacticValue', ''))}")
            for o in l.get("Outputs", []):
                if isinstance(o, dict) and "Format/Datatype" in o:
                    prec.add("FP16" if "Half" in o["Format/Datatype"] else "FP32" if "Float" in o["Format/Datatype"]
                             else o["Format/Datatype"].split()[-1])
    return len(L), "+".join(sorted(prec)), " ;; ".join(tac)


def g2_table():
    spec = {s["graph"]: s for s in json.load(open(f"{G2OUT}/spec.json"))}
    res = {}
    for line in open(f"{G2OUT}/results.jsonl"):
        j = json.loads(line)
        res.setdefault(j["label"], []).append(j)
    unlocked = open(f"{G2OUT}/unlocked.txt").read() if os.path.exists(f"{G2OUT}/unlocked.txt") else ""
    fails = open(f"{G2OUT}/failures.txt").read() if os.path.exists(f"{G2OUT}/failures.txt") else ""
    rows, per = [], {}
    for g, s in spec.items():
        for k in (1, 2, 3):
            lab = f"g2b{k}_{g}"
            runs = sorted(res.get(lab, []), key=lambda j: j["run"])
            ok = [j for j in runs if j["gpu_freq_hz_timed_min_med_max"] == [LOCK_HZ] * 3]
            p = [j["p50_ms"] for j in ok]
            base = f"{lab}_{s['flag']}"
            blog = f"{G2OUT}/logs/{base}_build.log" if k > 1 else f"{G2OUT}/logs/orig/{g}_{s['flag']}_build.log"
            bsec = ""
            if os.path.exists(blog):
                for l in open(blog):
                    if l.startswith("build_seconds"):
                        bsec = int(l.split()[1])
            lay = f"{G2OUT}/logs/{base}.layers.json" if k > 1 else f"{G2OUT}/logs/orig/{g}_{s['flag']}.layers.json"
            nl, prec, tac = _layers(lay)
            row = dict(graph=g, build=k, origin="original engine (hard link)" if k == 1 else "fresh build 2026-10-30",
                       flag=s["flag"], onnx_sha256=s["sha"], engine=runs[0]["engine"] if runs else "",
                       engine_file_MB=runs[0].get("engine_file_MB", "") if runs else "", build_seconds=bsec,
                       nruns=len(runs), nruns_locked=len(ok), n_unlocked_skips=unlocked.count(f" {base}_"),
                       failed="yes" if base in fails else "",
                       p50_runs=" ".join(f"{v:.4f}" for v in p), p50=round(st.median(p), 4) if len(p) == 3 else "",
                       p50_min=round(min(p), 4) if p else "", p50_max=round(max(p), 4) if p else "",
                       p95_median=round(st.median(j["p95_ms"] for j in ok), 4) if ok else "",
                       n_layers=nl, precision=prec, tactics=tac, e11_target_ms=s["e11_target_ms"])
            rows.append(row)
            if row["p50"] != "":
                per.setdefault(g, []).append(row["p50"])
    keys = list(rows[0])
    with open(f"{G2OUT}/builds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = []
    for g, s in spec.items():
        v = per.get(g, [])
        tacs = {r["tactics"] for r in rows if r["graph"] == g and r["tactics"]}
        summ.append(dict(graph=g, flag=s["flag"], n_builds=len(v), per_build_p50=" ".join(f"{x:.4f}" for x in v),
                         median_p50=round(st.median(v), 4) if v else "", min_p50=min(v) if v else "",
                         max_p50=max(v) if v else "", spread_max_over_min=round(max(v) / min(v), 4) if v else "",
                         build1_p50_this_session=v[0] if v else "", e11_target_ms=s["e11_target_ms"],
                         build1_vs_e11=round(v[0] / s["e11_target_ms"], 4) if v else "",
                         n_distinct_tactic_sets=len(tacs)))
    with open(f"{G2OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    # C-b comparisons over all 3 x 3 build pairs (BS-3 rule: "LRU is not slower than DH-SSM" only if every pair <= 1.0)
    pb = {g: {r["build"]: r["p50"] for r in rows if r["graph"] == g and r["p50"] != ""} for g in spec}
    PAIRS = [("LRU K50 / DH-SSM n128 cont K25", "lru_L2_H64_N128_lr0.001_vs_t1__K50", "dhssm_L2_n128_b4_cont__K25"),
             ("LRU K50 / DH-SSM n128 cont K50", "lru_L2_H64_N128_lr0.001_vs_t1__K50", "dhssm_L2_n128_b4_cont__K50"),
             ("LRU K50 / DH-SSM n128 spk K25", "lru_L2_H64_N128_lr0.001_vs_t1__K50", "dhssm_L2_n128_b4_spk__K25"),
             ("LRU K50 / DH-SSM n256 cont K25", "lru_L2_H64_N128_lr0.001_vs_t1__K50", "dhssm_L2_n256_b4_cont__K25"),
             ("dssmU K50 / DH-SSM n128 cont K50", "dssmU_L2_n128_b4_spk_vs_t1__K50", "dhssm_L2_n128_b4_cont__K50"),
             ("GRU h768 / LRU K50", "gru_h768_lr0.003_vs_t1__seq", "lru_L2_H64_N128_lr0.001_vs_t1__K50"),
             ("GRU h768 / dssmU K50", "gru_h768_lr0.003_vs_t1__seq", "dssmU_L2_n128_b4_spk_vs_t1__K50"),
             ("GRU h768 / DH-SSM n256 cont K25", "gru_h768_lr0.003_vs_t1__seq", "dhssm_L2_n256_b4_cont__K25"),
             ("GRU h768 / DH-SSM n128 cont K25", "gru_h768_lr0.003_vs_t1__seq", "dhssm_L2_n128_b4_cont__K25"),
             ("GRU h64 / DH-SSM n128 cont K25", "gru_h64__seq", "dhssm_L2_n128_b4_cont__K25"),
             ("GRU h256 / DH-SSM n128 cont K25", "gru_h256__seq", "dhssm_L2_n128_b4_cont__K25")]
    pr = []
    for name, a, b in PAIRS:
        rr = [pb[a][i] / pb[b][j] for i in pb[a] for j in pb[b]]
        if not rr:
            continue
        ma = st.median(pb[a].values())
        mb = st.median(pb[b].values())
        pr.append(dict(pair=name, numerator=a, denominator=b, n_build_pairs=len(rr),
                       ratio_of_build_medians=round(ma / mb, 4), min_ratio=round(min(rr), 4), max_ratio=round(max(rr), 4),
                       median_ratio=round(st.median(rr), 4), all_pairs_le_1=all(x <= 1.0 for x in rr),
                       all_pairs_ge_1=all(x >= 1.0 for x in rr),
                       build1_ratio=round(pb[a][1] / pb[b][1], 4) if 1 in pb[a] and 1 in pb[b] else ""))
    with open(f"{G2OUT}/pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pr[0]))
        w.writeheader()
        w.writerows(pr)
    for s in summ:
        print(s["graph"], s["flag"], s["per_build_p50"], "median", s["median_p50"], "spread", s["spread_max_over_min"],
              "b1/E11", s["build1_vs_e11"])
    for p in pr:
        print(p)


# ---------------------------------------------------------------- G1 energy
def e_spec():
    F = feat()
    b = {}
    for ph in ("1", "2"):
        for r in csv.DictReader(open(f"{LUT}/buildvar_builds_p{ph}.csv")):
            b.setdefault((r["graph"], r["build"]), []).append((float(r["p50"]), int(r["k"]), ph, r["label"]))
    out = []
    for g in G1:
        r = F[g]
        fl = target_flag(r)
        v = sorted(b[(g, fl)])
        assert len(v) == 5, (g, fl, v)
        p50, k, ph, lab = v[2]   # median of the 5 builds
        out.append(dict(graph=g, flag=fl, k=k, phase=ph, sha=r["sha256"], buildvar_p50=p50,
                        buildvar_median=st.median(x[0] for x in v), e11_target_ms=float(r["target"]),
                        engine=f"$HOME/{RD}/work_bv{ph}/engines/bv{k}_{g}_{fl}_{r['sha256'][:12]}.engine"))
    return out


def e_start():
    OT.check()
    if OT.ssh(f"test -e ~/{EWD} && echo EXISTS").stdout.strip() == "EXISTS":
        raise SystemExit(f"~/{EWD} exists: refusing to start twice")
    for w in ("work_e13", G2WD):
        if "batch done" not in OT.ssh(f"cat ~/{RD}/{w}/progress.txt").stdout:
            raise SystemExit(f"{w} not finished")
    spec = e_spec()
    for s in spec:
        p = s["engine"].replace("$HOME/", "~/")
        assert OT.ssh(f"test -s {p} && echo OK").stdout.strip() == "OK", (p, "engine missing")
    assert OT.ssh(f"test -x ~/dendritic-edge-bench/energy/bin/energy_one.sh -o -s ~/dendritic-edge-bench/energy/bin/"
                  f"energy_one.sh && test -s ~/{MS_SH} && echo OK").stdout.strip() == "OK"
    os.makedirs(EOUT, exist_ok=True)
    json.dump(spec, open(f"{EOUT}/spec.json", "w"), indent=1)
    lines = [f"{s['graph']} {s['engine']}" for s in spec]
    open(f"{EOUT}/energy_list.txt", "w").write("\n".join(lines) + "\n")
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{EWD}"], check=True, timeout=60)
    subprocess.run(["rsync", "-a", f"{EOUT}/energy_list.txt", f"{ORIN}:{EWD}/"], check=True)
    cmd = (f"cd ~/{EWD} && sed \"s# \\$HOME# $HOME#\" energy_list.txt > energy_list_abs.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash ~/{MS_SH} energy $HOME/{EWD}/energy_list_abs.txt "
           f"$HOME/{EWD}/out 3 > energy.out 2>&1 < /dev/null &")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("energy started:", len(lines), "engines x 3 reps")


def e_status():
    print("lock", OT.lock_state())
    print(OT.ssh(f"tail -3 ~/{EWD}/out/progress.txt; cat ~/{EWD}/out/failures.txt ~/{EWD}/out/unlocked.txt "
                 f"~/{EWD}/out/skipped.txt 2>/dev/null | tail -5").stdout)


def e_pull():
    subprocess.run(["rsync", "-a", f"{ORIN}:{EWD}/", f"{EOUT}/"], check=True)
    print("pulled ->", EOUT)


def e_table():
    import energy_parse as EP
    d = f"{EOUT}/out"
    spec = {s["graph"]: s for s in json.load(open(f"{EOUT}/spec.json"))}
    runs = [r for r in (EP.per_run(d, json.loads(l)) for l in open(f"{d}/runs.jsonl") if l.strip()) if r]
    med_idle = st.median(r["P_idle_mW"] for r in runs)
    for r in runs:   # same exclusion rules as energy_parse.main (PLAN 2026-10-02)
        extra = []
        if abs(r["P_idle_mW"] - med_idle) > 300:
            extra.append(f"idle_dev_{r['P_idle_mW'] - med_idle:+.0f}mW")
        if r["n_samples_active"] < 0.9 * r["duration_s"] / 0.1:
            extra.append(f"few_samples_{r['n_samples_active']}")
        if r["gpu_frac_at_max_active"] is not None and r["gpu_frac_at_max_active"] < 1.0:
            extra.append(f"gpu_not_at_lock_{r['gpu_frac_at_max_active']}")
        r["flags"] = ";".join(x for x in [r["flags"]] + extra if x)
        r["idle_median_of_set_mW"] = round(med_idle, 1)
    with open(f"{EOUT}/g1_energy_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    rows = []
    for g, s in spec.items():
        allr = [r for r in runs if r["label"] == g]
        good = [r for r in allr if not r["flags"]]
        rng = lambda v: f"{min(v):.4g}-{max(v):.4g}" if v else ""  # noqa: E731
        rows.append(dict(
            graph=g, flag=s["flag"], engine=os.path.basename(s["engine"]), buildvar_build=f"p{s['phase']} k{s['k']}",
            buildvar_p50_ms=s["buildvar_p50"], e11_target_ms=s["e11_target_ms"], n_runs=len(good),
            n_flagged=len(allr) - len(good), flagged=" ".join(f"{r['tag']}:{r['flags']}" for r in allr if r["flags"]),
            E_net_mJ=round(st.median(r["E_net_mJ"] for r in good), 5) if good else "",
            E_net_range=rng([r["E_net_mJ"] for r in good]),
            E_gross_mJ=round(st.median(r["E_gross_mJ"] for r in good), 5) if good else "",
            dP_mW=round(st.median(r["dP_mW"] for r in good), 1) if good else "", dP_range=rng([r["dP_mW"] for r in good]),
            P_idle_mW=round(st.median(r["P_idle_mW"] for r in good), 1) if good else "",
            P_active_mW=round(st.median(r["P_active_mW"] for r in good), 1) if good else "",
            mean_ms=round(st.median(r["mean_ms"] for r in good), 4) if good else "",
            n_inf_median=int(st.median(r["n_inf"] for r in good)) if good else "",
            duration_s_median=round(st.median(r["duration_s"] for r in good), 1) if good else "",
            gpu_MHz_active_median=st.median(r["gpu_MHz_active_median"] for r in good) if good else "",
            tj_max_C=max(r["tj_max_C"] for r in good) if good else "", tags=" ".join(r["tag"] for r in good)))
    out = f"{ROOT}/results/T6/g1_energy.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["graph"], r["flag"], "E_net", r["E_net_mJ"], f"({r['E_net_range']})", "dP", r["dP_mW"], "idle",
              r["P_idle_mW"], "ms", r["mean_ms"], "n", r["n_runs"], r["flagged"])
    print("->", out)


if __name__ == "__main__":
    {"g2_start": g2_start, "g2_status": g2_status, "g2_pull": g2_pull, "g2_table": g2_table,
     "e_start": e_start, "e_status": e_status, "e_pull": e_pull, "e_table": e_table}[sys.argv[1]]()
