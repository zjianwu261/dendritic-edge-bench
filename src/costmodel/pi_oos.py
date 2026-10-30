"""E-12: post-hoc out-of-sample check of the Pi cost-model constants (PLAN.md section 3g-10 B5;
protocol results/costmodel/pi_oos/PROTOCOL.md, committed as H1 in PLAN.md). Post hoc: outside the
pre-specified E-11 protocol.

  predict  (before any Pi timing) -> pi_oos/{PROTOCOL.md, graphs_pre.csv, pairs_pre.csv, pre.sha256}
           Recomputes K / F / B / F_eff / B_nnz with features.graph_features and asserts equality with the
           orin1 rows of features.csv; uses the point estimates of fit_pi.json (no refit). Refuses to run if
           pi_oos/raw/ exists. Reads no Pi data.
  verdict  (after timing)            -> pi_oos/{timing.csv, pairs_pred.csv, verdict.json}
           Checks pre.sha256 first, then P6 (validity, anchor drift), P7 (S1'-S3'), P8 (reading).
           Amendment PROTOCOL_A1.md (commit H3): throttled rule A1-1 (baseline line in env_log.txt), rows of
           aborted rounds in raw/results_aborted.jsonl are skipped (A1-2), void reasons are categorised (A1-4).

Run with the dendritic conda env: python src/costmodel/pi_oos.py {predict|verdict} [--out DIR]

Release notes. Paths are results/costmodel/pi_oos/ (the research repository used experiments/costmodel/pi_oos/).
The commits H1 (PLAN.md with the protocol), H2 (pre.sha256) and H3 (PROTOCOL_A1.md) live in the research
repository's git history, which is not part of the release. `verdict` looks for that history in DEB_RESEARCH_ROOT
(default: ROOT); without it the commit checks are skipped and verdict.json records the commit hashes as
`recorded, not verified` together with a note. The SHA-256 of the pre-specified files (pre.sha256) and of
PROTOCOL_A1.md are always checked. `--out DIR` writes timing.csv / pairs_pred.csv / verdict.json to DIR instead of
pi_oos/ (inputs are still read from pi_oos/). `predict` needs the exported ONNX graphs and the research git history
(neither is in the release) and refuses to run because pi_oos/raw/ exists.
"""
import csv
import hashlib
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

ROOT = data.ROOT
CM = f"{data.EXP}/costmodel"
OUTD = f"{CM}/pi_oos"
RAW = f"{OUTD}/raw"
ONNX = data.ONNX
H1 = "470787a6211d85e59c96350cb372829b6b695ef6"
# Release: values recorded in the research repository (used when its git history is unavailable)
H2_RECORDED = "29e3cefc174c687589f87fd01636cda61d15f88d"
H3_RECORDED = "b9a147dc404f781e6d5da3109e1d035ac32f7cd4"
A1_SHA256 = "9c64f271eab339de1785bf918ae97354db14127e1764728cabcefb051fde26e0"
RESEARCH_ROOT = os.environ.get("DEB_RESEARCH_ROOT", ROOT)
RESEARCH_PI_OOS = "experiments/costmodel/pi_oos"  # path of pi_oos/ inside the research repository
NO_HISTORY = ("research git history unavailable (not part of the release): commit checks skipped; "
              "hashes are the values recorded in the research repository, not verified here")
PRE_FILES = ("PROTOCOL.md", "graphs_pre.csv", "pairs_pre.csv")
LN125, LN15 = math.log(1.25), math.log(1.5)

# ------------------------------------------------------------------------------------- P2 graph list
GF = "g1f_dannG_p4_{}_s64_sg1_lr0.001_wd{}_fair_t1__bmm"
M8, M16, M32, M64 = GF.format("m8", "0"), GF.format("m16", "0.01"), GF.format("m32", "0.01"), GF.format("m64", "0.01")
H128 = "g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm"
H256 = "g1f_mlp_h256_lr0.0003_wd0_fair_t1__gemm"
H512 = "g1f_mlp_h512_lr0.0003_wd0.01_fair_t1__gemm"
H1024 = "g1f_mlp_h1024_lr0.0003_wd0_fair_t1__gemm"
C3264 = "g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv"
LRU = "lru_L2_H64_N128_lr0.001_vs_t1__K50"
DSSMU = "dssmU_L2_n128_b4_spk_vs_t1__K50"
GRU768 = "gru_h768_lr0.003_vs_t1__seq"
WIDTHS = (256, 512, 1024, 2048, 4096)

# (graph, subdir, role, group) in P2 order; role: main_dendritic / main_dense / secondary / anchor
GRAPHS = (
    [(g, "g1f", "main_dendritic", "G1") for g in (M8, M32, M64)]
    + [(g, "g1f", "main_dense", "G1") for g in (H128, H256, H512, H1024, C3264)]
    + [(g, "g2", "main_dendritic", "G2") for g in ("dhssm_L2_n128_b4_cont__K50", "dhssm_L2_n128_b4_cont__K25",
                                                    "dhssm_L2_n128_b4_spk__K25", "dhssm_L2_n256_b4_cont__K25")]
    + [(LRU, "g2x", "main_dense", "G2"), (DSSMU, "g2x", "main_dense", "G2"), (GRU768, "g2x", "main_dense", "G2"),
       ("gru_h64__seq", "g2", "main_dense", "G2"), ("gru_h256__seq", "g2", "main_dense", "G2")]
    + [(f"adc_h{W}_ffrand__gemm", "g3", "main_dendritic", "G3") for W in WIDTHS]
    + [(f"mlp_h{W}__gemm", "g3", "main_dense", "G3") for W in WIDTHS]
    + [(M16, "g1f", "secondary", "G1"), ("g1f_cnn_c8_16_lr0.001_wd0_fair_t1__conv", "g1f", "secondary", "G1"),
       ("g1f_cnn_c16_32_lr0.001_wd0_fair_t1__conv", "g1f", "secondary", "G1"),
       ("dhssm_L2_n128_b4_spk__K50", "g2", "secondary", "G2"), ("dhssm_L2_n256_b4_cont__K50", "g2", "secondary", "G2")]
)
ANCHORS = ("mlp_h85_match_d32_s128", "dann_lrf_d32_s128", "active_dend_k4_cached")

# P3: 19 pairs (PG id, dendritic, dense, group, E-11 Orin #1 counterpart)
PAIRS = (
    [("PG1", M8, H128, "G1", "O20"), ("PG2", M32, H128, "G1", "O21"), ("PG3", M32, H256, "G1", "O22"),
     ("PG4", M32, H512, "G1", "O23"), ("PG5", M32, H1024, "G1", "O24"), ("PG6", M64, H1024, "G1", "O25"),
     ("PG7", M32, C3264, "G1", "O26"), ("PG8", M64, C3264, "G1", "O27"),
     ("PG9", "dhssm_L2_n128_b4_cont__K50", LRU, "G2", "O28"),
     ("PG10", "dhssm_L2_n128_b4_cont__K25", LRU, "G2", "O29"),
     ("PG11", "dhssm_L2_n128_b4_spk__K25", DSSMU, "G2", "O30"),
     ("PG12", "dhssm_L2_n256_b4_cont__K25", GRU768, "G2", "O31"),
     ("PG13", "dhssm_L2_n128_b4_spk__K25", "gru_h64__seq", "G2", "O32"),
     ("PG14", "dhssm_L2_n128_b4_cont__K25", "gru_h256__seq", "G2", "O33")]
    + [(f"PG{15 + k}", f"adc_h{W}_ffrand__gemm", f"mlp_h{W}__gemm", "G3", f"O{15 + k}") for k, W in enumerate(WIDTHS)]
)
FEAT = ("K", "F", "B", "F_eff", "B_nnz")


def onnx_path(g, sub):
    return f"{data.T1M}/{g}.onnx" if sub == "T1" else f"{ONNX}/{sub}/{g}.onnx"


def cited_path(p):
    """Path as cited in features.csv / graphs_pre.csv (research layout: results/<exp>/ -> experiments/<exp>/)."""
    r = os.path.relpath(p, ROOT)
    return "experiments/" + r[len("results/"):] if r.startswith("results/") else r


def all_graphs():
    """P2 order: main 27, secondary 5, anchors 3 (35)."""
    return list(GRAPHS) + [(g, "T1", "anchor", "anchor") for g in ANCHORS]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_constants():
    c = json.load(open(f"{CM}/fit_pi.json"))["constants"]
    return c["a_ms"], c["kappa_ms_per_node"], c["inv_phi_ms_per_MAC"], c["inv_beta_ms_per_weight"]


def t_hat(K, F, B, consts=None):
    a, kap, iphi, ibeta = consts or load_constants()
    return a + kap * K + F * iphi + B * ibeta


def feature_rows():
    return {(r["tier"], r["graph"]): r for r in csv.DictReader(open(f"{CM}/features.csv"))}


def recompute(g, sub, FR):
    """Features of one graph, asserted equal to the orin1 (or, for anchors, pi) row of features.csv."""
    import features
    p = onnx_path(g, sub)
    f = features.graph_features(p)
    tier = "pi" if sub == "T1" else "orin1"
    ref = FR[(tier, g)]
    assert cited_path(p) == ref["file"], (g, p, ref["file"])
    for k in FEAT:
        assert int(f[k]) == int(float(ref[k])), (g, k, f[k], ref[k])
    assert int(f["fused_rnn"]) == int(ref["fused_rnn"]), (g, "fused_rnn")
    s = sha_file(p)
    assert s == ref["sha256"], (g, "sha256")
    pe = data.params_eff_record(g)
    if pe is None:
        pe = int(f["B_nnz"])
    assert int(pe) == int(float(ref["params_eff"])), (g, "params_eff", pe, ref["params_eff"])
    return dict({k: int(f[k]) for k in FEAT}, fused_rnn=int(f["fused_rnn"]), params_eff=int(pe), sha256=s,
                file=cited_path(p))


def research_git():
    """True when RESEARCH_ROOT is a git checkout of the research repository that contains commit H1."""
    r = subprocess.run(["git", "-C", RESEARCH_ROOT, "cat-file", "-e", f"{H1}^{{commit}}"], capture_output=True)
    return r.returncode == 0


def write_protocol():
    if not research_git():
        raise SystemExit("predict needs the research git history (commit H1); set DEB_RESEARCH_ROOT")
    plan = subprocess.run(["git", "-C", RESEARCH_ROOT, "show", f"{H1}:PLAN.md"], check=True, capture_output=True,
                          text=True).stdout.split("\n")
    ih = next(i for i, l in enumerate(plan) if l.startswith("*(3) 协议正文（E-12"))
    i0 = next(i for i in range(ih, len(plan)) if plan[i].startswith("*P1 目的。*"))
    i9 = next(i for i, l in enumerate(plan) if l.startswith("*P9 读法与报告"))
    i1 = next(i for i in range(i9, len(plan)) if plan[i].startswith("- **B5 验收**"))
    while plan[i1 - 1].strip() == "":
        i1 -= 1
    body = "\n".join(plan[i0:i1])
    hdr = f"照抄自 PLAN.md §3g-10 B5 (3)，提交 H1（{H1}，PLAN.md 第 {i0 + 1}–{i1} 行）。\n\n"
    open(f"{OUTD}/PROTOCOL.md", "w").write(hdr + body + "\n")


def predict():
    if os.path.exists(RAW):
        raise SystemExit(f"{RAW} exists: predict must run before any Pi timing")
    os.makedirs(OUTD, exist_ok=True)
    write_protocol()
    consts = load_constants()
    FR = feature_rows()
    G = {}
    rows = []
    for g, sub, role, grp in all_graphs():
        f = recompute(g, sub, FR)
        th = "" if f["fused_rnn"] else t_hat(f["K"], f["F"], f["B"], consts)
        G[g] = dict(f, T_hat_ms=th)
        rows.append(dict(graph=g, file=f["file"], onnx_sha256=f["sha256"], role=role, group=grp,
                         **{k: f[k] for k in FEAT}, params_eff=f["params_eff"], fused_rnn=f["fused_rnn"],
                         T_hat_ms=th))
        print(f"{role:15s} {g:58s} K={f['K']:6d} F={f['F']:11d} B={f['B']:10d} T_hat={th}", flush=True)
    with open(f"{OUTD}/graphs_pre.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    a, kap, iphi, ibeta = consts
    prs = []
    for pid, dn, ds, grp, oid in PAIRS:
        n, d = G[dn], G[ds]
        blind = a + kap * d["K"] + n["F_eff"] * iphi + n["params_eff"] * ibeta
        prs.append(dict(pair=pid, group=grp, orin1_pair=oid, dendritic=dn, dense=ds, T_hat_dend_ms=n["T_hat_ms"],
                        T_hat_blind_ms=blind, P_param=n["params_eff"] / d["params_eff"], P_MAC=n["F_eff"] / d["F_eff"],
                        T_hat_dense_ms=d["T_hat_ms"]))
    with open(f"{OUTD}/pairs_pre.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(prs[0].keys()))
        w.writeheader()
        w.writerows(prs)
    with open(f"{OUTD}/pre.sha256", "w") as fo:
        for fn in PRE_FILES:
            fo.write(f"{sha_file(f'{OUTD}/{fn}')}  {fn}\n")
    print(open(f"{OUTD}/pre.sha256").read())


# ------------------------------------------------------------------------------------- verdict
def check_pre():
    got = {}
    for line in open(f"{OUTD}/pre.sha256"):
        h, fn = line.split()
        got[fn] = h
        assert sha_file(f"{OUTD}/{fn}") == h, f"{fn} changed after pre.sha256"
    assert set(got) == set(PRE_FILES)
    return got


CHECK_TIME = "2026-10-27 08:14:00"  # A1-1: environment check; a Pi boot later than this = rebooted


def throttled_ok(val, baseline, rebooted):
    """PROTOCOL_A1.md A1-1. Rebooted: strict P6 rule (== 0x0). Not rebooted: bits 0-3 all 0 and bits 16-19
    identical to the session baseline (any new sticky bit = undervoltage / throttling during the session)."""
    v = int(val, 16)
    if rebooted:
        return v == 0
    return (v & 0xF) == 0 and (v & 0xF0000) == (int(baseline, 16) & 0xF0000)


def parse_env_log(path):
    """{(label, run): [(throttled, baseline, rebooted), ...]} for the last attempt of each (label, run).

    env_log.txt format (written by the timing driver):
      '## baseline throttled=0x50000 boot=2026-09-30 10:12:28'  (session start; a resumed session writes a new one)
      '## before <label> run <k> ...' / '## after <label> run <k> ...' blocks with a 'throttled=0x..' line.
    A new 'before' block for a key already seen starts a new attempt (an aborted round re-run, A1-2), so only the
    readings of the last attempt are kept."""
    out, cur, base, reb = {}, None, None, None
    for line in open(path):
        m = re.match(r"## baseline throttled=(0x[0-9a-fA-F]+) boot=(\S+ \S+)", line)
        if m:
            base, reb, cur = m.group(1), m.group(2) > CHECK_TIME, None
            continue
        m = re.match(r"## (before|after) (\S+) run (\d+)", line)
        if m:
            cur = (m.group(2), int(m.group(3)))
            if m.group(1) == "before":
                out[cur] = []
            else:
                out.setdefault(cur, [])
        elif line.startswith("throttled=") and cur:
            assert base is not None, "env_log.txt has no '## baseline' line before the first run (A1-1)"
            out[cur].append((line.strip().split("=", 1)[1], base, reb))
    return out


def _last_commit(path):
    r = subprocess.run(["git", "-C", RESEARCH_ROOT, "log", "-1", "--format=%H", "--", path],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def amendment(hist):
    f = f"{OUTD}/PROTOCOL_A1.md"
    s = sha_file(f)
    assert s == A1_SHA256, "PROTOCOL_A1.md differs from the amendment committed as H3"
    if hist:
        return dict(file=f"{RESEARCH_PI_OOS}/PROTOCOL_A1.md", sha256=s,
                    commit=_last_commit(f"{RESEARCH_PI_OOS}/PROTOCOL_A1.md"))
    return dict(file="results/costmodel/pi_oos/PROTOCOL_A1.md", sha256=s, commit=H3_RECORDED, commit_verified=False)


def commits(hist):
    if hist:
        return dict(H1=H1, H2=_last_commit(f"{RESEARCH_PI_OOS}/pre.sha256"))
    return dict(H1=H1, H2=H2_RECORDED, verified=False, note=NO_HISTORY)


def verdict(outd=None):
    shas = check_pre()
    hist = research_git()
    outd = outd or OUTD
    os.makedirs(outd, exist_ok=True)
    pre_g = {r["graph"]: r for r in csv.DictReader(open(f"{OUTD}/graphs_pre.csv"))}
    pre_p = list(csv.DictReader(open(f"{OUTD}/pairs_pre.csv")))
    res = {}
    for line in open(f"{RAW}/results.jsonl"):
        j = json.loads(line)
        res.setdefault(j["label"], []).append(j)
    env = parse_env_log(f"{RAW}/env_log.txt")
    fails = {}
    if os.path.exists(f"{RAW}/failures.txt"):
        for line in open(f"{RAW}/failures.txt"):
            m = re.match(r"PI FAILED: (\S+) run (\d+)(.*)", line.strip())
            if m:
                fails.setdefault(m.group(1), []).append(line.strip())
    t1 = {r["model"]: float(r["latency_p50"]) for r in csv.DictReader(open(f"{data.EXP}/T1/results.csv"))
          if r["device"].startswith("RPi")}

    aborted = 0  # A1-2: rows of aborted rounds are kept in raw/results_aborted.jsonl and never read here
    if os.path.exists(f"{RAW}/results_aborted.jsonl"):
        aborted = sum(1 for _ in open(f"{RAW}/results_aborted.jsonl"))
    timing, T, bad = [], {}, {}  # bad[g] = [(category, detail)]; categories as in A1-4
    for g, sub, role, grp in all_graphs():
        rr = sorted(res.get(g, []), key=lambda j: j["run"])
        why = []
        if g in fails:
            why += [("load or run failure", x) for x in fails[g]]
        if rr and len(rr) != 3:
            why.append(("fewer than 3 valid rounds", f"{len(rr)} valid rounds"))
        for j in rr:
            th = env.get((g, j["run"]), [])
            if len(th) < 2 or not all(throttled_ok(*t) for t in th):
                why.append(("throttled", f"run {j['run']} throttled {[t[0] for t in th]} "
                                         f"(baseline {th[0][1] if th else None}, rebooted {th[0][2] if th else None})"))
            assert j["iters"] == 500 and j["threads"] == 3 and j["affinity"] == [0, 2, 3] and j["warmup"] >= 50
        if not rr:
            if role == "secondary" and g not in fails:
                continue  # not timed (P2 allows skipping the secondary set)
            if g not in fails:
                why.append(("fewer than 3 valid rounds", "0 valid rounds"))
        if why:
            bad[g] = why
        if rr:
            p50 = [j["p50_ms"] for j in rr]
            T[g] = st.median(p50)
            timing.append(dict(graph=g, role=role, group=grp, **{f"p50_run{k + 1}": p50[k] if k < len(p50) else ""
                                                                  for k in range(3)},
                               p50_median=T[g], p95_median=st.median(j["p95_ms"] for j in rr),
                               vm_hwm_median=st.median(j["vm_hwm_MB"] for j in rr),
                               ort_version=";".join(sorted({j["runtime"] for j in rr})), n_runs=len(rr),
                               status="void" if g in bad else "ok"))
    with open(f"{outd}/timing.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(timing[0].keys()))
        w.writeheader()
        w.writerows(timing)

    sgn = lambda v: (v > 0) - (v < 0)  # noqa: E731
    out, void = [], []
    for p in pre_p:
        dn, ds = p["dendritic"], p["dense"]
        reasons = [dict(graph=g, category=c, detail=x) for g in (dn, ds) for c, x in bad.get(g, [])]
        row = dict(pair=p["pair"], tier="pi", dendritic=dn, dense=ds)
        if reasons or dn not in T or ds not in T:
            reasons = reasons or [dict(graph="", category="fewer than 3 valid rounds", detail="missing timing")]
            void.append(dict(pair=p["pair"], categories=sorted({x["category"] for x in reasons}), reasons=reasons))
            row.update(status="void")
            out.append(row)
            continue
        T_n, T_d = T[dn], T[ds]
        r = T_n / T_d
        rhat = float(p["T_hat_dend_ms"]) / T_d
        rhat2 = float(p["T_hat_dend_ms"]) / float(p["T_hat_dense_ms"]) if p["T_hat_dense_ms"] else None
        p_param, p_mac, p_blind = float(p["P_param"]), float(p["P_MAC"]), float(p["T_hat_blind_ms"]) / T_d
        row.update(T_dendritic_ms=T_n, T_dense_ms=T_d, dense_source="pi_oos/timing.csv p50_median", r=r,
                   ln_r=math.log(r), decisive=int(abs(math.log(r)) >= LN125), rhat=rhat,
                   rhat_preddenom="" if rhat2 is None else rhat2, rhat_P_param=p_param, rhat_P_MAC=p_mac,
                   rhat_blind=p_blind)
        for tag, v in (("model", rhat), ("P_param", p_param), ("P_MAC", p_mac), ("blind", p_blind)):
            row[f"dir_ok_{tag}"] = int(sgn(math.log(v)) == sgn(math.log(r)))
            row[f"abs_lnerr_{tag}"] = abs(math.log(v) - math.log(r))
        row["dir_ok_preddenom"] = "" if rhat2 is None else int(sgn(math.log(rhat2)) == sgn(math.log(r)))
        row["abs_lnerr_preddenom"] = "" if rhat2 is None else abs(math.log(rhat2) - math.log(r))
        row["status"] = "ok"
        out.append(row)
    cols = list(next(csv.DictReader(open(f"{CM}/pairs_pred.csv"))).keys()) + ["status"]
    with open(f"{outd}/pairs_pred.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    ok = [r for r in out if r["status"] == "ok"]
    dec = [r for r in ok if r["decisive"]]
    tags = ("model", "P_param", "P_MAC", "blind")
    n_dir = sum(r["dir_ok_model"] for r in dec)
    stat = {t: (st.median(r[f"abs_lnerr_{t}"] for r in dec) if dec else None) for t in tags}
    s1 = (n_dir / len(dec) >= 0.8) if dec else None
    s2 = (stat["model"] <= LN15) if dec else None
    s3d = {t: stat["model"] <= 0.5 * stat[t] for t in ("P_param", "P_MAC", "blind")} if dec else {}
    s3 = all(s3d.values()) if dec else None
    if len(dec) < 4:
        reading = "信息不足 (insufficient: < 4 decisive ratios)"
    elif s1 and s2 and s3:
        reading = "PASS′"
    elif s1 and s3:
        reading = "DIRECTION′"
    else:
        reading = "FAIL′"

    tie = [r for r in ok if not r["decisive"]]
    pdn = [r for r in dec if r["rhat_preddenom"] != ""]
    absd = []
    for g, sub, role, grp in all_graphs():
        if g in T and g not in bad and pre_g[g]["T_hat_ms"] and (role == "main_dendritic" or
                                                                 (role == "secondary" and data.is_dendritic(g))):
            e = math.log(float(pre_g[g]["T_hat_ms"]) / T[g])
            absd.append(dict(graph=g, role=role, T_hat_ms=float(pre_g[g]["T_hat_ms"]), T_meas_ms=T[g],
                             pred_over_meas=math.exp(e), abs_ln_err=abs(e)))
    groups = {}
    for grp in ("G1", "G2", "G3"):
        gr = [r for r in ok if PG_GROUP[r["pair"]] == grp]
        gd = [r for r in gr if r["decisive"]]
        groups[grp] = dict(n_ok=len(gr), n_decisive=len(gd), dir_correct=sum(r["dir_ok_model"] for r in gd),
                           median_abs_lnerr=(st.median(r["abs_lnerr_model"] for r in gd) if gd else None),
                           median_factor=(math.exp(st.median(r["abs_lnerr_model"] for r in gd)) if gd else None),
                           misses=[r["pair"] for r in gd if not r["dir_ok_model"]])
    drift = {}
    for g in ANCHORS:
        if g in T:
            drift[g] = dict(p50_median=T[g], t1_latency_p50=t1[g], ratio=T[g] / t1[g])
    rat = [v["ratio"] for v in drift.values()]
    reproduced = len(rat) == 3 and all(0.90 <= x <= 1.10 for x in rat)
    max_dev = max(rat, key=lambda x: abs(math.log(x))) if rat else None
    vers = sorted({j["runtime"] for v in res.values() for j in v})
    V = dict(
        experiment="E-12 post-hoc Pi out-of-sample check (outside the pre-specified E-11 protocol)",
        commits=commits(hist), pre_sha256=shas, protocol_amendment=amendment(hist), reading=reading,
        n_pairs=len(out), n_ok=len(ok), n_void=len(void), void=void,
        n_void_by_category={c: sum(c in v["categories"] for v in void)
                            for c in ("throttled", "fewer than 3 valid rounds", "load or run failure")},
        aborted_rows_skipped=aborted, n_decisive=len(dec),
        S1=dict(passed=s1, direction_correct=n_dir, n_decisive=len(dec), frac=(n_dir / len(dec) if dec else None),
                threshold=0.8),
        S2=dict(passed=s2, median_abs_lnerr=stat["model"],
                median_factor=(math.exp(stat["model"]) if dec else None), threshold=LN15, threshold_factor=1.5),
        S3=dict(passed=s3, proxy_statistic=stat,
                proxy_factor={t: (math.exp(v) if v is not None else None) for t, v in stat.items()},
                proxy_dir_correct={t: sum(r[f"dir_ok_{t}"] for r in dec) for t in tags},
                model_le_half_of=s3d),
        anchor_drift=dict(per_graph=drift, reproduced=reproduced, max_drift_factor=max_dev,
                          label=("环境复现" if reproduced else f"环境漂移（最大 {max_dev:.3f} 倍）" if max_dev else "无锚点")),
        ort_versions=vers,
        secondary=dict(
            tie_band=dict(n=len(tie), n_rhat_in_band=sum(abs(math.log(r["rhat"])) < LN125 for r in tie),
                          pairs=[dict(pair=r["pair"], r=r["r"], rhat=r["rhat"]) for r in tie]),
            predicted_denominator=dict(n_decisive_with_prediction=len(pdn),
                                       dir_correct=sum(r["dir_ok_preddenom"] for r in pdn),
                                       median_abs_lnerr=(st.median(r["abs_lnerr_preddenom"] for r in pdn) if pdn else None),
                                       note="fused GRU denominators have no prediction"),
            absolute_error_dendritic=dict(
                n=len(absd), median_factor=(math.exp(st.median(x["abs_ln_err"] for x in absd)) if absd else None),
                max_factor=(math.exp(max(x["abs_ln_err"] for x in absd)) if absd else None), graphs=absd),
            by_group=groups,
            secondary_set_timed=[g for g, s, role, grp in GRAPHS if role == "secondary" and g in T]),
        constants=dict(zip(("a_ms", "kappa_ms_per_node", "inv_phi_ms_per_MAC", "inv_beta_ms_per_weight"),
                           load_constants())),
    )
    json.dump(V, open(f"{outd}/verdict.json", "w"), indent=1, ensure_ascii=False)
    for r in out:
        if r["status"] == "ok":
            print(f"{r['pair']:5s} r={r['r']:8.4f} rhat={r['rhat']:8.4f} Pp={r['rhat_P_param']:8.4f} "
                  f"Pm={r['rhat_P_MAC']:8.4f} bl={r['rhat_blind']:8.4f} dec={r['decisive']} dir={r['dir_ok_model']}")
        else:
            print(r["pair"], "void")
    print(json.dumps({k: V[k] for k in ("reading", "n_ok", "n_void", "n_decisive", "S1", "S2", "S3", "anchor_drift",
                                        "ort_versions")}, indent=1, ensure_ascii=False))


PG_GROUP = {p[0]: p[3] for p in PAIRS}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "predict":
        predict()
    elif cmd == "verdict":
        verdict(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None)
    else:
        raise SystemExit("usage: pi_oos.py {predict|verdict}")
