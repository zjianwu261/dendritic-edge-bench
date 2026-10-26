"""E-11 data assembly: every timed graph per tier with its target value (PROTOCOL.md P4), its
role (P5: calib / dendritic / fused_rnn / excluded) and the 72 test pairs (P6).

Graph keys are '<tier>:<graph>'. Nothing here fits anything.
"""
import csv
import json
import os
import re
import statistics as st

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = f"{ROOT}/results"
T1M = f"{EXP}/T1/models"
LUT = f"{EXP}/T6/lut"
ONNX = f"{LUT}/onnx"
G6 = f"{EXP}/T6/models"
O2 = f"{EXP}/robustness_B/orin2"
E10 = f"{EXP}/T1m/e10"

TIERS = ("orin1", "pi", "orin2", "mcu")


def _rows(path):
    return list(csv.DictReader(open(path)))


def _f(x):
    return float(x) if x not in (None, "", "nan") else None


# ------------------------------------------------------------------------------------- classification
DENDRITIC_PREFIX = ("dann_lrf", "dhsrnn", "active_dend", "dann24_", "dannG_", "g1f_dannG", "dhssm_", "adc_",
                    "dann_d", "dann_prng")


def is_dendritic(name):
    return name.startswith(DENDRITIC_PREFIX)


# ------------------------------------------------------------------------------------- effective params
def _json_peff(path):
    if os.path.exists(path):
        d = json.load(open(path))
        if "params_effective" in d:
            return int(d["params_effective"])
    return None


def params_eff_record(graph):
    """Effective parameter count from the training records (None -> caller uses B_nnz)."""
    t1 = {r["model"]: int(r["params"]) for r in _rows(f"{EXP}/T1/results.csv")}
    if graph in t1:
        return t1[graph]
    g = re.sub(r"__(conv|gemm|bmm|seq|K\d+)$", "", graph)
    cands = []
    if g.startswith("g1f_"):
        cands.append(f"{G6}/g1/{g[4:].replace('_t1', '')}_t1.json")
    if g.startswith("adc_"):
        m = re.match(r"adc_h(\d+)_(ff\w+)", g)
        cands.append(f"{G6}/g3/adc_h{m.group(1)}_k4_{m.group(2)}_t1.json")
    if g.endswith("_vs_t1"):
        cands.append(f"{G6}/g2/{g}.json")
    for sub in ("g1", "g2", "g3"):
        cands.append(f"{G6}/{sub}/{g}_t1.json")
    for c in cands:
        v = _json_peff(c)
        if v is not None:
            return v
    if g.startswith("adc_"):
        return _adnet_peff(g)
    return None


def _adnet_peff(g):
    """AD widths without a trained record (h4096): the same ADNet.effective_params() that wrote the
    params_effective of the trained widths (reproduces h1024 7355402 and h2048 15759370 exactly)."""
    import sys
    sys.path.insert(0, f"{ROOT}/src/t6")
    import ad_cache
    m = re.match(r"adc_h(\d+)_ff(\w+)", g)
    return int(ad_cache.ADNet(h=int(m.group(1)), segments=4, seed=1, ff=m.group(2)).effective_params())


# ------------------------------------------------------------------------------------- per tier
def _t1_rows(device_prefix):
    out = []
    for r in _rows(f"{EXP}/T1/results.csv"):
        if not r["device"].startswith(device_prefix):
            continue
        out.append(r)
    return out


def tier_orin1():
    G = {}
    # T1 graphs (P4: locked column if present, else DVFS p50 if all three timed runs were at 625 MHz)
    for r in _t1_rows("Orin"):
        m = r["model"]
        lk = _f(r["latency_p50_locked"])
        if lk is not None:
            tgt, src = lk, "T1/results.csv latency_p50_locked"
        elif r["gpu_MHz_timed_median"] == "625/625/625":
            tgt, src = _f(r["latency_p50"]), "T1/results.csv latency_p50 (DVFS, 3x625 MHz)"
        else:
            tgt, src = None, f"excluded P4: GPU MHz {r['gpu_MHz_timed_median']}"
        G[m] = dict(onnx=f"{T1M}/{m}.onnx", target=tgt, source=src)
    # lut_g1 (old G1 sweep; its mlp rows were replaced by the g1v2 session -> taken from there)
    for r in _rows(f"{LUT}/lut_g1.csv"):
        if r["cfg"].startswith("mlp_"):
            continue
        g = f"{r['cfg']}__{r['impl']}"
        G[g] = _lut_entry(r, f"{ONNX}/g1/{g}.onnx", "T6/lut/lut_g1.csv")
    for r in _rows(f"{LUT}/lut_g1v2_session.csv"):
        g = r["label"]
        if not g.startswith(("g1v2_", "g1x2_")):
            continue  # session anchors: re-timings of lut_g1 graphs, not new graphs
        G[g] = _lut_entry(r, f"{ONNX}/g1v2/{g}.onnx", "T6/lut/lut_g1v2_session.csv")
    # G1 fair protocol: median of 5 builds (buildvar), faster flag
    bv = {}
    for r in _rows(f"{LUT}/buildvar_summary.csv"):
        if r["graph"].startswith("g1f_"):
            bv.setdefault(r["graph"], []).append((float(r["median_p50"]), r["build"]))
    for g, v in bv.items():
        t, b = min(v)
        G[g] = dict(onnx=f"{ONNX}/g1f/{g}.onnx", target=t,
                    source=f"T6/lut/buildvar_summary.csv median of 5 builds, faster flag ({b})")
    for r in _rows(f"{LUT}/lut_g2.csv"):
        g = f"{r['cfg']}__{r['impl']}"
        G[g] = _lut_entry(r, f"{ONNX}/g2/{g}.onnx", "T6/lut/lut_g2.csv")
    for r in _rows(f"{LUT}/lut_g2x.csv"):
        g = r["label"]
        if not os.path.exists(f"{ONNX}/g2x/{g}.onnx"):
            continue  # anchors: re-timings of lut_g2 graphs (dhssm K50, gru_h256), already in lut_g2
        G[g] = _lut_entry(r, f"{ONNX}/g2x/{g}.onnx", "T6/lut/lut_g2x.csv")
    for r in _rows(f"{LUT}/lut_g3.csv"):
        g = f"{r['cfg']}__{r['impl']}"
        G[g] = _lut_entry(r, f"{ONNX}/g3/{g}.onnx", "T6/lut/lut_g3.csv")
    return G


def _lut_entry(r, path, src):
    """P4 Orin #1 lut graphs: p50 of the faster of the two TensorRT flags (default / nocublas).
    The 2:4-sparsity builds (sp_*) are a different executed form and are not used."""
    v = [(float(r[k]), k[4:]) for k in ("p50_default", "p50_nocublas") if _f(r.get(k)) is not None]
    t, b = min(v)
    fm = r.get(f"fmax_frac_{b}", r.get(f"gpu_fmax_frac_{b}", ""))
    return dict(onnx=path, target=t, source=f"{src} p50_{b} (fmax_frac {fm})")


def tier_pi():
    G = {}
    for r in _t1_rows("RPi"):
        m = r["model"]
        G[m] = dict(onnx=f"{T1M}/{m}.onnx", target=_f(r["latency_p50"]), source="T1/results.csv latency_p50 (RPi)")
    return G


def orin2_onnx(g):
    for p in (f"{T1M}/{g}.onnx", f"{ONNX}/g1f/{g}.onnx", f"{ONNX}/g3/{g}.onnx", f"{ONNX}/g2/{g}.onnx",
              f"{ONNX}/g2x/{g}.onnx"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(g)


def orin2_medians():
    cell = {}
    for r in _rows(f"{O2}/orin2_builds.csv"):
        cell.setdefault((r["graph"], r["build"]), []).append(float(r["p50"]))
    return {k: st.median(v) for k, v in cell.items()}


def tier_orin2():
    med = orin2_medians()
    G = {}
    for g in sorted({k[0] for k in med}):
        v = [(med[(g, b)], b) for b in ("default", "nocublas") if (g, b) in med]
        t, b = min(v)
        G[g] = dict(onnx=orin2_onnx(g), target=t,
                    source=f"robustness_B/orin2/orin2_builds.csv median of 3 builds, faster flag ({b})")
    return G


MCU_CALIB = ["mlp_h16", "mlp_h32", "mlp_h48", "mlp_h64", "mlp_h72", "cnn_c2_4", "cnn_c3_6", "cnn_c4_8",
             "cnn_c6_12", "cnn_c8_16", "cnn_c16_32", "cnn_c32_64", "cnns4_c2_4", "cnns4_c4_8"]


def _mcu_build_dir(model, base):
    ds = [d for d in os.listdir(base) if re.fullmatch(re.escape(model) + r"_lr[\d.]+_wd[\d.]+_t1", d)]
    if len(ds) != 1:
        raise FileNotFoundError(f"{model}: {ds}")
    return f"{base}/{ds[0]}"


def tier_mcu():
    G = {}
    for r in _rows(f"{E10}/results.csv"):
        m = r["model"]
        cyc, ms = float(r["board_cycles"]), float(r["board_ms"])
        G[m] = dict(model_h=f"{_mcu_build_dir(m, E10 + '/build')}/model.h", target=cyc, ms=ms,
                    mhz=cyc / ms / 1000.0, source=f"T1m/e10/results.csv board_cycles ({r['board_ms_source']})")
    for r in _rows(f"{E10}/cnnvar/results.csv"):
        m = r["model"]
        if m not in ("cnns4_c2_4", "cnns4_c4_8"):
            continue
        cyc, ms = float(r["board_cycles_median"]), float(r["board_ms"])
        G[m] = dict(model_h=f"{_mcu_build_dir(m, E10 + '/cnnvar/build')}/model.h", target=cyc, ms=ms,
                    mhz=cyc / ms / 1000.0, source=f"T1m/e10/cnnvar/results.csv board_cycles_median ({r['board_ms_source']})")
    return G


# ------------------------------------------------------------------------------------- 72 test pairs (P6)
# (id, tier, dendritic graph, dense graph, dense-measured override or None)
GF = "g1f_dannG_p4_{}_s64_sg1_lr0.001_wd{}_fair_t1__bmm"
M8, M32, M64 = GF.format("m8", "0"), GF.format("m32", "0.01"), GF.format("m64", "0.01")
H128 = "g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm"     # cited cells (T6/lut/buildvar_cited.csv)
H256 = "g1f_mlp_h256_lr0.0003_wd0_fair_t1__conv"
H512 = "g1f_mlp_h512_lr0.0003_wd0.01_fair_t1__conv"
H1024 = "g1f_mlp_h1024_lr0.0003_wd0_fair_t1__gemm"
C3264 = "g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv"
LRU = "lru_L2_H64_N128_lr0.001_vs_t1__K50"
DSSMU = "dssmU_L2_n128_b4_spk_vs_t1__K50"
GRU768 = "gru_h768_lr0.003_vs_t1__seq"

_O_T1 = [
    ("1", "dann_lrf_d32_s128", "mlp_h85_match_d32_s128"),
    ("2", "dann_lrf_d32_s128", "vann_same_d32_s128"),
    ("3", "dann_lrf_d32_s128", "mlp_h1024_fmnist"),
    ("4", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seq"),
    ("5", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seqloop"),
    ("6", "dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val_seqloop"),
    ("7", "dhsrnn_n64_b8_step", "gru_h23_step"),
    ("8", "dhsrnn_n64_b8_step", "lif_srnn_n64_step"),
    ("9", "active_dend_k4", "mlp_h3592_match_k4"),
    ("10", "active_dend_k8", "mlp_h4965_match_k8"),
    ("11", "active_dend_k4", "mlp_h3941_matchstored_k4"),
    ("12", "active_dend_k8", "mlp_h5230_matchstored_k8"),
    ("13", "active_dend_k4_cached", "mlp_h3592_match_k4"),
    ("14", "active_dend_k8_cached", "mlp_h4965_match_k8"),
]


def pairs():
    P = []
    for i, num, den in _O_T1:
        P.append((f"O{i}", "orin1", num, den))
    for k, W in enumerate((256, 512, 1024, 2048, 4096)):
        P.append((f"O{15 + k}", "orin1", f"adc_h{W}_ffrand__gemm", f"mlp_h{W}__gemm"))
    P += [("O20", "orin1", M8, H128), ("O21", "orin1", M32, H128), ("O22", "orin1", M32, H256),
          ("O23", "orin1", M32, H512), ("O24", "orin1", M32, H1024), ("O25", "orin1", M64, H1024),
          ("O26", "orin1", M32, C3264), ("O27", "orin1", M64, C3264),
          ("O28", "orin1", "dhssm_L2_n128_b4_cont__K50", LRU),
          ("O29", "orin1", "dhssm_L2_n128_b4_cont__K25", LRU),
          ("O30", "orin1", "dhssm_L2_n128_b4_spk__K25", DSSMU),
          ("O31", "orin1", "dhssm_L2_n256_b4_cont__K25", GRU768),
          ("O32", "orin1", "dhssm_L2_n128_b4_spk__K25", "gru_h64__seq"),
          ("O33", "orin1", "dhssm_L2_n128_b4_cont__K25", "gru_h256__seq")]
    for i, num, den in _O_T1:
        P.append((f"P{i}", "pi", num, den))
    P += [("Q1", "orin2", "g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm", H256),
          ("Q2", "orin2", "g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm", H128),
          ("Q3", "orin2", "adc_h1024_ffrand__gemm", "mlp_h1024__gemm"),
          ("Q4", "orin2", "adc_h2048_ffrand__gemm", "mlp_h2048__gemm"),
          ("Q5", "orin2", "dhssm_L2_n128_b4_cont__K25", GRU768),
          ("Q6", "orin2", "dhssm_L2_n128_b4_cont__K25", LRU),
          ("Q7", "orin2", "dhssm_L2_n128_b4_cont__K50", LRU),
          ("Q8", "orin2", "dhsrnn_n64_b8_val_seqloop", "gru_pm_val_seq"),
          ("Q9", "orin2", "dann_lrf_d32_s128", "mlp_h85_match_d32_s128"),
          ("Q10", "orin2", "dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val_seqloop")]
    P += [("M1", "mcu", "dann_d16_s64", "mlp_h48"), ("M2", "mcu", "dann_prng_d16_s128", "mlp_h64"),
          ("M3", "mcu", "dann_prng_d20_s128", "mlp_h64")]
    k = 4
    for r in _rows(f"{E10}/cnnvar/matched.csv"):
        if r["variant"] == "primary_int8val_ddof1" and r["kind"] in ("dann", "prng"):
            P.append((f"M{k}", "mcu", r["model"], r["all_matched_cnn"]))
            k += 1
    assert len(P) == 72, len(P)
    return P


def dense_measured_override(pid):
    """Q1 names the dense cell explicitly: MLP h256 Conv graph, nocublas flag."""
    if pid == "Q1":
        return orin2_medians()[(H256, "nocublas")], "orin2_builds.csv h256 conv|nocublas (named in P6 Q1)"
    return None


def all_graphs():
    return dict(orin1=tier_orin1(), pi=tier_pi(), orin2=tier_orin2(), mcu=tier_mcu())
