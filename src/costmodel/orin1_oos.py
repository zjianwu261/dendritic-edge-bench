"""E-13 (BS-2): post-hoc out-of-sample check of the Orin #1 cost-model constants on NEW graph shapes
(PLAN.md section 3g-12 BS-2; protocol results/costmodel/orin1_oos/PROTOCOL.md = P1-P9 of commit H1 plus the
implementation rules P2 / P5 leave to the implementer, written before any timing and committed with H2).

  export   random-weight ONNX of the new shapes -> orin1_oos/onnx/ + onnx/manifest.csv (sha256, ops, params_eff,
           ORT check). Existing files are never overwritten. Asserts: op multiset of every new graph == op multiset of
           its E-11 reference graph (features.csv orin1 row), no Identity node, no initializer used by two nodes.
  predict  (before any Orin #1 timing of these graphs) -> orin1_oos/{PROTOCOL.md, graphs_pre.csv, pairs_pre.csv,
           pre.sha256}. Features with features.graph_features; constants = fit_orin1.json["constants"] point
           estimates (no refit). Asserts that no new shape equals any tier = orin1 row of features.csv (name-level
           shape key and (K, F, B) triple) and that no new ONNX sha256 appears in features.csv. Refuses to run if
           orin1_oos/raw/ exists.
  manifest (after H2) -> orin1_oos/onnx/orin_manifest.txt for src/t6/orin/t6_orin.sh (3 builds = 3 labels b{k}_*)
  verdict  (after timing) -> orin1_oos/{timing.csv, pairs_pred.csv, verdict.json}; checks pre.sha256 first.

Run with the dendritic conda env: python src/costmodel/orin1_oos.py {export|predict|manifest|verdict} [--out DIR]

Release notes. Paths are results/costmodel/orin1_oos/ (the research repository used experiments/costmodel/orin1_oos/).
The commits H1 (PLAN.md with the protocol) and H2 (pre.sha256) live in the research repository's git history, which
is not part of the release. `verdict` looks for that history in DEB_RESEARCH_ROOT (default: ROOT); without it the
commit check is skipped and verdict.json records the hashes as `recorded, not verified` together with a note. The
SHA-256 of the files fixed before timing (pre.sha256) is always checked. `--out DIR` writes timing.csv / pairs_pred.csv /
verdict.json to DIR instead of orin1_oos/ (inputs are still read from orin1_oos/). `export`, `predict` and `manifest`
need the exported ONNX graphs (orin1_oos/onnx/, not in the release); `predict` also needs the research git history
and refuses to run because orin1_oos/raw/ exists. File paths written into CSVs keep the research layout
(experiments/<exp>/), as in features.csv.
"""
import collections
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
OUTD = f"{CM}/orin1_oos"
ONNXD = f"{OUTD}/onnx"
RAW = f"{OUTD}/raw"
H1 = "720e9539c5ccba6277d83b9adb5da1aa36966249"
# Release: value recorded in the research repository (used when its git history is unavailable)
H2_RECORDED = "7e8f26d2018d81319864fb845b52d2bd91dca8b1"
RESEARCH_ROOT = os.environ.get("DEB_RESEARCH_ROOT", ROOT)
RESEARCH_OUTD = "experiments/costmodel/orin1_oos"  # path of orin1_oos/ inside the research repository
NO_HISTORY = ("research git history unavailable (not part of the release): commit check skipped; "
              "hashes are the values recorded in the research repository, not verified here")
PRE_FILES = ("PROTOCOL.md", "graphs_pre.csv", "pairs_pre.csv")
LN125, LN15 = math.log(1.25), math.log(1.5)
LOCK_HZ = 624750000
FLAGS = ("default", "nocublas")
NBUILDS, NRUNS, ITERS = 3, 3, 500
FEAT = ("K", "F", "B", "F_eff", "B_nnz")

# ------------------------------------------------------------------------------------- E-11 reference graphs
R_M8 = "g1f_dannG_p4_m8_s64_sg1_lr0.001_wd0_fair_t1__bmm"
R_H128G = "g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm"
R_H128C = "g1f_mlp_h128_lr0.001_wd0_fair_t1__conv"
C3264 = "g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv"
R_DH25, R_DH50 = "dhssm_L2_n128_b4_cont__K25", "dhssm_L2_n128_b4_cont__K50"
R_LRU = "lru_L2_H64_N128_lr0.001_vs_t1__K50"
R_DSSMU = "dssmU_L2_n128_b4_spk_vs_t1__K50"
R_GRU = "gru_h256__seq"
R_ADC = "adc_h256_ffrand__gemm"
R_MLP3 = "mlp_h256__gemm"
LUTO = data.ONNX  # results/T6/lut/onnx (research: experiments/T6/lut/onnx)

# ------------------------------------------------------------------------------------- P2 graph list
# (label, group, role, family, spec, reference graph for the op multiset (None = existing file))
# role: main_dendritic / main_dense / anchor (anchors: not in P2, report only - environment drift vs E-11)
G1_M, G1_H = (12, 24, 48), (192, 384, 768)
G2_DH = (("dhssm", 192, 25), ("dhssm", 192, 50), ("dhssm", 384, 25))
G3_W = (384, 768, 1536, 3072)


def dann(m):
    return f"e13_g1_dannG_p4_m{m}_s64_sg1__bmm"


def mlp1(h, form):
    return f"e13_g1_mlp_h{h}__{form}"


def dh(n, K):
    return f"e13_g2_dhssm_L2_n{n}_b4_cont__K{K}"


LRU192, DSSMU192 = "e13_g2_lru_L2_H64_N192__K50", "e13_g2_dssmU_L2_n192_b4_spk__K50"


def gru(h):
    return f"e13_g2_gru_h{h}__seq"


def adc(W):
    return f"e13_g3_adc_h{W}_ffrand__gemm"


def mlp3(W):
    return f"e13_g3_mlp_h{W}__gemm"


GRAPHS = (
    [(dann(m), "G1", "main_dendritic", "dannG", dict(m=m), R_M8) for m in G1_M]
    + [(mlp1(h, f), "G1", "main_dense", "mlp1", dict(h=h, form=f), R_H128G if f == "gemm" else R_H128C)
       for h in G1_H for f in ("gemm", "conv")]
    + [(C3264, "G1", "main_dense", "existing", dict(path=f"{LUTO}/g1f/{C3264}.onnx"), None)]
    + [(dh(n, K), "G2", "main_dendritic", "dhssm", dict(n=n, K=K), R_DH25 if K == 25 else R_DH50) for _, n, K in G2_DH]
    + [(LRU192, "G2", "main_dense", "lru", dict(N=192), R_LRU),
       (DSSMU192, "G2", "main_dense", "dssmU", dict(n=192), R_DSSMU)]
    + [(gru(h), "G2", "main_dense", "gru", dict(h=h), R_GRU) for h in (192, 384)]
    + [(adc(W), "G3", "main_dendritic", "adc", dict(h=W), R_ADC) for W in G3_W]
    + [(mlp3(W), "G3", "main_dense", "mlp3", dict(h=W), R_MLP3) for W in G3_W]
    + [(R_DH50, "anchor", "anchor", "existing", dict(path=f"{LUTO}/g2/{R_DH50}.onnx"), None),
       ("mlp_h1024__gemm", "anchor", "anchor", "existing", dict(path=f"{LUTO}/g3/mlp_h1024__gemm.onnx"), None)]
)
G1_FORMS = {h: (mlp1(h, "gemm"), mlp1(h, "conv")) for h in G1_H}   # dense target = faster form (see PROTOCOL A2)
MLP1 = {h: f"e13_g1_mlp_h{h}" for h in G1_H}                       # pair-level name of the two-form dense graph

# P3: 14 pairs (PG id, dendritic, dense, group); dense "e13_g1_mlp_h<h>" = the faster of its two forms
PAIRS = (
    [("PG1", dann(12), MLP1[192], "G1"), ("PG2", dann(24), MLP1[384], "G1"), ("PG3", dann(48), MLP1[768], "G1"),
     ("PG4", dann(24), C3264, "G1"), ("PG5", dann(48), C3264, "G1"),
     ("PG6", dh(192, 50), LRU192, "G2"), ("PG7", dh(192, 25), DSSMU192, "G2"), ("PG8", dh(384, 25), LRU192, "G2"),
     ("PG9", dh(192, 25), gru(192), "G2"), ("PG10", dh(384, 25), gru(384), "G2")]
    + [(f"PG{11 + k}", adc(W), mlp3(W), "G3") for k, W in enumerate(G3_W)]
)
PG_GROUP = {p[0]: p[3] for p in PAIRS}


def onnx_file(label):
    for g in GRAPHS:
        if g[0] == label:
            return g[4]["path"] if g[3] == "existing" else f"{ONNXD}/{label}.onnx"
    raise KeyError(label)


def dense_members(d):
    """Graphs that make up the dense side of a pair (two forms for the G1 MLPs)."""
    for h, nm in MLP1.items():
        if d == nm:
            return list(G1_FORMS[h])
    return [d]


# ------------------------------------------------------------------------------------- shape keys (P2 assertion)
def shape_key(name):
    """Family + widths of a graph name, independent of training tags (lr / wd / t1 / vs) and of the e13_ / g1f_ /
    g1v2_ prefixes (G1 and G3 MLPs share the 784-h-h-10 shape, so they are deliberately not told apart).
    Used for the P2 assertion 'no new shape equals an orin1 row of features.csv'."""
    n = re.sub(r"^(e13_g[123]_|g1f_|g1v2_|g1x2_)", "", name)
    return re.sub(r"_lr[0-9.e-]+|_wd[0-9.e-]+|_fair|_vs|_t1", "", n)


# ------------------------------------------------------------------------------------- export
def _distinct_bias(net):
    """g1v2 rule (src/t6/g1v2_orin.py): an all-zero bias would be de-duplicated by torch.onnx (Identity artifact);
    every all-zero bias is replaced by U(-1/sqrt(fan_in), 1/sqrt(fan_in)) from Generator(seed 1), one draw per tensor."""
    import torch
    g = torch.Generator().manual_seed(1)
    changed = []
    with torch.no_grad():
        for name, mod in net.named_modules():
            b = getattr(mod, "bias", None)
            w = getattr(mod, "weight", None)
            if isinstance(b, torch.nn.Parameter) and w is not None and b.numel() and not bool(b.abs().sum()):
                fan_in = w[0].numel() if w.dim() > 1 else w.numel()
                bd = 1 / math.sqrt(fan_in)
                b.copy_((torch.rand(b.shape, generator=g) * 2 - 1) * bd)
                changed.append(name or "<root>")
    return changed


def build_net(fam, spec):
    """(module to export, example input, params_eff, params_eff rule) - same constructors as the E-11 families."""
    import torch
    sys.path.insert(0, f"{ROOT}/src/t6")
    torch.manual_seed(0)
    if fam == "dannG":
        from dann_g import DANNG
        net = DANNG(4, spec["m"], 64, 1)
        net.impl = "bmm"
        x = torch.rand(1, 784, generator=torch.Generator().manual_seed(0))
        return net, x, int(net.effective_params()), "DANNG.effective_params() (as the g1f training records)"
    if fam == "mlp1":
        from dann_g import Dense24MLP
        net = Dense24MLP(spec["h"], spec["h"])
        net.impl = spec["form"]
        x = torch.rand(1, 784, generator=torch.Generator().manual_seed(0))
        return net, x, int(net.effective_params()), "Dense24MLP.effective_params() (as the g1f training records)"
    if fam in ("dhssm", "dssmU", "lru", "gru"):
        from dh_ssm import DHSSM, DHSSMExport, GRUNetD, LRUExport, LRUNet, N_IN, T_STEPS
        if fam == "dhssm":
            m = DHSSM(spec["n"], 4, 2, "cont").eval()
            net, K = DHSSMExport(m, spec["K"]), spec["K"]
            pe, rule = m.effective_params(), "DHSSM.effective_params() (as the g2 training records)"
        elif fam == "dssmU":
            m = DHSSM(spec["n"], 4, 2, "spk", grouped=False).eval()
            net = DHSSMExport(m, 50)
            pe, rule = m.effective_params(), "DHSSM.effective_params() (as the g2 training records)"
        elif fam == "lru":
            m = LRUNet(64, spec["N"], 2, 0.1, K=50).eval()
            net = LRUExport(m, 50)
            pe, rule = m.effective_params(), "LRUNet.effective_params() (as the g2 training records)"
        else:
            net = GRUNetD(spec["h"]).eval()
            pe, rule = None, "B_nnz (as the lut GRU rows of features.csv)"
        x = torch.zeros(1, T_STEPS, N_IN).bernoulli_(0.02, generator=torch.Generator().manual_seed(0))
        return net, x, (None if pe is None else int(pe)), rule
    if fam == "adc":
        import ad_cache
        net = ad_cache._LutAD(spec["h"], "rand")
        net.impl = "gemm"
        pe = int(ad_cache.ADNet(h=spec["h"], segments=4, seed=1, ff="rand").effective_params())
        x = torch.rand(1, 784, generator=torch.Generator().manual_seed(0))
        return net, x, pe, "ADNet(h, 4, seed=1, ff).effective_params() (data._adnet_peff)"
    if fam == "mlp3":
        import ad_cache
        net = ad_cache.MLP(spec["h"])
        net.impl = "gemm"
        x = torch.rand(1, 784, generator=torch.Generator().manual_seed(0))
        return net, x, int(net.effective_params()), "MLP.effective_params() (as the g3 training records)"
    raise ValueError(fam)


def onnx_checks(path):
    """No Identity node; no initializer consumed by two nodes (the g1v2 export artifact)."""
    import onnx
    m = onnx.load(path)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    inits = {i.name for i in m.graph.initializer}
    use = collections.Counter(i for n in m.graph.node for i in n.input if i in inits)
    shared = sorted(k for k, v in use.items() if v > 1)
    return dict(identity=ops.get("Identity", 0), shared_initializers=shared)


def ops_string(f):
    return ";".join(f"{k}:{v}" for k, v in sorted(f["ops"].items()))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cited_path(p):
    """Path as cited in features.csv / manifest.csv / graphs_pre.csv (research layout: results/ -> experiments/)."""
    r = os.path.relpath(p, ROOT)
    return "experiments/" + r[len("results/"):] if r.startswith("results/") else r


def research_git():
    """True when RESEARCH_ROOT is a git checkout of the research repository that contains commit H1."""
    r = subprocess.run(["git", "-C", RESEARCH_ROOT, "cat-file", "-e", f"{H1}^{{commit}}"], capture_output=True)
    return r.returncode == 0


def feature_rows():
    return {(r["tier"], r["graph"]): r for r in csv.DictReader(open(f"{CM}/features.csv"))}


def export():
    import features
    sys.path.insert(0, f"{ROOT}/src/t6")
    from common6 import export_onnx
    os.makedirs(ONNXD, exist_ok=True)
    FR = feature_rows()
    rows = []
    for lab, grp, role, fam, spec, ref in GRAPHS:
        if fam == "existing":
            continue
        path = f"{ONNXD}/{lab}.onnx"
        net, x, pe, rule = build_net(fam, spec)
        changed = _distinct_bias(net)
        net.eval()
        if not os.path.exists(path):
            info = export_onnx(net, (x,), path, ["x"], ["y"], check=True)
        else:
            info = {"onnx": os.path.basename(path), "note": "existing file kept"}
        chk = onnx_checks(path)
        f = features.graph_features(path)
        ops = ops_string(f)
        ref_ops = FR[("orin1", ref)]["ops"]
        assert ops == ref_ops, (lab, ops, ref, ref_ops)
        assert chk["identity"] == 0 and not chk["shared_initializers"], (lab, chk)
        if pe is None:
            pe = int(f["B_nnz"])
        rows.append(dict(graph=lab, group=grp, role=role, family=fam, spec=json.dumps(spec, sort_keys=True),
                         file=cited_path(path), onnx_sha256=sha_file(path), size_MB=info.get("size_MB", ""),
                         ort_max_abs_err=info.get("max_abs_err_vs_torch", ""), ops=ops, reference_graph=ref,
                         params_eff=pe, params_eff_rule=rule, zero_bias_replaced=";".join(changed)))
        print(lab, f"K={f['K']} F={f['F']} B={f['B']} ops==ref ({ref})", info.get("max_abs_err_vs_torch", ""), flush=True)
    with open(f"{ONNXD}/manifest.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "exported ->", f"{ONNXD}/manifest.csv")


# ------------------------------------------------------------------------------------- model
def load_constants():
    c = json.load(open(f"{CM}/fit_orin1.json"))["constants"]
    return c["a_ms"], c["kappa_ms_per_node"], c["inv_phi_ms_per_MAC"], c["inv_beta_ms_per_weight"]


def t_hat(K, F, B, consts=None):
    a, kap, iphi, ibeta = consts or load_constants()
    return a + kap * K + F * iphi + B * ibeta


def t_blind(dend, dense_K, consts=None):
    """mapping-blind (E-11 P7 S3): the dendritic graph as a dense model of its effective size, K of the dense graph."""
    a, kap, iphi, ibeta = consts or load_constants()
    return a + kap * dense_K + dend["F_eff"] * iphi + dend["params_eff"] * ibeta


def orin1_shapes():
    FR = feature_rows()
    o = [r for (t, g), r in FR.items() if t == "orin1"]
    return ({shape_key(r["graph"]) for r in o}, {(int(r["K"]), int(r["F"]), int(r["B"])) for r in o},
            {r["sha256"] for r in o if r.get("sha256")})


# ------------------------------------------------------------------------------------- PROTOCOL.md
APPENDIX = f"""

---

## 附：dn-exec 按 P2 / P5 写定的实现细则（计时之前写定，随 H2 提交；不改 P1–P9 的任何一条）

**A1 图（P2）。** 全部随机权重，`src/costmodel/orin1_oos.py export` 导出到 `experiments/costmodel/orin1_oos/onnx/`，
`onnx/manifest.csv` 记 sha256、算子清单、params_eff 及其规则。构造器与 E-11 各家族相同（`torch.manual_seed(0)`）：
- G1：dANN-G = `dann_g.DANNG(4, m, 64, 1)`，impl bmm（m = 12 / 24 / 48）；稠密 MLP = `dann_g.Dense24MLP(h, h)`
  （784-h-h-10，LeakyReLU 0.1，与 g1f 公平口径 MLP 同一个类），h = 192 / 384 / 768。
- G2：DH-SSM = `DHSSMExport(DHSSM(n, 4, 2, "cont"), K)`；LRU = `LRUExport(LRUNet(64, 192, 2, 0.1, K=50), 50)`；
  dssmU = `DHSSMExport(DHSSM(192, 4, 2, "spk", grouped=False), 50)`（与 E-11 的 dssmU n128 同为 b4 spk）；
  GRU = `GRUNetD(h)` 整序列（ONNX GRU 融合算子），h = 192 / 384。输入 1 × 1000 × 700。
- G3：AD-cache = `ad_cache._LutAD(W, "rand")`，impl gemm；稠密 = `ad_cache.MLP(W)`，impl gemm（ReLU，与 lut_g3 相同）。
- 偏置：全零的偏置张量按 g1v2 的规则换成 U(±1/√fan_in)（Generator 种子 1），避免 torch.onnx 把相同的零偏置合并成
  Identity（2026-10-19 的导出假象）。导出后断言：无 Identity 节点、无被两个节点共用的初始值、**每张新图的算子清单与
  其 E-11 参照图（features.csv 的 orin1 行）的 `ops` 列逐项相同**（dANN-G ← g1f m8；MLP gemm / conv ← g1f h128
  gemm / conv；DH-SSM K25 / K50 ← dhssm_L2_n128_b4_cont K25 / K50；LRU、dssmU ← E-11 的 n128 / N128 图；GRU ←
  gru_h256__seq；AD-cache ← adc_h256_ffrand__gemm；G3 MLP ← mlp_h256__gemm）。
- CNN c32_64：沿用已计时的 `experiments/T6/lut/onnx/g1f/{C3264}.onnx`（只作分母），在本会话里与新图一起重新计时，
  比值用本会话的值；E-11 的值（0.1180 ms）只用于漂移报告。
- params_eff：与 E-11 同一函数（dANN-G、MLP、DH-SSM、dssmU、LRU、G3 MLP 用各类的 `effective_params()`；AD-cache 用
  `ADNet(W, 4, seed=1, ff).effective_params()`；融合 GRU 用 B_nnz）。单测核对这些函数在 E-11 的参照形状上复现
  features.csv 的 params_eff。
- P2 断言（`predict`）：新图的形状键（家族 + 宽度，去掉训练标记）与 features.csv 中 tier = orin1 的 202 行都不同；
  (K, F, B) 三元组也都不同；新 ONNX 的 sha256 不在 features.csv 中。另查过 `experiments/T6/raw/orin1*/results.jsonl`
  全部标签，这些形状从未在 Orin #1 上计时。

**A2 稠密 MLP 的图形式（P2 "与 O20–O27 的 MLP 相同的图形式"，从 features.csv 核对）。** O20–O27 的 MLP 分母是论文
引用的单元：h128 Gemm、h256 Conv、h512 Conv、h1024 Gemm，即每个宽度在 {{Gemm, 1×1 Conv}} × {{default, nocublas}}
四个单元里 5 次构建中位数最快的那个（h256 两种形式的中位数同为 0.1041 ms，Conv 的构建最小值 0.1029 < 0.1032）。
因此 G1 的 MLP h192 / h384 / h768 两种形式都导出、都计时，分母 = 两种形式 × 两个 flag 中构建中位数最小者；
在 0.1 µs 精度上相等时取构建最小 p50 较小者。单测用 `buildvar_summary.csv` 核对此规则复现 O20–O27 的四个单元。
预测分母（次要结果）与 mapping-blind 的 K 用被选中那个形式的值（`pairs_pre.csv` 两种形式都列出）。
G3 的稠密 MLP 只用 Gemm（P2 写明）；G2 的 LRU / dssmU 只有一种图。

**A3 计时与目标值（P5，与 E-11 对 Orin #1 的规则相同）。** Orin #1（JetPack 5.1 / TensorRT 8.5.2，15 W 模式），
`src/t6/orin/t6_orin.sh batch`（T1r 同步端到端：`t6_sync_bench.py`，预热 ≥ 2 s 且 ≥ 50 次，计时 500 次，每次运行一个
新进程）；构建 `trtexec --fp16 --memPoolSize=workspace:1024`，flag = default 与 nocublas
（`--tacticSources=-CUBLAS,-CUBLAS_LT`）。每张图每个 flag 独立构建 3 次（标签 `b1_ / b2_ / b3_` + 图名，各自一个
engine），每个 engine 3 个进程 × 500 次，轮次交替（先构建全部，再 3 轮，每轮跑遍全部 engine）。
engine p50 = 3 次运行 p50 的中位数；(图, flag) 的值 = 3 次构建的中位数；图的目标值 = 两个 flag 中较小者
（E-11 P4 的"较快那个 flag"；G1 公平口径 MLP 另按 A2 在两种形式间取最快）。每次构建与运行都持有
`flock /tmp/dn_orin_bench.lock`，`REQUIRE_LOCK_HZ={LOCK_HZ}`：运行前 GPU devfreq min = cur = max ≠ 624.75 MHz 的
运行被跳过并记入 `unlocked.txt`。会话开始前只读确认锁频，未锁则不开始；不执行 jetson_clocks / nvpmodel，不 sudo。

**A4 有效性（P6 的落实）。** 一张图作废，若：任一次构建失败或被跳过；任一次运行失败；任一 engine 少于 3 次有效运行；
任一次运行被 `REQUIRE_LOCK_HZ` 跳过、运行前后 devfreq 不是 min = cur = max = 624.75 MHz、或计时期间的频率采样
min / med / max 不全是 624.75 MHz（原因 "clock"）。作废图所在的配对作废（G1 的 MLP 分母：两种形式任一作废即作废）。
不重导出、不换算子、不补测。

**A5 锚点（P2 之外，只报告环境漂移，不进任何判据）。** `dhssm_L2_n128_b4_cont__K50`（lut g2）与
`mlp_h1024__gemm`（lut g3）按同样规则计时，连同 CNN c32_64，与 E-11 目标值相比，报告比值；不改变任何配对或读法。

**A6 读法计算。** `orin1_oos.py verdict` 先校验 `pre.sha256`，再按 P6 / P7 / P8 计算：决定性 |ln r| ≥ ln 1.25；
S1′ 方向 ≥ 80%；S2′ 中位 |ln r̂ − ln r| ≤ ln 1.5；S3′ ≤ P-param、P-MAC、mapping-blind 各自统计量的一半；
决定性 < 4 记"信息不足"。PASS′ = S1′ ∧ S2′ ∧ S3′；DIRECTION′ = S1′ ∧ S3′ ∧ ¬S2′；FAIL′ = ¬S1′ ∨ ¬S3′。
"""


def write_protocol():
    if not research_git():
        raise SystemExit("predict needs the research git history (commit H1); set DEB_RESEARCH_ROOT")
    plan = subprocess.run(["git", "-C", RESEARCH_ROOT, "show", f"{H1}:PLAN.md"], check=True, capture_output=True,
                          text=True).stdout.split("\n")
    ih = next(i for i, l in enumerate(plan) if l.startswith("**BS-2 E-13"))
    i0 = next(i for i in range(ih, len(plan)) if plan[i].startswith("- *P1 目的*"))
    i9 = next(i for i in range(i0, len(plan)) if plan[i].startswith("- *P9 报告*"))
    body = "\n".join(plan[i0:i9 + 1])
    hdr = (f"# E-13 协议：Orin #1 样本外成本模型检验（新形状）\n\n"
           f"P1–P9 照抄自 PLAN.md §3g-12 BS-2，提交 H1（{H1}，2026-10-29 08:29 +0800，PLAN v0.12.27；"
           f"H1 的 PLAN.md 第 {i0 + 1}–{i9 + 1} 行）。性质（同节）：与 E-12 同样是协议之外、判定之后的事后检验；"
           f"E-11 的 v1 判定、72 个比值、`fit_orin1.json` 常数一律不动、不重拟合。\n\n")
    open(f"{OUTD}/PROTOCOL.md", "w").write(hdr + body + "\n" + APPENDIX)
    return i0 + 1, i9 + 1


# ------------------------------------------------------------------------------------- predict
def graph_rows():
    """Features of every P2 graph (+ anchors) recomputed from the ONNX files."""
    import features
    FR = feature_rows()
    man = {r["graph"]: r for r in csv.DictReader(open(f"{ONNXD}/manifest.csv"))}
    out = {}
    for lab, grp, role, fam, spec, ref in GRAPHS:
        p = onnx_file(lab)
        f = features.graph_features(p)
        s = sha_file(p)
        if fam == "existing":
            r = FR[("orin1", lab)]
            assert s == r["sha256"] and all(int(f[k]) == int(float(r[k])) for k in FEAT), lab
            pe, tgt = int(float(r["params_eff"])), float(r["target_ms"])
        else:
            m = man[lab]
            assert s == m["onnx_sha256"], (lab, "sha256 differs from onnx/manifest.csv")
            pe, tgt = int(m["params_eff"]), ""
        out[lab] = dict(graph=lab, file=cited_path(p), onnx_sha256=s, role=role, group=grp, family=fam,
                        **{k: int(f[k]) for k in FEAT}, params_eff=pe, fused_rnn=int(f["fused_rnn"]),
                        ops=ops_string(f), e11_target_ms=tgt)
    return out


def predict():
    if os.path.exists(RAW):
        raise SystemExit(f"{RAW} exists: predict must run before any Orin #1 timing of these graphs")
    os.makedirs(OUTD, exist_ok=True)
    lines = write_protocol()
    consts = load_constants()
    keys, triples, shas = orin1_shapes()
    G = graph_rows()
    for lab, grp, role, fam, spec, ref in GRAPHS:
        if fam == "existing":
            continue
        g = G[lab]
        assert shape_key(lab) not in keys, (lab, "shape key in features.csv orin1")
        assert (g["K"], g["F"], g["B"]) not in triples, (lab, "(K, F, B) equals an orin1 row")
        assert g["onnx_sha256"] not in shas, (lab, "sha256 in features.csv")
    rows = []
    for lab, *_ in GRAPHS:
        g = G[lab]
        g["T_hat_ms"] = "" if g["fused_rnn"] else t_hat(g["K"], g["F"], g["B"], consts)
        rows.append(g)
        print(f"{g['role']:15s} {lab:52s} K={g['K']:6d} F={g['F']:11d} B={g['B']:9d} peff={g['params_eff']:9d} "
              f"T_hat={g['T_hat_ms']}", flush=True)
    with open(f"{OUTD}/graphs_pre.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    prs = []
    for pid, dn, ds, grp in PAIRS:
        n = G[dn]
        mem = dense_members(ds)
        d0 = G[mem[0]]
        row = dict(pair=pid, group=grp, dendritic=dn, dense=ds, dense_graphs="|".join(mem), T_hat_dend_ms=n["T_hat_ms"],
                   P_param=n["params_eff"] / d0["params_eff"], P_MAC=n["F_eff"] / d0["F_eff"])
        for k, gm in enumerate(mem):
            d = G[gm]
            assert d["params_eff"] == d0["params_eff"] and d["F_eff"] == d0["F_eff"]
            tag = "" if len(mem) == 1 else ("_gemm" if gm.endswith("__gemm") else "_conv")
            row[f"T_hat_blind_ms{tag}"] = t_blind(n, d["K"], consts)
            row[f"T_hat_dense_ms{tag}"] = d["T_hat_ms"]
        prs.append(row)
    cols = ["pair", "group", "dendritic", "dense", "dense_graphs", "T_hat_dend_ms", "P_param", "P_MAC",
            "T_hat_blind_ms", "T_hat_dense_ms", "T_hat_blind_ms_gemm", "T_hat_dense_ms_gemm",
            "T_hat_blind_ms_conv", "T_hat_dense_ms_conv"]
    with open(f"{OUTD}/pairs_pre.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        w.writerows(prs)
    with open(f"{OUTD}/pre.sha256", "w") as fo:
        for fn in PRE_FILES:
            fo.write(f"{sha_file(f'{OUTD}/{fn}')}  {fn}\n")
    print("PROTOCOL.md = H1 PLAN.md lines", lines)
    print(open(f"{OUTD}/pre.sha256").read())


# ------------------------------------------------------------------------------------- Orin manifest (after H2)
RD = "dendritic-edge-bench/T6"


def manifest():
    """t6_orin.sh manifest: build k outer, graph inner (the 3 builds of a graph are spread over the build phase)."""
    lines = []
    for k in range(1, NBUILDS + 1):
        for lab, *_ in GRAPHS:
            lines.append(f"~/{RD}/onnx/e13/{os.path.basename(onnx_file(lab))} b{k}_{lab} {','.join(FLAGS)}")
    open(f"{ONNXD}/orin_manifest.txt", "w").write("\n".join(lines) + "\n")
    print(len(lines), "lines ->", f"{ONNXD}/orin_manifest.txt")


# ------------------------------------------------------------------------------------- verdict
def check_pre():
    got = {}
    for line in open(f"{OUTD}/pre.sha256"):
        h, fn = line.split()
        got[fn] = h
        assert sha_file(f"{OUTD}/{fn}") == h, f"{fn} changed after pre.sha256"
    assert set(got) == set(PRE_FILES)
    return got


def run_clock_ok(j, env_states):
    """A4: lock check before, devfreq min=cur=max before and after, all timed samples at LOCK_HZ."""
    return (j.get("gpu_freq_hz_timed_min_med_max") == [LOCK_HZ] * 3 and j.get("gpu_freq_timed_frac_at_max") == 1.0
            and len(env_states) >= 2 and all(s == (LOCK_HZ,) * 3 for s in env_states))


def gpu_states(envf):
    """(cur, min, max) of every env_dump in a run's env file (as src/t6/e7_g1_orin.gpu_states)."""
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


def select_target(cells):
    """cells: [(median_p50, min_p50, name)] -> the fastest (A2 / A3 rule: smallest median at 1e-4 ms, then min)."""
    return min(cells, key=lambda c: (round(c[0], 4), c[1]))


def graph_targets(res, fails, unlocked, envdir):
    """Per graph: target (min over flags of the median over 3 builds of the engine p50 = median of 3 runs) + void."""
    T, bad, timing = {}, {}, []
    for lab, grp, role, fam, spec, ref in GRAPHS:
        why, cells = [], []
        for b in FLAGS:
            per_build = []
            for k in range(1, NBUILDS + 1):
                bl = f"b{k}_{lab}"
                rr = sorted((j for j in res.get((bl, b), [])), key=lambda j: j["run"])
                if any(bl == f[0] and b == f[1] for f in fails):
                    why.append(("build or run failure", f"{bl} {b}"))
                if any(bl in u and f"_{b}_" in u for u in unlocked):
                    why.append(("clock", f"{bl} {b} run skipped by REQUIRE_LOCK_HZ"))
                if len(rr) != NRUNS:
                    why.append(("fewer than 3 valid runs", f"{bl} {b}: {len(rr)} runs"))
                for j in rr:
                    assert j["iters"] == ITERS and j["precision"] == "fp16"
                    base = j["engine"][:-len(".engine")]
                    if not run_clock_ok(j, gpu_states(f"{envdir}/{base}_r{j['run']}.txt")):
                        why.append(("clock", f"{bl} {b} run {j['run']}"))
                if rr:
                    per_build.append(st.median(j["p50_ms"] for j in rr))
            if len(per_build) == NBUILDS:
                cells.append((st.median(per_build), min(per_build), b, per_build))
        row = dict(graph=lab, role=role, group=grp)
        for c in cells:
            row[f"median_{c[2]}"] = c[0]
            row[f"builds_{c[2]}"] = " ".join(f"{x:.4f}" for x in c[3])
            row[f"build_range_{c[2]}"] = max(c[3]) / min(c[3])
        if cells:
            best = select_target([(c[0], c[1], c[2]) for c in cells])
            T[lab] = (best[0], best[1], best[2])
            row.update(target_ms=best[0], flag=best[2])
        if why or len(cells) != len(FLAGS):
            bad[lab] = why or [("fewer than 3 valid runs", "missing cells")]
        row["status"] = "void" if lab in bad else "ok"
        timing.append(row)
    return T, bad, timing


def parse_raw():
    res = {}
    for line in open(f"{RAW}/results.jsonl"):
        j = json.loads(line)
        res.setdefault((j["label"], j["build"]), []).append(j)
    fails = []
    if os.path.exists(f"{RAW}/failures.txt"):
        for line in open(f"{RAW}/failures.txt"):
            m = re.search(r"(?:FAILED|SKIPPED)[^ ]* (?:or SKIPPED \(low memory\) )?(\S+)", line)
            if not m:
                continue
            tok = os.path.basename(m.group(1)).replace(".engine", "")
            for b in FLAGS:
                mm = re.match(rf"(b\d_\S+?)_{b}(?:_[0-9a-f]{{12}})?$", tok)
                if mm:
                    fails.append((mm.group(1), b))
    unlocked = open(f"{RAW}/unlocked.txt").read().splitlines() if os.path.exists(f"{RAW}/unlocked.txt") else []
    return res, fails, unlocked


def commits(hist):
    if hist:
        h2 = subprocess.run(["git", "-C", RESEARCH_ROOT, "log", "-1", "--format=%H", "--", f"{RESEARCH_OUTD}/pre.sha256"],
                            capture_output=True, text=True).stdout.strip()
        return dict(H1=H1, H2=h2 or None)
    return dict(H1=H1, H2=H2_RECORDED, verified=False, note=NO_HISTORY)


def verdict(outd=None):
    shas = check_pre()
    hist = research_git()
    outd = outd or OUTD
    os.makedirs(outd, exist_ok=True)
    pre_g = {r["graph"]: r for r in csv.DictReader(open(f"{OUTD}/graphs_pre.csv"))}
    pre_p = list(csv.DictReader(open(f"{OUTD}/pairs_pre.csv")))
    res, fails, unlocked = parse_raw()
    T, bad, timing = graph_targets(res, fails, unlocked, f"{RAW}/env")
    keys = []
    for r in timing:
        keys += [k for k in r if k not in keys]
    with open(f"{outd}/timing.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        w.writerows(timing)

    sgn = lambda v: (v > 0) - (v < 0)  # noqa: E731
    out, void = [], []
    for p in pre_p:
        dn, ds = p["dendritic"], p["dense"]
        mem = p["dense_graphs"].split("|")
        reasons = [dict(graph=g, category=c, detail=x) for g in [dn] + mem for c, x in bad.get(g, [])]
        row = dict(pair=p["pair"], tier="orin1", dendritic=dn, dense=ds)
        if reasons or dn not in T or any(m not in T for m in mem):
            reasons = reasons or [dict(graph="", category="fewer than 3 valid runs", detail="missing timing")]
            void.append(dict(pair=p["pair"], categories=sorted({x["category"] for x in reasons}), reasons=reasons))
            row.update(status="void")
            out.append(row)
            continue
        sel = select_target([(T[m][0], T[m][1], m) for m in mem])[2]
        tag = "" if len(mem) == 1 else ("_gemm" if sel.endswith("__gemm") else "_conv")
        T_n, T_d = T[dn][0], T[sel][0]
        r = T_n / T_d
        rhat = float(p["T_hat_dend_ms"]) / T_d
        td = p[f"T_hat_dense_ms{tag}"]
        rhat2 = float(p["T_hat_dend_ms"]) / float(td) if td else None
        p_param, p_mac, p_blind = float(p["P_param"]), float(p["P_MAC"]), float(p[f"T_hat_blind_ms{tag}"]) / T_d
        row.update(T_dendritic_ms=T_n, T_dense_ms=T_d, dense_source=f"orin1_oos/timing.csv {sel} ({T[sel][2]})", r=r,
                   ln_r=math.log(r), decisive=int(abs(math.log(r)) >= LN125), rhat=rhat,
                   rhat_preddenom="" if rhat2 is None else rhat2, rhat_P_param=p_param, rhat_P_MAC=p_mac,
                   rhat_blind=p_blind)
        for t, v in (("model", rhat), ("P_param", p_param), ("P_MAC", p_mac), ("blind", p_blind)):
            row[f"dir_ok_{t}"] = int(sgn(math.log(v)) == sgn(math.log(r)))
            row[f"abs_lnerr_{t}"] = abs(math.log(v) - math.log(r))
        row["dir_ok_preddenom"] = "" if rhat2 is None else int(sgn(math.log(rhat2)) == sgn(math.log(r)))
        row["abs_lnerr_preddenom"] = "" if rhat2 is None else abs(math.log(rhat2) - math.log(r))
        row["status"] = "ok"
        out.append(row)
    cols = list(next(csv.DictReader(open(f"{CM}/pairs_pred.csv"))).keys()) + ["status"]
    with open(f"{outd}/pairs_pred.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    V = reading(out)
    tie = [r for r in out if r["status"] == "ok" and not r["decisive"]]
    pdn = [r for r in out if r["status"] == "ok" and r["decisive"] and r["rhat_preddenom"] != ""]
    absd = []
    for lab, grp, role, *_ in GRAPHS:
        if role == "main_dendritic" and lab in T and lab not in bad:
            e = math.log(float(pre_g[lab]["T_hat_ms"]) / T[lab][0])
            absd.append(dict(graph=lab, T_hat_ms=float(pre_g[lab]["T_hat_ms"]), T_meas_ms=T[lab][0],
                             pred_over_meas=math.exp(e), abs_ln_err=abs(e)))
    groups = {}
    for grp in ("G1", "G2", "G3"):
        gr = [r for r in out if r["status"] == "ok" and PG_GROUP[r["pair"]] == grp]
        gd = [r for r in gr if r["decisive"]]
        groups[grp] = dict(n_ok=len(gr), n_decisive=len(gd), dir_correct=sum(r["dir_ok_model"] for r in gd),
                           median_abs_lnerr=(st.median(r["abs_lnerr_model"] for r in gd) if gd else None),
                           median_factor=(math.exp(st.median(r["abs_lnerr_model"] for r in gd)) if gd else None),
                           misses=[r["pair"] for r in gd if not r["dir_ok_model"]])
    drift = {g: dict(session_ms=T[g][0], e11_target_ms=float(pre_g[g]["e11_target_ms"]),
                     ratio=T[g][0] / float(pre_g[g]["e11_target_ms"]))
             for g, *_ in GRAPHS if pre_g[g]["e11_target_ms"] and g in T}
    V = dict(experiment="E-13 post-hoc Orin #1 out-of-sample check, new shapes (outside the pre-specified E-11 protocol)",
             commits=commits(hist), pre_sha256=shas, **V, n_void=len(void), void=void,
             n_void_by_category={c: sum(c in v["categories"] for v in void)
                                 for c in ("clock", "build or run failure", "fewer than 3 valid runs")},
             secondary=dict(
                 tie_band=dict(n=len(tie), n_rhat_in_band=sum(abs(math.log(r["rhat"])) < LN125 for r in tie),
                               pairs=[dict(pair=r["pair"], r=r["r"], rhat=r["rhat"]) for r in tie]),
                 predicted_denominator=dict(n=len(pdn), dir_correct=sum(r["dir_ok_preddenom"] for r in pdn),
                                            median_abs_lnerr=(st.median(r["abs_lnerr_preddenom"] for r in pdn)
                                                              if pdn else None),
                                            note="fused GRU denominators have no prediction"),
                 absolute_error_dendritic=dict(n=len(absd),
                                               median_factor=(math.exp(st.median(x["abs_ln_err"] for x in absd))
                                                              if absd else None),
                                               max_factor=(math.exp(max(x["abs_ln_err"] for x in absd)) if absd else None),
                                               graphs=absd),
                 by_group=groups, anchor_drift=drift),
             constants=dict(zip(("a_ms", "kappa_ms_per_node", "inv_phi_ms_per_MAC", "inv_beta_ms_per_weight"),
                                load_constants())))
    json.dump(V, open(f"{outd}/verdict.json", "w"), indent=1, ensure_ascii=False)
    for r in out:
        if r["status"] == "ok":
            print(f"{r['pair']:5s} r={r['r']:8.4f} rhat={r['rhat']:8.4f} Pp={r['rhat_P_param']:8.4f} "
                  f"Pm={r['rhat_P_MAC']:8.4f} bl={r['rhat_blind']:8.4f} dec={r['decisive']} dir={r['dir_ok_model']}")
        else:
            print(r["pair"], "void")
    print(json.dumps({k: V[k] for k in ("reading", "n_ok", "n_void", "n_decisive", "S1", "S2", "S3")}, indent=1,
                     ensure_ascii=False))


def reading(out):
    """P7 / P8 on the pair rows (status ok)."""
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
        rd = "信息不足 (insufficient: < 4 decisive ratios)"
    elif s1 and s2 and s3:
        rd = "PASS′"
    elif s1 and s3:
        rd = "DIRECTION′"
    else:
        rd = "FAIL′"
    return dict(reading=rd, n_pairs=len(out), n_ok=len(ok), n_decisive=len(dec),
                S1=dict(passed=s1, direction_correct=n_dir, n_decisive=len(dec),
                        frac=(n_dir / len(dec) if dec else None), threshold=0.8),
                S2=dict(passed=s2, median_abs_lnerr=stat["model"],
                        median_factor=(math.exp(stat["model"]) if dec else None), threshold=LN15, threshold_factor=1.5),
                S3=dict(passed=s3, proxy_statistic=stat,
                        proxy_factor={t: (math.exp(v) if v is not None else None) for t, v in stat.items()},
                        proxy_dir_correct={t: sum(r[f"dir_ok_{t}"] for r in dec) for t in tags},
                        model_le_half_of=s3d))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"export": export, "predict": predict, "manifest": manifest, "verdict": verdict}.get(cmd)
    if fn is None:
        raise SystemExit("usage: orin1_oos.py {export|predict|manifest|verdict} [--out DIR]")
    if cmd == "verdict":
        verdict(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None)
    else:
        fn()
