# E-13 协议：Orin #1 样本外成本模型检验（新形状）

P1–P9 照抄自 PLAN.md §3g-12 BS-2，提交 H1（720e9539c5ccba6277d83b9adb5da1aa36966249，2026-10-29 08:29 +0800，PLAN v0.12.27；H1 的 PLAN.md 第 1636–1648 行）。性质（同节）：与 E-12 同样是协议之外、判定之后的事后检验；E-11 的 v1 判定、72 个比值、`fit_orin1.json` 常数一律不动、不重拟合。

- *P1 目的*：用 E-11 已拟合的 Orin #1 常数预测**从未在 Orin #1 上计时过的新形状**树突图的树突 / 稠密延迟比值，然后计时检验。作者在写定时不知道任何结果。
- *P2 图清单（固定；随机权重；形状不得与 `features.csv` 中 tier = orin1 的任何一行相同，由 `predict` 断言）*：
  - G1：dANN-G p4 m12 / m24 / m48 s64（bmm）；稠密 MLP h192 / h384 / h768（与 O20–O27 的 MLP 相同的图形式，由 dn-exec 从 features.csv 核对后写进 PROTOCOL）；CNN c32_64 已计时，只作分母。
  - G2：DH-SSM L2 n192 b4 cont K25 / K50，n384 b4 cont K25（T = 1000，700 通道）；稠密 LRU L2 H64 N192 K50、dssmU L2 n192 K50、GRU h192 / h384 fused seq。
  - G3：AD-cache ffrand h384 / h768 / h1536 / h3072（gemm）；稠密 MLP 同宽（gemm）。
  - 导出：沿用 `src/t6` 各家族的导出代码，只改宽度；ONNX 放 `experiments/costmodel/orin1_oos/onnx/`，manifest 记 sha256。
- *P3 配对（14 个，写定于任何计时之前）*：PG1–PG3 dANN-G m12 / m24 / m48 对 MLP h192 / h384 / h768；PG4–PG5 m24、m48 对 CNN c32_64；PG6–PG8 DH-SSM n192 K50 / LRU N192 K50，n192 K25 / dssmU n192 K50，n384 K25 / LRU N192 K50；PG9–PG10 DH-SSM n192 K25 / GRU h192 fused，n384 K25 / GRU h384 fused；PG11–PG14 AD-cache hW / MLP hW，W = 384、768、1536、3072。r = 实测 p50 之比；r̂ = T̂(树突) / T_meas(稠密)；分母用预测值的版本作次要结果。
- *P4 模型与代理*：与 E-11 / E-12 完全相同（四项式、`fit_orin1.json["constants"]` 点估计；P-param、P-MAC、mapping-blind 定义同 E-11 P7 S3）。
- *P5 计时*：Orin #1，TensorRT 8.5 FP16，目标值的 flag 规则与 E-11 对 Orin #1 的规则相同（写进 PROTOCOL），每图 3 次构建、每 engine 3 进程 × 500 次、预热 ≥ 2 s、同步端到端 p50，取构建中位数；锁频只读确认、每次运行记录；不改时钟。
- *P6 有效性*：构建或运行失败的图所在配对作废（不重导出、不换算子）；任一运行发现时钟不在锁定值则该图所在配对作废（原因 "clock"）。
- *P7 判据*：与 E-12 P7 相同（决定性 |ln r| ≥ ln 1.25；S1′ 方向 ≥ 80%；S2′ 中位 |ln r̂ − ln r| ≤ ln 1.5；S3′ ≤ 三个代理各自的一半；决定性 < 4 记"信息不足"）。
- *P8 读法*：PASS′ / DIRECTION′ / FAIL′ 同 E-12，不改 v1 判定。
- *P9 报告*：§6 在树莓派段之后加一段 ≤ 90 词，段首 `\textbf{Post-hoc out-of-sample check (Orin~\#1, new shapes).}`，必含 "after the verdict"、"shapes never timed on this device"、"committed before timing"、"outside the pre-specified protocol and does not alter the verdict above"，数字句同 E-12 模板；补充新增 S7.8（设计时间、H1 / H2、sha256、图与配对、环境、14 行表、S1′–S3′、读法、作废清单）；局限 (vii) 末尾按 PASS′ / DIRECTION′ / FAIL′ 三种模板各加一句（FAIL′："…so on Orin~\#1 the constants do not transfer to [failing group]"）；摘要、亮点、投稿信、结论不改；数据可用性加 `experiments/costmodel/orin1_oos/`。表 S21 的"设备 / 软件栈"行与"引擎构建"行按 BS-1 / BS-2 结果更新。


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
- CNN c32_64：沿用已计时的 `experiments/T6/lut/onnx/g1f/g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv.onnx`（只作分母），在本会话里与新图一起重新计时，
  比值用本会话的值；E-11 的值（0.1180 ms）只用于漂移报告。
- params_eff：与 E-11 同一函数（dANN-G、MLP、DH-SSM、dssmU、LRU、G3 MLP 用各类的 `effective_params()`；AD-cache 用
  `ADNet(W, 4, seed=1, ff).effective_params()`；融合 GRU 用 B_nnz）。单测核对这些函数在 E-11 的参照形状上复现
  features.csv 的 params_eff。
- P2 断言（`predict`）：新图的形状键（家族 + 宽度，去掉训练标记）与 features.csv 中 tier = orin1 的 202 行都不同；
  (K, F, B) 三元组也都不同；新 ONNX 的 sha256 不在 features.csv 中。另查过 `experiments/T6/raw/orin1*/results.jsonl`
  全部标签，这些形状从未在 Orin #1 上计时。

**A2 稠密 MLP 的图形式（P2 "与 O20–O27 的 MLP 相同的图形式"，从 features.csv 核对）。** O20–O27 的 MLP 分母是论文
引用的单元：h128 Gemm、h256 Conv、h512 Conv、h1024 Gemm，即每个宽度在 {Gemm, 1×1 Conv} × {default, nocublas}
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
`flock /tmp/dn_orin_bench.lock`，`REQUIRE_LOCK_HZ=624750000`：运行前 GPU devfreq min = cur = max ≠ 624.75 MHz 的
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
