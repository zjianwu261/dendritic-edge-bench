照抄自 PLAN.md §3g-10 B5 (3)，提交 H1（470787a6211d85e59c96350cb372829b6b695ef6，PLAN.md 第 1384–1427 行）。

*P1 目的。*用 E-11 已拟合的树莓派常数（`experiments/costmodel/fit_pi.json`，不重拟合）预测公平口径 G1 / G2 / G3 部署图在树莓派 4B 上的树突/稠密延迟比值，然后计时检验。这些图在写本协议时从未在树莓派上计时过，作者不知道任何结果。本检验在 E-11 协议之外、在 E-11 判定之后设计，是事后（post hoc）附加检验；无论结果如何，E-11 的 v1 判定、72 个比值和常数都不变（E-11 P9）。

*P2 图清单（固定；只用 `experiments/T6/lut/onnx/` 下的文件；`experiments/costmodel/features.csv` 中 tier = pi 的 36 张 T1 图全部排除）。*
- 主集，27 张：
  - G1 公平口径（`onnx/g1f/`）：树突 `g1f_dannG_p4_m8_s64_sg1_lr0.001_wd0_fair_t1__bmm`、`g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm`、`g1f_dannG_p4_m64_s64_sg1_lr0.001_wd0.01_fair_t1__bmm`；稠密 `g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm`、`g1f_mlp_h256_lr0.0003_wd0_fair_t1__gemm`、`g1f_mlp_h512_lr0.0003_wd0.01_fair_t1__gemm`、`g1f_mlp_h1024_lr0.0003_wd0_fair_t1__gemm`、`g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv`。CPU 上 MLP 只取 Gemm 形式（Conv 形式是 TensorRT 的构建技巧，Pi 的标定集全是 Gemm 图）。
  - G2 公平口径（`onnx/g2/`、`onnx/g2x/`）：树突 `dhssm_L2_n128_b4_cont__K50`、`dhssm_L2_n128_b4_cont__K25`、`dhssm_L2_n128_b4_spk__K25`、`dhssm_L2_n256_b4_cont__K25`；稠密 `lru_L2_H64_N128_lr0.001_vs_t1__K50`、`dssmU_L2_n128_b4_spk_vs_t1__K50`、`gru_h768_lr0.003_vs_t1__seq`、`gru_h64__seq`、`gru_h256__seq`（三个融合 GRU 只作分母，不预测，同 E-11 P2）。
  - G3 等宽（`onnx/g3/`）：树突 `adc_h{256,512,1024,2048,4096}_ffrand__gemm`；稠密 `mlp_h{256,512,1024,2048,4096}__gemm`。
- 次集，5 张（只进绝对延迟误差，不进配对）：`g1f_dannG_p4_m16_s64_sg1_lr0.001_wd0.01_fair_t1__bmm`、`g1f_cnn_c8_16_lr0.001_wd0_fair_t1__conv`、`g1f_cnn_c16_32_lr0.001_wd0_fair_t1__conv`、`dhssm_L2_n128_b4_spk__K50`、`dhssm_L2_n256_b4_cont__K50`。主集全部跑完后才跑次集；时间不够可以不跑，但要写明。
- 锚点，3 张（T1 图，不进检验，只查环境漂移）：`experiments/T1/models/{mlp_h85_match_d32_s128, dann_lrf_d32_s128, active_dend_k4_cached}.onnx`。
- 输入：batch = 1；SHD 图 T = 1000、700 通道；Fashion-MNIST 图 1×784；pMNIST 图 1×784 加上下文输入；全部按 `bench_ort.py` 的规则喂随机 / 零输入，与 T1 相同。有动态维度的图用 `--shape` 显式给出上述形状。
- 说明：`g1f_mlp_h{256,512,1024}` 与 T1 已计时的 `mlp_h{256,512,1024}_fmnist` 形状相同（权重不同）；它们只作实测分母，这一点不影响对树突图预测的检验。

*P3 配对（19 个，与 E-11 的 Orin #1 配对 O15–O33 一一对应，写定于看到任何树莓派结果之前）。*比值 r = 树突 / 稠密，实测 p50 之比；预测 $\hat r = \hat T_{pi}(\text{树突}) / T_{meas}(\text{稠密})$；分母用预测值的版本作次要结果。
- PG1–PG8（G1，对应 O20–O27）：m8 / h128；m32 / h128；m32 / h256；m32 / h512；m32 / h1024；m64 / h1024；m32 / CNN c32_64；m64 / CNN c32_64。
- PG9–PG14（G2，对应 O28–O33）：DH-SSM n128 cont K50 / LRU K50；DH-SSM n128 cont K25 / LRU K50；DH-SSM n128 spk K25 / dssmU K50；DH-SSM n256 cont K25 / GRU h768 融合；DH-SSM n128 spk K25 / GRU h64 融合；DH-SSM n128 cont K25 / GRU h256 融合。
- PG15–PG19（G3，对应 O15–O19）：AD-cache hW ffrand (gemm) / MLP hW (gemm)，W = 256、512、1024、2048、4096。

*P4 模型与代理（全部沿用 E-11，不改）。*
- 模型：$\hat T = a + \kappa K + F/\phi + B/\beta$，四个常数取 `fit_pi.json["constants"]` 的点估计；K、F、B 按 E-11 P2 的定义由 `src/costmodel/features.py` 算出，并与 `features.csv` 的同名 orin1 行断言相等。
- 代理：P-param = 有效参数比；P-MAC = 有效 MAC 比；mapping-blind = 同一组 Pi 常数，树突图按"有效规模的稠密模型"计（F = F_eff，B = params_eff，K 取配对稠密图的 K）。定义与 E-11 P7 S3 相同。
- 预测在计时前算出并写入 `graphs_pre.csv`、`pairs_pre.csv`，三文件 sha256 记入 `pre.sha256` 与 PROGRESS；计时后 `verdict` 先校验这三个文件未变。

*P5 计时方法（与 T1 的树莓派口径完全相同）。*树莓派 4B 8 GB，onnxruntime CPUExecutionProvider FP32，`src/bench/bench_ort.py`，`taskset -c 0,2,3`，`--threads 3`，预热 ≥ 50 次且 ≥ 2 s，计时 500 次，同步端到端（`sync_e2e_python`），每张图 3 个独立进程（外层轮次、内层图，交替），p50 取 3 次 p50 的中位数（`src/t1/make_results.py` 的规则）。不改 governor（`ondemand`），不停 `garbage.service`，不 sudo，不装包，不留后台进程。每次运行前后记录温度、ARM 频率、throttled、governor、`garbage.service` 所在核。ORT 版本与 T1 不同时照常计时，但在报告中写明，并以锚点漂移为准判断可比性。

*P6 目标值与有效性。*
- 目标值：每张图的 p50 中位数（ms）。
- 作废：图在 ORT 上加载或运行失败，或 3 轮中任一轮 `throttled` 非 0x0，则该图所在配对作废，报告作废数和原因；不重导出、不换算子、不重测补救。
- 锚点漂移：三张锚点图的 p50 中位数 / T1 的 `latency_p50`，三者都在 [0.90, 1.10] 内记"环境复现"；否则记"环境漂移（最大 x 倍）"，检验照常报告。

*P7 判据（E-11 P7 的单设备版本；阈值不变）。*
- 决定性比值：$|\ln r| \ge \ln 1.25$（r 在 [0.8, 1.25] 之外）；其余为平局带。决定性比值少于 4 个时，本检验记"信息不足"，只报告数字，不作读法。
- S1′ 方向：决定性比值中 $\mathrm{sign}\ln\hat r = \mathrm{sign}\ln r$ 的占 ≥ 80%（E-11 P7 对单台设备的阈值）。
- S2′ 幅度：决定性比值上 $|\ln\hat r - \ln r|$ 的中位数 ≤ ln 1.5。
- S3′ 胜过代理：S2′ 的统计量不超过 P-param、P-MAC、mapping-blind 各自统计量的一半。
- 次要（只报告）：平局带中 $|\ln\hat r| < \ln 1.25$ 的比例；分母用预测值的版本；主集 + 次集全部树突图的绝对延迟误差中位数与最大值；按 G1 / G2 / G3 分组的方向与误差。

*P8 读法（写死）。*PASS′ = S1′ ∧ S2′ ∧ S3′；DIRECTION′ = S1′ ∧ S3′ ∧ ¬S2′；FAIL′ = ¬S1′ ∨ ¬S3′。任何读法都不改变 E-11 的 v1 判定（PASS）、72 个比值或常数；不允许为本检验重拟合、改特征、改配对或改判据，想改只能标为 exploratory 并列报告。

*P9 读法与报告（三种结果都要写，措辞事先定）。*
- 正文 §6（RQ3）在 "Alternative choices for the five ratios … (Supplementary Section~S7)." 之后加一段，≤ 90 词，段首 `\textbf{Post-hoc out-of-sample check (Raspberry~Pi).}`，必须含："after the verdict"、"never timed on the Pi before"、"pairs, predictions and criterion were committed before timing"、"outside the pre-specified protocol and does not alter the verdict above"，然后一句数字：N 个决定性比值中方向对 D 个，模型中位误差 X×，P-param / P-MAC / mapping-blind 为 Y / Z / W×。结果词按读法：PASS′ 写 "held"；DIRECTION′ 写 "the direction held but not the size criterion"；FAIL′ 写 "did not hold"，并加一句失败集中在哪一组（G1 / G2 / G3）。
- 补充新增 `\subsection{Post-hoc out-of-sample test on the Raspberry Pi (outside the pre-specified protocol)}`，放在 S7 末尾（S7.7）：设计时间与提交哈希 H1 / H2、三文件 sha256、图清单与配对、计时环境（ORT 版本、governor、锚点漂移）、一张表（19 行：配对、两图、$\hat T$、实测 T、r、$\hat r$、四个代理、决定性、方向）、S1′–S3′ 数值与读法、作废清单、次要指标；首句写明 "This test was designed on 2026-10-27, after the verdict of Section~S7.5, and is outside the pre-specified protocol; its graphs had never been timed on the Pi."。
- 局限 (vii) 末尾加一句：PASS′ / DIRECTION′ → "A post-hoc check on 19 Pi ratios that were not part of the test set (Section~S7.7) gave [D of N directions, median error X×]; it was designed after the verdict and is reported as supporting evidence only."；FAIL′ → "A post-hoc check on 19 Pi ratios not in the test set (Section~S7.7) did not hold ([D of N directions, median error X×]), so the Pi constants do not transfer to [the failing group] and the pass above should be read with this limit."
- 摘要、亮点、投稿信、结论**不改**（摘要已满 248 词；本检验不是主张的一部分）。
- 发布仓库加 `results/costmodel/pi_oos/` 与 `src/costmodel/pi_oos.py`，数据可用性在成本模型一句后加 "the post-hoc Pi check (Section~S7.7) is in \path{experiments/costmodel/pi_oos/} with its own \path{PROTOCOL.md}, prediction files and their SHA-256 (\path{pre.sha256}), committed before timing"。
