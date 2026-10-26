# E-11 协议：映射感知成本模型（PLAN.md §3g-8 (3) "协议正文"，v0.12.22）

**声明（协议第一段）**：所有测量值（Orin #1、Orin #2、树莓派的延迟，8051 的板上周期）在写本协议之前作者都已看过；事先写定的只是模型形式、特征、划分、判据和读法。本文件写定之后才开始写特征提取和拟合代码；`src/costmodel/fit.py` 启动时校验本文件的 sha256 与 PROGRESS.md 中记录的值一致，不一致就退出。

以下为 PLAN.md §3g-8 (3) 协议正文原文（照抄）。

---

*P1 目的。*检验"部署成本由部署图的映射决定"：用部署图上算出的特征、只在非树突模型上标定常数，能否预测树突/稠密延迟比值，并且比参数量、MAC 数和"不看映射"的预测好。

*P2 GPU / CPU 档的模型（Orin #1、Orin #2、树莓派，各自独立标定）。*
$\hat T = a + \kappa K + F/\phi + B/\beta$，四个常数都 ≥ 0。
- $K$：执行节点数。用 onnxruntime 以 `ORT_ENABLE_BASIC` 优化级别保存优化后的图（常量折叠、去冗余），数节点，不计 Constant / Identity / Shape 类节点；ONNX Loop：外层节点 + 循环次数 × 循环体节点（SHD 循环次数 1000）。
- $F$：部署图实际执行的乘加次数，逐个 MatMul / Gemm / Conv（含分组、批量）按张量形状计算；掩码实现里矩阵中的零照算；Loop 体乘以循环次数；逐元素运算不计。
- $B$：部署图中常量张量（权重）的元素个数，折叠之后计。
- 拟合：最小化 $\sum(\ln\hat T-\ln T)^2$，`scipy.optimize.least_squares`，下界 0，初值取线性空间 NNLS 的解。
- 融合 RNN 算子（ONNX GRU / LSTM）：不在模型的映射类之内，不进标定，也不做预测，只作为比值的实测分母。

*P3 8051 的模型。*板上周期 $= c_0 + c_{fc}F_{fc} + c_{conv}F_{conv} + c_{out}N_{out} + c_g F_{gather} + c_p F_{prng}$。
- 各项由 Python int8 参考实现插桩数出（每类 MAC 和每个重量化输出各计一次），并与生成的 C 内核的循环次数做断言比对。
- $F_{gather}$ 是存储掩码层的非零权重 MAC，$F_{prng}$ 是 PRNG 层的 MAC，$N_{out}$ 是重量化输出的个数。
- $c_0, c_{fc}, c_{conv}, c_{out}$ 用 14 个非树突、板上实测的构建拟合：5 个 MLP、7 个普通 CNN、s4 c2_4、s4 c4_8。$c_g, c_p$ 是稠密模型用不到的项：预测某个 dANN 时，在稠密常数固定的前提下，只用同一掩码类的其余 dANN 拟合（留一法）。
- 目标值：board_cycles；比值用板上 ms。

*P4 目标值。*
- Orin #1：用论文所引的那个值。G1 公平协议的图用 5 次构建的中位数（buildvar）；T6 lut 的图用较快那个 flag 的 p50；T1 的图在有锁频列时用锁频列，否则用 DVFS p50，前提是计时时 GPU 频率三次都是 625 MHz，不满足就剔除。
- 树莓派：T1 的 p50。
- Orin #2：`orin2_builds.csv` 中各图 3 次构建的中位数，较快的 flag，不做频率修正。

*P5 标定集。*各设备上全部被计时的非树突、非融合 RNN 的图，包括 MLP 各宽度和各实现形式、vANN、CNN、GRU Loop、LIF Loop、展开的 LIF、单步模型、LRU、dssmU、dense_D4096。**树突模型一个都不进标定**（8051 的 $c_g, c_p$ 按留一法处理，见 P3）。

*P6 测试集：72 个树突/稠密比值，事先列死。*比值 $r$ = 树突 / 稠密。预测比值 $\hat r = \hat T(\text{树突}) / T_{\text{实测}}(\text{稠密})$，即分母用实测值；用预测值作分母的版本作为次要结果一并报告。
- **Orin #1（33 个）**
  - O1–O3：dANN-LRF 掩码 / MLP h85、vANN、h1024
  - O4–O6：DH-SRNN Loop / GRU-pm 融合、GRU-pm Loop、LIF Loop
  - O7–O8：DH-SRNN 单步 / GRU h23 单步、LIF 单步
  - O9–O12：主动树突朴素实现 k4 / h3592、k8 / h4965、k4 / h3941、k8 / h5230
  - O13–O14：缓存 k4 / h3592、缓存 k8 / h4965
  - O15–O19：AD-cache hW (ffrand, gemm) / MLP hW (gemm)，W = 256、512、1024、2048、4096（lut_g3，各取较快 flag）
  - O20–O25：dANN-G m8 / h128；m32 / h128、h256、h512、h1024；m64 / h1024
  - O26–O27：m32 / CNN c32_64、m64 / CNN c32_64
  - O28–O30：DH-SSM n128 cont K50 / LRU K50、DH-SSM n128 cont K25 / LRU K50、DH-SSM n128 b4 / dssmU
  - O31–O33：DH-SSM n256 / GRU h768 融合、DH-SSM n128 spiking / GRU h64、DH-SSM n128 cont / GRU h256
- **树莓派（14 个）**：P1–P14 与 O1–O14 一一对应（同样的图）。
- **Orin #2（10 个）**
  - Q1–Q2：m32 / h256（Conv nocublas）、m32 / h128
  - Q3–Q4：AD-cache h1024、h2048 / 同宽 MLP
  - Q5–Q7：DH-SSM K25 / GRU h768；DH-SSM K25、K50 / LRU K50
  - Q8–Q10：DH-SRNN Loop / GRU-pm 融合；dANN-LRF 掩码 / MLP h85；DH-SRNN Loop / LIF Loop
- **8051（15 个）**
  - M1–M3：d16_s64 / MLP h48、PRNG d16_s128 / h64、PRNG d20_s128 / h64
  - M4–M15：12 个 dANN 各自 / 四族主规则下的匹配 CNN（`cnnvar/report_board.log` "matched, primary_int8val_ddof1"）

*P7 判据。*
- 决定性比值：$|\ln r| \ge \ln 1.25$，也就是 r 落在 [0.8, 1.25] 之外，超过最大的重建差异 1.22；其余为平局带。
- **S1 方向**：决定性比值中 $\mathrm{sign}\ln\hat r=\mathrm{sign}\ln r$ 的占 ≥ 90%（四台合计）；Orin #1、树莓派、8051 各自 ≥ 80%（只对决定性比值 ≥ 4 个的设备检查）。
- **S2 幅度**：Orin #1、树莓派、8051 的决定性比值上，$|\ln\hat r-\ln r|$ 的中位数 ≤ ln 1.5。Orin #2 处在 DVFS 下，只看方向，不进 S2。
- **S3 胜过代理**：S2 的那个统计量不超过下面三个代理各自统计量的一半。
  - P-param：$\hat r$ = 有效参数比；
  - P-MAC：$\hat r$ = 有效 MAC 比（理想稀疏执行）；
  - mapping-blind：同一组拟合常数，但树突模型的特征按"有效参数的稠密模型"计算（$F$ = 有效 MAC，$B$ = 有效参数，$K$ 取配对稠密模型的 $K$）；8051 上把 $c_{fc}$ 用于全部有效 MAC。
- 次要指标（只报告，不进判定）：
  - 平局带比值中 $|\ln\hat r|<\ln1.25$ 的比例；
  - 各设备上全部树突配置的绝对延迟误差，包括 lut_g1/g2/g3 的扫描配置（如 DH-SSM 分支数 b1/b2/b4/b8）和 8051 全部 dANN；
  - 分母用预测值的版本；
  - 每台设备的拟合常数及其 bootstrap 区间（按标定模型重采样 1,000 次）。

*P8 读法（写死）。*
- PASS = S1 ∧ S2 ∧ S3：模型作为主贡献 C1 的一部分写进正文。
- DIRECTION = S1 ∧ S3 ∧ ¬S2：正文只说能预测方向。
- FAIL = ¬S1 ∨ ¬S3：正文只用一段话如实报告"没有通过"，细节放补充，论文退回 (i) + (iii)。

*P9 不许迭代。*
- 拟合之后，只有特征计数的 bug 可以修。是不是 bug 要用单元测试来证明（手算 3 张图：MLP h128、DH-SRNN Loop、dANN-G m32），每次修复都记进 PROGRESS，并报告修复前后两个版本的判定。
- 模型形式、特征定义、测试集和判据一律不改。想改的话，只能作为"exploratory v2"与 v1 并列报告，判定以 v1 为准。
