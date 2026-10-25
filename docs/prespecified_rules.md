# Pre-specified rules: verbatim plan excerpts

Verbatim excerpts (in Chinese, as written) from the authors' internal project plan
(`PLAN.md`), covering the stop-loss rule of the dendritic-adapter test (plan v0.10,
2026-10-03) and the fixed-mechanism rerun under the same rule (plan v0.12.2, 2026-10-10).
The rest of the plan is internal and not released. The dates are the authors' own
records; the rules were not registered publicly (see the paper's Data availability).

Summary (English): S1 — the context-gated dendritic adapter with the context ID given
must reach a final mean error at least 5% lower than the parameter-matched routed LoRA
bank, and be lower in at least 2 of 3 seeds (v3 main order, test set). S2 — inferred
context accuracy at least 85%. S1 failing triggers the stop-loss (switch to the analysis
paper). The v0.12.2 rerun after the R-044 fix uses the same S1/S2 rules.

## Plan v0.10 (2026-10-03)

- **v0.10 规划决定（2026-10-03，阶段总结后，用户选定"H5 最后一试"）**：
  - **现状**：v2 基准种子 0，dend2 是参数高效类方法里最好的（最终 2.64，BWT ≈ 0；LoRA 3.27，LoRA r50 2.99），但学完当前任务时的误差 2.65，ft 约 1.7，可塑性不够；经验回放最终 1.78、BWT 0.006，全面领先。上下文推断 54%，新上下文检出 17%。给定 ID 好于随机上下文（2.81 vs 3.25）。验证集收益检查（门限 15%，事先写定）只有 5 个任务入选。
  - **止损规则（事先写定，结果出来后不改）**：在 v3 主顺序、3 个种子、测试集上：
    - (S1) dend2 给定 ID 的最终平均误差比参数匹配的 LoRA 库（给定 ID）低 ≥ 5%，并且 3 个种子中 ≥ 2 个都更低；
    - (S2) dend2 自动推断的上下文准确率 ≥ 85%（H5 原定 90%，这里作为最低线）。
    - S1 不过 → **止损**：树突机制没有超出"按上下文路由的适配器"，主线转为分析型论文（工作标题"When do dendritic networks pay off on edge robots?"），用 H1 / H7（公平口径）/ H5 / H8 的完整结果。S1 过、S2 不过 → 论文只报给定 ID 的设置，自动推断作为局限。两者都过 → 按原主线继续，下一步 T4b。
    - 不论结果如何，经验回放都作为参照报告；论文的卖点只能落在"不存数据"这个约束下的比较。
  - **H7 收尾**：修 `g2v_lru_*`（optimizer 参数组重复："some parameters appear in more than one parameter group"），重跑 12 个任务。PROGRESS 中按公平口径重写 H7 结论：说话人划分 + 调参后 DH-SSM 88.0±2 vs GRU h768 87.1±1 vs 非树突对角 SSM 87.1；延迟优势来自并行扫描 SSM，不是树突。按验证集选的最好模型和 Orin 延迟一起给。

## Plan v0.12.2 (2026-10-10)

> 变更（v0.12.1 → v0.12.2）：**第 4 轮评审阻断 R-044**：dend2 自 v2 起的所有运行中，树突段 0 因标准化后原型≈0、初始化除以 |z|² 而被放大约 10⁶ 倍，之后每个上下文都选中它，段 1–N 从未训练（own_seg_frac_out：任务 0 为 1.0，其余全是 0.0，已由主会话核对 4 个文件）。v3 止损判定作为事先登记的程序结果照原样报告，但它检验的是机制没有生效的实现。→ 修复后按**同一套 S1 / S2 规则**重跑（补充实验，事先写定），同时处理 R-045（训练模式统一）和 R-046（适配器类 lr 网格扩到 1e-3）。主线（论文 A）不变；如果修复后 S1 通过，是否回到 H5 由用户决定。

- **v0.12.2 规划决定（2026-10-10，R-044）**：
  1. **修复**：树突段初始化加 ε，标准化基准改为全部已见上下文的原型（或第一个任务的原型样本加正则），并在每个任务结束后断言 own_seg_frac_out ≥ 0.9（给定 ID 时）；训练 / 评测模式在 dend2 和 LoRA 库之间统一（R-045，都用 eval() 下的 dropout=0，或都 train()，写明）。
  2. **lr（R-046）**：dend2 和 LoRA 库都在 {1e-4, 3e-4, 1e-3} 上按种子 0 的验证集选；输出偏置已在 {3e-4, 3e-3} 中选过，保持。
  3. **重跑（补充实验 v3fix，prio 160，排在 A4 / A8 前面，约 15–20 个任务）**：dend2 给定 ID / 推断 / 随机上下文、LoRA 库 r10 给定 ID / 推断，v3 主顺序，3 个种子（再补种子 3 / 4 到 prio 100）。
  4. **判定（事先写定，与 v0.10 相同）**：S1 = dend2 给定 ID 比参数匹配 LoRA 库低 ≥ 5% 且 ≥ 2/3 个种子更低；S2 = 推断准确率 ≥ 0.85。原判定（v3）与修复后（v3fix）并列报告，都写进论文 B。
  5. **之后**：S1 不过 → 论文 B 的 §VI 结论成立（并且这次检验的是生效的机制）；S1 过 → 报告用户，由用户决定是否回到 H5（论文 A 仍是主线）。论文 B §VI-B、C3、判据 4 和摘要相关句先标"待 R-044"。
