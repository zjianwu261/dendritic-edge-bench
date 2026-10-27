照抄自 PLAN.md §3g-10 B5-A1，提交 H1 / H2 之后

*A1-0 性质与生效方式。*本修正案在 H2（29e3cefc）之后、任何树莓派计时之前写定；写定时不存在任何计时数据（`pi_oos/raw/` 不存在），不以任何结果为据。它只补 P6 的 throttled 歧义、加一道计时前的环境门、改一处 P2 的计数笔误；P2 的图清单、P3 的配对、P4 的模型与预测、P7–P9 的判据和读法一字不改。生效步骤：dn-exec 把本块（从 *A1-0* 到 *A1-4* 末尾）**逐字复制**为 `experiments/costmodel/pi_oos/PROTOCOL_A1.md`（首行加"照抄自 PLAN.md §3g-10 B5-A1，提交 H1 / H2 之后"），**不改** `PROTOCOL.md`、`graphs_pre.csv`、`pairs_pre.csv`（`pre.sha256` 保持有效）；主会话提交，哈希记为 H3，并把 `PROTOCOL_A1.md` 的 sha256 写进 PROGRESS。**H3 之前不得开始计时。**`verdict.json` 加 `protocol_amendment: {file, sha256, commit}`；S7.7 的设计信息里加一句 "a timing-environment amendment (PROTOCOL\_A1.md, commit H3) was committed after the predictions and before timing"。

*A1-1 throttled 判据（替代 P6 第二条中"3 轮中任一轮 `throttled` 非 0x0"的部分；其余 P6 不变）。*`vcgencmd get_throttled` 的低 4 位（bit 0–3：当前欠压、当前限频、当前降频、当前软温限）是当前状态，bit 16–19 是开机以来的累计位。
- 若树莓派**没有重启**（开机时间延续自核对时的 26 天）：会话开始时先读一次 `get_throttled` 作为基准值并写进 `env_log.txt` 首行和 PROGRESS（核对时为 0x50000，即 bit 16、18 已置位）。一张图有效的条件是：该图 3 轮每轮运行前后的读数都满足 **(i) bit 0–3 全为 0，且 (ii) bit 16–19 与基准值完全相同**（会话中出现任何新的累计位，视同该次读数发生了欠压 / 降频）。两条任一不满足，该图所在配对作废（计入 P9 作废数，原因写 "throttled"），不重测。
- 若树莓派在计时前**重启过**（`uptime -s` 晚于核对时间 2026-10-27 08:14）：P6 原文的严格规则照用，任一读数非 0x0 即作废。重启后必须重做步骤 3a 的全部核对，并确认 `garbage.service` 仍为 active、仍在单独一核上（T1 状态）、governor 仍为 `ondemand`；任何一项与 T1 不同，记进 PROGRESS 和 S7.7，但不中止。
- 报告：S7.7 写明走的是哪条规则、基准值是多少、会话中累计位有无变化。

*A1-2 计时前环境门（新增；P5 的补充）。*核对时的后台负载（root 进程 `/tmp/ShellCrash/CrashCore`，PID 1091，占 2.5–2.9 核；load 13.6–14.5，空闲 0–0.8%）与 T1 时（load ≈ 1.3，空闲 ≈ 50%）不可比。处理它需要 root，由**用户**决定重启该服务、计时期间停掉它或重启树莓派；dn-exec 不 sudo、不碰该进程。只有下面的门通过才能开始计时：
- **轮次门（每轮 k = 1..3 开始前各做一次，前一轮结束后至少等 90 s 再做）**：在树莓派上连续采样 60 s（`vmstat 1 60`，取 `id` 列的平均；采样结束时再读 `cat /proc/loadavg`），要求 **1 分钟 load ≤ 2.0 且平均空闲 ≥ 40%**；同时 `ps -eo pcpu,psr,pid,user,comm --sort=-pcpu | head -8` 中，除 `garbage.service` 的进程（core 1）之外，**没有任何进程占 CPU > 20%**（跑在 core 0、2、3 上的尤其如此）。三个输出原样追加到 `gate_log.txt`，并写明通过 / 未通过。
- **每图门（每次运行前后，与 P5 的温度、频率、throttled 一起记）**：`cat /proc/loadavg` 和同一条 `ps` 命令追加到 `env_log.txt`。运行前要求除 `garbage.service` 和本次 `bench_ort.py` 外没有进程 > 20% CPU（1 分钟 load 在这里只记录不判，因为前一张图的 3 线程负载要 1 分钟才衰减）。
- **门未通过时**：
  - 轮次门未通过：不开始该轮；每 10 分钟重做一次轮次门，累计 60 分钟仍不过则结束本次会话，拉回已有日志，PROGRESS 记"环境门未通过，等待用户处理"，交回主会话。已完成的轮次仍然有效。
  - 每图门在一轮中途未通过（出现非基准进程 > 20% CPU）：**立即中止该轮**，该轮已跑过的图的结果保留在 `raw/results_aborted.jsonl` 中备查但**不进 verdict**；等轮次门重新通过后，该轮从第一张图起整轮重跑。这不是按数值挑选重测：触发条件只看环境日志，不看延迟，且规则在计时前写定。
  - 会话（可跨日恢复，但 A1-1 的 throttled 规则按恢复时的重启状态重新确定并记录）结束时仍不足 3 个完整轮次的图，所在配对作废，原因写 "fewer than 3 valid rounds"。
- 锚点图照 P2 末、3e 的规定每轮跟着跑，漂移判断不因本修正案改变。

*A1-3 P2 计数笔误。*P2 括号中"tier = pi 的 36 张 T1 图"应为 **32 张**（`features.csv` 中 `tier = pi` 共 32 行）。排除集合的定义（`features.csv` 中 tier = pi 的全部行）不变，图清单不变；PLAN.md 已改，`PROTOCOL.md` 因 sha256 已固定不改，以本条为准。

*A1-4 对 pi_oos.py verdict 的最小改动（dn-exec，计时前做完并随 H3 提交）。*`verdict` 解析 `env_log.txt` 时按 A1-1 判 throttled（无重启：bit 0–3 为 0 且 bit 16–19 等于首行基准；有重启：等于 0x0），按 A1-2 识别 `results_aborted.jsonl` 并跳过，`verdict.json` 的作废清单里区分 "throttled" / "fewer than 3 valid rounds" / "load or run failure"。不改 `pre.sha256` 校验、不改任何统计量。
