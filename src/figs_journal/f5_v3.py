"""Fig. 4: v3 continual low-light benchmark - final act_mse_gc (test, mean over 5 tasks after the last task,
main order) per method, horizontal bars = 3-seed mean (seeds 0-2), dots = individual seeds.  Vertical lines: zero-shot
and the per-task mean-trajectory reference.  Lower is better.

Order: ORDER below (full fine-tuning family, PEFT, LoRA bank, dend2 v3, then the post-fix v3fix rows used for the
pre-registered S1/S2 verdict).  Not plotted: dend2old_* (pre-v3 dend2 config) and bias at the non-selected lr 3e-4
(bias lr 3e-3 was selected on seed-0 validation, v3_verdict.json).

Source CSVs: results/T4a/v3_summary.csv, results/T4a/v3fix_summary.csv (test_final_mean, test_final_per_seed)
"""
import csv
import json

import numpy as np

from style import COL1, OI, ROOT, plt, save

ORDER = [("v3", "ft", "full FT", "full"), ("v3", "ewc", "EWC", "full"), ("v3", "replay", "replay", "full"),
         ("v3", "ft_head", "head FT", "peft"), ("v3", "lora", "LoRA r8", "peft"), ("v3", "lora_r50", "LoRA r50", "peft"),
         ("v3", "bias_lr0.003", "bias", "peft"),
         ("v3", "lora_bank_r10_id", "LoRA bank r10, ID", "bank"), ("v3", "lora_bank_r10_infer", "LoRA bank r10, inferred", "bank"),
         ("v3", "lora_bank_r50_id", "LoRA bank r50, ID", "bank"), ("v3", "lora_bank_r50_infer", "LoRA bank r50, inferred", "bank"),
         ("v3", "dend2v3_id", "dend2, ID", "dend"), ("v3", "dend2v3_infer", "dend2, inferred", "dend"),
         ("v3", "dend2v3_random", "dend2, random ctx", "dend"),
         ("fix", "lora_bank_r10fix_id", "LoRA bank r10, ID (fix)", "bank"),
         ("fix", "lora_bank_r10fix_infer", "LoRA bank r10, inferred (fix)", "bank"),
         ("fix", "dend2fix_id", "dend2, ID (fix)", "dend"), ("fix", "dend2fix_infer", "dend2, inferred (fix)", "dend"),
         ("fix", "dend2fix_random", "dend2, random ctx (fix)", "dend")]
COL = {"full": OI["grey"], "peft": OI["sky"], "bank": OI["orange"], "dend": OI["blue"]}


def main():
    T = {"v3": {r["method"]: r for r in csv.DictReader(open(f"{ROOT}/results/T4a/v3_summary.csv"))},
         "fix": {r["method"]: r for r in csv.DictReader(open(f"{ROOT}/results/T4a/v3fix_summary.csv"))}}
    fig, ax = plt.subplots(figsize=(COL1 - 0.95, 3.6))  # tight bbox incl. tick labels <= 3.5 in
    n = len(ORDER); y = np.arange(n)[::-1]
    for yi, (src, m, lab, grp) in zip(y, ORDER):
        r = T[src][m]
        ps = list(json.loads(r["test_final_per_seed"]).values())
        assert len(ps) == 3, (m, ps)
        ax.barh(yi, float(r["test_final_mean"]), 0.7, color=COL[grp], zorder=2)
        ax.scatter(ps, [yi] * 3, s=6, color="k", zorder=3, linewidths=0)
    zs = float(T["v3"]["zero_shot"]["test_final_mean"]); ref = float(T["v3"]["mean_traj_ref"]["test_final_mean"])
    ax.axvline(zs, color=OI["vermillion"], lw=0.8, ls="--", zorder=1)
    ax.axvline(ref, color="k", lw=0.8, ls=":", zorder=1)
    ax.text(zs + 0.04, y[-1] - 0.4, "zero-shot", fontsize=7, rotation=90, ha="left", va="bottom", color=OI["vermillion"])
    ax.text(ref, n - 0.3, "mean traj.", fontsize=7, ha="right", va="bottom")
    ax.axhline(y[13] - 0.5, color="0.5", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([o[2] for o in ORDER], fontsize=7)
    ax.set_ylim(-0.6, n + 0.3)
    ax.set_xlim(0.0, 3.4)  # bars start at 0 (no truncated baseline)
    ax.set_xlabel("final action MSE (test, lower is better)")
    save(fig, "f5-v3.pdf")


if __name__ == "__main__":
    main()
