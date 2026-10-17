"""dend2 plasticity choice for the v3 benchmark (VALIDATION only): waits for the 4 sweep runs
(kWTA {5%,10%} x width {x1 (h=192), x2 (h=384)}, context id, seed 0, v3 main order, v3 context features,
novelty q = 0.95), picks the config with the lowest validation act_mse_gc final average among the configs with
trainable params <= 5% of the backbone, writes results/T4a/cl/v3_dend2_selection.json, copies the selected
sweep run as the seed-0 id result (dend2_id_..._v3new_s0.json) and appends the remaining v3 dend2 runs
(id seeds 1/2, infer / random seeds 0/1/2) to the task file (prio 220; src/tools/run_tasks.py).
Run in the background: python v3_select_dend2.py [--dry]"""
import json
import os
import shutil
import sys
import time

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CL = f"{ROOT}/results/T4a/cl"
TASKS = os.path.join(os.environ.get("DEB_QUEUE", f"{ROOT}/work/queue"), "tasks.txt")
PY = os.environ.get("PYTHON", sys.executable)
SEQ = "huron_bww8:dark,recon:occl,huron_cory1:dark,huron_bww8:occl,recon:dark"
CFGS = [(k, h) for h in (192, 384) for k in (0.05, 0.10)]
COMMON = f"--mask_frac 0.10 --lr 3e-4 --holdout_val 1 --ctx_feat v3 --novel_q 0.95 --seq {SEQ}"


def sweep_file(k, h):
    return f"{CL}/dend2_id_k{int(round(k * 100)):02d}_mf10_v3sel_h{h}_s0.json"


def main(dry=False):
    while not all(os.path.exists(sweep_file(k, h)) for k, h in CFGS):
        time.sleep(120)
    rows = []
    for k, h in CFGS:
        d = json.load(open(sweep_file(k, h)))
        rows.append({"kwta": k, "h": h, "file": os.path.basename(sweep_file(k, h)), "n_trainable": d["n_trainable"],
                     "trainable_frac": d["trainable_frac"], "eligible": d["trainable_frac"] <= 0.05,
                     "val_act_mse_gc_final_avg": d["summary_val"]["act_mse_gc"]["final_avg"],
                     "val_act_mse_gc_bwt": d["summary_val"]["act_mse_gc"]["bwt"],
                     "val_act_mse_gc_diag_avg": d["summary_val"]["act_mse_gc"]["diag_avg"]})
    el = [r for r in rows if r["eligible"]]
    best = min(el, key=lambda r: r["val_act_mse_gc_final_avg"])
    best_all = min(rows, key=lambda r: r["val_act_mse_gc_final_avg"])
    kk, h = best["kwta"], best["h"]
    final_tag = f"_v3new_h{h}"
    sel = {"rule": "lowest VALIDATION act_mse_gc final_avg (context id, seed 0, v3 main order) among configs with "
                   "trainable params <= 5% of the backbone; test numbers not looked at",
           "candidates": rows, "selected": {"kwta": kk, "h": h}, "best_ignoring_param_cap": {"kwta": best_all["kwta"], "h": best_all["h"]},
           "final_tag": final_tag, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(sel, indent=1))
    if dry:
        return
    json.dump(sel, open(f"{CL}/v3_dend2_selection.json", "w"), indent=1)
    kt = f"k{int(round(kk * 100)):02d}"
    dst = f"{CL}/dend2_id_{kt}_mf10{final_tag}_s0.json"
    if not os.path.exists(dst):
        d = json.load(open(sweep_file(kk, h)))
        d["note"] = f"copy of the selection sweep run {os.path.basename(sweep_file(kk, h))} (same config and seed)"
        d["args"]["tag"] = final_tag
        json.dump(d, open(dst, "w"), indent=1)
    lines = [f"# dend2 v3 selected by validation (v3_dend2_selection.json): kWTA {kk}, h {h}"]
    for s in (0, 1, 2):
        for c in ("id", "infer", "random"):
            if s == 0 and c == "id":
                continue
            lines.append(f"t4a_v3new_dend2_{c}_s{s} | 2500 | prio=220 | {PY} {ROOT}/src/t4a/cl_dend2.py --context {c} "
                         f"--kwta {kk} --h {h} {COMMON} --seed {s} --tag {final_tag}")
    # standing backlog (tier 4, prio 100): seeds 3/4 of the selected dend2
    for s in (3, 4):
        for c in ("id", "infer", "random"):
            lines.append(f"t4a_v3new_dend2_{c}_s{s} | 2500 | prio=100 | {PY} {ROOT}/src/t4a/cl_dend2.py --context {c} "
                         f"--kwta {kk} --h {h} {COMMON} --seed {s} --tag {final_tag}")
    # v3 shuffled order (tier 3 seed 0 prio 200, tier 4 seeds 1/2 prio 100)
    shuf = "recon:occl,huron_bww8:occl,huron_bww8:dark,huron_cory1:dark,recon:dark"
    for s in (0, 1, 2):
        for c in ("id", "infer", "random"):
            lines.append(f"t4a_v3shufnew_dend2_{c}_s{s} | 2500 | prio={200 if s == 0 else 100} | {PY} {ROOT}/src/t4a/cl_dend2.py "
                         f"--context {c} --kwta {kk} --h {h} {COMMON.replace(SEQ, shuf)} --seed {s} --tag {final_tag}_shuf")
    with open(TASKS, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("queued", len(lines) - 1)


if __name__ == "__main__":
    main("--dry" in sys.argv)
