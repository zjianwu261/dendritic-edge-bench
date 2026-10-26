"""v3fix re-run of dend2 and the LoRA bank r10 on the v3 main order.

Fixes (cl_dend2.py / cl_run.py flags): --seg_eps 0.1 (segment init z / (|z|^2 + 0.1)); --std_ref normal (inferer
standardisation from the first task's prototype sample indices on NORMAL-light frames; used ONLY for the
dend2 segment coordinates (--seg_ref normal) -- the context inferer of dend2 and of the LoRA bank keeps the pre-specified
v3 standardisation (--std_ref first), so S2 is judged with the v3 inferer); --seg_orth 1 (orthogonalised segment init with
adaptive gain; needed because eps + std_ref alone did not separate the segments); both methods train in train() mode,
prototypes / inference features /
evaluation in eval() (dend2 --train_mode train; LoRA bank --proto_eval 1); dend2 with the ID given asserts
own_seg_frac_out >= 0.9 for every seen task after every task (--assert_own 0.9).
lr: per method in {1e-4, 3e-4, 1e-3}, chosen on seed-0 VALIDATION final avg act_mse_gc of the given-ID run.
Usage:  python v3fix_select.py --enqueue_select     (6 seed-0 lr runs, prio 160)
        python v3fix_select.py                      (waits, writes results/T4a/cl/v3fix_selection.json, enqueues
                                                     seeds 0/1/2 at prio 160 and 3/4 at prio 100)
"""
import argparse
import json
import os
import sys
import time

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CL = f"{ROOT}/results/T4a/cl"
Q = os.environ.get("DEB_QUEUE", f"{ROOT}/work/queue")  # task file read by src/tools/run_tasks.py
PY = os.environ.get("PYTHON", sys.executable)
SEQ = "huron_bww8:dark,recon:occl,huron_cory1:dark,huron_bww8:occl,recon:dark"
LRS = [1e-4, 3e-4, 1e-3]
D2 = (f"{ROOT}/src/t4a/cl_dend2.py --kwta 0.05 --h 192 --mask_frac 0.10 --holdout_val 1 --ctx_feat v3 --novel_q 0.95 "
      f"--seg_eps 0.1 --seg_orth 1 --std_ref first --seg_ref normal --train_mode train --seq {SEQ}")
LB = (f"{ROOT}/src/t4a/cl_run.py --method lora_bank --lora_r 10 --ctx_feat v3 --novel_q 0.95 --holdout_val 1 "
      f"--std_ref first --proto_eval 1 --seq {SEQ}")
NEED = {"d2": 3200, "lb": 3300}


def d2_job(ctx, lr, s, prio):
    tag = f"_v3fix_lr{lr:g}"
    extra = " --assert_own 0.9" if ctx == "id" else ""
    return (f"t4a_v3fix_dend2_{ctx}_lr{lr:g}_s{s} | {NEED['d2']} | prio={prio} | PYTHONDONTWRITEBYTECODE=1 {PY} {D2} --context {ctx} "
            f"--lr {lr:g} --seed {s} --tag {tag}{extra}"), f"{CL}/dend2_{ctx}_k05_mf10{tag}_s{s}.json"


def lb_job(ctx, lr, s, prio):
    tag = f"_v3fix_r10_lr{lr:g}"
    return (f"t4a_v3fix_lora_bank_r10_{ctx}_lr{lr:g}_s{s} | {NEED['lb']} | prio={prio} | PYTHONDONTWRITEBYTECODE=1 {PY} {LB} "
            f"--context {ctx} --lr {lr:g} --seed {s} --tag {tag}"), f"{CL}/lora_bank_{ctx}{tag}_s{s}.json"


def append(lines, header):
    existing = open(f"{Q}/tasks.txt").read()
    new = [l for l in lines if l.split(" | ")[0] + " " not in existing]
    if new:
        with open(f"{Q}/tasks.txt", "a") as fh:
            fh.write("\n".join([header] + new) + "\n")
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue_select", action="store_true")
    a = ap.parse_args()
    sel_jobs = {"dend2": [d2_job("id", lr, 0, 160) for lr in LRS], "lora_bank_r10": [lb_job("id", lr, 0, 160) for lr in LRS]}
    if a.enqueue_select:
        n = append([j for v in sel_jobs.values() for j, _ in v], "# v3fix: seed-0 lr selection, given ID (src/t4a/v3fix_select.py)")
        print("appended", n); return
    while True:
        miss = [f for v in sel_jobs.values() for _, f in v if not os.path.exists(f)]
        failed = [p for p in os.listdir(f"{Q}/state") if p.startswith("t4a_v3fix_") and p.endswith(".failed")]
        if not miss:
            break
        print(time.strftime("%F %T"), f"waiting for {len(miss)} lr-selection runs" + (f"; FAILED: {failed}" if failed else ""), flush=True)
        time.sleep(300)
    sel = {"rule": "argmin over lr in {1e-4,3e-4,1e-3} of seed-0 summary_val.act_mse_gc.final_avg, given-ID run", "made": time.strftime("%F %T")}
    for m, v in sel_jobs.items():
        sc = {}
        for (job, f), lr in zip(v, LRS):
            sc[f"{lr:g}"] = json.load(open(f))["summary_val"]["act_mse_gc"]["final_avg"]
        sel[m] = {"lr": float(min(sc, key=sc.get)), "val_final_by_lr": sc}
    json.dump(sel, open(f"{CL}/v3fix_selection.json", "w"), indent=1)
    print(json.dumps(sel, indent=1), flush=True)
    l2, l3 = sel["dend2"]["lr"], sel["lora_bank_r10"]["lr"]
    lines = []
    for seeds, prio in (([0, 1, 2], 160), ([3, 4], 100)):
        for s in seeds:
            for ctx in ("id", "infer", "random"):
                if not (ctx == "id" and s == 0):  # seed-0 given-ID run at the selected lr = the selection run
                    lines.append(d2_job(ctx, l2, s, prio)[0])
            for ctx in ("id", "infer"):
                if not (ctx == "id" and s == 0):
                    lines.append(lb_job(ctx, l3, s, prio)[0])
    print("appended", append(lines, "# v3fix main runs (selected lr; appended by src/t4a/v3fix_select.py)"))


if __name__ == "__main__":
    main()
