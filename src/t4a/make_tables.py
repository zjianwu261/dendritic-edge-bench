"""Summaries for T4a: zero-shot table (results/T4a/zero_shot.csv), CL table (results/T4a/cl_summary.csv),
Orin table (results/T4a/orin_latency.csv) from the raw JSON / JSONL files."""
import csv
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP  # noqa: E402


def zero_shot():
    rows = []
    for f in sorted(glob.glob(os.path.join(EXP, "zero_shot", "*.json"))):
        d = json.load(open(f))
        rows.append({"task": d["task"], "light": d["light"], "n": d["n"], "n_masked": d["n_masked"],
                     **{k: round(d[k], 4) for k in ("eps_mse_gc", "eps_mse_uc", "act_mse_gc", "act_cos_gc", "act_mse_uc",
                                                     "final_err_m", "dist_mse")},
                     "ref_act_mse": round(d["ref_mean_traj"]["act_mse"], 4),
                     "ref_final_err_m": round(d["ref_mean_traj"]["final_err_m"], 4)})
    if rows:
        with open(os.path.join(EXP, "zero_shot.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def cl():
    """CL summary. zero-shot reference per (sequence tag, seed) comes from any cl_run.py run with the same tag
    and seed (they all evaluate the untouched backbone first with identical settings)."""
    runs = []
    for f in sorted(glob.glob(os.path.join(EXP, "cl", "*.json"))):
        if "_calib" in f or "smoke" in f or os.path.basename(f).startswith("FAILED_"):
            continue
        d = json.load(open(f)); d["_file"] = os.path.basename(f)[:-5]
        if "R" not in d or "summary" not in d:  # selection files etc.
            continue
        runs.append(d)
    zs = {}
    for d in runs:
        z = next((r["row"] for r in d["R"] if r["after"] == "zero_shot"), None)
        if z:
            zs[(d["args"].get("tag", ""), d["args"]["seed"])] = z
            zs[(tuple(d["names"]), d["args"]["seed"], bool(d["args"].get("holdout_val", 0)))] = z  # v3: same order / seed / split
    rows = []
    for d in runs:
        s = d["summary"]; ar = d["args"]
        meth = d.get("method", ar.get("method"))
        if meth == "dend":
            meth = f"dend_{ar['context']}"
        if meth == "dend2":
            meth = f"dend2_{ar['context']}_k{int(round(ar['kwta'] * 100)):02d}" + ("" if ar.get("gradmask", 1) else "_nomask")
            if ar.get("h", 192) != 192:
                meth += f"_h{ar['h']}"
            if ar.get("ctx_feat", "v2") != "v2":
                meth += f"_ctx{ar['ctx_feat']}"
        if meth == "lora_bank":
            meth = f"lora_bank_r{ar['lora_r']}_{ar['context']}"
        elif meth == "lora" and ar.get("lora_r", 8) != 8:
            meth = f"lora_r{ar['lora_r']}"
        r = {"run": d["_file"], "seq_tag": ar.get("tag", ""), "method": meth, "seed": ar["seed"], "lr": d.get("lr", ar.get("lr")),
             "trainable": d["n_trainable"], "trainable_frac": round(d["trainable_frac"], 4),
             "train_s_per_task": round(st.mean(d["train_s"]), 1), "train_s_list": d["train_s"], "peak_MB": max(d["peak_MB"])}
        z = zs.get((ar.get("tag", ""), ar["seed"])) or zs.get((tuple(d["names"]), ar["seed"], bool(ar.get("holdout_val", 0))))
        for met in ("act_mse_gc", "final_err_m", "eps_mse_gc"):
            r[f"{met}_final_avg"] = round(s[met]["final_avg"], 4); r[f"{met}_diag_avg"] = round(s[met]["diag_avg"], 4)
            r[f"{met}_bwt"] = round(s[met]["bwt"], 4)
            if z:
                zavg = st.mean(z[n][met] for n in d["names"])
                r[f"{met}_zs_avg"] = round(zavg, 4)
                r[f"{met}_gain_vs_zs_pct"] = round(100 * (zavg - s[met]["final_avg"]) / zavg, 2)
        # mean-trajectory reference (v3_ref.py; held-out-val runs with eval_n 500 only)
        if ar.get("holdout_val") and ar.get("eval_n", 500) == 500 and "final" in s["act_mse_gc"]:
            from v3_ref import get_refs
            refs = get_refs(d["names"])
            fr = [f / refs[n]["test"]["act_mse"] for f, n in zip(s["act_mse_gc"]["final"], d["names"])]
            dr = [f / refs[n]["test"]["act_mse"] for f, n in zip(s["act_mse_gc"]["diag"], d["names"])]
            r["act_mse_gc_ref_avg"] = round(st.mean(refs[n]["test"]["act_mse"] for n in d["names"]), 4)
            r["act_mse_gc_final_over_ref_avg"] = round(st.mean(fr), 4); r["act_mse_gc_diag_over_ref_avg"] = round(st.mean(dr), 4)
            r["act_mse_gc_final_over_ref_per_task"] = json.dumps({n: round(x, 4) for n, x in zip(d["names"], fr)})
            r["n_tasks_final_ge_ref"] = sum(x >= 1 for x in fr)
        r["ctx_infer_acc"] = round(s.get("ctx_infer_acc_final", float("nan")), 4)
        sv = d.get("summary_val")
        if sv:
            r["val_act_mse_gc_final_avg"] = round(sv["act_mse_gc"]["final_avg"], 4); r["val_act_mse_gc_bwt"] = round(sv["act_mse_gc"]["bwt"], 4)
            r["val_ctx_infer_acc"] = round(sv.get("ctx_infer_acc_final", float("nan")), 4)
        r["novel_frac_at_arrival_mean"] = round(st.mean(s["novel_frac_at_arrival"]), 4) if s.get("novel_frac_at_arrival") else ""
        r["false_new_final"] = s.get("false_new_final", "")
        r["winners_jaccard_out"] = (s.get("winners_jaccard_mean") or {}).get("out", "")
        # own-segment fraction (dend2) after the last task, per task, and at each task's own step
        if meth.startswith("dend2"):
            last = d["R"][-1]["row"]
            r["own_seg_frac_final"] = round(s.get("own_seg_frac_final", float("nan")), 4)
            r["own_seg_frac_final_per_task"] = json.dumps({k: round(v["own_seg_frac_out"], 4) for k, v in last.items() if v and "own_seg_frac_out" in v})
            r["own_seg_frac_diag_per_task"] = json.dumps({n: round(R["row"][n]["own_seg_frac_out"], 4) for n, R in zip(d["names"], d["R"])
                                                          if R["row"].get(n) and "own_seg_frac_out" in R["row"][n]})
            r["seg_fix"] = json.dumps({k: ar.get(k) for k in ("seg_eps", "std_ref", "train_mode")})
        rows.append(r)
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(EXP, "cl_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows


def orin():
    rows = []
    for f in glob.glob(os.path.join(EXP, "raw", "orin1", "*.jsonl")):
        for ln in open(f):
            d = json.loads(ln)
            if "error" in d:
                continue
            rows.append(d)
    groups = {}
    for d in rows:
        groups.setdefault((d["label"], d.get("build", "")), []).append(d)
    out = []
    for (lab, b), ds in sorted(groups.items()):
        k = "total_p50_ms" if "total_p50_ms" in ds[0] else "p50_ms"
        p50 = [d[k] for d in ds]
        r = {"label": lab, "build": b, "runs": len(ds), "p50_ms_median": round(st.median(p50), 3),
             "p50_ms_min": min(p50), "p50_ms_max": max(p50),
             "hz": round(1000 / st.median(p50), 2),
             "vm_hwm_MB_max": max(d.get("vm_hwm_MB", 0) for d in ds),
             "mem_avail_drop_MB_max": max(d.get("mem_avail_drop_MB", 0) for d in ds)}
        for extra in ("enc_p50_ms", "unet_step_p50_ms", "host_step_p50_ms", "infer_only_p50_ms"):
            if extra in ds[0]:
                r[extra] = round(st.median(d[extra] for d in ds), 3)
        out.append(r)
    if out:
        keys = sorted({k for r in out for k in r}, key=lambda k: list(out[0]).index(k) if k in out[0] else 99)
        with open(os.path.join(EXP, "orin_latency.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    return out


if __name__ == "__main__":
    for fn in (zero_shot, cl, orin):
        for r in fn():
            print(r)
