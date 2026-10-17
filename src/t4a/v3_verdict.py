"""v3 benchmark summary (--fix: the same rules on the v3fix runs) + pre-registered stop-loss verdict (S1 / S2).

Main metric: act_mse_gc final average over the 5 tasks (TEST split), main order, seeds 0/1/2.
  S1: dend2 (given ID, selected v3 config) final avg <= 0.95 x lora_bank r10 (given ID), and dend2 lower in >= 2 of 3 seeds.
  S2: dend2 (inferred context) context-inference accuracy (test, after the last task) >= 0.85 (mean over seeds).
Output-bias lr: the one with the lower seed-0 VALIDATION final avg (both lrs are run for all seeds).
Writes results/T4a/v3_summary.csv (per method: mean / std / per-seed of test + val metrics;
per-task mean-trajectory reference (v3_ref.py) and adapted / reference ratios, final and diag, test and val) and
results/T4a/v3_verdict.json.  Usage: python v3_verdict.py [--order main|shuf] [--seeds 0 1 2]
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP  # noqa: E402
from v3_ref import get_refs  # noqa: E402

MAIN = ["huron_bww8:dark", "recon:occl", "huron_cory1:dark", "huron_bww8:occl", "recon:dark"]
SHUF = ["recon:occl", "huron_bww8:occl", "huron_bww8:dark", "huron_cory1:dark", "recon:dark"]


def method_of(d, sel_tag, fix_sel=None):
    """fix_sel None: original v3 verdict (v3fix files excluded).  fix_sel = v3fix_selection.json: only v3fix runs at
    the selected lr."""
    ar = d["args"]; m = d.get("method", ar.get("method")); tag = ar.get("tag", "")
    if fix_sel is not None:
        if not tag.startswith("_v3fix"):
            return None
        if m == "dend2" and tag == f"_v3fix_lr{fix_sel['dend2']['lr']:g}":
            return f"dend2fix_{ar['context']}"
        if m == "lora_bank" and tag == f"_v3fix_r10_lr{fix_sel['lora_bank_r10']['lr']:g}":
            return f"lora_bank_r10fix_{ar['context']}"
        return None
    if tag.startswith("_v3fix"):
        return None
    if m == "dend2":
        if tag.startswith("_v3new") or tag.startswith("_v3sel"):
            if sel_tag and not tag.startswith(sel_tag):
                return None
            return f"dend2v3_{ar['context']}"
        return f"dend2old_{ar['context']}"
    if m == "lora_bank":
        return f"lora_bank_r{ar['lora_r']}_{ar['context']}"
    if m == "lora":
        return "lora_r50" if ar.get("lora_r", 8) == 50 else "lora"
    if m == "bias":
        return f"bias_lr{ar['lr']:g}"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="main", choices=["main", "shuf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fix", action="store_true", help="v3fix verdict (same S1 / S2 rules) -> v3fix_summary.csv, "
                                                     "v3fix_verdict.json; the original verdict files are not touched")
    a = ap.parse_args()
    fix_sel = None
    if a.fix:
        fix_sel = json.load(open(os.path.join(EXP, "cl", "v3fix_selection.json")))
    names = MAIN if a.order == "main" else SHUF
    selp = os.path.join(EXP, "cl", "v3_dend2_selection.json")
    sel = json.load(open(selp)) if os.path.exists(selp) else None
    sel_tag = sel["final_tag"] if sel else None
    runs = {}
    zs = {}
    for f in sorted(glob.glob(os.path.join(EXP, "cl", "*_v3*.json"))):
        if "selection" in f:
            continue
        d = json.load(open(f))
        if d.get("names") != names or not d["args"].get("holdout_val") or d["args"]["seed"] not in a.seeds:
            continue
        if "_v3sel" in d["args"].get("tag", ""):
            continue  # sweep runs: the selected one is copied to the _v3new name
        if d["args"].get("tag", "").startswith("_v3fix") and fix_sel is None:
            continue
        m = method_of(d, sel_tag, fix_sel)
        if m is None:
            continue
        runs.setdefault(m, {})[d["args"]["seed"]] = (os.path.basename(f), d)
        z = next((r for r in d["R"] if r["after"] == "zero_shot"), None)
        if z:
            zs[d["args"]["seed"]] = z
    refs = get_refs(names)
    rows = []
    for m, bys in sorted(runs.items()):
        def vals(fn):
            return {s: fn(d) for s, (f, d) in sorted(bys.items())}
        te = vals(lambda d: d["summary"]["act_mse_gc"]["final_avg"])
        bwt = vals(lambda d: d["summary"]["act_mse_gc"]["bwt"])
        diag = vals(lambda d: d["summary"]["act_mse_gc"]["diag_avg"])
        va = vals(lambda d: d.get("summary_val", {}).get("act_mse_gc", {}).get("final_avg"))
        acc = vals(lambda d: d["summary"].get("ctx_infer_acc_final"))
        own = vals(lambda d: d["summary"].get("own_seg_frac_final"))
        anyd = next(iter(bys.values()))[1]
        r = {"method": m, "n_seeds": len(te), "seeds": list(te), "trainable": anyd["n_trainable"],
             "trainable_frac": round(anyd["trainable_frac"], 4), "lr": anyd.get("lr", anyd["args"].get("lr")),
             "test_final_mean": round(st.mean(te.values()), 4), "test_final_std": round(st.stdev(te.values()), 4) if len(te) > 1 else "",
             "test_bwt_mean": round(st.mean(bwt.values()), 4), "test_diag_mean": round(st.mean(diag.values()), 4),
             "val_final_mean": round(st.mean([v for v in va.values() if v is not None]), 4) if any(v is not None for v in va.values()) else "",
             "ctx_infer_acc_mean": round(st.mean([v for v in acc.values() if v is not None]), 4) if any(v is not None for v in acc.values()) else "",
             "own_seg_frac_final_mean": round(st.mean([v for v in own.values() if v is not None]), 4) if any(v is not None for v in own.values()) else "",
             "test_final_per_seed": json.dumps({k: round(v, 4) for k, v in te.items()}),
             "files": json.dumps([f for f, _ in bys.values()])}
        # per-task mean-trajectory reference and "adapted / reference" ratios
        for split, skey in (("test", "summary"), ("val", "summary_val")):
            fin = [d[skey]["act_mse_gc"]["final"] for f, d in bys.values() if skey in d]
            dg = [d[skey]["act_mse_gc"]["diag"] for f, d in bys.values() if skey in d]
            if not fin:
                continue
            fr, dr = [], []
            for j, n in enumerate(names):
                lab = n.replace("huron_", "").replace(":", "_")
                ref = refs[n][split]["act_mse"]
                fm, dm = st.mean(x[j] for x in fin), st.mean(x[j] for x in dg)
                r[f"{split}_ref_{lab}"] = round(ref, 4)
                r[f"{split}_final_{lab}"] = round(fm, 4); r[f"{split}_final_over_ref_{lab}"] = round(fm / ref, 4)
                r[f"{split}_diag_over_ref_{lab}"] = round(dm / ref, 4)
                fr.append(fm / ref); dr.append(dm / ref)
            r[f"{split}_final_over_ref_avg"] = round(st.mean(fr), 4); r[f"{split}_diag_over_ref_avg"] = round(st.mean(dr), 4)
            r[f"{split}_n_tasks_final_ge_ref"] = sum(x >= 1 for x in fr); r[f"{split}_n_tasks_diag_ge_ref"] = sum(x >= 1 for x in dr)
        rows.append(r)
    rr = {"method": "mean_traj_ref", "n_seeds": "", "test_final_mean": round(st.mean(refs[n]["test"]["act_mse"] for n in names), 4),
          "val_final_mean": round(st.mean(refs[n]["val"]["act_mse"] for n in names), 4),
          "files": "v3_mean_traj_ref.json (src/t4a/v3_ref.py: mean training trajectory, train split without val recordings, "
                   "scored on the same first-500 test / val samples)"}
    for split in ("test", "val"):
        for n in names:
            rr[f"{split}_ref_{n.replace('huron_', '').replace(':', '_')}"] = round(refs[n][split]["act_mse"], 4)
    rows.append(rr)
    if zs:
        zm = [st.mean(z["row"][n]["act_mse_gc"] for n in names) for z in zs.values()]
        zr = {"method": "zero_shot", "n_seeds": len(zm), "test_final_mean": round(st.mean(zm), 4)}
        fr = []
        for n in names:
            lab = n.replace("huron_", "").replace(":", "_")
            v = st.mean(z["row"][n]["act_mse_gc"] for z in zs.values()) / refs[n]["test"]["act_mse"]
            zr[f"test_final_over_ref_{lab}"] = round(v, 4); fr.append(v)
        zr["test_final_over_ref_avg"] = round(st.mean(fr), 4); zr["test_n_tasks_final_ge_ref"] = sum(x >= 1 for x in fr)
        rows.append(zr)
    out = os.path.join(EXP, f"v3{'fix' if a.fix else ''}_summary{'' if a.order == 'main' else '_shuf'}.csv")
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["method"], r.get("n_seeds"), r.get("test_final_mean"), r.get("test_final_std", ""), "bwt", r.get("test_bwt_mean", ""),
              "val", r.get("val_final_mean", ""), "acc", r.get("ctx_infer_acc_mean", ""))
    if a.order != "main":
        return
    ver = {"rule": "stop-loss rule (pre-registered 2026-10-03): S1 dend2 id <= 0.95 x lora_bank r10 id (test act_mse_gc final avg, "
                   "3 seeds) and lower in >= 2/3 seeds; S2 dend2 infer ctx acc >= 0.85", "dend2_selection": sel}
    if a.fix:
        ver["rule"] += "  [v3fix: post-hoc supplementary run after the segment-initialisation bug; lr per method selected on seed-0 val]"
        ver["v3fix_selection"] = fix_sel
        d2, lb, d2i = runs.get("dend2fix_id", {}), runs.get("lora_bank_r10fix_id", {}), runs.get("dend2fix_infer", {})
    else:
        d2, lb, d2i = runs.get("dend2v3_id", {}), runs.get("lora_bank_r10_id", {}), runs.get("dend2v3_infer", {})
    common = sorted(set(d2) & set(lb) & set(a.seeds))
    if len(common) == len(a.seeds):
        dv = {s: d2[s][1]["summary"]["act_mse_gc"]["final_avg"] for s in common}
        lv = {s: lb[s][1]["summary"]["act_mse_gc"]["final_avg"] for s in common}
        ratio = st.mean(dv.values()) / st.mean(lv.values())
        wins = sum(dv[s] < lv[s] for s in common)
        ver["S1"] = {"dend2_id": dv, "lora_bank_r10_id": lv, "ratio_of_means": round(ratio, 4), "seed_wins": wins,
                     "pass": bool(ratio <= 0.95 and wins >= 2)}
    else:
        ver["S1"] = {"pending": True, "have_dend2": sorted(d2), "have_lora_bank": sorted(lb)}
    if len(set(d2i) & set(a.seeds)) == len(a.seeds):
        accs = {s: d2i[s][1]["summary"]["ctx_infer_acc_final"] for s in a.seeds}
        ver["S2"] = {"ctx_infer_acc": accs, "mean": round(st.mean(accs.values()), 4), "pass": bool(st.mean(accs.values()) >= 0.85)}
    else:
        ver["S2"] = {"pending": True, "have": sorted(d2i)}
    b = {lr: runs.get(f"bias_lr{lr}", {}).get(0) for lr in ("0.0003", "0.003")}
    if all(b.values()):
        ver["bias_lr_selected_on_seed0_val"] = min(b, key=lambda k: b[k][1]["summary_val"]["act_mse_gc"]["final_avg"])
    json.dump(ver, open(os.path.join(EXP, "v3fix_verdict.json" if a.fix else "v3_verdict.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in ver.items() if k not in ("dend2_selection", "v3fix_selection")}, indent=1))
    other = os.path.join(EXP, "v3_verdict.json" if a.fix else "v3fix_verdict.json")
    if os.path.exists(other):  # side by side
        o = json.load(open(other)); orig, fx = (o, ver) if a.fix else (ver, o)
        print("SIDE BY SIDE  original v3 | v3fix")
        for k in ("S1", "S2"):
            print(k, json.dumps({x: orig.get(k, {}).get(x) for x in ("ratio_of_means", "seed_wins", "mean", "pass")}), "|",
                  json.dumps({x: fx.get(k, {}).get(x) for x in ("ratio_of_means", "seed_wins", "mean", "pass")}))


if __name__ == "__main__":
    main()
