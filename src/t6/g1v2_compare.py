"""Old vs new pareto_g1 / latmatch_g1 after the g1v2 MLP LUT fix.  The old tables (pareto_g1.csv.bak_v2,
latmatch_g1.csv.bak_v2, written by pareto.py before lut.py substituted the g1v2 rows) are not distributed; the
comparison itself is in results/T6/g1v2_diff_{pareto,latmatch}.csv.
  python g1v2_compare.py [subdir]     subdir '' (default) or 'fmax'  -> results/T6/<subdir>/g1v2_diff_{pareto,latmatch}.csv"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import T6  # noqa: E402


def rd(p):
    return {r["cfg"]: r for r in csv.DictReader(open(p))}


def main(sub=""):
    d = f"{T6}/{sub}" if sub else T6
    po, pn = rd(f"{d}/pareto_g1.csv.bak_v2"), rd(f"{d}/pareto_g1.csv")
    lo, ln = rd(f"{d}/latmatch_g1.csv.bak_v2"), rd(f"{d}/latmatch_g1.csv")
    out = []
    for c in sorted(set(po) | set(pn)):
        a, b = po.get(c, {}), pn.get(c, {})
        r = {"cfg": c, "family": (a or b)["family"], "acc_mean": (b or a).get("acc_mean"),
             "p50_old": a.get("p50_ms", ""), "p50_new": b.get("p50_ms", ""), "impl_build_old": f"{a.get('impl', '')}/{a.get('build', '')}",
             "impl_build_new": f"{b.get('impl', '')}/{b.get('build', '')}",
             "pareto_family_old": a.get("pareto_family", ""), "pareto_family_new": b.get("pareto_family", ""),
             "pareto_all_old": a.get("pareto_all", ""), "pareto_all_new": b.get("pareto_all", "")}
        r["flag_changed"] = (r["pareto_family_old"], r["pareto_all_old"]) != (r["pareto_family_new"], r["pareto_all_new"])
        out.append(r)
    with open(f"{d}/g1v2_diff_pareto.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("pareto rows with p50 or flag change:")
    for r in out:
        if r["flag_changed"] or r["p50_old"] != r["p50_new"]:
            print(" ", r)
    lm = []
    K = ["p50_ms", "best_baseline_within_lat", "baseline_p50_ms", "delta_acc_pp", "fastest_baseline_same_acc", "its_p50_ms", "speedup_vs_it"]
    for c in sorted(set(lo) | set(ln), key=lambda c: float((ln.get(c) or lo.get(c))["p50_ms"])):
        a, b = lo.get(c, {}), ln.get(c, {})
        r = {"cfg": c, "in_old": bool(a), "in_new": bool(b)}
        for k in K:
            r[k + "_old"], r[k + "_new"] = a.get(k, ""), b.get(k, "")
        lm.append(r)
    with open(f"{d}/g1v2_diff_latmatch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lm[0])); w.writeheader(); w.writerows(lm)
    print("latmatch old -> new:")
    for r in lm:
        print(f"  {r['cfg']}: base {r['best_baseline_within_lat_old']}->{r['best_baseline_within_lat_new']} "
              f"dacc {r['delta_acc_pp_old']}->{r['delta_acc_pp_new']} | fastest {r['fastest_baseline_same_acc_old']}->"
              f"{r['fastest_baseline_same_acc_new']} speedup {r['speedup_vs_it_old']}->{r['speedup_vs_it_new']}"
              f"{'' if r['in_old'] and r['in_new'] else ' (in_old %s in_new %s)' % (r['in_old'], r['in_new'])}")
    for x, y in (("dann24_D512_s128", "mlp_h512"), ("dannG_p4_m16_s128_sg4_sp24", "mlp_h256")):
        f = lambda P: float(P[x]["p50_ms"]) / float(P[y]["p50_ms"])
        print(f"ratio {y}/{x} (speed-up): old {1 / f(po):.3f} new {1 / f(pn):.3f}; {x}/{y} old {f(po):.3f} new {f(pn):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
