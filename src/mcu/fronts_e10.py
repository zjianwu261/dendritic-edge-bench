"""Paper B 8051 Pareto fronts with the E-10 re-tuned rows.

Merged 8051 table:
  - the 10 E-10 architectures (results/T1m/e10/results.csv): acc = test_int8_mean (3 seeds, sample SD
    test_int8_sd), flash = flash_bytes_e10, cycles = ucsim_cycles (unchanged);
  - every other dANN / MLP row of results/T1m/results.csv (acc_int8, flash_bytes, cycles_ucsim_mc);
  - CNNs c16_32 / c32_64 from results/T1m/cnn/cnn_results.csv.
Deployable = flash <= 65 024 B (63.5 KiB; T1m results.csv `fits`). Fronts are over deployable models only.
Front definition (same as f3_tiers.pareto for F, generalised): a model is on a front if no other deployable model in
the same pool has acc >= and every cost <= with at least one strict. Costs: F = flash; C = ucsim machine cycles;
3 = flash and cycles (three objectives: acc, flash, cycles). Pools: dANN + MLP ("nocnn") and all incl. CNNs ("cnn").
Output: results/T1m/e10/fronts_e10.csv (+ printed member lists, best MLP, Table-1 flash savings).
  python src/mcu/fronts_e10.py [--no-e10] [--e10-only] [--old-cycles]
E-10b (2026-10-25): e10/results.csv now holds every deployable Table-3 row (10 E-10 + 14 E-10b), so the
  merged table is single-protocol. Cycles = ucsim of the E-10/E-10b seed-1 selected builds (ucsim_cycles_e10build)
  unless --old-cycles (first-run builds). --e10-only reproduces the E-10 (10-row) version from
  e10/results_e10only.csv. cnn_results.csv SDs are ddof 1 (statistics.stdev), sd_kind fixed.
"""
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FLASH_MAX = 65024


E10_ONLY = "--e10-only" in sys.argv
OLD_CYC = "--old-cycles" in sys.argv


def table(e10=True):
    fn = "results_e10only.csv" if E10_ONLY else "results.csv"
    e = {r["model"]: r for r in csv.DictReader(open(f"{ROOT}/results/T1m/e10/{fn}"))} if e10 else {}
    out = []
    for r in csv.DictReader(open(f"{ROOT}/results/T1m/results.csv")):
        out.append(dict(model=r["model"], kind=r["kind"], mask_source=r["mask_source"], acc=float(r["acc_int8"]),
                        sd=float(r["acc_int8_std"]), sd_kind="ddof0", flash=int(r["flash_bytes"]),
                        cycles=float(r["cycles_ucsim_mc"]) if r["cycles_ucsim_mc"] else None, src="T1m/results.csv"))
    for r in csv.DictReader(open(f"{ROOT}/results/T1m/cnn/cnn_results.csv")):
        out.append(dict(model=r["model"], kind="cnn", mask_source="cnn", acc=float(r["acc_int8"]),
                        sd=float(r["acc_int8_std"]), sd_kind="ddof1", flash=int(r["flash_bytes"]),
                        cycles=float(r["cycles_ucsim_mc"]), src="T1m/cnn/cnn_results.csv"))
    for o in out:
        if o["model"] in e:
            r = e[o["model"]]
            cyc = r.get("ucsim_cycles_e10build") or ""
            o.update(acc=float(r["test_int8_mean"]), sd=float(r["test_int8_sd"]), sd_kind="ddof1",
                     flash=int(r["flash_bytes_e10"]),
                     cycles=float(r["ucsim_cycles"]) if (OLD_CYC or not cyc) else float(cyc),
                     src=f"T1m/e10/{fn}:{r.get('protocol', 'E-10')}")
    assert not e10 or sum(o["src"].startswith("T1m/e10") for o in out) == (10 if E10_ONLY else 24)
    for o in out:
        o["fits"] = int(o["flash"] <= FLASH_MAX)
    return out


def front(pool, costs):
    def dom(q, p):
        ge = q["acc"] >= p["acc"] and all(q[c] <= p[c] for c in costs)
        st = q["acc"] > p["acc"] or any(q[c] < p[c] for c in costs)
        return ge and st
    return {p["model"] for p in pool if not any(dom(q, p) for q in pool if q is not p)}


def short(m):
    return m.replace("dann_prng_", "PRNG ").replace("dann_", "").replace("mlp_", "mlp ").replace("cnn_", "")


def main():
    e10 = "--no-e10" not in sys.argv
    T = table(e10)
    dep = [o for o in T if o["fits"]]
    pools = {"nocnn": [o for o in dep if o["kind"] != "cnn"], "cnn": dep}
    COST = {"F": ("flash",), "C": ("cycles",), "3": ("flash", "cycles")}
    for pn, pool in pools.items():
        for fn, c in COST.items():
            mem = front(pool, c)
            for o in T:
                o[f"front_{fn}_{pn}"] = int(o["model"] in mem)
    path = f"{ROOT}/results/T1m/e10/fronts_e10{'' if e10 else '_old'}{'only' if E10_ONLY else ''}" \
           f"{'_oldcycles' if OLD_CYC else ''}.csv"
    keys = list(T[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for o in sorted(T, key=lambda o: o["flash"]):
            w.writerow(o)
    print(f"{'E-10' if e10 else 'old'} table -> {path}")
    for pn in pools:
        for fn in COST:
            k = f"front_{fn}_{pn}"
            m = sorted([o for o in T if o[k]], key=lambda o: o["flash" if fn != "C" else "cycles"])
            kinds = {}
            for o in m:
                kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
            print(f"{k}: {len(m)} {kinds}\n   " + ", ".join(f"{short(o['model'])} ({100*o['acc']:.2f})" for o in m))
    mlp = [o for o in dep if o["kind"] == "mlp"]
    b = max(mlp, key=lambda o: o["acc"])
    print(f"most accurate deployable MLP: {b['model']} {100*b['acc']:.2f}% (flash {b['flash']})")
    print("deployable MLPs on any dANN+MLP front:",
          [o["model"] for o in mlp if any(o[f"front_{f}_nocnn"] for f in COST)])
    g = {o["model"]: o for o in T}
    for a, m in (("dann_d16_s64", "mlp_h48"), ("dann_prng_d16_s128", "mlp_h64")):
        s = 1 - g[a]["flash"] / g[m]["flash"]
        print(f"flash saving {a} ({g[a]['flash']}) vs {m} ({g[m]['flash']}): {100*s:.2f}%")


if __name__ == "__main__":
    main()
