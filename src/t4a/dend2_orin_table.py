"""results/T4a/dend2_orin/raw/{enc,pipeline}.jsonl (orin_run6.sh) -> results/T4a/dend2_orin/dend2_latency.csv
(median of the 3 runs' p50, min-max range, overhead vs the base gnd encoder measured in the same session)."""
import collections
import csv
import json
import os
import numpy as np
D = os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results/T4a/dend2_orin")
rows = []
enc = collections.defaultdict(list)
for r in map(json.loads, open(f"{D}/raw/enc.jsonl")):
    enc[r["label"]].append(r)
pipe = collections.defaultdict(list)
for r in map(json.loads, open(f"{D}/raw/pipeline.jsonl")):
    pipe[r["label"]].append(r)
base_e = np.median([r["p50_ms"] for r in enc["nomad_enc_explore_b1_gnd"]])
for lab, rs in enc.items():
    p = [r["p50_ms"] for r in rs]; m = float(np.median(p))
    rows.append({"scope": "encoder", "label": lab, "n_samples": "", "n_runs": len(rs), "p50_ms": round(m, 3),
                 "range_ms": f"{min(p):.3f}-{max(p):.3f}", "hz": round(1000 / m, 2), "overhead_ms": round(m - base_e, 3),
                 "overhead_pct": round(100 * (m - base_e) / base_e, 2),
                 "gpu_frac_at_max": float(np.median([r["gpu_freq_timed_frac_at_max"] for r in rs]))})
for n in (8, 1):
    b = np.median([r["total_p50_ms"] for r in pipe[f"chain_nomad_enc_explore_b1_gnd_n{n}"]])
    for lab, rs in pipe.items():
        if not lab.endswith(f"_n{n}"):
            continue
        p = [r["total_p50_ms"] for r in rs]; m = float(np.median(p))
        rows.append({"scope": "chain enc + 10 x UNet", "label": lab, "n_samples": n, "n_runs": len(rs), "p50_ms": round(m, 3),
                     "range_ms": f"{min(p):.3f}-{max(p):.3f}", "hz": round(1000 / m, 2), "overhead_ms": round(m - b, 3),
                     "overhead_pct": round(100 * (m - b) / b, 2), "gpu_frac_at_max": ""})
with open(f"{D}/dend2_latency.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
