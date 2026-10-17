"""Join results/T6/energy.csv (Orin #1 energy per inference, src/bench/energy_parse.py) with accuracy
(pareto_<task>.csv) -> results/T6/energy_compare.csv.  E_net = (P_active - P_idle) x time / inferences."""
import csv
import os

T6 = os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results/T6")
acc = {}
for t in ("g2", "g3"):
    for r in csv.DictReader(open(f"{T6}/pareto_{t}.csv")):
        acc[r["cfg"]] = r
# energy label -> accuracy cfg
MAP = {"dhssm_L2_n128_b4_spk__K25": "dhssm_L2_n128_b4_spk", "dhssm_L2_n128_b4_cont__K25": "dhssm_L2_n128_b4_cont",
       "gru_h64__seq": "gru_h64_r1", "gru_h256__seq": "gru_h256_r1",
       "g3_adc_h1024_ffrand__gemm": "adc_h1024_k4_ffrand", "g3_adc_h2048_ffrand__gemm": "adc_h2048_k4_ffrand",
       "g3_mlp_h1024__gemm": "mlp_h1024", "g3_mlp_h2048__gemm": "mlp_h2048", "g3_mlp_h4096__gemm": "mlp_h4096"}
T1_NOTE = {"t1_dhsrnn_n64_b8_seqloop": ("T1 DH-SRNN (ONNX Loop), val-selected T1r model", "")}
out = []
for e in csv.DictReader(open(f"{T6}/energy.csv")):
    lab = e["label"]
    a = acc.get(MAP.get(lab, ""), {})
    out.append({"task": "G3" if lab.startswith("g3_") else "G2", "label": lab,
                "acc_cfg": MAP.get(lab, T1_NOTE.get(lab, ("",))[0]), "acc_mean": a.get("acc_mean", ""),
                "acc_std": a.get("acc_std", ""), "E_net_mJ": e["E_net_mJ"], "E_net_range": e["E_net_range"],
                "E_gross_mJ": e["E_gross_mJ"], "mean_ms_energy_run": e["mean_ms"], "n_runs": e["n_runs"],
                "lut_p50_ms": a.get("p50_ms", "")})
with open(f"{T6}/energy_compare.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r["task"], r["label"], r["acc_mean"], r["acc_std"], r["E_net_mJ"], r["n_runs"], r["lut_p50_ms"])
