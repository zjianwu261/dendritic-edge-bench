"""Build the T1 result tables from raw benchmark JSON lines + model metadata.

Outputs (results/T1/):
  results.csv              main table, one row per model x device. Orin: everything (latency,
                           memory, on-device accuracy, energy) from the nocublas build
                           (--tacticSources=-CUBLAS,-CUBLAS_LT); the
                           default-build latency / memory are only repeated as reference columns.
                           energy_mJ = net-of-idle energy per inference from
                           results/T1/energy_orin1.csv (src/bench/energy_parse.py; jtop VDD_IN,
                           >= 30 s windows, locked clocks), gross energy in energy_gross_mJ.
  results_orin_builds.csv  appendix: every model x TensorRT build (default / nocublas) on the Orin,
                           with per-run GPU frequencies and the number of cuBLAS layers in the engine.


Per (model, device[, build]) the last NRUNS runs are used; p50 / p95 / memory = median over runs of
the per-run values; latency_p50_range = min-max of the per-run p50. Every run: warm-up >= 2 s
and >= 50 iterations, then 500 timed iterations, batch 1, fresh process.
peak_mem_MB: Orin = drop of system MemAvailable from process start to after timing (CUDA
context + TensorRT (+cuBLAS if loaded) + engine + buffers; the iGPU shares system RAM);
RPi = VmHWM (process peak RSS).
GPU frequency (Orin, not locked, DVFS): gpu_MHz_warmup_end = cur_freq at the end of the warm-up of
each run; gpu_MHz_timed_median = median of the cur_freq samples taken every 25 timed iterations.
"""
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

T1 = os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results/T1")
ORIN_RAW, RPI_RAW = "orin1_v07", "rpi4b_v07"
# Appendix column: small models (G1 + G2 single step) re-timed with the Orin clocks
# locked (GPU 624.75 MHz; locked externally, not by these scripts),
# nocublas build, same protocol; raw/orin1_v07_locked (src/t1/run_t1r_locked.sh).
ORIN_LOCKED_RAW = "orin1_v07_locked"
NRUNS = 3
FMAX = 624750000

G1, G2, G2S, G2U, G3, G3D = ("G1 dANN / Fashion-MNIST", "G2 SHD (full sequence T=1000)",
                             "G2 SHD (streaming step, 1 ms)", "G2 SHD (supplementary: unrolled graph, RPi only)",
                             "G3 active dendrites / pMNIST-10", "G3 deployment variants (supplementary)")
# label -> (group, dendritic reference it is a baseline of, meta name, note)
MODELS = {
    "dann_lrf_d32_s128": (G1, "", "dann_lrf_d32_s128", "mask implementation (dense matrices with zeros)"),
    "mlp_h85_match_d32_s128": (G1, "dann_lrf_d32_s128", "mlp_h85_match_d32_s128", "same #params (effective)"),
    "vann_same_d32_s128": (G1, "dann_lrf_d32_s128", "vann_same_d32_s128", "same #params (stored) = dense MLP with the dANN's layer shape (what the mask implementation computes)"),
    "mlp_h256_fmnist": (G1, "dann_lrf_d32_s128", "mlp_h256_fmnist", "accuracy-oriented baseline"),
    "mlp_h512_fmnist": (G1, "dann_lrf_d32_s128", "mlp_h512_fmnist", "accuracy-oriented baseline"),
    "mlp_h1024_fmnist": (G1, "dann_lrf_d32_s128", "mlp_h1024_fmnist", "accuracy-oriented baseline"),
    # G2, validation-selected weights; graph = v0.5 graph
    "dhsrnn_n64_b8_val_seqloop": (G2, "", "dhsrnn_n64_b8_val", "ONNX Loop (TorchScript); epoch selected on validation"),
    "lif_srnn_n64_val_seqloop": (G2, "dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val", "ONNX Loop (TorchScript); epoch selected on validation"),
    "gru_pm_val_seqloop": (G2, "dhsrnn_n64_b8_val_seqloop", "gru_pm_val", "parameter-matched GRU, tuned on validation; ONNX Loop (TorchScript), same construction as the SNNs"),
    "gru_pm_val_seq": (G2, "dhsrnn_n64_b8_val_seqloop", "gru_pm_val", "parameter-matched GRU, tuned on validation; fused ONNX GRU op"),
    "gru_acc_val_seqloop": (G2, "dhsrnn_n64_b8_val_seqloop", "gru_acc_val", "accuracy-oriented GRU (best validation accuracy in the grid); ONNX Loop"),
    "gru_acc_val_seq": (G2, "dhsrnn_n64_b8_val_seqloop", "gru_acc_val", "accuracy-oriented GRU (best validation accuracy in the grid); fused ONNX GRU op"),
    # G2 with the v0.5 weights (last epoch, no validation split); same graphs
    "dhsrnn_n64_b8_seqloop": (G2 + " v0.5 weights", "", "dhsrnn_n64_b8", "ONNX Loop; v0.5 weights (last epoch)"),
    "lif_srnn_n64_seqloop": (G2 + " v0.5 weights", "dhsrnn_n64_b8_seqloop", "lif_srnn_n64", "ONNX Loop; v0.5 weights"),
    "gru_h23_seqloop": (G2 + " v0.5 weights", "dhsrnn_n64_b8_seqloop", "gru_h23", "ONNX Loop; v0.5 weights (untuned)"),
    "gru_h23_seq": (G2 + " v0.5 weights", "dhsrnn_n64_b8_seqloop", "gru_h23", "fused ONNX GRU op; v0.5 weights (untuned)"),
    "dhsrnn_n64_b8_seq": (G2U, "", "dhsrnn_n64_b8", "loop unrolled (~28k nodes); TensorRT 8.5 parser did not finish in 50 min (v0.5)"),
    "lif_srnn_n64_seq": (G2U, "dhsrnn_n64_b8_seq", "lif_srnn_n64", "loop unrolled"),
    "dhsrnn_n64_b8_step": (G2S, "", "dhsrnn_n64_b8", "one step, explicit state I/O"),
    "lif_srnn_n64_step": (G2S, "dhsrnn_n64_b8_step", "lif_srnn_n64", "one step, explicit state I/O"),
    "gru_h23_step": (G2S, "dhsrnn_n64_b8_step", "gru_h23", "one step, explicit state I/O"),
    "gru_acc_val_step": (G2S, "dhsrnn_n64_b8_step", "gru_acc_val", "one step, explicit state I/O"),
    "active_dend_k4": (G3, "", "active_dend_k4", "naive: inputs x + context, segments recomputed per input (prototype lookup not included)"),
    "mlp_h3592_match_k4": (G3, "active_dend_k4", "mlp_h3592_match_k4", "same #params (effective)"),
    "mlp_h3941_matchstored_k4": (G3, "active_dend_k4", "mlp_h3941_matchstored_k4", "same #params (stored)"),
    "active_dend_k8": (G3, "", "active_dend_k8", "naive: inputs x + context (prototype lookup not included)"),
    "mlp_h4965_match_k8": (G3, "active_dend_k8", "mlp_h4965_match_k8", "same #params (effective)"),
    "mlp_h5230_matchstored_k8": (G3, "active_dend_k8", "mlp_h5230_matchstored_k8", "same #params (stored)"),
    "active_dend_k4_proto": (G3D, "active_dend_k4", "active_dend_k4_proto", "naive + nearest-prototype lookup inside the graph (input x only)"),
    "active_dend_k4_cached": (G3D, "active_dend_k4", "active_dend_k4_cached", "gates cached per task + prototype lookup (deployment optimisation, T2 territory)"),
    "active_dend_k8_proto": (G3D, "active_dend_k8", "active_dend_k8_proto", "naive + nearest-prototype lookup inside the graph"),
    "active_dend_k8_cached": (G3D, "active_dend_k8", "active_dend_k8_cached", "gates cached per task + prototype lookup"),
}
ENERGY_NOTE = {
    "rpi4b-8gb": "not measured: no power meter for the RPi",
}


def load_energy():
    p = f"{T1}/energy_orin1.csv"
    return {r["label"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def load_jsonl(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def cublas_layers(label, build):
    p = f"{T1}/raw/{ORIN_RAW}/{label}_fp16_{build}.layers.json"
    if not os.path.exists(p):
        return ""
    try:
        L = json.load(open(p))["Layers"]
    except Exception:  # noqa: BLE001
        return ""
    return sum(1 for l in L if "cublas" in str(l.get("LayerType", "")).lower())


def mhz(x):
    return int(round(int(x) / 1e6))


def orin_summary(rs):
    p50s = [r["p50_ms"] for r in rs]
    fw = [mhz(r["gpu_freq_hz_warmup_end"]) for r in rs]
    ft = [mhz(r["gpu_freq_hz_timed_min_med_max"][1]) for r in rs]
    out = {
        "p50": round(st.median(p50s), 4), "p95": round(st.median([r["p95_ms"] for r in rs]), 4),
        "range": f"{min(p50s):.4f}-{max(p50s):.4f}",
        "rel_range": round((max(p50s) - min(p50s)) / st.median(p50s) / 2, 4),
        "mem": round(st.median([r["mem_avail_drop_MB"] for r in rs]), 1),
        "infer_only": round(st.median([r["infer_only_p50_ms"] for r in rs]), 4),
        "stateful": round(st.median([r["stateful_p50_ms"] for r in rs]), 4) if "stateful_p50_ms" in rs[0] else "",
        "fw": "/".join(map(str, fw)), "ft": "/".join(map(str, ft)),
        "below_max": sum(1 for r in rs if r["gpu_freq_hz_timed_min_med_max"][1] < FMAX),
        "engine_MB": rs[-1]["engine_file_MB"], "warmup": "/".join(str(r["warmup"]) for r in rs),
        "n": len(rs), "sha": rs[-1].get("onnx_sha256", "")[:12],
    }
    return out


def locked_cols(rs):
    rs = sorted([r for r in rs if r.get("build") == "nocublas"], key=lambda r: r["time"])[-NRUNS:]
    if not rs:
        return {"latency_p50_locked": "", "latency_p50_locked_range": "", "locked_runs_not_at_max": ""}
    p50s = [r["p50_ms"] for r in rs]
    return {"latency_p50_locked": round(st.median(p50s), 4),
            "latency_p50_locked_range": f"{min(p50s):.4f}-{max(p50s):.4f}",
            "locked_runs_not_at_max": sum(1 for r in rs if r["gpu_freq_hz_timed_min_med_max"][0] < FMAX)}


def main():
    runs = defaultdict(list)
    for r in load_jsonl(f"{T1}/raw/{ORIN_RAW}/results.jsonl"):
        runs[(r["label"], "orin1", r["build"])].append(r)
    for r in load_jsonl(f"{T1}/raw/{RPI_RAW}/results.jsonl"):
        runs[(r["label"], "rpi4b-8gb", "")].append(r)
    energy = load_energy()
    locked = defaultdict(list)
    for r in load_jsonl(f"{T1}/raw/{ORIN_LOCKED_RAW}/results.jsonl"):
        locked[r["label"]].append(r)
    dacc = {}
    for r in load_jsonl(f"{T1}/raw/{ORIN_RAW}/device_acc.jsonl"):
        dacc[(r["model"], r["build"])] = r  # last one wins
    last = lambda k: sorted(runs.get(k, []), key=lambda r: r["time"])[-NRUNS:]  # noqa: E731

    rows, brows = [], []
    for label, (group, base, meta_name, note) in MODELS.items():
        try:
            meta = json.load(open(f"{T1}/models/{meta_name}.json"))
        except FileNotFoundError:
            continue
        common = {"model": label, "baseline_of": base, "group": group, "dataset": meta["dataset"],
                  "acc": round(meta["test_acc"], 4), "params": meta["params_effective"],
                  "params_stored": meta.get("params_stored_deployed", meta["params_stored"])}
        # ---------------- Orin: both builds
        summ = {}
        for b in ("default", "nocublas"):
            rs = last((label, "orin1", b))
            if rs:
                summ[b] = orin_summary(rs)
                da = dacc.get((label, b))
                s = summ[b]
                brows.append({**common, "device": "Orin Nano #1 (15W)", "trt_build": b,
                              "latency_p50": s["p50"], "latency_p95": s["p95"], "latency_p50_range": s["range"],
                              "latency_p50_rel_halfrange": s["rel_range"], "peak_mem_MB": s["mem"],
                              "engine_file_MB": s["engine_MB"], "cublas_layers": cublas_layers(label, b),
                              "infer_only_p50": s["infer_only"], "stateful_p50": s["stateful"],
                              "gpu_MHz_warmup_end": s["fw"], "gpu_MHz_timed_median": s["ft"],
                              "runs_below_max_freq": s["below_max"], "warmup_iters": s["warmup"], "runs": s["n"],
                              "acc_on_device": round(da["acc_trt_fp16_orin"], 4) if da else "",
                              "onnx_sha256_12": s["sha"], "notes": note})
        if "nocublas" in summ:
            best = "nocublas"   # main table always uses the nocublas build
            s = summ[best]
            mem_b = best
            da = dacc.get((label, best))
            e = energy.get(label)
            if e:
                e_eng_sha = e["engine"].rsplit("_", 1)[-1].split(".")[0]
                enote = (f"net of idle; median of {e['n_runs']} runs x >= 30 s (range {e['E_net_range']} mJ); "
                         f"jtop VDD_IN 100 ms; idle {e['P_idle_mW']} mW, active {e['P_active_mW']} mW; "
                         f"measured with LOCKED clocks (GPU 624.75 MHz) while latency_p50 is DVFS (not locked); "
                         f"engine {e['engine']}"
                         + ("" if e_eng_sha == s["sha"] else " (WARNING: engine sha differs from the latency engine)")
                         + (f"; excluded flagged runs: {e['flagged_runs']}" if e.get("flagged_runs") else "")
                         + "; see results/T1/energy_orin1.csv")
            else:
                enote = "not measured for this model (no entry in results/T1/energy_orin1.csv)"
            rows.append({**common, "device": "Orin Nano #1 (15W)", "precision": "fp16", "batch": 1,
                         "latency_p50": s["p50"], "latency_p95": s["p95"],
                         "energy_mJ": e["E_net_mJ"] if e else "",
                         "peak_mem_MB": summ[mem_b]["mem"], "trt_build": best,
                         "energy_gross_mJ": e["E_gross_mJ"] if e else "",
                         "energy_runs": e["n_runs"] if e else "",
                         "latency_p50_default_ref": summ.get("default", {}).get("p50", ""),
                         "peak_mem_default_ref_MB": summ.get("default", {}).get("mem", ""),
                         "latency_p50_range": s["range"], "runs": s["n"],
                         **locked_cols(locked.get(label, [])),
                         "engine_file_MB": summ[mem_b]["engine_MB"],
                         "infer_only_p50": s["infer_only"], "stateful_p50": s["stateful"],
                         "gpu_MHz_warmup_end": s["fw"], "gpu_MHz_timed_median": s["ft"],
                         "runs_below_max_freq": s["below_max"],
                         "acc_on_device": round(da["acc_trt_fp16_orin"], 4) if da else "",
                         "acc_on_device_n": da["n"] if da else "",
                         "acc_ort_fp32": round(da["acc_ort_fp32_workstation"], 4) if da else "",
                         "runtime": "TensorRT 8.5.2.2 python (ctypes cudart)", "timing_scope": "sync_e2e_pinned",
                         "peak_mem_def": "MemAvailable drop (system)", "energy_note": enote,
                         "notes": note})
        # ---------------- RPi
        rs = last((label, "rpi4b-8gb", ""))
        if rs:
            p50s = [r["p50_ms"] for r in rs]
            rows.append({**common, "device": "RPi 4B 8GB", "precision": "fp32", "batch": 1,
                         "latency_p50": round(st.median(p50s), 4),
                         "latency_p95": round(st.median([r["p95_ms"] for r in rs]), 4), "energy_mJ": "",
                         "peak_mem_MB": round(st.median([r["vm_hwm_MB"] for r in rs]), 1),
                         "trt_build": "", "energy_gross_mJ": "", "energy_runs": "",
                         "latency_p50_default_ref": "", "peak_mem_default_ref_MB": "",
                         **locked_cols([]), "latency_p50_range": f"{min(p50s):.4f}-{max(p50s):.4f}",
                         "runs": len(rs), "engine_file_MB": "", "infer_only_p50": "", "stateful_p50": "",
                         "gpu_MHz_warmup_end": "", "gpu_MHz_timed_median": "", "runs_below_max_freq": "",
                         "acc_on_device": "", "acc_on_device_n": "", "acc_ort_fp32": "",
                         "runtime": rs[0]["runtime"] + f", {rs[0]['threads']} threads pinned to cores {rs[0]['affinity']}",
                         "timing_scope": rs[0]["timing_scope"], "peak_mem_def": "VmHWM (process peak RSS)",
                         "energy_note": ENERGY_NOTE["rpi4b-8gb"],
                         "notes": "; ".join([note, "background garbage.service kept running on core 1"])})
    for path, rr in ((f"{T1}/results.csv", rows), (f"{T1}/results_orin_builds.csv", brows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader()
            w.writerows(rr)
        print(f"wrote {len(rr)} rows to {path}")
    for r in rows:
        print(f"{r['model']:28s} {r['device'][:4]:4s} acc {r['acc']:.4f} dev {r['acc_on_device']!s:6s} "
              f"p50 {r['latency_p50']:9.4f} [{r['trt_build']:8s}] def {r['latency_p50_default_ref']!s:7s} "
              f"E {r['energy_mJ']!s:8s} mem {r['peak_mem_MB']} (def {r['peak_mem_default_ref_MB']}) "
              f"f {r['gpu_MHz_timed_median']} st {r['stateful_p50']}")


if __name__ == "__main__":
    main()
