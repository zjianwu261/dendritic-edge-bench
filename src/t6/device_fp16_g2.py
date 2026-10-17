"""Device-side TensorRT FP16 accuracy of the fair-protocol (speaker split)
G2 models on Orin #1, seed 1 trained weights, full SHD test set (2264 samples, eval_shd.npz = T1 eval file).

Models (seed 1, *_vs_t1):
  lru_L2_H64_N128_lr0.001        reuse results/T6/lut/onnx/g2x ONNX (LRUExport K=50) -> same engines as lut_g2x
  dssmU_L2_n128_b4_spk           reuse g2x ONNX (DHSSMExport K=50)
  gru_h768_lr0.003               reuse g2x ONNX (ONNX GRU op, full sequence)
  dhssm_L2_n128_b4_cont          new export, DHSSMExport K=50 -> results/T6/device/onnx_g2f/
  dhssm_L2_n256_b4_cont          new export, DHSSMExport K=50
Both builds (default, nocublas; 1 GB workspace, memory guard) are evaluated; nocublas is the primary column
(same engine as the energy runs), default is reported alongside.

  python device_fp16_g2.py ref      PyTorch FP32 (training module) + ORT FP32 (the ONNX) accuracy, workstation CPU
  python device_fp16_g2.py push     rsync ONNX/npz/scripts, start on the Orin (setsid): t6_orin.sh batch (nruns 0,
                                    builds missing engines in ~/dendritic-edge-bench/T6/work) -> per-engine eval,
                                    every build / eval under flock /tmp/dn_orin_bench.lock; clocks never touched
  python device_fp16_g2.py collect  pull outputs -> results/T6/device_fp16_acc.csv
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, MODELS, T6, export_onnx, save_json  # noqa: E402
import dh_ssm as D  # noqa: E402

ORIN = os.environ.get("ORIN_HOST", "orin")
RD = "dendritic-edge-bench/T6"
OD = f"{T6}/device/onnx_g2f"
REF = f"{T6}/device/ref_g2f"
RAW = f"{T6}/device/raw_g2f"
EVAL = f"{T6}/data/eval_shd.npz"
G2X = f"{LUT}/onnx/g2x"
# label -> (checkpoint name, onnx path local, onnx path on the Orin)
MODELS_G2F = {
    "lru_L2_H64_N128_lr0.001_vs_t1__K50": ("lru_L2_H64_N128_lr0.001_vs_t1", f"{G2X}/lru_L2_H64_N128_lr0.001_vs_t1__K50.onnx", "onnx/g2x"),
    "dssmU_L2_n128_b4_spk_vs_t1__K50": ("dssmU_L2_n128_b4_spk_vs_t1", f"{G2X}/dssmU_L2_n128_b4_spk_vs_t1__K50.onnx", "onnx/g2x"),
    "gru_h768_lr0.003_vs_t1__seq": ("gru_h768_lr0.003_vs_t1", f"{G2X}/gru_h768_lr0.003_vs_t1__seq.onnx", "onnx/g2x"),
    "dhssm_L2_n128_b4_cont_vs_t1__K50": ("dhssm_L2_n128_b4_cont_vs_t1", f"{OD}/dhssm_L2_n128_b4_cont_vs_t1__K50.onnx", "onnx/g2f"),
    "dhssm_L2_n256_b4_cont_vs_t1__K50": ("dhssm_L2_n256_b4_cont_vs_t1", f"{OD}/dhssm_L2_n256_b4_cont_vs_t1__K50.onnx", "onnx/g2f"),
}


def load(name):
    meta = json.load(open(f"{MODELS}/g2/{name}.json"))
    a = meta["args"]
    sd = torch.load(f"{MODELS}/g2/{name}.pt", map_location="cpu")
    if a["arch"] in ("dhssm", "dssmU"):
        m = D.DHSSM(a["n"], a["nb"], a["L"], a["act"], a["dropout"], K=50, grouped=(a["arch"] == "dhssm"))
        m.load_state_dict(sd)
        return m.eval(), D.DHSSMExport(m.eval(), 50).eval(), meta
    if a["arch"] == "lru":
        m = D.LRUNet(a["H"], a["N"], a["L"], a["dropout"], K=50)
        m.load_state_dict(sd)
        return m.eval(), D.LRUExport(m.eval(), 50).eval(), meta
    m = D.GRUNetD(a["h"], a["dropout"])
    m.load_state_dict(sd)
    return m.eval(), m.eval(), meta


def xs(data, idx):
    return torch.tensor(np.stack([np.unpackbits(data["x__packbits"][i])[:D.T_STEPS * D.N_IN] for i in idx])
                        .astype(np.float32)).view(len(idx), D.T_STEPS, D.N_IN)


def ref():
    import onnxruntime as ort
    torch.set_num_threads(6)
    os.makedirs(REF, exist_ok=True)
    data = np.load(EVAL)
    y = D.load_shd("test", "cpu")[2].numpy()
    N = len(y)
    # sanity: the T1 eval npz is the same raster / order as load_shd("test")
    fl, of, _ = D.load_shd("test", "cpu")
    for i in (0, 1234, N - 1):
        assert torch.equal(D.make_batch(fl, of, torch.tensor([i])), xs(data, [i])), f"npz sample {i} != load_shd"
    for lab, (name, onnx, _) in MODELS_G2F.items():
        out = f"{REF}/{lab}.json"
        if os.path.exists(out):
            continue
        train_m, exp_m, meta = load(name)
        info = None
        if not os.path.exists(onnx):
            torch.manual_seed(0)
            x0 = torch.zeros(1, D.T_STEPS, D.N_IN).bernoulli_(0.02)
            info = export_onnx(exp_m, (x0,), onnx, ["x"], ["y"], check=True)
            print(lab, info, flush=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 6
        sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
        po, eo, oo = [], [], []
        with torch.no_grad():
            for s in range(0, N, 16):
                idx = list(range(s, min(N, s + 16)))
                xb = xs(data, idx)
                po.append(train_m(xb).numpy())
                if exp_m is not train_m:
                    eo.append(torch.cat([exp_m(xb[j:j + 1]) for j in range(len(idx))]).numpy())
                for j in range(len(idx)):
                    oo.append(sess.run(None, {"x": xb[j:j + 1].numpy()})[0].reshape(-1))
        po, oo = np.concatenate(po), np.stack(oo)
        eo = np.concatenate(eo) if eo else po
        np.save(f"{REF}/{lab}_ort.npy", oo)
        r = {"label": lab, "model": name, "cfg": meta["cfg"], "seed": meta["args"]["seed"], "n_eval": int(N),
             "acc_meta_test": meta["test_acc"], "val_acc_meta": meta["val_acc"],
             "acc_torch_fp32": float((po.argmax(1) == y).mean()),
             "acc_torch_export_graph_fp32": float((eo.argmax(1) == y).mean()),
             "acc_ort_fp32": float((oo.argmax(1) == y).mean()),
             "argmax_agree_ort_vs_torch": float((oo.argmax(1) == po.argmax(1)).mean()),
             "max_abs_ort_vs_export_graph": float(np.abs(oo - eo).max()),
             "onnx": os.path.relpath(onnx, T6), "export_info": info}
        save_json(out, r)
        print(json.dumps({k: r[k] for k in ("label", "acc_meta_test", "acc_torch_fp32", "acc_torch_export_graph_fp32",
                                             "acc_ort_fp32", "max_abs_ort_vs_export_graph")}), flush=True)


def push():
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/onnx/g2f ~/{RD}/device/eval ~/{RD}/device/g2f/devout ~/{RD}/bin"], check=True)
    subprocess.run(["rsync", "-a", f"{OD}/", f"{ORIN}:{RD}/onnx/g2f/"], check=True)
    subprocess.run(["rsync", "-a", EVAL, f"{ORIN}:{RD}/device/eval/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/", f"{ORIN}:{RD}/bin/"], check=True)
    man = "".join(f"$HOME/{RD}/{sub}/{lab}.onnx {lab} default,nocublas\n" for lab, (_, _, sub) in MODELS_G2F.items())
    script = f"""set -u
cd $HOME/{RD}
W=$HOME/{RD}/work
G=$HOME/{RD}/device/g2f
printf '%s' "{man}" > $G/manifest.txt
echo "$(date -Iseconds) start" >> $G/progress.txt
bash bin/t6_orin.sh batch $G/manifest.txt $W 0
cp $W/engines_manifest.txt.txt $G/engines.txt
while read -r eng lab b sha; do
  [ -s $G/devout/${{lab}}_${{b}}.npy ] && continue
  flock /tmp/dn_orin_bench.lock python3 bin/t6_trt_eval.py $eng device/eval/eval_shd.npz $G/devout/${{lab}}_${{b}}.npy >> $G/eval.log 2>&1 \\
    || echo "$(date -Iseconds) EVAL FAILED $lab $b" >> $G/failures.txt
  echo "$(date -Iseconds) evaluated $lab $b $(basename $eng)" >> $G/progress.txt
done < $G/engines.txt
echo "$(date -Iseconds) done" >> $G/progress.txt
"""
    subprocess.run(["ssh", ORIN, f"cat > ~/{RD}/device/g2f/run.sh"], input=script.encode(), check=True)
    subprocess.run(["ssh", "-f", ORIN, f"cd ~/{RD} && setsid nohup bash device/g2f/run.sh > device/g2f/run.out 2>&1 < /dev/null & echo started"],
                   check=True, timeout=60)


def collect():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{ORIN}:{RD}/device/g2f/", f"{RAW}/"], check=True)
    y = D.load_shd("test", "cpu")[2].numpy()
    eng = {}
    for l in open(f"{RAW}/engines.txt"):
        e, lab, b, sha = l.split()
        eng[(lab, b)] = (os.path.basename(e), sha)
    rows = []
    for lab in MODELS_G2F:
        r = json.load(open(f"{REF}/{lab}.json"))
        ort_o = np.load(f"{REF}/{lab}_ort.npy")
        row = {"model": r["model"], "cfg": r["cfg"], "seed": r["seed"], "n_eval": r["n_eval"],
               "acc_torch_fp32": round(r["acc_torch_fp32"], 5), "acc_ort_fp32": round(r["acc_ort_fp32"], 5)}
        for b in ("nocublas", "default"):
            p = f"{RAW}/devout/{lab}_{b}.npy"
            if not os.path.exists(p):
                continue
            o = np.load(p)
            acc = float((o.argmax(1) == y).mean())
            sfx = "" if b == "nocublas" else "_default"
            row[f"acc_orin_trt_fp16{sfx}"] = round(acc, 5)
            row[f"delta_fp16_vs_torch_pp{sfx}"] = round(100 * (acc - r["acc_torch_fp32"]), 2)
            row[f"delta_fp16_vs_ort_pp{sfx}"] = round(100 * (acc - r["acc_ort_fp32"]), 2)
            row[f"argmax_agree_fp16_vs_ort{sfx}"] = round(float((o.argmax(1) == ort_o.argmax(1)).mean()), 4)
            row[f"max_abs_fp16_vs_ort{sfx}"] = round(float(np.abs(o - ort_o).max()), 4)
            row[f"engine{sfx}"] = eng.get((lab, b), ("", ""))[0]
        row["onnx"] = r["onnx"]
        row["onnx_sha256"] = eng.get((lab, "nocublas"), ("", ""))[1]
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f"{T6}/device_fp16_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    {"ref": ref, "push": push, "collect": collect}[sys.argv[1]]()
