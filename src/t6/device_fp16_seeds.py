"""PLAN §3g-12 BS-1 (B-g2): device-side TensorRT FP16 accuracy of the fair-protocol G2 models, seeds 2-3, on Orin #1,
full SHD test set (2264 samples, eval_shd.npz). Same method as device_fp16_g2.py (seed 1, results/T6/device_fp16_acc.csv):

  models (seeds 2, 3; *_vs_t2 / *_vs_t3):  dhssm_L2_n128_b4_cont (DHSSMExport K=50), lru_L2_H64_N128_lr0.001 (LRUExport
  K=50), dssmU_L2_n128_b4_spk (DHSSMExport K=50, ungrouped), gru_h768_lr0.003 (ONNX GRU op, full sequence).
  Export: common6.export_onnx, opset / dummy input identical to seed 1 (torch.manual_seed(0), bernoulli 0.02);
  the op multiset of every new ONNX must equal the seed-1 ONNX of the same family (asserted in ref()).
  Engines: both builds (default, nocublas), 1 GB workspace, memory guard, built by t6_orin.sh batch (nruns 0) in a
  separate workdir ~/RD/work_g2s; every build / eval under flock /tmp/dn_orin_bench.lock; clocks never touched.

  python device_fp16_seeds.py ref      PyTorch FP32 + ORT FP32 accuracy on the workstation CPU (exports missing ONNX)
  python device_fp16_seeds.py push     rsync ONNX + scripts, start (setsid) build + per-engine eval on the Orin
  python device_fp16_seeds.py status
  python device_fp16_seeds.py collect  pull -> results/T6/fp16_acc_seeds.csv (seed-1 rows copied from
                                       device_fp16_acc.csv for the same 4 configs, marked source=seed1_existing)

Release notes. Device-side driver (needs the Orin #1, the trained G2 checkpoints and the SHD data; not runnable from
the release alone). Paths are results/T6/ via common6 (the research repository used experiments/T6/).
"""
import collections
import csv
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import LUT, T6, export_onnx, save_json  # noqa: E402
import dh_ssm as D  # noqa: E402
import device_fp16_g2 as B  # noqa: E402

ORIN, RD = B.ORIN, B.RD
OD = f"{T6}/device/onnx_g2s"
REF = f"{T6}/device/ref_g2s"
RAW = f"{T6}/device/raw_g2s"
EVAL = B.EVAL
WD = "work_g2s"
OUT = f"{T6}/fp16_acc_seeds.csv"
CFG = {  # cfg -> (export suffix, seed-1 ONNX used for the op-multiset check)
    "dhssm_L2_n128_b4_cont": ("K50", f"{T6}/device/onnx_g2f/dhssm_L2_n128_b4_cont_vs_t1__K50.onnx"),
    "lru_L2_H64_N128_lr0.001": ("K50", f"{LUT}/onnx/g2x/lru_L2_H64_N128_lr0.001_vs_t1__K50.onnx"),
    "dssmU_L2_n128_b4_spk": ("K50", f"{LUT}/onnx/g2x/dssmU_L2_n128_b4_spk_vs_t1__K50.onnx"),
    "gru_h768_lr0.003": ("seq", f"{LUT}/onnx/g2x/gru_h768_lr0.003_vs_t1__seq.onnx"),
}
SEEDS = (2, 3)
MODELS = {f"{c}_vs_t{s}__{sfx}": (f"{c}_vs_t{s}", c, s, ref1) for s in SEEDS for c, (sfx, ref1) in CFG.items()}


def ops(p):
    import onnx
    return dict(collections.Counter(n.op_type for n in onnx.load(p).graph.node))


def ref():
    import onnxruntime as ort
    torch.set_num_threads(8)
    os.makedirs(REF, exist_ok=True)
    data = np.load(EVAL)
    y = D.load_shd("test", "cpu")[2].numpy()
    N = len(y)
    fl, of, _ = D.load_shd("test", "cpu")
    for i in (0, 1234, N - 1):
        assert torch.equal(D.make_batch(fl, of, torch.tensor([i])), B.xs(data, [i])), f"npz sample {i} != load_shd"
    for lab, (name, cfg, seed, ref1) in MODELS.items():
        out = f"{REF}/{lab}.json"
        if os.path.exists(out):
            continue
        train_m, exp_m, meta = B.load(name)
        assert meta["args"]["seed"] == seed, (name, meta["args"]["seed"])
        onnx_p = f"{OD}/{lab}.onnx"
        info = None
        if not os.path.exists(onnx_p):
            torch.manual_seed(0)
            x0 = torch.zeros(1, D.T_STEPS, D.N_IN).bernoulli_(0.02)
            info = export_onnx(exp_m, (x0,), onnx_p, ["x"], ["y"], check=True)
            print(lab, info, flush=True)
        o_new, o_ref = ops(onnx_p), ops(ref1)
        assert o_new == o_ref, (lab, "op multiset differs from seed 1", o_new, o_ref)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        sess = ort.InferenceSession(onnx_p, so, providers=["CPUExecutionProvider"])
        po, eo, oo = [], [], []
        with torch.no_grad():
            for s in range(0, N, 16):
                idx = list(range(s, min(N, s + 16)))
                xb = B.xs(data, idx)
                po.append(train_m(xb).numpy())
                if exp_m is not train_m:
                    eo.append(torch.cat([exp_m(xb[j:j + 1]) for j in range(len(idx))]).numpy())
                for j in range(len(idx)):
                    oo.append(sess.run(None, {"x": xb[j:j + 1].numpy()})[0].reshape(-1))
        po, oo = np.concatenate(po), np.stack(oo)
        eo = np.concatenate(eo) if eo else po
        np.save(f"{REF}/{lab}_ort.npy", oo)
        np.save(f"{REF}/{lab}_torch.npy", po)
        r = {"label": lab, "model": name, "cfg": cfg, "seed": seed, "n_eval": int(N),
             "acc_meta_test": meta["test_acc"], "val_acc_meta": meta["val_acc"],
             "acc_torch_fp32": float((po.argmax(1) == y).mean()),
             "acc_torch_export_graph_fp32": float((eo.argmax(1) == y).mean()),
             "acc_ort_fp32": float((oo.argmax(1) == y).mean()),
             "argmax_agree_ort_vs_torch": float((oo.argmax(1) == po.argmax(1)).mean()),
             "max_abs_ort_vs_export_graph": float(np.abs(oo - eo).max()),
             "onnx": os.path.relpath(onnx_p, T6), "onnx_ops": o_new, "export_info": info}
        save_json(out, r)
        print(json.dumps({k: r[k] for k in ("label", "acc_meta_test", "acc_torch_fp32", "acc_torch_export_graph_fp32",
                                             "acc_ort_fp32", "max_abs_ort_vs_export_graph")}), flush=True)


def push():
    sys.path.insert(0, f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/costmodel")
    import orin1_oos_time as OT
    OT.check()   # read-only lock / disk check; stops if not at 624.75 MHz
    if OT.ssh(f"test -e ~/{RD}/{WD} && echo EXISTS").stdout.strip() == "EXISTS":
        raise SystemExit(f"~/{RD}/{WD} exists: refusing to start twice")
    for lab in MODELS:
        assert os.path.exists(f"{REF}/{lab}.json"), (lab, "run ref first")
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/onnx/g2s ~/{RD}/device/eval ~/{RD}/device/g2s/devout ~/{RD}/bin"],
                   check=True, timeout=60)
    subprocess.run(["rsync", "-a", "--include=*.onnx", "--exclude=*", f"{OD}/", f"{ORIN}:{RD}/onnx/g2s/"], check=True)
    subprocess.run(["rsync", "-a", EVAL, f"{ORIN}:{RD}/device/eval/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/t6_orin.sh", f"{here}/orin/t6_sync_bench.py", f"{here}/orin/t6_trt_eval.py",
                    f"{ORIN}:{RD}/bin/"], check=True)
    man = "".join(f"$HOME/{RD}/onnx/g2s/{lab}.onnx {lab} default,nocublas\n" for lab in MODELS)
    script = f"""set -u
cd $HOME/{RD}
W=$HOME/{RD}/{WD}
G=$HOME/{RD}/device/g2s
mkdir -p $W
printf '%s' "{man}" > $G/manifest.txt
echo "$(date -Iseconds) start" >> $G/progress.txt
bash bin/t6_orin.sh batch $G/manifest.txt $W 0
cp $W/engines_manifest.txt.txt $G/engines.txt
cp $W/failures.txt $G/build_failures.txt 2>/dev/null
mkdir -p $G/logs && cp $W/logs/*_build.log $G/logs/ 2>/dev/null
while read -r eng lab b sha; do
  [ -s $G/devout/${{lab}}_${{b}}.npy ] && continue
  flock /tmp/dn_orin_bench.lock python3 bin/t6_trt_eval.py $eng device/eval/eval_shd.npz $G/devout/${{lab}}_${{b}}.npy >> $G/eval.log 2>&1 \\
    || echo "$(date -Iseconds) EVAL FAILED $lab $b" >> $G/failures.txt
  echo "$(date -Iseconds) evaluated $lab $b $(basename $eng)" >> $G/progress.txt
done < $G/engines.txt
echo "$(date -Iseconds) done" >> $G/progress.txt
"""
    subprocess.run(["ssh", ORIN, f"cat > ~/{RD}/device/g2s/run.sh"], input=script.encode(), check=True, timeout=60)
    subprocess.run(["ssh", "-f", ORIN, f"cd ~/{RD} && setsid nohup bash device/g2s/run.sh > device/g2s/run.out 2>&1 "
                    f"< /dev/null &"], check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("started")


def status():
    r = subprocess.run(["ssh", ORIN, f"tail -4 ~/{RD}/device/g2s/progress.txt; tail -3 ~/{RD}/{WD}/progress.txt; "
                        f"cat ~/{RD}/{WD}/failures.txt ~/{RD}/device/g2s/failures.txt 2>/dev/null; df -h / | tail -1"],
                       capture_output=True, text=True, timeout=60)
    print(r.stdout)


def collect():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{ORIN}:{RD}/device/g2s/", f"{RAW}/"], check=True)
    y = D.load_shd("test", "cpu")[2].numpy()
    eng = {}
    for l in open(f"{RAW}/engines.txt"):
        e, lab, b, sha = l.split()
        eng[(lab, b)] = (os.path.basename(e), sha)
    rows = []
    s1 = {r["cfg"]: r for r in csv.DictReader(open(f"{T6}/device_fp16_acc.csv"))}
    for cfg in CFG:   # seed-1 rows, copied unchanged (same columns), so the file shows all 3 seeds
        if cfg in s1:
            rows.append({**s1[cfg], "source": "seed1_existing (experiments/T6/device_fp16_acc.csv)"})  # research path, as in the CSV
    for lab, (name, cfg, seed, _) in MODELS.items():
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
        row["source"] = "BS-1 2026-10-30 (src/t6/device_fp16_seeds.py)"
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r.get(k) for k in ("cfg", "seed", "acc_torch_fp32", "acc_orin_trt_fp16", "delta_fp16_vs_torch_pp",
                                     "acc_orin_trt_fp16_default", "delta_fp16_vs_torch_pp_default")})
    print("->", OUT)


if __name__ == "__main__":
    {"ref": ref, "push": push, "status": status, "collect": collect}[sys.argv[1]]()
