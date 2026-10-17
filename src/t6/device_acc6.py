"""Trained T6 models (real weights) on Orin #1 - TensorRT FP16 device accuracy and
latency of the very same engines, against PyTorch FP32 and onnxruntime FP32 (workstation CPU).

  python device_acc6.py export <set>    real-weight ONNX -> results/T6/device/onnx/dev_<task>_<model>.onnx,
                                        PyTorch / ORT FP32 outputs + accuracy -> device/ref/*.json
  python device_acc6.py push <set>      copy ONNX + eval npz + scripts to Orin #1 and start (detached):
                                        builds (nocublas, 1 GB workspace, memory guard) -> device eval ->
                                        sync timing, rounds interleaved across ALL engines of the set
                                        (G1 / G3: >= 10 runs in the same period; G2: 3)
  python device_acc6.py collect         pull outputs -> results/T6/device_acc.csv
Labels carry the task prefix (dev_g1_ / dev_g2_ / dev_g3_) so that same-named graphs of different
tasks are never mixed. Eval inputs = the T1 eval npz files (copied to results/T6/data):
F-MNIST full test set; SHD full test set; pMNIST 1000 test images x 10 tasks (as T1 device_acc).
Every Orin build / eval / timing run holds flock /tmp/dn_orin_bench.lock; clocks are not touched.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import MODELS, T6, export_onnx, save_json  # noqa: E402

DEV = f"{T6}/device"
EVAL = f"{T6}/data"
ORIN = os.environ.get("ORIN_HOST", "orin")
RD = "dendritic-edge-bench/T6"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = (1, 2, 3)
SETS = {
    "r028a": {
        "g1": ["dannG_p4_m32_s64_sg1", "dannG_p4_m64_s128_sg4", "mlp_h256", "mlp_h1024"],
        "g2": ["dhssm_L2_n128_b4_cont", "dhssm_L2_n128_b4_spk", "dhssm_L1_n64_b8_spk", "gru_h256_r1", "gru_h64_r1"],
        "g3": ["adc_h1024_k4_ff24", "adc_h2048_k4_ffrand", "mlp_h1024"],
    },
}


def labels_np(task):
    from common6 import load_vision
    if task == "g1":
        return load_vision("fmnist", "cpu")["test"][1].numpy()
    if task == "g3":
        return load_vision("mnist", "cpu")["test"][1][:1000].repeat(10).numpy()
    from dh_ssm import load_shd
    return load_shd("test", "cpu")[2].numpy()


def npz_path(task):
    return f"{EVAL}/eval_{ {'g1': 'fmnist', 'g2': 'shd', 'g3': 'pmnist'}[task] }.npz"


def load_model(task, name):
    """-> (deploy module taking x (1, ...) , meta)."""
    meta = json.load(open(f"{MODELS}/{task}/{name}.json"))
    sd = torch.load(f"{MODELS}/{task}/{name}.pt", map_location="cpu")
    a = types.SimpleNamespace(**meta["args"])
    if task == "g1":
        from dann_g import build
        m, _ = build(a.arch, a)
        m.load_state_dict(sd)
        m.eval()
        m.impl = "bmm" if a.arch == "dannG" else ("conv" if a.arch == "cnn" else "gemm")
        return m, meta
    if task == "g2":
        import dh_ssm as D
        if a.arch in ("dhssm", "dssmU"):
            m = D.DHSSM(a.n, a.nb, a.L, a.act, a.dropout, K=50, grouped=(a.arch == "dhssm"))
            m.load_state_dict(sd)
            return D.DHSSMExport(m.eval(), 25).eval(), meta
        if a.arch == "lru":
            m = D.LRUNet(a.H, a.N, a.L, a.dropout, K=50)
            m.load_state_dict(sd)
            return D.LRUExport(m.eval(), 25).eval(), meta
        m = (D.GRUNetD if a.arch == "gru" else D.LSTMNetD)(a.h, a.dropout)
        m.load_state_dict(sd)
        return m.eval(), meta
    import ad_cache as A
    _, _, _, _, perms, protos = A.pmnist("cpu")
    if a.arch == "ad":
        m = A.ADNet(a.h, a.k, seed=a.seed, ff=a.ff)
        m.load_state_dict(sd)
        return A.ADCached(m.eval(), protos).eval(), meta
    if a.arch == "mlpctx":
        m = A.MLPCtx(a.h, protos)
        m.load_state_dict(sd)
        return m.eval(), meta
    m = A.MLP(a.h)
    m.load_state_dict(sd)
    return m.eval(), meta


def x_iter(task, data, i):
    if task == "g2":
        return torch.tensor(np.unpackbits(data["x__packbits"][i])[:1000 * 700].astype(np.float32)).view(1, 1000, 700)
    return torch.tensor(data["x"][i]).view(1, 784)


def export(setname):
    os.makedirs(f"{DEV}/onnx", exist_ok=True)
    os.makedirs(f"{DEV}/ref", exist_ok=True)
    os.makedirs(EVAL, exist_ok=True)
    for n in ("fmnist", "shd", "pmnist"):
        if not os.path.exists(f"{EVAL}/eval_{n}.npz"):
            shutil.copy(os.path.join(ROOT, "results/T1/data", f"eval_{n}.npz"), EVAL)
    import onnxruntime as ort
    torch.set_num_threads(4)
    lines = []
    for task, names in SETS[setname].items():
        y = labels_np(task)
        data = np.load(npz_path(task))
        N = len(y)
        for base in names:
            for s in SEEDS:
                name = f"{base}_t{s}"
                lab = f"dev_{task}_{name}"
                onnx = f"{DEV}/onnx/{lab}.onnx"
                ref = f"{DEV}/ref/{lab}.json"
                lines.append(f"~/{RD}/device/onnx/{lab}.onnx {lab} nocublas")
                if os.path.exists(ref):
                    continue
                m, meta = load_model(task, name)
                x0 = x_iter(task, data, 0)
                info = export_onnx(m, (x0,), onnx, ["x"], ["y"])
                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
                po, oo = [], []
                with torch.no_grad():
                    for i in range(N):
                        xi = x_iter(task, data, i)
                        po.append(m(xi).numpy().reshape(-1))
                        oo.append(sess.run(None, {"x": xi.numpy()})[0].reshape(-1))
                po, oo = np.stack(po), np.stack(oo)
                np.save(f"{DEV}/ref/{lab}_ort.npy", oo)
                r = {"label": lab, "task": task, "model": name, "cfg": meta["cfg"] + (meta["args"].get("tag", "") if task == "g2" else ""),
                     "n_eval": int(N), "acc_torch_fp32": float((po.argmax(1) == y).mean()),
                     "acc_ort_fp32": float((oo.argmax(1) == y).mean()),
                     "acc_reported_meta": meta["test_acc"], "export": info,
                     "note": "G3: 1000 test images x 10 tasks (T1 device_acc subset)" if task == "g3" else ""}
                save_json(ref, r)
                print(json.dumps({k: r[k] for k in ("label", "acc_torch_fp32", "acc_ort_fp32", "acc_reported_meta")}), flush=True)
    with open(f"{DEV}/onnx/manifest_{setname}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} entries -> {DEV}/onnx/manifest_{setname}.txt")


def push(setname):
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/device/onnx ~/{RD}/device/eval ~/{RD}/device/work"], check=True)
    subprocess.run(["rsync", "-a", f"{DEV}/onnx/", f"{ORIN}:{RD}/device/onnx/"], check=True)
    subprocess.run(["rsync", "-a", f"{EVAL}/eval_fmnist.npz", f"{EVAL}/eval_shd.npz", f"{EVAL}/eval_pmnist.npz",
                    f"{ORIN}:{RD}/device/eval/"], check=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["rsync", "-a", f"{here}/orin/", f"{ORIN}:{RD}/bin/"], check=True)
    man = f"manifest_{setname}.txt"
    W = f"$HOME/{RD}/device/work"
    # small models (G1 / G3): >= 10 interleaved runs in one period; G2: 3 runs
    cmd = (f"cd ~/{RD} && sed \"s#^~#$HOME#\" device/onnx/{man} > device/work/{setname}_all.txt && "
           f"grep -v ' dev_g2_' device/work/{setname}_all.txt > device/work/{setname}_small.txt; "
           f"grep ' dev_g2_' device/work/{setname}_all.txt > device/work/{setname}_g2.txt; "
           f"setsid nohup bash -c 'bash bin/t6_orin.sh batch device/work/{setname}_all.txt {W} 0 && "
           f"bash bin/t6_orin.sh evalall {W}/engines_{setname}_all.txt.txt $HOME/{RD}/device/eval {W}; "
           f"bash bin/t6_orin.sh batch device/work/{setname}_small.txt {W} 10; "
           f"bash bin/t6_orin.sh batch device/work/{setname}_g2.txt {W} 3' "
           f"> device/work/{setname}_chain.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)


def collect():
    os.makedirs(f"{DEV}/raw", exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/device/work/", f"{DEV}/raw/"], check=True)
    rows = []
    res = [json.loads(l) for l in open(f"{DEV}/raw/results.jsonl")] if os.path.exists(f"{DEV}/raw/results.jsonl") else []
    for ref in sorted(glob.glob(f"{DEV}/ref/*.json")):
        r = json.load(open(ref))
        lab = r["label"]
        out = f"{DEV}/raw/devout/{lab}_nocublas.npy"
        if os.path.exists(out):
            y = labels_np(r["task"])
            o = np.load(out)
            ort_o = np.load(f"{DEV}/ref/{lab}_ort.npy")
            r["acc_trt_fp16"] = float((o.argmax(1) == y).mean())
            r["argmax_agree_fp16_vs_ort"] = float((o.argmax(1) == ort_o.argmax(1)).mean())
            r["max_abs_diff_fp16_vs_ort"] = float(np.abs(o - ort_o).max())
        rr = [x for x in res if x["label"] == lab]
        if rr:
            p = [x["p50_ms"] for x in rr]
            r["n_runs"] = len(rr)
            r["p50_ms_median"] = round(float(np.median(p)), 4)
            r["p50_ms_min_max"] = f"{min(p):.4f}-{max(p):.4f}"
            r["infer_only_p50_ms"] = round(float(np.median([x["infer_only_p50_ms"] for x in rr])), 4)
        r.pop("export", None)
        rows.append(r)
    import csv
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f"{T6}/device_acc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {T6}/device_acc.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "push", "collect"])
    ap.add_argument("set", nargs="?", default="r028a")
    a = ap.parse_args()
    {"export": lambda: export(a.set), "push": lambda: push(a.set), "collect": collect}[a.cmd]()
