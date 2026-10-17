"""Check that the streaming single-step ONNX graphs, chained over T = 1000
steps with onnxruntime (state fed back), reproduce the full-sequence graphs.

For each model the first N SHD test samples are run (a) through <name>_step.onnx step by step
from zero state, accumulating the model readout exactly as the full model does (SNN: sum over
t > 10 of softmax(readout); GRU: mean over t > 10 of the logits), and (b) through the reference
full-sequence graph (<name>_seqloop.onnx; the Loop graphs themselves were checked against the
PyTorch model at export). Reports max |diff| of the outputs and argmax agreement, and the
accuracy of both on those N samples. Output: results/T1/logs/step_chain_check.json
Usage: python step_chain_check.py [N=200] [model ...]
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, T1  # noqa: E402
from shd import load_shd, make_batch  # noqa: E402


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    names = sys.argv[2:] or ["dhsrnn_n64_b8", "lif_srnn_n64", "gru_h23"]
    flat, offs, y = load_shd("test", "cpu")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    out = {}
    for name in names:
        step = ort.InferenceSession(f"{MODELS}/{name}_step.onnx", so, providers=["CPUExecutionProvider"])
        seq = ort.InferenceSession(f"{MODELS}/{name}_seqloop.onnx", so, providers=["CPUExecutionProvider"])
        ins = step.get_inputs()
        outs = [o.name for o in step.get_outputs()]
        state_names = [i.name for i in ins[1:]]
        is_gru = name.startswith("gru")
        diffs, agree, acc_step, acc_seq = [], 0, 0, 0
        for k in range(n):
            x = make_batch(flat, offs, torch.tensor([k])).numpy()[0]           # T x 700
            st = {i.name: np.zeros(i.shape, np.float32) for i in ins[1:]}
            acc = np.zeros((1, 20), np.float32)
            for t in range(x.shape[0]):
                res = step.run(None, {ins[0].name: x[t:t + 1], **st})
                o = res[0]
                st = {sn: res[outs.index(sn + "_o")] for sn in state_names}
                if t > 10:
                    acc += o if is_gru else softmax(o)
            if is_gru:
                acc /= (x.shape[0] - 11)
            ref = seq.run(None, {"x": x[None]})[0]
            diffs.append(float(np.abs(acc - ref).max()))
            agree += int(acc.argmax() == ref.argmax())
            acc_step += int(acc.argmax() == int(y[k]))
            acc_seq += int(ref.argmax() == int(y[k]))
        out[name] = {"n": n, "max_abs_diff": max(diffs), "median_abs_diff": float(np.median(diffs)),
                     "argmax_agreement": agree / n, "acc_step_chained": acc_step / n, "acc_seqloop": acc_seq / n,
                     "reference": f"{name}_seqloop.onnx"}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(f"{T1}/logs/step_chain_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
