"""C-c multi-seed retraining (2026-10-26): T1 models, extra seeds.

Runs the ORIGINAL T1 training entry points (src/t1/*.py, unchanged recipe) with two changes made
from the outside (monkeypatching, the T1 sources are not edited):
  * output directory: results/robustness_B/Cc_multiseed/models/s<seed>/ (file names identical
    to results/T1/models, so the original T1 bench/build scripts and engine names apply);
  * device: CPU when --cpu (the shared workstation GPU was fully booked by other jobs at the
    time). Tensor.cuda / Module.cuda become no-ops, set_per_process_memory_fraction
    is ignored, load_shd defaults to the CPU. Numerics: FP32 on CPU (the v0.5 shd.py recipe asked for
    TF32 on the GPU; there is no TF32 on CPU).

Seed mapping ("seed 1" = the original T1 model):
  G1 dann_fmnist.py --trial 1 (orig) / 2 / 3
  G2 shd_val.py / shd.py --seed 0 (orig) / 1 / 2
  G3 active_dendrites.py --seed 0 (orig) / 1 / 2
Usage:
  python ms_train.py [--cpu] <job> <seed_index 2|3>
  jobs: g1_dann g1_mlp85 g1_vann g1_mlp1024 |
        g2v_dhsrnn g2v_lif g2v_grupm g2v_gruacc  (shd_val.py, validation-selected) |
        g2o_dhsrnn g2o_lif g2o_gru23             (shd.py v0.5 recipe, last epoch) |
        g3_ad4 g3_ad8 g3_mlp3592 g3_mlp4965 g3_mlp3941 g3_mlp5230
  python ms_train.py export <seed_index>   -> ONNX exports (val SHD models, AD proto/cached)
"""
import os
import sys

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = f"{ROOT}/results/robustness_B/Cc_multiseed"
sys.path.insert(0, f"{ROOT}/src/t1")

SEEDMAP = {"g1": {1: 1, 2: 2, 3: 3}, "g2": {1: 0, 2: 1, 3: 2}, "g3": {1: 0, 2: 1, 3: 2}}

JOBS = {
    "g1_dann": ("dann_fmnist", ["--model", "dann"]),
    "g1_mlp85": ("dann_fmnist", ["--model", "mlp"]),
    "g1_vann": ("dann_fmnist", ["--model", "vann_same"]),
    "g1_mlp1024": ("dann_fmnist", ["--model", "mlp", "--hidden", "1024"]),
    # shd_val.py: SNN args as in dhsrnn_n64_b8_val.json / lif_srnn_n64_val.json (defaults, mem_frac
    # irrelevant); GRU args as in gru_pm_val.json / gru_acc_val.json (shd_gru_grid: 60 ep, micro 4)
    "g2v_dhsrnn": ("shd_val", ["--model", "dhsrnn"]),
    "g2v_lif": ("shd_val", ["--model", "srnn"]),
    "g2v_grupm": ("shd_val", ["--model", "gru", "--hidden", "23", "--lr", "0.003", "--dropout", "0.3",
                              "--wd", "0.01", "--epochs", "60", "--micro", "4"]),
    "g2v_gruacc": ("shd_val", ["--model", "gru", "--hidden", "256", "--lr", "0.003", "--dropout", "0.3",
                               "--wd", "0.01", "--epochs", "60", "--micro", "4"]),
    "g2o_dhsrnn": ("shd", ["--model", "dhsrnn"]),
    "g2o_lif": ("shd", ["--model", "srnn"]),
    "g2o_gru23": ("shd", ["--model", "gru"]),
    "g3_ad4": ("active_dendrites", ["--model", "ad", "--segments", "4"]),
    "g3_ad8": ("active_dendrites", ["--model", "ad", "--segments", "8"]),
    "g3_mlp3592": ("active_dendrites", ["--model", "mlp", "--segments", "4"]),
    "g3_mlp4965": ("active_dendrites", ["--model", "mlp", "--segments", "8"]),
    "g3_mlp3941": ("active_dendrites", ["--model", "mlp", "--segments", "4", "--match", "stored"]),
    "g3_mlp5230": ("active_dendrites", ["--model", "mlp", "--segments", "8", "--match", "stored"]),
}


def patch_srnn_unbind():
    """Implementation-only change of shd.SRNN.forward: xp[:, t] -> xp.unbind(1)[t].
    On the CPU the backward of xp[:, t] materialises a zero tensor of the full (B, T, n*branch) shape at
    every step (DH-SRNN: 36 s per batch); unbind does one stack. Checked 2026-10-26 on a B=100, T=1000
    batch: loss and every parameter gradient bitwise identical to the original forward (DH and LIF),
    39.6 s -> 1.2 s (DH). Export (SNNStep / SNNLoop) does not use SRNN.forward and is unaffected."""
    import torch.nn.functional as F
    import shd

    def forward(self, x):
        b, T, _ = x.shape
        xs = self.in_proj(x).unbind(1)
        st = self.init_state(b, x.device)
        out = 0
        for t in range(T):
            st, ro = self.step(xs[t], st)
            if t > 10:
                out = out + F.softmax(ro, dim=1)
        return out
    shd.SRNN.forward = forward


def patch_cpu():
    import torch
    import torch.nn as nn
    torch.Tensor.cuda = lambda self, *a, **k: self
    nn.Module.cuda = lambda self, *a, **k: self
    torch.cuda.set_per_process_memory_fraction = lambda *a, **k: None
    os.environ["T1_DEV"] = "cpu"
    import shd
    shd.load_shd.__defaults__ = ("cpu",)


def redirect(mdir):
    """Point every T1 module's MODELS (and shd_val's OUTM/OUTL) at mdir."""
    os.makedirs(mdir, exist_ok=True)
    import common
    common.MODELS = mdir
    for modname in ("shd", "shd_val", "dann_fmnist", "active_dendrites", "shd_loop_export",
                    "gru_loop_export", "ad_cached_export"):
        mod = __import__(modname)
        if hasattr(mod, "MODELS"):
            mod.MODELS = mdir
        if modname == "shd_val":
            mod.OUTM = f"{mdir}/shd_val"
            mod.OUTL = f"{mdir}/shd_val_logs"


def export_all(si):
    """Val SHD exports (export_val_models.py logic) + AD proto/cached (ad_cached_export.py)."""
    import json
    import shutil
    import torch
    import shd
    import shd_loop_export
    import gru_loop_export
    import ad_cached_export
    mdir = f"{OUT}/models/s{si}"
    s = SEEDMAP["g2"][si]
    runs = {"dhsrnn_n64_b8_val": f"dhsrnn_n64_b8_s{s}", "lif_srnn_n64_val": f"lif_srnn_n64_s{s}",
            "gru_pm_val": f"gru_h23_lr0.003_do0.3_wd0.01_s{s}", "gru_acc_val": f"gru_h256_lr0.003_do0.3_wd0.01_s{s}"}
    for alias, run in runs.items():
        if not os.path.exists(f"{mdir}/shd_val/{run}.json") or os.path.exists(f"{mdir}/{alias}_seqloop.onnx"):
            continue
        vm = json.load(open(f"{mdir}/shd_val/{run}.json"))
        sd = torch.load(f"{mdir}/shd_val/{run}.pt", map_location="cpu")
        if vm["model"] == "gru":
            h = vm["args"]["hidden"] or sd["gru.weight_hh_l0"].shape[1]
            m = shd.GRUNet(h)
        else:
            h = 0
            m = shd.SRNN(64, 8 if vm["model"] == "dhsrnn" else 0, seed=vm["args"]["seed"])
        m.load_state_dict(sd)
        meta = {k: v for k, v in vm.items() if k != "history"}
        meta.update({"name": alias, "group": "shd", "source_run": run, "T": 1000, "dt_ms": 1,
                     "selection_note": "epoch chosen on the 10% validation split; test_acc = test accuracy of that epoch"})
        shutil.copy(f"{mdir}/shd_val/{run}.pt", f"{mdir}/{alias}.pt")
        shd.export(m, alias, meta)
        import common
        common.save_meta(alias, meta)
        if vm["model"] == "gru":
            sys.argv = ["gru_loop_export.py", f"{mdir}/{alias}.pt", str(h), alias]
            gru_loop_export.main()
        else:
            sys.argv = ["shd_loop_export.py", alias]
            shd_loop_export.main()
    # v0.5 SNNs: shd.py already exported _seq/_step; ONNX Loop graphs as for T1
    for name in ("dhsrnn_n64_b8", "lif_srnn_n64"):
        if os.path.exists(f"{mdir}/{name}.pt") and not os.path.exists(f"{mdir}/{name}_seqloop.onnx"):
            sys.argv = ["shd_loop_export.py", name]
            shd_loop_export.main()
    if os.path.exists(f"{mdir}/gru_h23.pt") and not os.path.exists(f"{mdir}/gru_h23_seqloop.onnx"):
        sys.argv = ["gru_loop_export.py", f"{mdir}/gru_h23.pt", "23", "gru_h23"]
        gru_loop_export.main()
    for k in (4, 8):
        if os.path.exists(f"{mdir}/active_dend_k{k}.pt") and not os.path.exists(f"{mdir}/active_dend_k{k}_cached.onnx"):
            ad_cached_export_main(k, mdir)


def ad_cached_export_main(k, mdir):
    """ad_cached_export.main with the seed taken from the meta json (ADNet(k) defaults to seed 0,
    which only matters for the sparse masks, and those are overwritten by load_state_dict)."""
    import ad_cached_export
    sys.argv = ["ad_cached_export.py", str(k)]
    ad_cached_export.main()


def main():
    args = sys.argv[1:]
    cpu = "--cpu" in args
    args = [a for a in args if a != "--cpu"]
    if cpu:
        patch_cpu()
    patch_srnn_unbind()
    job, si = args[0], int(args[1])
    redirect(f"{OUT}/models/s{si}")
    if job == "export":
        export_all(si)
        return
    modname, extra = JOBS[job]
    grp = job.split("_")[0][:2]
    seed = SEEDMAP[grp][si]
    flag = "--trial" if modname == "dann_fmnist" else "--seed"
    mod = __import__(modname)
    sys.argv = [f"{modname}.py"] + extra + [flag, str(seed)] + args[2:]
    print(f"[ms_train] {job} seed_index {si} -> {flag} {seed}; cpu={cpu}; argv={sys.argv}", flush=True)
    mod.main()


if __name__ == "__main__":
    main()
