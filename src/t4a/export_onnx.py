"""Export NoMaD / ViNT / GNM components to static-shape ONNX (opset 16, TRT 8.5-friendly).

NoMaD graphs (official defaults: context_size=3, image 96x96, 10 DDPM steps, 8 samples):
  nomad_enc_explore_b1     obs[1,12,96,96], goal[1,3,96,96] -> cond[1,256]      (goal masked, explore.py)
  nomad_enc_nav_b10        obs[10,12,..], goal[10,3,..]     -> cond[10,256], dist[10,1] (navigate.py, radius 4)
  nomad_unet_b{1,8}        sample[B,8,2], t[B], cond[B,256] -> eps[B,8,2]       (one denoising step, UNet only)
  nomad_full_explore_b8    obs, goal, z0[8,8,2], z[9,8,8,2] -> actions[8,8,2]   (encoder + 10 unrolled
                           UNet steps + DDPM update (fixed_small variance, clip_sample) + unnormalize + cumsum)
  nomad_full_explore_b1    same with 1 sample
ViNT / GNM: vint_b1 / gnm_b1 obs[1,18,64,85], goal[1,3,64,85] -> dist, action.

Each export is checked against PyTorch with onnxruntime (CPU). The fused graph is checked against the
*official* diffusers DDPMScheduler loop driven by the same noise sequence.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, ddpm_scheduler, load_model  # noqa: E402

OUT = os.path.join(EXP, "models")
ACTION_MIN = torch.tensor([-2.5, -4.0])
ACTION_MAX = torch.tensor([5.0, 4.0])


def no_swish(m):
    for mod in m.modules():
        if hasattr(mod, "set_swish"):
            mod.set_swish(memory_efficient=False)


class EncExplore(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, obs, goal):
        mask = torch.ones(obs.shape[0], dtype=torch.long)
        return self.m.vision_encoder(obs, goal, input_goal_mask=mask)


class EncNav(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, obs, goal):
        mask = torch.zeros(obs.shape[0], dtype=torch.long)
        c = self.m.vision_encoder(obs, goal, input_goal_mask=mask)
        return c, self.m.dist_pred_net(c)


class UNetStep(nn.Module):
    def __init__(self, m):
        super().__init__(); self.u = m.noise_pred_net

    def forward(self, sample, t, cond):
        return self.u(sample=sample, timestep=t, global_cond=cond)


class DDPMUnrolled(nn.Module):
    """Encoder + all denoising steps + DDPM update, replicating diffusers DDPMScheduler.step
    (variance_type fixed_small, clip_sample=True, prediction_type epsilon, T_inference == T_train)."""

    def __init__(self, m, sched, n_samples, horizon=8):
        super().__init__()
        self.m, self.n, self.h = m, n_samples, horizon
        sched.set_timesteps(sched.config.num_train_timesteps)
        self.ts = [int(t) for t in sched.timesteps]
        ac = sched.alphas_cumprod.double()
        self.coef = []
        for t in self.ts:
            prev = t - 1
            ap = ac[t]; ap_prev = ac[prev] if prev >= 0 else torch.tensor(1.0, dtype=torch.float64)
            bp, bp_prev = 1 - ap, 1 - ap_prev
            cur_a = ap / ap_prev; cur_b = 1 - cur_a
            c = dict(sa=float(ap.sqrt()), sb=float(bp.sqrt()),
                     cx0=float(ap_prev.sqrt() * cur_b / bp), cxt=float(cur_a.sqrt() * bp_prev / bp),
                     std=float(torch.clamp(bp_prev / bp * cur_b, min=1e-20).sqrt()) if t > 0 else 0.0)
            self.coef.append(c)
        self.register_buffer("tri", torch.tril(torch.ones(horizon, horizon)))  # cumsum as matmul

    def forward(self, obs, goal, z0, z):
        mask = torch.ones(obs.shape[0], dtype=torch.long)
        cond = self.m.vision_encoder(obs, goal, input_goal_mask=mask)
        cond = cond.repeat(self.n, 1)
        x = z0
        for i, t in enumerate(self.ts):
            tt = torch.full((self.n,), float(t))
            eps = self.m.noise_pred_net(sample=x, timestep=tt, global_cond=cond)
            c = self.coef[i]
            x0 = torch.clamp((x - c["sb"] * eps) / c["sa"], -1.0, 1.0)
            x = c["cx0"] * x0 + c["cxt"] * x
            if t > 0:
                x = x + c["std"] * z[i]
        d = (x + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
        return torch.matmul(self.tri, d)


def official_loop(m, sched, obs, goal, n, gen_seed):
    """Reference: the official explore.py loop (diffusers step with a seeded generator)."""
    g = torch.Generator().manual_seed(gen_seed)
    z0 = torch.randn((n, 8, 2), generator=g)
    cond = m("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=torch.ones(1).long()).repeat(n, 1)
    sched.set_timesteps(sched.config.num_train_timesteps)
    x = z0
    noises = []
    g2 = torch.Generator().manual_seed(gen_seed + 1)
    for k in sched.timesteps:
        eps = m("noise_pred_net", sample=x, timestep=k, global_cond=cond)
        state = g2.get_state()
        x = sched.step(model_output=eps, timestep=k, sample=x, generator=g2).prev_sample
        if int(k) > 0:  # recover the variance noise diffusers drew
            g3 = torch.Generator(); g3.set_state(state)
            noises.append(torch.randn(eps.shape, generator=g3))
    d = (x + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
    return z0, torch.stack(noises + [torch.zeros_like(z0)]), torch.cumsum(d, 1)


def export(mod, args, names_in, names_out, path, check=True):
    mod.eval()  # torch.onnx.export restores the wrapper's (train) mode recursively afterwards
    torch.onnx.export(mod, args, path, input_names=names_in, output_names=names_out, opset_version=16,
                      do_constant_folding=True, dynamo=False)
    info = {"onnx": os.path.basename(path), "inputs": {n: list(a.shape) for n, a in zip(names_in, args)}}
    if check:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        outs = s.run(None, {n: a.numpy() for n, a in zip(names_in, args)})
        mod.eval()
        with torch.no_grad():
            ref = mod(*args)
        ref = ref if isinstance(ref, tuple) else (ref,)
        info["max_abs_err_vs_torch"] = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref))
    print(json.dumps(info))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)  # fused transformer kernel has no ONNX symbolic
    torch.manual_seed(0)
    log = []
    m, cfg = load_model("nomad")
    no_swish(m)
    sched = ddpm_scheduler(cfg)
    obs1, goal1 = torch.randn(1, 12, 96, 96), torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        log.append(export(EncExplore(m), (obs1, goal1), ["obs", "goal"], ["cond"], f"{OUT}/nomad_enc_explore_b1.onnx"))
        log.append(export(EncNav(m), (obs1.repeat(10, 1, 1, 1), torch.randn(10, 3, 96, 96)), ["obs", "goal"],
                          ["cond", "dist"], f"{OUT}/nomad_enc_nav_b10.onnx"))
        for b in (1, 8):
            log.append(export(UNetStep(m), (torch.randn(b, 8, 2), torch.full((b,), 5.0), torch.randn(b, 256)),
                              ["sample", "t", "cond"], ["eps"], f"{OUT}/nomad_unet_b{b}.onnx"))
        for n in (8, 1):
            full = DDPMUnrolled(m, sched, n).eval()
            z0, z, ref = official_loop(m, sched, obs1, goal1, n, 123)
            mine = full(obs1, goal1, z0, z)
            err_off = float((mine - ref).abs().max())
            print(f"fused vs official diffusers loop (n={n}): max abs err {err_off:.2e} m")
            info = export(full, (obs1, goal1, z0, z), ["obs", "goal", "z0", "z"], ["actions"],
                          f"{OUT}/nomad_full_explore_b{n}.onnx")
            info["max_abs_err_vs_official_loop"] = err_off
            log.append(info)
    for name in ("vint", "gnm"):
        mm, c = load_model(name)
        no_swish(mm)
        w, h = c["image_size"]
        o, g = torch.randn(1, 3 * (c["context_size"] + 1), h, w), torch.randn(1, 3, h, w)
        with torch.no_grad():
            log.append(export(mm, (o, g), ["obs", "goal"], ["dist", "action"], f"{OUT}/{name}_b1.onnx"))
    with open(f"{OUT}/export_log.json", "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
