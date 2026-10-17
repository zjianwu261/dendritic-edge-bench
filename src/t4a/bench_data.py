"""T4a benchmark task loader + NoMaD metrics (shared by zero-shot eval and continual-learning runs).

Frames are kept in host memory as uint8 and gathered per batch; normalization matches the official
training transform (ImageNet mean/std after [0,1] scaling).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
from build_bench import CTX, OUT, make_sample, valid_currs  # noqa: E402

ACTION_MIN = torch.tensor([-2.5, -4.0])
ACTION_MAX = torch.tensor([5.0, 4.0])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Task:
    def __init__(self, name, light="normal", device="cuda", holdout_val=False):
        """name: task folder in $DEB_DATA/t4a_bench; light: 'normal', 'dark' (synthetic low light) or
        'occl' (fixed grey box over 40% of every frame, lower-left: a persistent lens obstruction)."""
        d = os.path.join(OUT, name)
        self.name, self.light, self.device = name, light, device
        self.frames = torch.from_numpy(np.load(os.path.join(d, "frames_dark.npy" if light == "dark" else "frames.npy")))
        if light == "occl":
            self.frames[:, 38:96, 0:64] = 128  # 58 x 64 = 40% of 96 x 96
        m = np.load(os.path.join(d, "meta.npz"))
        self.meta = {k: m[k] for k in m.files}
        self.meta["metric_spacing"] = float(self.meta["metric_spacing"])
        t = np.load(os.path.join(d, "test.npz"))
        self.test = {k: torch.from_numpy(t[k]) for k in t.files}
        st, sp = self.meta["traj_start"], self.meta["traj_split"]
        self.train_k = [k for k in range(len(sp)) if sp[k] == 0]
        self.val = None
        if holdout_val:  # train without the validation recordings (make_val_split.py)
            v = np.load(os.path.join(d, "val.npz"))
            vk = set(int(k) for k in v["val_traj"])
            self.train_k = [k for k in self.train_k if k not in vk]
            self.val = {k: torch.from_numpy(v[k]) for k in ("ctx", "goal", "actions", "dist", "neg", "mask")}
        self.train_pool = np.concatenate([np.arange(st[k], st[k + 1]) for k in self.train_k])
        es = int(self.meta["end_slack"])
        self.train_cand = np.array([(k, c) for k in self.train_k for c in valid_currs(st[k + 1] - st[k], es)])

    def images(self, ctx, goal):
        """ctx [B,4] frame idx, goal [B] -> obs [B,12,96,96], goal [B,3,96,96] normalized, on device."""
        B = ctx.shape[0]
        f = self.frames[torch.cat([ctx.reshape(-1), goal.reshape(-1)])].to(self.device, non_blocking=True)
        f = f.permute(0, 3, 1, 2).float().div_(255)
        f = (f - MEAN.to(f.device)) / STD.to(f.device)
        obs = f[: B * (CTX + 1)].reshape(B, (CTX + 1) * 3, 96, 96)
        return obs, f[B * (CTX + 1):]

    def test_batches(self, bs=250):
        n = len(self.test["goal"])
        for i in range(0, n, bs):
            yield {k: v[i:i + bs] for k, v in self.test.items()}, i

    def sample_train(self, n, rng):
        """n random training samples (goals re-drawn each call, as the official loader does)."""
        idx = rng.randint(len(self.train_cand), size=n)
        rows = [make_sample(self.meta, int(self.train_cand[i][0]), int(self.train_cand[i][1]), rng, self.train_pool)
                for i in idx]
        return {"ctx": torch.tensor(np.array([r[0] for r in rows])), "goal": torch.tensor([r[1] for r in rows]),
                "actions": torch.from_numpy(np.stack([r[2] for r in rows])), "dist": torch.tensor([r[3] for r in rows]),
                "neg": torch.tensor([r[4] for r in rows]), "mask": torch.tensor([r[5] for r in rows])}


def norm_deltas(actions):
    """cumulative waypoints [B,8,2] -> normalized deltas in [-1,1] (official get_delta + normalize_data)."""
    d = torch.diff(actions, dim=1, prepend=torch.zeros_like(actions[:, :1]))
    lo, hi = ACTION_MIN.to(actions.device), ACTION_MAX.to(actions.device)
    return (d - lo) / (hi - lo) * 2 - 1


def unnorm_cumsum(nd):
    lo, hi = ACTION_MIN.to(nd.device), ACTION_MAX.to(nd.device)
    return torch.cumsum((nd + 1) / 2 * (hi - lo) + lo, dim=1)


def masked_mean(x, mask):
    m = mask.float()
    return float((x * m).sum() / m.sum().clamp(min=1))


@torch.no_grad()
def evaluate(model, sched, task, n_action_samples=4, seed=0, bs=250, cond_fn=None):
    """Deterministic metrics on the task's fixed test set.
    eps_mse_gc / eps_mse_uc: diffusion (training) loss with goal / goal masked, fixed noise+timesteps,
        averaged over action-masked samples (as the official loss).
    act_mse_gc / act_cos_gc: goal-conditioned sampled trajectories (n_action_samples DDPM samples, 10 steps)
        vs label (normalized units, official gc_action_loss / gc_multi_action_waypts_cos_sim), action-masked.
    act_mse_uc: same with the goal masked (exploration mode).  dist_mse: distance head, all samples.
    final_err_m: L2 error of the last (8th) waypoint in metres (goal-conditioned).
    cond_fn(model, obs, goal, mask_long) optionally replaces the vision encoder (adapters)."""
    dev = task.device
    g = torch.Generator(device=dev)
    sums = {k: [] for k in ("eps_mse_gc", "eps_mse_uc", "act_mse_gc", "act_cos_gc", "act_mse_uc", "dist_mse", "final_err_m")}
    masks = []
    enc = cond_fn or (lambda m, o, gl, mk: m("vision_encoder", obs_img=o, goal_img=gl, input_goal_mask=mk))
    for b, i0 in task.test_batches(bs):
        obs, goal = task.images(b["ctx"], b["goal"])
        B = obs.shape[0]
        act = b["actions"].to(dev).float(); mask = b["mask"].to(dev); dist = b["dist"].to(dev).float()
        g.manual_seed(seed * 100003 + i0)
        c_gc = enc(model, obs, goal, torch.zeros(B, dtype=torch.long, device=dev))
        c_uc = enc(model, obs, goal, torch.ones(B, dtype=torch.long, device=dev))
        na = norm_deltas(act)
        noise = torch.randn(na.shape, generator=g, device=dev)
        t = torch.randint(0, sched.config.num_train_timesteps, (B,), generator=g, device=dev)
        noisy = sched.add_noise(na, noise, t)
        for key, c in (("eps_mse_gc", c_gc), ("eps_mse_uc", c_uc)):
            pred = model("noise_pred_net", sample=noisy, timestep=t, global_cond=c)
            sums[key].append(((pred - noise) ** 2).mean(dim=(1, 2)))
        sums["dist_mse"].append((model("dist_pred_net", obsgoal_cond=c_gc).squeeze(-1) - dist) ** 2)
        for key, c in (("gc", c_gc), ("uc", c_uc)):
            mse, cos, fe = 0, 0, 0
            for s in range(n_action_samples):
                x = torch.randn(na.shape, generator=g, device=dev)
                sched.set_timesteps(sched.config.num_train_timesteps)
                for k in sched.timesteps:
                    eps = model("noise_pred_net", sample=x, timestep=k.to(dev).expand(B), global_cond=c)
                    x = sched.step(model_output=eps, timestep=k, sample=x, generator=g).prev_sample
                pa = unnorm_cumsum(x)
                mse = mse + ((pa - act) ** 2).mean(dim=(1, 2))
                cos = cos + F.cosine_similarity(pa.flatten(1), act.flatten(1), dim=-1)
                fe = fe + (pa[:, -1] - act[:, -1]).norm(dim=-1) * task.meta["metric_spacing"]
            if key == "gc":
                sums["act_mse_gc"].append(mse / n_action_samples); sums["act_cos_gc"].append(cos / n_action_samples)
                sums["final_err_m"].append(fe / n_action_samples)
            else:
                sums["act_mse_uc"].append(mse / n_action_samples)
        masks.append(mask)
    mask = torch.cat(masks)
    out = {k: masked_mean(torch.cat(v), mask) for k, v in sums.items() if k != "dist_mse"}
    out["dist_mse"] = float(torch.cat(sums["dist_mse"]).mean())
    out["n"] = int(len(mask)); out["n_masked"] = int(mask.sum())
    return out
