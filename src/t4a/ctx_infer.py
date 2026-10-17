"""Shared context inferer for dend2 and lora_bank (both use the same inferer).

Context vector of a sample (computed from the frozen backbone and the current frame only, no labels):
  group "feat" : frozen goal-masked NoMaD conditioning vector (256)
  group "hist" : 16-bin brightness histogram of the current frame (gray, 0..255)
  group "low"  : [log10 variance of the Laplacian (sharpness), occlusion ratio]  (feature set "v3" only)
                 occlusion ratio = fraction of the 144 8x8 blocks whose gray std < OCC_STD (flat patches,
                 e.g. a lens obstruction)
Standardisation ("group", v3): per-dimension z-score with mean / std fixed from the first task's prototype
samples (std floored per group, FLOOR), then every group is divided by sqrt(its dimension) so that each group
contributes equally to the Euclidean distance.  Feature set "v2" (= the old cl_dend2 behaviour) is
[feat, hist] with per-dimension z-score and no group weighting.
Prototype of context k = mean standardised vector over the task's 512 prototype samples (stored: one vector +
one threshold per context, no raw data).  Inference: nearest prototype (Euclidean).  New-context detection:
min_k ||z - mu_k|| / tau_k > 1 with tau_k = q-quantile of the distances of context k's own *validation*
samples to mu_k (q calibrated on the v2sup validation run); without a validation split tau_k is
the 0.99-quantile of the prototype samples (old behaviour).

The frame is recovered exactly from the normalised observation tensor (round((x*std+mean)*255)), so the
inferer only needs (obs, goal), as at deployment.
"""
import math

import torch
import torch.nn.functional as F

from cl_methods import encode

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OCC_STD = 2.0
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
FLOOR = {"feat": 1e-3, "hist": 1e-2, "low": 5e-2}


def current_gray(obs):
    """obs [B, 12, 96, 96] normalised -> gray of the current (last) frame, [B, 96, 96], values k/3, k int."""
    x = obs[:, -3:]
    rgb = ((x * STD.to(x.device) + MEAN.to(x.device)) * 255).round().clamp(0, 255)
    return rgb.mean(1)


def low_level(gray):
    """gray [B,96,96] (0..255) -> hist16 [B,16], low [B,2] = [log10 lap var, occlusion ratio]."""
    B = gray.shape[0]
    bins = (gray / 256 * 16).long().clamp(0, 15).view(B, -1)
    hist = F.one_hot(bins, 16).float().mean(1)
    g = gray.unsqueeze(1) / 255.0
    lap = F.conv2d(g, LAP.to(g.device))
    lv = torch.log10(lap.flatten(1).var(1) + 1e-6)
    blocks = gray.view(B, 12, 8, 12, 8).permute(0, 1, 3, 2, 4).reshape(B, 144, 64)
    occ = (blocks.std(-1) < OCC_STD).float().mean(1)
    return hist, torch.stack([lv, occ], -1)


class ContextInferer:
    def __init__(self, venc, feat_set="v3", q=0.95, hist_w=1.0):
        self.venc, self.feat_set, self.q, self.hist_w = venc, feat_set, q, hist_w
        self.groups = [("feat", 256), ("hist", 16)] + ([("low", 2)] if feat_set == "v3" else [])
        self.dim = sum(d for _, d in self.groups)
        self.mu0 = self.sd0 = None
        self.mus, self.taus = [], []

    @torch.no_grad()
    def raw(self, obs, goal):
        f = encode(self.venc, obs, goal, torch.ones(obs.shape[0], dtype=torch.long, device=obs.device))
        hist, low = low_level(current_gray(obs))
        parts = [f, hist * self.hist_w] + ([low] if self.feat_set == "v3" else [])
        return torch.cat(parts, -1)

    def fit_std(self, zr):
        self.mu0 = zr.mean(0)
        sd = zr.std(0)
        if self.feat_set == "v3":
            fl, o = [], 0
            for g, d in self.groups:
                fl.append(torch.full((d,), FLOOR[g], device=zr.device)); o += d
            self.sd0 = torch.maximum(sd, torch.cat(fl))
            self.gw = torch.cat([torch.full((d,), 1.0 / math.sqrt(d), device=zr.device) for _, d in self.groups])
        else:
            self.sd0 = sd + 1e-3
            self.gw = torch.ones_like(sd)

    def std(self, zr):
        return (zr - self.mu0) / self.sd0 * self.gw

    def add_context(self, z_proto, z_val=None):
        """z_* standardised. Returns (k, mu)."""
        mu = z_proto.mean(0)
        if z_val is not None:
            tau = torch.quantile((z_val - mu).norm(dim=-1), self.q)
        else:
            tau = torch.quantile((z_proto - mu).norm(dim=-1), 0.99)
        self.mus.append(mu); self.taus.append(tau)
        return len(self.mus) - 1, mu

    def nearest(self, z):
        M = torch.stack(self.mus)
        d = torch.cdist(z, M)
        k = d.argmin(-1)
        novel = (d / torch.stack(self.taus)[None]).min(-1).values > 1
        return k, novel

    def novel_frac(self, z):
        if not self.mus:
            return None
        return float(self.nearest(z)[1].float().mean())

    def state(self):
        return {"feat_set": self.feat_set, "q": self.q, "mu0": self.mu0.cpu(), "sd0": self.sd0.cpu(), "gw": self.gw.cpu(),
                "mus": [m.cpu() for m in self.mus], "taus": [float(t) for t in self.taus]}
