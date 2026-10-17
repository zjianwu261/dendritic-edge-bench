"""Shared helpers for T4a: paths, model loading (strict key check), DDPM scheduler.

The official visualnav-transformer loader uses load_state_dict(strict=False); here we load
with strict=False too but *report* missing / unexpected keys and fail if any weight is missing.
"""
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VNT = os.path.join(ROOT, "src/third_party/visualnav-transformer")
sys.path.insert(0, os.path.join(VNT, "train"))
sys.path.insert(0, os.path.join(ROOT, "src/third_party/diffusion_policy"))
DEB_DATA = os.environ.get("DEB_DATA", os.path.join(ROOT, "data"))
WEIGHTS = os.path.join(DEB_DATA, "gnm_weights/checkpoints")
DATA = os.path.join(DEB_DATA, "gnm")
EXP = os.path.join(ROOT, "results/T4a")

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402
from vint_train.models.gnm.gnm import GNM  # noqa: E402
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork  # noqa: E402
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn  # noqa: E402
from vint_train.models.vint.vint import ViNT  # noqa: E402


def load_config(name):
    with open(os.path.join(VNT, "train/config/defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(VNT, f"train/config/{name}.yaml")) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def build_model(name, cfg):
    if name == "nomad":
        venc = NoMaD_ViNT(obs_encoding_size=cfg["encoding_size"], context_size=cfg["context_size"],
                          mha_num_attention_heads=cfg["mha_num_attention_heads"],
                          mha_num_attention_layers=cfg["mha_num_attention_layers"],
                          mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
        venc = replace_bn_with_gn(venc)
        unet = ConditionalUnet1D(input_dim=2, global_cond_dim=cfg["encoding_size"],
                                 down_dims=cfg["down_dims"], cond_predict_scale=cfg["cond_predict_scale"])
        return NoMaD(vision_encoder=venc, noise_pred_net=unet,
                     dist_pred_net=DenseNetwork(embedding_dim=cfg["encoding_size"]))
    if name == "vint":
        return ViNT(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"],
                    learn_angle=cfg["learn_angle"], obs_encoder=cfg["obs_encoder"],
                    obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
                    mha_num_attention_heads=cfg["mha_num_attention_heads"],
                    mha_num_attention_layers=cfg["mha_num_attention_layers"],
                    mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
    if name == "gnm":
        return GNM(cfg["context_size"], cfg["len_traj_pred"], cfg["learn_angle"],
                   cfg["obs_encoding_size"], cfg["goal_encoding_size"])
    raise ValueError(name)


def load_model(name, device="cpu", verbose=True):
    cfg = load_config(name)
    model = build_model(name, cfg)
    ck = torch.load(os.path.join(WEIGHTS, f"{name}.pth"), map_location="cpu", weights_only=False)
    if name == "nomad":
        sd = ck
    else:
        m = ck["model"]
        sd = m.module.state_dict() if hasattr(m, "module") else m.state_dict()
    res = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[{name}] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
              f" params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    if res.missing_keys:
        raise RuntimeError(f"missing keys: {res.missing_keys[:10]}")
    return model.to(device).eval(), cfg


def ddpm_scheduler(cfg):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=cfg["num_diffusion_iters"], beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")
