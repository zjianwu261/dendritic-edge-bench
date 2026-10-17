"""Continual-learning methods on top of a frozen / partly trainable NoMaD (T4a step 3, first version).

encode(): re-implementation of NoMaD_ViNT.forward with two insertion points
   token adapter  : on the [B, ctx+2, 256] observation/goal tokens before positional encoding + transformer
   output adapter : on the pooled 256-d conditioning vector (input of the diffusion UNet and distance head)
Checked to equal the official forward bit-for-bit when no adapter is given (test_encode()).

Methods
  ft        full fine-tuning (all NoMaD weights)
  ft_head   fine-tune diffusion UNet + distance head only (vision encoder frozen)
  lora      LoRA (rank r) on every transformer weight of the vision encoder (in_proj, out_proj, FFN) via
            torch parametrizations; backbone frozen
  replay    ft + reservoir replay buffer of <= 500 past samples (frame indices + labels; i.e. stores raw data)
  dend      dendritic context-gated adapter (Active-Dendrites style, Iyer et al. 2022):
              y = x + Up( kWTA( Down(x) * sigmoid( d_{s*}(z) ) ) ),  d_s(z) = <seg_s,h , z>,
              s* = argmax_s |d_s(z)| (absolute-max gating), z = context vector
            z = task prototype: mean of the *frozen* backbone feature (goal-masked conditioning vector) over
            the task's training samples (stored: 256 floats per task, no raw data).
            context "id": the true task's prototype; context "infer": nearest stored prototype to the
            sample's own frozen feature (cosine). Up is zero-initialised (adapter starts as identity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.utils.checkpoint import checkpoint

GRAD_CKPT = False  # set by cl_run --grad_ckpt: recompute EfficientNet activations in backward


def _feat(enc, x):
    if GRAD_CKPT and torch.is_grad_enabled() and any(p.requires_grad for p in enc.parameters()):
        return checkpoint(enc.extract_features, x, use_reentrant=False)
    return enc.extract_features(x)


def encode(venc, obs_img, goal_img, goal_mask, tok_adapter=None, out_adapter=None, z=None):
    cs = venc.context_size
    og = torch.cat([obs_img[:, 3 * cs:], goal_img], dim=1)
    g = venc.goal_encoder._avg_pooling(_feat(venc.goal_encoder, og))
    if venc.goal_encoder._global_params.include_top:
        g = venc.goal_encoder._dropout(g.flatten(start_dim=1))
    g = venc.compress_goal_enc(g)
    if g.dim() == 2:
        g = g.unsqueeze(1)
    o = torch.cat(torch.split(obs_img, 3, dim=1), dim=0)
    o = venc.obs_encoder._avg_pooling(_feat(venc.obs_encoder, o))
    if venc.obs_encoder._global_params.include_top:
        o = venc.obs_encoder._dropout(o.flatten(start_dim=1))
    o = venc.compress_obs_enc(o).unsqueeze(1)
    o = o.reshape((cs + 1, -1, venc.obs_encoding_size)).transpose(0, 1)
    tok = torch.cat((o, g), dim=1)
    if tok_adapter is not None:
        tok = tok_adapter(tok, z)
    dev = obs_img.device
    kpm = torch.index_select(venc.all_masks.to(dev), 0, goal_mask.long())
    tok = venc.positional_encoding(tok)
    tok = venc.sa_encoder(tok, src_key_padding_mask=kpm)
    avg = torch.index_select(venc.avg_pool_mask.to(dev), 0, goal_mask.long()).unsqueeze(-1)
    c = torch.mean(tok * avg, dim=1)
    if out_adapter is not None:
        c = out_adapter(c, z)
    return c


class KWTA(nn.Module):
    def __init__(self, frac):
        super().__init__(); self.frac = frac

    def forward(self, x):
        if self.frac >= 1:
            return x
        k = max(1, int(round(self.frac * x.shape[-1])))
        thr = x.topk(k, dim=-1).values[..., -1:]
        return x * (x >= thr).to(x.dtype)


class DendriticAdapter(nn.Module):
    def __init__(self, d=256, h=128, n_seg=8, ctx_dim=256, kwta=0.25):
        super().__init__()
        self.down = nn.Linear(d, h)
        self.up = nn.Linear(h, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.seg = nn.Parameter(torch.randn(n_seg, h, ctx_dim) * (1.0 / ctx_dim ** 0.5))
        self.kwta = KWTA(kwta)
        self.last_seg = None

    def gate(self, z):
        dend = torch.einsum("shc,bc->bsh", self.seg, z)          # [B, S, H]
        idx = dend.abs().argmax(dim=1, keepdim=True)              # absolute-max segment per unit
        self.last_seg = idx.squeeze(1)
        return torch.sigmoid(torch.gather(dend, 1, idx).squeeze(1))  # [B, H]

    def forward(self, x, z):
        g = self.gate(z)
        a = F.relu(self.down(x))
        a = a * (g.unsqueeze(1) if x.dim() == 3 else g)
        return x + self.up(self.kwta(a))


class LoRA(nn.Module):
    """Parametrization W -> W + (alpha/r) B A for a [out, in] weight."""

    def __init__(self, out_f, in_f, r=8, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_f) / in_f ** 0.5)
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.s = alpha / r

    def forward(self, W):
        return W + self.s * self.B @ self.A


def add_lora(venc, r=8):
    params = []
    for layer in venc.sa_encoder.layers:
        for mod, name in ((layer.self_attn, "in_proj_weight"), (layer.self_attn.out_proj, "weight"),
                          (layer.linear1, "weight"), (layer.linear2, "weight")):
            W = getattr(mod, name)
            lo = LoRA(W.shape[0], W.shape[1], r).to(W.device)
            P.register_parametrization(mod, name, lo)
            params += list(lo.parameters())
    return params


def test_encode():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import load_model
    torch.backends.mha.set_fastpath_enabled(False)
    m, _ = load_model("nomad")
    o, g = torch.randn(3, 12, 96, 96), torch.randn(3, 3, 96, 96)
    for mk in (torch.zeros(3).long(), torch.ones(3).long(), torch.tensor([0, 1, 0])):
        with torch.no_grad():
            a = m("vision_encoder", obs_img=o, goal_img=g, input_goal_mask=mk)
            b = encode(m.vision_encoder, o, g, mk)
        print("encode max abs diff vs official:", float((a - b).abs().max()))


if __name__ == "__main__":
    test_encode()


class DendriticAdapterV2(nn.Module):
    """v2: one dendritic segment per unit per context (allocated when the context appears);
    absolute-max segment selection over the allocated segments; sparse kWTA; gradient masking of the
    shared Down/Up weights to the units active for the current context; old contexts' segments frozen
    (done by the caller through snapshot/restore, see cl_run.py).
    The new context's segment is initialised aligned with its (standardised) context vector so that it
    produces |d| ~ seg_gain for its own context: seg[k, u] = s_u * seg_gain * z_k / |z_k|^2, s_u = +-1."""

    def __init__(self, d=256, h=192, max_ctx=8, ctx_dim=272, kwta=0.05, seg_gain=3.0):
        super().__init__()
        self.down = nn.Linear(d, h)
        self.up = nn.Linear(h, d, bias=False)
        nn.init.zeros_(self.up.weight)
        self.seg = nn.Parameter(torch.zeros(max_ctx, h, ctx_dim))
        self.n_ctx = 0
        self.kwta = KWTA(kwta)
        self.seg_gain = seg_gain
        self.last_seg = None; self.last_active = None
        # temp_norm=True -> gate = sigmoid(seg_gain * d / target_k) with target_k
        # the init gain of the selected segment, so every context's gate starts at sigmoid(+-seg_gain) (default off)
        self.temp_norm = False

    @torch.no_grad()
    def add_context(self, z, gen=None, eps=0.0, orth=False, margin=1.5):
        """eps > 0 (v3fix): seg = s * gain * z / (|z|^2 + eps), so a near-zero context vector can
        no longer produce a huge segment that wins the absolute-max selection for every other context.
        orth (v3fix): the segment direction is the part of z orthogonal to the earlier contexts' vectors (so the new
        segment gives d = 0 on every earlier prototype and cannot steal it), and its gain is raised to
        max(seg_gain, margin * max |d_old(z)|) so that it wins on its own prototype against all earlier segments
        (eps / std_ref alone did not guarantee this)."""
        k = self.n_ctx
        s = (torch.randint(0, 2, (self.seg.shape[1], 1), generator=gen, device=z.device) * 2 - 1).float()
        if not orth:
            self.seg[k] = s * self.seg_gain * z.view(1, -1) / (z.pow(2).sum() + eps)
        else:
            if not hasattr(self, "ctx_z"):
                self.ctx_z, self.gains = [], []
            zp = z.clone()
            if self.ctx_z:
                Qm, _ = torch.linalg.qr(torch.stack(self.ctx_z, 1))
                zp = z - Qm @ (Qm.T @ z)
            dold = float(torch.einsum("shc,c->sh", self.seg[:k], z).abs().max()) if k else 0.0
            target = max(self.seg_gain, margin * dold)
            n2 = float(zp.pow(2).sum())
            self.seg[k] = s * target * zp.view(1, -1) / n2  # d_k(z) = target exactly
            self.ctx_z.append(z.clone()); self.gains.append(target)
        self.n_ctx += 1
        return k

    def gate(self, z):
        dend = torch.einsum("shc,bc->bsh", self.seg[: self.n_ctx], z)
        idx = dend.abs().argmax(dim=1, keepdim=True)
        self.last_seg = idx.squeeze(1)
        d = torch.gather(dend, 1, idx).squeeze(1)
        if self.temp_norm:
            tg = torch.tensor(self.gains, device=d.device, dtype=d.dtype)[self.last_seg]  # [B, h]
            d = self.seg_gain * d / tg
        return torch.sigmoid(d)

    def forward(self, x, z):
        g = self.gate(z)
        a = F.relu(self.down(x)) * (g.unsqueeze(1) if x.dim() == 3 else g)
        a = self.kwta(a)
        act = (a > 0)
        self.last_active = act.reshape(-1, act.shape[-1]).any(0)
        self.last_act_samples = act.reshape(x.shape[0], -1, act.shape[-1]).any(1)  # [B, h]
        flat = act.reshape(-1, act.shape[-1])
        self.last_act_count = flat.float().sum(0); self.last_act_n = flat.shape[0]  # per (sample, token) instance
        return x + self.up(a)


class LoRABank(nn.Module):
    """LoRA bank: one independent LoRA per context on the same weight; `active` (set by the caller
    through set_bank_active) selects which context's low-rank update is applied. Context k's B starts at zero
    when the bank is created, so an untrained context equals the backbone. Only the active context receives
    gradients (others have grad None and are skipped by AdamW), so contexts trained earlier stay unchanged."""

    def __init__(self, out_f, in_f, n_ctx, r=10, alpha=16):
        super().__init__()
        # separate tensors per context: an inactive context is not in the graph -> grad None -> AdamW skips it
        # (no drift of old contexts through stale Adam moments)
        self.A = nn.ParameterList([nn.Parameter(torch.randn(r, in_f) / in_f ** 0.5) for _ in range(n_ctx)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(out_f, r)) for _ in range(n_ctx)])
        self.s = alpha / r
        self.active = 0

    def forward(self, W):
        k = self.active
        if k < 0:  # -1 = no context: plain frozen backbone (used for the context inferer's features)
            return W
        return W + self.s * self.B[k] @ self.A[k]


def add_lora_bank(venc, n_ctx, r=10):
    banks = []
    for layer in venc.sa_encoder.layers:
        for mod, name in ((layer.self_attn, "in_proj_weight"), (layer.self_attn.out_proj, "weight"),
                          (layer.linear1, "weight"), (layer.linear2, "weight")):
            W = getattr(mod, name)
            lo = LoRABank(W.shape[0], W.shape[1], n_ctx, r).to(W.device)
            P.register_parametrization(mod, name, lo)
            banks.append(lo)
    return banks


def set_bank_active(banks, k):
    for b in banks:
        b.active = int(k)


class BiasAdapter(nn.Module):
    """"Only learn an output bias" reference: a single shared learnable
    bias on the 256-d conditioning vector (input of the diffusion UNet and the distance head); backbone frozen."""

    def __init__(self, d=256):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, c, z=None):
        return c + self.b
