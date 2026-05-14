import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class CrossModalMamba(nn.Module):
    """Mamba block with radar-modulated step size (Δ) and readout (C) — RMS BE mode.

    Δ and C are augmented from a radar token stream; B and A remain image-only.
    Radar-side projections are zero-initialized (gate bias -2) so the block starts
    numerically equivalent to vanilla Mamba and must *learn* to use radar.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        d_inner = self.mamba.d_inner
        self.radar_dt_proj = nn.Linear(d_model, d_inner, bias=False)
        self.radar_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.radar_dt_proj.weight)
        nn.init.zeros_(self.radar_gate.weight)
        nn.init.constant_(self.radar_gate.bias, -2.0)
        self.radar_c_proj = nn.Linear(d_model, d_state, bias=False)
        nn.init.zeros_(self.radar_c_proj.weight)

    def forward(self, x_img, x_rad=None):
        if x_rad is None:
            return self.mamba(x_img)
        m = self.mamba
        batch, seqlen, _ = x_img.shape
        xz = rearrange(m.in_proj.weight @ rearrange(x_img, 'b l d -> d (b l)'), 'd (b l) -> b d l', l=seqlen)
        if m.in_proj.bias is not None:
            xz = xz + rearrange(m.in_proj.bias.to(dtype=xz.dtype), 'd -> d 1')
        A = -torch.exp(m.A_log.float())
        x, z = xz.chunk(2, dim=1)
        x = m.act(m.conv1d(x)[..., :seqlen])
        x_dbl = m.x_proj(rearrange(x, 'b d l -> (b l) d'))
        dt_img, B_img, C_img = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
        dt_img = m.dt_proj.weight @ dt_img.t()
        dt_img = rearrange(dt_img, 'd (b l) -> b d l', l=seqlen)
        B_img = rearrange(B_img, '(b l) n -> b n l', l=seqlen).contiguous()
        C_img = rearrange(C_img, '(b l) n -> b n l', l=seqlen).contiguous()
        dt_rad = self.radar_dt_proj(x_rad)
        dt_rad = rearrange(dt_rad, 'b l d -> b d l')
        gate = torch.sigmoid(self.radar_gate(x_rad))
        gate = rearrange(gate, 'b l n -> b n l')
        dt_rad = dt_rad.to(dtype=dt_img.dtype)
        gate = gate.to(dtype=dt_img.dtype)
        dt = dt_img + gate * dt_rad
        C_rad = self.radar_c_proj(x_rad)
        C_rad = rearrange(C_rad, 'b l n -> b n l').contiguous()
        C = C_img + C_rad.to(dtype=C_img.dtype)
        y = selective_scan_fn(x, dt, A, B_img, C, m.D.float(), z=z, delta_bias=m.dt_proj.bias.float(), delta_softplus=True)
        y = rearrange(y, 'b d l -> b l d')
        return m.out_proj(y)
