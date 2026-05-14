import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_modal_mamba import CrossModalMamba


class DepthNormalizer:

    def __init__(self, mn=0.5, mx=80.0):
        self.mn, self.mx = (mn, mx)
        self.log_mn, self.log_mx = (math.log(mn), math.log(mx))
        self.log_r = self.log_mx - self.log_mn

    def normalize(self, d):
        return 2.0 * (torch.log(d.clamp(min=self.mn, max=self.mx)) - self.log_mn) / self.log_r - 1.0

    def denormalize(self, x):
        return torch.exp((x + 1.0) / 2.0 * self.log_r + self.log_mn)


class RFMConvBlock(nn.Module):

    def __init__(self, img_ch, radar_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(img_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.mod1 = nn.Sequential(nn.Conv2d(radar_ch, out_ch, 1), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch * 2, 1))
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.mod2 = nn.Sequential(nn.Conv2d(radar_ch, out_ch, 1), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch * 2, 1))
        nn.init.zeros_(self.mod1[-1].weight)
        nn.init.zeros_(self.mod1[-1].bias)
        nn.init.zeros_(self.mod2[-1].weight)
        nn.init.zeros_(self.mod2[-1].bias)

    def forward(self, img_feat, radar_feat):
        x = self.bn1(self.conv1(img_feat))
        g1, b1 = self.mod1(radar_feat).chunk(2, dim=1)
        x = F.relu((1 + g1) * x + b1)
        x = self.bn2(self.conv2(x))
        g2, b2 = self.mod2(radar_feat).chunk(2, dim=1)
        x = F.relu((1 + g2) * x + b2)
        return x


def _make_mamba(d_model, d_state, d_conv, expand):
    return CrossModalMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class MultiDirMambaFusion(nn.Module):
    """Global 4-direction Mamba scan over pure-image tokens, with radar entering via Δ/C in the scan."""

    def __init__(self, img_ch, radar_ch, d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.img_proj = nn.Sequential(nn.Conv2d(img_ch, d_model, 1), nn.ReLU(inplace=True))
        self.rad_proj = nn.Sequential(nn.Conv2d(radar_ch, d_model, 1), nn.ReLU(inplace=True))
        self.n_layers = n_layers
        self.mamba_layers = nn.ModuleList([nn.ModuleDict({
            'norm': nn.LayerNorm(d_model),
            'norm_rad': nn.LayerNorm(d_model),
            'mamba_lr': _make_mamba(d_model, d_state, d_conv, expand),
            'mamba_rl': _make_mamba(d_model, d_state, d_conv, expand),
            'mamba_tb': _make_mamba(d_model, d_state, d_conv, expand),
            'mamba_bt': _make_mamba(d_model, d_state, d_conv, expand),
            'merge': nn.Linear(d_model * 4, d_model),
        }) for _ in range(n_layers)])
        self.out_proj = nn.Sequential(nn.Conv2d(d_model, img_ch, 1), nn.BatchNorm2d(img_ch), nn.ReLU(inplace=True))

    def _scan_directions(self, tokens_2d, rad_tokens_2d, layer):
        B, H, W, D = tokens_2d.shape
        normed = layer['norm'](tokens_2d)
        normed_rad = layer['norm_rad'](rad_tokens_2d)
        lr = normed.reshape(B, H * W, D)
        lr_rad = normed_rad.reshape(B, H * W, D)
        lr_out = layer['mamba_lr'](lr, lr_rad).reshape(B, H, W, D)
        rl = normed.flip(2).reshape(B, H * W, D)
        rl_rad = normed_rad.flip(2).reshape(B, H * W, D)
        rl_out = layer['mamba_rl'](rl, rl_rad).reshape(B, H, W, D).flip(2)
        tb = normed.permute(0, 2, 1, 3).reshape(B, H * W, D)
        tb_rad = normed_rad.permute(0, 2, 1, 3).reshape(B, H * W, D)
        tb_out = layer['mamba_tb'](tb, tb_rad).reshape(B, W, H, D).permute(0, 2, 1, 3)
        bt = normed.permute(0, 2, 1, 3).flip(2).reshape(B, H * W, D)
        bt_rad = normed_rad.permute(0, 2, 1, 3).flip(2).reshape(B, H * W, D)
        bt_out = layer['mamba_bt'](bt, bt_rad).reshape(B, W, H, D).flip(2).permute(0, 2, 1, 3)
        merged = torch.cat([lr_out, rl_out, tb_out, bt_out], dim=-1)
        merged = layer['merge'](merged)
        return tokens_2d + merged

    def forward(self, img_feat, radar_feat):
        img = self.img_proj(img_feat)
        rad = self.rad_proj(radar_feat)
        tokens = img.permute(0, 2, 3, 1)
        rad_tokens = rad.permute(0, 2, 3, 1)
        for layer in self.mamba_layers:
            tokens = self._scan_directions(tokens, rad_tokens, layer)
        out = tokens.permute(0, 3, 1, 2)
        return img_feat + self.out_proj(out)


class RadarCenteredMamba(nn.Module):
    """Windowed 4-direction Mamba scan around radar pixels with Gaussian-blended scatter back."""

    def __init__(self, img_ch, radar_ch, d_model=128, window_size=8, n_layers=1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.ws = window_size
        self.sigma = window_size / 2.5
        self.img_proj = nn.Sequential(nn.Conv2d(img_ch, d_model, 1), nn.ReLU(inplace=True))
        self.rad_proj = nn.Sequential(nn.Conv2d(radar_ch, d_model, 1), nn.ReLU(inplace=True))
        self.mamba_layers = nn.ModuleList([nn.ModuleDict({
            'norm': nn.LayerNorm(d_model),
            'norm_rad': nn.LayerNorm(d_model),
            'mamba_lr': _make_mamba(d_model, d_state, d_conv, expand),
            'mamba_rl': _make_mamba(d_model, d_state, d_conv, expand),
            'mamba_tb': _make_mamba(d_model, d_state, d_conv, expand),
            'mamba_bt': _make_mamba(d_model, d_state, d_conv, expand),
            'merge': nn.Linear(d_model * 4, d_model),
        }) for _ in range(n_layers)])
        self.out_proj = nn.Conv2d(d_model, img_ch, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _get_radar_positions(self, sparse_depth, H, W):
        B = sparse_depth.shape[0]
        device = sparse_depth.device
        radar_mask = (sparse_depth > 0).float().unsqueeze(1)
        radar_mask_scaled = F.adaptive_max_pool2d(radar_mask, (H, W)).squeeze(1)
        positions = []
        for b in range(B):
            ys, xs = torch.where(radar_mask_scaled[b] > 0)
            ys = ys.clamp(self.ws // 2, H - 1 - self.ws // 2)
            xs = xs.clamp(self.ws // 2, W - 1 - self.ws // 2)
            positions.append(torch.stack([ys, xs], dim=1))
        return positions

    def _extract_windows(self, feat, positions, H, W):
        ws = self.ws
        half = ws // 2
        B, C = (feat.shape[0], feat.shape[1])
        device = feat.device
        batch_ids_list, cy_list, cx_list = ([], [], [])
        for b in range(B):
            pos = positions[b]
            if len(pos) > 0:
                batch_ids_list.append(torch.full((len(pos),), b, device=device, dtype=torch.long))
                cy_list.append(pos[:, 0])
                cx_list.append(pos[:, 1])
        if len(batch_ids_list) == 0:
            return (None, None, None, None)
        batch_ids = torch.cat(batch_ids_list)
        cy_all = torch.cat(cy_list)
        cx_all = torch.cat(cx_list)
        N_total = len(batch_ids)
        y0 = (cy_all - half).clamp(0, H - ws)
        x0 = (cx_all - half).clamp(0, W - ws)
        dy = torch.arange(ws, device=device)
        dx = torch.arange(ws, device=device)
        abs_y = (y0[:, None, None] + dy[None, :, None]).expand(N_total, ws, ws).long()
        abs_x = (x0[:, None, None] + dx[None, None, :]).expand(N_total, ws, ws).long()
        flat_idx = batch_ids[:, None, None] * H * W + abs_y * W + abs_x
        feat_flat = feat.reshape(B * H * W, C)
        windows = feat_flat[flat_idx.reshape(-1)].reshape(N_total, ws, ws, C).permute(0, 3, 1, 2)
        return (windows, batch_ids, y0, x0)

    def _scan_4dir_windows(self, windows, rad_windows):
        N, D, H, W = windows.shape
        tokens = windows.permute(0, 2, 3, 1)
        rad_tokens = rad_windows.permute(0, 2, 3, 1)
        for layer in self.mamba_layers:
            normed = layer['norm'](tokens)
            normed_rad = layer['norm_rad'](rad_tokens)
            lr = normed.reshape(N, H * W, D)
            lr_rad = normed_rad.reshape(N, H * W, D)
            lr_out = layer['mamba_lr'](lr, lr_rad).reshape(N, H, W, D)
            rl = normed.flip(2).reshape(N, H * W, D)
            rl_rad = normed_rad.flip(2).reshape(N, H * W, D)
            rl_out = layer['mamba_rl'](rl, rl_rad).reshape(N, H, W, D).flip(2)
            tb = normed.permute(0, 2, 1, 3).reshape(N, H * W, D)
            tb_rad = normed_rad.permute(0, 2, 1, 3).reshape(N, H * W, D)
            tb_out = layer['mamba_tb'](tb, tb_rad).reshape(N, W, H, D).permute(0, 2, 1, 3)
            bt = normed.permute(0, 2, 1, 3).flip(2).reshape(N, H * W, D)
            bt_rad = normed_rad.permute(0, 2, 1, 3).flip(2).reshape(N, H * W, D)
            bt_out = layer['mamba_bt'](bt, bt_rad).reshape(N, W, H, D).flip(2).permute(0, 2, 1, 3)
            merged = torch.cat([lr_out, rl_out, tb_out, bt_out], dim=-1)
            merged = layer['merge'](merged)
            tokens = tokens + merged
        return tokens.permute(0, 3, 1, 2)

    def _scatter_windows(self, base_shape, windows, batch_ids, y0s, x0s, sigma, device):
        B, C, H, W = base_shape
        ws = self.ws
        N = len(batch_ids)
        if N == 0:
            return torch.zeros(B, C, H, W, device=device)
        half = ws // 2
        gy = torch.arange(ws, device=device).float() - half
        gx = torch.arange(ws, device=device).float() - half
        yy, xx = torch.meshgrid(gy, gx, indexing='ij')
        gauss = torch.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
        dy = torch.arange(ws, device=device).long()
        dx = torch.arange(ws, device=device).long()
        abs_y = (y0s[:, None].long() + dy[None, :]).clamp(0, H - 1)
        abs_x = (x0s[:, None].long() + dx[None, :]).clamp(0, W - 1)
        flat_idx = (batch_ids[:, None, None] * H * W + abs_y[:, :, None] * W + abs_x[:, None, :]).reshape(-1)
        g_windows = windows * gauss[None, None, :, :]
        g_flat = g_windows.permute(0, 2, 3, 1).reshape(-1, C)
        gauss_flat = gauss.reshape(-1).repeat(N)
        accum = torch.zeros(B * H * W, C, device=device)
        accum.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), g_flat)
        accum = accum.reshape(B, H, W, C).permute(0, 3, 1, 2)
        weight = torch.zeros(B * H * W, device=device)
        weight.scatter_add_(0, flat_idx, gauss_flat)
        weight = weight.reshape(B, 1, H, W)
        out = accum / weight.clamp(min=1e-06)
        out = out * (weight > 1e-06).float()
        return out

    def forward(self, img_feat, radar_feat, sparse_depth):
        B, _, H, W = img_feat.shape
        img = self.img_proj(img_feat)
        rad = self.rad_proj(radar_feat)
        positions = self._get_radar_positions(sparse_depth, H, W)
        result = self._extract_windows(img, positions, H, W)
        if result[0] is None:
            return img_feat
        windows, batch_ids, y0s, x0s = result
        r = self._extract_windows(rad, positions, H, W)
        if r[0] is None:
            return img_feat
        rad_windows = r[0]
        processed = self._scan_4dir_windows(windows, rad_windows)
        delta = self._scatter_windows((B, self.d_model, H, W), processed, batch_ids, y0s, x0s, self.sigma, img_feat.device)
        return img_feat + self.out_proj(delta)
