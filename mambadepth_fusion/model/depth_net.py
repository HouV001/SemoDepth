import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .fusion_blocks import DepthNormalizer, RFMConvBlock, MultiDirMambaFusion, RadarCenteredMamba
from .radar_gse import BatchRadarGSE


class MambaFusionDepth(nn.Module):

    def __init__(self, min_depth=0.5, max_depth=80.0, lambda_grad=0.5, lambda_linear=1.0,
                 lambda_log=1.0, lambda_sparse=1.0, radar_proj_ch=64, gnn_dim=64, gnn_layers=3,
                 max_radar_points=512, mamba_d_model=256, mamba_n_layers=2, window_size=8,
                 window_mamba_d_model=128, use_mono_depth=False):
        super().__init__()
        self.use_mono_depth = use_mono_depth
        self.min_d = min_depth
        self.max_d = max_depth
        self.lambda_grad = lambda_grad
        self.lambda_linear = lambda_linear
        self.lambda_log = lambda_log
        self.lambda_sparse = lambda_sparse
        self.dn = DepthNormalizer(min_depth, max_depth)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.enc_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc_pool = resnet.maxpool
        self.enc_layer1 = resnet.layer1
        self.enc_layer2 = resnet.layer2
        self.enc_layer3 = resnet.layer3
        self.enc_layer4 = resnet.layer4
        self.radar_gse = BatchRadarGSE(hidden_dim=gnn_dim, out_dim=gnn_dim, n_layers=gnn_layers, max_points=max_radar_points, max_depth=max_depth)
        if use_mono_depth:
            self.mono_embed = nn.Sequential(nn.Conv2d(1, gnn_dim // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(gnn_dim // 2, gnn_dim, 3, padding=1))
            nn.init.zeros_(self.mono_embed[-1].weight)
            nn.init.zeros_(self.mono_embed[-1].bias)
        rpc = radar_proj_ch
        self.radar_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(gnn_dim, rpc, 1), nn.ReLU(inplace=True)) for _ in range(5)])
        self.mamba_fuse4 = MultiDirMambaFusion(512, rpc, d_model=mamba_d_model, n_layers=mamba_n_layers)
        self.mamba_fuse3 = MultiDirMambaFusion(256 + 256, rpc, d_model=mamba_d_model, n_layers=mamba_n_layers)
        self.radar_mamba2 = RadarCenteredMamba(img_ch=256 + 128, radar_ch=rpc, d_model=window_mamba_d_model, window_size=window_size, n_layers=1)
        self.dec4 = RFMConvBlock(512, rpc, 256)
        self.dec3 = RFMConvBlock(256 + 256, rpc, 256)
        self.dec2 = RFMConvBlock(256 + 128, rpc, 128)
        self.dec1 = RFMConvBlock(128 + 64, rpc, 64)
        self.dec0 = RFMConvBlock(64 + 64, rpc, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def get_encoder_params(self):
        return list(self.enc_conv1.parameters()) + list(self.enc_pool.parameters()) + list(self.enc_layer1.parameters()) + list(self.enc_layer2.parameters()) + list(self.enc_layer3.parameters()) + list(self.enc_layer4.parameters())

    def get_decoder_params(self):
        params = list(self.radar_gse.parameters())
        modules = [*self.radar_proj, self.dec4, self.dec3, self.dec2, self.dec1, self.dec0,
                   self.mamba_fuse4, self.mamba_fuse3, self.radar_mamba2, self.head]
        if self.use_mono_depth:
            modules.append(self.mono_embed)
        for m in modules:
            params.extend(m.parameters())
        return params

    def _encode_image(self, image):
        x = image.permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        c0 = self.enc_conv1(x)
        x = self.enc_pool(c0)
        c1 = self.enc_layer1(x)
        c2 = self.enc_layer2(c1)
        c3 = self.enc_layer3(c2)
        c4 = self.enc_layer4(c3)
        return (c0, c1, c2, c3, c4)

    def _build_radar_info(self, sparse_depth):
        return self.radar_gse(sparse_depth)

    def _fuse_mono(self, radar_info, mono_depth):
        if not self.use_mono_depth or mono_depth is None:
            return radar_info
        m = mono_depth.unsqueeze(1) / self.max_d
        m = m.clamp(0.0, 1.0)
        mono_feat = self.mono_embed(m)
        if mono_feat.shape[-2:] != radar_info.shape[-2:]:
            mono_feat = F.interpolate(mono_feat, size=radar_info.shape[-2:], mode='bilinear', align_corners=False)
        return radar_info + mono_feat

    @staticmethod
    def _downsample(x, target_size):
        return F.interpolate(x, target_size, mode='bilinear', align_corners=False)

    def _decode(self, c0, c1, c2, c3, c4, radar_info, sparse_depth):

        def up(x, target):
            return F.interpolate(x, target.shape[2:], mode='bilinear', align_corners=False)
        r4 = self.radar_proj[0](self._downsample(radar_info, c4.shape[2:]))
        r3 = self.radar_proj[1](self._downsample(radar_info, c3.shape[2:]))
        r2 = self.radar_proj[2](self._downsample(radar_info, c2.shape[2:]))
        r1 = self.radar_proj[3](self._downsample(radar_info, c1.shape[2:]))
        r0 = self.radar_proj[4](self._downsample(radar_info, c0.shape[2:]))
        c4_fused = self.mamba_fuse4(c4, r4)
        d4 = self.dec4(c4_fused, r4)
        l3_in = torch.cat([up(d4, c3), c3], dim=1)
        l3_fused = self.mamba_fuse3(l3_in, r3)
        d3 = self.dec3(l3_fused, r3)
        l2_in = torch.cat([up(d3, c2), c2], dim=1)
        l2_fused = self.radar_mamba2(l2_in, r2, sparse_depth)
        d2 = self.dec2(l2_fused, r2)
        d1 = self.dec1(torch.cat([up(d2, c1), c1], dim=1), r1)
        feat = self.dec0(torch.cat([up(d1, c0), c0], dim=1), r0)
        feat = F.interpolate(feat, radar_info.shape[2:], mode='bilinear', align_corners=False)
        return self.head(feat)

    def forward_train(self, image, sparse_depth, gt_depth, sparse_gt=None, mono_depth=None):
        device = image.device
        c0, c1, c2, c3, c4 = self._encode_image(image)
        radar_info = self._build_radar_info(sparse_depth)
        radar_info = self._fuse_mono(radar_info, mono_depth)
        pred_norm = self._decode(c0, c1, c2, c3, c4, radar_info, sparse_depth)
        gt_valid = (gt_depth > self.min_d) & (gt_depth < self.max_d)
        if gt_valid.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero, {'loss': 0, 'loss_log': 0, 'loss_linear': 0, 'loss_grad': 0, 'loss_sparse': 0, 'mae_mm': 0})
        gt_norm = self.dn.normalize(gt_depth.clamp(min=self.min_d)).unsqueeze(1)
        mask = gt_valid.unsqueeze(1).float()
        pred_meters = self.dn.denormalize(pred_norm.clamp(-1, 1))
        have_sparse = False
        if sparse_gt is not None:
            sg_valid = (sparse_gt > self.min_d) & (sparse_gt < self.max_d)
            if sg_valid.sum() > 0:
                sg_norm = self.dn.normalize(sparse_gt.clamp(min=self.min_d)).unsqueeze(1)
                sg_mask = sg_valid.unsqueeze(1).float()
                have_sparse = True
        if have_sparse:
            loss_log = (torch.abs(pred_norm - sg_norm) * sg_mask).sum() / sg_mask.sum()
        else:
            loss_log = (torch.abs(pred_norm - gt_norm) * mask).sum() / mask.sum()
        huber = F.huber_loss(pred_meters, gt_depth.unsqueeze(1), reduction='none', delta=5.0)
        loss_linear = (huber * mask).sum() / mask.sum() / self.max_d
        pdy = pred_norm[:, :, :-1, :] - pred_norm[:, :, 1:, :]
        pdx = pred_norm[:, :, :, :-1] - pred_norm[:, :, :, 1:]
        gdy = gt_norm[:, :, :-1, :] - gt_norm[:, :, 1:, :]
        gdx = gt_norm[:, :, :, :-1] - gt_norm[:, :, :, 1:]
        mdy = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        mdx = mask[:, :, :, :-1] * mask[:, :, :, 1:]
        loss_grad = (torch.abs(pdx - gdx) * mdx).sum() / mdx.sum().clamp(min=1) + (torch.abs(pdy - gdy) * mdy).sum() / mdy.sum().clamp(min=1)
        loss_sparse = torch.tensor(0.0, device=device)
        if have_sparse:
            loss_sparse = (torch.abs(pred_meters - sparse_gt.unsqueeze(1)) * sg_mask).sum() / sg_mask.sum() / self.max_d
        loss = self.lambda_log * loss_log + self.lambda_linear * loss_linear + self.lambda_grad * loss_grad + self.lambda_sparse * loss_sparse
        with torch.no_grad():
            mae_m = (torch.abs(pred_meters - gt_depth.unsqueeze(1)) * mask).sum() / mask.sum() * 1000
        return (loss, {'loss': loss.item(), 'loss_log': loss_log.item(), 'loss_linear': loss_linear.item(), 'loss_grad': loss_grad.item(), 'loss_sparse': loss_sparse.item(), 'mae_mm': mae_m.item()})

    @torch.no_grad()
    def forward_test(self, image, sparse_depth, mono_depth=None, paste_radar=False):
        c0, c1, c2, c3, c4 = self._encode_image(image)
        radar_info = self._build_radar_info(sparse_depth)
        radar_info = self._fuse_mono(radar_info, mono_depth)
        pred_norm = self._decode(c0, c1, c2, c3, c4, radar_info, sparse_depth)
        depth = self.dn.denormalize(pred_norm.clamp(-1, 1))
        if paste_radar:
            radar_mask = (sparse_depth > 0).unsqueeze(1)
            depth[radar_mask] = sparse_depth.unsqueeze(1)[radar_mask]
        return (depth, None)

    def load_pretrained_mae(self, pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if 'gse_state_dict' in ckpt:
            self.radar_gse.gse.load_state_dict(ckpt['gse_state_dict'], strict=False)
            print(f'Loaded pre-trained GSE from {pretrained_path}')
        elif 'gnn_state_dict' in ckpt:
            missing, unexpected = self.radar_gse.gse.load_state_dict(ckpt['gnn_state_dict'], strict=False)
            print(f'Loaded legacy GNN weights from {pretrained_path} (missing={len(missing)}, unexpected={len(unexpected)})')

    def save(self, p):
        torch.save(self.state_dict(), p)

    def load(self, p):
        self.load_state_dict(torch.load(p, map_location='cpu'), strict=False)
