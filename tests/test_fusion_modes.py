import math
import pytest
import torch
from mambadepth_fusion.model.cross_modal_mamba import CrossModalMamba
from mambadepth_fusion.model.radar_gse import BatchRadarGSE, RadarGSE

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = 'CUDA not available — mamba-ssm requires a GPU'
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)


def test_cross_modal_mamba_forward():
    d_model, B, L = (64, 2, 32)
    device = 'cuda'
    m = CrossModalMamba(d_model=d_model).to(device)
    x_img = torch.randn(B, L, d_model, device=device)
    x_rad = torch.randn(B, L, d_model, device=device)
    y = m(x_img, x_rad)
    assert y.shape == (B, L, d_model), f'expected {(B, L, d_model)}, got {y.shape}'
    assert torch.isfinite(y).all(), 'non-finite output'


def test_zero_init_parity():
    """Radar-side projections are zero-initialized, so output must match vanilla Mamba at init."""
    d_model, B, L = (64, 2, 32)
    torch.manual_seed(0)
    m_fus = CrossModalMamba(d_model=d_model).cuda()
    x_img = torch.randn(B, L, d_model, device='cuda')
    x_rad = torch.randn(B, L, d_model, device='cuda')
    y_with_rad = m_fus(x_img, x_rad)
    y_without_rad = m_fus(x_img)
    diff = (y_with_rad - y_without_rad).abs().max().item()
    assert diff < 0.001, f'zero-init parity broken, max diff = {diff}'


def test_gradients_flow_into_radar_params():
    d_model, B, L = (64, 2, 32)
    m = CrossModalMamba(d_model=d_model).cuda()
    x_img = torch.randn(B, L, d_model, device='cuda', requires_grad=True)
    x_rad = torch.randn(B, L, d_model, device='cuda', requires_grad=True)
    y = m(x_img, x_rad).sum()
    y.backward()
    radar_params = [(n, p) for n, p in m.named_parameters() if n.startswith('radar_')]
    assert len(radar_params) > 0, 'no radar-side parameters found'
    for name, p in radar_params:
        assert p.grad is not None, f'{name}: no gradient'


def test_gse_full_adjacency_forward():
    N = 24
    gse = RadarGSE(in_dim=3, hidden_dim=32, out_dim=32, n_layers=2).cuda()
    pts = torch.rand(N, 3, device='cuda')
    out = gse(pts)
    assert out.shape == (N, 32)
    assert torch.isfinite(out).all()


def test_gse_empty_input_is_safe():
    gse = RadarGSE(in_dim=3, hidden_dim=32, out_dim=32, n_layers=1).cuda()
    out = gse(torch.zeros(0, 3, device='cuda'))
    assert out.shape == (0, 32)


def test_batch_gse_subsamples_when_overcap():
    H, W, cap = (32, 32, 16)
    bg = BatchRadarGSE(hidden_dim=16, out_dim=16, n_layers=1, max_points=cap).cuda()
    sd = torch.zeros(1, H, W, device='cuda')
    idx = torch.randperm(H * W, device='cuda')[:100]
    ys = idx // W
    xs = idx % W
    sd[0, ys, xs] = torch.rand(100, device='cuda') * 40.0 + 1.0
    feat = bg(sd)
    assert feat.shape == (1, 16, H, W)
    nonzero = (feat.abs().sum(dim=1) > 0).sum().item()
    assert nonzero <= cap, f'expected ≤{cap} nonzero pixels, got {nonzero}'


def _tiny_depth_net():
    from mambadepth_fusion.model import MambaFusionDepth
    return MambaFusionDepth(gnn_layers=1, max_radar_points=32, mamba_n_layers=1).cuda()


def _tiny_inputs(B=1, H=64, W=128):
    image = torch.rand(B, H, W, 3, device='cuda') * 255
    sparse_depth = torch.zeros(B, H, W, device='cuda')
    sparse_depth[0, 20, 40] = 5.0
    sparse_depth[0, 40, 90] = 15.0
    gt = torch.zeros(B, H, W, device='cuda')
    gt[0, 10:50, 30:100] = torch.rand(40, 70, device='cuda') * 40 + 2
    sparse_gt = torch.zeros(B, H, W, device='cuda')
    sparse_gt[0, 15, 50] = 8.0
    sparse_gt[0, 45, 85] = 22.0
    return (image, sparse_depth, gt, sparse_gt)


def test_depth_net_forward():
    image, sparse_depth, _, _ = _tiny_inputs()
    model = _tiny_depth_net()
    model.eval()
    with torch.no_grad():
        depth, _ = model.forward_test(image, sparse_depth)
    assert depth.shape == (1, 1, 64, 128)
    assert torch.isfinite(depth).all()


def test_depth_net_honors_max_depth():
    from mambadepth_fusion.model import MambaFusionDepth
    image, sparse_depth, _, _ = _tiny_inputs()
    for max_d in (80.0, 120.0):
        model = MambaFusionDepth(max_depth=max_d, gnn_layers=1, max_radar_points=32, mamba_n_layers=1).cuda()
        model.eval()
        with torch.no_grad():
            depth, _ = model.forward_test(image, sparse_depth)
        non_radar = (sparse_depth == 0).unsqueeze(1)
        pred_max = depth[non_radar].max().item()
        assert pred_max <= max_d + 0.001, f'max_depth={max_d}: pred max {pred_max:.2f} exceeds ceiling at non-radar pixels'
        assert math.isclose(model.dn.log_mx, math.log(max_d))


def test_depth_net_forward_train():
    image, sparse_depth, gt, sparse_gt = _tiny_inputs()
    model = _tiny_depth_net()
    model.train()
    loss, ld = model.forward_train(image, sparse_depth, gt, sparse_gt=sparse_gt)
    assert torch.isfinite(loss), 'loss is not finite'
    loss.backward()
    for key in ('loss', 'loss_log', 'loss_linear', 'loss_grad', 'loss_sparse', 'mae_mm'):
        assert key in ld, f'missing key {key} in loss dict'
