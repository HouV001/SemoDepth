from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'third_party/Sparse-Beats-Dense'))
from mambadepth_fusion.data import data_utils
from mambadepth_fusion.model.depth_net import MambaFusionDepth
from utils import colorize_depth_map
DEFAULT_DATA_ROOT = '/path/to/ZJU-4DRadarCam/data'
DEFAULT_LOG_ROOT = '/path/to/ZJU-4DRadarCam/log/mambadepth'
DEFAULT_CKPT = os.path.join(DEFAULT_LOG_ROOT, 'model-best.pth')
ZJU_CROP_Y0, ZJU_CROP_Y1 = (240, 540)
DEFAULT_FRAMES = [50, 600, 1700, 2800]

@dataclass
class ModelSpec:
    name: str
    checkpoint: str

def build_and_load(spec: ModelSpec, device, max_depth, min_depth):
    model = MambaFusionDepth(max_depth=max_depth, min_depth=min_depth).to(device).eval()
    state = torch.load(spec.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[ckpt] {spec.name}: {os.path.basename(spec.checkpoint)} (missing={len(missing)} unexpected={len(unexpected)})')
    return model

@torch.no_grad()
def predict(model, image_u8, radar_sparse, device):
    image_t = torch.from_numpy(image_u8).unsqueeze(0).to(device).float()
    radar_t = torch.from_numpy(radar_sparse).unsqueeze(0).to(device).float()
    depth, _ = model.forward_test(image_t, radar_t)
    return depth.squeeze().cpu().numpy()

def load_depth_png(path):
    return np.array(Image.open(path), dtype=np.float32) / 256.0

def load_zju_frame(image_path, radar_path, gt_path):
    img = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    radar = load_depth_png(radar_path)
    gt = load_depth_png(gt_path)
    if img.shape[0] == 720:
        img = img[ZJU_CROP_Y0:ZJU_CROP_Y1, :, :]
        radar = radar[ZJU_CROP_Y0:ZJU_CROP_Y1, :]
        gt = gt[ZJU_CROP_Y0:ZJU_CROP_Y1, :]
    return (img, radar, gt)

def render_depth(d, vmax, mask=None):
    return colorize_depth_map(d / vmax, mask=mask)

def overlay_radar(rgb_u8, radar, vmax, point_size=4):
    out = rgb_u8.astype(np.float32) / 255.0
    cmap = plt.get_cmap('jet')
    ys, xs = np.where(radar > 0)
    for y, x in zip(ys, xs):
        d = float(radar[y, x])
        c = np.array(cmap(min(d, vmax) / vmax)[:3])
        y0, y1 = (max(0, y - point_size), min(out.shape[0], y + point_size + 1))
        x0, x1 = (max(0, x - point_size), min(out.shape[1], x + point_size + 1))
        out[y0:y1, x0:x1] = c
    return np.clip(out, 0, 1)

def per_frame_metrics(pred, gt, eval_cap_m=80.0, min_eval_m=0.5):
    valid = (gt > 0) & (gt <= eval_cap_m)
    if not valid.any():
        return (float('nan'), float('nan'))
    p = np.where(np.isnan(pred), min_eval_m, pred)
    p = np.where(np.isinf(p), eval_cap_m, p)
    p = np.clip(p, min_eval_m, eval_cap_m)
    o, g = (p[valid], gt[valid])
    return (float(np.mean(np.abs(o - g))) * 1000.0, float(np.sqrt(np.mean((o - g) ** 2))) * 1000.0)

def render_grid(frames, column_titles, output_path, vmax=80.0, scene_labels=None, cell_aspect=300.0 / 1280.0):
    n_rows = len(frames)
    n_cols = len(column_titles)
    left, right, top, bottom = (0.005, 0.995, 0.92, 0.005)
    wspace_target = 0.01
    hspace_target = wspace_target / cell_aspect
    cell_w_in = 2.4
    fig_w = cell_w_in * n_cols
    usable_w = fig_w * (right - left)
    cell_w_est = usable_w / (n_cols + (n_cols - 1) * wspace_target)
    cell_h_est = cell_w_est * cell_aspect
    usable_h = cell_h_est * (n_rows + (n_rows - 1) * hspace_target)
    fig_h = usable_h / (top - bottom)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[None, :]
    for r, frame in enumerate(frames):
        col = 0
        axes[r, col].imshow(frame['rgb'])
        axes[r, col].axis('off')
        if scene_labels and r < len(scene_labels) and scene_labels[r]:
            axes[r, col].text(0.02, 0.92, scene_labels[r], transform=axes[r, col].transAxes, fontsize=9, color='white', weight='bold', va='top', ha='left', bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none'))
        if r == 0:
            axes[r, col].set_title(column_titles[col], fontsize=9)
        col += 1
        axes[r, col].imshow(overlay_radar(frame['rgb'], frame['radar'], vmax))
        axes[r, col].axis('off')
        if r == 0:
            axes[r, col].set_title(column_titles[col], fontsize=9)
        col += 1
        for pred in frame['preds']:
            axes[r, col].imshow(render_depth(pred['depth'], vmax))
            axes[r, col].axis('off')
            if pred.get('mae_mm') is not None:
                axes[r, col].text(0.02, 0.92, f"MAE {pred['mae_mm']:.0f}\nRMSE {pred['rmse_mm']:.0f}", transform=axes[r, col].transAxes, fontsize=7, color='white', linespacing=1.05, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, edgecolor='none'))
            if r == 0:
                axes[r, col].set_title(pred['name'], fontsize=9)
            col += 1
        if col < len(column_titles) and column_titles[col] == 'Ground truth':
            gt_mask = frame['gt'] > 0
            axes[r, col].imshow(render_depth(frame['gt'], vmax, mask=gt_mask), interpolation='nearest', resample=False)
            axes[r, col].axis('off')
            if r == 0:
                axes[r, col].set_title(column_titles[col], fontsize=9)
    plt.subplots_adjust(wspace=wspace_target, hspace=hspace_target, left=left, right=right, top=top, bottom=bottom)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[render] wrote {output_path}')

def auto_select_frames(model, image_paths, radar_paths, gt_paths, device, args):
    n = len(image_paths)
    pool = list(range(n))
    if args.search_pool > 0 and args.search_pool < n:
        step = max(1, n // args.search_pool)
        pool = list(range(0, n, step))[:args.search_pool]
    print(f'[auto-select] scoring {len(pool)} of {n} test frames')
    scored = []
    for k, idx in enumerate(pool):
        img, radar, gt = load_zju_frame(image_paths[idx], radar_paths[idx], gt_paths[idx])
        if (gt > 0).sum() < 100:
            continue
        pred = predict(model, img, radar, device)
        mae, _ = per_frame_metrics(pred, gt, eval_cap_m=args.vmax, min_eval_m=args.min_depth)
        if not np.isfinite(mae):
            continue
        scored.append((mae, idx))
        if (k + 1) % 200 == 0:
            print(f'  ...scored {k + 1}/{len(pool)}')
    scored.sort(key=lambda x: x[0])
    picked = []
    for mae, idx in scored:
        if all((abs(idx - p_idx) >= args.min_frame_gap for _, p_idx in picked)):
            picked.append((mae, idx))
        if len(picked) >= args.auto_select:
            break
    print(f'\n[auto-select] top {len(picked)} frames by Ours MAE@{args.vmax}, min gap={args.min_frame_gap}:')
    for mae, idx in picked:
        print(f'  frame {idx:5d}  MAE={mae:.0f} mm')
    return [idx for _, idx in picked]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--checkpoint', default=DEFAULT_CKPT)
    p.add_argument('--frames', type=int, nargs='+', default=DEFAULT_FRAMES)
    p.add_argument('--auto_select', type=int, default=0)
    p.add_argument('--search_pool', type=int, default=0)
    p.add_argument('--min_frame_gap', type=int, default=200)
    p.add_argument('--random', type=int, default=0, help='Pick this many test frames at random (seeded); overrides --frames and --auto_select.')
    p.add_argument('--seed', type=int, default=0, help='RNG seed used by --random.')
    p.add_argument('--output', default=None)
    p.add_argument('--max_depth', type=float, default=80.0)
    p.add_argument('--min_depth', type=float, default=0.5)
    p.add_argument('--vmax', type=float, default=80.0)
    p.add_argument('--scene_labels', nargs='+', default=None)
    p.add_argument('--no_gt', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(REPO, 'docs/fig/qualitative_zju.pdf')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f'[output] {args.output}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_list = os.path.join(args.data_root, 'test.txt')
    image_paths = data_utils.load_data_path(args.data_root + '/image/', test_list, '.png')
    radar_paths = data_utils.load_data_path(args.data_root + '/radar_png/', test_list, '.png')
    gt_paths = data_utils.load_data_path(args.data_root + '/gt/', test_list, '.png')
    print(f'[manifest] {len(image_paths)} test frames')
    spec = ModelSpec(name='Ours (SemoDepth)', checkpoint=args.checkpoint)
    model = build_and_load(spec, device, args.max_depth, args.min_depth)
    if args.random > 0:
        rng = np.random.default_rng(args.seed)
        frame_indices = sorted((int(i) for i in rng.choice(len(image_paths), size=args.random, replace=False)))
        print(f'[random] seed={args.seed} picked {len(frame_indices)} frames: {frame_indices}')
    elif args.auto_select > 0:
        frame_indices = auto_select_frames(model, image_paths, radar_paths, gt_paths, device, args)
    else:
        frame_indices = args.frames
    column_titles = ['RGB', 'Radar overlay', spec.name]
    if not args.no_gt:
        column_titles.append('Ground truth')
    out_frames = []
    for idx in frame_indices:
        if idx >= len(image_paths):
            print(f'[skip] frame {idx} out of range')
            continue
        img, radar, gt = load_zju_frame(image_paths[idx], radar_paths[idx], gt_paths[idx])
        pred = predict(model, img, radar, device)
        mae_mm, rmse_mm = per_frame_metrics(pred, gt, eval_cap_m=args.vmax, min_eval_m=args.min_depth)
        print(f'[frame {idx}] MAE={mae_mm:.0f}, RMSE={rmse_mm:.0f}, radar pts={int((radar > 0).sum())}, gt pts={int((gt > 0).sum())}')
        out_frames.append({'rgb': img, 'radar': radar, 'gt': gt, 'preds': [{'name': spec.name, 'depth': pred, 'mae_mm': mae_mm, 'rmse_mm': rmse_mm}]})
    render_grid(out_frames, column_titles, args.output, vmax=args.vmax, scene_labels=args.scene_labels)
if __name__ == '__main__':
    main()
