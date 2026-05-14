import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.hub as _hub
_hub._check_repo_is_trusted = lambda *_a, **_kw: None

def load_depth_png(path):
    return np.array(Image.open(path), dtype=np.float32) / 256.0

def save_depth_png(depth_m, path):
    clipped = np.clip(depth_m, 0.0, 255.0)
    arr = np.asarray(clipped * 256.0, dtype=np.uint32).astype(np.int32)
    Image.fromarray(arr.astype(np.int32), mode='I').save(path)

def load_radar_depth(path, H, W):
    if path.endswith('.png'):
        depth = load_depth_png(path)
    elif path.endswith('.npy'):
        pts = np.load(path).astype(np.float32)
        depth = np.zeros((H, W), dtype=np.float32)
        if pts.shape[0] > 0:
            for p in pts:
                y, x, d = (int(round(p[0])), int(round(p[1])), float(p[2]))
                if 0 <= y < H and 0 <= x < W and (d > 0):
                    if depth[y, x] == 0 or d < depth[y, x]:
                        depth[y, x] = d
    else:
        raise ValueError(f'Unrecognised radar format: {path}')
    if depth.shape != (H, W):
        depth_t = torch.from_numpy(depth)[None, None]
        depth = F.interpolate(depth_t, size=(H, W), mode='nearest').squeeze().numpy()
    return depth

def fit_scale_shift(prediction, target, mask):
    m = mask.astype(np.float32)
    a00 = np.sum(m * prediction * prediction)
    a01 = np.sum(m * prediction)
    a11 = np.sum(m)
    b0 = np.sum(m * prediction * target)
    b1 = np.sum(m * target)
    det = a00 * a11 - a01 * a01
    if det <= 0 or a11 < 2:
        return (0.0, 0.0)
    scale = (a11 * b0 - a01 * b1) / det
    shift = (-a01 * b0 + a00 * b1) / det
    return (float(scale), float(shift))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_manifest', type=str, required=True)
    p.add_argument('--radar_manifest', type=str, required=True)
    p.add_argument('--output_root', type=str, required=True, help='Directory to save aligned mono-depth PNGs.')
    p.add_argument('--split_tag', type=str, required=True, help="Used to name the output manifest, e.g. 'val' or 'train'.")
    p.add_argument('--dataset_tag', type=str, default='nuscenes', help='Prefix in the output manifest filename.')
    p.add_argument('--dpt_variant', type=str, default='DPT_Hybrid', choices=['DPT_Hybrid', 'DPT_Large'])
    p.add_argument('--dpt_net_size', type=int, default=384, help='DPT input size on the long side.')
    p.add_argument('--min_depth', type=float, default=2.0)
    p.add_argument('--max_depth', type=float, default=80.0)
    p.add_argument('--native_h', type=int, default=900, help='Target output resolution height (matches dataset).')
    p.add_argument('--native_w', type=int, default=1600)
    p.add_argument('--limit', type=int, default=-1, help='Only process first N frames (debug).')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_root, exist_ok=True)
    print(f'[DPT] loading {args.dpt_variant} ...')
    dpt = torch.hub.load('intel-isl/MiDaS', args.dpt_variant, trust_repo=True)
    dpt.eval().to(device)
    with open(args.image_manifest) as f:
        image_paths = [ln.strip() for ln in f if ln.strip()]
    with open(args.radar_manifest) as f:
        radar_paths = [ln.strip() for ln in f if ln.strip()]
    assert len(image_paths) == len(radar_paths), f'manifest length mismatch: {len(image_paths)} images vs {len(radar_paths)} radar files'
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
        radar_paths = radar_paths[:args.limit]
    print(f'[data] {len(image_paths)} frames to process')
    net = args.dpt_net_size
    H, W = (args.native_h, args.native_w)
    aspect = W / H
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    manifest_path = os.path.join(args.output_root, f'{args.dataset_tag}_{args.split_tag}_mono_ga.txt')
    out_f = open(manifest_path, 'w')
    t_start = time.time()
    n_ok = 0
    n_skip = 0
    with torch.no_grad():
        for i, (img_p, rad_p) in enumerate(zip(image_paths, radar_paths)):
            try:
                pil = Image.open(img_p).convert('RGB')
                img = np.asarray(pil, dtype=np.uint8)
                if img.shape[:2] != (H, W):
                    pil = pil.resize((W, H), Image.BILINEAR)
                    img = np.asarray(pil, dtype=np.uint8)
                t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                t_img = t_img.unsqueeze(0).to(device)
                if aspect >= 1.0:
                    new_h = net
                    new_w = int(round(net * aspect / 32.0)) * 32
                else:
                    new_w = net
                    new_h = int(round(net / aspect / 32.0)) * 32
                t_in = F.interpolate(t_img, size=(new_h, new_w), mode='bilinear', align_corners=True)
                t_in = (t_in - mean) / std
                disp = dpt(t_in)
                if disp.dim() == 3:
                    disp = disp.unsqueeze(1)
                disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=True)
                disp_np = disp.squeeze().cpu().numpy().astype(np.float32)
                radar_depth = load_radar_depth(rad_p, H, W)
                valid = (radar_depth >= args.min_depth) & (radar_depth <= args.max_depth)
                target_inv = np.zeros_like(radar_depth)
                target_inv[valid] = 1.0 / radar_depth[valid]
                s, t = fit_scale_shift(disp_np, target_inv, valid)
                if s == 0.0:
                    n_skip += 1
                    continue
                aligned_inv = disp_np * s + t
                min_inv = 1.0 / args.max_depth
                max_inv = 1.0 / args.min_depth
                aligned_inv = np.clip(aligned_inv, min_inv, max_inv)
                aligned_depth = 1.0 / aligned_inv
                rel = os.path.relpath(rad_p, start='/')
                out_path = os.path.join(args.output_root, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if not out_path.endswith('.png'):
                    out_path = os.path.splitext(out_path)[0] + '.png'
                save_depth_png(aligned_depth, out_path)
                out_f.write(out_path + '\n')
                n_ok += 1
                if i % 200 == 0:
                    dt = time.time() - t_start
                    rate = (i + 1) / max(dt, 1e-06)
                    eta = (len(image_paths) - i - 1) / max(rate, 1e-06)
                    print(f'[{i + 1:5d}/{len(image_paths)}] ok={n_ok} skip={n_skip}  rate={rate:.2f} fr/s  eta={eta / 60:.1f} min')
            except Exception as e:
                print(f'[warn] frame {i} ({img_p}) failed: {e}')
                n_skip += 1
    out_f.close()
    dt = time.time() - t_start
    print()
    print(f'=== Done ===')
    print(f'  processed : {len(image_paths)}')
    print(f'  saved     : {n_ok}')
    print(f'  skipped   : {n_skip}')
    print(f'  wall time : {dt / 60:.1f} min')
    print(f'  manifest  : {manifest_path}')
if __name__ == '__main__':
    main()
