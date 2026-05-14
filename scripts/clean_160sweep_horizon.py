import argparse
import os
import time
import numpy as np
from PIL import Image
SRC_ROOT_DEFAULT = '/path/to/nuScenes_derived/gt_lidar_160sweep'

def load_depth_png(path):
    return np.array(Image.open(path), dtype=np.float32) / 256.0

def save_depth_png(path, depth):
    arr = np.clip(depth * 256.0, 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)

def local_mad_outlier(gt, window=7, min_neighbors=10, k=3.0, abs_floor=10.0, candidate_mask=None):
    H, W = gt.shape
    out = np.zeros_like(gt, dtype=bool)
    pad = window // 2
    if candidate_mask is None:
        candidate_mask = gt > 0
    ys, xs = np.where(candidate_mask)
    if len(ys) == 0:
        return out
    gt_padded = np.pad(gt, pad, mode='constant', constant_values=0)
    yp = ys + pad
    xp = xs + pad
    dy = np.arange(-pad, pad + 1)
    dx = np.arange(-pad, pad + 1)
    DY, DX = np.meshgrid(dy, dx, indexing='ij')
    DY = DY.ravel()
    DX = DX.ravel()
    neigh = gt_padded[yp[:, None] + DY[None, :], xp[:, None] + DX[None, :]]
    valid = neigh > 0
    n_valid = valid.sum(axis=1)
    enough = n_valid >= min_neighbors
    masked = np.where(valid, neigh, np.nan)
    med = np.nanmedian(masked, axis=1)
    mad = np.nanmedian(np.abs(masked - med[:, None]), axis=1)
    d = gt[ys, xs]
    is_low = med - d > k * mad
    is_low &= med - d > abs_floor
    is_low &= enough
    out[ys[is_low], xs[is_low]] = True
    return out

def clean_frame(gt, mono, y_frac=0.45, near_thresh=20.0, mono_disagree=30.0, near_thresh_mono=25.0, window=7, k=3.0, abs_floor=10.0, min_neighbors=10):
    H, W = gt.shape
    y_idx = np.arange(H, dtype=np.float32).reshape(-1, 1)
    drop_a = (gt > 0) & (gt < near_thresh) & (y_idx < y_frac * H)
    drop_b = (gt > 0) & (mono > 0) & (mono - gt > mono_disagree) & (gt < near_thresh_mono)
    drop_c = local_mad_outlier(gt, window=window, min_neighbors=min_neighbors, k=k, abs_floor=abs_floor, candidate_mask=drop_a)
    drop = drop_a & (drop_b | drop_c)
    cleaned = gt.copy()
    cleaned[drop] = 0.0
    return (cleaned, drop, {'A': drop_a, 'B': drop_b, 'C': drop_c})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest_dir', default='/path/to/manifest_dir')
    p.add_argument('--gt_manifest_name', default='nuscenes_train_D_acc.txt')
    p.add_argument('--mono_manifest', default='/path/to/nuScenes_derived/mono_ga_dpt_hybrid/nuscenes_train_mono_ga.txt')
    p.add_argument('--radar_manifest_name', default='nuscenes_train_radar.txt', help='Used to derive per-frame mono paths (mono manifest may be shorter than GT when LS fits failed).')
    p.add_argument('--src_root', default=SRC_ROOT_DEFAULT, help='Source dir prefix in the GT manifest (replaced by --output_root in output paths).')
    p.add_argument('--output_root', default='/path/to/nuScenes_derived/gt_lidar_160sweep_cleaned')
    p.add_argument('--output_manifest_name', default='nuscenes_train_D_acc_160clean.txt', help='Written into manifest_dir/training/nuscenes/ (only on full runs — skipped when --indices is given).')
    p.add_argument('--indices', type=int, nargs='*', default=None, help='Frame indices to process (sample run). Omit to process all frames.')
    p.add_argument('--y_frac', type=float, default=0.45, help='Pass A: drop only above y_frac * H (upper image).')
    p.add_argument('--near_thresh', type=float, default=20.0, help='Pass A: drop only where gt < this (meters).')
    p.add_argument('--mono_disagree', type=float, default=30.0, help='Pass B: drop where (mono - gt) > this (meters).')
    p.add_argument('--abs_floor', type=float, default=10.0, help='Pass C: drop only when (local_med - gt) > this (meters).')
    p.add_argument('--mad_k', type=float, default=3.0, help='Pass C: drop when (local_med - gt) > k * MAD.')
    p.add_argument('--window', type=int, default=7, help='Pass C: square window side for local stats.')
    p.add_argument('--min_neighbors', type=int, default=10, help='Pass C: minimum valid neighbors required.')
    args = p.parse_args()
    gt_manifest_path = os.path.join(args.manifest_dir, 'training/nuscenes', args.gt_manifest_name)
    radar_manifest_path = os.path.join(args.manifest_dir, 'training/nuscenes', args.radar_manifest_name)
    out_manifest_path = os.path.join(args.manifest_dir, 'training/nuscenes', args.output_manifest_name)
    with open(gt_manifest_path) as f:
        gt_paths = [l.strip() for l in f if l.strip()]
    with open(args.mono_manifest) as f:
        mono_paths = [l.strip() for l in f if l.strip()]
    with open(radar_manifest_path) as f:
        radar_paths = [l.strip() for l in f if l.strip()]
    mono_root = os.path.dirname(args.mono_manifest)
    mono_set = set(mono_paths)

    def _mono_for_radar(radar_path):
        rel = os.path.relpath(radar_path, start='/')
        p = os.path.join(mono_root, rel)
        if not p.endswith('.png'):
            p = os.path.splitext(p)[0] + '.png'
        return p if p in mono_set else None
    aligned_mono = [_mono_for_radar(r) for r in radar_paths]
    n_with_mono = sum((p is not None for p in aligned_mono))
    print(f'Mono coverage: {n_with_mono}/{len(gt_paths)} frames ({len(gt_paths) - n_with_mono} fall back to Pass-C-only)')
    assert len(gt_paths) == len(radar_paths) == len(aligned_mono)
    out_paths = [p.replace(args.src_root, args.output_root) for p in gt_paths]
    sample_changed = sum((1 for a, b in zip(gt_paths, out_paths) if a != b))
    if sample_changed != len(gt_paths):
        raise RuntimeError(f"src_root '{args.src_root}' did not appear in {len(gt_paths) - sample_changed} of {len(gt_paths)} gt paths. First gt path: {gt_paths[0]}")
    indices = args.indices if args.indices is not None else list(range(len(gt_paths)))
    full_run = args.indices is None
    print(f"Processing {len(indices)} frames ({('full run' if full_run else 'sample run')})")
    print(f'Thresholds: y_frac={args.y_frac}  near_thresh={args.near_thresh}m  mono_disagree={args.mono_disagree}m  abs_floor={args.abs_floor}m  MAD k={args.mad_k}  window={args.window}')
    n_dropped_total = 0
    n_pixels_total = 0
    n_dropped_a = n_dropped_b = n_dropped_c = 0
    t_start = time.time()
    for k_idx, i in enumerate(indices):
        gt = load_depth_png(gt_paths[i])
        if aligned_mono[i] is not None:
            mono = load_depth_png(aligned_mono[i])
        else:
            mono = np.zeros_like(gt)
        cleaned, drop, subs = clean_frame(gt, mono, y_frac=args.y_frac, near_thresh=args.near_thresh, mono_disagree=args.mono_disagree, window=args.window, k=args.mad_k, abs_floor=args.abs_floor, min_neighbors=args.min_neighbors)
        save_depth_png(out_paths[i], cleaned)
        n_orig = int((gt > 0).sum())
        n_drop = int(drop.sum())
        n_pixels_total += n_orig
        n_dropped_total += n_drop
        n_dropped_a += int(subs['A'].sum())
        n_dropped_b += int((subs['A'] & subs['B']).sum())
        n_dropped_c += int((subs['A'] & subs['C']).sum())
        cov_orig = (gt > 0).mean() * 100
        cov_clean = (cleaned > 0).mean() * 100
        if full_run and k_idx % 200 == 0:
            t = time.time() - t_start
            rate = (k_idx + 1) / max(t, 1e-06)
            eta = (len(indices) - k_idx) / max(rate, 1e-06) / 60
            print(f'[{k_idx:5d}/{len(indices)}] cov {cov_orig:5.2f}% → {cov_clean:5.2f}%  dropped {n_drop:6d}  ({rate:.1f} fps, ETA {eta:.1f} min)')
        if not full_run:
            drop_pct = n_drop / max(n_orig, 1) * 100
            print(f"[{i:5d}] cov: {cov_orig:5.2f}% → {cov_clean:5.2f}%  dropped: {n_drop:5d} ({drop_pct:5.2f}% of valid)  A={int(subs['A'].sum()):5d}  B={int((subs['A'] & subs['B']).sum()):5d}  C={int((subs['A'] & subs['C']).sum()):5d}")
    if full_run:
        with open(out_manifest_path, 'w') as f:
            f.write('\n'.join(out_paths) + '\n')
        print(f'\nWrote manifest: {out_manifest_path}')
    if n_pixels_total > 0:
        print(f'\nSummary across {len(indices)} frames:')
        print(f'  Total valid pixels:     {n_pixels_total}')
        print(f'  Total dropped pixels:   {n_dropped_total} ({n_dropped_total / n_pixels_total * 100:.3f}%)')
        print(f'  A-region candidates:    {n_dropped_a} (pre-gate)')
        print(f'  A∧B (mono-disagree):    {n_dropped_b}')
        print(f'  A∧C (local-MAD):        {n_dropped_c}')
        print(f'  Elapsed:                {time.time() - t_start:.1f} s')
if __name__ == '__main__':
    main()
