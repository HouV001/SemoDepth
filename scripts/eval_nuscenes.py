from __future__ import annotations
import argparse
import json
import os
import sys
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mambadepth_fusion.data import data_utils
from mambadepth_fusion.data.dataset_nuscenes import NuScenesDataset
from mambadepth_fusion.model.depth_net import MambaFusionDepth
from mambadepth_fusion.train.trainer import evaluate


def get_scene_subset_indices(val_image_paths, dataroot, subset, cache_path):
    if subset == 'all':
        return list(range(len(val_image_paths)))
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        return cache[subset]
    from nuscenes.nuscenes import NuScenes
    print(f'[scene-split] building cache (one-time, ~30s)...')
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    fname_to_desc = {}
    for scene in nusc.scene:
        desc = scene['description']
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            fname_to_desc[os.path.basename(sd['filename'])] = desc
            sample_token = sample['next']
    day_idx, night_idx = ([], [])
    for i, p in enumerate(val_image_paths):
        desc = fname_to_desc.get(os.path.basename(p), '')
        (night_idx if 'night' in desc.lower() else day_idx).append(i)
    cache = {'day': day_idx, 'night': night_idx}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f'[scene-split] day={len(day_idx)} night={len(night_idx)} → {cache_path}')
    return cache[subset]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to model-best.pth')
    p.add_argument('--manifest_dir', default='/path/to/manifest_dir')
    p.add_argument('--max_depth', type=float, default=120.0)
    p.add_argument('--min_depth', type=float, default=0.5)
    p.add_argument('--no_default_to_max', action='store_true', help="Disable CaFNet/Singh-style eval-time clipping (pred → [min_eval, max_d]; NaN → min, inf → max).")
    p.add_argument('--frame_subset', choices=['all', 'day', 'night'], default='all', help="Filter val frames by scene description (TacoDepth Table 7 protocol).")
    p.add_argument('--dataroot', default='/path/to/nuScenes', help='nuScenes raw dataroot (only needed for --frame_subset day|night).')
    p.add_argument('--scene_cache', default='/path/to/nuScenes_derived/val_scene_subset.json', help='Cache path for the day/night index lists.')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_image_file = os.path.join(args.manifest_dir, 'validation/nuscenes/nuscenes_val_image.txt')
    val_radar_file = os.path.join(args.manifest_dir, 'validation/nuscenes/nuscenes_val_radar.txt')
    val_gt_file = os.path.join(args.manifest_dir, 'validation/nuscenes/nuscenes_val_D_gt.txt')
    for f in (val_image_file, val_radar_file, val_gt_file):
        assert os.path.exists(f), f'missing manifest: {f}'
    val_image_paths = data_utils.read_paths(val_image_file)
    val_radar_paths = data_utils.read_paths(val_radar_file)
    val_gt_paths = data_utils.read_paths(val_gt_file)
    print(f'[manifest] {len(val_image_paths)} val samples')
    if args.frame_subset != 'all':
        keep = get_scene_subset_indices(val_image_paths, args.dataroot, args.frame_subset, args.scene_cache)
        val_image_paths = [val_image_paths[i] for i in keep]
        val_radar_paths = [val_radar_paths[i] for i in keep]
        val_gt_paths = [val_gt_paths[i] for i in keep]
        print(f'[subset={args.frame_subset}] {len(val_image_paths)} frames retained')
    model = MambaFusionDepth(max_depth=args.max_depth, min_depth=args.min_depth).to(device).eval()
    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[ckpt] {args.checkpoint}')
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}')
    results = evaluate(val_image_paths, val_radar_paths, val_gt_paths, model=model, device=device, dataset_class=NuScenesDataset, max_depth=args.max_depth, default_to_max=not args.no_default_to_max)
    print()
    print('=' * 60)
    print(f' Re-eval: max_depth={args.max_depth} default_to_max={not args.no_default_to_max}')
    print('=' * 60)
    for max_d in (50, 70, 80):
        r = results[max_d]
        print(f" {max_d}m: MAE={r['mae']:7.1f} RMSE={r['rmse']:7.1f} iMAE={r['imae']:5.2f} iRMSE={r['irmse']:5.2f} d1={r['delta1']:.4f}")
    print('=' * 60)


if __name__ == '__main__':
    main()
