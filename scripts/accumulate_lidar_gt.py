from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box, view_points
IMG_H, IMG_W = (900, 1600)
MAX_DEPTH_CLIP_M = 255.996
_MOVABLE_CATEGORY_PREFIXES = ('vehicle.', 'human.', 'animal')
_NUSC: NuScenes | None = None
_SCENE_ID_BY_TOKEN: dict[str, int] = {}

def _init_worker(version: str, dataroot: str) -> None:
    global _NUSC, _SCENE_ID_BY_TOKEN
    _NUSC = NuScenes(version=version, dataroot=dataroot, verbose=False)
    _SCENE_ID_BY_TOKEN = {s['token']: i for i, s in enumerate(_NUSC.scene)}

def _walk_lidar_sweeps(start_sd: dict, n_back: int, n_fwd: int) -> list[str]:
    tokens = [start_sd['token']]
    prev = start_sd
    for _ in range(n_back):
        if prev['prev'] == '':
            break
        prev = _NUSC.get('sample_data', prev['prev'])
        tokens.append(prev['token'])
    nxt = start_sd
    for _ in range(n_fwd):
        if nxt['next'] == '':
            break
        nxt = _NUSC.get('sample_data', nxt['next'])
        tokens.append(nxt['token'])
    return tokens

def _is_movable_category(name: str) -> bool:
    return any((name.startswith(p) for p in _MOVABLE_CATEGORY_PREFIXES))

@lru_cache(maxsize=None)
def _box_velocity_xy(token: str) -> float:
    try:
        v = _NUSC.box_velocity(token)
    except Exception:
        return float('inf')
    if v is None:
        return float('inf')
    v_xy = float(np.linalg.norm(v[:2]))
    return v_xy if np.isfinite(v_xy) else float('inf')

def _is_moving_box(box, velocity_threshold: float) -> bool:
    return _box_velocity_xy(box.token) > velocity_threshold

def _drop_points_in_moving_boxes(pcl: LidarPointCloud, sd: dict, sensor_calib: dict, sensor_ego: dict, velocity_threshold: float) -> None:
    boxes_global = _NUSC.get_boxes(sd['token'])
    if not boxes_global:
        return
    inv_R_ego = Quaternion(sensor_ego['rotation']).inverse
    inv_R_sensor = Quaternion(sensor_calib['rotation']).inverse
    t_ego = np.array(sensor_ego['translation'])
    t_sensor = np.array(sensor_calib['translation'])
    moving_boxes_sensor = []
    for box in boxes_global:
        if not _is_movable_category(box.name):
            continue
        if not _is_moving_box(box, velocity_threshold):
            continue
        box.translate(-t_ego)
        box.rotate(inv_R_ego)
        box.translate(-t_sensor)
        box.rotate(inv_R_sensor)
        moving_boxes_sensor.append(box)
    if not moving_boxes_sensor:
        return
    pts3 = pcl.points[:3, :]
    keep = np.ones(pts3.shape[1], dtype=bool)
    for box in moving_boxes_sensor:
        keep &= ~points_in_box(box, pts3)
    pcl.points = pcl.points[:, keep]

def _project_sweep_to_target_cam(sd: dict, dataroot: str, tgt_cam_calib: dict, tgt_cam_ego: dict, cam_intr: np.ndarray, filter_moving: bool=False, velocity_threshold: float=0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_calib = _NUSC.get('calibrated_sensor', sd['calibrated_sensor_token'])
    sensor_ego = _NUSC.get('ego_pose', sd['ego_pose_token'])
    pcl = LidarPointCloud.from_file(os.path.join(dataroot, sd['filename']))
    if filter_moving:
        _drop_points_in_moving_boxes(pcl, sd, sensor_calib, sensor_ego, velocity_threshold)
        if pcl.points.shape[1] == 0:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    pcl.rotate(Quaternion(sensor_calib['rotation']).rotation_matrix)
    pcl.translate(np.array(sensor_calib['translation']))
    pcl.rotate(Quaternion(sensor_ego['rotation']).rotation_matrix)
    pcl.translate(np.array(sensor_ego['translation']))
    pcl.translate(-np.array(tgt_cam_ego['translation']))
    pcl.rotate(Quaternion(tgt_cam_ego['rotation']).rotation_matrix.T)
    pcl.translate(-np.array(tgt_cam_calib['translation']))
    pcl.rotate(Quaternion(tgt_cam_calib['rotation']).rotation_matrix.T)
    depth_vals = pcl.points[2, :].astype(np.float32)
    in_front = (depth_vals > 1.0) & (depth_vals < MAX_DEPTH_CLIP_M)
    if not np.any(in_front):
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    pts_kept = pcl.points[:3, in_front]
    d_kept = depth_vals[in_front]
    pts_2d = view_points(pts_kept, cam_intr, normalize=True)
    u = pts_2d[0, :]
    v = pts_2d[1, :]
    in_canvas = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u = u[in_canvas].astype(np.int32)
    v = v[in_canvas].astype(np.int32)
    d = d_kept[in_canvas]
    return (u, v, d)

def _accumulate_one_keyframe(sample_token_with_ctx: tuple) -> str:
    sample_token, n_back, n_fwd, dataroot, out_root, output_subdir, filter_moving, velocity_threshold = sample_token_with_ctx
    sample = _NUSC.get('sample', sample_token)
    scene_id = _SCENE_ID_BY_TOKEN[sample['scene_token']]
    cam_sd = _NUSC.get('sample_data', sample['data']['CAM_FRONT'])
    cam_calib = _NUSC.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ego = _NUSC.get('ego_pose', cam_sd['ego_pose_token'])
    cam_intr = np.array(cam_calib['camera_intrinsic'])
    lidar_sd = _NUSC.get('sample_data', sample['data']['LIDAR_TOP'])
    sweep_tokens = _walk_lidar_sweeps(lidar_sd, n_back, n_fwd)
    depth = np.full((IMG_H, IMG_W), np.inf, dtype=np.float32)
    for tok in sweep_tokens:
        sd = _NUSC.get('sample_data', tok)
        u, v, d = _project_sweep_to_target_cam(sd, dataroot, cam_calib, cam_ego, cam_intr, filter_moving=filter_moving, velocity_threshold=velocity_threshold)
        if u.size == 0:
            continue
        np.minimum.at(depth, (v, u), d)
    depth[np.isinf(depth)] = 0.0
    encoded = np.clip(depth * 256.0, 0, 65535).astype(np.uint16)
    cam_filename = os.path.basename(cam_sd['filename']).replace('.jpg', '.png')
    scene_dir = os.path.join(out_root, output_subdir, f'scene_{scene_id}', 'CAM_FRONT')
    os.makedirs(scene_dir, exist_ok=True)
    out_path = os.path.join(scene_dir, cam_filename)
    Image.fromarray(encoded).save(out_path)
    return out_path

def main() -> None:
    p = argparse.ArgumentParser(description='Accumulate LIDAR_TOP sweeps into CAM_FRONT depth maps.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataroot', type=str, required=True, help='nuScenes raw data root (contains samples/, sweeps/, v1.0-*/)')
    p.add_argument('--derived_dir', type=str, required=True, help='Where to write <output_subdir>/scene_*/CAM_FRONT/*.png')
    p.add_argument('--output_subdir', type=str, required=True, help='Subfolder name, e.g. gt_lidar_1sweep or gt_lidar_160sweep')
    p.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-test', 'v1.0-mini'])
    p.add_argument('--n_sweeps', type=int, required=True, help='Total LIDAR_TOP sweeps to accumulate per keyframe. 1 = single sweep (D_gt). 160 = 8-sec window (D_acc).')
    p.add_argument('--n_thread', type=int, default=40, help='Worker processes (each loads ~5 GB nuScenes metadata)')
    p.add_argument('--limit', type=int, default=None, help='(testing) process only the first N keyframes')
    p.add_argument('--filter_moving', action='store_true', help='Drop LiDAR returns falling inside moving 3D-annotated boxes (vehicles/humans/animals with velocity above --velocity_threshold). Eliminates ghost trails in accumulated GT. Recommended whenever n_sweeps > ~10.')
    p.add_argument('--velocity_threshold', type=float, default=0.5, help='Box-velocity threshold in m/s for the moving filter. Boxes at or below this are treated as static and their returns are kept (parked-car geometry).')
    args = p.parse_args()
    if args.n_sweeps < 1:
        raise ValueError('--n_sweeps must be >= 1')
    n_side = (args.n_sweeps - 1) // 2
    n_back = n_side
    n_fwd = args.n_sweeps - 1 - n_side
    print(f'Loading {args.version} metadata from {args.dataroot} ...')
    t0 = time.time()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f'  loaded in {time.time() - t0:.1f}s. {len(nusc.scene)} scenes, {len(nusc.sample)} keyframes.')
    sample_tokens = [s['token'] for s in nusc.sample]
    if args.limit is not None:
        sample_tokens = sample_tokens[:args.limit]
        print(f'  LIMITED to first {args.limit} keyframes for testing')
    print(f'Accumulating {args.n_sweeps} sweeps per keyframe ({n_back} back + 1 current + {n_fwd} forward)')
    if args.filter_moving:
        print(f'Moving filter: ON (drop returns inside boxes with |v_xy| > {args.velocity_threshold:.2f} m/s)')
    else:
        print('Moving filter: OFF (raw min-depth-wins; ghost trails preserved)')
    print(f'Output: {args.derived_dir}/{args.output_subdir}/scene_*/CAM_FRONT/')
    tasks = [(tok, n_back, n_fwd, args.dataroot, args.derived_dir, args.output_subdir, args.filter_moving, args.velocity_threshold) for tok in sample_tokens]
    os.makedirs(os.path.join(args.derived_dir, args.output_subdir), exist_ok=True)
    t0 = time.time()
    done = 0
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.n_thread, initializer=_init_worker, initargs=(args.version, args.dataroot)) as pool:
        for _ in tqdm(pool.map(_accumulate_one_keyframe, tasks), total=len(tasks), desc='accumulating'):
            done += 1
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-06)
    print(f'\nDone. {done} keyframes in {elapsed / 60:.1f} min ({rate:.1f} kf/s)')
    print(f'Output: {args.derived_dir}/{args.output_subdir}/')
if __name__ == '__main__':
    main()
