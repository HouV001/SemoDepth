import argparse
import datetime
import os
from mambadepth_fusion.data import NuScenesDataset, data_utils
from mambadepth_fusion.train import train


def _mono_path_from_radar(radar_path, mono_root):
    rel = os.path.relpath(radar_path, start='/')
    p = os.path.join(mono_root, rel)
    if not p.endswith('.png'):
        p = os.path.splitext(p)[0] + '.png'
    return p


def _align_to_mono(radar_paths, mono_paths, mono_root, tag):
    mono_set = set(mono_paths)
    derived = [_mono_path_from_radar(r, mono_root) for r in radar_paths]
    mask = [d in mono_set for d in derived]
    n_keep = sum(mask)
    if n_keep != len(mono_paths):
        raise RuntimeError(f'[{tag}] mono manifest has {len(mono_paths)} entries but only {n_keep} align with the radar manifest; mono_root={mono_root}')
    aligned = [d for d, k in zip(derived, mask) if k]
    return (mask, aligned)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest_dir', type=str, default=os.environ.get('NUSCENES_MANIFEST_DIR', '/path/to/manifest_dir'))
    p.add_argument('--train_gt_acc_file', type=str, default=None, help='Override the D_acc (accumulated LiDAR) train manifest. Defaults to <manifest_dir>/training/nuscenes/nuscenes_train_D_acc_160clean.txt (160-sweep with the horizon-cleaning pass of scripts/clean_160sweep_horizon.py applied).')
    p.add_argument('--log_root', type=str, default=os.environ.get('NUSCENES_LOG_ROOT', '/path/to/nuScenes_derived/log/mambadepth'))
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=50, help="Matches TacoDepth's 50-epoch schedule.")
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--lambda_grad', type=float, default=0.5, help='Gradient-matching weight on main target (D_acc). Default 0.5 on the cleaned 160-sweep manifest.')
    p.add_argument('--lambda_linear', type=float, default=1.0, help='Huber weight on main target (D_acc).')
    p.add_argument('--lambda_log', type=float, default=1.0, help='Log-space L1 weight on sparse anchor (D_gt).')
    p.add_argument('--lambda_sparse', type=float, default=1.0, help="Linear-meter L1 weight on sparse anchor (D_gt).")
    p.add_argument('--max_depth', type=float, default=120.0, help='Model-internal depth ceiling (meters). Decoupled from eval, which is fixed at {50, 70, 80} m. Default 120 m on nuScenes to keep the ~15%% of radar points at 80-120 m from being clipped into a false wall.')
    p.add_argument('--min_depth', type=float, default=0.5, help='Model-internal depth floor (meters).')
    p.add_argument('--pretrained_mae', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--mono_manifest_train', type=str, default=None, help='Path to the train split mono_ga manifest (produced by scripts/precompute_mono_ga.py). Setting this enables Plug-in mode.')
    p.add_argument('--mono_manifest_val', type=str, default=None, help='Path to the val split mono_ga manifest (required when --mono_manifest_train is set).')
    p.add_argument('--seed', type=int, default=None, help='Seed every RNG (Python, NumPy, PyTorch CPU+CUDA, DataLoader workers) for reproducible runs.')
    p.add_argument('--mono_dropout', type=float, default=0.5, help='TacoDepth-style auxiliary-branch dropout: probability of replacing the per-sample mono depth with zeros during training (default 0.5). Set 0.0 for plug-in only.')
    return p.parse_args()


def main():
    args = _parse_args()
    train_image_file = os.path.join(args.manifest_dir, 'training/nuscenes/nuscenes_train_image.txt')
    train_gt_file = os.path.join(args.manifest_dir, 'training/nuscenes/nuscenes_train_D_gt.txt')
    train_gt_acc_file = args.train_gt_acc_file or os.path.join(args.manifest_dir, 'training/nuscenes/nuscenes_train_D_acc_160clean.txt')
    val_image_file = os.path.join(args.manifest_dir, 'validation/nuscenes/nuscenes_val_image.txt')
    val_gt_file = os.path.join(args.manifest_dir, 'validation/nuscenes/nuscenes_val_D_gt.txt')
    train_radar_file = os.path.join(args.manifest_dir, 'training/nuscenes/nuscenes_train_radar.txt')
    val_radar_file = os.path.join(args.manifest_dir, 'validation/nuscenes/nuscenes_val_radar.txt')
    train_image_paths = data_utils.read_paths(train_image_file)
    train_radar_paths = data_utils.read_paths(train_radar_file)
    train_gt_acc_paths = data_utils.read_paths(train_gt_acc_file)
    train_gt_paths = data_utils.read_paths(train_gt_file)
    val_image_paths = data_utils.read_paths(val_image_file)
    val_radar_paths = data_utils.read_paths(val_radar_file)
    val_gt_paths = data_utils.read_paths(val_gt_file)
    train_mono_paths = None
    val_mono_paths = None
    plug_in = args.mono_manifest_train is not None
    if plug_in:
        assert args.mono_manifest_val is not None, '--mono_manifest_train set but --mono_manifest_val missing'
        train_mono_paths = data_utils.read_paths(args.mono_manifest_train)
        val_mono_paths = data_utils.read_paths(args.mono_manifest_val)
        train_mono_root = os.path.dirname(args.mono_manifest_train)
        val_mono_root = os.path.dirname(args.mono_manifest_val)
        if len(train_mono_paths) != len(train_image_paths):
            mask, train_mono_paths = _align_to_mono(train_radar_paths, train_mono_paths, train_mono_root, 'train')
            n_drop = len(train_image_paths) - sum(mask)
            print(f'[plug-in] dropping {n_drop} train frames missing from mono manifest')
            train_image_paths = [p for p, k in zip(train_image_paths, mask) if k]
            train_radar_paths = [p for p, k in zip(train_radar_paths, mask) if k]
            train_gt_paths = [p for p, k in zip(train_gt_paths, mask) if k]
            train_gt_acc_paths = [p for p, k in zip(train_gt_acc_paths, mask) if k]
        if len(val_mono_paths) != len(val_image_paths):
            mask, val_mono_paths = _align_to_mono(val_radar_paths, val_mono_paths, val_mono_root, 'val')
            n_drop = len(val_image_paths) - sum(mask)
            print(f'[plug-in] dropping {n_drop} val frames missing from mono manifest')
            val_image_paths = [p for p, k in zip(val_image_paths, mask) if k]
            val_radar_paths = [p for p, k in zip(val_radar_paths, mask) if k]
            val_gt_paths = [p for p, k in zip(val_gt_paths, mask) if k]
        assert len(train_mono_paths) == len(train_image_paths), f'train mono manifest length {len(train_mono_paths)} != {len(train_image_paths)} images'
        assert len(val_mono_paths) == len(val_image_paths), f'val mono manifest length {len(val_mono_paths)} != {len(val_image_paths)} images'
    print(f'Train: {len(train_image_paths)} samples')
    print(f'Val:   {len(val_image_paths)} samples')
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    depth_tag = f'd{int(args.max_depth)}'
    plugin_tag = '_plugin' if plug_in else ''
    exp_name = f'semodepth_ns_{depth_tag}{plugin_tag}_{ts}'
    train(
        train_image_paths=train_image_paths,
        train_radar_paths=train_radar_paths,
        train_gt_paths=train_gt_acc_paths,
        train_sparse_gt_paths=train_gt_paths,
        train_mono_paths=train_mono_paths,
        val_mono_paths=val_mono_paths,
        eval_mono_paths=val_mono_paths,
        val_image_paths=None,
        val_radar_paths=None,
        val_sparse_gt_paths=None,
        eval_image_paths=val_image_paths,
        eval_radar_paths=val_radar_paths,
        eval_sparse_gt_paths=val_gt_paths,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_grad=args.lambda_grad,
        lambda_linear=args.lambda_linear,
        lambda_log=args.lambda_log,
        lambda_sparse=args.lambda_sparse,
        pretrained_mae=args.pretrained_mae,
        mono_dropout_prob=args.mono_dropout,
        resume=args.resume,
        augment=False,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        checkpoint_dirpath=os.path.join(args.log_root, exp_name),
        dataset_class=NuScenesDataset,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
