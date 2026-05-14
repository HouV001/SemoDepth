import argparse
import datetime
import os
from mambadepth_fusion.data import data_utils
from mambadepth_fusion.train import train


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default=os.environ.get('ZJU_DATA_ROOT', '/path/to/ZJU-4DRadarCam/data'))
    p.add_argument('--log_root', type=str, default=os.environ.get('ZJU_LOG_ROOT', '/path/to/ZJU-4DRadarCam/log/mambadepth'))
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--lambda_grad', type=float, default=0.5)
    p.add_argument('--lambda_linear', type=float, default=1.0)
    p.add_argument('--max_depth', type=float, default=80.0, help='Model-internal depth ceiling (meters). Eval is fixed at {50, 70, 80} m regardless. ZJU default 80 m matches the 4D-radar range distribution.')
    p.add_argument('--min_depth', type=float, default=0.5, help='Model-internal depth floor (meters).')
    p.add_argument('--pretrained_mae', type=str, default=None, help='Path to MAE pre-trained GNN weights.')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=None, help='Seed every RNG (Python, NumPy, PyTorch CPU+CUDA, DataLoader workers) for reproducible runs.')
    return p.parse_args()


def main():
    args = _parse_args()
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f'semodepth_zju_{ts}'
    root = args.data_root
    train_list = os.path.join(root, 'train.txt')
    val_list = os.path.join(root, 'val.txt')
    test_list = os.path.join(root, 'test.txt')
    gt_interp_dir = root + '/gt_interp/'
    gt_scatter_dir = root + '/gt/'
    train(
        train_image_paths=data_utils.load_data_path(root + '/image/', train_list, '.png'),
        train_radar_paths=data_utils.load_data_path(root + '/radar_png/', train_list, '.png'),
        train_gt_paths=data_utils.load_data_path(gt_interp_dir, train_list, '.png'),
        train_sparse_gt_paths=data_utils.load_data_path(gt_scatter_dir, train_list, '.png'),
        val_image_paths=data_utils.load_data_path(root + '/image/', val_list, '.png'),
        val_radar_paths=data_utils.load_data_path(root + '/radar_png/', val_list, '.png'),
        val_sparse_gt_paths=data_utils.load_data_path(root + '/gt/', val_list, '.png'),
        eval_image_paths=data_utils.load_data_path(root + '/image/', test_list, '.png'),
        eval_radar_paths=data_utils.load_data_path(root + '/radar_png/', test_list, '.png'),
        eval_sparse_gt_paths=data_utils.load_data_path(root + '/gt/', test_list, '.png'),
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_grad=args.lambda_grad,
        lambda_linear=args.lambda_linear,
        pretrained_mae=args.pretrained_mae,
        resume=args.resume,
        augment=False,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        checkpoint_dirpath=os.path.join(args.log_root, exp_name),
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
