# SemoDepth

> Paper: [Selection, Not Fusion: Radar-Modulated State Space Models for Radar-Camera Depth Estimation](https://arxiv.org/abs/2605.11840)

Radar-camera depth estimation with **Radar-Modulated Selection (RMS)** — radar features are injected inside the Mamba selective scan via the step size $\Delta_t$ and readout $\mathbf{C}_t$, while the input projection $\mathbf{B}_t$ and state-evolution matrix $\mathbf{A}$ remain image-only. The Mamba main stream carries pure-image tokens; radar enters only via the $\Delta_t$/$\mathbf{C}_t$ paths.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Two dependencies need a CUDA build environment and are installed separately:

```bash
pip install mamba-ssm causal-conv1d
```

For the nuScenes data path (ground-truth accumulation, day/night split):

```bash
pip install nuscenes-devkit
```

## Datasets

- **nuScenes** — full driving dataset; download from the official nuScenes website. Train/val/test manifests follow the radar-camera-fusion-depth conventions used by recent radar-camera depth work (TacoDepth, CaFNet, Li et al., Singh et al.). Pre-built manifests live under a single `manifest_dir`.
- **ZJU-4DRadarCam** — campus dataset with 4D radar; download from the original release.

Hard paths are not baked in. Each script accepts CLI flags or environment variables:

| Variable                 | Used by                  | Purpose                                          |
|--------------------------|--------------------------|--------------------------------------------------|
| `NUSCENES_MANIFEST_DIR`  | `train_nuscenes.py`      | Default `--manifest_dir`                         |
| `NUSCENES_LOG_ROOT`      | `train_nuscenes.py`      | Default `--log_root` (where checkpoints land)    |
| `ZJU_DATA_ROOT`          | `train_zju.py`           | Default `--data_root`                            |
| `ZJU_LOG_ROOT`           | `train_zju.py`           | Default `--log_root`                             |

## Train

```bash
python scripts/train_nuscenes.py \
    --manifest_dir /path/to/manifests \
    --epochs 50 --batch_size 12 --lambda_grad 0.5

python scripts/train_zju.py \
    --data_root /path/to/ZJU-4DRadarCam/data \
    --epochs 100 --batch_size 12 --lambda_grad 0.5
```

## Evaluate

```bash
python scripts/eval_nuscenes.py \
    --checkpoint /path/to/checkpoint.pt \
    --manifest_dir /path/to/manifests
```

For the day/night protocol (TacoDepth Table 7):

```bash
python scripts/eval_nuscenes.py \
    --checkpoint /path/to/checkpoint.pt \
    --manifest_dir /path/to/manifests \
    --frame_subset day \
    --dataroot /path/to/nuScenes
```

## Build accumulated LiDAR ground truth (nuScenes)

The model is trained on a 160-sweep accumulated supervision target with a horizon-cleaning pass. Build it from the raw dataset:

```bash
python scripts/accumulate_lidar_gt.py \
    --dataroot /path/to/nuScenes \
    --derived_dir /path/to/nuScenes_derived \
    --output_subdir gt_lidar_160sweep --n_sweeps 160 --n_thread 40

python scripts/clean_160sweep_horizon.py \
    --src_root /path/to/nuScenes_derived/gt_lidar_160sweep \
    --output_dir /path/to/nuScenes_derived/gt_lidar_160sweep_cleaned
```

## Tests

```bash
pytest tests/
```

Tests cover the zero-init parity guarantee (the radar-side projections start at zero so the block matches vanilla Mamba at init) and gradient flow through each pathway. They require CUDA and `mamba-ssm`; tests are skipped automatically without them.

## Repository layout

- `mambadepth_fusion/model/` — RMS block (`cross_modal_mamba.py`), top-level model (`depth_net.py`), MVSP blocks (`fusion_blocks.py`), Radar GSE (`radar_gse.py`)
- `mambadepth_fusion/data/` — nuScenes and ZJU dataset loaders
- `mambadepth_fusion/train/` — training loop with the composite loss
- `mambadepth_fusion/utils/` — evaluation metrics
- `scripts/` — CLI wrappers for training, evaluation, qualitative visualization, ground-truth construction, and supervision cleaning
- `tests/` — smoke tests for the RMS block
