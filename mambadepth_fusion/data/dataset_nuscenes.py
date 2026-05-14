import os
import torch.utils.data
import numpy as np
from PIL import Image
NU_H, NU_W = (900, 1600)

def load_depth_png(path):
    z = np.array(Image.open(path), dtype=np.float32) / 256.0
    z[z <= 0] = 0.0
    return z

def load_image(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)

def radar_points_to_depth_map(npy_path, H=NU_H, W=NU_W):
    pts = np.load(npy_path).astype(np.float32)
    depth_map = np.zeros((H, W), dtype=np.float32)
    if pts.shape[0] == 0:
        return depth_map
    x = np.round(pts[:, 0]).astype(np.int32)
    y = np.round(pts[:, 1]).astype(np.int32)
    d = pts[:, 2]
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (d > 0)
    x, y, d = (x[valid], y[valid], d[valid])
    order = np.argsort(-d)
    x, y, d = (x[order], y[order], d[order])
    depth_map[y, x] = d
    return depth_map

class NuScenesDataset(torch.utils.data.Dataset):

    def __init__(self, image_paths, radar_paths, gt_paths, sparse_gt_paths, augment=False, max_depth=80.0, mono_paths=None, mono_dropout_prob=0.0):
        self.n_sample = len(image_paths)
        assert all((len(p) == self.n_sample for p in [radar_paths, gt_paths, sparse_gt_paths]))
        if mono_paths is not None:
            assert len(mono_paths) == self.n_sample, f'mono_paths length {len(mono_paths)} != {self.n_sample}'
        assert 0.0 <= mono_dropout_prob <= 1.0
        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.gt_paths = gt_paths
        self.sparse_gt_paths = sparse_gt_paths
        self.mono_paths = mono_paths
        self.mono_dropout_prob = mono_dropout_prob
        self.augment = augment
        self.max_depth = max_depth

    def __getitem__(self, index):
        image = load_image(self.image_paths[index])
        radar_path = self.radar_paths[index]
        if radar_path.endswith('.npy'):
            radar = radar_points_to_depth_map(radar_path, H=image.shape[0], W=image.shape[1])
        else:
            radar = load_depth_png(radar_path)
        if self.max_depth is not None:
            radar = np.clip(radar, 0, self.max_depth)
        gt = load_depth_png(self.gt_paths[index])
        sparse_gt = load_depth_png(self.sparse_gt_paths[index])
        mono = None
        if self.mono_paths is not None:
            if self.mono_dropout_prob > 0.0 and np.random.rand() < self.mono_dropout_prob:
                mono = np.zeros_like(gt)
            else:
                mono = load_depth_png(self.mono_paths[index])
                if self.max_depth is not None:
                    mono = np.clip(mono, 0, self.max_depth)
        if self.augment and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            radar = np.ascontiguousarray(radar[:, ::-1])
            gt = np.ascontiguousarray(gt[:, ::-1])
            sparse_gt = np.ascontiguousarray(sparse_gt[:, ::-1])
            if mono is not None:
                mono = np.ascontiguousarray(mono[:, ::-1])
        out = (image.astype(np.float32), radar.astype(np.float32), gt.astype(np.float32), sparse_gt.astype(np.float32))
        if mono is not None:
            out = out + (mono.astype(np.float32),)
        return out

    def __len__(self):
        return self.n_sample
