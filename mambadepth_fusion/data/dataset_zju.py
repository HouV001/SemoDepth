import os
import torch.utils.data
import numpy as np
from PIL import Image

def load_depth_png(path):
    z = np.array(Image.open(path), dtype=np.float32) / 256.0
    z[z <= 0] = 0.0
    return z

def load_image(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)

def _try_npy(png_path, loader_fn):
    npy_path = png_path.replace('/image/', '/image_npy/').replace('/radar_png/', '/radar_npy/').replace('/gt/', '/gt_npy/').replace('/gt_interp/', '/gt_interp_npy/').replace('.png', '.npy')
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return loader_fn(png_path)

class ZJUDataset(torch.utils.data.Dataset):

    def __init__(self, image_paths, radar_paths, gt_paths, sparse_gt_paths, z_paths=None, augment=False, prop_dir=None, max_depth=None):
        self.n_sample = len(image_paths)
        assert all((len(p) == self.n_sample for p in [radar_paths, gt_paths, sparse_gt_paths]))
        if z_paths is not None:
            assert len(z_paths) == self.n_sample
        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.gt_paths = gt_paths
        self.sparse_gt_paths = sparse_gt_paths
        self.z_paths = z_paths
        self.augment = augment
        self.prop_dir = prop_dir
        self.max_depth = max_depth

    def __getitem__(self, index):
        image = _try_npy(self.image_paths[index], load_image)
        radar = _try_npy(self.radar_paths[index], load_depth_png)
        gt = _try_npy(self.gt_paths[index], load_depth_png)
        sparse_gt = _try_npy(self.sparse_gt_paths[index], load_depth_png)
        if image.shape[0] == 720:
            y0, y1 = (720 // 3, 720 // 4 * 3)
            image = image[y0:y1, :, :]
            radar = radar[y0:y1, :]
            gt = gt[y0:y1, :]
            sparse_gt = sparse_gt[y0:y1, :]
        if self.max_depth is not None:
            radar = np.clip(radar, 0, self.max_depth)
        prop_data = None
        if self.prop_dir is not None:
            basename = os.path.basename(self.image_paths[index]).split('.')[0]
            npz_path = os.path.join(self.prop_dir, f'{basename}.npz')
            if os.path.exists(npz_path):
                npz = np.load(npz_path)
                prop_data = {'d0': npz['d0'].astype(np.float32), 'bilat_conf': npz['bilat_conf'].astype(np.float32), 'radar_feats': npz['radar_feats'].astype(np.float32), 'edge_map': npz['edge_map'].astype(np.float32), 'd0_mismatch': npz['d0_mismatch'].astype(np.float32)}
        z = None
        if self.z_paths is not None:
            z = load_depth_png(self.z_paths[index])
        if self.augment and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            radar = np.ascontiguousarray(radar[:, ::-1])
            gt = np.ascontiguousarray(gt[:, ::-1])
            sparse_gt = np.ascontiguousarray(sparse_gt[:, ::-1])
            if z is not None:
                z = np.ascontiguousarray(z[:, ::-1])
            if prop_data is not None:
                prop_data = {k: np.ascontiguousarray(v[..., ::-1]) for k, v in prop_data.items()}
        out = (image.astype(np.float32), radar.astype(np.float32), gt.astype(np.float32), sparse_gt.astype(np.float32))
        if prop_data is not None:
            out = out + (prop_data['d0'], prop_data['bilat_conf'], prop_data['radar_feats'], prop_data['edge_map'], prop_data['d0_mismatch'])
        elif z is not None:
            out = out + (z.astype(np.float32),)
        return out

    def __len__(self):
        return self.n_sample
