import numpy as np
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
import matplotlib.pyplot as plt

def load_data_path(root, file_name_txt, data_type):
    with open(file_name_txt, 'r') as f:
        data_path = f.readlines()
    data_path = [root + x.strip() + data_type for x in data_path]
    return data_path

def load_data_path_nu(root, name_list, data_type):
    data_path = [root + x.strip() + data_type for x in name_list]
    return data_path

def read_paths(filepath):
    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            if path == '':
                break
            path_list.append(path)
    return path_list

def write_paths(filepath, paths):
    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=False, data_format='HWC'):
    image = Image.open(path).convert('RGB')
    image = np.asarray(image, np.float32)
    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))
    image = image / 255.0 if normalize else image
    return image

def load_depth(path, multiplier=256.0, data_format='HW'):
    z = np.array(Image.open(path), dtype=np.float32)
    z = z / multiplier
    z[z <= 0] = 0.0
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))
    return z

def save_depth(z, path, multiplier=256.0):
    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def save_color_depth(z, path):
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    colormap = plt.cm.viridis
    z_color = colormap(z_normalized)
    z_color = np.uint8(z_color * 255)
    image = Image.fromarray(z_color)
    image.save(path)

def load_response(path, multiplier=2 ** 14, data_format='HW'):
    response = np.array(Image.open(path), dtype=np.float32)
    response = response / multiplier
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        response = np.expand_dims(response, axis=0)
    elif data_format == 'HWC':
        response = np.expand_dims(response, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))
    return response

def save_response(response, path, multiplier=2 ** 14):
    response = np.uint32(response * multiplier)
    response = Image.fromarray(response, mode='I')
    response.save(path)

def interpolate_depth(depth_map, validity_map, log_space=False):
    assert depth_map.ndim == 2 and validity_map.ndim == 2
    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]
    if log_space:
        depth_values = np.log(depth_values)
    interpolator = LinearNDInterpolator(points=np.stack([data_row_idx, data_col_idx], axis=1), values=depth_values, fill_value=0 if not log_space else np.log(0.001))
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    Z = interpolator(query_coord).reshape([rows, cols])
    if log_space:
        Z = np.exp(Z)
        Z[Z < 0.1] = 0.0
    return Z

def interpolate_depth_ZJU(depth_map, validity_map=None, log_space=False, window_size=12):
    assert depth_map.ndim == 2
    if validity_map is None:
        validity_map = depth_map > 0.0
    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]
    if log_space:
        depth_values = np.log(depth_values)
    interpolator = LinearNDInterpolator(points=np.stack([data_row_idx, data_col_idx], axis=1), values=depth_values, fill_value=0 if not log_space else np.log(0.001))
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    Z = np.zeros_like(depth_map)
    query_indices = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    window_indices = np.indices((window_size, window_size)).reshape(2, -1) - window_size // 2
    window_row_indices = np.clip(query_indices[:, 0, None] + window_indices[0], 0, rows - 1)
    window_col_indices = np.clip(query_indices[:, 1, None] + window_indices[1], 0, cols - 1)
    window_values = depth_map[window_row_indices, window_col_indices]
    valid_indices = np.any(window_values > 0, axis=1)
    valid_query_indices = np.where(valid_indices)[0]
    valid_query_coords = query_indices[valid_query_indices]
    Z.ravel()[valid_query_indices] = interpolator(valid_query_coords)
    if log_space:
        Z = np.exp(Z)
        Z[Z < 0.1] = 0.0
    return Z
