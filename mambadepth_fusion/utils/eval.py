import numpy as np

def root_mean_sq_err(src, tgt):
    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    return np.sqrt(np.mean((1.0 / tgt - 1.0 / src) ** 2))

def inv_mean_abs_err(src, tgt):
    return np.mean(np.abs(1.0 / tgt - 1.0 / src))

def mean_abs_rel_err(src, tgt):
    return np.mean(np.abs(src - tgt) / tgt)

def mean_sq_rel_err(src, tgt):
    return np.mean((src - tgt) ** 2 / tgt)

def thr_acc(src, tgt, thr=1.25):
    return np.mean(np.maximum(tgt / src, src / tgt) < thr)
