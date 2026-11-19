"""Geometry-preserving brightness experiments on BGR uint8 images."""
import math
import cv2
import numpy as np


def settings(mode='none', gamma=0.8, clip=2.0, grid=8):
    if mode == 'none':
        return {'mode': 'none'}
    if mode == 'gamma':
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError('gamma must be finite and positive')
        return {'mode': mode, 'gamma': gamma}
    if mode == 'clahe':
        if not math.isfinite(clip) or clip <= 0 or grid < 1:
            raise ValueError('CLAHE clip and grid must be positive')
        return {'mode': mode, 'clip_limit': clip, 'grid': grid}
    raise ValueError(mode)


def tag(config):
    if config['mode'] == 'gamma':
        return f"gamma{config['gamma']:g}"
    if config['mode'] == 'clahe':
        return f"clahe{config['clip_limit']:g}_grid{config['grid']}"
    return 'original'


def apply(bgr, config):
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Expected H x W x 3 uint8 BGR image')
    if config['mode'] == 'none':
        return bgr.copy()
    if config['mode'] == 'gamma':
        # Explicit convention: output = 255 * (input/255)**gamma; gamma < 1 brightens.
        lut = np.rint(255 * (np.arange(256, dtype=np.float64) / 255) ** config['gamma']).astype(np.uint8)
        return cv2.LUT(bgr, lut)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    grid = config['grid']
    clahe = cv2.createCLAHE(clipLimit=config['clip_limit'], tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def read_processed(path, config):
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return apply(bgr, config)
