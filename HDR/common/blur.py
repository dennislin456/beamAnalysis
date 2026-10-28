"""Separable Gaussian blur used before an HDR exposure is fused.

Copied from FourStepAdj ``merge_exposures.gaussian_blur``.
"""

from __future__ import annotations

import numpy as np


def gaussian_blur(data: np.ndarray, width_x: int, width_y: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. A kernel width of 1 leaves that axis unchanged."""
    values = np.asarray(data, dtype=float)
    kx = int(width_x)
    ky = int(width_y)
    if values.ndim != 2 or values.size == 0 or (kx <= 1 and ky <= 1):
        return values
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return values
    from scipy.ndimage import convolve1d

    out = values
    if ky > 1:
        out = convolve1d(out, _gaussian_kernel(ky, sigma), axis=0, mode="nearest")
    if kx > 1:
        out = convolve1d(out, _gaussian_kernel(kx, sigma), axis=1, mode="nearest")
    return out


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    half = (int(size) - 1) / 2.0
    positions = np.arange(int(size), dtype=np.float64) - half
    kernel = np.exp(-0.5 * (positions / float(sigma)) ** 2)
    return kernel / kernel.sum()
