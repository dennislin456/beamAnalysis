"""Local Mertens exposure fusion: contrast / saturation / well-exposedness + Laplacian pyramid.

This is a self-contained implementation used by hdr_test.py. It follows the same
weighting idea as OpenCV MergeMertens, but the exposure-bell width (sigma) is
adjustable, and grayscale stays single-channel so saturation is skipped.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_SIGMA = 0.2


def _to_uint8(image: np.ndarray, full_scale: float) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    scale = float(full_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("滿量必須是大於 0 的數字。")
    scaled = np.clip(np.asarray(image, dtype=np.float64) / scale * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(np.rint(scaled).astype(np.uint8))


def _as_float01(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data.astype(np.float32) * (1.0 / 255.0))
    return np.ascontiguousarray(np.clip(data.astype(np.float32), 0.0, 1.0))


def _power_safe(values: np.ndarray, exponent: float) -> np.ndarray:
    """Match OpenCV: 0**0 becomes 1; other non-positive bases stay 0 for positive exponents."""
    out = np.empty(values.shape, dtype=np.float32)
    cv2.pow(np.asarray(values, dtype=np.float32), float(exponent), out)
    return out


def weight_map(
    image01: np.ndarray,
    contrast: float,
    saturation: float,
    exposure: float,
    sigma: float,
) -> np.ndarray:
    """Build one exposure's weight map on the 0–1 image."""
    data = np.asarray(image01, dtype=np.float32)
    if data.ndim == 2:
        gray = data
        channels = [data]
        use_saturation = False
    elif data.ndim == 3 and data.shape[2] == 3:
        gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        channels = [data[:, :, index] for index in range(3)]
        use_saturation = True
    else:
        raise ValueError(f"影像必須是灰階或 RGB，形狀為 {data.shape}")

    contrast_map = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    contrast_map = _power_safe(contrast_map, contrast)

    if use_saturation:
        mean = (channels[0] + channels[1] + channels[2]) * (1.0 / 3.0)
        sat = np.zeros(gray.shape, dtype=np.float32)
        for channel in channels:
            deviation = channel - mean
            sat += deviation * deviation
        sat = np.sqrt(sat)
        sat = _power_safe(sat, saturation)
        weights = contrast_map * sat
    else:
        weights = contrast_map

    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma 必須是大於 0 的數字。")
    denom = 2.0 * sigma * sigma
    well = np.ones(gray.shape, dtype=np.float32)
    for channel in channels:
        well *= np.exp(-((channel - 0.5) ** 2) / denom)
    well = _power_safe(well, exposure)
    return weights * well + 1e-12


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build levels+1 images: original plus `levels` successive pyrDown results."""
    current = np.ascontiguousarray(image, dtype=np.float32)
    pyramid = [current]
    for _ in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = _gaussian_pyramid(image, levels)
    pyramid: list[np.ndarray] = []
    for level in range(levels):
        up = cv2.pyrUp(gauss[level + 1], dstsize=(gauss[level].shape[1], gauss[level].shape[0]))
        pyramid.append(gauss[level] - up)
    pyramid.append(gauss[levels])
    return pyramid


def fuse_laplacian(
    images01: list[np.ndarray],
    weights: list[np.ndarray],
) -> np.ndarray:
    """Blend float images with normalized weights using a Laplacian pyramid."""
    if len(images01) < 2 or len(images01) != len(weights):
        raise ValueError("影像和權重張數要相同，而且至少兩張。")
    height, width = images01[0].shape[:2]
    levels = int(np.floor(np.log2(min(height, width))))
    if levels < 1:
        stacked = np.stack([np.asarray(image, dtype=np.float32) for image in images01], axis=0)
        weight = np.stack([np.asarray(item, dtype=np.float32) for item in weights], axis=0)
        weight_sum = np.sum(weight, axis=0, keepdims=True)
        weight = weight / weight_sum
        if stacked.ndim == 3:
            return np.sum(stacked * weight, axis=0)
        return np.sum(stacked * weight[:, :, :, None], axis=0)

    weight_sum = np.zeros(weights[0].shape, dtype=np.float32)
    for weight in weights:
        weight_sum += weight
    norm_weights = [weight / weight_sum for weight in weights]

    result_levels: list[np.ndarray | None] = [None] * (levels + 1)
    for image, weight in zip(images01, norm_weights):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            img_pyr = _laplacian_pyramid(image, levels)
            weight_pyr = _gaussian_pyramid(weight, levels)
            for level in range(levels + 1):
                blended = img_pyr[level] * weight_pyr[level]
                if result_levels[level] is None:
                    result_levels[level] = blended
                else:
                    result_levels[level] = result_levels[level] + blended
        else:
            img_pyr = _laplacian_pyramid(image, levels)
            weight_pyr = _gaussian_pyramid(weight, levels)
            for level in range(levels + 1):
                pieces = []
                for channel in range(image.shape[2]):
                    pieces.append(img_pyr[level][:, :, channel] * weight_pyr[level])
                blended = np.stack(pieces, axis=2)
                if result_levels[level] is None:
                    result_levels[level] = blended
                else:
                    result_levels[level] = result_levels[level] + blended

    result = result_levels[levels]
    assert result is not None
    for level in range(levels - 1, -1, -1):
        current = result_levels[level]
        assert current is not None
        up = cv2.pyrUp(result, dstsize=(current.shape[1], current.shape[0]))
        result = current + up
    return np.asarray(result, dtype=np.float32)


def merge_mertens_local(
    images: list[np.ndarray],
    full_scale: float,
    settings: dict[str, float | bool],
) -> tuple[np.ndarray, None]:
    """Fuse exposures locally. Returns a float32 fusion image and no response curve."""
    if len(images) < 2:
        raise ValueError("至少需要兩張曝光。")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("每張影像的尺寸和通道數要相同。")

    gray = images[0].ndim == 2
    ldr = []
    for image in images:
        uint8 = _to_uint8(image, full_scale)
        if gray:
            if uint8.ndim != 2:
                raise ValueError("灰階輸入在壓成 8-bit 後仍須是二維。")
            ldr.append(uint8)
        else:
            if uint8.ndim == 2:
                uint8 = np.repeat(uint8[:, :, None], 3, axis=2)
            ldr.append(np.ascontiguousarray(uint8))

    contrast = float(settings.get("contrast", 1.0))
    saturation = float(settings.get("saturation", 1.0))
    exposure = float(settings.get("exposure", 0.0))
    sigma = float(settings.get("sigma", DEFAULT_SIGMA))

    images01 = [_as_float01(frame) for frame in ldr]
    weights = [
        weight_map(frame, contrast, saturation, exposure, sigma) for frame in images01
    ]
    merged = fuse_laplacian(images01, weights)
    if gray and merged.ndim == 3:
        merged = merged[:, :, 0]
    merged = np.maximum(np.asarray(merged, dtype=np.float32), 0) * np.float32(255.0)
    return merged, None
