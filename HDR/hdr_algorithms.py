"""Debevec, Robertson, and OpenCV Mertens HDR merge helpers used by hdr_test.py.

The self-written Mertens lives in mertens_local.py and is dispatched from run_hdr.
"""

from __future__ import annotations

import cv2
import numpy as np

from mertens_local import merge_mertens_local


def to_uint8(image: np.ndarray, full_scale: float) -> np.ndarray:
    """Map one exposure onto the 8-bit range OpenCV HDR expects."""
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    scale = float(full_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("滿量必須是大於 0 的數字。")
    scaled = np.clip(np.asarray(image, dtype=np.float64) / scale * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(np.rint(scaled).astype(np.uint8))


def _prepare_ldr(images: list[np.ndarray], full_scale: float) -> tuple[list[np.ndarray], bool]:
    gray = images[0].ndim == 2
    ldr = []
    for image in images:
        uint8 = to_uint8(image, full_scale)
        if uint8.ndim == 2:
            uint8 = np.repeat(uint8[:, :, None], 3, axis=2)
        ldr.append(np.ascontiguousarray(uint8))
    return ldr, gray


def _take_gray_channel(
    merged: np.ndarray,
    response: np.ndarray | None,
    gray: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    merged = np.asarray(merged, dtype=np.float32)
    if response is not None:
        response = np.asarray(response, dtype=np.float32)
    if gray:
        if merged.ndim == 3:
            merged = merged[:, :, 0]
        if response is not None and response.ndim == 3:
            response = response[:, :, :1]
    return merged, response


def merge_debevec(
    images: list[np.ndarray],
    times_us: list[float],
    full_scale: float,
    settings: dict[str, float | bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a response curve, then merge into a float32 radiance map."""
    ldr, gray = _prepare_ldr(images, full_scale)
    seconds = np.ascontiguousarray(np.asarray(times_us, dtype=np.float32) * 1e-6)
    calibrate = cv2.createCalibrateDebevec()
    calibrate.setSamples(max(int(settings["samples"]), 1))
    calibrate.setLambda(float(settings["lambda"]))
    calibrate.setRandom(bool(settings["random"]))
    response = calibrate.process(ldr, seconds)
    merged = cv2.createMergeDebevec().process(ldr, seconds, response)
    return _take_gray_channel(merged, response, gray)


def merge_robertson(
    images: list[np.ndarray],
    times_us: list[float],
    full_scale: float,
    settings: dict[str, float | bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively estimate a response curve, then merge into a float32 radiance map."""
    ldr, gray = _prepare_ldr(images, full_scale)
    seconds = np.ascontiguousarray(np.asarray(times_us, dtype=np.float32) * 1e-6)
    calibrate = cv2.createCalibrateRobertson()
    calibrate.setMaxIter(max(int(settings["max_iter"]), 1))
    calibrate.setThreshold(float(settings["threshold"]))
    response = calibrate.process(ldr, seconds)
    merged = cv2.createMergeRobertson().process(ldr, seconds, response)
    return _take_gray_channel(merged, response, gray)


def merge_mertens(
    images: list[np.ndarray],
    full_scale: float,
    settings: dict[str, float | bool],
) -> tuple[np.ndarray, None]:
    """Fuse exposures by contrast, saturation, and well-exposedness. No response curve."""
    ldr, gray = _prepare_ldr(images, full_scale)
    merge = cv2.createMergeMertens(
        float(settings["contrast"]),
        float(settings["saturation"]),
        float(settings["exposure"]),
    )
    merged = merge.process(ldr)
    merged = np.maximum(np.asarray(merged, dtype=np.float32), 0) * np.float32(255.0)
    return _take_gray_channel(merged, None, gray)


def run_hdr(
    images: list[np.ndarray],
    times_us: list[float],
    algorithm: str,
    full_scale: float,
    settings: dict[str, float | bool],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run one HDR method. Returns the float result and, for calibration methods, the response."""
    if len(images) < 2:
        raise ValueError("至少需要兩張曝光。")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("每張影像的尺寸和通道數要相同。")
    times = np.asarray(times_us, dtype=np.float32)
    if times.shape != (len(images),) or not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("每張曝光時間都要是大於 0 的數字。")
    if algorithm not in {"mertens", "mertens_local"} and np.allclose(times, times[0]):
        raise ValueError("Debevec 和 Robertson 需要不相同的曝光時間。")
    if algorithm == "debevec":
        return merge_debevec(images, times_us, full_scale, settings)
    if algorithm == "robertson":
        return merge_robertson(images, times_us, full_scale, settings)
    if algorithm == "mertens":
        return merge_mertens(images, full_scale, settings)
    if algorithm == "mertens_local":
        return merge_mertens_local(images, full_scale, settings)
    raise ValueError(f"未知的演算法: {algorithm}")
