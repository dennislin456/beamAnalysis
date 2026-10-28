"""Merge four Basler exposures onto the 50000 us intensity scale.

Each checked exposure can be raised to a power, then color-scaled. Values at or below its
floor become 0, and values at or above its ceiling become that ceiling. A blank
bound is skipped. Defaults are 256 and 4096. A shorter exposure then keeps a
pixel when the scaled value times 10 reaches the threshold (default 4000). Kept pixels are multiplied by that
exposure's scale. Defaults:

    50 us    -> value * 1000
    500 us   -> value * 100
    5000 us  -> value * 10
    50000 us -> value * 1

50000 us is the base when it is included with other exposures: color scale, then
multiply, with no keep test. If only one exposure checkbox is on, the result is
that exposure after color scale and its scale, for every pixel.
If several shorter exposures qualify at the same pixel, the shorter one is used.
Overwrite mode then smooths that finished image. Linear and log modes instead
smooth each exposure before its color scale.
Linear and log combine modes skip the keep test and add every checked exposure
after its own power, smooth, color scale, and scale. Log uses log10(max(value, 1)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_FULL_SCALE = 4000.0
NEXT_EXPOSURE_FACTOR = 10.0
EXPOSURE_COLOR_FLOOR = 256.0
EXPOSURE_COLOR_CEIL = 4096.0

# token, exposure_us, scale onto 50000 us
EXPOSURE_SPECS = (
    ("50us", 50, 1000.0),
    ("500us", 500, 100.0),
    ("5000us", 5000, 10.0),
    ("50000us", 50000, 1.0),
)


@dataclass(frozen=True)
class MergeResult:
    image: np.ndarray
    source: np.ndarray  # exposure us that supplied each pixel
    masks: dict[str, np.ndarray]
    counts: dict[str, int | str]


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


def keep_mask(image: np.ndarray, threshold: float = CAMERA_FULL_SCALE) -> np.ndarray:
    """Pixels whose value * 10 is already at or above the threshold."""
    values = np.asarray(image, dtype=np.float64)
    return values * NEXT_EXPOSURE_FACTOR >= float(threshold)


def raise_power(image: np.ndarray, exponent: float) -> np.ndarray:
    """Raise each pixel to exponent. Non-finite results, including negative bases with a fractional power, become 0."""
    values = np.asarray(image, dtype=np.float64)
    power = float(exponent)
    if not np.isfinite(power):
        raise ValueError(f"次方必須是有限數字，目前為 {exponent}")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = np.power(values, power)
    return np.where(np.isfinite(out), out, 0.0)


def clip_exposure(image: np.ndarray, floor: float | None, ceil: float | None) -> np.ndarray:
    """<= floor becomes 0, >= ceil becomes ceil. A missing bound is not applied."""
    values = np.asarray(image, dtype=np.float64)
    lo = None if floor is None else float(floor)
    hi = None if ceil is None else float(ceil)
    if lo is not None and not np.isfinite(lo):
        raise ValueError(f"color scale 下限必須是有限數字，目前為 {floor}")
    if hi is not None and not np.isfinite(hi):
        raise ValueError(f"color scale 上限必須是有限數字，目前為 {ceil}")
    if lo is not None and hi is not None and hi < lo:
        lo, hi = hi, lo
    out = np.array(values, dtype=np.float64, copy=True)
    if lo is not None:
        out[out <= lo] = 0.0
    if hi is not None:
        out[out >= hi] = hi
    return out


def merge_four_exposures(
    img_50us: np.ndarray | None,
    img_500us: np.ndarray | None,
    img_5000us: np.ndarray | None,
    img_50000us: np.ndarray | None,
    threshold: float = CAMERA_FULL_SCALE,
    scale_50us: float = 1000.0,
    scale_500us: float = 100.0,
    scale_5000us: float = 10.0,
    scale_50000us: float = 1.0,
    floor_50us: float | None = EXPOSURE_COLOR_FLOOR,
    ceil_50us: float | None = EXPOSURE_COLOR_CEIL,
    floor_500us: float | None = EXPOSURE_COLOR_FLOOR,
    ceil_500us: float | None = EXPOSURE_COLOR_CEIL,
    floor_5000us: float | None = EXPOSURE_COLOR_FLOOR,
    ceil_5000us: float | None = EXPOSURE_COLOR_CEIL,
    floor_50000us: float | None = EXPOSURE_COLOR_FLOOR,
    ceil_50000us: float | None = EXPOSURE_COLOR_CEIL,
    use_50us: bool = True,
    use_500us: bool = True,
    use_5000us: bool = True,
    use_50000us: bool = True,
    power_50us: float | None = None,
    power_500us: float | None = None,
    power_5000us: float | None = None,
    power_50000us: float | None = None,
    smooth_x: int = 1,
    smooth_y: int = 1,
    sigma: float = 0.8,
    combine: str = "overwrite",
) -> MergeResult:
    raw = {
        "50us": img_50us,
        "500us": img_500us,
        "5000us": img_5000us,
        "50000us": img_50000us,
    }
    uses = {
        "50us": bool(use_50us),
        "500us": bool(use_500us),
        "5000us": bool(use_5000us),
        "50000us": bool(use_50000us),
    }
    loaded = {name: np.asarray(img) for name, img in raw.items() if img is not None}
    if not loaded:
        raise ValueError("這一筆沒有任何曝光圖")
    if not any(uses.values()):
        raise ValueError("請至少勾選一段曝光")
    images = {name: img for name, img in loaded.items() if uses[name]}
    if not images:
        raise ValueError("勾選的曝光都沒有載入影像")
    shapes = {name: arr.shape for name, arr in images.items()}
    if any(arr.ndim != 2 for arr in images.values()):
        raise ValueError(f"每張圖都必須是二維陣列，目前形狀為 {shapes}")
    ref_shape = next(iter(images.values())).shape
    if any(arr.shape != ref_shape for arr in images.values()):
        raise ValueError(f"曝光圖尺寸必須相同，目前形狀為 {shapes}")

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"門檻必須是大於等於 0 的數字，目前為 {threshold}")
    threshold = float(threshold)
    scales = {
        "50us": float(scale_50us),
        "500us": float(scale_500us),
        "5000us": float(scale_5000us),
        "50000us": float(scale_50000us),
    }
    floors = {
        "50us": floor_50us,
        "500us": floor_500us,
        "5000us": floor_5000us,
        "50000us": floor_50000us,
    }
    ceils = {
        "50us": ceil_50us,
        "500us": ceil_500us,
        "5000us": ceil_5000us,
        "50000us": ceil_50000us,
    }
    powers = {
        "50us": power_50us,
        "500us": power_500us,
        "5000us": power_5000us,
        "50000us": power_50000us,
    }
    for name, scale in scales.items():
        if not np.isfinite(scale):
            raise ValueError(f"{name} 倍率必須是有限數字，目前為 {scale}")
    smooth_x = max(1, int(smooth_x))
    smooth_y = max(1, int(smooth_y))
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma 必須是大於等於 0 的有限數字，目前為 {sigma}")
    blur = smooth_x > 1 or smooth_y > 1
    prepared = {}
    for name in images:
        values = images[name]
        if powers[name] is not None:
            values = raise_power(values, powers[name])
        if combine in {"linear", "log"} and blur:
            values = gaussian_blur(values, smooth_x, smooth_y, sigma)
        prepared[name] = clip_exposure(values, floors[name], ceils[name])
    exposure_us = {"50us": 50, "500us": 500, "5000us": 5000, "50000us": 50000}
    checked_count = sum(1 for name, enabled in uses.items() if enabled)
    solo = next(iter(prepared)) if checked_count == 1 else None
    if combine not in {"overwrite", "linear", "log"}:
        raise ValueError(f"合成方式必須是覆寫、線性相加或 log 相加，目前為 {combine}")

    mask_5000 = np.zeros(ref_shape, dtype=bool)
    mask_500 = np.zeros(ref_shape, dtype=bool)
    mask_50 = np.zeros(ref_shape, dtype=bool)
    if combine in {"linear", "log"}:
        merged = np.zeros(ref_shape, dtype=np.float64)
        for name in ("50000us", "5000us", "500us", "50us"):
            if name not in prepared:
                continue
            term = prepared[name] * scales[name]
            if combine == "log":
                term = np.log10(np.maximum(term, 1.0))
            merged = merged + term
        source = np.full(ref_shape, -1, dtype=np.int32)
        solo = None
    elif solo is not None:
        merged = prepared[solo] * scales[solo]
        source = np.full(ref_shape, exposure_us[solo], dtype=np.int32)
    else:
        if "50000us" in prepared:
            merged = prepared["50000us"] * scales["50000us"]
            source = np.full(ref_shape, 50000, dtype=np.int32)
        else:
            merged = np.zeros(ref_shape, dtype=np.float64)
            source = np.zeros(ref_shape, dtype=np.int32)

        mask_5000 = keep_mask(prepared["5000us"], threshold) if "5000us" in prepared else mask_5000
        mask_500 = keep_mask(prepared["500us"], threshold) if "500us" in prepared else mask_500
        mask_50 = keep_mask(prepared["50us"], threshold) if "50us" in prepared else mask_50

        # Longer qualifying exposures are written first so a shorter exposure
        # overwrites the same pixel.
        if "5000us" in prepared:
            merged[mask_5000] = prepared["5000us"][mask_5000] * scales["5000us"]
            source[mask_5000] = 5000
        if "500us" in prepared:
            merged[mask_500] = prepared["500us"][mask_500] * scales["500us"]
            source[mask_500] = 500
        if "50us" in prepared:
            merged[mask_50] = prepared["50us"][mask_50] * scales["50us"]
            source[mask_50] = 50

    if combine == "overwrite" and blur:
        merged = gaussian_blur(merged, smooth_x, smooth_y, sigma)

    if combine in {"linear", "log"}:
        pixel_count = int(ref_shape[0] * ref_shape[1])
        used_50 = pixel_count if "50us" in prepared else 0
        used_500 = pixel_count if "500us" in prepared else 0
        used_5000 = pixel_count if "5000us" in prepared else 0
        used_50000 = pixel_count if "50000us" in prepared else 0
    else:
        used_50 = int(np.sum(source == 50))
        used_500 = int(np.sum(source == 500))
        used_5000 = int(np.sum(source == 5000))
        used_50000 = int(np.sum(source == 50000))

    counts = {
        "kept_50us": int(mask_50.sum()),
        "kept_500us": int(mask_500.sum()),
        "kept_5000us": int(mask_5000.sum()),
        "used_50us": int(used_50),
        "used_500us": int(used_500),
        "used_5000us": int(used_5000),
        "used_50000us": int(used_50000),
        "solo": solo or "",
        "combine": combine,
    }
    return MergeResult(
        image=merged,
        source=source,
        masks={"50us": mask_50, "500us": mask_500, "5000us": mask_5000},
        counts=counts,
    )
