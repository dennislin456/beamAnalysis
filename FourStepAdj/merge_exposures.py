"""Merge four Basler exposures onto the 50000 us intensity scale.

A pixel is taken from a shorter exposure when that exposure, scaled by 10,
would already reach the threshold (default 4095). That pixel is then scaled
to the 50000 us equivalent:

    50 us    -> value * 1000
    500 us   -> value * 100
    5000 us  -> value * 10
    50000 us -> original value

If several exposures qualify at the same pixel, the shorter one is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_FULL_SCALE = 4000.0
NEXT_EXPOSURE_FACTOR = 10.0

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
    counts: dict[str, int]


def keep_mask(image: np.ndarray, threshold: float = CAMERA_FULL_SCALE) -> np.ndarray:
    """Pixels whose value * 10 is already at or above the threshold."""
    values = np.asarray(image, dtype=np.float64)
    return values * NEXT_EXPOSURE_FACTOR >= float(threshold)


def merge_four_exposures(
    img_50us: np.ndarray | None,
    img_500us: np.ndarray | None,
    img_5000us: np.ndarray | None,
    img_50000us: np.ndarray | None,
    threshold: float = CAMERA_FULL_SCALE,
) -> MergeResult:
    raw = {
        "50us": img_50us,
        "500us": img_500us,
        "5000us": img_5000us,
        "50000us": img_50000us,
    }
    images = {name: np.asarray(img) for name, img in raw.items() if img is not None}
    if not images:
        raise ValueError("這一筆沒有任何曝光圖")
    shapes = {name: arr.shape for name, arr in images.items()}
    if any(arr.ndim != 2 for arr in images.values()):
        raise ValueError(f"每張圖都必須是二維陣列，目前形狀為 {shapes}")
    ref_shape = next(iter(images.values())).shape
    if any(arr.shape != ref_shape for arr in images.values()):
        raise ValueError(f"曝光圖尺寸必須相同，目前形狀為 {shapes}")

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"門檻必須是大於等於 0 的數字，目前為 {threshold}")
    threshold = float(threshold)

    if "50000us" in images:
        merged = images["50000us"].astype(np.float64, copy=True)
        source = np.full(ref_shape, 50000, dtype=np.int32)
    else:
        merged = np.zeros(ref_shape, dtype=np.float64)
        source = np.zeros(ref_shape, dtype=np.int32)

    mask_5000 = keep_mask(images["5000us"], threshold) if "5000us" in images else np.zeros(ref_shape, dtype=bool)
    mask_500 = keep_mask(images["500us"], threshold) if "500us" in images else np.zeros(ref_shape, dtype=bool)
    mask_50 = keep_mask(images["50us"], threshold) if "50us" in images else np.zeros(ref_shape, dtype=bool)

    # Longer qualifying exposures are written first so a shorter exposure
    # overwrites the same pixel.
    if "5000us" in images:
        merged[mask_5000] = images["5000us"].astype(np.float64)[mask_5000] * 10.0
        source[mask_5000] = 5000
    if "500us" in images:
        merged[mask_500] = images["500us"].astype(np.float64)[mask_500] * 100.0
        source[mask_500] = 500
    if "50us" in images:
        merged[mask_50] = images["50us"].astype(np.float64)[mask_50] * 1000.0
        source[mask_50] = 50

    used_50 = source == 50
    used_500 = source == 500
    used_5000 = source == 5000
    used_50000 = source == 50000

    counts = {
        "kept_50us": int(mask_50.sum()),
        "kept_500us": int(mask_500.sum()),
        "kept_5000us": int(mask_5000.sum()),
        "used_50us": int(used_50.sum()),
        "used_500us": int(used_500.sum()),
        "used_5000us": int(used_5000.sum()),
        "used_50000us": int(used_50000.sum()),
    }
    return MergeResult(
        image=merged,
        source=source,
        masks={"50us": mask_50, "500us": mask_500, "5000us": mask_5000},
        counts=counts,
    )
