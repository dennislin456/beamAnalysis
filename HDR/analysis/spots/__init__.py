"""Dual-spot finders split by center mode.

Modes (one module each):
- shape: 形狀中心
- intensity: 強度質心
- peak: 條帶波峰
- line: 顯示用斜剖面
- slant_iter: 斜向迭代（測試）
"""

from __future__ import annotations

from analysis.spots import intensity, line, peak, shape, slant_iter
from analysis.spots.common import (
    empty_result,
    find_upper_spot,
    slant_tan,
    slanted_line_profile,
    slanted_strip_outline,
    slanted_strip_sum,
)

MODES = {
    shape.MODE_ID: shape,
    intensity.MODE_ID: intensity,
    peak.MODE_ID: peak,
    line.MODE_ID: line,
    slant_iter.MODE_ID: slant_iter,
}


def locate_dual_spots(
    image,
    x: int,
    y: int,
    width: int,
    height: int,
    min_peak_distance: int = 50,
    min_peak_ratio: float = 0.10,
    use_log: bool = False,
    thresh_percent: float = 50.0,
    lower_thresh_percent: float | None = None,
    center_mode: str = "shape",
) -> dict:
    """Dispatch to the module that implements ``center_mode``."""
    module = MODES.get(str(center_mode), shape)
    return module.locate(
        image,
        x,
        y,
        width,
        height,
        min_peak_distance=min_peak_distance,
        min_peak_ratio=min_peak_ratio,
        use_log=use_log,
        thresh_percent=thresh_percent,
        lower_thresh_percent=lower_thresh_percent,
    )


__all__ = [
    "MODES",
    "empty_result",
    "find_upper_spot",
    "locate_dual_spots",
    "slant_tan",
    "slanted_line_profile",
    "slanted_strip_outline",
    "slanted_strip_sum",
]
