"""Display helpers used by the spot analysis window.

Copied from FourStepAdj ``image_canvas``. Blur and color-scale clipping
come from ``common``.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from common.blur import gaussian_blur
from common.widgets import clip_color_scale


def apply_abs_then_color_scale(
    data: np.ndarray,
    abs_on: bool,
    lo: float | None,
    hi: float | None,
    clip_on: bool,
    smooth: tuple[int, int, float] = (1, 1, 0.0),
) -> np.ndarray:
    """Absolute value, then Gaussian blur, then the color-scale thresholds."""
    out = np.asarray(data, dtype=float)
    if abs_on:
        out = np.abs(out)
    width_x, width_y, sigma = smooth
    if int(width_x) > 1 or int(width_y) > 1:
        out = gaussian_blur(out, int(width_x), int(width_y), float(sigma))
    if clip_on:
        out = clip_color_scale(out, lo, hi)
    return out


def fix_readout_height(label: QLabel, lines: int) -> None:
    """Keep the cursor readout tall enough for its longest message."""
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    label.setWordWrap(False)
    label.setFixedHeight(label.fontMetrics().lineSpacing() * int(lines) + 6)


def display_threshold(value: float, log_on: bool) -> float:
    """Map a linear threshold onto the displayed axis."""
    value = float(value)
    if log_on:
        return float(np.log10(max(value, 1.0)))
    return value


def log10_bound(value: float | None) -> float | None:
    """Show a linear color-scale bound as log10. Empty stays empty. <= 0 becomes 0."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(np.log10(value))


def linear_bound(value: float | None) -> float | None:
    """Turn a log10 color-scale bound back into a linear threshold."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return 1.0
    with np.errstate(over="ignore"):
        converted = float(10.0 ** value)
    if not np.isfinite(converted):
        return converted
    nearest = round(converted)
    if abs(converted - nearest) <= 1e-4 * max(1.0, abs(nearest)):
        converted = float(nearest)
    return converted


def build_bw_clip_colormap(base_cmap, vmin, vmax, black_floor=None, white_ceil=None):
    """Black at and below the floor, white at and above the ceiling, jet in between."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return base_cmap
    span = vmax - vmin
    t_lo = 0.0
    t_hi = 1.0
    if black_floor is not None and np.isfinite(black_floor):
        t_lo = float(np.clip((float(black_floor) - vmin) / span, 0.0, 1.0))
    if white_ceil is not None and np.isfinite(white_ceil):
        t_hi = float(np.clip((float(white_ceil) - vmin) / span, 0.0, 1.0))
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    if t_lo <= 0.0 and t_hi >= 1.0:
        return base_cmap
    if t_lo >= 1.0:
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (0, 0, 0)])
    if t_hi <= 0.0:
        return pg.ColorMap([0.0, 1.0], [(255, 255, 255), (255, 255, 255)])

    pos = []
    colors = []
    if t_lo > 0.0:
        pos.extend([0.0, max(t_lo - 1e-9, 0.0)])
        colors.extend([(0, 0, 0), (0, 0, 0)])
    mid_span = max(t_hi - t_lo, 1e-9)
    samples = np.linspace(0.0, 1.0, 48)
    mapped = base_cmap.map(samples, mode="byte")
    for index, sample in enumerate(samples):
        pos.append(t_lo + mid_span * float(sample))
        colors.append((int(mapped[index, 0]), int(mapped[index, 1]), int(mapped[index, 2])))
    if t_hi < 1.0:
        pos.extend([min(t_hi + 1e-9, 1.0), 1.0])
        colors.extend([(255, 255, 255), (255, 255, 255)])
    return pg.ColorMap(pos, colors)


# Colorbar triangles from bottom to top. Position 0 is the bottom, 1 is the top.
COLOR_STOPS = (
    ("黑", "black", (0, 0, 0), 0.06),
    ("藍", "blue", (0, 0, 255), 0.21),
    ("青", "cyan", (0, 255, 255), 0.35),
    ("綠", "green", (0, 255, 0), 0.50),
    ("黃", "yellow", (255, 255, 0), 0.65),
    ("紅", "red", (255, 0, 0), 0.79),
    ("白", "white", (255, 255, 255), 0.94),
)


def colormap_from_stops(positions: list[float]) -> pg.ColorMap:
    """Black at the bottom, jet through the middle, white at the top."""
    colors = [color for _zh, _en, color, _pos in COLOR_STOPS]
    return pg.ColorMap([float(pos) for pos in positions], colors)


def jet_bw_colormap() -> pg.ColorMap:
    """Black at the bottom, jet through the middle, white at the top."""
    return colormap_from_stops([pos for _zh, _en, _color, pos in COLOR_STOPS])
