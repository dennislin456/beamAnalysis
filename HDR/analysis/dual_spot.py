"""Dual spot finder used by the HDR analysis window.

Implementation is split by center mode under ``analysis.spots``:
shape / intensity / peak / line. This module keeps the previous public API.
"""

from __future__ import annotations

from analysis.spots import (
    find_upper_spot,
    locate_dual_spots,
    slant_tan,
    slanted_line_profile,
    slanted_strip_outline,
    slanted_strip_sum,
)

__all__ = [
    "find_upper_spot",
    "locate_dual_spots",
    "slant_tan",
    "slanted_line_profile",
    "slanted_strip_outline",
    "slanted_strip_sum",
]
