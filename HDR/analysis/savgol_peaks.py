"""Local Savitzky–Golay refine for strip-sum lower-peak Y (sub-pixel).

Used only by the 條帶波峰 path after raw peak / valley finding. Does not
touch 2D images.
"""

from __future__ import annotations

import numpy as np

from analysis.peak_refine_common import (
    METHOD_SAVGOL,
    RefinedLowerPeak,
    extract_window,
    finish_refine,
    normalize_window,
    parabolic_peak_offset,
    window_bounds,
)

# Re-exports for older imports.
SavgolLowerPeak = RefinedLowerPeak


def normalize_savgol_params(
    window: int,
    polyorder: int,
    length: int,
) -> tuple[int, int] | None:
    """Clamp to a valid odd window and polyorder for ``scipy.signal.savgol_filter``."""
    w = normalize_window(window, length)
    if w is None:
        return None
    order = int(polyorder)
    order = max(1, min(order, w - 1))
    return w, order


def refine_lower_peak(
    profile: np.ndarray,
    crest_y: float,
    *,
    window: int = 7,
    polyorder: int = 2,
) -> RefinedLowerPeak | None:
    """S-G-smooth the window around ``crest_y`` and return a sub-pixel peak Y."""
    extracted = extract_window(profile, crest_y, window)
    if extracted is None:
        return None
    segment, lo, hi, seg_win = extracted
    seg_params = normalize_savgol_params(seg_win, polyorder, int(segment.size))
    if seg_params is None:
        return None
    seg_win, seg_order = seg_params
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return None
    smooth = np.asarray(
        savgol_filter(
            segment,
            window_length=seg_win,
            polyorder=seg_order,
            mode="interp",
        ),
        dtype=np.float64,
    )
    return finish_refine(
        smooth,
        crest_y=crest_y,
        lo=lo,
        hi=hi,
        win=seg_win,
        profile_len=int(np.asarray(profile).size),
        method=METHOD_SAVGOL,
    )


__all__ = [
    "METHOD_SAVGOL",
    "RefinedLowerPeak",
    "SavgolLowerPeak",
    "normalize_savgol_params",
    "normalize_window",
    "parabolic_peak_offset",
    "refine_lower_peak",
    "window_bounds",
]
