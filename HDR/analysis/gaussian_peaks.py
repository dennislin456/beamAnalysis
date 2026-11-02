"""Local 1D Gaussian refine for strip-sum lower-peak Y (sub-pixel).

Used only by the 條帶波峰 path after raw peak / valley finding. Does not
touch 2D images. ``window`` selects the neighborhood; ``sigma`` is derived
from window as ``(window - 1) / 6`` (half-width ≈ 3σ), not tuned by hand.
"""

from __future__ import annotations

import numpy as np

from analysis.peak_refine_common import (
    METHOD_GAUSSIAN,
    RefinedLowerPeak,
    extract_window,
    finish_refine,
    normalize_window,
    parabolic_peak_offset,
    window_bounds,
)

GaussianLowerPeak = RefinedLowerPeak


def sigma_from_window(window: int) -> float:
    """Derive Gaussian sigma from odd window length (half-width ≈ 3σ)."""
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    return float(max((w - 1) / 6.0, 0.01))


def refine_lower_peak(
    profile: np.ndarray,
    crest_y: float,
    *,
    window: int = 7,
    sigma: float | None = None,
) -> RefinedLowerPeak | None:
    """Gaussian-smooth the window around ``crest_y`` and return a sub-pixel peak Y.

    ``sigma`` is optional; when omitted (or non-positive), it follows ``window``.
    """
    extracted = extract_window(profile, crest_y, window)
    if extracted is None:
        return None
    segment, lo, hi, seg_win = extracted
    if sigma is None or not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
        sig = sigma_from_window(seg_win)
    else:
        sig = float(sigma)
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        return None
    smooth = np.asarray(
        gaussian_filter1d(segment, sigma=sig, mode="nearest"),
        dtype=np.float64,
    )
    return finish_refine(
        smooth,
        crest_y=crest_y,
        lo=lo,
        hi=hi,
        win=seg_win,
        profile_len=int(np.asarray(profile).size),
        method=METHOD_GAUSSIAN,
    )


__all__ = [
    "METHOD_GAUSSIAN",
    "GaussianLowerPeak",
    "RefinedLowerPeak",
    "normalize_window",
    "parabolic_peak_offset",
    "refine_lower_peak",
    "sigma_from_window",
    "window_bounds",
]
