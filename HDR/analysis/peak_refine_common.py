"""Shared helpers for local 1D lower-peak refine (S-G / Gaussian)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHOD_NONE = "none"
METHOD_SAVGOL = "savgol"
METHOD_GAUSSIAN = "gaussian"
METHOD_QUADRATIC = "quadratic"
METHODS = (METHOD_NONE, METHOD_SAVGOL, METHOD_GAUSSIAN, METHOD_QUADRATIC)


@dataclass(frozen=True)
class RefinedLowerPeak:
    """Refined lower-peak location and the filter window used."""

    y: float
    """Sub-pixel row index in the same coordinates as ``profile``."""

    lo: int
    """Inclusive start index of the window on ``profile``."""

    hi: int
    """Inclusive end index of the window on ``profile``."""

    smooth: np.ndarray
    """Smoothed values for ``profile[lo:hi+1]`` (length ``hi - lo + 1``)."""

    method: str
    """``savgol``, ``gaussian``, or ``quadratic``."""

    poly_coeff: tuple[float, ...] | None = None
    """Optional ``np.polyfit`` coefficients (highest degree first) for dense redraw."""

    peak_value: float | None = None
    """Intensity of the refined peak on the fitted/smoothed curve (linear scale)."""

    poly_x_mid: float | None = None
    """Local-index midpoint used to normalize poly ``x`` before fitting."""

    poly_x_scale: float | None = None
    """Local-index scale used to normalize poly ``x`` before fitting."""


def normalize_window(window: int, length: int) -> int | None:
    """Clamp to a valid odd window length that fits ``length``."""
    n = int(length)
    if n < 3:
        return None
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    max_odd = n if n % 2 == 1 else n - 1
    w = min(w, max_odd)
    if w < 3:
        return None
    return w


def window_bounds(center: float, length: int, window: int) -> tuple[int, int]:
    """Inclusive ``[lo, hi]`` of length up to ``window``, clipped to ``[0, length)``."""
    n = int(length)
    w = max(1, int(window))
    if n < 1:
        return 0, -1
    half = (w - 1) // 2
    c = int(np.clip(int(round(float(center))), 0, n - 1))
    lo = c - half
    hi = lo + w - 1
    if lo < 0:
        hi -= lo
        lo = 0
    if hi >= n:
        lo -= hi - (n - 1)
        hi = n - 1
        lo = max(0, lo)
    return int(lo), int(hi)


def parabolic_peak_offset(values: np.ndarray, index: int) -> float:
    """Sub-pixel peak index via three-point parabola; returns absolute index."""
    y = np.asarray(values, dtype=np.float64)
    i = int(index)
    if y.size < 3 or i <= 0 or i >= int(y.size) - 1:
        return float(i)
    left = float(y[i - 1])
    mid = float(y[i])
    right = float(y[i + 1])
    if not (np.isfinite(left) and np.isfinite(mid) and np.isfinite(right)):
        return float(i)
    denom = left - 2.0 * mid + right
    if abs(denom) < 1e-18:
        return float(i)
    delta = 0.5 * (left - right) / denom
    if not np.isfinite(delta):
        return float(i)
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(i) + delta


def pick_local_peak(smooth: np.ndarray, crest_y: float, lo: int, win: int) -> int:
    """Argmax on smoothed segment, preferring the raw crest when far away."""
    local = int(np.argmax(smooth))
    raw_i = int(np.clip(int(round(float(crest_y))) - lo, 0, int(smooth.size) - 1))
    if abs(local - raw_i) > max(1, int(win) // 2):
        local = raw_i
    return local


def extract_window(
    profile: np.ndarray,
    crest_y: float,
    window: int,
) -> tuple[np.ndarray, int, int, int] | None:
    """Return ``(segment, lo, hi, odd_window)`` around ``crest_y``, or None."""
    data = np.asarray(profile, dtype=np.float64)
    n = int(data.size)
    if n < 3 or not np.isfinite(float(crest_y)):
        return None
    win = normalize_window(window, n)
    if win is None:
        return None
    lo, hi = window_bounds(crest_y, n, win)
    if hi < lo:
        return None
    segment = np.nan_to_num(data[lo : hi + 1], nan=0.0)
    seg_win = normalize_window(win, int(segment.size))
    if seg_win is None:
        return None
    return segment, lo, hi, seg_win


def finish_refine(
    smooth: np.ndarray,
    *,
    crest_y: float,
    lo: int,
    hi: int,
    win: int,
    profile_len: int,
    method: str,
) -> RefinedLowerPeak | None:
    """Parabola-fit the local peak on ``smooth`` and pack the result."""
    if smooth.size < 1:
        return None
    local = pick_local_peak(smooth, crest_y, lo, win)
    peak = parabolic_peak_offset(smooth, local)
    y = float(np.clip(float(lo) + float(peak), 0.0, float(profile_len - 1)))
    local = float(y) - float(lo)
    peak_value = _parabola_sample(smooth, local)
    return RefinedLowerPeak(
        y=y,
        lo=int(lo),
        hi=int(hi),
        smooth=np.asarray(smooth, dtype=np.float64),
        method=str(method),
        peak_value=peak_value,
    )


def _parabola_sample(values: np.ndarray, local_x: float) -> float:
    """Sample a 3-point parabola around the nearest interior index at ``local_x``."""
    y = np.asarray(values, dtype=np.float64)
    n = int(y.size)
    if n < 1:
        return 0.0
    if n == 1 or not np.isfinite(float(local_x)):
        value = float(y[0])
        return 0.0 if not np.isfinite(value) else value
    i = int(np.clip(int(round(float(local_x))), 1, n - 2))
    left = float(y[i - 1])
    mid = float(y[i])
    right = float(y[i + 1])
    t = float(local_x) - float(i)
    value = mid + 0.5 * (right - left) * t + 0.5 * (left - 2.0 * mid + right) * t * t
    if not np.isfinite(value):
        value = float(np.interp(float(local_x), np.arange(n, dtype=float), y))
    return float(value)
