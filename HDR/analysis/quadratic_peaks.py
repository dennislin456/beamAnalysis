"""Local polynomial refine for strip-sum lower-peak Y (sub-pixel).

Fits a polynomial to all samples inside ``window`` (no S-G / Gaussian
pre-filter). Degree follows the half-width:

- 左右各 1 點（window=3）→ 二次（degree 2）
- 左右各 2 點（window=5）→ 四次（degree 4）
- 左右各 k 點（window=2k+1）→ degree 2k

Peak Y is the maximum of the fitted polynomial inside the window.
Coordinates are scaled to about [-1, 1] before ``polyfit`` to avoid
ill-conditioned Vandermonde / RankWarning spam on larger windows.
"""

from __future__ import annotations

import warnings

import numpy as np

from analysis.peak_refine_common import (
    METHOD_QUADRATIC,
    RefinedLowerPeak,
    extract_window,
    finish_refine,
    normalize_window,
    parabolic_peak_offset,
    window_bounds,
)

QuadraticLowerPeak = RefinedLowerPeak


def poly_degree_for_window(point_count: int) -> int:
    """Degree = 2k when there are k samples on each side of the center."""
    n = int(point_count)
    if n < 3:
        return 0
    side = (n - 1) // 2
    degree = 2 * side
    # Need at least degree+1 samples; with odd n=2k+1, degree=2k fits exactly.
    return int(min(degree, n - 1))


def _normalize_x(xs: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map local indices to roughly [-1, 1] for stable polynomial fitting."""
    x = np.asarray(xs, dtype=np.float64)
    x_mid = float(np.mean(x)) if x.size else 0.0
    x_scale = float(np.max(np.abs(x - x_mid))) if x.size else 1.0
    if not np.isfinite(x_scale) or x_scale < 1e-12:
        x_scale = 1.0
    return (x - x_mid) / x_scale, x_mid, x_scale


def _to_normalized(local_x: float, x_mid: float, x_scale: float) -> float:
    return (float(local_x) - float(x_mid)) / float(x_scale)


def _from_normalized(norm_x: float, x_mid: float, x_scale: float) -> float:
    return float(norm_x) * float(x_scale) + float(x_mid)


def eval_poly_local(
    coeff,
    local_x: np.ndarray | float,
    *,
    x_mid: float,
    x_scale: float,
) -> np.ndarray | float:
    """Evaluate normalized-basis poly coefficients at local window indices."""
    x = np.asarray(local_x, dtype=np.float64)
    yn = np.polyval(np.asarray(coeff, dtype=float), (x - float(x_mid)) / float(x_scale))
    if np.ndim(local_x) == 0:
        return float(yn)
    return np.asarray(yn, dtype=np.float64)


def _peak_on_polynomial(coeff: np.ndarray, x_lo: float, x_hi: float) -> float | None:
    """Return local-x of the highest value of poly(coeff) on ``[x_lo, x_hi]``."""
    if coeff.size < 1 or not np.all(np.isfinite(coeff)):
        return None
    candidates = [float(x_lo), float(x_hi)]
    if coeff.size >= 2:
        deriv = np.polyder(coeff)
        if deriv.size >= 1 and np.any(np.abs(deriv) > 1e-18):
            try:
                roots = np.roots(deriv)
            except np.linalg.LinAlgError:
                roots = np.array([], dtype=np.complex128)
            for root in roots:
                if abs(float(np.imag(root))) > 1e-8:
                    continue
                x = float(np.real(root))
                if x_lo - 1e-9 <= x <= x_hi + 1e-9:
                    candidates.append(float(np.clip(x, x_lo, x_hi)))
    best_x = None
    best_y = -np.inf
    for x in candidates:
        y = float(np.polyval(coeff, x))
        if not np.isfinite(y):
            continue
        if y > best_y:
            best_y = y
            best_x = float(x)
    return best_x


def refine_lower_peak(
    profile: np.ndarray,
    crest_y: float,
    *,
    window: int = 7,
) -> RefinedLowerPeak | None:
    """Fit degree-2k poly in the window (k = half-width) and return sub-pixel Y."""
    extracted = extract_window(profile, crest_y, window)
    if extracted is None:
        return None
    segment, lo, hi, seg_win = extracted
    n = int(segment.size)
    degree = poly_degree_for_window(n)
    if degree < 2:
        return None
    xs = np.arange(n, dtype=np.float64)
    ys = np.asarray(segment, dtype=np.float64)
    profile_len = int(np.asarray(profile).size)
    xs_n, x_mid, x_scale = _normalize_x(xs)
    try:
        with warnings.catch_warnings():
            # NumPy 1.x: numpy.RankWarning; 2.x: numpy.exceptions.RankWarning
            rank_warning = getattr(np, "RankWarning", None)
            if rank_warning is None:
                try:
                    from numpy.exceptions import RankWarning as rank_warning
                except Exception:
                    rank_warning = UserWarning
            warnings.simplefilter("ignore", rank_warning)
            warnings.filterwarnings(
                "ignore",
                message="Polyfit may be poorly conditioned",
            )
            coeff = np.polyfit(xs_n, ys, degree)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return finish_refine(
            ys,
            crest_y=crest_y,
            lo=lo,
            hi=hi,
            win=seg_win,
            profile_len=profile_len,
            method=METHOD_QUADRATIC,
        )
    fitted = np.asarray(np.polyval(coeff, xs_n), dtype=np.float64)
    peak_norm = _peak_on_polynomial(coeff, float(xs_n[0]), float(xs_n[-1]))
    if peak_norm is None:
        return finish_refine(
            fitted if np.all(np.isfinite(fitted)) else ys,
            crest_y=crest_y,
            lo=lo,
            hi=hi,
            win=seg_win,
            profile_len=profile_len,
            method=METHOD_QUADRATIC,
        )
    peak_local = _from_normalized(peak_norm, x_mid, x_scale)
    peak_local = float(np.clip(peak_local, 0.0, float(n - 1)))
    y = float(np.clip(float(lo) + float(peak_local), 0.0, float(profile_len - 1)))
    peak_value = float(np.polyval(coeff, _to_normalized(peak_local, x_mid, x_scale)))
    return RefinedLowerPeak(
        y=y,
        lo=int(lo),
        hi=int(hi),
        smooth=fitted if np.all(np.isfinite(fitted)) else ys,
        method=METHOD_QUADRATIC,
        poly_coeff=tuple(float(v) for v in coeff),
        peak_value=peak_value,
        poly_x_mid=float(x_mid),
        poly_x_scale=float(x_scale),
    )


__all__ = [
    "METHOD_QUADRATIC",
    "QuadraticLowerPeak",
    "RefinedLowerPeak",
    "eval_poly_local",
    "normalize_window",
    "parabolic_peak_offset",
    "poly_degree_for_window",
    "refine_lower_peak",
    "window_bounds",
]
