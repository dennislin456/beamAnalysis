"""Shared strip geometry and valley finding for all dual-spot modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def slant_tan(upper, lower) -> float:
    """Horizontal shift per row from the upper spot toward the lower spot."""
    if upper is None or lower is None:
        return 0.0
    dy = float(lower[1]) - float(upper[1])
    if abs(dy) < 1e-6:
        return 0.0
    return (float(lower[0]) - float(upper[0])) / dy


def sample_x(row: np.ndarray, x: float) -> float:
    """Linear interpolation of one row at a continuous X."""
    width = int(row.shape[0])
    if width < 1:
        return 0.0
    if width == 1:
        value = float(row[0])
        return 0.0 if not np.isfinite(value) else value
    if x <= 0.0:
        value = float(row[0])
        return 0.0 if not np.isfinite(value) else value
    if x >= width - 1:
        value = float(row[-1])
        return 0.0 if not np.isfinite(value) else value
    left = int(np.floor(x))
    right = left + 1
    t = float(x - left)
    v0 = float(row[left])
    v1 = float(row[right])
    if not np.isfinite(v0):
        v0 = 0.0
    if not np.isfinite(v1):
        v1 = 0.0
    return v0 * (1.0 - t) + v1 * t


def integrate_row_x(row: np.ndarray, x_lo: float, x_hi: float) -> float:
    """Integrate one row from x_lo to x_hi with linear interpolation between pixels."""
    values = np.asarray(row, dtype=np.float64)
    width = int(values.shape[0])
    if width < 1 or not np.isfinite(x_lo) or not np.isfinite(x_hi):
        return 0.0
    lo = float(min(x_lo, x_hi))
    hi = float(max(x_lo, x_hi))
    lo = max(lo, 0.0)
    hi = min(hi, float(width - 1))
    if hi <= lo:
        return 0.0
    total = 0.0
    x = lo
    while x < hi - 1e-12:
        left = int(np.floor(x))
        next_knot = float(left + 1)
        segment_hi = min(hi, next_knot)
        if segment_hi <= x:
            break
        v0 = sample_x(values, x)
        v1 = sample_x(values, segment_hi)
        total += 0.5 * (v0 + v1) * (segment_hi - x)
        x = segment_hi
    return float(total)


def slanted_line_profile(image: np.ndarray, upper, lower) -> np.ndarray:
    """Sample each row on the upper-to-lower line with linear interpolation in X."""
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2 or data.size == 0 or upper is None:
        return np.zeros(0, dtype=np.float64)
    height = int(data.shape[0])
    tan = slant_tan(upper, lower)
    x_ref = float(upper[0])
    y_ref = float(upper[1])
    out = np.zeros(height, dtype=np.float64)
    for y in range(height):
        xc = x_ref + (float(y) - y_ref) * tan
        out[y] = sample_x(data[y], xc)
    return out


def slanted_strip_sum(image: np.ndarray, upper, lower, width: int) -> np.ndarray:
    """Sum each image row inside a strip centered on the upper-to-lower line."""
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2 or data.size == 0 or upper is None:
        return np.zeros(0, dtype=np.float64)
    height, _img_w = data.shape
    strip_w = max(1, int(width))
    half = strip_w / 2.0
    tan = slant_tan(upper, lower)
    x_ref = float(upper[0])
    y_ref = float(upper[1])
    out = np.zeros(height, dtype=np.float64)
    for y in range(height):
        xc = x_ref + (float(y) - y_ref) * tan
        out[y] = integrate_row_x(data[y], xc - half, xc + half)
    return out


def slanted_strip_mask(
    shape: tuple[int, ...],
    upper,
    lower,
    width: int,
) -> np.ndarray:
    """Boolean mask of a strip centered on the upper-to-lower line (full image)."""
    height = int(shape[0]) if len(shape) >= 1 else 0
    img_w = int(shape[1]) if len(shape) >= 2 else 0
    mask = np.zeros((height, img_w), dtype=bool)
    if upper is None or height < 1 or img_w < 1:
        return mask
    strip_w = max(1, int(width))
    half = strip_w / 2.0
    tan = slant_tan(upper, lower)
    x_ref = float(upper[0])
    y_ref = float(upper[1])
    xs = np.arange(img_w, dtype=np.float64)
    for y in range(height):
        xc = x_ref + (float(y) - y_ref) * tan
        mask[y] = (xs >= xc - half) & (xs <= xc + half)
    return mask


def slanted_strip_outline(
    upper,
    lower,
    width: int,
    height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed polygon for the slanted strip covering the full image height."""
    if upper is None or height < 1 or image_width < 1:
        return np.zeros(0), np.zeros(0)
    strip_w = max(1, int(width))
    half = strip_w / 2.0
    tan = slant_tan(upper, lower)
    x_ref = float(upper[0])
    y_ref = float(upper[1])
    y_top = 0.0
    y_bot = float(height)
    x_top = x_ref + (y_top - y_ref) * tan
    x_bot = x_ref + (y_bot - y_ref) * tan
    xs = np.array(
        [x_top - half, x_top + half, x_bot + half, x_bot - half, x_top - half],
        dtype=float,
    )
    ys = np.array([y_top, y_top, y_bot, y_bot, y_top], dtype=float)
    return xs, ys


def empty_result() -> dict:
    return {
        "bounds": None,
        "profile": np.array([]),
        "smooth": np.array([]),
        "mask": None,
        "upper": None,
        "lower": None,
        "raw_sum": np.array([]),
    }


def clip_roi(shape, x, y, width, height):
    height_img, width_img = shape
    try:
        x0 = int(round(float(x)))
        y0 = int(round(float(y)))
        rw = int(round(float(width)))
        rh = int(round(float(height)))
    except (TypeError, ValueError):
        return None
    if rw <= 0 or rh <= 0 or width_img <= 0 or height_img <= 0:
        return None
    x0 = int(np.clip(x0, 0, width_img - 1))
    y0 = int(np.clip(y0, 0, height_img - 1))
    x1 = int(np.clip(x0 + rw, x0 + 1, width_img))
    y1 = int(np.clip(y0 + rh, y0 + 1, height_img))
    return x0, y0, x1, y1


def threshold_upper(work: np.ndarray, thresh_percent: float, y_limit: float | None = None):
    """Core of the strongest pixel, and the mask used for it."""
    from scipy.ndimage import label as ndi_label

    signal = np.clip(np.asarray(work, dtype=np.float64), 0.0, None)
    if y_limit is not None and signal.ndim == 2:
        cut = int(np.clip(np.floor(float(y_limit)), 0, signal.shape[0]))
        signal[cut:, :] = 0.0
    if signal.size == 0 or not np.isfinite(float(np.max(signal))) or float(np.max(signal)) <= 0:
        return None, np.zeros(work.shape, dtype=bool)
    peak_index = np.unravel_index(int(np.argmax(signal)), signal.shape)
    peak = float(signal[peak_index])
    frac = float(np.clip(thresh_percent, 1.0, 99.0)) / 100.0
    raw_mask = signal >= peak * frac
    labeled, count = ndi_label(raw_mask)
    if count < 1:
        return None, np.zeros(work.shape, dtype=bool)
    peak_label = int(labeled[peak_index])
    if peak_label == 0:
        return None, np.zeros(work.shape, dtype=bool)
    best = labeled == peak_label
    weights = signal[best]
    total = float(np.sum(weights))
    ys, xs = np.where(best)
    if total <= 1e-12:
        center = (float(np.mean(xs)), float(np.mean(ys)))
    else:
        center = (
            float(np.sum(xs * weights) / total),
            float(np.sum(ys * weights) / total),
        )
    return center, best


def find_upper_spot(
    image: np.ndarray,
    thresh_percent: float = 50.0,
) -> tuple[tuple[float, float] | None, np.ndarray | None]:
    """Threshold centroid of the brightest core on the full image."""
    matrix = np.asarray(image, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return None, None
    work = np.nan_to_num(matrix, nan=0.0)
    upper, mask = threshold_upper(work, thresh_percent)
    if upper is None:
        return None, None
    return (float(upper[0]), float(upper[1])), mask


def other_peak_lobe(
    profile: np.ndarray,
    peak1: int,
    min_peak_distance: int = 50,
    valley_frac: float = 0.65,
    min_peak_ratio: float = 0.10,
):
    """Valley between the upper spot and the tallest separated crest below it.

    Valley depth is judged against the *upper* peak (not the lower crest), so a
    real lower lobe behind a deep gap still passes when the lower crest is weak
    or the gap floor sits above ``valley_frac × lower``. Mid-slope wiggles that
    never drop far below the upper peak are rejected. No smoothing.
    """
    profile = np.asarray(profile, dtype=np.float64)
    count = int(profile.size)
    if count < 5:
        return None
    peak1 = int(np.clip(peak1, 0, max(0, count - 1)))
    min_dist = max(1, int(min_peak_distance))
    ratio = max(0.0, float(min_peak_ratio))
    upper_v = float(profile[peak1])
    if not np.isfinite(upper_v) or upper_v <= 0.0:
        return None
    min_crest = ratio * upper_v
    # Valley must sit at least this far below the upper peak index.
    min_valley_gap = max(2, min_dist // 5)
    best = None
    best_height = -1.0
    for crest in range(peak1 + min_dist, count - 1):
        # Allow flat-topped crests (>= both neighbours); strict < missed broad lobes.
        if profile[crest] < profile[crest - 1] or profile[crest] < profile[crest + 1]:
            continue
        crest_v = float(profile[crest])
        if crest_v < min_crest:
            continue
        between = profile[peak1 : crest + 1]
        if between.size < 3:
            continue
        valley_i = peak1 + int(np.argmin(between))
        valley_v = float(profile[valley_i])
        if valley_i <= peak1 or valley_i >= crest:
            continue
        if valley_i < peak1 + min_valley_gap:
            continue
        # Separation from the upper lobe: valley must fall below valley_frac × upper.
        if valley_v > float(valley_frac) * upper_v:
            continue
        if crest_v - valley_v <= 0.0:
            continue
        if crest_v <= best_height:
            continue
        end = crest
        while end < count - 1 and float(profile[end + 1]) > valley_v:
            end += 1
        if end <= valley_i:
            continue
        best_height = crest_v
        best = (float(crest), float(valley_i), int(end))
    if best is not None:
        return best
    # Fallback: deepest valley after the upper peak, then tallest point beyond it.
    # Helps when the lower lobe has no strict single-pixel local maximum.
    search_lo = peak1 + min_valley_gap
    search_hi = count - 1
    if search_hi - search_lo < 2:
        return None
    valley_i = search_lo + int(np.argmin(profile[search_lo:search_hi]))
    valley_v = float(profile[valley_i])
    if valley_v > float(valley_frac) * upper_v:
        return None
    after = profile[valley_i + 1 : count]
    if after.size < 1:
        return None
    crest = valley_i + 1 + int(np.argmax(after))
    if crest < peak1 + min_dist:
        return None
    crest_v = float(profile[crest])
    if crest_v < min_crest or crest_v - valley_v <= 0.0:
        return None
    end = crest
    while end < count - 1 and float(profile[end + 1]) > valley_v:
        end += 1
    if end <= valley_i:
        return None
    return (float(crest), float(valley_i), int(end))


def lower_centroid(
    work: np.ndarray,
    valley: float,
    y_end: int,
    center_mode: str = "shape",
    thresh_percent: float = 50.0,
):
    """Center of the lower blob below the valley (shape or intensity)."""
    from scipy.ndimage import label as ndi_label

    height, width = work.shape
    if height < 3 or width < 1:
        return None, None
    cut = int(np.clip(np.floor(float(valley)), 0, height - 2))
    y_end = int(np.clip(max(int(y_end), cut + 1), cut + 1, height - 1))
    if y_end <= cut:
        return None, None
    signal = np.clip(np.asarray(work, dtype=np.float64), 0.0, None)
    signal[: cut + 1, :] = 0.0
    signal[y_end + 1 :, :] = 0.0
    peak = float(np.max(signal))
    if not np.isfinite(peak) or peak <= 0.0:
        return None, signal
    frac = float(np.clip(thresh_percent, 1.0, 99.0)) / 100.0
    raw_mask = signal >= peak * frac
    if not np.any(raw_mask):
        return None, signal
    peak_index = np.unravel_index(int(np.argmax(signal)), signal.shape)
    labeled, count = ndi_label(raw_mask)
    if count < 1:
        return None, signal
    peak_label = int(labeled[peak_index])
    if peak_label == 0:
        return None, signal
    support = labeled == peak_label
    ys, xs = np.where(support)
    if ys.size == 0:
        return None, signal
    if center_mode == "intensity":
        weights = signal[support]
        total = float(np.sum(weights))
        if total > 1e-12:
            return (
                float(np.sum(xs * weights) / total),
                float(np.sum(ys * weights) / total),
            ), signal
    return (float(np.mean(xs)), float(np.mean(ys))), signal


def highest_peak_above(profile: np.ndarray, valley: float) -> float | None:
    """Tallest local crest of the strip sum in rows strictly below ``valley``."""
    count = int(profile.size)
    end = int(np.clip(np.floor(float(valley)), 0, count))
    best_i = None
    best_v = -1.0
    for crest in range(1, end):
        if crest >= count - 1:
            break
        if profile[crest] < profile[crest - 1] or profile[crest] <= profile[crest + 1]:
            continue
        height = float(profile[crest])
        if height > best_v:
            best_v = height
            best_i = crest
    if best_i is None and end > 0:
        best_i = int(np.argmax(profile[:end]))
        if float(profile[best_i]) <= 0:
            return None
    if best_i is None:
        return None
    return float(best_i)


def row_peak_x(work: np.ndarray, row_y: float, fallback_x: float) -> float:
    """X of the brightest pixel on one strip row."""
    height, width = work.shape
    if height < 1 or width < 1:
        return float(fallback_x)
    row_index = int(np.clip(round(float(row_y)), 0, height - 1))
    row = np.asarray(work[row_index], dtype=np.float64)
    if row.size == 0 or not np.isfinite(row).any() or float(np.nanmax(row)) <= 0.0:
        return float(fallback_x)
    return float(np.nanargmax(row))


def strip_peak_center(work: np.ndarray, crest_y: float, fallback_x: float) -> tuple[float, float] | None:
    """Lower spot from the strip-sum crest. X is the brightest pixel on that row."""
    height = work.shape[0]
    if height < 1:
        return None
    row_index = int(np.clip(round(float(crest_y)), 0, height - 1))
    return row_peak_x(work, row_index, fallback_x), float(row_index)


@dataclass
class StripContext:
    """Prepared strip ROI shared by every center mode."""

    matrix: np.ndarray
    bounds: tuple[int, int, int, int]
    work: np.ndarray
    raw_sum: np.ndarray
    profile: np.ndarray
    seed_x: float
    seed_y: float
    upper_local: tuple[float, float]
    roi_mask: np.ndarray
    full_mask: np.ndarray
    lower_local: tuple[float, float, int] | None
    min_dist: int
    thresh_percent: float
    lower_thresh_percent: float
    use_log: bool


def prepare_strip(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    min_peak_distance: int = 50,
    min_peak_ratio: float = 0.10,
    use_log: bool = False,
    thresh_percent: float = 50.0,
    lower_thresh_percent: float | None = None,
    *,
    peak_seed: bool = False,
) -> StripContext | None:
    """Clip ROI, threshold upper seed, locate valley. Modes then resolve centers."""
    matrix = np.asarray(image, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return None
    bounds = clip_roi(matrix.shape, x, y, width, height)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    work = np.nan_to_num(matrix[y0:y1, x0:x1], nan=0.0)
    upper_thresh = float(thresh_percent)
    lower_thresh = (
        float(upper_thresh) if lower_thresh_percent is None else float(lower_thresh_percent)
    )
    upper_local, roi_mask = threshold_upper(work, upper_thresh)
    if upper_local is None:
        return None
    full_mask = np.zeros(matrix.shape, dtype=bool)
    full_mask[y0:y1, x0:x1] = roi_mask
    raw_sum = np.asarray(work.sum(axis=1), dtype=np.float64)
    profile = np.log10(np.maximum(raw_sum, 1.0)) if use_log else np.asarray(raw_sum, dtype=np.float64)
    seed_x = float(upper_local[0])
    seed_y = float(upper_local[1])
    min_dist = max(1, int(min_peak_distance))
    if peak_seed:
        strip_peak = highest_peak_above(raw_sum, float(raw_sum.size))
        peak1 = int(round(strip_peak if strip_peak is not None else seed_y))
    else:
        peak1 = int(round(seed_y))
    lower_local = other_peak_lobe(
        raw_sum,
        peak1,
        min_dist,
        min_peak_ratio=float(min_peak_ratio),
    )
    if lower_local is not None:
        limited, limited_mask = threshold_upper(work, upper_thresh, y_limit=lower_local[1])
        if limited is not None and not peak_seed:
            upper_local = limited
            seed_x = float(upper_local[0])
            seed_y = float(upper_local[1])
        if limited_mask is not None:
            roi_mask = limited_mask
            full_mask = np.zeros(matrix.shape, dtype=bool)
            full_mask[y0:y1, x0:x1] = roi_mask
    return StripContext(
        matrix=matrix,
        bounds=bounds,
        work=work,
        raw_sum=raw_sum,
        profile=profile,
        seed_x=seed_x,
        seed_y=seed_y,
        upper_local=upper_local,
        roi_mask=roi_mask,
        full_mask=full_mask,
        lower_local=lower_local,
        min_dist=min_dist,
        thresh_percent=float(upper_thresh),
        lower_thresh_percent=float(lower_thresh),
        use_log=bool(use_log),
    )


def pack_result(
    ctx: StripContext,
    upper: tuple[float, float] | None,
    lower: tuple[float, float] | None,
    profile: np.ndarray | None = None,
) -> dict:
    x0, y0, _x1, _y1 = ctx.bounds
    return {
        "bounds": ctx.bounds,
        "profile": ctx.profile if profile is None else profile,
        "smooth": ctx.profile if profile is None else profile,
        "mask": ctx.full_mask,
        "upper": upper,
        "lower": lower,
        "raw_sum": ctx.raw_sum,
        "origin": (float(x0), float(y0)),
    }
