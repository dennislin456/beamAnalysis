"""Dual spot finder.

The upper spot is the intensity centroid inside a signal threshold, using
the same background-subtracted mask as the semi-transparent threshold
overlay. The strip sum locates the valley below the upper spot. The center of that
valley is the boundary.     The lower spot is either the shape center or the intensity centroid of
the brightest connected region below that boundary.
"""

from __future__ import annotations

import numpy as np


def locate_dual_spots(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    smooth_win: int = 7,
    min_peak_distance: int = 12,
    use_log: bool = False,
    thresh_percent: float = 50.0,
    center_mode: str = "shape",
) -> dict:
    """Return strip profile and upper/lower centroids in full-image coordinates."""
    matrix = np.asarray(image, dtype=np.float64)
    empty = {
        "bounds": None,
        "profile": np.array([]),
        "smooth": np.array([]),
        "smooth_image": None,
        "mask": None,
        "upper": None,
        "lower": None,
    }
    if matrix.ndim != 2 or matrix.size == 0:
        return empty
    bounds = _clip_roi(matrix.shape, x, y, width, height)
    if bounds is None:
        return empty
    x0, y0, x1, y1 = bounds
    work = np.nan_to_num(matrix[y0:y1, x0:x1], nan=0.0)
    upper_local, roi_mask = _threshold_upper(work, thresh_percent)
    full_mask = np.zeros(matrix.shape, dtype=bool)
    if roi_mask is not None:
        full_mask[y0:y1, x0:x1] = roi_mask

    raw_sum = work.sum(axis=1)
    linear = np.clip(raw_sum - float(np.median(raw_sum)), 0.0, None)
    profile = np.log10(np.maximum(linear, 1.0)) if use_log else linear
    upper = None
    lower = None
    if upper_local is not None:
        peak1 = int(round(upper_local[1]))
        lower_local = _other_peak_lobe(linear, peak1)
        if lower_local is not None:
            limited, limited_mask = _threshold_upper(work, thresh_percent, y_limit=lower_local[1])
            if limited is not None:
                upper_local = limited
                roi_mask = limited_mask
                full_mask = np.zeros(matrix.shape, dtype=bool)
                full_mask[y0:y1, x0:x1] = roi_mask
        upper = (float(x0) + float(upper_local[0]), float(y0) + float(upper_local[1]))
        if lower_local is not None and lower_local[0] > upper_local[1] + 2.0:
            center, _lower_smooth = _lower_centroid(
                work,
                lower_local[1],
                work.shape[0] - 1,
                smooth_win,
                center_mode,
            )
            if center is not None and center[1] > lower_local[1]:
                lower = (float(x0) + center[0], float(y0) + center[1])
    smooth_image = _smooth_full_image(matrix, smooth_win)
    return {
        "bounds": bounds,
        "profile": profile,
        "smooth": profile,
        "smooth_image": smooth_image,
        "mask": full_mask,
        "upper": upper,
        "lower": lower,
    }


def _clip_roi(shape, x, y, width, height):
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


def _threshold_upper(work: np.ndarray, thresh_percent: float, y_limit: float | None = None):
    """Core of the strongest pixel, and the mask used for it.

    The threshold is a fraction of the brightest signal after border-background
    removal. Only the connected component that contains that peak is kept.
    Rows at and below y_limit are excluded, so the upper spot stays above the
    valley center.
    """
    from scipy.ndimage import label as ndi_label

    signal = np.clip(work - _border_background(work), 0.0, None)
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
    if best is None:
        return None, np.zeros(work.shape, dtype=bool)
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


def _border_background(work: np.ndarray) -> float:
    if work.size == 0:
        return 0.0
    height, width = work.shape
    if height < 2 or width < 2:
        return float(np.median(work))
    border = np.concatenate([
        work[0, :],
        work[-1, :],
        work[1:-1, 0],
        work[1:-1, -1],
    ])
    return float(np.median(border))


def _other_peak_lobe(profile: np.ndarray, peak1: int):
    """Valley between the upper spot and the tallest separated crest below it.

    The crest only identifies which valley is the gap. The returned valley is
    the midpoint of that trough. Returns (crest_y, valley_center, y_end).
    y_end is only the crest's far side; the lower spot itself continues below it.
    """
    count = int(profile.size)
    peak1 = int(np.clip(peak1, 0, max(0, count - 1)))
    if count < 8 or float(np.max(profile)) <= 0:
        return None
    best_height = -1.0
    best = None
    for crest in range(peak1 + 6, count - 1):
        if profile[crest] < profile[crest - 1] or profile[crest] <= profile[crest + 1]:
            continue
        crest_v = float(profile[crest])
        if crest_v <= 0:
            continue
        between = profile[peak1 : crest + 1]
        if between.size < 3:
            continue
        valley_i = peak1 + int(np.argmin(between))
        valley_v = float(profile[valley_i])
        if valley_i in (peak1, crest) or valley_v > 0.65 * crest_v:
            continue
        far_side = profile[crest + 1 :]
        if far_side.size < 2 or float(np.min(far_side)) > 0.75 * crest_v:
            continue
        if crest_v <= best_height:
            continue
        end = crest
        while end < count - 1 and float(profile[end + 1]) > valley_v:
            end += 1
        if end <= valley_i:
            continue
        best_height = crest_v
        best = (float(crest), _valley_center(profile, peak1, valley_i, crest), int(end))
    return best


def _valley_center(profile: np.ndarray, peak1: int, valley_i: int, crest: int) -> float:
    """Midpoint of the valley floor between the upper peak and the lower crest."""
    floor = float(profile[valley_i])
    rise = min(float(profile[peak1]), float(profile[crest])) - floor
    gate = floor + 0.15 * max(0.0, rise)
    left = int(valley_i)
    while left > int(peak1) and float(profile[left - 1]) <= gate:
        left -= 1
    right = int(valley_i)
    while right < int(crest) and float(profile[right + 1]) <= gate:
        right += 1
    return 0.5 * (float(left) + float(right))


def _smooth_full_image(image: np.ndarray, smooth_win: int) -> np.ndarray:
    """Circular average of the whole image, for display only."""
    from scipy.ndimage import convolve

    data = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0)
    kernel = _disk_kernel(smooth_win)
    if kernel.size <= 1:
        return data
    return convolve(data, kernel, mode="nearest")


def _disk_kernel(diameter: int) -> np.ndarray:
    """Circular mean kernel. Even diameters are rounded up to the next odd size."""
    diameter = max(1, int(diameter))
    if diameter % 2 == 0:
        diameter += 1
    radius = diameter // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float64)
    total = float(kernel.sum())
    if total <= 0:
        return np.ones((1, 1), dtype=np.float64)
    return kernel / total


def _lower_centroid(
    work: np.ndarray,
    valley: float,
    y_end: int,
    smooth_win: int = 1,
    center_mode: str = "shape",
):
    """Center of the lower blob. The window only rounds that blob.

    shape: every pixel in the circle has the same weight.
    intensity: weights are the smoothed intensities.
    """
    from scipy.ndimage import convolve
    from scipy.ndimage import label as ndi_label

    height, width = work.shape
    if height < 3 or width < 1:
        return None, None
    cut = int(np.clip(np.floor(float(valley)), 0, height - 2))
    y_end = int(np.clip(max(int(y_end), cut + 1), cut + 1, height - 1))
    if y_end <= cut:
        return None, None
    signal = np.clip(np.asarray(work, dtype=np.float64) - _border_background(work), 0.0, None)
    band = signal[cut + 1 : y_end + 1]
    edge_cols = max(1, min(2, width // 4))
    edge = np.concatenate([band[:, :edge_cols], band[:, -edge_cols:]], axis=None)
    signal = np.clip(signal - float(np.median(edge)), 0.0, None)
    signal[: cut + 1, :] = 0.0
    signal[y_end + 1 :, :] = 0.0
    labeled, count = ndi_label(signal > 0.0)
    if count < 1:
        return None, signal
    best_id = 0
    best_mass = 0.0
    for label_id in range(1, int(count) + 1):
        component = labeled == label_id
        mass = float(np.sum(signal[component]))
        if mass > best_mass:
            best_mass = mass
            best_id = label_id
    if best_id == 0:
        return None, signal
    support = labeled == best_id
    kernel = _disk_kernel(smooth_win)
    if kernel.size <= 1:
        smoothed = signal
    else:
        smoothed = np.zeros_like(signal)
        smoothed[cut + 1 : y_end + 1] = convolve(signal[cut + 1 : y_end + 1], kernel, mode="nearest")
    smoothed = np.where(support, smoothed, 0.0)
    ys, xs = np.where(support)
    if ys.size == 0:
        return None, smoothed
    if center_mode == "intensity":
        weights = smoothed[support]
        total = float(np.sum(weights))
        if total > 1e-12:
            return (
                float(np.sum(xs * weights) / total),
                float(np.sum(ys * weights) / total),
            ), smoothed
    return (float(np.mean(xs)), float(np.mean(ys))), smoothed


def _band_centroid_x(work: np.ndarray, y0: int, y1: int, thresh_percent: float) -> float:
    """Intensity centroid across X inside the lower lobe, at that lobe's own threshold."""
    height, width = work.shape
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(max(y1, y0 + 1), y0 + 1, height))
    band = np.asarray(work[y0:y1, :], dtype=np.float64)
    signal = np.clip(band - float(np.median(band)), 0.0, None)
    peak = float(np.max(signal)) if signal.size else 0.0
    if peak <= 1e-12:
        return float(width) / 2.0
    frac = float(np.clip(thresh_percent, 1.0, 99.0)) / 100.0
    mask = signal >= peak * frac
    if not np.any(mask):
        mask = signal >= peak * 0.5
    weights = signal[mask]
    total = float(np.sum(weights))
    columns = np.where(mask)[1].astype(np.float64)
    if total <= 1e-12:
        return float(np.mean(columns))
    return float(np.sum(columns * weights) / total)


def _width_at_frac(values: np.ndarray, peak_index: int, frac: float) -> float:
    count = len(values)
    if count < 3 or peak_index < 0 or peak_index >= count:
        return float(count)
    amplitude = float(values[peak_index])
    if amplitude <= 0:
        return float(count)
    level = float(np.clip(frac, 0.05, 0.95)) * amplitude
    left = int(peak_index)
    while left > 0 and float(values[left]) >= level:
        left -= 1
    right = int(peak_index)
    while right < count - 1 and float(values[right]) >= level:
        right += 1
    return float(max(1, right - left))


def _lobe_center(values: np.ndarray, peak_index: int, frac: float) -> float:
    """Intensity centroid of the lobe around one peak."""
    count = len(values)
    peak_index = int(np.clip(peak_index, 0, max(0, count - 1)))
    amplitude = float(values[peak_index]) if count else 0.0
    if amplitude <= 0:
        return float(peak_index)
    level = float(np.clip(frac, 0.15, 0.85)) * amplitude
    left = peak_index
    while left > 0 and float(values[left - 1]) >= level:
        left -= 1
    right = peak_index
    while right < count - 1 and float(values[right + 1]) >= level:
        right += 1
    weights = np.asarray(values[left : right + 1], dtype=np.float64)
    weights = np.where(weights >= level, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(peak_index)
    positions = np.arange(left, right + 1, dtype=np.float64)
    return float(np.sum(positions * weights) / total)


def _upper_lobe_center(smooth: np.ndarray) -> float | None:
    """Lock the bright upper spot on its clean upper side, so a lower shoulder cannot pull it down."""
    if smooth.size < 3 or float(np.max(smooth)) <= 0:
        return None
    margin = max(2, int(smooth.size) // 40)
    band = smooth[margin : len(smooth) - margin] if smooth.size > margin * 2 + 1 else smooth
    offset = margin if band is not smooth else 0
    peak_index = offset + int(np.argmax(band))
    amplitude = float(smooth[peak_index])
    level = 0.50 * amplitude
    left = peak_index
    while left > 0 and float(smooth[left - 1]) >= level and float(smooth[left - 1]) <= float(smooth[left]) * 1.05:
        left -= 1
    weights = np.asarray(smooth[left : peak_index + 1], dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(peak_index)
    positions = np.arange(left, peak_index + 1, dtype=np.float64)
    upper_side = float(np.sum(positions * weights) / total)
    return float(peak_index) + (float(peak_index) - upper_side)


def _noise_scale(values: np.ndarray) -> float:
    """Robust wiggle of the profile, ignoring the tall upper peak."""
    data = np.asarray(values, dtype=np.float64)
    if data.size < 5:
        return 0.0
    floor = data[data <= np.percentile(data, 60)]
    if floor.size < 3:
        floor = data
    median = float(np.median(floor))
    mad = float(np.median(np.abs(floor - median)))
    return 1.4826 * mad


def _connected_end(smooth: np.ndarray, start: int, floor: float) -> int:
    """End of the lobe still attached to the upper spot. A gap returns to the floor."""
    count = int(smooth.size)
    last = int(start)
    gap = 0
    for index in range(int(start), count):
        if float(smooth[index]) > floor:
            last = index + 1
            gap = 0
        else:
            gap += 1
            if gap >= 6:
                return max(int(start) + 4, last)
    return count


def _lower_lobe_center(
    smooth: np.ndarray,
    upper_y: float,
    min_sep: int,
    core_w: float,
) -> float | None:
    """First real lobe below the upper spot.

    A distant spot shows up as a valley then a rise. A close spot often only
    makes a shoulder on the falling side, so that case is the centroid of the
    light left over after removing a symmetric copy of the upper spot.
    """
    count = int(smooth.size)
    if count < 8:
        return None
    peak_index = int(np.clip(round(float(upper_y)), 0, count - 1))
    gap = max(int(min_sep), int(round(0.55 * max(6.0, float(core_w)))))
    start = int(np.clip(peak_index + gap, 0, count - 1))
    if count - start < 6:
        return None
    primary_amp = max(float(smooth[peak_index]), 1e-12)
    noise = _noise_scale(smooth[start:])
    min_rise = max(2.0 * noise, 0.002 * primary_amp)

    valley_center = _valley_then_crest(smooth, start, min_rise, primary_amp)
    shoulder_center = _shoulder_after_upper(smooth, peak_index, start, min_rise, primary_amp)
    core = max(6.0, float(core_w))
    shoulder_is_core_wing = (
        shoulder_center is not None and (float(shoulder_center) - peak_index) <= core
    )
    if valley_center is not None and (shoulder_center is None or shoulder_is_core_wing):
        if shoulder_center is None or float(valley_center) >= peak_index + core:
            return float(valley_center)
    if shoulder_center is not None:
        return float(shoulder_center)
    return None if valley_center is None else float(valley_center)


def _valley_then_crest(
    smooth: np.ndarray,
    start: int,
    min_rise: float,
    primary_amp: float,
) -> float | None:
    count = int(smooth.size)
    valley = None
    for index in range(max(start, 1), count - 2):
        if not (smooth[index] <= smooth[index - 1] and smooth[index] <= smooth[index + 1]):
            continue
        left_max = float(np.max(smooth[start : index + 1]))
        depth = left_max - float(smooth[index])
        if depth < max(min_rise, 0.12 * left_max):
            continue
        crest = index + 1
        while crest < count - 1 and smooth[crest + 1] >= smooth[crest]:
            crest += 1
        rise = float(smooth[crest] - smooth[index])
        if rise < min_rise or (crest - index) < 3:
            continue
        if float(smooth[crest]) < 0.002 * primary_amp:
            continue
        span = max(8, crest - index)
        tail_end = min(count, crest + span + 1)
        drop = float(smooth[crest] - np.min(smooth[crest:tail_end]))
        if drop < 0.25 * rise:
            continue
        valley = int(index)
        break
    if valley is None:
        return None
    crest = valley + 1
    while crest < count - 1 and smooth[crest + 1] >= smooth[crest]:
        crest += 1
    rise = float(smooth[crest] - smooth[valley])
    end = crest
    while end < count - 1 and float(smooth[end]) >= float(smooth[valley]) + 0.5 * rise:
        end += 1
    return _lobe_center_range(smooth, valley + 1, end + 1, lobe_frac=0.40)


def _shoulder_after_upper(
    smooth: np.ndarray,
    peak_index: int,
    start: int,
    min_rise: float,
    primary_amp: float,
) -> float | None:
    """Extra light below a symmetric copy of the upper spot."""
    count = int(smooth.size)
    half = 0
    level = 0.35 * primary_amp
    while peak_index - half > 0 and float(smooth[peak_index - half]) >= level:
        if half > 0 and float(smooth[peak_index - half]) > float(smooth[peak_index - half + 1]) * 1.08:
            break
        half += 1
        if half > count // 3:
            break
    half = max(4, half)
    model = np.zeros(count, dtype=np.float64)
    for offset in range(half + 1):
        sample = float(smooth[peak_index - offset]) if peak_index - offset >= 0 else 0.0
        if peak_index + offset < count:
            model[peak_index + offset] = sample
    residual = np.clip(smooth - model, 0.0, None)
    if start >= count - 3:
        return None
    end = _connected_end(smooth, start, max(min_rise, 0.015 * primary_amp))
    segment = residual[start:end]
    height = float(np.max(segment)) if segment.size else 0.0
    if height < min_rise or height < 0.004 * primary_amp:
        return None
    level = 0.25 * height
    weights = np.where(segment >= level, segment, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return None
    positions = np.arange(start, start + len(segment), dtype=np.float64)
    return float(np.sum(positions * weights) / total)


def _lobe_center_range(values: np.ndarray, y0: int, y1: int, lobe_frac: float) -> float | None:
    if y1 - y0 < 3:
        return None
    segment = np.asarray(values[y0:y1], dtype=np.float64)
    if segment.size < 3 or float(np.max(segment)) <= 0:
        return None
    peak_index = int(np.argmax(segment))
    return float(y0) + _lobe_center(segment, peak_index, lobe_frac)


def _core_x(
    work: np.ndarray,
    center_y: float,
    half_band: float,
    core_frac: float,
    y_min: float = 0.0,
) -> float:
    """Bright-core centroid across X inside a Y band. Ignores the dim halo."""
    height, width = work.shape
    half = max(3, int(round(float(half_band))))
    y0 = int(np.clip(int(np.floor(center_y)) - half, 0, height - 1))
    y0 = max(y0, int(np.clip(int(np.floor(y_min)), 0, height - 1)))
    y1 = int(np.clip(int(np.ceil(center_y)) + half + 1, y0 + 1, height))
    band = work[y0:y1, :]
    weights = np.clip(band - float(np.median(band)), 0.0, None)
    peak = float(np.max(weights)) if weights.size else 0.0
    if peak <= 1e-12:
        return float(width) / 2.0
    mask = weights >= peak * float(np.clip(core_frac, 0.2, 0.9))
    if not np.any(mask):
        return float(width) / 2.0
    columns = np.where(mask)[1].astype(np.float64)
    selected = weights[mask]
    total = float(np.sum(selected))
    if total <= 1e-12:
        return float(np.mean(columns))
    return float(np.sum(columns * selected) / total)
