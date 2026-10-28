"""斜向迭代（測試） dual-spot mode.

DOC_TITLE: 斜向迭代（測試）
DOC_SUMMARY: 先用直條帶強度質心抓兩點，再沿連線做斜條帶、重找波谷與質心。直條帶已包住兩團、ΔX 小時結果幾乎不變；下光斑被直條帶切邊時才會明顯移動。若第一次沒抓到下光斑，會先用加寬直條帶找種子再斜向重抓。

Pipeline after the shared strip / valley step:
1. 第一次：與強度質心相同（垂直條帶）。
2. 若只有上光斑：用加寬直條帶暫找下光斑種子（仍可能失敗）。
3. 若上下都有：沿兩點連線做斜條帶加總，重找下峰／波谷；在斜條帶遮罩內重算強度質心（寬度用使用者設定的「寬」）。
"""

from __future__ import annotations

import numpy as np

from analysis.spots import intensity
from analysis.spots.common import (
    StripContext,
    empty_result,
    lower_centroid,
    other_peak_lobe,
    pack_result,
    prepare_strip,
    slanted_strip_mask,
    slanted_strip_sum,
    threshold_upper,
)


MODE_ID = "slant_iter"
MODE_LABEL = "斜向迭代（測試）"


def locate(
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
) -> dict:
    """Vertical intensity pass, then optional slanted-strip second pass."""
    ctx = prepare_strip(
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
        peak_seed=False,
    )
    if ctx is None:
        return empty_result()
    first = intensity.resolve(ctx)
    upper = first.get("upper")
    lower = first.get("lower")
    if upper is None:
        return first
    if lower is None:
        lower = _wide_seed_lower(
            ctx,
            upper,
            strip_width=int(width),
            min_peak_ratio=float(min_peak_ratio),
        )
    if lower is None:
        return first
    return _second_pass(
        ctx,
        upper,
        lower,
        strip_width=int(width),
        min_peak_ratio=float(min_peak_ratio),
    )


def _wide_seed_lower(
    ctx: StripContext,
    upper: tuple[float, float],
    *,
    strip_width: int,
    min_peak_ratio: float,
) -> tuple[float, float] | None:
    """If the user strip missed the lower blob, try a wider vertical window once."""
    matrix = np.nan_to_num(np.asarray(ctx.matrix, dtype=np.float64), nan=0.0)
    height, img_w = matrix.shape
    wide = max(int(strip_width) * 3, int(strip_width) + 40)
    half = wide / 2.0
    cx = float(upper[0])
    x0 = int(np.clip(np.floor(cx - half), 0, max(0, img_w - 1)))
    x1 = int(np.clip(np.ceil(cx + half), x0 + 1, img_w))
    work = matrix[:, x0:x1]
    raw_sum = np.asarray(work.sum(axis=1), dtype=np.float64)
    peak1 = int(np.clip(round(float(upper[1])), 0, max(0, raw_sum.size - 1)))
    lobe = other_peak_lobe(
        raw_sum,
        peak1,
        ctx.min_dist,
        min_peak_ratio=min_peak_ratio,
    )
    if lobe is None:
        return None
    center, _ = lower_centroid(
        work,
        float(lobe[1]),
        work.shape[0] - 1,
        "intensity",
        ctx.lower_thresh_percent,
    )
    if center is None or float(center[1]) <= float(upper[1]) + float(ctx.min_dist):
        return None
    return (float(x0) + float(center[0]), float(center[1]))


def _second_pass(
    ctx: StripContext,
    upper0: tuple[float, float],
    lower0: tuple[float, float],
    *,
    strip_width: int,
    min_peak_ratio: float,
) -> dict:
    """Re-find valley and intensity centroids inside the slanted strip on the full image."""
    matrix = np.nan_to_num(np.asarray(ctx.matrix, dtype=np.float64), nan=0.0)
    strip_w = max(1, int(strip_width))
    mask = slanted_strip_mask(matrix.shape, upper0, lower0, strip_w)
    if not np.any(mask):
        return pack_result(ctx, upper0, lower0)

    slant_sum = slanted_strip_sum(matrix, upper0, lower0, strip_w)
    peak1 = int(np.clip(round(float(upper0[1])), 0, max(0, slant_sum.size - 1)))
    lobe = other_peak_lobe(
        slant_sum,
        peak1,
        ctx.min_dist,
        min_peak_ratio=min_peak_ratio,
    )

    masked = np.where(mask, np.clip(matrix, 0.0, None), 0.0)
    if lobe is not None:
        upper_c, upper_mask = threshold_upper(
            masked, ctx.thresh_percent, y_limit=float(lobe[1])
        )
        lower_c, _ = lower_centroid(
            masked,
            float(lobe[1]),
            masked.shape[0] - 1,
            "intensity",
            ctx.lower_thresh_percent,
        )
    else:
        upper_c, upper_mask = threshold_upper(masked, ctx.thresh_percent)
        lower_c = None

    upper = upper0 if upper_c is None else (float(upper_c[0]), float(upper_c[1]))
    lower = None
    if lower_c is not None and float(lower_c[1]) > float(upper[1]) + float(ctx.min_dist):
        lower = (float(lower_c[0]), float(lower_c[1]))
    if lower is None:
        lower = lower0

    if upper_mask is not None and np.any(upper_mask):
        full_mask = np.asarray(upper_mask, dtype=bool)
    else:
        full_mask = mask
    ctx.full_mask = full_mask

    profile = (
        np.log10(np.maximum(slant_sum, 1.0))
        if ctx.use_log
        else np.asarray(slant_sum, dtype=np.float64)
    )
    result = pack_result(ctx, upper, lower, profile=profile)
    result["raw_sum"] = np.asarray(slant_sum, dtype=np.float64)
    result["slant_seed"] = (upper0, lower0)
    return result
