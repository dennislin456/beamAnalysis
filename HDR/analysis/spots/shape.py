"""形狀中心 dual-spot mode.

DOC_TITLE: 形狀中心
DOC_SUMMARY: 上光斑用門檻核心質心擺條帶；下光斑是波谷以下最亮連通區的幾何中心（每像素權重相同）。下光斑門檻可與上光斑分開設定。

Pipeline after the shared strip / valley step:
1. Upper spot = threshold-core center above the valley (same seed used for strip X).
2. Lower spot = unweighted mean of pixels in the brightest connected blob below the valley
   (uses lower_thresh_percent vs local peak below the valley).
"""

from __future__ import annotations

from analysis.spots.common import StripContext, lower_centroid, pack_result, prepare_strip


MODE_ID = "shape"
MODE_LABEL = "形狀中心"


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
    """Locate dual spots with shape-center lower blob."""
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
        from analysis.spots.common import empty_result

        return empty_result()
    return resolve(ctx)


def resolve(ctx: StripContext) -> dict:
    x0, y0, _x1, _y1 = ctx.bounds
    upper = (float(x0) + float(ctx.upper_local[0]), float(y0) + float(ctx.upper_local[1]))
    lower = None
    if ctx.lower_local is not None and ctx.lower_local[0] > ctx.upper_local[1] + float(ctx.min_dist):
        center, _ = lower_centroid(
            ctx.work,
            ctx.lower_local[1],
            ctx.work.shape[0] - 1,
            "shape",
            ctx.lower_thresh_percent,
        )
        if center is not None and center[1] > ctx.lower_local[1]:
            lower = (float(x0) + center[0], float(y0) + center[1])
    return pack_result(ctx, upper, lower)
