"""顯示用斜剖面 dual-spot mode.

DOC_TITLE: 顯示用斜剖面
DOC_SUMMARY: 抓點方式與強度質心相同；Y 剖面／加總改沿上下光斑連線做 X 線性內插，只影響顯示。

Pipeline after the shared strip / valley step:
1. Upper / lower coordinates match intensity-centroid mode（下光斑用獨立門檻）。
2. 回傳的剖面是穿過兩光斑的斜線取樣（每列 X 線性內插），供顯示用 Y 剖面／加總。
"""

from __future__ import annotations

from analysis.spots.common import (
    StripContext,
    lower_centroid,
    pack_result,
    prepare_strip,
    slanted_line_profile,
)


MODE_ID = "line"
MODE_LABEL = "顯示用斜剖面"


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
    """Locate dual spots like intensity mode; profile follows the slanted line."""
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
            "intensity",
            ctx.lower_thresh_percent,
        )
        if center is not None and center[1] > ctx.lower_local[1]:
            lower = (float(x0) + center[0], float(y0) + center[1])
    upper_local = (float(upper[0]) - float(x0), float(upper[1]) - float(y0))
    lower_local_pt = (
        None
        if lower is None
        else (float(lower[0]) - float(x0), float(lower[1]) - float(y0))
    )
    profile = slanted_line_profile(ctx.work, upper_local, lower_local_pt)
    return pack_result(ctx, upper, lower, profile=profile)
