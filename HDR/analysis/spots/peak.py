"""條帶波峰 dual-spot mode.

DOC_TITLE: 條帶波峰
DOC_SUMMARY: 只看條帶 Y 加總的兩個峰；X 固定在條帶中心（上光斑種子 X），距離用兩個 Y 的差。

Pipeline after the shared strip / valley step:
1. 上峰 Y：波谷以上（沒有波谷則整段條帶）加總曲線最高峰。
2. 下峰 Y：通過距離／下峰比／谷深的那個下峰。
3. 兩邊 X 都固定為條帶中心（門檻種子 X），不再取各列最亮點。
"""

from __future__ import annotations

from analysis.spots.common import (
    StripContext,
    highest_peak_above,
    pack_result,
    prepare_strip,
)


MODE_ID = "peak"
MODE_LABEL = "條帶波峰"


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
    """Locate dual spots from strip-sum Y peaks with fixed strip-center X."""
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
        peak_seed=True,
    )
    if ctx is None:
        from analysis.spots.common import empty_result

        return empty_result()
    return resolve(ctx)


def resolve(ctx: StripContext) -> dict:
    x0, y0, _x1, _y1 = ctx.bounds
    strip_peak = highest_peak_above(ctx.raw_sum, float(ctx.raw_sum.size))
    peak1 = int(round(strip_peak if strip_peak is not None else ctx.seed_y))
    upper_y = float(peak1)
    if ctx.lower_local is not None:
        found = highest_peak_above(ctx.raw_sum, float(ctx.lower_local[1]))
        if found is not None:
            upper_y = found
    # Fixed X: strip center from the threshold upper seed (same for both peaks).
    fixed_x = float(ctx.seed_x)
    upper = (float(x0) + fixed_x, float(y0) + float(upper_y))
    lower = None
    if ctx.lower_local is not None:
        lower_y = float(ctx.lower_local[0])
        if lower_y > float(ctx.lower_local[1]) and lower_y > float(upper_y) + float(ctx.min_dist):
            lower = (float(x0) + fixed_x, float(y0) + lower_y)
    return pack_result(ctx, upper, lower)
