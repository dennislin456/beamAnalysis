"""條帶波峰 dual-spot mode.

DOC_TITLE: 條帶波峰
DOC_SUMMARY: 只看條帶 Y 加總的兩個峰；X 固定在條帶中心（上光斑種子 X），距離用兩個 Y 的差。

Pipeline after the shared strip / valley step:
1. 上峰 Y：波谷以上（沒有波谷則整段條帶）加總曲線最高峰。
2. 下峰 Y：通過峰距／下峰比／谷深的那個下峰（raw 加總）。
3. 可選 Refine：S-G／Gaussian／多項式擬合 → float Y（距離用 float）。
   多項式為左右各 k 點 → 2k 次，見 docs/quadratic_peaks.md。
4. 兩邊 X 都固定為條帶中心（門檻種子 X），不再取各列最亮點。
"""

from __future__ import annotations

from analysis import gaussian_peaks, quadratic_peaks, savgol_peaks
from analysis.peak_refine_common import (
    METHOD_GAUSSIAN,
    METHOD_NONE,
    METHOD_QUADRATIC,
    METHOD_SAVGOL,
)
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
    *,
    refine_enabled: bool = False,
    refine_method: str = METHOD_NONE,
    refine_window: int = 7,
    refine_order: int = 2,
    refine_sigma: float = 1.0,
    # Back-compat kwargs from the first S-G-only wiring.
    savgol_enabled: bool | None = None,
    savgol_window: int | None = None,
    savgol_order: int | None = None,
) -> dict:
    """Locate dual spots from strip-sum Y peaks with fixed strip-center X."""
    if savgol_enabled is not None:
        refine_enabled = bool(savgol_enabled)
        if refine_enabled and str(refine_method) in ("", METHOD_NONE):
            refine_method = "savgol"
    if savgol_window is not None:
        refine_window = int(savgol_window)
    if savgol_order is not None:
        refine_order = int(savgol_order)
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
    return resolve(
        ctx,
        refine_enabled=refine_enabled,
        refine_method=refine_method,
        refine_window=refine_window,
        refine_order=refine_order,
        refine_sigma=refine_sigma,
    )


def resolve(
    ctx: StripContext,
    *,
    refine_enabled: bool = False,
    refine_method: str = METHOD_NONE,
    refine_window: int = 7,
    refine_order: int = 2,
    refine_sigma: float = 1.0,
) -> dict:
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
    refine_meta = None
    method = str(refine_method or METHOD_NONE).strip().lower()
    if ctx.lower_local is not None:
        lower_y = float(ctx.lower_local[0])
        if lower_y > float(ctx.lower_local[1]) and lower_y > float(upper_y) + float(ctx.min_dist):
            if refine_enabled and method not in ("", METHOD_NONE, "off", "none"):
                if method in ("sg", "s-g", "savitzky-golay", "savitzky_golay"):
                    method = METHOD_SAVGOL
                if method in ("gauss", "gaussian_filter", "gaussian-filter"):
                    method = METHOD_GAUSSIAN
                if method in ("quad", "parabola", "poly2", "quadratic_fit"):
                    method = METHOD_QUADRATIC
                if method == METHOD_SAVGOL:
                    refined = savgol_peaks.refine_lower_peak(
                        ctx.raw_sum,
                        lower_y,
                        window=int(refine_window),
                        polyorder=int(refine_order),
                    )
                elif method == METHOD_GAUSSIAN:
                    # sigma follows window; do not take a separate UI sigma.
                    win = int(refine_window)
                    refined = gaussian_peaks.refine_lower_peak(
                        ctx.raw_sum,
                        lower_y,
                        window=win,
                    )
                    refine_sigma = gaussian_peaks.sigma_from_window(win)
                elif method == METHOD_QUADRATIC:
                    refined = quadratic_peaks.refine_lower_peak(
                        ctx.raw_sum,
                        lower_y,
                        window=int(refine_window),
                    )
                else:
                    refined = None
                if refined is not None:
                    lower_y = float(refined.y)
                    refine_meta = {
                        "enabled": True,
                        "method": str(refined.method),
                        "window": int(refine_window),
                        "order": int(refine_order),
                        "sigma": float(refine_sigma),
                        "lo": int(y0) + int(refined.lo),
                        "hi": int(y0) + int(refined.hi),
                        "local_lo": int(refined.lo),
                        "local_hi": int(refined.hi),
                        "smooth": refined.smooth,
                        "poly_coeff": (
                            None
                            if refined.poly_coeff is None
                            else [float(v) for v in refined.poly_coeff]
                        ),
                        "peak_value": (
                            None
                            if refined.peak_value is None
                            else float(refined.peak_value)
                        ),
                        "poly_x_mid": refined.poly_x_mid,
                        "poly_x_scale": refined.poly_x_scale,
                    }
            lower = (float(x0) + fixed_x, float(y0) + lower_y)
    return pack_result(ctx, upper, lower, savgol=refine_meta)
