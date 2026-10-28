# -*- coding: utf-8 -*-
"""對照現行下光斑 X，與「小平滑後亮核形狀中心」。不改正式演算法。

形狀中心：同一 Y 帶上做 n×n 平均，扣中位數後，
只留 ≥ 峰值 55% 的像素，取遮罩的不加權 X 平均。

用法（在 basler/）:
  python test/try_smooth_shape_center.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_BASLER_ROOT = Path(__file__).resolve().parent.parent
if str(_BASLER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASLER_ROOT))

import numpy as np
from scipy.ndimage import uniform_filter

from shared_components import (
    _peak_width_at_frac_1d,
    _peak_x_in_y_band,
    _smooth_1d_profile,
    _valley_between_peaks_1d,
    compute_auto_spot_center,
    find_dual_spots_axial_1d_projection,
)
from test.try_sum_secondary_all_test import discover_root_numbered_npy

CORE_FRAC = 0.55
SMOOTH_NS = (3, 5)


def _secondary_y_band(work, profile, py0, sy0):
    """與 _refine_primary_secondary_spots 相同的下光斑 Y 帶。"""
    h = int(work.shape[0])
    smooth = _smooth_1d_profile(profile, 7)
    pi = int(np.clip(round(float(py0)), 0, max(0, h - 1)))
    core_w = max(6.0, _peak_width_at_frac_1d(smooth, pi, frac=0.65))
    sep = abs(float(sy0) - float(py0))
    y_half = int(np.clip(0.40 * sep if sep > 1 else 12, 10, 32))
    sign = 1.0 if float(sy0) >= float(py0) else -1.0
    sy = float(sy0) + sign * float(min(8.0, max(3.0, 0.10 * sep)))
    sy = float(np.clip(sy, 0.0, max(0.0, h - 1.0)))
    valley = _valley_between_peaks_1d(smooth, py0, sy0)
    if not np.isfinite(float(valley)):
        valley = 0.5 * (float(py0) + float(sy0))
    y_toward = int(np.clip(0.45 * y_half, 6, y_half))
    y_away = int(np.clip(1.35 * y_half, y_half, 44))
    sy_i = int(round(float(sy)))
    if sign >= 0.0:
        y0 = max(int(np.floor(valley)) + 1, sy_i - y_toward)
        y1 = min(h, sy_i + y_away + 1)
    else:
        y0 = max(0, sy_i - y_away)
        y1 = min(int(np.ceil(valley)), sy_i + y_toward + 1)
    if y1 <= y0 + 2:
        y0 = max(0, sy_i - y_half)
        y1 = min(h, sy_i + y_half + 1)
    return y0, y1, core_w, sep


def _shape_center_x(band, n, core_frac=CORE_FRAC):
    """n×n 平滑後，亮核遮罩的形狀中心 X（遮罩內 X 的平均）。"""
    sm = uniform_filter(np.asarray(band, dtype=np.float64), size=int(n), mode="nearest")
    bg = float(np.median(sm))
    weights = np.clip(sm - bg, 0.0, None)
    peak = float(np.max(weights)) if weights.size else 0.0
    if peak <= 1e-12:
        return None, 0
    mask = weights >= (peak * float(core_frac))
    if not np.any(mask):
        return None, 0
    xs = np.where(mask)[1].astype(np.float64)
    return float(np.mean(xs)), int(xs.size)


def _run_pair(short_path, long_path, *, thresh=50.0, smooth=7, roi_w=160, roi_h=512, min_sep=12):
    short = np.load(str(short_path)).astype(float)
    long = np.load(str(long_path)).astype(float)
    ss = compute_auto_spot_center(
        short,
        "centroid",
        use_threshold=True,
        thresh_percent=float(thresh),
        bg_subtract=True,
        largest_cc_only=True,
        subpixel=True,
    )
    if ss is None:
        return None
    result = find_dual_spots_axial_1d_projection(
        long,
        ss,
        roi_w=roi_w,
        roi_h=roi_h,
        smooth_win=smooth,
        min_peak_distance=min_sep,
    )
    spots = result.get("spots")
    roi = result.get("roi_xywh")
    if not spots or len(spots) < 2 or roi is None:
        return None
    pidx = int(result.get("primary_index", 0) or 0)
    pri = spots[pidx]
    sec = spots[1 - pidx]
    x0, y0, rw, rh = [int(v) for v in roi]
    work = long[y0 : y0 + rh, x0 : x0 + rw]
    raw = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
    profile = np.clip(raw - float(np.median(raw)), 0.0, None)
    py = float(pri[1]) - float(y0)
    sy = float(sec[1]) - float(y0)
    band_y0, band_y1, core_w, sep = _secondary_y_band(work, profile, py, sy)
    band = work[band_y0:band_y1, :]
    cur = _peak_x_in_y_band(work, band_y0, band_y1, core_frac=CORE_FRAC)
    row = {
        "file": Path(long_path).name,
        "sec_x": float(sec[0]),
        "cur_x": None if cur is None else float(cur) + float(x0),
        "short_x": float(ss[0]),
        "core_w": float(core_w),
        "sep": float(sep),
        "band_h": int(band_y1 - band_y0),
    }
    for n in SMOOTH_NS:
        sx, npx = _shape_center_x(band, n)
        row[f"shape{n}_x"] = None if sx is None else float(sx) + float(x0)
        row[f"shape{n}_n"] = npx
    return row


def _summ(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n=0"
    jumps = np.abs(np.diff(a)) if a.size >= 2 else np.array([0.0])
    return (
        f"n={a.size}  mean={np.mean(a):+.2f}  std={np.std(a):.2f}  "
        f"|max|={np.max(np.abs(a)):.2f}  jump_p90={np.percentile(jumps, 90):.2f}"
    )


def main():
    test_dir = _BASLER_ROOT / "test"
    pairs = discover_root_numbered_npy(test_dir)
    basler = sorted(test_dir.glob("Basler_50us_*_00*_*.npy"))
    if len(basler) >= 2:
        pairs.append(
            {
                "dataset": "basler_50us",
                "pair_id": "001_002",
                "short_path": basler[0],
                "long_path": basler[1],
            }
        )
    print(f"pairs: {len(pairs)}")
    rows = []
    for p in pairs:
        row = _run_pair(p["short_path"], p["long_path"])
        if row is None:
            print(f"  fail {p['pair_id']}")
            continue
        rows.append(row)
        d3 = row["shape3_x"] - row["cur_x"]
        d5 = row["shape5_x"] - row["cur_x"]
        print(
            f"{p['pair_id']}: cur={row['cur_x']:.2f}  "
            f"shape3={row['shape3_x']:.2f} ({d3:+.2f}, n={row['shape3_n']})  "
            f"shape5={row['shape5_x']:.2f} ({d5:+.2f}, n={row['shape5_n']})  "
            f"algo={row['sec_x']:.2f}  short={row['short_x']:.2f}"
        )

    if not rows:
        print("no pairs")
        return
    cur = np.array([r["cur_x"] for r in rows], dtype=float)
    s3 = np.array([r["shape3_x"] for r in rows], dtype=float)
    s5 = np.array([r["shape5_x"] for r in rows], dtype=float)
    algo = np.array([r["sec_x"] for r in rows], dtype=float)
    print("\n=== vs current bright-core X (px) ===")
    print("algo - cur ", _summ(algo - cur))
    print("shape3 - cur", _summ(s3 - cur))
    print("shape5 - cur", _summ(s5 - cur))
    print("\n=== frame-to-frame |dX| (px) ===")
    print("current ", _summ(np.abs(np.diff(cur))))
    print("shape3  ", _summ(np.abs(np.diff(s3))))
    print("shape5  ", _summ(np.abs(np.diff(s5))))


if __name__ == "__main__":
    main()
