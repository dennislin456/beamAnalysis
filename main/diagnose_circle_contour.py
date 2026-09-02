#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""診斷：內切圓是否超出門檻 contour（同幀、同 %）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as ndi_label

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shared_components import (
    estimate_border_background,
    find_dual_peak_valley_y,
    intersect_half_with_locate_band,
    split_y_index,
)
from analyze_inscribed_verify import fit_inscribed_circle, load_npy


def build_half_mask(matrix, split_y, half, thresh_percent, use_threshold=True):
    """同 tab_batch._build_threshold_mask 邏輯（全半區、全寬 X）。"""
    from shared_components import build_robust_threshold_mask

    matrix = np.asarray(matrix, dtype=np.float64)
    y_i = split_y_index(split_y)
    if half == "below":
        region = matrix[:y_i, :]
        y_off = 0
    else:
        region = matrix[y_i + 1:, :]
        y_off = y_i + 1
    region_mask = build_robust_threshold_mask(
        region, use_threshold, thresh_percent, bg_subtract=True, largest_cc_only=True,
    )
    if region_mask is None:
        return None, y_off
    full = np.zeros(matrix.shape, dtype=bool)
    h_sub = region_mask.shape[0]
    full[y_off: y_off + h_sub, :] = region_mask
    return full, y_off


def circle_pixels(cx, cy, r, shape):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def diagnose(path: Path, thresh: float = 60.0):
    mat = load_npy(path)
    h, w = mat.shape
    info = find_dual_peak_valley_y(mat, min_peak_distance=5, pixel_pitch_um=5.5)
    split_y = info["valley_y"]
    locate = info.get("locate_bounds")

    print(f"file: {path.name}")
    print(f"valley_y={split_y:.1f} peaks={info.get('peak_ys')} locate={locate}")

    for half in ("below", "above"):
        print(f"\n--- {half.upper()} @ {thresh}% ---")
        mask_full, _ = build_half_mask(mat, split_y, half, thresh)
        sub_xy, x0, y0 = intersect_half_with_locate_band(mat, split_y, half, locate)
        sub_y, x0y, y0y = intersect_half_with_locate_band(
            mat, split_y, half, locate, y_only=True
        )

        for label, sub, xo, yo in (
            ("full_half",
             mat[: split_y_index(split_y), :] if half == "below" else mat[split_y_index(split_y) + 1:, :],
             0, 0 if half == "below" else split_y_index(split_y) + 1),
            ("locate_xy", sub_xy, x0, y0),
            ("locate_yonly", sub_y, x0y, y0y),
        ):
            r = fit_inscribed_circle(sub, True, thresh)
            if not r or not r.get("ok"):
                print(f"  [{label}] no circle")
                continue
            cx = r["cx"] + xo
            cy = r["cy"] + yo
            rad = r["radius"]
            bg = estimate_border_background(sub)
            work = np.clip(sub - bg, 0, None)
            peak_local = float(np.max(work))
            print(
                f"  [{label}] region={sub.shape} peak_local={peak_local:.0f} "
                f"thresh_abs={r['thresh']:.0f} r={rad:.2f}px center=({cx:.1f},{cy:.1f})"
            )

            circ = circle_pixels(cx, cy, rad, (h, w))
            if mask_full is None:
                continue
            outside = circ & (~mask_full)
            n_out = int(np.count_nonzero(outside))
            n_in = int(np.count_nonzero(circ & mask_full))
            print(f"           circle vs overlay_mask: inside={n_in}px outside={n_out}px")
            if n_out > 0:
                ys, xs = np.where(outside)
                print(f"           *** LEAK: max_outside_dist={np.max(np.hypot(xs-cx, ys-cy)):.1f} r={rad:.2f}")


if __name__ == "__main__":
    p = Path(
        r"d:\AESDproject\2026\BeamAnalysis\beamImage\20260902_Jenny"
        r"\5d_119.4_60.3_p268.02_1100ms\Basler_20260902_200448_066_20260902_200452.npy"
    )
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    diagnose(p, thresh)
