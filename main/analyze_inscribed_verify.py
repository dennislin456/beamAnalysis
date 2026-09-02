#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""驗證指定 .npy 在 60% 門檻下的最大內切圓半徑（對照 App 與全半區）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndi_label

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shared_components import (
    estimate_border_background,
    find_dual_peak_valley_y,
    intersect_half_with_locate_band,
    split_y_index,
)


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.dtype == np.uint16:
        return arr.astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def fit_inscribed_circle(matrix, use_threshold=True, thresh_percent=60.0):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return None
    bg = estimate_border_background(matrix)
    work = np.clip(matrix - bg, 0.0, None)
    peak_val = float(np.max(work)) if work.size > 0 else 0.0
    if not np.isfinite(peak_val) or peak_val <= 0:
        return None
    thresh_val = peak_val * (thresh_percent / 100.0) if use_threshold else peak_val * 0.5
    mask = work >= thresh_val
    if not np.any(mask):
        return {"ok": False, "reason": "empty mask"}

    labeled, num = ndi_label(mask)
    if num <= 0:
        return {"ok": False, "reason": "no cc"}
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_id = int(np.argmax(counts))
    blob = labeled == largest_id
    blob_pixels = int(np.count_nonzero(blob))

    dist = distance_transform_edt(blob)
    max_idx = np.argmax(dist)
    cy, cx = np.unravel_index(max_idx, dist.shape)
    radius = float(dist[cy, cx])
    return {
        "ok": True,
        "cx": float(cx),
        "cy": float(cy),
        "radius": radius,
        "peak": peak_val,
        "thresh": thresh_val,
        "blob_pixels": blob_pixels,
        "region_shape": matrix.shape,
        "bg": bg,
    }


def analyze(path: Path, thresh_percent: float = 60.0, out_png: Path | None = None):
    mat = load_npy(path)
    h, w = mat.shape
    print(f"=== 檔案: {path.name} ===")
    print(f"shape: {h} x {w}, dtype loaded float64, max={mat.max():.1f}, min={mat.min():.1f}")

    info = find_dual_peak_valley_y(mat, min_peak_distance=5, pixel_pitch_um=5.5)
    split_y = info["valley_y"]
    split_y_i = split_y_index(split_y)
    locate = info.get("locate_bounds")
    peaks = info.get("peak_ys")
    print(f"\n--- 波谷切分 ---")
    print(f"valley_y={split_y:.2f}, cut_x={info['cx']:.2f}, peak_ys={peaks}")
    print(f"locate_bounds={locate}")

    for half in ("below", "above"):
        print(f"\n========== {half.upper()} @ {thresh_percent}% ==========")
        y_i = split_y_index(split_y)
        if half == "below":
            full = mat[:y_i, :]
            y_off = 0
        else:
            full = mat[y_i + 1:, :]
            y_off = y_i + 1

        r_full = fit_inscribed_circle(full, True, thresh_percent)
        print(f"[全半區] shape={full.shape} -> {fmt_result(r_full, y_off, 0)}")

        if locate is not None:
            sub, x0, y0 = intersect_half_with_locate_band(mat, split_y, half, locate)
            r_loc = fit_inscribed_circle(sub, True, thresh_percent) if sub is not None else None
            print(f"[locate band] offset=({x0},{y0}) shape={None if sub is None else sub.shape}")
            print(f"            -> {fmt_result(r_loc, y0, x0)}")

            # Y-only locate (full X width in half)
            lx0, ly0, lx1, ly1 = locate
            y_i = split_y_index(split_y)
            if half == "below":
                sy0, sy1 = max(0, ly0), min(y_i, ly1)
                sub_y = mat[sy0:sy1, :]
                y0y = sy0
            else:
                sy0, sy1 = max(y_i + 1, ly0), min(h, ly1)
                sub_y = mat[sy0:sy1, :]
                y0y = sy0
            r_yonly = fit_inscribed_circle(sub_y, True, thresh_percent) if sub_y.size else None
            print(f"[Y-only locate, full X] shape={None if sub_y is None else sub_y.shape}")
            print(f"            -> {fmt_result(r_yonly, y0y, 0)}")

    # App path (after fix: inscribed uses y_only locate band)
    print("\n========== App (M2 Batch inscribed, full half) ==========")
    for half in ("below", "above"):
        y_i = split_y_index(split_y)
        if half == "below":
            sub, x0, y0 = mat[:y_i, :], 0, 0
        else:
            sub, x0, y0 = mat[y_i + 1:, :], 0, y_i + 1
        r = fit_inscribed_circle(sub, True, thresh_percent) if sub is not None else None
        if r and r.get("ok"):
            print(f"{half}: center=({r['cx']+x0:.1f}, {r['cy']+y0:.1f}), r={r['radius']:.2f}px "
                  f"({r['radius']*5.5:.1f}um), region={sub.shape}")
        else:
            print(f"{half}: FAILED")

    if out_png is not None:
        save_comparison_png(mat, split_y, locate, thresh_percent, out_png)
        print(f"\nWrote: {out_png}")


def save_comparison_png(mat, split_y, locate, thresh_percent, out_png: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["full half", "locate XY (old bug)", "locate Y-only (fixed)"]
    modes = ["full", "xy", "yonly"]
    h, w = mat.shape
    y_i = split_y_index(split_y)

    for ax, title, mode in zip(axes, titles, modes):
        work = np.clip(mat - estimate_border_background(mat), 0, None)
        ax.imshow(work, cmap="jet", origin="upper")
        ax.set_title(title)
        for half, color in (("below", "cyan"), ("above", "magenta")):
            if mode == "full":
                if half == "below":
                    sub, x0, y0 = mat[:y_i, :], 0, 0
                else:
                    sub, x0, y0 = mat[y_i + 1:, :], 0, y_i + 1
            elif mode == "xy":
                sub, x0, y0 = intersect_half_with_locate_band(mat, split_y, half, locate)
            else:
                sub, x0, y0 = intersect_half_with_locate_band(
                    mat, split_y, half, locate, y_only=True
                )
            if sub is None or sub.size == 0:
                continue
            r = fit_inscribed_circle(sub, True, thresh_percent)
            if not r or not r.get("ok"):
                continue
            cx, cy, rad = r["cx"] + x0, r["cy"] + y0, r["radius"]
            ax.add_patch(Circle((cx, cy), rad, fill=False, ec=color, lw=2))
            ax.plot(cx, cy, "+", color=color, ms=12, mew=2)
        if locate is not None:
            lx0, ly0, lx1, ly1 = locate
            ax.add_patch(Rectangle(
                (lx0, ly0), lx1 - lx0, ly1 - ly0,
                fill=False, ec="lime", ls="--", lw=1.5,
            ))
        ax.axhline(split_y, color="white", ls="--", lw=1)
    fig.suptitle(f"inscribed @ {thresh_percent}% — {out_png.stem}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def blob_diagnostics(matrix, thresh_percent):
    bg = estimate_border_background(matrix)
    work = np.clip(matrix - bg, 0.0, None)
    peak = float(np.max(work))
    mask = work >= peak * (thresh_percent / 100.0)
    labeled, num = ndi_label(mask)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    blob = labeled == int(np.argmax(counts))
    ys, xs = np.where(blob)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    equiv_r = 0.5 * min(bw, bh)  # 粗略：外接框較短邊的一半
    return {
        "bbox": (x0, y0, x1, y1),
        "equiv_radius": equiv_r,
        "pixels": int(np.count_nonzero(blob)),
    }


def fmt_result(r, y_off, x_off):
    if r is None:
        return "None"
    if not r.get("ok"):
        return f"FAIL ({r.get('reason')})"
    return (
        f"center=({r['cx']+x_off:.1f}, {r['cy']+y_off:.1f}), "
        f"r={r['radius']:.2f}px ({r['radius']*5.5:.1f}um), "
        f"blob_px={r['blob_pixels']}, peak={r['peak']:.1f}, thresh={r['thresh']:.1f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "npy",
        nargs="?",
        default=r"d:\AESDproject\2026\BeamAnalysis\beamImage\20260902_Jenny"
        r"\1c_203_138_p268.32_1100ms\Basler_20260902_194248_001_20260902_194251.npy",
    )
    parser.add_argument("--thresh", type=float, default=60.0)
    parser.add_argument("--png", type=str, default="")
    args = parser.parse_args()
    out_png = Path(args.png) if args.png else None
    analyze(Path(args.npy), args.thresh, out_png)
