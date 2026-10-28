# -*- coding: utf-8 -*-
"""獨立測試：標記控制流域分割（Marker-controlled Watershed）抓黏連光斑。

對照概念：
  - 亮度＝高度；黏連雙峰在鞍部立「水壩」切開
  - 先高斯平滑降噪，再給前景／背景標記，避免過度分割

流程（長曝光）:
  0. < noise_floor → 0
  1. gaussian_filter(σ)
  2. 相對峰門檻 → 光斑二值區
  3. 距離轉換找確定前景峰（標記）；膨脹得確定背景
  4. cv2.watershed 切開黏連區
  5. 各流域加權質心；短曝光錨點對應＝上光斑，其餘取下方最優＝下光斑

用法（在 basler/）:
  python test/try_watershed_spot_algo.py
  python test/try_watershed_spot_algo.py --pairs 20 --sigma 4 --noise-floor 256
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

_BASLER_ROOT = Path(__file__).resolve().parent.parent
if str(_BASLER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASLER_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label as ndi_label,
    maximum_filter,
)

from shared_components import compute_auto_spot_center

PX_TO_UM = 3.45
_DEFAULT_NOISE_FLOOR = 256.0


def apply_noise_floor(data, floor=256.0):
    out = np.asarray(data, dtype=np.float64).copy()
    floor = float(floor)
    if np.isfinite(floor):
        out = np.where(out < floor, 0.0, out)
    return out


def _load_pairs(data_dir: Path):
    files = sorted(data_dir.glob("Basler_*.json"))
    by_pair = {}
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        by_pair.setdefault(int(d["pair_index"]), {})[d["exposure_role"]] = p
    return by_pair


def _load_npy(data_dir: Path, json_path: Path) -> np.ndarray:
    return np.load(str(data_dir / json_path.name.replace(".json", ".npy"))).astype(
        float
    )


def _short_spot(mat, thresh):
    return compute_auto_spot_center(
        mat,
        "centroid",
        use_threshold=True,
        thresh_percent=float(thresh),
        bg_subtract=True,
        largest_cc_only=True,
        subpixel=True,
    )


def _stats(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0}
    jumps = np.abs(np.diff(a)) if a.size >= 2 else np.array([0.0])
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "jump_p90": float(np.percentile(jumps, 90)),
        "jump_max": float(np.max(jumps)),
    }


def _weighted_centroid(weights, mask):
    if mask is None or not np.any(mask):
        return None
    ww = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    sel = mask & (ww > 0)
    if not np.any(sel):
        sel = mask
    ys, xs = np.where(sel)
    if ys.size == 0:
        return None
    w = ww[sel]
    wsum = float(np.sum(w))
    if wsum <= 1e-12:
        return float(np.mean(xs)), float(np.mean(ys))
    return float(np.sum(xs * w) / wsum), float(np.sum(ys * w) / wsum)


def find_dual_spots_watershed(
    matrix,
    center_xy,
    *,
    sigma=4.0,
    mask_frac=0.03,
    peak_frac=0.08,
    min_peak_distance=28,
    min_sep_y=40,
    fg_dilate=1,
    bg_dilate=12,
):
    """標記控制流域分割：回傳 primary／secondary 與診斷圖層。

    地形用「反轉平滑圖」：亮核＝盆地，從標記往外淹，在鞍部立壩。
    前景標記＝平滑圖局部極大（弱下斑也能成種子）＋短曝光錨點。
    """
    empty = {
        "primary": None,
        "secondary": None,
        "spots": None,
        "labels": None,
        "markers": None,
        "mask": None,
        "smooth": None,
        "n_basins": 0,
        "reason": "empty",
    }
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0 or center_xy is None:
        return empty

    smooth = gaussian_filter(matrix, sigma=float(sigma), mode="nearest")
    peak = float(np.max(smooth)) if smooth.size else 0.0
    if peak <= 1e-12:
        return {**empty, "smooth": smooth, "reason": "no_signal"}

    # 較寬的光斑區（含弱下斑暈）：相對峰門檻放低
    mask = smooth >= (peak * float(mask_frac))
    # 至少包含平滑後 > 0 的連通主體外圍
    if not np.any(mask):
        mask = smooth > 0
    if not np.any(mask):
        return {**empty, "smooth": smooth, "reason": "no_mask"}

    h, w = smooth.shape
    sx = int(np.clip(round(float(center_xy[0])), 0, w - 1))
    sy = int(np.clip(round(float(center_xy[1])), 0, h - 1))

    # 確定背景：光斑區大幅膨脹後外側
    bg = binary_dilation(mask, iterations=max(1, int(bg_dilate)))
    sure_bg = ~bg

    # 強度局部極大當前景種子（比距離轉換更能抓弱下斑）
    min_sep = max(5, int(min_peak_distance))
    neighborhood = maximum_filter(smooth, size=min_sep)
    local_max = (smooth == neighborhood) & (smooth >= peak * float(peak_frac)) & mask
    # 排除太貼邊上光斑的假峰：仍保留下方／側向峰
    sure_fg = local_max.copy()
    if int(fg_dilate) > 0:
        sure_fg = binary_dilation(sure_fg, iterations=int(fg_dilate)) & mask

    # 強制短曝光錨點為種子
    r_seed = max(3, min_sep // 5)
    yy, xx = np.ogrid[0:h, 0:w]
    sure_fg |= ((xx - sx) ** 2 + (yy - sy) ** 2) <= (r_seed * r_seed)

    fg_labeled, n_fg = ndi_label(sure_fg)
    if n_fg <= 0:
        return {
            **empty,
            "smooth": smooth,
            "mask": mask,
            "reason": "no_fg_markers",
        }

    # OpenCV markers: 0=unknown, 1=background, 2..=foreground
    markers = np.zeros(smooth.shape, dtype=np.int32)
    markers[sure_bg] = 1
    markers[fg_labeled > 0] = fg_labeled[fg_labeled > 0] + 1

    # 反轉地形：亮＝低＝盆地；8-bit 供 watershed
    inv = float(np.max(smooth)) - smooth
    inv = inv - float(np.min(inv))
    imax = float(np.max(inv)) if inv.size else 0.0
    if imax > 1e-12:
        g8 = (255.0 * inv / imax).astype(np.uint8)
    else:
        g8 = np.zeros_like(smooth, dtype=np.uint8)
    color = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    labels = markers.copy()
    basin_ids = sorted(int(v) for v in np.unique(labels) if int(v) > 1)
    centroids = []
    for bid in basin_ids:
        sel = labels == bid
        c = _weighted_centroid(smooth, sel & mask)
        if c is None:
            c = _weighted_centroid(smooth, sel)
        if c is None:
            continue
        area = int(np.count_nonzero(sel & mask))
        if area < 8:
            continue
        centroids.append(
            {
                "id": bid,
                "xy": c,
                "area": area,
                "peak": float(np.max(smooth[sel])) if np.any(sel) else 0.0,
            }
        )

    if not centroids:
        return {
            **empty,
            "smooth": smooth,
            "mask": mask,
            "markers": markers,
            "labels": labels,
            "n_basins": 0,
            "reason": "no_centroids",
        }

    cx, cy = float(center_xy[0]), float(center_xy[1])
    centroids.sort(key=lambda d: (d["xy"][0] - cx) ** 2 + (d["xy"][1] - cy) ** 2)
    primary = centroids[0]["xy"]
    primary_y = float(primary[1])

    min_sep_y = max(8.0, float(min_sep_y))
    candidates = []
    for d in centroids[1:]:
        x, y = d["xy"]
        dy = float(y) - primary_y
        if abs(dy) < min_sep_y:
            continue
        # 偏好下方伴峰
        side = 1.35 if dy > 0 else 0.75
        score = side * d["peak"] * (max(d["area"], 1) ** 0.5)
        candidates.append((score, d))
    secondary = None
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        secondary = candidates[0][1]["xy"]

    spots = None
    if secondary is not None:
        if primary[1] <= secondary[1]:
            spots = [primary, secondary]
        else:
            spots = [secondary, primary]

    return {
        "primary": primary,
        "secondary": secondary,
        "spots": spots,
        "labels": labels,
        "markers": markers,
        "mask": mask,
        "smooth": smooth,
        "n_basins": len(centroids),
        "centroids": centroids,
        "reason": "ok" if secondary is not None else "no_secondary",
    }


def save_demo(pi, display_long, result, short_xy, out_dir, *, sigma, noise_floor):
    pri = result["primary"]
    sec = result["secondary"]
    labels = result.get("labels")
    mask = result.get("mask")
    smooth = result.get("smooth")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle(
        f"Pair {pi} Marker Watershed | noise<{noise_floor:g}→0 | σ={sigma} | "
        f"basins={result.get('n_basins', 0)} | {result.get('reason')}",
        fontsize=11,
    )

    ax = axes[0]
    ax.imshow(display_long, cmap="jet", aspect="equal", origin="upper")
    ax.plot(short_xy[0], short_xy[1], "w+", ms=12, mew=2, label="short seed")
    if pri is not None:
        ax.plot(pri[0], pri[1], "g+", ms=11, mew=2, label="primary")
    if sec is not None:
        ax.plot(sec[0], sec[1], "rx", ms=10, mew=2, label="secondary")
    ax.set_title("Long (noise floor)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    show = smooth if smooth is not None else display_long
    lo, hi = np.percentile(show[np.isfinite(show)], [1, 99.5])
    ax.imshow(show, cmap="jet", aspect="equal", origin="upper", vmin=lo, vmax=hi)
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors="w", linewidths=0.6)
    ax.set_title(f"Smooth σ={sigma} + mask")

    ax = axes[2]
    if labels is not None:
        vis = np.ma.masked_where(labels <= 1, labels)
        ax.imshow(display_long, cmap="gray", aspect="equal", origin="upper")
        ax.imshow(vis, cmap="tab20", aspect="equal", origin="upper", alpha=0.55)
        # watershed lines
        lines = labels == -1
        if np.any(lines):
            ax.contour(lines.astype(float), levels=[0.5], colors="yellow", linewidths=0.8)
    else:
        ax.imshow(display_long, cmap="jet", aspect="equal", origin="upper")
    if pri is not None:
        ax.plot(pri[0], pri[1], "g+", ms=12, mew=2, label="primary")
    if sec is not None:
        ax.plot(sec[0], sec[1], "rx", ms=10, mew=2, label="secondary")
    ax.set_title("Watershed basins")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = out_dir / f"watershed_demo_pair{pi:03d}.png"
    try:
        buf = io.BytesIO()
        fig.savefig(buf, dpi=140, format="png")
        path.write_bytes(buf.getvalue())
    except OSError as exc:
        print(f"  warn: skip demo {path.name}: {exc}")
        path = None
    finally:
        plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"test/700 um")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--thresh", type=float, default=20.0, help="短曝光門檻％")
    ap.add_argument("--noise-floor", type=float, default=_DEFAULT_NOISE_FLOOR)
    ap.add_argument("--sigma", type=float, default=4.0, help="高斯平滑 σ（px）")
    ap.add_argument(
        "--mask-frac",
        type=float,
        default=0.03,
        help="光斑區相對峰門檻（平滑後，宜偏低以涵蓋弱下斑）",
    )
    ap.add_argument(
        "--peak-frac",
        type=float,
        default=0.08,
        help="強度局部極大門檻（相對峰，當前景種子）",
    )
    ap.add_argument(
        "--min-peak-distance",
        type=int,
        default=28,
        help="前景種子最小間距（px）",
    )
    ap.add_argument(
        "--min-sep-y",
        type=float,
        default=40.0,
        help="上下光斑最小 Y 分離（px）",
    )
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (_BASLER_ROOT / data_dir).resolve()
    out_dir = data_dir / "watershed_spot_algo_trial"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pair = _load_pairs(data_dir)
    if not by_pair:
        print(f"ERROR: no Basler_*.json pairs under {data_dir}")
        return 1

    rows = []
    for pi in range(1, int(args.pairs) + 1):
        if pi not in by_pair or "short" not in by_pair[pi] or "long" not in by_pair[pi]:
            continue
        sm = apply_noise_floor(
            _load_npy(data_dir, by_pair[pi]["short"]), args.noise_floor
        )
        lm = apply_noise_floor(
            _load_npy(data_dir, by_pair[pi]["long"]), args.noise_floor
        )
        ss = _short_spot(sm, args.thresh)
        if ss is None:
            continue

        result = find_dual_spots_watershed(
            lm,
            ss,
            sigma=args.sigma,
            mask_frac=args.mask_frac,
            peak_frac=args.peak_frac,
            min_peak_distance=args.min_peak_distance,
            min_sep_y=args.min_sep_y,
        )
        pri, sec = result["primary"], result["secondary"]
        row = {
            "pair": pi,
            "noise_floor": float(args.noise_floor),
            "sigma": float(args.sigma),
            "short_x": float(ss[0]),
            "short_y": float(ss[1]),
            "pri_x": None if pri is None else float(pri[0]),
            "pri_y": None if pri is None else float(pri[1]),
            "sec_x": None if sec is None else float(sec[0]),
            "sec_y": None if sec is None else float(sec[1]),
            "dx_um": None,
            "dy_um": None,
            "dist_um": None,
            "n_basins": int(result.get("n_basins") or 0),
            "reason": result.get("reason", ""),
            "ok": sec is not None,
        }
        if sec is not None:
            dx = float(sec[0] - ss[0])
            dy = float(sec[1] - ss[1])
            dist = float(np.hypot(dx, dy))
            row["dx_um"] = dx * PX_TO_UM
            row["dy_um"] = dy * PX_TO_UM
            row["dist_um"] = dist * PX_TO_UM
        rows.append(row)

        if pi <= 5:
            save_demo(
                pi,
                lm,
                result,
                ss,
                out_dir,
                sigma=args.sigma,
                noise_floor=args.noise_floor,
            )
        if pi % 10 == 0:
            print(f"  progress {pi}/{args.pairs}")

    if not rows:
        print("ERROR: no pairs processed")
        return 1

    csv_path = out_dir / f"watershed_algo_pairs_001_{int(args.pairs):03d}.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})

    ok = [r for r in rows if r["ok"]]
    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(
        f"noise floor: <{args.noise_floor:g} → 0  | gaussian sigma={args.sigma} | "
        f"mask_frac={args.mask_frac} peak_frac={args.peak_frac}"
    )
    print(
        f"params: min_peak_distance={args.min_peak_distance} "
        f"min_sep_y={args.min_sep_y}"
    )
    print(f"success: {len(ok)}/{len(rows)}")
    print("--- Watershed dist_um (short→secondary) ---")
    print(_stats([r["dist_um"] for r in ok]))
    print(f"csv: {csv_path.name}")
    print("demo: watershed_demo_pair001.png … (up to 005)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
