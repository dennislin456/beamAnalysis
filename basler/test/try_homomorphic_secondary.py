# -*- coding: utf-8 -*-
"""測試同態濾波（Homomorphic Filter）對下光斑可見度／定位的影響。

流程（對齊說明書）:
  Log → FFT2 → 高通濾波 → IFFT2 → Exp

用法（在 basler/）:
  python test/try_homomorphic_secondary.py
  python test/try_homomorphic_secondary.py --pairs 100 --data "test/700 um"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_BASLER_ROOT = Path(__file__).resolve().parent.parent
if str(_BASLER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASLER_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shared_components as sc
from shared_components import (
    clip_roi_to_matrix,
    compute_auto_spot_center,
    find_dual_spots_axial_1d_projection,
)

PX_TO_UM = 3.45


# ---------------------------------------------------------------------------
# 同態濾波
# ---------------------------------------------------------------------------
def homomorphic_filter(
    image,
    *,
    d0=40.0,
    gamma_l=0.5,
    gamma_h=2.0,
    c=1.0,
    eps=1.0,
):
    """Gaussian 型同態高通：壓低照明(低頻)、抬高反射/細節(高頻)。

    H(u,v) = (γH - γL) * (1 - exp(-c * (D²/D0²))) + γL
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        raise ValueError("image must be 2D")
    # 避免 log(0)
    work = np.clip(img, 0.0, None) + float(eps)
    log_img = np.log(work)

    f = np.fft.fft2(log_img)
    fshift = np.fft.fftshift(f)

    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    d0 = max(float(d0), 1e-6)
    h_filter = (float(gamma_h) - float(gamma_l)) * (
        1.0 - np.exp(-float(c) * (d2 / (d0 * d0)))
    ) + float(gamma_l)

    filtered = np.fft.ifftshift(fshift * h_filter)
    out_log = np.fft.ifft2(filtered).real
    out = np.exp(out_log)
    # 數值穩定性：去掉非有限值
    out = np.where(np.isfinite(out), out, 0.0)
    return out, h_filter


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


def _strip_profile(matrix, center_xy, *, roi_w, roi_h):
    xywh = sc._roi_centered_on_xy(matrix, center_xy, roi_w=roi_w, roi_h=roi_h)
    bounds = clip_roi_to_matrix(matrix, xywh)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    work = matrix[y0:y1, x0:x1]
    raw = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
    bg = float(np.median(raw)) if raw.size else 0.0
    profile = np.clip(raw - bg, 0.0, None)
    return {
        "work": work,
        "bounds": (x0, y0, x1, y1),
        "y0": float(y0),
        "x0": float(x0),
        "profile": profile,
        "xywh": xywh,
    }


def _secondary_from_matrix(matrix, short_xy, *, thresh_unused, smooth, roi_w, roi_h, min_sep):
    """對任意長曝光矩陣跑現行 SUM 次峰管線。"""
    result = find_dual_spots_axial_1d_projection(
        np.asarray(matrix, dtype=float),
        short_xy,
        roi_w=roi_w,
        roi_h=roi_h,
        smooth_win=smooth,
        min_peak_distance=min_sep,
    )
    spots = result.get("spots")
    if not spots or len(spots) < 2:
        return None, result
    pidx = int(result.get("primary_index", 0) or 0)
    sec = spots[1 - pidx]
    return (float(sec[0]), float(sec[1])), result


def _amp_ratio(profile, pri_y_full, sec_y_full, y0, smooth_win=8):
    smooth = sc._smooth_1d_profile(profile, smooth_win)
    n = len(smooth)

    def peak_near(y_full, half=12):
        i = int(np.clip(round(y_full - y0), 0, n - 1))
        a = max(0, i - half)
        b = min(n, i + half + 1)
        return float(np.max(smooth[a:b]))

    ap = peak_near(pri_y_full)
    as_ = peak_near(sec_y_full)
    if as_ <= 1e-12:
        return None
    return ap / as_


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


def save_demo(pair_i, short_xy, long_raw, long_homo, out_raw, out_homo, out_dir, *, d0, gl, gh):
    """存對照圖：原圖／同態圖 + 兩者 SUM 剖面。"""
    ss_raw, res_raw = out_raw
    ss_homo, res_homo = out_homo

    strip_r = _strip_profile(long_raw, short_xy, roi_w=160, roi_h=512)
    strip_h = _strip_profile(long_homo, short_xy, roi_w=160, roi_h=512)
    if strip_r is None or strip_h is None:
        return None

    y0 = strip_r["y0"]
    y_full = np.arange(len(strip_r["profile"])) + y0
    s_r = sc._smooth_1d_profile(strip_r["profile"], 8)
    s_h = sc._smooth_1d_profile(strip_h["profile"], 8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Pair {pair_i} Homomorphic | D0={d0} gL={gl} gH={gh}",
        fontsize=12,
    )

    # ROI crops
    wr = strip_r["work"]
    wh = strip_h["work"]
    ax = axes[0, 0]
    ax.imshow(wr, cmap="jet", aspect="auto", origin="upper")
    ax.set_title("Long ROI (raw)")
    ax.set_ylabel("Y local")
    ax = axes[0, 1]
    # display scale for homo
    lo, hi = np.percentile(wh[np.isfinite(wh)], [1, 99.5]) if wh.size else (0, 1)
    ax.imshow(wh, cmap="jet", aspect="auto", origin="upper", vmin=lo, vmax=hi)
    ax.set_title("Long ROI (homomorphic)")

    ax = axes[1, 0]
    ax.plot(y_full, strip_r["profile"], color="#90A4AE", lw=0.8, label="sum raw")
    ax.plot(y_full, s_r, color="#1565C0", lw=1.3, label="sum smooth 8")
    if ss_raw is not None:
        ax.axvline(ss_raw[1], color="#C62828", ls="--", label=f"sec Y={ss_raw[1]:.1f}")
    if res_raw.get("spots"):
        pidx = int(res_raw.get("primary_index", 0) or 0)
        ax.axvline(res_raw["spots"][pidx][1], color="#2E7D32", ls=":", label="pri")
    ax.set_title("SUM profile on RAW")
    ax.set_xlabel("Y full px")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(y_full, strip_h["profile"], color="#90A4AE", lw=0.8, label="sum raw")
    ax.plot(y_full, s_h, color="#EF6C00", lw=1.3, label="sum smooth 8")
    if ss_homo is not None:
        ax.axvline(ss_homo[1], color="#C62828", ls="--", label=f"sec Y={ss_homo[1]:.1f}")
    if res_homo.get("spots"):
        pidx = int(res_homo.get("primary_index", 0) or 0)
        ax.axvline(res_homo["spots"][pidx][1], color="#2E7D32", ls=":", label="pri")
    ax.set_title("SUM profile on HOMOMORPHIC")
    ax.set_xlabel("Y full px")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"homo_demo_pair{pair_i:03d}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"test/700 um")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=20.0)
    ap.add_argument("--smooth", type=int, default=8)
    ap.add_argument("--roi-w", type=int, default=160)
    ap.add_argument("--roi-h", type=int, default=512)
    ap.add_argument("--min-sep", type=int, default=12)
    ap.add_argument("--d0", type=float, default=40.0, help="高通截止（px 頻率尺度）")
    ap.add_argument("--gamma-l", type=float, default=0.5, help="低頻增益 <1 壓照明")
    ap.add_argument("--gamma-h", type=float, default=2.0, help="高頻增益 >1 抬細節")
    ap.add_argument("--c", type=float, default=1.0, help="過渡銳利度")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (_BASLER_ROOT / data_dir).resolve()
    out_dir = data_dir / "axial_1d_homomorphic_trial"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pair = _load_pairs(data_dir)
    rows = []
    for pi in range(1, int(args.pairs) + 1):
        if pi not in by_pair or "short" not in by_pair[pi] or "long" not in by_pair[pi]:
            continue
        sm = _load_npy(data_dir, by_pair[pi]["short"])
        lm = _load_npy(data_dir, by_pair[pi]["long"])
        ss = _short_spot(sm, args.thresh)
        if ss is None:
            continue

        # 同態濾波：對整張長曝光做（再進既有 ROI/SUM 管線）
        lm_h, _hf = homomorphic_filter(
            lm,
            d0=args.d0,
            gamma_l=args.gamma_l,
            gamma_h=args.gamma_h,
            c=args.c,
        )

        sec_r, res_r = _secondary_from_matrix(
            lm,
            ss,
            thresh_unused=args.thresh,
            smooth=args.smooth,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            min_sep=args.min_sep,
        )
        sec_h, res_h = _secondary_from_matrix(
            lm_h,
            ss,
            thresh_unused=args.thresh,
            smooth=args.smooth,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            min_sep=args.min_sep,
        )

        def pack(sec):
            if sec is None:
                return None, None, None
            dx = sec[0] - ss[0]
            dy = sec[1] - ss[1]
            dist = float(np.hypot(dx, dy))
            return sec[0], sec[1], dist * PX_TO_UM

        sx_r, sy_r, dist_r = pack(sec_r)
        sx_h, sy_h, dist_h = pack(sec_h)

        # 振幅比（各自 SUM 剖面）
        ratio_r = ratio_h = None
        strip_r = _strip_profile(lm, ss, roi_w=args.roi_w, roi_h=args.roi_h)
        strip_h = _strip_profile(lm_h, ss, roi_w=args.roi_w, roi_h=args.roi_h)
        if strip_r and sec_r and res_r.get("spots"):
            pidx = int(res_r.get("primary_index", 0) or 0)
            pri_y = res_r["spots"][pidx][1]
            ratio_r = _amp_ratio(strip_r["profile"], pri_y, sec_r[1], strip_r["y0"])
        if strip_h and sec_h and res_h.get("spots"):
            pidx = int(res_h.get("primary_index", 0) or 0)
            pri_y = res_h["spots"][pidx][1]
            ratio_h = _amp_ratio(strip_h["profile"], pri_y, sec_h[1], strip_h["y0"])

        rows.append(
            {
                "pair": pi,
                "short_x": float(ss[0]),
                "short_y": float(ss[1]),
                "sec_raw_x": sx_r,
                "sec_raw_y": sy_r,
                "dist_raw_um": dist_r,
                "sec_homo_x": sx_h,
                "sec_homo_y": sy_h,
                "dist_homo_um": dist_h,
                "amp_ratio_raw": ratio_r,
                "amp_ratio_homo": ratio_h,
                "ok_raw": sec_r is not None,
                "ok_homo": sec_h is not None,
            }
        )

        if pi <= 3:
            save_demo(
                pi,
                ss,
                lm,
                lm_h,
                (sec_r, res_r),
                (sec_h, res_h),
                out_dir,
                d0=args.d0,
                gl=args.gamma_l,
                gh=args.gamma_h,
            )
        if pi % 25 == 0:
            print(f"  progress {pi}/{args.pairs}")

    csv_path = out_dir / f"homo_vs_raw_pairs_001_{int(args.pairs):03d}.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})

    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(
        f"params: D0={args.d0} gamma_l={args.gamma_l} gamma_h={args.gamma_h} c={args.c}"
    )
    print(f"pairs: {len(rows)}")
    print("--- RAW (SUM pipeline) dist_um ---")
    print(_stats([r["dist_raw_um"] for r in rows]))
    print("--- HOMOMORPHIC then SUM pipeline dist_um ---")
    print(_stats([r["dist_homo_um"] for r in rows]))
    print("--- amp ratio primary/secondary (median) ---")
    rr = [r["amp_ratio_raw"] for r in rows if r["amp_ratio_raw"] is not None]
    rh = [r["amp_ratio_homo"] for r in rows if r["amp_ratio_homo"] is not None]
    if rr:
        print(f"  raw  median ratio = {np.median(rr):.1f}x")
    if rh:
        print(f"  homo median ratio = {np.median(rh):.1f}x")
    ok_h = sum(1 for r in rows if r["ok_homo"])
    ok_r = sum(1 for r in rows if r["ok_raw"])
    print(f"success: raw={ok_r}/{len(rows)}  homo={ok_h}/{len(rows)}")
    print(f"csv: {csv_path.name}")
    print("demo: homo_demo_pair001.png (+002,003)")


if __name__ == "__main__":
    main()
