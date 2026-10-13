# -*- coding: utf-8 -*-
"""獨立測試：同態濾波抓雙光斑（不接一維投影／SUM 管線）。

演算法:
  0. 雜訊門檻：強度 < noise_floor（預設 256）視為雜訊，設為 0
     （可選再做 white above 硬裁切）
  1. Log → FFT2 → 高通 → IFFT2 → Exp（對長曝光）
  2. 以短曝光光斑為上光斑錨點
  3. 在同態影像、上光斑下方做門檻＋最大連通區質心 → 下光斑

用法（在 basler/）:
  python test/try_homomorphic_spot_algo.py
  python test/try_homomorphic_spot_algo.py --pairs 100 --noise-floor 256
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
from scipy.ndimage import label as ndi_label

from shared_components import compute_auto_spot_center

PX_TO_UM = 3.45
# 與 UI color scale「black below」語意一致：低於此視為雜訊 → 0
_DEFAULT_NOISE_FLOOR = 256.0


def apply_noise_floor(data, floor=256.0, white_above=None):
    """< floor 視為雜訊設為 0；若給 white_above 則 >hi 裁成 hi。"""
    out = np.asarray(data, dtype=np.float64).copy()
    floor = float(floor)
    if np.isfinite(floor):
        out = np.where(out < floor, 0.0, out)
    if white_above is not None and np.isfinite(float(white_above)):
        hi = float(white_above)
        out = np.where(out > hi, hi, out)
    return out


def homomorphic_filter(
    image,
    *,
    d0=40.0,
    gamma_l=0.5,
    gamma_h=2.0,
    c=1.0,
    eps=1.0,
):
    """同態高通：Log → FFT → H → IFFT → Exp。"""
    img = np.asarray(image, dtype=np.float64)
    work = np.clip(img, 0.0, None) + float(eps)
    log_img = np.log(work)
    fshift = np.fft.fftshift(np.fft.fft2(log_img))
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    d0 = max(float(d0), 1e-6)
    h_filter = (float(gamma_h) - float(gamma_l)) * (
        1.0 - np.exp(-float(c) * (d2 / (d0 * d0)))
    ) + float(gamma_l)
    out = np.exp(np.fft.ifft2(np.fft.ifftshift(fshift * h_filter)).real)
    return np.where(np.isfinite(out), out, 0.0)


def _centroid_largest_cc(weights, mask):
    labeled, num = ndi_label(mask)
    if num <= 0:
        return None
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    lid = int(np.argmax(counts))
    sel = labeled == lid
    ww = weights[sel]
    ys, xs = np.where(sel)
    wsum = float(np.sum(ww))
    if wsum <= 1e-12 or ys.size == 0:
        return None
    cx = float(np.sum(xs.astype(np.float64) * ww) / wsum)
    cy = float(np.sum(ys.astype(np.float64) * ww) / wsum)
    return cx, cy, int(ys.size)


def find_dual_spots_homomorphic(
    long_mat,
    short_xy,
    *,
    d0=40.0,
    gamma_l=0.5,
    gamma_h=2.0,
    c=1.0,
    thresh_frac=0.35,
    search_half_x=80,
    min_sep_y=40,
    primary_half=24,
):
    """同態影像上抓上／下光斑。

    Returns:
        dict: filtered, primary (cx,cy), secondary (cx,cy) 或 None
    """
    filtered = homomorphic_filter(
        long_mat, d0=d0, gamma_l=gamma_l, gamma_h=gamma_h, c=c
    )
    h, w = filtered.shape
    sx, sy = float(short_xy[0]), float(short_xy[1])

    # ---- 上光斑：短曝光錨點附近局部亮核質心 ----
    x0 = int(np.clip(round(sx) - primary_half, 0, w - 1))
    x1 = int(np.clip(round(sx) + primary_half + 1, x0 + 1, w))
    y0 = int(np.clip(round(sy) - primary_half, 0, h - 1))
    y1 = int(np.clip(round(sy) + primary_half + 1, y0 + 1, h))
    patch = filtered[y0:y1, x0:x1]
    bg = float(np.median(patch)) if patch.size else 0.0
    weights = np.clip(patch - bg, 0.0, None)
    peak = float(np.max(weights)) if weights.size else 0.0
    if peak <= 1e-12:
        primary = (sx, sy)
    else:
        mask = weights >= (peak * 0.50)
        local = _centroid_largest_cc(weights, mask)
        if local is None:
            primary = (sx, sy)
        else:
            primary = (local[0] + x0, local[1] + y0)

    # ---- 下光斑：上光斑下方、軸向窄帶內門檻最大連通區 ----
    px, py = primary
    y_lo = int(np.clip(round(py) + int(min_sep_y), 0, h - 1))
    if y_lo >= h - 3:
        return {
            "filtered": filtered,
            "primary": primary,
            "secondary": None,
            "reason": "no_room_below",
        }

    xa = int(np.clip(round(px) - search_half_x, 0, w - 1))
    xb = int(np.clip(round(px) + search_half_x + 1, xa + 1, w))
    region = filtered[y_lo:h, xa:xb]
    bg2 = float(np.median(region)) if region.size else 0.0
    weights2 = np.clip(region - bg2, 0.0, None)
    peak2 = float(np.max(weights2)) if weights2.size else 0.0
    if peak2 <= 1e-12:
        return {
            "filtered": filtered,
            "primary": primary,
            "secondary": None,
            "reason": "no_secondary_signal",
        }

    mask2 = weights2 >= (peak2 * float(thresh_frac))
    local2 = _centroid_largest_cc(weights2, mask2)
    if local2 is None:
        return {
            "filtered": filtered,
            "primary": primary,
            "secondary": None,
            "reason": "no_cc",
        }

    secondary = (local2[0] + xa, local2[1] + y_lo)
    return {
        "filtered": filtered,
        "primary": primary,
        "secondary": secondary,
        "sec_peak": peak2,
        "pri_peak": peak,
        "amp_ratio": (peak / peak2) if peak2 > 1e-12 else None,
        "reason": "ok",
    }


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


def save_demo(pi, display_long, result, short_xy, out_dir, *, d0, gl, gh, noise_floor):
    filtered = result["filtered"]
    pri = result["primary"]
    sec = result["secondary"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Pair {pi} Homomorphic (noise<{noise_floor:g}→0) | "
        f"D0={d0} gL={gl} gH={gh}",
        fontsize=11,
    )

    ax = axes[0]
    ax.imshow(display_long, cmap="jet", aspect="equal", origin="upper")
    ax.plot(short_xy[0], short_xy[1], "w+", ms=12, mew=2, label="short")
    if pri is not None:
        ax.plot(pri[0], pri[1], "g+", ms=12, mew=2, label="primary")
    if sec is not None:
        ax.plot(sec[0], sec[1], "rx", ms=10, mew=2, label="secondary")
    ax.set_title(f"Long after noise floor (<{noise_floor:g}=0)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    lo, hi = np.percentile(filtered, [1, 99.5])
    ax.imshow(filtered, cmap="jet", aspect="equal", origin="upper", vmin=lo, vmax=hi)
    if pri is not None:
        ax.plot(pri[0], pri[1], "g+", ms=12, mew=2, label="primary")
    if sec is not None:
        ax.plot(sec[0], sec[1], "rx", ms=10, mew=2, label="secondary")
    ax.set_title("Homomorphic + spots")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = out_dir / f"homo_algo_demo_pair{pi:03d}.png"
    try:
        import io

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
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=20.0, help="短曝光抓點門檻％")
    ap.add_argument(
        "--noise-floor",
        type=float,
        default=_DEFAULT_NOISE_FLOOR,
        help="低於此視為雜訊→0（對應 color scale black below，預設 256）",
    )
    ap.add_argument(
        "--white",
        type=float,
        default=None,
        help="可選：white above 硬裁上界（預設不做，只清雜訊）",
    )
    ap.add_argument("--d0", type=float, default=40.0)
    ap.add_argument("--gamma-l", type=float, default=0.5)
    ap.add_argument("--gamma-h", type=float, default=2.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--sec-thresh", type=float, default=0.35, help="下光斑相對峰門檻")
    ap.add_argument("--half-x", type=int, default=80, help="下光斑搜尋半寬 px")
    ap.add_argument("--min-sep-y", type=int, default=40, help="與上光斑最小 Y 間距")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (_BASLER_ROOT / data_dir).resolve()
    out_dir = data_dir / "homomorphic_spot_algo_trial"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pair = _load_pairs(data_dir)
    rows = []
    for pi in range(1, int(args.pairs) + 1):
        if pi not in by_pair or "short" not in by_pair[pi] or "long" not in by_pair[pi]:
            continue
        sm = _load_npy(data_dir, by_pair[pi]["short"])
        lm = _load_npy(data_dir, by_pair[pi]["long"])
        # 先清雜訊：< noise_floor → 0，再抓短曝光／同態
        sm_cs = apply_noise_floor(sm, args.noise_floor, args.white)
        lm_cs = apply_noise_floor(lm, args.noise_floor, args.white)
        ss = _short_spot(sm_cs, args.thresh)
        if ss is None:
            continue

        result = find_dual_spots_homomorphic(
            lm_cs,
            ss,
            d0=args.d0,
            gamma_l=args.gamma_l,
            gamma_h=args.gamma_h,
            c=args.c,
            thresh_frac=args.sec_thresh,
            search_half_x=args.half_x,
            min_sep_y=args.min_sep_y,
        )
        pri = result["primary"]
        sec = result["secondary"]
        row = {
            "pair": pi,
            "noise_floor": float(args.noise_floor),
            "white_above": args.white,
            "short_x": float(ss[0]),
            "short_y": float(ss[1]),
            "pri_x": None if pri is None else float(pri[0]),
            "pri_y": None if pri is None else float(pri[1]),
            "sec_x": None if sec is None else float(sec[0]),
            "sec_y": None if sec is None else float(sec[1]),
            "dx_um": None,
            "dy_um": None,
            "dist_um": None,
            "amp_ratio": result.get("amp_ratio"),
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
        if pi <= 3:
            save_demo(
                pi,
                lm_cs,
                result,
                ss,
                out_dir,
                d0=args.d0,
                gl=args.gamma_l,
                gh=args.gamma_h,
                noise_floor=args.noise_floor,
            )
        if pi % 25 == 0:
            print(f"  progress {pi}/{args.pairs}")

    csv_path = out_dir / f"homo_algo_pairs_001_{int(args.pairs):03d}.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})

    ok = [r for r in rows if r["ok"]]
    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(
        f"noise floor: <{args.noise_floor:g} → 0"
        + (
            f"  | white_above={args.white:g}"
            if args.white is not None
            else "  | white_above=off"
        )
    )
    print(
        f"params: D0={args.d0} gL={args.gamma_l} gH={args.gamma_h} "
        f"sec_thresh={args.sec_thresh} half_x={args.half_x} min_sep_y={args.min_sep_y}"
    )
    print(f"success: {len(ok)}/{len(rows)}")
    print("--- Homomorphic standalone dist_um (short→secondary) ---")
    print(_stats([r["dist_um"] for r in ok]))
    ratios = [r["amp_ratio"] for r in ok if r["amp_ratio"] is not None]
    if ratios:
        print(
            f"amp_ratio primary/secondary on filtered: "
            f"median={np.median(ratios):.2f}x  mean={np.mean(ratios):.2f}x"
        )
    print(f"csv: {csv_path.name}")
    print("demo: homo_algo_demo_pair001.png (+002,003)")


if __name__ == "__main__":
    main()
