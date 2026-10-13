# -*- coding: utf-8 -*-
"""測試：用 log10(sum+1) 剖面找下光斑 Y，與現行線性剖面對照。

用法（在 basler/ 目錄）:
  python test/try_log_secondary_y.py
  python test/try_log_secondary_y.py --pairs 100 --data "test/700 um"
"""
from __future__ import annotations

import argparse
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


def _short_spot(mat: np.ndarray, thresh: float):
    return compute_auto_spot_center(
        mat,
        "centroid",
        use_threshold=True,
        thresh_percent=float(thresh),
        bg_subtract=True,
        largest_cc_only=True,
        subpixel=True,
    )


def _strip_profile(long_mat, short_xy, *, roi_w, roi_h):
    xywh = sc._roi_centered_on_xy(long_mat, short_xy, roi_w=roi_w, roi_h=roi_h)
    bounds = clip_roi_to_matrix(long_mat, xywh)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    work = long_mat[y0:y1, x0:x1]
    raw = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
    bg = float(np.median(raw)) if raw.size else 0.0
    profile = np.clip(raw - bg, 0.0, None)
    return {
        "work": work,
        "bounds": bounds,
        "y0": float(y0),
        "profile": profile,
        "raw_sum": raw,
    }


def _primary_y_from_profile(profile, short_xy, y0, smooth_win):
    peaks = sc._peaks_by_derivative_1d(
        profile, smooth_win=smooth_win, min_prominence_frac=0.04
    )
    hint = float(short_xy[1]) - float(y0)
    return sc._primary_peak_from_hint(peaks, profile, hint)


def _secondary_on_profile(profile, primary_y, *, smooth_win, min_sep):
    """現行次峰搜尋（可對線性或 log 剖面呼叫）。"""
    return sc._find_secondary_peak_away_from_primary(
        profile,
        primary_y,
        smooth_win=smooth_win,
        min_sep=min_sep,
        min_prominence_frac=0.02,
    )


def _refine(work, profile_lin, primary_y, secondary_y, axis_x):
    return sc._refine_primary_secondary_spots(
        work,
        profile_lin,
        primary_y,
        secondary_y,
        axis_x=axis_x,
    )


def _px_to_um(px: float) -> float:
    return float(px) * 3.45


def run_pair(data_dir, by_pair, pi, *, thresh, smooth, roi_w, roi_h, min_sep):
    ps = by_pair[pi]
    sm = _load_npy(data_dir, ps["short"])
    lm = _load_npy(data_dir, ps["long"])
    ss = _short_spot(sm, thresh)
    if ss is None:
        return None

    strip = _strip_profile(lm, ss, roi_w=roi_w, roi_h=roi_h)
    if strip is None:
        return None
    profile = strip["profile"]
    y0 = strip["y0"]
    work = strip["work"]
    axis_x = float(ss[0]) - float(strip["bounds"][0])

    # --- A: 現行完整管線（線性）---
    r_lin = find_dual_spots_axial_1d_projection(
        lm,
        ss,
        roi_w=roi_w,
        roi_h=roi_h,
        smooth_win=smooth,
        min_peak_distance=min_sep,
    )
    spots = r_lin.get("spots")
    if not spots or len(spots) < 2:
        sec_lin = None
        seed_lin = None
    else:
        pidx = int(r_lin.get("primary_index", 0) or 0)
        sec_lin = spots[1 - pidx]
        # seed from linear path for reference
        py = _primary_y_from_profile(profile, ss, y0, smooth)
        seed_lin = _secondary_on_profile(
            profile, py, smooth_win=smooth, min_sep=min_sep
        )

    # --- B: log 剖面定次峰 Y，再用線性影像做 2D 精修 ---
    log_prof = np.log10(profile + 1.0)
    py = _primary_y_from_profile(profile, ss, y0, smooth)  # 主峰仍用線性（較穩）
    if py is None:
        return None
    seed_log = _secondary_on_profile(
        log_prof, py, smooth_win=smooth, min_sep=min_sep
    )
    sec_log = None
    if seed_log is not None:
        _pri, c_sec, _v = _refine(work, profile, py, seed_log, axis_x)
        sec_log = (
            float(c_sec[0] + strip["bounds"][0]),
            float(c_sec[1] + y0),
        )

    def _delta(sec):
        if sec is None:
            return None, None, None
        dx = float(sec[0] - ss[0])
        dy = float(sec[1] - ss[1])
        dist = float(np.hypot(dx, dy))
        return dx, dy, dist

    dx_l, dy_l, dist_l = _delta(sec_lin)
    dx_g, dy_g, dist_g = _delta(sec_log)

    return {
        "pair": pi,
        "short_x": float(ss[0]),
        "short_y": float(ss[1]),
        "primary_y_local": float(py),
        "seed_lin_y_full": (float(seed_lin) + y0) if seed_lin is not None else None,
        "seed_log_y_full": (float(seed_log) + y0) if seed_log is not None else None,
        "sec_lin_x": None if sec_lin is None else float(sec_lin[0]),
        "sec_lin_y": None if sec_lin is None else float(sec_lin[1]),
        "sec_log_x": None if sec_log is None else float(sec_log[0]),
        "sec_log_y": None if sec_log is None else float(sec_log[1]),
        "dist_lin_um": None if dist_l is None else _px_to_um(dist_l),
        "dist_log_um": None if dist_g is None else _px_to_um(dist_g),
        "dy_lin_um": None if dy_l is None else _px_to_um(dy_l),
        "dy_log_um": None if dy_g is None else _px_to_um(dy_g),
        "_strip": strip,
        "_profile": profile,
        "_log": log_prof,
        "_py": py,
        "_seed_lin": seed_lin,
        "_seed_log": seed_log,
        "_ss": ss,
    }


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
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


def save_demo_plot(row, out_dir: Path, smooth: int):
    strip = row["_strip"]
    profile = row["_profile"]
    log_prof = row["_log"]
    y0 = strip["y0"]
    y_full = np.arange(len(profile)) + y0
    smooth_lin = sc._smooth_1d_profile(profile, smooth)
    smooth_log = sc._smooth_1d_profile(log_prof, smooth)
    extra = max(15, smooth * 2 + 1)
    extra_lin = sc._smooth_1d_profile(smooth_lin, extra)
    extra_log = sc._smooth_1d_profile(smooth_log, extra)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Pair {row['pair']} linear vs log secondary | smooth={smooth} extra={extra}",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(y_full, profile, color="#90A4AE", lw=0.8, label="sum raw")
    ax.plot(y_full, smooth_lin, color="#1565C0", lw=1.2, label=f"smooth {smooth}")
    ax.plot(y_full, extra_lin, color="#E65100", lw=1.4, label=f"extra {extra}")
    if row["_seed_lin"] is not None:
        ax.axvline(
            row["_seed_lin"] + y0,
            color="#C62828",
            ls="--",
            label=f"lin seed Y={row['_seed_lin']+y0:.1f}",
        )
    ax.set_title("Linear profile (full)")
    ax.set_ylabel("sum")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(y_full, log_prof, color="#90A4AE", lw=0.8, label="log10(sum+1)")
    ax.plot(y_full, smooth_log, color="#1565C0", lw=1.2, label=f"smooth {smooth}")
    ax.plot(y_full, extra_log, color="#E65100", lw=1.4, label=f"extra {extra}")
    if row["_seed_log"] is not None:
        ax.axvline(
            row["_seed_log"] + y0,
            color="#C62828",
            ls="--",
            label=f"log seed Y={row['_seed_log']+y0:.1f}",
        )
    if row["_py"] is not None:
        ax.axvline(row["_py"] + y0, color="#2E7D32", ls=":", label="primary")
    ax.set_title("Log profile (full)")
    ax.set_ylabel("log10")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # zoom lower region
    z0, z1 = 280, 420
    m = (y_full >= z0) & (y_full <= z1)
    ax = axes[1, 0]
    ax.plot(y_full[m], profile[m], color="#90A4AE", lw=0.9, label="raw")
    ax.plot(y_full[m], smooth_lin[m], color="#1565C0", lw=1.2, label=f"smooth {smooth}")
    ax.plot(y_full[m], extra_lin[m], color="#E65100", lw=1.6, label=f"extra {extra}")
    if row["_seed_lin"] is not None:
        ax.axvline(row["_seed_lin"] + y0, color="#C62828", ls="--")
    ax.set_xlim(z0, z1)
    ax.set_title("Linear zoom (lower spot)")
    ax.set_xlabel("Y full px")
    ax.set_ylabel("sum")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(y_full[m], log_prof[m], color="#90A4AE", lw=0.9, label="log raw")
    ax.plot(y_full[m], smooth_log[m], color="#1565C0", lw=1.2, label=f"smooth {smooth}")
    ax.plot(y_full[m], extra_log[m], color="#E65100", lw=1.6, label=f"extra {extra}")
    if row["_seed_log"] is not None:
        ax.axvline(row["_seed_log"] + y0, color="#C62828", ls="--")
    ax.set_xlim(z0, z1)
    ax.set_title("Log zoom (lower spot) — weak lobe more visible")
    ax.set_xlabel("Y full px")
    ax.set_ylabel("log10")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"log_vs_linear_demo_pair{row['pair']:03d}.png"
    fig.savefig(path, dpi=150)
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
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    basler = _BASLER_ROOT
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (basler / data_dir).resolve()
    out_dir = data_dir / "axial_1d_log_trial"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pair = _load_pairs(data_dir)
    rows = []
    for pi in range(1, int(args.pairs) + 1):
        if pi not in by_pair or "short" not in by_pair[pi] or "long" not in by_pair[pi]:
            continue
        row = run_pair(
            data_dir,
            by_pair,
            pi,
            thresh=args.thresh,
            smooth=args.smooth,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            min_sep=args.min_sep,
        )
        if row is None:
            continue
        rows.append(row)
        if pi <= 3:
            save_demo_plot(row, out_dir, args.smooth)

    # summary CSV (no private caches)
    csv_path = out_dir / f"log_vs_linear_pairs_001_{int(args.pairs):03d}.csv"
    keys = [
        "pair",
        "short_x",
        "short_y",
        "primary_y_local",
        "seed_lin_y_full",
        "seed_log_y_full",
        "sec_lin_x",
        "sec_lin_y",
        "sec_log_x",
        "sec_log_y",
        "dist_lin_um",
        "dist_log_um",
        "dy_lin_um",
        "dy_log_um",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    "" if r.get(k) is None else f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                    for k in keys
                )
                + "\n"
            )

    dist_lin = [r["dist_lin_um"] for r in rows if r["dist_lin_um"] is not None]
    dist_log = [r["dist_log_um"] for r in rows if r["dist_log_um"] is not None]
    seed_lin = [r["seed_lin_y_full"] for r in rows if r["seed_lin_y_full"] is not None]
    seed_log = [r["seed_log_y_full"] for r in rows if r["seed_log_y_full"] is not None]

    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(f"pairs ok: {len(rows)} / requested {args.pairs}")
    print("--- linear (current) dist_um ---")
    print(_stats(dist_lin))
    print("--- log-secondary dist_um ---")
    print(_stats(dist_log))
    print("--- seed Y std (lin / log) ---")
    sl = np.asarray(seed_lin, float)
    sg = np.asarray(seed_log, float)
    print(
        f"lin seed Y std={sl.std():.3f}  log seed Y std={sg.std():.3f}"
        if sl.size and sg.size
        else "n/a"
    )
    if sl.size and sg.size and sl.size == sg.size:
        print(f"mean |seed_log - seed_lin| = {np.mean(np.abs(sg - sl)):.3f} px")
    print(f"csv: {csv_path.name}")
    print("demo png: log_vs_linear_demo_pair001.png (+002,003)")


if __name__ == "__main__":
    main()
