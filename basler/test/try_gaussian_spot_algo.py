# -*- coding: utf-8 -*-
"""獨立測試：高斯濾波後抓雙光斑（不接一維投影）。

流程:
  0. < noise_floor（預設 256）→ 0
  1. 短曝光質心／門檻 → 上光斑
  2. 長曝光 2D 高斯平滑 → 抹主斑鄰域 → 下方帶抓下光斑

用法（在 basler/）:
  python test/try_gaussian_spot_algo.py
  python test/try_gaussian_spot_algo.py --pairs 100 --sigma 4.0 --noise-floor 256
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared_components import compute_auto_spot_center, find_dual_spots_gaussian

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


def save_demo(pi, display_long, result, short_xy, out_dir, *, sigma, noise_floor):
    pri = result["primary"]
    sec = result["secondary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Pair {pi} Gaussian spot algo | noise<{noise_floor:g}→0 | sigma={sigma}",
        fontsize=11,
    )
    ax = axes[0]
    ax.imshow(display_long, cmap="jet", aspect="equal", origin="upper")
    ax.plot(short_xy[0], short_xy[1], "w+", ms=12, mew=2, label="short=primary")
    if sec is not None:
        ax.plot(sec[0], sec[1], "rx", ms=10, mew=2, label="secondary")
    ax.set_title("Long after noise floor")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    lo, hi = np.percentile(display_long, [1, 99.5])
    ax.imshow(
        display_long, cmap="jet", aspect="equal", origin="upper", vmin=lo, vmax=hi
    )
    if pri is not None:
        ax.plot(pri[0], pri[1], "g+", ms=12, mew=2, label="primary(=short)")
    if sec is not None:
        ax.plot(sec[0], sec[1], "rx", ms=10, mew=2, label="secondary")
    ax.set_title(f"Spots (σ={sigma})")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = out_dir / f"gauss_algo_demo_pair{pi:03d}.png"
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
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=20.0)
    ap.add_argument("--noise-floor", type=float, default=_DEFAULT_NOISE_FLOOR)
    ap.add_argument("--sigma", type=float, default=4.0, help="高斯 σ（px）")
    ap.add_argument("--sec-thresh", type=float, default=0.35)
    ap.add_argument("--half-x", type=int, default=80)
    ap.add_argument("--min-sep-y", type=int, default=120, help="上光斑下方排除帶（px）")
    ap.add_argument(
        "--blank-r",
        type=float,
        default=90.0,
        help="抹掉上光斑鄰域半徑（px），避免抓到主峰下緣",
    )
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (_BASLER_ROOT / data_dir).resolve()
    out_dir = data_dir / "gaussian_spot_algo_trial"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pair = _load_pairs(data_dir)
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

        result = find_dual_spots_gaussian(
            lm,
            ss,
            sigma=args.sigma,
            thresh_frac=args.sec_thresh,
            search_half_x=args.half_x,
            min_sep_y=args.min_sep_y,
            primary_blank_r=args.blank_r,
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
                lm,
                result,
                ss,
                out_dir,
                sigma=args.sigma,
                noise_floor=args.noise_floor,
            )
        if pi % 25 == 0:
            print(f"  progress {pi}/{args.pairs}")

    csv_path = out_dir / f"gauss_algo_pairs_001_{int(args.pairs):03d}.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})

    ok = [r for r in rows if r["ok"]]
    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(f"noise floor: <{args.noise_floor:g} → 0  | gaussian sigma={args.sigma}")
    print(
        f"params: sec_thresh={args.sec_thresh} half_x={args.half_x} "
        f"min_sep_y={args.min_sep_y} blank_r={args.blank_r}"
    )
    print(f"success: {len(ok)}/{len(rows)}")
    print("--- Gaussian standalone dist_um (short→secondary) ---")
    print(_stats([r["dist_um"] for r in ok]))
    ratios = [r["amp_ratio"] for r in ok if r["amp_ratio"] is not None]
    if ratios:
        print(
            f"amp_ratio primary/secondary: "
            f"median={np.median(ratios):.1f}x  mean={np.mean(ratios):.1f}x"
        )
    print(f"csv: {csv_path.name}")
    print("demo: gauss_algo_demo_pair001.png (+002,003)")


if __name__ == "__main__":
    main()
