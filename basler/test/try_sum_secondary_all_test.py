# -*- coding: utf-8 -*-
"""用現行 SUM 方法抓下光斑，跑完 basler/test 內所有可用短／長曝光配對。

資料來源:
  - test/700 um/          Basler_*.json + .npy（pair_index + exposure_role）
  - test/sum 10/          sum_odd_*_short.npy + sum_even_*_long.npy
  - test/*.npy            「700 um-2 ..._NNNN_...」奇數=short、偶數=long

用法（在 basler/）:
  python test/try_sum_secondary_all_test.py
  python test/try_sum_secondary_all_test.py --thresh 20 --smooth 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_BASLER_ROOT = Path(__file__).resolve().parent.parent
if str(_BASLER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASLER_ROOT))

import numpy as np

from shared_components import (
    compute_auto_spot_center,
    find_dual_spots_axial_1d_projection,
)

PX_TO_UM = 3.45


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


def _run_one(short_mat, long_mat, *, thresh, smooth, roi_w, roi_h, min_sep):
    ss = _short_spot(np.asarray(short_mat, dtype=float), thresh)
    if ss is None:
        return {"ok": False, "reason": "short_spot_fail"}
    result = find_dual_spots_axial_1d_projection(
        np.asarray(long_mat, dtype=float),
        ss,
        roi_w=roi_w,
        roi_h=roi_h,
        smooth_win=smooth,
        min_peak_distance=min_sep,
    )
    spots = result.get("spots")
    if not spots or len(spots) < 2:
        return {
            "ok": False,
            "reason": "no_dual_peak",
            "short_x": float(ss[0]),
            "short_y": float(ss[1]),
            "roi_xywh": result.get("roi_xywh"),
        }
    pidx = int(result.get("primary_index", 0) or 0)
    sec = spots[1 - pidx]
    pri = spots[pidx]
    dx = float(sec[0] - ss[0])
    dy = float(sec[1] - ss[1])
    dist = float(np.hypot(dx, dy))
    return {
        "ok": True,
        "short_x": float(ss[0]),
        "short_y": float(ss[1]),
        "pri_x": float(pri[0]),
        "pri_y": float(pri[1]),
        "sec_x": float(sec[0]),
        "sec_y": float(sec[1]),
        "dx_px": dx,
        "dy_px": dy,
        "dist_px": dist,
        "dx_um": dx * PX_TO_UM,
        "dy_um": dy * PX_TO_UM,
        "dist_um": dist * PX_TO_UM,
        "roi_xywh": result.get("roi_xywh"),
    }


# ---------- discover datasets ----------
def discover_basler_json_pairs(folder: Path):
    """700 um 風格：json 含 pair_index / exposure_role。"""
    files = sorted(folder.glob("Basler_*.json"))
    by_pair = {}
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        pi = int(d["pair_index"])
        role = d.get("exposure_role")
        npy = folder / p.name.replace(".json", ".npy")
        if not npy.exists():
            continue
        by_pair.setdefault(pi, {})[role] = npy
    pairs = []
    for pi in sorted(by_pair):
        if "short" in by_pair[pi] and "long" in by_pair[pi]:
            pairs.append(
                {
                    "dataset": folder.name,
                    "pair_id": str(pi),
                    "short_path": by_pair[pi]["short"],
                    "long_path": by_pair[pi]["long"],
                }
            )
    return pairs


def discover_sum10_pairs(folder: Path):
    shorts = sorted(folder.glob("*short*.npy"))
    longs = sorted(folder.glob("*long*.npy"))
    pairs = []
    # 若各一張：配成一對；若多張依排序 zip
    n = min(len(shorts), len(longs))
    for i in range(n):
        pairs.append(
            {
                "dataset": folder.name,
                "pair_id": f"{i+1}",
                "short_path": shorts[i],
                "long_path": longs[i],
            }
        )
    return pairs


def discover_root_numbered_npy(test_dir: Path):
    """根目錄 700 um-2 ..._NNNN_...npy：奇數 short、偶數 long。"""
    files = sorted(test_dir.glob("*.npy"))
    indexed = []
    for p in files:
        m = re.search(r"_(\d{4})_", p.name)
        if not m:
            continue
        indexed.append((int(m.group(1)), p))
    indexed.sort()
    by_i = {i: p for i, p in indexed}
    pairs = []
    for i in sorted(by_i):
        if i % 2 == 1 and (i + 1) in by_i:
            pairs.append(
                {
                    "dataset": "root_700um2_npy",
                    "pair_id": f"{i:04d}_{i+1:04d}",
                    "short_path": by_i[i],
                    "long_path": by_i[i + 1],
                }
            )
    return pairs


def discover_all(test_dir: Path):
    all_pairs = []
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        # skip output folders
        if "trial" in sub.name.lower() or "export" in sub.name.lower():
            continue
        if list(sub.glob("Basler_*.json")):
            all_pairs.extend(discover_basler_json_pairs(sub))
        elif list(sub.glob("*.npy")):
            all_pairs.extend(discover_sum10_pairs(sub))
    all_pairs.extend(discover_root_numbered_npy(test_dir))
    return all_pairs


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="test")
    ap.add_argument("--thresh", type=float, default=20.0)
    ap.add_argument("--smooth", type=int, default=8)
    ap.add_argument("--roi-w", type=int, default=160)
    ap.add_argument("--roi-h", type=int, default=512)
    ap.add_argument("--min-sep", type=int, default=12)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = (_BASLER_ROOT / test_dir).resolve()
    out_dir = test_dir / "axial_1d_sum_all_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_all(test_dir)
    print(f"test_dir: {test_dir}")
    print(f"discovered pairs: {len(pairs)}")
    from collections import Counter

    print("by dataset:", dict(Counter(p["dataset"] for p in pairs)))

    rows = []
    fail = 0
    for i, p in enumerate(pairs, 1):
        sm = np.load(str(p["short_path"]))
        lm = np.load(str(p["long_path"]))
        r = _run_one(
            sm,
            lm,
            thresh=args.thresh,
            smooth=args.smooth,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            min_sep=args.min_sep,
        )
        row = {
            "dataset": p["dataset"],
            "pair_id": p["pair_id"],
            "short_file": p["short_path"].name,
            "long_file": p["long_path"].name,
            "ok": bool(r.get("ok")),
            "reason": r.get("reason", ""),
            "short_x": r.get("short_x"),
            "short_y": r.get("short_y"),
            "sec_x": r.get("sec_x"),
            "sec_y": r.get("sec_y"),
            "dx_px": r.get("dx_px"),
            "dy_px": r.get("dy_px"),
            "dist_px": r.get("dist_px"),
            "dx_um": r.get("dx_um"),
            "dy_um": r.get("dy_um"),
            "dist_um": r.get("dist_um"),
        }
        rows.append(row)
        if not row["ok"]:
            fail += 1
        if i % 50 == 0 or i == len(pairs):
            print(f"  progress {i}/{len(pairs)} fail={fail}")

    csv_path = out_dir / "sum_secondary_all_pairs.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: ("" if row[k] is None else row[k])
                    for k in keys
                }
            )

    # per-dataset summary
    summary_path = out_dir / "sum_secondary_summary_by_dataset.csv"
    datasets = sorted({r["dataset"] for r in rows})
    sum_rows = []
    print("\n=== SUM secondary results ===")
    for ds in datasets:
        ok_rows = [r for r in rows if r["dataset"] == ds and r["ok"]]
        dists = [r["dist_um"] for r in ok_rows]
        st = _stats(dists)
        print(f"[{ds}] ok={len(ok_rows)}/{sum(1 for r in rows if r['dataset']==ds)}  dist_um {st}")
        sum_rows.append({"dataset": ds, **{f"dist_{k}": v for k, v in st.items()}})

    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        if sum_rows:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader()
            w.writerows(sum_rows)

    print(f"\ncsv: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"params: thresh={args.thresh}% smooth={args.smooth} roi={args.roi_w}x{args.roi_h}")


if __name__ == "__main__":
    main()
