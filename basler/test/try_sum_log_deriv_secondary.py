# -*- coding: utf-8 -*-
"""測試：sum / log / 微分 三種下光斑 Y 定法（各自條件），再共用 2D 亮核精修 X。

用法（在 basler/）:
  python test/try_sum_log_deriv_secondary.py
  python test/try_sum_log_deriv_secondary.py --pairs 100 --data "test/700 um"
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
from shared_components import clip_roi_to_matrix, compute_auto_spot_center


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------
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


def _strip(long_mat, short_xy, *, roi_w, roi_h):
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
        "x0": float(x0),
        "y0": float(y0),
        "sum": profile,
        "log": np.log10(profile + 1.0),
    }


def _px_um(px):
    return float(px) * 3.45


# ---------------------------------------------------------------------------
# 共用：主峰（短曝光 hint，在指定剖面上）
# ---------------------------------------------------------------------------
def _primary_y(profile, short_xy, y0, *, smooth_win, prom_frac):
    peaks = sc._peaks_by_derivative_1d(
        profile, smooth_win=smooth_win, min_prominence_frac=prom_frac
    )
    hint = float(short_xy[1]) - float(y0)
    return sc._primary_peak_from_hint(peaks, profile, hint)


def _exclude_width(smooth, primary_y, min_sep):
    n = len(smooth)
    pi = int(np.clip(round(float(primary_y)), 0, n - 1))
    core_w = max(6.0, sc._peak_width_at_frac_1d(smooth, pi, frac=0.65))
    return int(np.clip(max(float(min_sep), 1.05 * core_w), 8, max(12, n // 5))), core_w


def _refine_full(strip, short_xy, primary_y, secondary_y, *, smooth_win):
    """用線性 sum 剖面 + 2D 亮核精修，回傳全圖 (x,y)。"""
    if secondary_y is None or primary_y is None:
        return None
    axis_x = float(short_xy[0]) - float(strip["x0"])
    # refine 內部會再平滑 profile；給線性 sum
    _pri, c_sec, _v = sc._refine_primary_secondary_spots(
        strip["work"],
        strip["sum"],
        primary_y,
        secondary_y,
        axis_x=axis_x,
    )
    return (
        float(c_sec[0] + strip["x0"]),
        float(c_sec[1] + strip["y0"]),
    )


# ---------------------------------------------------------------------------
# 方法 A：SUM — 現行條件（線性振幅、波谷後瓣質心、加強平滑）
# ---------------------------------------------------------------------------
def find_secondary_sum(profile_sum, primary_y, *, smooth_win, min_sep):
    """條件：線性剖面 + 現行 _find_secondary_peak_away_from_primary。"""
    return sc._find_secondary_peak_away_from_primary(
        profile_sum,
        primary_y,
        smooth_win=smooth_win,
        min_sep=min_sep,
        min_prominence_frac=0.02,  # 相對主峰振幅（線性）
    )


# ---------------------------------------------------------------------------
# 方法 B：LOG — log 空間專用條件
# ---------------------------------------------------------------------------
def find_secondary_log(profile_log, primary_y, *, smooth_win, min_sep):
    """條件（log 專用）:
    - 剖面 = log10(sum+1)
    - 排除帶：用 log 剖面的 0.65 峰寬（與 sum 分開算）
    - 波谷：log 加強平滑後，找 exclude 後第一個「谷→上升 ≥ min_log_rise」
    - 次峰：波谷以下 log 瓣質心（extra_smooth）；不要求相對主峰振幅比
    - min_log_rise: 絕對 log 上升（預設 0.12 ≈ ×1.32），適弱峰
    """
    y = np.asarray(profile_log, dtype=np.float64)
    n = int(y.size)
    if n < 8 or primary_y is None:
        return None
    base_win = int(smooth_win) if smooth_win else 7
    smooth = sc._smooth_1d_profile(y, base_win)
    exclude, _core = _exclude_width(smooth, primary_y, min_sep)
    py = float(primary_y)
    pi = int(np.clip(round(py), 0, n - 1))
    primary_amp = float(smooth[pi])
    extra = max(11, base_win * 2 - 1)  # log 動態範圍已壓過，加強窗略小於線性
    extra = extra if (extra % 2) else extra + 1
    blur = sc._smooth_1d_profile(smooth, extra)

    min_log_rise = 0.12  # 絕對條件，不跟主峰比
    start = int(np.clip(pi + exclude + 1, 0, n - 1))
    valley = None
    for j in range(max(start, 1), n - 2):
        if not (
            float(blur[j]) <= float(blur[j - 1])
            and float(blur[j]) <= float(blur[j + 1])
        ):
            continue
        k = j + 1
        while k < n - 1 and float(blur[k + 1]) >= float(blur[k]):
            k += 1
        rise = float(blur[k]) - float(blur[j])
        if rise < min_log_rise:
            continue
        if (k - j) < 3:
            continue
        valley = int(j)
        break
    if valley is None:
        mid = min(n, start + max(40, (n - start) // 2))
        if mid - start < 3:
            return None
        valley = int(start + int(np.argmin(blur[start:mid])))

    # blur 已是加強平滑：直接在波谷以下做瓣質心（不再二次平滑）
    seg = blur[valley + 1 :]
    if seg.size < 3 or float(np.max(seg)) <= 0:
        return None
    j = int(np.argmax(seg))
    peak = float(seg[j])
    thr = 0.30 * peak
    left = j
    while left > 0 and float(seg[left - 1]) >= thr:
        left -= 1
    right = j
    while right < seg.size - 1 and float(seg[right + 1]) >= thr:
        right += 1
    ww = np.asarray(seg[left : right + 1], dtype=np.float64)
    ww = np.where(ww >= thr, ww, 0.0)
    wsum = float(np.sum(ww))
    if wsum <= 1e-12:
        return float(valley + 1 + j)
    xs = np.arange(left, right + 1, dtype=np.float64)
    return float(valley + 1) + float(np.sum(xs * ww) / wsum)


# ---------------------------------------------------------------------------
# 方法 C：微分 — 零交越專用條件
# ---------------------------------------------------------------------------
def _deriv_peaks(smooth, *, min_amp=None, min_prom_abs=None):
    """dy 正→負零交越；振幅門檻用絕對值（或相對），回傳 (y, amp, sharpness)。"""
    dy = np.gradient(smooth)
    n = len(smooth)
    margin = max(2, n // 50)
    peaks = []
    for i in range(margin, n - 1 - margin):
        if dy[i] >= 0.0 and dy[i + 1] < 0.0:
            denom = float(dy[i] - dy[i + 1])
            frac = float(dy[i] / denom) if abs(denom) > 1e-15 else 0.5
            pos = float(i) + frac
            amp = float(np.interp(pos, np.arange(n, dtype=np.float64), smooth))
            sharp = float(dy[i] - dy[i + 1])  # 交越陡峭度
            if min_amp is not None and amp < min_amp:
                continue
            if min_prom_abs is not None:
                # 相對鄰近谷的突出度
                lo = max(0, int(pos) - 30)
                hi = min(n, int(pos) + 31)
                valley = float(np.min(smooth[lo:hi]))
                if (amp - valley) < min_prom_abs:
                    continue
            peaks.append((pos, amp, sharp))
    peaks.sort(key=lambda t: t[1], reverse=True)
    return peaks


def find_secondary_deriv(profile_sum, primary_y, *, smooth_win, min_sep):
    """條件（微分專用）:
    - 在線性 sum 平滑後取 dy 零交越峰
    - 主峰：由呼叫端給定（短曝光 hint）
    - 次峰：只取 primary+exclude 下方的零交越
    - 評分：突出度(amp−local_valley) × 交越陡峭度；不要求 amp/primary 比
    - 若多峰挤在 ±12 px，改取權重質心（壓散斑）
    """
    y = np.asarray(profile_sum, dtype=np.float64)
    n = int(y.size)
    if n < 8 or primary_y is None:
        return None
    base_win = int(smooth_win) if smooth_win else 7
    smooth = sc._smooth_1d_profile(y, base_win)
    exclude, core_w = _exclude_width(smooth, primary_y, min_sep)
    py = float(primary_y)
    y_down = py + exclude

    # 弱峰：絕對突出度用主峰的小比例下限，但有 floor
    primary_amp = float(smooth[int(np.clip(round(py), 0, n - 1))])
    min_prom = max(primary_amp * 0.003, float(np.median(smooth)) * 0.5, 200.0)

    peaks = _deriv_peaks(smooth, min_prom_abs=min_prom)
    cands = []
    for pos, amp, sharp in peaks:
        if pos < y_down:
            continue
        lo = int(np.clip(np.floor(min(pos, py)), 0, n - 1))
        hi = int(np.clip(np.ceil(max(pos, py)), lo, n - 1))
        valley = float(np.min(smooth[lo : hi + 1])) if hi > lo else 0.0
        prom = amp - valley
        if prom < min_prom:
            continue
        # 不懲罰遠距弱峰的 amp/primary；用突出度×陡峭
        score = prom * max(sharp, 1e-6)
        cands.append((score, pos, amp, prom, sharp))

    if not cands:
        # 後備：波谷後瓣心（與 sum 類似但用較低 lobe）
        return sc._secondary_y_after_valley(
            smooth,
            py,
            exclude,
            primary_amp=primary_amp,
            min_amp_frac=0.002,
            extra_smooth=max(15, base_win * 2 + 1),
        )

    cands.sort(key=lambda t: t[0], reverse=True)
    # 散斑：前幾名若擠在一起 → 權重質心
    top = [c for c in cands if c[0] >= 0.5 * cands[0][0]]
    ys = np.array([c[1] for c in top], dtype=float)
    ws = np.array([c[3] for c in top], dtype=float)  # prom weights
    if len(top) >= 2 and (ys.max() - ys.min()) <= max(12.0, 0.25 * core_w):
        return float(np.sum(ys * ws) / max(float(np.sum(ws)), 1e-12))
    return float(cands[0][1])


# ---------------------------------------------------------------------------
# 跑一組
# ---------------------------------------------------------------------------
def run_pair(data_dir, by_pair, pi, *, thresh, smooth, roi_w, roi_h, min_sep):
    ps = by_pair[pi]
    sm = _load_npy(data_dir, ps["short"])
    lm = _load_npy(data_dir, ps["long"])
    ss = _short_spot(sm, thresh)
    if ss is None:
        return None
    strip = _strip(lm, ss, roi_w=roi_w, roi_h=roi_h)
    if strip is None:
        return None

    # 主峰：三種方法都用「線性 sum + 短曝光 hint」（主峰夠亮，log/微分不必另定）
    py = _primary_y(
        strip["sum"], ss, strip["y0"], smooth_win=smooth, prom_frac=0.04
    )
    if py is None:
        return None

    seed_sum = find_secondary_sum(
        strip["sum"], py, smooth_win=smooth, min_sep=min_sep
    )
    seed_log = find_secondary_log(
        strip["log"], py, smooth_win=smooth, min_sep=min_sep
    )
    seed_der = find_secondary_deriv(
        strip["sum"], py, smooth_win=smooth, min_sep=min_sep
    )

    out = {
        "pair": pi,
        "short_x": float(ss[0]),
        "short_y": float(ss[1]),
        "primary_y_local": float(py),
        "primary_y_full": float(py) + float(strip["y0"]),
    }
    for name, seed in (
        ("sum", seed_sum),
        ("log", seed_log),
        ("deriv", seed_der),
    ):
        out[f"seed_{name}_y_full"] = (
            None if seed is None else float(seed) + float(strip["y0"])
        )
        sec = _refine_full(strip, ss, py, seed, smooth_win=smooth)
        if sec is None:
            out[f"sec_{name}_x"] = None
            out[f"sec_{name}_y"] = None
            out[f"dist_{name}_um"] = None
            out[f"dy_{name}_um"] = None
        else:
            out[f"sec_{name}_x"] = sec[0]
            out[f"sec_{name}_y"] = sec[1]
            dx = sec[0] - ss[0]
            dy = sec[1] - ss[1]
            out[f"dist_{name}_um"] = _px_um(np.hypot(dx, dy))
            out[f"dy_{name}_um"] = _px_um(dy)

    out["_cache"] = {
        "strip": strip,
        "ss": ss,
        "py": py,
        "seed_sum": seed_sum,
        "seed_log": seed_log,
        "seed_der": seed_der,
    }
    return out


def _stats(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
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


def save_demo(row, out_dir: Path, smooth: int):
    c = row["_cache"]
    strip = c["strip"]
    y0 = strip["y0"]
    y_full = np.arange(len(strip["sum"])) + y0
    s_sum = sc._smooth_1d_profile(strip["sum"], smooth)
    s_log = sc._smooth_1d_profile(strip["log"], smooth)
    dy = np.gradient(s_sum)
    extra = max(15, smooth * 2 + 1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f"Pair {row['pair']} | sum / log / deriv (own conditions) | smooth={smooth}",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(y_full, strip["sum"], color="#B0BEC5", lw=0.7, label="sum raw")
    ax.plot(y_full, s_sum, color="#1565C0", lw=1.2, label=f"sum smooth {smooth}")
    ax.axvline(c["py"] + y0, color="#2E7D32", ls=":", label="primary")
    if c["seed_sum"] is not None:
        ax.axvline(
            c["seed_sum"] + y0,
            color="#C62828",
            ls="--",
            label=f"SUM seed {c['seed_sum']+y0:.1f}",
        )
    ax.set_ylabel("sum")
    ax.set_title("SUM method")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(y_full, strip["log"], color="#B0BEC5", lw=0.7, label="log raw")
    ax.plot(y_full, s_log, color="#6A1B9A", lw=1.2, label=f"log smooth {smooth}")
    ax.axvline(c["py"] + y0, color="#2E7D32", ls=":")
    if c["seed_log"] is not None:
        ax.axvline(
            c["seed_log"] + y0,
            color="#EF6C00",
            ls="--",
            label=f"LOG seed {c['seed_log']+y0:.1f}",
        )
    ax.set_ylabel("log10(sum+1)")
    ax.set_title("LOG method (abs rise / lobe on log)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(y_full, dy, color="#00838F", lw=1.0, label="d/dy sum-smooth")
    ax.axhline(0, color="#424242", lw=0.8)
    ax.axvline(c["py"] + y0, color="#2E7D32", ls=":")
    if c["seed_der"] is not None:
        ax.axvline(
            c["seed_der"] + y0,
            color="#AD1457",
            ls="--",
            label=f"DERIV seed {c['seed_der']+y0:.1f}",
        )
    ax.set_ylabel("derivative")
    ax.set_xlabel("Y full px")
    ax.set_title("DERIV method (zero-cross, prom x sharpness)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"sum_log_deriv_demo_pair{row['pair']:03d}.png"
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
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (_BASLER_ROOT / data_dir).resolve()
    out_dir = data_dir / "axial_1d_sum_log_deriv_trial"
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
            save_demo(row, out_dir, args.smooth)

    keys = [
        "pair",
        "short_x",
        "short_y",
        "primary_y_full",
        "seed_sum_y_full",
        "seed_log_y_full",
        "seed_deriv_y_full",
        "sec_sum_x",
        "sec_sum_y",
        "sec_log_x",
        "sec_log_y",
        "sec_deriv_x",
        "sec_deriv_y",
        "dist_sum_um",
        "dist_log_um",
        "dist_deriv_um",
        "dy_sum_um",
        "dy_log_um",
        "dy_deriv_um",
    ]
    csv_path = out_dir / f"sum_log_deriv_pairs_001_{int(args.pairs):03d}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(f"pairs ok: {len(rows)}")
    for name in ("sum", "log", "deriv"):
        dist = [r[f"dist_{name}_um"] for r in rows]
        seed = [r[f"seed_{name}_y_full"] for r in rows]
        print(f"--- {name.upper()} dist_um ---")
        print(_stats(dist))
        ss = np.asarray([s for s in seed if s is not None], float)
        if ss.size:
            print(f"    seed Y std = {ss.std():.3f} px")
    print(f"csv: {csv_path.name}")
    print("demo: sum_log_deriv_demo_pair001.png (+002,003)")


if __name__ == "__main__":
    main()
