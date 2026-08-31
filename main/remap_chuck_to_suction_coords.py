"""Chuck 座標反轉後對齊 Suction 的 x_rel_mm / y_rel_mm 網格，數值跟著對應點走。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_chuck(chuck_path: Path) -> pd.DataFrame:
    df = pd.read_csv(chuck_path)
    rename = {str(col).replace("\ufeff", "").strip().lower(): col for col in df.columns}
    if "x_rel_mm" not in rename or "y_rel_mm" not in rename:
        raise ValueError("Chuck 檔需要 x_rel_mm、y_rel_mm 欄位。")
    value_col = rename.get("value") or rename.get("dist1")
    if value_col is None:
        raise ValueError("Chuck 檔需要 value 或 Dist1 欄位。")
    out = df[[rename["x_rel_mm"], rename["y_rel_mm"], value_col]].copy()
    out.columns = ["x_rel_mm", "y_rel_mm", "value"]
    return out


def _read_suction(suction_path: Path) -> pd.DataFrame:
    df = pd.read_csv(suction_path)
    rename = {str(col).replace("\ufeff", "").strip().lower(): col for col in df.columns}
    if "x_rel_mm" in rename and "y_rel_mm" in rename:
        x_col, y_col = rename["x_rel_mm"], rename["y_rel_mm"]
    elif "x" in rename and "y" in rename:
        x_col, y_col = rename["x"], rename["y"]
    else:
        raise ValueError("Suction 檔需要 x_rel_mm/y_rel_mm 或 x/y 欄位。")
    out = df[[x_col, y_col]].copy()
    out.columns = ["x_rel_mm", "y_rel_mm"]
    return out


def _coord_key(x: float, y: float) -> tuple[float, float]:
    return round(float(x), 6), round(float(y), 6)


def _inverse_flip(x: float, y: float, flip_x: bool, flip_y: bool) -> tuple[float, float]:
    """Suction 座標反推 Chuck 原始座標。"""
    cx = -x if flip_x else x
    cy = -y if flip_y else y
    return _coord_key(cx, cy)


def remap_chuck_to_suction(
    chuck_path: Path,
    suction_path: Path,
    output_path: Path,
    flip_x: bool = True,
    flip_y: bool = False,
) -> pd.DataFrame:
    chuck = _read_chuck(chuck_path)
    suction = _read_suction(suction_path)

    chuck_lookup = {
        _coord_key(x, y): row["value"]
        for x, y, row in zip(
            chuck["x_rel_mm"],
            chuck["y_rel_mm"],
            chuck.to_dict("records"),
        )
    }

    rows = []
    for _, srow in suction.iterrows():
        sx = float(srow["x_rel_mm"])
        sy = float(srow["y_rel_mm"])
        cx, cy = _inverse_flip(sx, sy, flip_x, flip_y)
        value = chuck_lookup.get((cx, cy), np.nan)
        rows.append({"x_rel_mm": sx, "y_rel_mm": sy, "value": value})

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def flip_chuck_file(
    chuck_path: Path,
    output_path: Path,
    flip_x: bool = True,
    flip_y: bool = True,
) -> pd.DataFrame:
    """無 Suction 參考時，只反轉 Chuck 座標並保留原列順序。"""
    chuck = _read_chuck(chuck_path)
    out = chuck.copy()
    if flip_x:
        out["x_rel_mm"] = -out["x_rel_mm"].astype(float)
    if flip_y:
        out["y_rel_mm"] = -out["y_rel_mm"].astype(float)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chuck 對齊 Suction 的 x_rel_mm / y_rel_mm 網格"
    )
    parser.add_argument("chuck_csv", type=Path)
    parser.add_argument("suction_csv", type=Path, nargs="?", default=None)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--flip-x", dest="flip_x", action="store_true", default=True)
    parser.add_argument("--no-flip-x", dest="flip_x", action="store_false")
    parser.add_argument("--flip-y", dest="flip_y", action="store_true", default=False)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    args = parser.parse_args()

    if args.suction_csv is None:
        result = flip_chuck_file(
            args.chuck_csv,
            args.output,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
        )
    else:
        result = remap_chuck_to_suction(
            args.chuck_csv,
            args.suction_csv,
            args.output,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
        )
    finite = int(np.isfinite(result["value"]).sum()) if "value" in result else len(result)
    print(f"已輸出 {len(result)} 筆（有效值 {finite} 筆）-> {args.output}")


if __name__ == "__main__":
    main()
