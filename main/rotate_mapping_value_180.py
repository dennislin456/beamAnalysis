"""將 mapping CSV 的 value 做 180° 反轉：座標不變，值取自 (-x, -y) 的原始數值。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _coord_key(x: float, y: float) -> tuple[float, float]:
    return round(float(x), 6), round(float(y), 6)


def _read_mapping_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {str(col).replace("\ufeff", "").strip().lower(): col for col in df.columns}
    required = ("x_rel_mm", "y_rel_mm", "value")
    missing = [name for name in required if name not in rename]
    if missing:
        raise ValueError(f"缺少欄位：{', '.join(missing)}")
    out = df[[rename[name] for name in required]].copy()
    out.columns = list(required)
    return out


def rotate_value_180(df: pd.DataFrame) -> pd.DataFrame:
    """x_rel_mm、y_rel_mm 不變；value 取自 (-x, -y) 對應點。"""
    lookup = {
        _coord_key(x, y): value
        for x, y, value in zip(df["x_rel_mm"], df["y_rel_mm"], df["value"])
    }
    values = []
    for x, y in zip(df["x_rel_mm"], df["y_rel_mm"]):
        key = _coord_key(-float(x), -float(y))
        values.append(lookup.get(key, np.nan))
    out = df.copy()
    out["value"] = values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="value 180° 反轉：依 (-x, -y) 查值，x_rel_mm / y_rel_mm 維持原樣"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("c-s_e.csv"),
        help="輸入 CSV（預設：同目錄 c-s_e.csv）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="輸出 CSV（預設覆寫輸入檔）",
    )
    args = parser.parse_args()

    input_path = args.input_csv.resolve()
    output_path = args.output.resolve() if args.output else input_path

    df = _read_mapping_csv(input_path)
    result = rotate_value_180(df)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    finite = int(np.isfinite(result["value"]).sum())
    print(f"已輸出 {len(result)} 筆（有效值 {finite} 筆）-> {output_path}")


if __name__ == "__main__":
    main()
