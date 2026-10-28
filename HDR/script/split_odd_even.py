"""依檔名序號把一個資料夾裡的檔案分成奇數與偶數兩個資料夾。

序號是檔名裡 `_數字_日期_時間` 的那一組，例如
`Basler_20261021_134654_0001_20261021_134848.npy` 的序號是 1。
同一序號的 `.npy` 與 `.json` 會進同一個資料夾。奇數序號進 `--odd`，
偶數序號進 `--even`。檔案是複製，來源資料夾不會被改動。

輸出固定在本腳本旁邊的 output/。

用法：
  python split_odd_even.py DATA_DIR --odd 20us --even 50us
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FRAME_RE = re.compile(r"_(\d+)_\d{8}_\d{6}\.[^.]+$", re.IGNORECASE)
INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def frame_index(path: Path) -> int | None:
    match = FRAME_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def folder_name(name: str) -> str:
    cleaned = name.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or cleaned != name.strip()
        or INVALID_NAME_RE.search(cleaned)
        or cleaned.endswith(".")
        or cleaned.endswith(" ")
    ):
        raise SystemExit(f"資料夾名稱不合法: {name!r}")
    return cleaned


def split_folder(src: Path, odd_name: str, even_name: str, overwrite: bool) -> None:
    src = src.expanduser().resolve()
    output_root = OUTPUT_ROOT.resolve()
    if not src.is_dir():
        raise SystemExit(f"找不到資料夾: {src}")
    if src == output_root or output_root in src.parents or src in output_root.parents:
        raise SystemExit("輸入資料夾不能是輸出位置，也不能包住輸出位置")
    if odd_name == even_name:
        raise SystemExit("兩個資料夾名稱不能相同")

    files = sorted((path for path in src.iterdir() if path.is_file()), key=lambda path: path.name.lower())
    if not files:
        subs = sorted(path.name for path in src.iterdir() if path.is_dir())
        hint = f" 請改指其中一個子資料夾，例如: {', '.join(subs)}" if subs else ""
        raise SystemExit(f"資料夾裡沒有檔案: {src}.{hint}")

    odd_files: list[Path] = []
    even_files: list[Path] = []
    skipped: list[Path] = []
    for path in files:
        index = frame_index(path)
        if index is None:
            skipped.append(path)
        elif index % 2:
            odd_files.append(path)
        else:
            even_files.append(path)
    if not odd_files and not even_files:
        raise SystemExit("沒有檔名帶序號的檔案，無法分辨奇偶")

    odd_dir = output_root / odd_name
    even_dir = output_root / even_name
    for dest, items in ((odd_dir, odd_files), (even_dir, even_files)):
        if not items:
            continue
        if dest.exists() and any(dest.iterdir()) and not overwrite:
            raise SystemExit(f"輸出資料夾已有內容，加上 --overwrite 才會覆寫: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        for src_file in items:
            target = dest / src_file.name
            if target.exists() and not overwrite:
                raise SystemExit(f"檔案已存在: {target}")
            shutil.copy2(src_file, target)

    print(f"來源: {src}")
    print(f"奇數 {len(odd_files)} 個 -> {odd_dir}")
    print(f"偶數 {len(even_files)} 個 -> {even_dir}")
    if skipped:
        names = ", ".join(path.name for path in skipped[:8])
        more = "" if len(skipped) <= 8 else f" …共 {len(skipped)} 個"
        print(f"略過沒有序號的檔案: {names}{more}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="依檔名序號把資料夾裡的檔案分成奇數與偶數兩個資料夾。輸出固定在 script/output。",
    )
    parser.add_argument("src", type=Path, help="要分類的資料夾（只處理這一層的檔案）")
    parser.add_argument("--odd", required=True, help="奇數序號的輸出資料夾名稱")
    parser.add_argument("--even", required=True, help="偶數序號的輸出資料夾名稱")
    parser.add_argument("--overwrite", action="store_true", help="輸出資料夾已有同名檔時覆寫")
    args = parser.parse_args(argv)
    split_folder(args.src, folder_name(args.odd), folder_name(args.even), args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
