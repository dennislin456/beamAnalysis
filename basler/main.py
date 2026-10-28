#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""獨立啟動 Basler 短／長曝光雙檔分頁。"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

BASLER_DIR = Path(__file__).resolve().parent
MAIN_DIR = BASLER_DIR.parent / "main"
TAB_TITLE = "Basler 短／長曝光"


def _prefer_paths(*dirs: Path) -> None:
    """強制把指定目錄依序放到 sys.path 最前（後寫入的在更前面）。"""
    resolved = [str(p.resolve()) for p in dirs]
    # 先清掉舊項目，避免腳本目錄已在 path 時 insert 被跳過、導致 main 蓋過 basler
    cleaned = []
    for entry in sys.path:
        try:
            if str(Path(entry).resolve()) in resolved:
                continue
        except OSError:
            pass
        cleaned.append(entry)
    sys.path[:] = cleaned
    for path in resolved:
        sys.path.insert(0, path)


# 先放 main（依賴），再放 basler（本資料夾 dual tab 必須優先）
_prefer_paths(MAIN_DIR, BASLER_DIR)

from tab_batch_basler_dual import BaslerBatchDualTab  # noqa: E402


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BaslerBatchDualTab()
    window.setWindowTitle(TAB_TITLE)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
