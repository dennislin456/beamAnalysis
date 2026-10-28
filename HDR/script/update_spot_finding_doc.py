"""Regenerate docs/spot_finding.md from analysis.spots module docstrings.

Run from beamAnalysis/HDR:
  python script/update_spot_finding_doc.py
"""

from __future__ import annotations

import importlib
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "spot_finding.md"
MODE_MODULES = (
    "analysis.spots.shape",
    "analysis.spots.intensity",
    "analysis.spots.peak",
    "analysis.spots.line",
    "analysis.spots.slant_iter",
)


def _field(doc: str, name: str) -> str:
    match = re.search(rf"^DOC_{name}:\s*(.+)$", doc, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _body(doc: str) -> str:
    lines = []
    past_meta = False
    for line in (doc or "").splitlines():
        stripped = line.strip()
        if not past_meta:
            if stripped.startswith("DOC_") or not stripped:
                continue
            # Drop the one-line English module blurb before DOC_ fields if any leaked.
            if stripped.endswith("dual-spot mode.") or stripped.endswith("mode for dual-spot finding."):
                continue
            past_meta = True
        if stripped.startswith("DOC_"):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines).strip()
    # Keep the pipeline section readable in Chinese docs.
    text = text.replace("Pipeline after the shared strip / valley step:", "波谷之後：")
    return text


def build() -> str:
    sections = []
    for mod_name in MODE_MODULES:
        module = importlib.import_module(mod_name)
        importlib.reload(module)
        doc = module.__doc__ or ""
        title = _field(doc, "TITLE") or getattr(module, "MODE_LABEL", mod_name)
        summary = _field(doc, "SUMMARY")
        body = _body(doc)
        file_path = Path(module.__file__).name if module.__file__ else mod_name
        block = [f"## {title}", "", f"- 模組：`analysis/spots/{file_path}`", f"- 代碼：`{getattr(module, 'MODE_ID', '')}`"]
        if summary:
            block.extend(["", summary])
        if body:
            block.extend(["", body])
        sections.append("\n".join(block))

    today = date.today().isoformat()
    return f"""# 上下光斑抓法

此檔由 `script/update_spot_finding_doc.py` **自動產生**。改四種抓法的模組 docstring（含 `DOC_TITLE` / `DOC_SUMMARY`）後執行：

```text
python script/update_spot_finding_doc.py
```

對齊程式：`analysis/spots/`（{today}）

共用流程在 `analysis/spots/common.py`：整張圖上門檻質心決定條帶 X → 條帶加總找下峰與波谷（距離、下峰比、谷深 65%）→ 各模式決定上下座標（下光斑 blob 用下門檻）。

---

## 共用流程

1. 在整張圖用門檻百分比抓最強連通核心的質心，得到上光斑種子 xy，並以此 X 擺直條帶。
2. 條帶內每一列沿 X 加總（不扣中位數）。
3. 由種子 Y（或條帶波峰模式的加總最高峰）往下找通過「距離／下峰比／谷深」的下峰；兩峰間最低點當波谷。
4. 再依所選演算法決定上、下光斑座標（見下列四節）。

---

""" + "\n\n---\n\n".join(sections) + "\n"


def main() -> None:
    text = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
