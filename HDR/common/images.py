"""Folder listing and exposure-token names used by the HDR window.

Copied from FourStepAdj ``app.py``. Only the image extensions, exposure
tokens, and file listing stay here.
"""

from __future__ import annotations

import re
from pathlib import Path

TOKENS = ("100000us", "50000us", "5000us", "500us", "50us", "20us")
BARE_TOKENS = {
    "100000": "100000us",
    "50000": "50000us",
    "5000": "5000us",
    "500": "500us",
    "50": "50us",
    "20": "20us",
}
IMAGE_EXTS = {".npy", ".png", ".bmp", ".tif", ".tiff", ".csv"}
FRAME_RE = re.compile(r"_(\d+)_\d{8}_\d{6}\.[^.]+$", re.IGNORECASE)
US_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)us(?!\d)", re.IGNORECASE)
# Basler-style: X16p00_Y123p80 → (16.00, 123.80); also X16.00_Y123.80.
_BASLER_XY_RE = re.compile(
    r"(?:^|_)X(-?\d+)p(\d+)_Y(-?\d+)p(\d+)(?:_|$)",
    re.IGNORECASE,
)
_DOT_XY_RE = re.compile(
    r"(?:^|_)X(-?\d+(?:\.\d+)?)_Y(-?\d+(?:\.\d+)?)(?:_|$)",
    re.IGNORECASE,
)


def parse_xy_from_name(name: str) -> tuple[float, float] | None:
    """Stage XY from a file or folder name. ``p`` means the decimal point."""
    match = _BASLER_XY_RE.search(name)
    if match is not None:
        x = float(f"{match.group(1)}.{match.group(2)}")
        y = float(f"{match.group(3)}.{match.group(4)}")
        return x, y
    match = _DOT_XY_RE.search(name)
    if match is not None:
        return float(match.group(1)), float(match.group(2))
    return None


def xy_key(xy: tuple[float, float]) -> tuple[float, float]:
    """Stable key for grouping stage positions."""
    return (round(float(xy[0]), 4), round(float(xy[1]), 4))


def format_xy_label(xy: tuple[float, float]) -> str:
    """UI / folder-style label, e.g. ``X16.00_Y123.80``."""
    x, y = float(xy[0]), float(xy[1])
    return f"X{x:.2f}_Y{y:.2f}"


def exposure_us_from_name(name: str) -> float | None:
    """Exposure in microseconds from a file or folder name.

    Known tokens are matched longest first, so 50000us is not read as 500us.
    Any ``Nus`` token in the name wins over bare digits (so ``...10000us...``
    is never read as ``50`` from a stray ``50`` elsewhere in the string).
    """
    lowered = name.lower().replace(" ", "")
    token = classify_exposure(name)
    if token is not None:
        return float(token[:-2] if token.endswith("us") else token)
    # Prefer the longest explicit Nus match (e.g. 10000us over a later 50us).
    matches = list(US_RE.finditer(lowered))
    if matches:
        best = max(matches, key=lambda m: (len(m.group(1)), m.start()))
        value = float(best.group(1))
        if value > 0:
            return value
    return None


def classify_exposure(name: str) -> str | None:
    lowered = name.lower().replace(" ", "")
    for token in TOKENS:
        if re.fullmatch(token, lowered) or re.search(rf"(?<!\d){re.escape(token)}(?!\d)", lowered):
            return token
    # Bare digits only when the whole name is that number (folder "50"), not a
    # substring inside 10000 / timestamps / Basler ids.
    for bare, token in sorted(BARE_TOKENS.items(), key=lambda item: -len(item[0])):
        if re.fullmatch(bare, lowered):
            return token
    return None


def frame_sort_key(path: Path) -> tuple:
    match = FRAME_RE.search(path.name)
    if match:
        return (int(match.group(1)), path.name.lower())
    parts = [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]
    return (10**9, parts)


def is_merged_name(name: str) -> bool:
    """True for fusion product files such as ``..._merged_YYYYMMDD_HHMMSS.npy``."""
    return bool(re.search(r"(?<![a-z0-9])merged(?![a-z0-9])", name.lower()))


def list_images(folder: Path) -> list[Path]:
    files = [
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and not is_merged_name(path.name)
    ]
    return sorted(files, key=frame_sort_key)
