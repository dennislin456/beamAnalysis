"""Standalone test window for Debevec, Robertson, and Mertens HDR.

Launch from beamAnalysis/HDR with: python hdr_test.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, QRectF, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from common import (
    OptionalNumberBox,
    ask_scientific,
    clip_color_scale,
    exposure_us_from_name,
    function_frame,
    gaussian_blur,
    install_input_undo,
    list_images,
)
from common.images import is_merged_name
from common.sci_format import format_sci
from analysis import SpotPanel
from analysis.display import COLOR_STOPS, colormap_from_stops, jet_bw_colormap
from analysis.spot_window import ANALYSIS_PANEL_W
from hdr_algorithms import run_hdr

CROP_SIDE = 512


def _natural_name_key(name: str) -> list[tuple[int, float | str]]:
    """Sort like Explorer: numbers compare numerically inside the name."""
    parts = re.split(r"(\d+(?:\.\d+)?)", name.lower())
    key: list[tuple[int, float | str]] = []
    for part in parts:
        if not part:
            continue
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part))
    return key


CANVAS = "#E8ECF0"
FUSION_PANEL_W = 440


def load_hdr_image(path: str | Path) -> np.ndarray:
    """Load a 2D image or an RGB image. Alpha is dropped."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    elif ext in {".png", ".bmp", ".tif", ".tiff"}:
        from PIL import Image

        arr = np.array(Image.open(path))
    else:
        raise ValueError(f"不支援的檔案格式: {path.suffix}")
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3):
        return arr
    raise ValueError(f"{path.name} 不是二維或 RGB 影像，形狀為 {arr.shape}")


def guess_exposure_us(path: Path) -> float | None:
    for name in (path.parent.name, path.stem, path.name):
        value = exposure_us_from_name(name)
        if value is not None:
            return value
    return None


def blur_hdr_image(image: np.ndarray, width_x: int, width_y: int, sigma: float) -> np.ndarray:
    """Gaussian-blur one exposure or result. Each color channel is blurred on its own."""
    data = np.asarray(image)
    if int(width_x) <= 1 and int(width_y) <= 1:
        return data
    if data.ndim == 2:
        return gaussian_blur(data, width_x, width_y, sigma)
    if data.ndim == 3:
        channels = [gaussian_blur(data[:, :, index], width_x, width_y, sigma) for index in range(data.shape[2])]
        return np.stack(channels, axis=2)
    return data


def preprocess_exposure(
    image: np.ndarray,
    gain: float,
    floor: float | None,
    ceil: float | None,
    width_x: int,
    width_y: int,
    sigma: float,
) -> np.ndarray:
    """Gain, then this exposure's color scale, then Gaussian blur. Used before HDR."""
    scale = float(gain)
    if not np.isfinite(scale):
        raise ValueError("倍率必須是有限數字。")
    out = np.asarray(image, dtype=np.float64) * scale
    out = clip_color_scale(out, floor, ceil)
    return blur_hdr_image(out, width_x, width_y, sigma)


def display_log10(image: np.ndarray) -> np.ndarray:
    """log10 display of non-negative intensity.

    Camera-scale data (peak > 1.5) uses floor 1, same as the main canvas.
    Fusion results in about 0–1 use a small floor so the image is not flattened to all zeros.
    """
    data = np.asarray(image, dtype=np.float64)
    data = np.where(np.isfinite(data), np.maximum(data, 0.0), 0.0)
    finite = data[np.isfinite(data)]
    peak = float(finite.max()) if finite.size else 0.0
    floor = 1.0 if peak > 1.5 else 1e-6
    return np.log10(np.maximum(data, floor))


def _spin(
    low: float,
    high: float,
    value: float,
    step: float,
    decimals: int,
    tip: str,
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setToolTip(tip)
    return box


def _panel_style() -> str:
    return """
    QWidget#sidePanel {
        background: #F4F6F8;
        border: 1px solid #e2e7ec;
        border-radius: 14px;
    }
    QScrollArea#sideScroll {
        background: transparent;
        border: none;
    }
    QLabel#hint {
        color: #5a646e;
        background: #ffffff;
        border: 1px solid #e4e9ee;
        border-radius: 10px;
        padding: 11px 13px;
        line-height: 1.4;
    }
    QLabel#section {
        color: #1f2933;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.8px;
        padding: 14px 2px 4px 2px;
        border-top: 1px solid #e6eaee;
    }
    QFrame#groupCard {
        background: #ffffff;
        border: 1px solid #e4e9ee;
        border-radius: 10px;
    }
    QLabel#card {
        background: #ffffff;
        color: #2c333a;
        border: 1px solid #e4e9ee;
        border-radius: 10px;
        padding: 12px 14px;
        line-height: 1.45;
    }
    QFrame#fileRow {
        background: #ffffff;
        border: 1px solid #e4e9ee;
        border-radius: 10px;
    }
    QFrame#fileRow:hover { border-color: #c5ced8; background: #fcfdfe; }
    QPushButton {
        background: #ffffff;
        color: #2c333a;
        border: 1px solid #d5dbe2;
        border-radius: 8px;
        padding: 7px 8px;
        min-height: 24px;
    }
    QPushButton:hover { background: #f2f5f8; border-color: #c5ced8; }
    QPushButton:disabled { color: #a0a8b0; background: #f3f4f6; }
    QPushButton#primary {
        background: #243039;
        color: #ffffff;
        border: 1px solid #243039;
        font-weight: 600;
        padding: 7px 12px;
        min-height: 24px;
    }
    QPushButton#primary:hover { background: #33404a; border-color: #33404a; }
    QPushButton#primary:disabled {
        background: #c5cbd3;
        color: #f5f6f7;
        border-color: #c5cbd3;
    }
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
        background: #ffffff;
        color: #2c333a;
        border: 1px solid #d5dbe2;
        border-radius: 7px;
        padding: 4px 8px;
        min-height: 24px;
    }
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
        border-color: #b8c0ca;
    }
    QCheckBox { color: #2c333a; spacing: 6px; }
    QScrollArea#nameScroll, QScrollArea#fileScroll { background: transparent; border: none; }
    QLabel#fileName { color: #1f2933; background: transparent; font-weight: 500; }
    QLabel#fileMeta { color: #6b7580; background: transparent; }
    QProgressBar {
        border: 1px solid #d5dbe2;
        border-radius: 8px;
        background: #ffffff;
        text-align: center;
        min-height: 18px;
        color: #2c333a;
    }
    QProgressBar::chunk { background: #6f8196; border-radius: 7px; }
    """


def _section_label(text: str, *, ruled: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    if not ruled:
        label.setStyleSheet(
            "QLabel#section { border-top: none; padding-top: 2px; }"
        )
    return label


def _group_card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("groupCard")
    frame.setAttribute(Qt.WA_StyledBackground, True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    return frame, layout


class _NameScroll(QScrollArea):
    """One filename, scrolled sideways with the wheel.

    The label can be wider than the viewport; this widget reports a small
    sizeHint so long Basler names never widen the fusion panel. Hover and
    use the mouse wheel to pan; full path stays in the tooltip.
    """

    def __init__(self, text: str, tooltip: str) -> None:
        super().__init__()
        self.setObjectName("nameScroll")
        label = QLabel(text)
        label.setObjectName("fileName")
        label.setWordWrap(False)
        label.setToolTip(tooltip)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._label = label
        self.setWidget(label)
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(label.fontMetrics().height() + 4)
        self.setMinimumWidth(0)
        # Ignored: take only the width the row layout gives; never push the panel wider.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setToolTip("檔名過長時在此處滾輪左右滑動；完整路徑見提示。")
        self._sync_label_width()

    def minimumSizeHint(self):
        return QSize(40, self.height())

    def sizeHint(self):
        return QSize(80, self.height())

    def _sync_label_width(self) -> None:
        text = self._label.text() or ""
        width = max(self._label.fontMetrics().horizontalAdvance(text) + 8, 1)
        self._label.setFixedWidth(width)

    def set_text(self, text: str, tooltip: str) -> None:
        self._label.setText(text)
        self._label.setToolTip(tooltip)
        self.setToolTip(f"{tooltip}\n（在檔名上滾輪可左右滑動）" if tooltip else "在檔名上滾輪可左右滑動")
        self._sync_label_width()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta)
        event.accept()


def _muted_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color: #5c6570; background: transparent;")
    return label


class ExposureRow(QFrame):
    """One imported exposure. Rows are created only after a file is loaded."""

    # Same four-line card as before; fixed columns keep every input lined up.
    ROW_HEIGHT = 142
    _LABEL_W = 64
    _FIELD_W = 92
    _LABEL2_W = 32

    def __init__(self, time_us: float, on_remove, on_change) -> None:
        super().__init__()
        self.slot_us = float(time_us)
        self.path: Path | None = None
        self.image: np.ndarray | None = None
        self.file_count = 0
        self._meta_core = ""
        self._file_full_name = ""
        self._on_remove = on_remove
        self.setObjectName("fileRow")
        self.setFixedHeight(self.ROW_HEIGHT)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        # col0 check | col1 label | col2 field | col3 label2 | col4 field2 | col5 count | col6 remove
        grid.setColumnMinimumWidth(1, self._LABEL_W)
        grid.setColumnMinimumWidth(2, self._FIELD_W)
        grid.setColumnMinimumWidth(3, self._LABEL2_W)
        grid.setColumnMinimumWidth(4, self._FIELD_W)
        grid.setColumnStretch(5, 1)

        def field_label(text: str, width: int) -> QLabel:
            label = _muted_label(text)
            label.setFixedWidth(width)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            return label

        self.use = QCheckBox()
        self.use.setToolTip("取消後這張不參與這次計算。")
        self.use.toggled.connect(on_change)
        us_label = field_label("曝光", self._LABEL_W)
        self.time = _spin(
            1e-3,
            1e12,
            self.slot_us,
            1.0,
            3,
            "這張影像的曝光時間，單位是 us。由檔名或資料夾名自動偵測，可再改。OpenCV 只用各張的比例。",
        )
        self.time.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.time.setFixedWidth(self._FIELD_W)
        self.count_value = QLabel("共 — 張")
        self.count_value.setObjectName("fileMeta")
        self.count_value.setStyleSheet(
            "color: #1f2933; background: transparent; font-weight: 600;"
        )
        self.remove = QPushButton("移除")
        self.remove.setFixedWidth(48)
        self.remove.setToolTip("移除此曝光列。")
        self.remove.clicked.connect(lambda: self._on_remove(self))

        self.average_n = OptionalNumberBox(
            None,
            "排序後每 N 張加總平均成一張；餘數丟掉。"
            "按計算時取「目前張數」對應的那一段，只做一次融合。"
            "空白＝不採用（維持原樣）。N 大於總張數會擋下計算。"
            "有勾批次融合時此欄暫不套用，改依原始張數逐筆融合。",
            width=self._FIELD_W,
        )
        self.average_n.setPlaceholderText("不採用")
        self.average_n.numberChanged.connect(on_change)
        self.gain = _spin(
            -1e12,
            1e12,
            1.0,
            0.1,
            4,
            "這張曝光先乘上這個倍率，再套這張自己的 color scale，最後才做高斯模糊。1 表示不改變。",
        )
        self.gain.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.gain.setFixedWidth(self._FIELD_W)
        self.gain.valueChanged.connect(on_change)
        self.color_floor = OptionalNumberBox(
            None,
            "這張曝光的 color scale 下限。小於等於這個值的像素改成 0。留空不採用。",
            width=self._FIELD_W,
        )
        self.color_ceil = OptionalNumberBox(
            None,
            "這張曝光的 color scale 上限。大於等於這個值的像素改成這個值。留空不採用。",
            width=self._FIELD_W,
        )
        self.color_floor.numberChanged.connect(on_change)
        self.color_ceil.numberChanged.connect(on_change)

        # Row 0: checkbox | 曝光 | time | … | 共 N 張 | 移除
        grid.addWidget(self.use, 0, 0, Qt.AlignVCenter)
        grid.addWidget(us_label, 0, 1, Qt.AlignVCenter)
        grid.addWidget(self.time, 0, 2, Qt.AlignVCenter)
        grid.addWidget(self.count_value, 0, 5, Qt.AlignVCenter)
        grid.addWidget(self.remove, 0, 6, Qt.AlignVCenter)

        # Row 1: filename
        self.file_label = QLabel("")
        self.file_label.setObjectName("fileMeta")
        self.file_label.setWordWrap(False)
        self.file_label.setMinimumWidth(0)
        self.file_label.setMinimumHeight(self.file_label.fontMetrics().height() + 2)
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.file_label.setStyleSheet("color: #8a939c; background: transparent;")
        grid.addWidget(self.file_label, 1, 1, 1, 6)

        # Row 2: 加總平均 | N | 倍率 | gain  (fields share columns with row 0 / 3)
        grid.addWidget(field_label("加總平均", self._LABEL_W), 2, 1, Qt.AlignVCenter)
        grid.addWidget(self.average_n, 2, 2, Qt.AlignVCenter)
        grid.addWidget(field_label("倍率", self._LABEL2_W), 2, 3, Qt.AlignVCenter)
        grid.addWidget(self.gain, 2, 4, Qt.AlignVCenter)

        # Row 3: 下限 | floor | 上限 | ceil
        grid.addWidget(field_label("下限", self._LABEL_W), 3, 1, Qt.AlignVCenter)
        grid.addWidget(self.color_floor, 3, 2, Qt.AlignVCenter)
        grid.addWidget(field_label("上限", self._LABEL2_W), 3, 3, Qt.AlignVCenter)
        grid.addWidget(self.color_ceil, 3, 4, Qt.AlignVCenter)

        outer.addLayout(grid)
        self.use.setChecked(True)

    def minimumSizeHint(self):
        return QSize(200, self.ROW_HEIGHT)

    def sizeHint(self):
        return QSize(280, self.ROW_HEIGHT)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_file_label()

    def _refresh_file_label(self) -> None:
        name = self._file_full_name
        if not name:
            self.file_label.setText("")
            return
        metrics = self.file_label.fontMetrics()
        width = max(self.file_label.width(), 40)
        self.file_label.setText(metrics.elidedText(name, Qt.ElideMiddle, width))

    def average_count(self) -> int | None:
        """N for chunk averaging, or None when unused."""
        value = self.average_n.number()
        if value is None:
            return None
        if not np.isfinite(value) or value < 1:
            return None
        return int(round(float(value)))

    def preprocess_state(
        self,
    ) -> tuple[float, float | None, float | None, bool, int | None]:
        return (
            float(self.gain.value()),
            self.color_floor.number(),
            self.color_ceil.number(),
            bool(self.use.isChecked()) if self.use.isEnabled() else True,
            self.average_count(),
        )

    def set_file_count(self, count: int) -> None:
        self.file_count = max(0, int(count))
        if self.file_count > 0:
            self.count_value.setText(f"共 {self.file_count} 張")
        else:
            self.count_value.setText("共 — 張")

    def set_file(
        self,
        path: Path,
        image: np.ndarray,
        *,
        keep_preprocess: bool = False,
        preprocess: tuple | None = None,
    ) -> None:
        self.path = path
        self.image = image
        name, meta = self._describe()
        self._meta_core = meta
        self._file_full_name = name
        tip = f"{path}\n{meta}"
        self.file_label.setToolTip(tip)
        self.count_value.setToolTip(tip)
        self._refresh_file_label()
        self.time.setValue(self.slot_us)
        self.time.setEnabled(True)
        if keep_preprocess and preprocess is not None:
            gain, floor, ceil, use = preprocess[:4]
            avg = preprocess[4] if len(preprocess) > 4 else None
            self.gain.setValue(float(gain))
            self.color_floor.set_number(floor, emit=False)
            self.color_ceil.set_number(ceil, emit=False)
            self.use.setChecked(bool(use))
            self.average_n.set_number(None if avg is None else float(avg), emit=False)
        elif keep_preprocess:
            pass
        else:
            self.gain.setValue(1.0)
            self.color_floor.set_number(None, emit=False)
            self.color_ceil.set_number(None, emit=False)
            self.use.setChecked(True)
            self.average_n.set_number(None, emit=False)
        self.gain.setEnabled(True)
        self.color_floor.setEnabled(True)
        self.color_ceil.setEnabled(True)
        self.average_n.setEnabled(True)
        self.use.setEnabled(True)
        self.remove.setEnabled(True)

    def _describe(self) -> tuple[str, str]:
        data = self.image
        shape = "×".join(str(size) for size in data.shape[:2])
        dtype = str(data.dtype)
        finite = np.asarray(data, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            span = "無有限值"
        else:
            span = f"{finite.min():.4g}–{finite.max():.4g}"
        return self.path.name, f"{shape} {dtype} {span}"


class SpotPickDialog(QDialog):
    """Pick exactly one spot folder; checkboxes are exclusive."""

    def __init__(
        self,
        spots: list[Path],
        current: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("選擇點位")
        self.resize(420, 520)
        self._spots = list(spots)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("勾選一個點位（不可多選）。依資料夾名稱排序。"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(4, 4, 4, 4)
        host_layout.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        current_key = current.resolve() if current is not None else None
        checked = False
        for index, spot in enumerate(self._spots):
            box = QCheckBox(spot.name)
            box.setToolTip(str(spot))
            self._group.addButton(box, index)
            host_layout.addWidget(box)
            if current_key is not None and spot.resolve() == current_key:
                box.setChecked(True)
                checked = True
        if not checked and self._spots:
            first = self._group.button(0)
            if first is not None:
                first.setChecked(True)
        host_layout.addStretch(1)
        scroll.setWidget(host)
        layout.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        ok = QPushButton("確定")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    def selected_spot(self) -> Path | None:
        button_id = self._group.checkedId()
        if button_id < 0 or button_id >= len(self._spots):
            return None
        return self._spots[button_id]


class CropDialog(QDialog):
    """Pick calculation ROI: full frame, 512×512, or custom size."""

    MODE_FULL = "full"
    MODE_512 = "512"
    MODE_CUSTOM = "custom"

    def __init__(
        self,
        image: np.ndarray,
        origin: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("計算範圍")
        self.resize(780, 720)
        data = np.asarray(image)
        self.height, self.width = int(data.shape[0]), int(data.shape[1])
        view = data[:, :, 0] if data.ndim == 3 else data
        view = np.asarray(view, dtype=np.float64)
        finite = view[np.isfinite(view)]
        if finite.size:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        cmap = jet_bw_colormap()
        hist = getattr(parent, "hist", None)
        if hist is not None:
            cmap = hist.gradient.colorMap()

        mode = self._infer_mode(origin, size)
        crop_w, crop_h = self._initial_size(mode, size)
        x0, y0 = self._initial_origin(origin, crop_w, crop_h)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("選擇計算範圍。全屏用整張影像；512×512 與自行輸入可拖移方框。計算和批次融合都用這一塊。"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self.mode_full = QRadioButton("全屏")
        self.mode_512 = QRadioButton("512 × 512")
        self.mode_custom = QRadioButton("自行輸入")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_full)
        self.mode_group.addButton(self.mode_512)
        self.mode_group.addButton(self.mode_custom)
        mode_row.addWidget(self.mode_full)
        mode_row.addWidget(self.mode_512)
        mode_row.addWidget(self.mode_custom)
        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(0, 0, 0, 0)
        custom_row.setSpacing(6)
        custom_row.addWidget(QLabel("寬"))
        self.width_box = QSpinBox()
        self.width_box.setRange(1, self.width)
        self.width_box.setValue(crop_w)
        self.width_box.setFixedWidth(72)
        self.width_box.setToolTip("自行輸入的寬度（像素）。")
        custom_row.addWidget(self.width_box)
        custom_row.addWidget(QLabel("高"))
        self.height_box = QSpinBox()
        self.height_box.setRange(1, self.height)
        self.height_box.setValue(crop_h)
        self.height_box.setFixedWidth(72)
        self.height_box.setToolTip("自行輸入的高度（像素）。")
        custom_row.addWidget(self.height_box)
        self.custom_host = QWidget()
        self.custom_host.setLayout(custom_row)
        mode_row.addWidget(self.custom_host)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.pos_label = QLabel()
        layout.addWidget(self.pos_label)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground("#111111")
        plot = self.win.addPlot()
        plot.setAspectLocked(True)
        plot.invertY(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMouseEnabled(x=False, y=False)
        item = pg.ImageItem()
        item.setColorMap(cmap)
        plot.addItem(item)
        item.setImage(np.nan_to_num(view, nan=0.0).T, autoLevels=False, levels=(lo, hi))
        plot.setXRange(0, self.width, padding=0)
        plot.setYRange(0, self.height, padding=0)
        self.roi = pg.RectROI(
            [x0, y0],
            [crop_w, crop_h],
            pen=pg.mkPen("#FF4FD8", width=2),
            movable=True,
            resizable=False,
        )
        for handle in [entry["item"] for entry in list(self.roi.handles)]:
            self.roi.removeHandle(handle)
        self.roi.setAcceptedMouseButtons(Qt.LeftButton)
        self.roi.sigRegionChanged.connect(self._keep_inside)
        plot.addItem(self.roi)
        self._moving = False
        self._applying = False

        if mode == self.MODE_FULL:
            self.mode_full.setChecked(True)
        elif mode == self.MODE_512:
            self.mode_512.setChecked(True)
        else:
            self.mode_custom.setChecked(True)
        self.mode_full.toggled.connect(self._on_mode_changed)
        self.mode_512.toggled.connect(self._on_mode_changed)
        self.mode_custom.toggled.connect(self._on_mode_changed)
        self.width_box.valueChanged.connect(self._on_custom_size_changed)
        self.height_box.valueChanged.connect(self._on_custom_size_changed)
        self._apply_mode(mode, reset_size=False)
        layout.addWidget(self.win, 1)
        buttons = QHBoxLayout()
        ok = QPushButton("確定")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    def _infer_mode(
        self,
        origin: tuple[int, int] | None,
        size: tuple[int, int] | None,
    ) -> str:
        if size is None:
            if self.width >= CROP_SIDE and self.height >= CROP_SIDE:
                return self.MODE_512
            return self.MODE_FULL
        w, h = int(size[0]), int(size[1])
        if w == self.width and h == self.height and (origin is None or origin == (0, 0)):
            return self.MODE_FULL
        if w == CROP_SIDE and h == CROP_SIDE:
            return self.MODE_512
        return self.MODE_CUSTOM

    def _initial_size(self, mode: str, size: tuple[int, int] | None) -> tuple[int, int]:
        if mode == self.MODE_FULL:
            return self.width, self.height
        if mode == self.MODE_512:
            return min(CROP_SIDE, self.width), min(CROP_SIDE, self.height)
        if size is not None:
            return (
                int(np.clip(size[0], 1, self.width)),
                int(np.clip(size[1], 1, self.height)),
            )
        return min(CROP_SIDE, self.width), min(CROP_SIDE, self.height)

    def _initial_origin(
        self,
        origin: tuple[int, int] | None,
        crop_w: int,
        crop_h: int,
    ) -> tuple[int, int]:
        limit_x = max(0, self.width - crop_w)
        limit_y = max(0, self.height - crop_h)
        if origin is None:
            return limit_x // 2, limit_y // 2
        return int(np.clip(origin[0], 0, limit_x)), int(np.clip(origin[1], 0, limit_y))

    def _current_mode(self) -> str:
        if self.mode_full.isChecked():
            return self.MODE_FULL
        if self.mode_512.isChecked():
            return self.MODE_512
        return self.MODE_CUSTOM

    def _on_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        self._apply_mode(self._current_mode(), reset_size=True)

    def _on_custom_size_changed(self) -> None:
        if self._applying or not self.mode_custom.isChecked():
            return
        self._set_roi_size(int(self.width_box.value()), int(self.height_box.value()))

    def _apply_mode(self, mode: str, *, reset_size: bool) -> None:
        self._applying = True
        self.custom_host.setVisible(mode == self.MODE_CUSTOM)
        if mode == self.MODE_FULL:
            w, h = self.width, self.height
            self.roi.translatable = False
        elif mode == self.MODE_512:
            if self.width < CROP_SIDE or self.height < CROP_SIDE:
                self._applying = False
                QMessageBox.warning(
                    self,
                    "影像太小",
                    f"512×512 需要至少 {CROP_SIDE}×{CROP_SIDE}，這張是 {self.width}×{self.height}。",
                )
                if self.width >= 1 and self.height >= 1:
                    self.mode_full.blockSignals(True)
                    self.mode_full.setChecked(True)
                    self.mode_full.blockSignals(False)
                    self._apply_mode(self.MODE_FULL, reset_size=True)
                return
            w, h = CROP_SIDE, CROP_SIDE
            self.roi.translatable = True
        else:
            if reset_size:
                w = min(CROP_SIDE, self.width)
                h = min(CROP_SIDE, self.height)
                self.width_box.setValue(w)
                self.height_box.setValue(h)
            else:
                w = int(self.width_box.value())
                h = int(self.height_box.value())
            self.roi.translatable = True
        self._applying = False
        self._set_roi_size(w, h, center_if_full=(mode == self.MODE_FULL))

    def _set_roi_size(self, crop_w: int, crop_h: int, *, center_if_full: bool = False) -> None:
        crop_w = int(np.clip(crop_w, 1, self.width))
        crop_h = int(np.clip(crop_h, 1, self.height))
        if center_if_full and crop_w == self.width and crop_h == self.height:
            x, y = 0, 0
        else:
            pos = self.roi.pos()
            limit_x = max(0, self.width - crop_w)
            limit_y = max(0, self.height - crop_h)
            x = float(np.clip(pos.x(), 0, limit_x))
            y = float(np.clip(pos.y(), 0, limit_y))
        self._moving = True
        self.roi.setSize([crop_w, crop_h], update=False)
        self.roi.setPos(x, y, update=False)
        self._moving = False
        self._update_pos_label(x, y, crop_w, crop_h)

    def _keep_inside(self) -> None:
        if self._moving:
            return
        size = self.roi.size()
        crop_w = int(round(size.x()))
        crop_h = int(round(size.y()))
        crop_w = int(np.clip(crop_w, 1, self.width))
        crop_h = int(np.clip(crop_h, 1, self.height))
        limit_x = max(0, self.width - crop_w)
        limit_y = max(0, self.height - crop_h)
        pos = self.roi.pos()
        x = float(np.clip(pos.x(), 0, limit_x))
        y = float(np.clip(pos.y(), 0, limit_y))
        self._moving = True
        if abs(x - pos.x()) > 0.01 or abs(y - pos.y()) > 0.01:
            self.roi.setPos(x, y, update=False)
        if abs(crop_w - size.x()) > 0.01 or abs(crop_h - size.y()) > 0.01:
            self.roi.setSize([crop_w, crop_h], update=False)
        self._moving = False
        self._update_pos_label(x, y, crop_w, crop_h)

    def _update_pos_label(self, x: float, y: float, crop_w: int, crop_h: int) -> None:
        self.pos_label.setText(
            f"左上角 ({int(round(x))}, {int(round(y))})    {crop_w}×{crop_h}"
        )

    def origin(self) -> tuple[int, int]:
        crop_w, crop_h = self.crop_size()
        pos = self.roi.pos()
        limit_x = max(0, self.width - crop_w)
        limit_y = max(0, self.height - crop_h)
        return int(np.clip(round(pos.x()), 0, limit_x)), int(np.clip(round(pos.y()), 0, limit_y))

    def crop_size(self) -> tuple[int, int]:
        if self.mode_full.isChecked():
            return self.width, self.height
        if self.mode_512.isChecked():
            return CROP_SIDE, CROP_SIDE
        return int(self.width_box.value()), int(self.height_box.value())


class HdrTestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HDR 測試")
        self.resize(1480, 860)
        self.rows: list[ExposureRow] = []
        self.raw: np.ndarray | None = None
        self.preview: np.ndarray | None = None
        self.shown: np.ndarray | None = None
        self.response: np.ndarray | None = None
        self.cross_xy: tuple[int, int] | None = None
        self._manual_x = False
        self._manual_y = False
        self._profile_sync_pending = False
        self._profile_syncing = False
        self._x_values: np.ndarray | None = None
        self._y_values: np.ndarray | None = None
        self._color_updating = False
        self._stops_updating = False
        self.color_stop_boxes: list[QDoubleSpinBox] = []
        self.analysis: SpotPanel | None = None
        self._export_full_scale = True
        self.parent_root: Path | None = None
        self.spot_dirs: list[Path] = []
        self.source_root: Path | None = None
        self.source_groups: dict[float, list[Path]] | None = None
        self.crop_origin: tuple[int, int] | None = None
        self.crop_size: tuple[int, int] | None = None
        self.batch_planes: list[np.ndarray] = []
        self.batch_labels: list[str] = []
        self._batch_running = False
        self._batch_stop = False
        # Persist across spot switches under the same parent folder.
        # (gain, floor, ceil, use, average_n)
        self._exposure_prefs: dict[float, tuple] = {}
        self._removed_exposures: set[float] = set()
        install_input_undo()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root.setStyleSheet(
            "QWidget#appRoot {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            "    stop:0 #EEF1F4, stop:1 #E4E9EE);"
            "}"
        )
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        left = QWidget()
        left.setObjectName("sidePanel")
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(_panel_style())
        left.setMinimumWidth(0)
        left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)

        hint = QLabel(
            "一張可看熱圖與剖面；兩張以上才跑 HDR。"
            "可載入父資料夾選點位，或載入單一主資料夾。載入後先選計算範圍。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        hint.setToolTip(
            "獨立測試 Debevec、Robertson、Mertens，不影響合成視窗。\n"
            "曝光時間由檔名或資料夾名自動偵測（例如 100us、500us），只顯示已匯入的列。\n"
            "每列可設倍率和 color scale。融合前是倍率、各自的 color scale、再高斯模糊。\n"
            "Debevec、Robertson 先估相機響應再合成輻射圖；Mertens 是曝光融合。\n"
            "OpenCV 吃 8-bit：本身是 8-bit 直接用，其他用滿量壓到 0–255。"
        )
        left_layout.addWidget(hint)

        left_layout.addWidget(_section_label("資料", ruled=False))
        folder_buttons = QHBoxLayout()
        folder_buttons.setSpacing(6)
        add_parent = QPushButton("載入父資料夾")
        add_parent.setToolTip(
            "選擇含多個點位子資料夾的父資料夾。子資料夾名稱當點位名，依名稱排序。"
            "每個點位內再依曝光時間載入，和「載入主資料夾」相同。"
        )
        add_parent.clicked.connect(self._add_parent_folder)
        add_folder = QPushButton("載入主資料夾")
        add_folder.setToolTip("直接載入單一點位／曝光資料夾。行為與以前相同。")
        add_folder.clicked.connect(self._add_folder)
        for btn in (add_parent, add_folder):
            btn.setMinimumWidth(48)
            btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        folder_buttons.addWidget(add_parent, 1)
        folder_buttons.addWidget(add_folder, 1)
        left_layout.addLayout(folder_buttons)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        add_files = QPushButton("加入檔案")
        add_files.clicked.connect(self._add_files)
        self.btn_crop = QPushButton("調整範圍")
        self.btn_crop.setToolTip(
            "選擇計算範圍：全屏、512×512，或自行輸入寬高後拖移方框。"
            "計算和批次融合都用這一塊。清空、換點位或重新載入後要再選一次。"
        )
        self.btn_crop.clicked.connect(self._adjust_crop)
        clear = QPushButton("清空")
        clear.clicked.connect(self._clear_rows)
        for btn in (add_files, self.btn_crop, clear):
            btn.setMinimumWidth(48)
            btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        buttons.addWidget(add_files, 1)
        buttons.addWidget(self.btn_crop, 1)
        buttons.addWidget(clear, 1)
        left_layout.addLayout(buttons)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        source_caption = QLabel("主資料夾")
        source_caption.setStyleSheet("color: #5c6570;")
        source_row.addWidget(source_caption)
        self.source_name = QLabel("尚未載入")
        self.source_name.setStyleSheet("color: #5c6570;")
        self.source_name.setWordWrap(False)
        self.source_name.setMinimumWidth(0)
        self.source_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.source_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_row.addWidget(self.source_name, 1)
        left_layout.addLayout(source_row)

        spot_row = QHBoxLayout()
        spot_row.setSpacing(8)
        spot_caption = QLabel("點位")
        spot_caption.setStyleSheet("color: #5c6570;")
        spot_row.addWidget(spot_caption)
        self.spot_name = QLabel("尚未選擇")
        self.spot_name.setStyleSheet("color: #5c6570;")
        self.spot_name.setWordWrap(False)
        self.spot_name.setMinimumWidth(0)
        self.spot_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.spot_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        spot_row.addWidget(self.spot_name, 1)
        self.btn_spot = QPushButton("選擇點位")
        self.btn_spot.setEnabled(False)
        self.btn_spot.setToolTip(
            "開啟視窗勾選一個點位（不可多選）。"
            "切換後會重載該點位影像並再選計算範圍；倍率、color scale、演算法等參數會保留。"
        )
        self.btn_spot.clicked.connect(self._pick_spot)
        spot_row.addWidget(self.btn_spot)
        left_layout.addLayout(spot_row)

        frame_row = QHBoxLayout()
        frame_row.setSpacing(8)
        frame_label = QLabel("主資料夾筆數")
        frame_label.setStyleSheet("color: #5c6570;")
        frame_row.addWidget(frame_label)
        self.frame_index = QSpinBox()
        self.frame_index.setRange(0, 999999)
        self.frame_index.setToolTip(
            "載入主資料夾或點位時，每個曝光用排序後的這個索引。0 是第一張。"
            "某個曝光張數不夠時用該曝光最後一張，不會略過。"
            "例如 50us 1 張、500us 2 張、5000us 3 張時，第 3 筆會用 50us 第 1 張、500us 第 2 張、5000us 第 3 張。"
        )
        frame_row.addWidget(self.frame_index, 1)
        left_layout.addLayout(frame_row)

        self.file_empty = QLabel("尚未載入曝光。匯入後依檔名或資料夾名自動偵測曝光時間（不限預設六段）。")
        self.file_empty.setObjectName("hint")
        self.file_empty.setWordWrap(True)
        left_layout.addWidget(self.file_empty)
        self.file_host = QWidget()
        self.file_host.setMinimumWidth(0)
        self.file_host.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.file_layout = QVBoxLayout(self.file_host)
        self.file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_layout.setSpacing(8)
        self.rows: list[ExposureRow] = []
        self.file_scroll = QScrollArea()
        self.file_scroll.setObjectName("fileScroll")
        self.file_scroll.setWidgetResizable(True)
        self.file_scroll.setWidget(self.file_host)
        self.file_scroll.setFrameShape(QFrame.NoFrame)
        self.file_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.file_scroll.setMinimumWidth(0)
        self.file_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._sync_file_scroll_height()
        left_layout.addWidget(self.file_scroll)

        self.full_scale = _spin(
            1e-6,
            1e12,
            4095.0,
            1.0,
            3,
            "非 8-bit 影像共用的滿量。像素值 / 滿量 × 255 後裁到 0–255。12-bit 常用 4095，這套相機滿量是 4000。",
        )
        self.algorithm = QComboBox()
        self.algorithm.addItem("Debevec", "debevec")
        self.algorithm.addItem("Robertson", "robertson")
        self.algorithm.addItem("Mertens", "mertens")
        self.algorithm.addItem("Mertens 自寫", "mertens_local")
        self.algorithm.setCurrentIndex(self.algorithm.findData("mertens"))
        self.algorithm.currentIndexChanged.connect(self._sync_algorithm_page)

        self.debevec_samples = QSpinBox()
        self.debevec_samples.setRange(1, 100000)
        self.debevec_samples.setValue(70)
        self.debevec_samples.setToolTip("用來估計響應曲線的樣本數。預設 70。")
        self.debevec_lambda = _spin(0.0, 1000.0, 10.0, 0.5, 3, "響應曲線的平滑強度。越大越平滑。預設 10。")
        self.debevec_random = QCheckBox("隨機抽樣")
        self.debevec_random.setToolTip("勾選後樣本位置隨機。不勾時用固定位置。")
        self.robertson_iter = QSpinBox()
        self.robertson_iter.setRange(1, 10000)
        self.robertson_iter.setValue(30)
        self.robertson_iter.setToolTip("Robertson 疊代次數上限。預設 30。")
        self.robertson_threshold = _spin(
            1e-8, 10.0, 0.01, 0.001, 6, "Robertson 收斂門檻。預設 0.01。"
        )
        self.mertens_contrast = _spin(0.0, 20.0, 1.0, 0.1, 2, "對比權重。預設 1。")
        self.mertens_saturation = _spin(
            0.0, 20.0, 0.0, 0.1, 2, "飽和權重。預設 0。灰階影像的飽和度是 0，這個權重幾乎不作用。"
        )
        self.mertens_exposure = _spin(
            0.0,
            20.0,
            1.0,
            0.1,
            2,
            "曝光良好程度的權重。OpenCV 預設是 1。調高會偏好接近中灰的像素。",
        )
        self.local_contrast = _spin(0.0, 20.0, 1.0, 0.1, 2, "對比權重。預設 1。")
        self.local_saturation = _spin(
            0.0,
            20.0,
            1.0,
            0.1,
            2,
            "飽和權重。預設 1。灰階單通道時不使用飽和。",
        )
        self.local_exposure = _spin(
            0.0,
            20.0,
            0.0,
            0.1,
            2,
            "曝光良好程度的權重。預設 0。調高會偏好接近中灰的像素。",
        )
        self.local_sigma = _spin(
            0.01,
            2.0,
            0.2,
            0.01,
            3,
            "中灰鐘形的 sigma。OpenCV 寫死約 0.2（分母 0.08）。預設 0.2。",
        )
        # Wide numeric ranges make Qt size-hint huge; cap so the file rows are not clipped.
        for box in (
            self.full_scale,
            self.debevec_samples,
            self.debevec_lambda,
            self.robertson_iter,
            self.robertson_threshold,
            self.mertens_contrast,
            self.mertens_saturation,
            self.mertens_exposure,
            self.local_contrast,
            self.local_saturation,
            self.local_exposure,
            self.local_sigma,
        ):
            box.setMinimumWidth(0)
            box.setMaximumWidth(160)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.algorithm.setMinimumWidth(0)
        self.algorithm.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.algo_stack = QStackedWidget()
        self.algo_stack.addWidget(self._form_page([
            ("樣本數", self.debevec_samples),
            ("lambda", self.debevec_lambda),
            ("", self.debevec_random),
        ]))
        self.algo_stack.addWidget(self._form_page([
            ("疊代次數", self.robertson_iter),
            ("收斂門檻", self.robertson_threshold),
        ]))
        self.algo_stack.addWidget(self._form_page([
            ("對比", self.mertens_contrast),
            ("飽和", self.mertens_saturation),
            ("曝光", self.mertens_exposure),
        ]))
        self.algo_stack.addWidget(self._form_page([
            ("對比", self.local_contrast),
            ("飽和", self.local_saturation),
            ("曝光", self.local_exposure),
            ("sigma", self.local_sigma),
        ]))

        left_layout.addWidget(_section_label("演算法"))
        algo_card, algo_card_layout = _group_card()
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.addRow("滿量", self.full_scale)
        form.addRow("演算法", self.algorithm)
        algo_card_layout.addLayout(form)
        algo_card_layout.addWidget(self.algo_stack)
        left_layout.addWidget(algo_card)

        left_layout.addWidget(_section_label("動作"))
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.btn_run = QPushButton("計算")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._compute)
        self.btn_analyze = QPushButton("分析")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setToolTip(
            "把目前融合結果送進分析區並重抓光斑。"
            "數值是 × 255 之後的 8-bit 尺度。兩種 Mertens 已經 × 255；"
            "Debevec、Robertson 和單張會再 × 255。不含 log。彩色影像用三個通道的平均。"
        )
        self.btn_analyze.clicked.connect(self._open_analysis)
        self.btn_export = QPushButton("匯出")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            "選一個資料夾，一次寫出剖面 CSV、熱圖、npy 和參數 JSON。"
            "檔名都帶時間戳。熱圖沿用畫布的色階和剖面，不含色條與十字。"
            "npy 不含 log。Debevec、Robertson 和單張再 × 滿量；兩種 Mertens 已經 × 255。"
        )
        self.btn_export.clicked.connect(self._export_bundle)
        actions.addWidget(self.btn_run, 1)
        actions.addWidget(self.btn_analyze, 1)
        actions.addWidget(self.btn_export, 1)
        left_layout.addLayout(actions)
        self.batch_check = QCheckBox("批次融合")
        self.batch_check.setToolTip(
            "勾選後用目前的演算法、倍率、color scale 和高斯模糊，"
            "依原始張數逐筆融合目前點位（或主資料夾）。不會跨點位；加總平均 N 暫不套用。"
            "未勾選時按計算：有填加總平均則平均後只融一次。"
            "分析區可逐筆查看；過程中取消勾選會停下來。"
        )
        self.batch_check.toggled.connect(self._on_batch_toggled)
        left_layout.addWidget(self.batch_check)
        self.batch_bar = QProgressBar()
        self.batch_bar.setRange(0, 1)
        self.batch_bar.setValue(0)
        self.batch_bar.setFormat("尚未批次融合")
        self.batch_bar.setTextVisible(True)
        left_layout.addWidget(self.batch_bar)

        left_layout.addWidget(_section_label("結果"))
        self.stats = QLabel("尚未計算。")
        self.stats.setObjectName("card")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.stats.setMinimumHeight(72)
        left_layout.addWidget(self.stats)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("sideScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(FUSION_PANEL_W)
        left_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Fusion canvas stays available for export / response curves, but is not shown.
        # The visible canvas is the analysis view (shared colorbar, log, abs).
        view_host = QWidget()
        view_host.setObjectName("viewHost")
        view_host.hide()
        view_layout = QVBoxLayout(view_host)
        view_layout.setContentsMargins(8, 8, 8, 4)
        view_layout.setSpacing(4)
        self._install_y_mode_bar(view_layout)
        self.log_check = QCheckBox("log")
        self.log_check.setChecked(False)
        self.log_check.toggled.connect(self._on_log_toggled)
        self.color_max = self._make_color_box(
            "上限: ",
            "色條上限，一般小數。只改熱圖和色條的顯示範圍，不重算演算法。雙擊色條回到資料範圍。",
        )
        self.color_min = self._make_color_box(
            "下限: ",
            "色條下限，一般小數。只改熱圖和色條的顯示範圍，不重算演算法。雙擊色條回到資料範圍。",
        )
        self.color_min.valueChanged.connect(self._on_color_limit_changed)
        self.color_max.valueChanged.connect(self._on_color_limit_changed)
        for group in (COLOR_STOPS[:4], COLOR_STOPS[4:]):
            for name, _en, _color, pos in group:
                box = self._make_stop_box(name, pos)
                box.valueChanged.connect(self._on_color_stop_changed)
                self.color_stop_boxes.append(box)
        pg.setConfigOption("background", CANVAS)
        pg.setConfigOption("foreground", "#333333")
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground(CANVAS)
        curve_pen = pg.mkPen("#444444", width=1.5)
        curve_brush = pg.mkBrush("#444444")
        self.plot_y = self.win.addPlot(row=0, col=0)
        self.plot_y.setLabel("left", "Y (px)")
        self.plot_y.setLabel("bottom", "強度")
        self.plot_y.getAxis("bottom").setToolTip("點最小值輸入下限，點最大值輸入上限。雙擊剖面復原。")
        self.plot_y.showGrid(x=True, y=True, alpha=0.35)
        self.plot_y.invertY(True)
        self.plot_y.enableAutoRange(x=False, y=False)
        self.curve_y = self.plot_y.plot(pen=curve_pen)
        self.marker_y = self.plot_y.plot(pen=None, symbol="o", symbolSize=7, symbolBrush=curve_brush)
        self.tip_y = self._make_profile_tip()
        self.win.scene().addItem(self.tip_y)
        self.plot_y.vb.sigRangeChanged.connect(lambda *_: self._pin_profile_tip("y"))
        self.plot = self.win.addPlot(row=0, col=1)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.45)
        self.plot.setLabel("bottom", "X")
        self.plot.setLabel("left", "Y")
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        cross_pen = pg.mkPen("#666666", width=1, style=Qt.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.plot.addItem(self.vline)
        self.plot.addItem(self.hline)
        self.vline.hide()
        self.hline.hide()
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(jet_bw_colormap())
        self.hist.plot.setPen(pg.mkPen("#666666", width=1))
        self.hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self.win.addItem(self.hist, row=0, col=2)
        self.response_plot = self.win.addPlot(row=1, col=0, title="相機響應")
        self.response_plot.showGrid(x=True, y=True, alpha=0.45)
        self.response_plot.setLabel("bottom", "8-bit 像素")
        self.response_plot.setLabel("left", "響應")
        self.response_plot.setXRange(0, 255, padding=0.02)
        self.plot_x = self.win.addPlot(row=1, col=1, title="X 剖面")
        self.plot_x.setLabel("bottom", "X (px)")
        self.plot_x.setLabel("left", "強度")
        self.plot_x.getAxis("left").setToolTip("點最小值輸入下限，點最大值輸入上限。雙擊剖面復原。")
        self.plot_x.showGrid(x=True, y=True, alpha=0.35)
        self.plot_x.enableAutoRange(x=False, y=False)
        self.curve_x = self.plot_x.plot(pen=curve_pen)
        self.marker_x = self.plot_x.plot(pen=None, symbol="o", symbolSize=7, symbolBrush=curve_brush)
        self.tip_x = self._make_profile_tip()
        self.win.scene().addItem(self.tip_x)
        self.plot_x.vb.sigRangeChanged.connect(lambda *_: self._pin_profile_tip("x"))
        self.win.ci.layout.setRowStretchFactor(0, 4)
        self.win.ci.layout.setRowStretchFactor(1, 2)
        self.win.ci.layout.setColumnStretchFactor(0, 2)
        self.win.ci.layout.setColumnStretchFactor(1, 4)
        self.win.ci.layout.setColumnStretchFactor(2, 0)
        self._pin_axis_frame(self.plot_y, 64, 48)
        self._pin_axis_frame(self.plot, 48, 42)
        self._pin_axis_frame(self.response_plot, 56, 48)
        self._pin_axis_frame(self.plot_x, 80, 48)
        self.win.installEventFilter(self)
        self.win.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.win.scene().sigMouseClicked.connect(self._on_scene_clicked)
        view_host.setAutoFillBackground(True)
        palette = view_host.palette()
        palette.setColor(view_host.backgroundRole(), QColor(CANVAS))
        view_host.setPalette(palette)
        self.hist.gradient.sigGradientChangeFinished.connect(self._on_gradient_ticks_moved)
        view_layout.addWidget(self.win, 1)
        self.hover = QLabel("游標：—")
        self._fusion_view_host = view_host

        self.analysis = SpotPanel(embed=True)
        analysis_scroll = self.analysis.controls_scroll
        analysis_scroll.setObjectName("sideScroll")
        analysis_scroll.setFixedWidth(ANALYSIS_PANEL_W)
        analysis_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.analysis.view_host.setStyleSheet(
            "QWidget#analysisViewHost {"
            "  background: #E8ECF0;"
            "  border: 1px solid #d0d6dd;"
            "  border-radius: 14px;"
            "}"
        )
        self.analysis.view_host.setAttribute(Qt.WA_StyledBackground, True)
        # Shared visible display / preprocess controls live on the analysis canvas.
        self.smooth_check = self.analysis.smooth_check
        self.smooth_kx = self.analysis.smooth_kx
        self.smooth_ky = self.analysis.smooth_ky
        self.smooth_sigma = self.analysis.smooth_sigma
        self.smooth_check.toggled.connect(self._on_smooth_toggled)
        for box in (self.smooth_kx, self.smooth_ky, self.smooth_sigma):
            box.valueChanged.connect(self._on_preprocess_changed)
        self.analysis.log_check.toggled.connect(self._on_shared_log_toggled)
        self.analysis.hist.gradient.sigGradientChangeFinished.connect(
            self._on_shared_gradient_moved
        )
        for box in self.analysis.color_stop_boxes:
            box.valueChanged.connect(self._on_shared_color_stop_changed)
        self.analysis.color_min.valueChanged.connect(self._on_shared_color_limit_changed)
        self.analysis.color_max.valueChanged.connect(self._on_shared_color_limit_changed)
        self._apply_color_stops([float(box.value()) for box in self.analysis.color_stop_boxes])

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(
            "QSplitter::handle {"
            "  background: #d7dde4;"
            "  margin: 8px 2px;"
            "  border-radius: 4px;"
            "}"
            "QSplitter::handle:hover { background: #b8c2cd; }"
        )
        splitter.addWidget(left_scroll)
        splitter.addWidget(analysis_scroll)
        splitter.addWidget(self.analysis.view_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([FUSION_PANEL_W, ANALYSIS_PANEL_W, 900])
        outer.addWidget(splitter, 1)

        self._sync_algorithm_page()

    def _form_page(self, rows: list[tuple[str, QWidget]]) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(2, 2, 2, 2)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for label, widget in rows:
            form.addRow(label, widget)
        return page

    def _sync_algorithm_page(self) -> None:
        self.algo_stack.setCurrentIndex(self.algorithm.currentIndex())

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "加入曝光檔案",
            "",
            "影像 (*.npy *.png *.bmp *.tif *.tiff *.csv)",
        )
        if not paths:
            return
        self._add_paths([(Path(path), guess_exposure_us(Path(path))) for path in paths])

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇主資料夾")
        if not folder:
            return
        root = Path(folder)
        self._clear_spot_list()
        self._clear_exposure_session()
        added, reused = self._load_exposure_folder(root, keep_preprocess=False, offer_crop=True)
        if reused:
            self.stats.setText("已載入。張數不夠的曝光固定用最後一張：" + "、".join(reused))
        elif added:
            self.stats.setText(f"已載入主資料夾 {root.name}。")

    def _add_parent_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇父資料夾")
        if not folder:
            return
        parent = Path(folder)
        spots = self._scan_spots(parent)
        if not spots:
            QMessageBox.warning(
                self,
                "沒有點位",
                "這個父資料夾底下沒有含曝光影像的子資料夾。",
            )
            return
        dialog = SpotPickDialog(spots, None, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        spot = dialog.selected_spot()
        if spot is None:
            QMessageBox.warning(self, "尚未勾選", "請勾選一個點位。")
            return
        self.parent_root = parent
        self.spot_dirs = spots
        self.btn_spot.setEnabled(True)
        self._clear_exposure_session()
        self._refresh_source_label()
        added, reused = self._load_exposure_folder(
            spot, keep_preprocess=False, offer_crop=True
        )
        self._refresh_spot_label()
        note = f"已載入父資料夾，共 {len(spots)} 個點位，目前 {spot.name}。"
        if reused:
            note += " 張數不夠的曝光固定用最後一張：" + "、".join(reused)
        if added or reused:
            self.stats.setText(note)

    def _pick_spot(self) -> None:
        if not self.spot_dirs:
            QMessageBox.warning(self, "沒有點位", "請先載入父資料夾。")
            return
        current = self.source_root if self.parent_root is not None else None
        dialog = SpotPickDialog(self.spot_dirs, current, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        spot = dialog.selected_spot()
        if spot is None:
            QMessageBox.warning(self, "尚未勾選", "請勾選一個點位。")
            return
        if self.source_root is not None and spot.resolve() == self.source_root.resolve():
            return
        added, reused = self._load_exposure_folder(
            spot, keep_preprocess=True, offer_crop=True
        )
        self._refresh_spot_label()
        note = f"已切換點位 {spot.name}。"
        if reused:
            note += " 張數不夠的曝光固定用最後一張：" + "、".join(reused)
        if added or reused:
            self.stats.setText(note)

    def _clear_spot_list(self) -> None:
        self.parent_root = None
        self.spot_dirs = []
        self.btn_spot.setEnabled(False)
        self.spot_name.setText("尚未選擇")
        self.spot_name.setToolTip("")

    def _refresh_spot_label(self) -> None:
        if self.parent_root is None or self.source_root is None:
            self.spot_name.setText("尚未選擇")
            self.spot_name.setToolTip("")
            return
        self.spot_name.setText(self.source_root.name)
        self.spot_name.setToolTip(str(self.source_root))

    def _scan_spots(self, parent: Path) -> list[Path]:
        spots: list[Path] = []
        try:
            children = list(parent.iterdir())
        except OSError:
            return []
        for path in sorted(children, key=lambda item: _natural_name_key(item.name)):
            if not path.is_dir():
                continue
            if self._scan_source(path):
                spots.append(path)
        return spots

    @staticmethod
    def _exposure_key(time_us: float) -> float:
        return round(float(time_us), 6)

    def _clear_exposure_session(self) -> None:
        self._exposure_prefs.clear()
        self._removed_exposures.clear()

    def _sync_exposure_prefs_from_rows(self) -> None:
        for row in self.rows:
            self._exposure_prefs[self._exposure_key(row.slot_us)] = row.preprocess_state()

    def _load_exposure_folder(
        self,
        root: Path,
        *,
        keep_preprocess: bool,
        offer_crop: bool,
    ) -> tuple[int, list[str]]:
        groups = self._scan_source(root)
        if not groups:
            QMessageBox.warning(self, "沒有影像", "這個資料夾裡沒有可辨認曝光時間的影像檔。")
            return 0, []
        if not keep_preprocess:
            self._clear_exposure_session()
        else:
            self._sync_exposure_prefs_from_rows()
        index = int(self.frame_index.value())
        chosen: list[tuple[Path, float]] = []
        reused: list[str] = []
        skipped_removed: list[str] = []
        for time_us, files in sorted(groups.items()):
            key = self._exposure_key(time_us)
            if keep_preprocess and key in self._removed_exposures:
                skipped_removed.append(f"{time_us:g} us")
                continue
            pick = min(index, len(files) - 1)
            chosen.append((files[pick], time_us))
            if pick != index:
                reused.append(f"{time_us:g} us 只有 {len(files)} 張，用第 {pick} 筆")
        if not chosen:
            QMessageBox.warning(
                self,
                "沒有曝光",
                "這個點位的曝光都已在先前點位移除；請重新載入父資料夾，或先清掉曝光列。",
            )
            return 0, []
        memory = dict(self._exposure_prefs) if keep_preprocess else {}
        active = {self._exposure_key(time_us) for _, time_us in chosen}
        for row in list(self.rows):
            if self._exposure_key(row.slot_us) not in active:
                self._detach_row(row)
        self.crop_origin = None
        self.crop_size = None
        self.raw = None
        self.preview = None
        self.shown = None
        self.response = None
        self.batch_planes = []
        self.batch_labels = []
        self.btn_analyze.setEnabled(False)
        if self.batch_check.isChecked():
            self.batch_check.blockSignals(True)
            self.batch_check.setChecked(False)
            self.batch_check.blockSignals(False)
        self.batch_bar.setRange(0, 1)
        self.batch_bar.setValue(0)
        self.batch_bar.setFormat("尚未批次融合")
        self.source_root = root
        self.source_groups = groups
        self._refresh_source_label()
        if self.analysis is not None:
            self.analysis.set_source_folder(root)
        added = self._add_paths(
            chosen,
            compute=False,
            keep_preprocess=keep_preprocess,
            preprocess_memory=memory,
        )
        # Belt-and-suspenders: never leave a previously removed exposure row on screen.
        for row in list(self.rows):
            if self._exposure_key(row.slot_us) in self._removed_exposures:
                self._detach_row(row)
        self._refresh_row_file_counts()
        if offer_crop and added:
            self._offer_crop()
        if skipped_removed:
            reused.append("沿用已移除曝光（未載入）：" + "、".join(skipped_removed))
        return added, reused

    def _scan_source(self, root: Path) -> dict[float, list[Path]]:
        groups: dict[float, list[Path]] = {}
        for path in sorted(root.iterdir(), key=lambda item: _natural_name_key(item.name)):
            if not path.is_dir():
                continue
            time_us = exposure_us_from_name(path.name)
            if time_us is None:
                continue
            key = self._exposure_key(time_us)
            files = list_images(path)
            if files:
                # Same exposure key from two folder names: keep separate lists concatenated
                # only when keys truly match (normalized), never mix 50us with 10000us.
                groups.setdefault(key, []).extend(files)
        if groups:
            return groups
        for path in list_images(root):
            time_us = exposure_us_from_name(path.name)
            if time_us is None:
                continue
            key = self._exposure_key(time_us)
            groups.setdefault(key, []).append(path)
        return groups

    def _files_for_exposure(
        self,
        folders: dict[float, list[Path]],
        time_us: float,
    ) -> list[Path] | None:
        key = self._exposure_key(time_us)
        if key in folders:
            return folders[key]
        for stored, files in folders.items():
            if self._exposure_key(stored) == key:
                return files
        return None

    def _add_paths(
        self,
        items: list[tuple[Path, float | None]],
        compute: bool = True,
        *,
        keep_preprocess: bool = False,
        preprocess_memory: dict[float, tuple[float, float | None, float | None, bool]] | None = None,
    ) -> int:
        errors: list[str] = []
        added = 0
        memory = preprocess_memory or {}
        for path, time_us in items:
            if is_merged_name(path.name):
                continue
            if time_us is None or time_us <= 0:
                errors.append(f"{path.name}：檔名或資料夾名沒有可辨認的曝光時間（例如 500us）")
                continue
            slot = self._ensure_slot(float(time_us))
            try:
                image = load_hdr_image(path)
            except Exception as exc:
                errors.append(f"{path.name}：{exc}")
                continue
            slot.set_file(
                path,
                image,
                keep_preprocess=keep_preprocess,
                preprocess=self._memory_lookup(memory, float(time_us)),
            )
            added += 1
        if errors:
            QMessageBox.warning(self, "部分檔案沒有載入", "\n".join(errors))
        elif added == 0:
            self.stats.setText("沒有加入影像。")
        self._refresh_row_file_counts()
        if compute and added and len(self._selected()) == 1 and self.raw is None:
            self._compute()
        return added

    def _memory_lookup(self, memory: dict[float, tuple], time_us: float) -> tuple | None:
        key = self._exposure_key(time_us)
        if key in memory:
            return memory[key]
        for stored, value in memory.items():
            if abs(float(stored) - float(time_us)) < 1e-6:
                return value
        return None

    def _refresh_row_file_counts(self) -> None:
        folders = self.source_groups or {}
        for row in self.rows:
            files = self._files_for_exposure(folders, row.slot_us)
            row.set_file_count(len(files) if files else (1 if row.image is not None else 0))

    def _slot_for(self, time_us: float | None) -> ExposureRow | None:
        if time_us is None:
            return None
        key = self._exposure_key(time_us)
        for row in self.rows:
            if self._exposure_key(row.slot_us) == key:
                return row
        return None

    def _detach_row(self, row: ExposureRow) -> None:
        if row in self.rows:
            self.rows.remove(row)
        self.file_layout.removeWidget(row)
        row.deleteLater()
        self._sync_file_scroll_height()

    def _remove_row(self, row: ExposureRow) -> None:
        key = self._exposure_key(row.slot_us)
        self._removed_exposures.add(key)
        self._exposure_prefs.pop(key, None)
        if self.source_groups:
            for stored in list(self.source_groups):
                if self._exposure_key(stored) == key:
                    self.source_groups.pop(stored, None)
        self._detach_row(row)
        self._invalidate_fusion_after_exposure_change()
        if not self.rows:
            self.stats.setText("已移除全部曝光列。")

    def _invalidate_fusion_after_exposure_change(self) -> None:
        """Drop stale batch / preview so removed exposures cannot keep driving frame count."""
        self.batch_planes = []
        self.batch_labels = []
        self.raw = None
        self.preview = None
        self.shown = None
        self.response = None
        self.btn_analyze.setEnabled(False)
        if self.batch_check.isChecked():
            self.batch_check.blockSignals(True)
            self.batch_check.setChecked(False)
            self.batch_check.blockSignals(False)
        self.batch_bar.setRange(0, 1)
        self.batch_bar.setValue(0)
        self.batch_bar.setFormat("曝光已變更，請重新計算")
        if self.analysis is not None:
            self.analysis.set_batch([], [])

    def _sync_file_scroll_height(self) -> None:
        n = len(self.rows)
        gap = max(n - 1, 0) * 8
        row_h = ExposureRow.ROW_HEIGHT
        self.file_scroll.setFixedHeight(n * row_h + gap)
        self.file_scroll.setVisible(n > 0)
        self.file_empty.setVisible(n == 0)

    def _ensure_slot(self, time_us: float) -> ExposureRow:
        found = self._slot_for(time_us)
        if found is not None:
            return found
        row = ExposureRow(time_us, self._remove_row, self._on_preprocess_changed)
        self.rows.append(row)
        self.rows.sort(key=lambda item: item.slot_us)
        while self.file_layout.count():
            self.file_layout.takeAt(0)
        for item in self.rows:
            self.file_layout.addWidget(item)
        self._sync_file_scroll_height()
        return row

    def _clear_rows(self) -> None:
        for row in list(self.rows):
            self._detach_row(row)
        self._clear_exposure_session()
        self.crop_origin = None
        self.crop_size = None
        self.source_root = None
        self.source_groups = None
        self.batch_planes = []
        self.batch_labels = []
        self._clear_spot_list()
        self._refresh_source_label()
        if self.analysis is not None:
            self.analysis.set_source_folder(None)

    def _crop_source_image(self) -> np.ndarray | None:
        for row in self.rows:
            if row.image is not None:
                return row.image
        return None

    def _offer_crop(self) -> bool:
        image = self._crop_source_image()
        if image is None:
            QMessageBox.warning(self, "沒有影像", "請先載入影像。")
            return False
        dialog = CropDialog(image, self.crop_origin, self.crop_size, self)
        if dialog.exec_() != QDialog.Accepted:
            return False
        origin = dialog.origin()
        size = dialog.crop_size()
        crop_w, crop_h = int(size[0]), int(size[1])
        if crop_w < 1 or crop_h < 1:
            QMessageBox.warning(self, "範圍無效", "寬和高都必須至少 1 像素。")
            return False
        for row in self.rows:
            if row.image is None:
                continue
            row_h, row_w = int(row.image.shape[0]), int(row.image.shape[1])
            if origin[0] + crop_w > row_w or origin[1] + crop_h > row_h:
                QMessageBox.warning(self, "範圍超出影像", "這個範圍超出其中一張曝光。")
                return False
        self.crop_origin = origin
        self.crop_size = (crop_w, crop_h)
        self.stats.setText(f"計算範圍 ({origin[0]}, {origin[1]})  {crop_w}×{crop_h}")
        return True

    def _adjust_crop(self) -> None:
        self._offer_crop()

    def _crop_image(self, image: np.ndarray) -> np.ndarray:
        if self.crop_origin is None or self.crop_size is None:
            raise RuntimeError("請先選擇計算範圍。")
        x, y = self.crop_origin
        crop_w, crop_h = self.crop_size
        data = np.asarray(image)
        height, width = int(data.shape[0]), int(data.shape[1])
        if x < 0 or y < 0 or x + crop_w > width or y + crop_h > height:
            raise RuntimeError("計算範圍超出影像。")
        return data[y : y + crop_h, x : x + crop_w]

    def _selected(self) -> list[ExposureRow]:
        return [row for row in self.rows if row.image is not None and row.use.isChecked()]

    def _algorithm_settings(self) -> dict[str, float | bool]:
        settings: dict[str, float | bool] = {
            "samples": int(self.debevec_samples.value()),
            "lambda": float(self.debevec_lambda.value()),
            "random": bool(self.debevec_random.isChecked()),
            "max_iter": int(self.robertson_iter.value()),
            "threshold": float(self.robertson_threshold.value()),
            "contrast": float(self.mertens_contrast.value()),
            "saturation": float(self.mertens_saturation.value()),
            "exposure": float(self.mertens_exposure.value()),
            "sigma": float(self.local_sigma.value()),
        }
        if str(self.algorithm.currentData()) == "mertens_local":
            settings["contrast"] = float(self.local_contrast.value())
            settings["saturation"] = float(self.local_saturation.value())
            settings["exposure"] = float(self.local_exposure.value())
        return settings

    def _compute(self) -> None:
        # Batch checkbox → raw-frame batch. Otherwise one fuse (average-N applied if set).
        if self.batch_check.isChecked():
            self._compute_batch()
            return
        selected = self._selected()
        if not selected:
            QMessageBox.warning(self, "沒有影像", "請至少勾選一張影像。")
            return
        if (self.crop_origin is None or self.crop_size is None) and not self._offer_crop():
            QMessageBox.warning(self, "尚未框選", "請先選擇計算範圍。")
            return
        try:
            series, notes = self._prepare_exposure_series(selected)
            images = [planes[0] for planes in series]
            raw, response, single = self._fuse_images(selected, images)
        except RuntimeError as exc:
            QMessageBox.warning(self, "無法計算", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "計算失敗", str(exc))
            return
        self.batch_planes = []
        self.batch_labels = []
        self._show_fused(raw, response, single, selected)
        if notes:
            self.stats.setText("\n".join(notes) + "\n" + self.stats.text())

    def _on_batch_toggled(self, checked: bool) -> None:
        if self._batch_running:
            self._batch_stop = not checked
            return
        if checked:
            if not self._compute_batch():
                self.batch_check.blockSignals(True)
                self.batch_check.setChecked(False)
                self.batch_check.blockSignals(False)

    def _batch_sources(self) -> tuple[dict[float, list[Path]], list[ExposureRow], int]:
        if self.source_groups:
            raw_folders = self.source_groups
        elif self.source_root is not None and self.source_root.is_dir():
            raw_folders = self._scan_source(self.source_root)
        else:
            raise RuntimeError("請先用「載入主資料夾」或「載入父資料夾」載入資料。")
        selected = self._selected()
        if not selected:
            raise RuntimeError("請至少勾選一個曝光。")
        # Only the currently checked rows participate. Removed exposures must not
        # inflate the batch length (e.g. max(1, 300) when 10000us was left in groups).
        folders: dict[float, list[Path]] = {}
        missing: list[str] = []
        for row in selected:
            files = self._files_for_exposure(raw_folders, row.slot_us)
            if not files:
                missing.append(f"{row.slot_us:g} us")
                continue
            folders[self._exposure_key(row.slot_us)] = files
        if missing:
            raise RuntimeError("這些曝光在目前點位／主資料夾裡沒有影像：" + "、".join(missing))
        count = max(len(files) for files in folders.values())
        if count < 1:
            raise RuntimeError("目前點位／主資料夾裡沒有可融合的影像。")
        return folders, selected, count

    def _compute_batch(self) -> bool:
        if self._batch_running:
            return False
        if (self.crop_origin is None or self.crop_size is None) and not self._offer_crop():
            return False
        try:
            folders, selected, count = self._batch_sources()
        except RuntimeError as exc:
            QMessageBox.warning(self, "無法批次融合", str(exc))
            return False
        self._batch_running = True
        self._batch_stop = False
        self.btn_run.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.batch_bar.setRange(0, count)
        self.batch_bar.setValue(0)
        width_x, width_y, sigma = self._smooth_widths()
        algorithm = str(self.algorithm.currentData())
        settings = self._algorithm_settings()
        full_scale = float(self.full_scale.value())
        planes: list[np.ndarray] = []
        labels: list[str] = []
        shown_raw = None
        shown_response = None
        shown_single = False
        target = int(self.frame_index.value())
        canceled = False
        failed_at: int | None = None
        try:
            for index in range(count):
                self.batch_bar.setFormat(f"正在融合第 {index + 1} / {count} 筆")
                QApplication.processEvents()
                if self._batch_stop:
                    canceled = True
                    break
                try:
                    images = []
                    times = []
                    for row in selected:
                        files = self._files_for_exposure(folders, row.slot_us) or []
                        path = files[min(index, len(files) - 1)]
                        image = self._crop_image(load_hdr_image(path))
                        images.append(preprocess_exposure(
                            image,
                            float(row.gain.value()),
                            row.color_floor.number(),
                            row.color_ceil.number(),
                            width_x,
                            width_y,
                            sigma,
                        ))
                        times.append(float(row.time.value()))
                    if len(images) == 1:
                        raw = np.asarray(images[0], dtype=np.float32)
                        response = None
                        single = True
                    else:
                        raw, response = run_hdr(images, times, algorithm, full_scale, settings)
                        single = False
                except Exception as exc:
                    QMessageBox.warning(self, "批次融合失敗", f"第 {index} 筆：{exc}")
                    failed_at = index
                    break
                plane = self._analysis_from_raw(raw, single)
                planes.append(np.asarray(plane, dtype=np.float32))
                label_row = max(
                    selected,
                    key=lambda row: len(self._files_for_exposure(folders, row.slot_us) or []),
                )
                label_files = self._files_for_exposure(folders, label_row.slot_us) or []
                labels.append(label_files[min(index, len(label_files) - 1)].name)
                if shown_raw is None or index == target:
                    shown_raw = raw
                    shown_response = response
                    shown_single = single
                self.batch_bar.setValue(index + 1)
        finally:
            self._batch_running = False
            self.btn_run.setEnabled(True)
            if self.raw is not None or self.batch_planes:
                self.btn_analyze.setEnabled(True)
        if not planes or shown_raw is None:
            self.batch_bar.setFormat("批次融合失敗" if failed_at is not None else "尚未批次融合")
            self.batch_bar.setValue(0)
            return False
        self.batch_planes = planes
        self.batch_labels = labels
        if not self.batch_check.isChecked():
            self.batch_check.blockSignals(True)
            self.batch_check.setChecked(True)
            self.batch_check.blockSignals(False)
        self._show_fused(shown_raw, shown_response, shown_single, selected)
        done = len(planes)
        if failed_at is not None:
            self.batch_bar.setFormat(f"第 {failed_at} 筆失敗，已保留 {done} 筆")
            self.stats.setText(
                f"第 {failed_at} 筆失敗，已保留前面 {done} 筆。\n" + self.stats.text()
            )
        elif canceled:
            self.batch_bar.setFormat(f"已停止 {done} / {count}")
            self.stats.setText(
                f"批次融合已停止，完成 {done} / {count} 筆。\n" + self.stats.text()
            )
        else:
            self.batch_bar.setFormat(f"已完成 {done} / {count}")
            self.stats.setText(f"批次融合完成 {done} 筆。\n" + self.stats.text())
        self.btn_analyze.setEnabled(True)
        return True

    def _fuse_rows(self, selected: list[ExposureRow]) -> tuple[np.ndarray, np.ndarray | None, bool]:
        width_x, width_y, sigma = self._smooth_widths()
        images = [self._preprocess_row(row, width_x, width_y, sigma) for row in selected]
        return self._fuse_images(selected, images)

    def _fuse_images(
        self,
        selected: list[ExposureRow],
        images: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        algorithm = str(self.algorithm.currentData())
        if len(images) == 1:
            return np.asarray(images[0], dtype=np.float32), None, True
        raw, response = run_hdr(
            images,
            [float(row.time.value()) for row in selected],
            algorithm,
            float(self.full_scale.value()),
            self._algorithm_settings(),
        )
        return raw, response, False

    def _prepare_exposure_series(
        self,
        selected: list[ExposureRow],
    ) -> tuple[list[list[np.ndarray]], list[str]]:
        """One preprocessed plane per exposure (chunk-averaged when N is set)."""
        folders = self.source_groups
        if not folders and self.source_root is not None and self.source_root.is_dir():
            folders = self._scan_source(self.source_root)
        folders = folders or {}
        width_x, width_y, sigma = self._smooth_widths()
        frame_index = int(self.frame_index.value())
        series: list[list[np.ndarray]] = []
        notes: list[str] = []
        for row in selected:
            files = self._files_for_exposure(folders, row.slot_us)
            if not files:
                if row.image is None:
                    raise RuntimeError(f"{row.slot_us:g} us 沒有影像。")
                files = [row.path] if row.path is not None else []
                if not files:
                    raise RuntimeError(f"{row.slot_us:g} us 沒有影像路徑。")
            n = row.average_count()
            if n is None:
                pick = min(frame_index, len(files) - 1)
                image = self._crop_image(load_hdr_image(files[pick]))
                series.append(
                    [
                        preprocess_exposure(
                            image,
                            float(row.gain.value()),
                            row.color_floor.number(),
                            row.color_ceil.number(),
                            width_x,
                            width_y,
                            sigma,
                        )
                    ]
                )
                continue
            if n > len(files):
                raise RuntimeError(
                    f"{row.slot_us:g} us 加總平均 N={n} 大於總張數 {len(files)}。"
                )
            chunks = len(files) // n
            if chunks < 1:
                raise RuntimeError(f"{row.slot_us:g} us 加總平均後沒有完整段。")
            leftover = len(files) - chunks * n
            chunk_i = min(frame_index, chunks - 1)
            chunk_paths = files[chunk_i * n : (chunk_i + 1) * n]
            stack = [
                np.asarray(self._crop_image(load_hdr_image(path)), dtype=np.float64)
                for path in chunk_paths
            ]
            mean = np.mean(np.stack(stack, axis=0), axis=0)
            series.append(
                [
                    preprocess_exposure(
                        mean,
                        float(row.gain.value()),
                        row.color_floor.number(),
                        row.color_ceil.number(),
                        width_x,
                        width_y,
                        sigma,
                    )
                ]
            )
            note = (
                f"{row.slot_us:g} us：N={n}，共 {chunks} 段，"
                f"用第 {chunk_i + 1} 段（{chunk_i * n}–{chunk_i * n + n - 1}）加總平均後單次融合。"
            )
            if leftover:
                note += f" 餘 {leftover} 張丟掉。"
            notes.append(note)
        return series, notes

    def _analysis_from_raw(self, raw: np.ndarray, single: bool) -> np.ndarray:
        plane = self._as_plane(raw)
        if plane is None:
            raise RuntimeError("沒有可分析的結果。")
        algorithm = str(self.algorithm.currentData())
        if single or algorithm not in {"mertens", "mertens_local"}:
            return plane * 255.0
        return plane

    def _show_fused(
        self,
        raw: np.ndarray,
        response: np.ndarray | None,
        single: bool,
        selected: list[ExposureRow],
    ) -> None:
        algorithm = str(self.algorithm.currentData())
        width_x, width_y, sigma = self._smooth_widths()
        self.raw = raw
        self.response = response
        self.preview = raw
        self._export_full_scale = single or algorithm not in {"mertens", "mertens_local"}
        self._draw_preview(raw)
        self._draw_response(response)
        self.btn_export.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self._push_analysis(switch_view=False)
        finite = raw[np.isfinite(raw)]
        missing = int(np.size(raw) - finite.size)
        if finite.size == 0:
            span = "沒有有限值"
        else:
            span = f"最小 {finite.min():.6g}，最大 {finite.max():.6g}"
            if missing:
                span += f"，另有 {missing} 個非有限值"
        steps = [self._exposure_prep_text(row) for row in selected]
        steps = [text for text in steps if text]
        if self.smooth_check.isChecked():
            steps.append(f"平滑 {width_x}×{width_y} σ={sigma:g}")
        prep = ("；輸入先" + "、".join(steps)) if steps else ""
        if single:
            title = f"單張顯示  {selected[0].path.name}"
            kind = "原始影像"
        else:
            title = self.algorithm.currentText()
            kind = "融合結果" if algorithm in {"mertens", "mertens_local"} else "輻射圖"
        scale = float(self.full_scale.value())
        if self._export_full_scale:
            saved = f"匯出 npy 會再 × 滿量 {scale:g}。"
        else:
            saved = "Mertens 結果已 × 255，回到 8-bit 尺度。匯出 npy 就是這個值，不再乘滿量。"
        self.stats.setText(
            f"{title} {kind}\n形狀 {'×'.join(str(size) for size in raw.shape)}，{span}{prep}\n"
            f"畫面和游標是這個浮點結果（已含輸入的平滑和 color scale）。"
            f"{saved}勾 log 只改畫面和熱圖匯出。"
        )

    def _smooth_widths(self) -> tuple[int, int, float]:
        if not self.smooth_check.isChecked():
            return 1, 1, 0.0
        return int(self.smooth_kx.value()), int(self.smooth_ky.value()), float(self.smooth_sigma.value())

    def _preprocess_row(self, row: ExposureRow, width_x: int, width_y: int, sigma: float) -> np.ndarray:
        return preprocess_exposure(
            self._crop_image(row.image),
            float(row.gain.value()),
            row.color_floor.number(),
            row.color_ceil.number(),
            width_x,
            width_y,
            sigma,
        )

    def _exposure_prep_text(self, row: ExposureRow) -> str:
        parts: list[str] = []
        gain = float(row.gain.value())
        if abs(gain - 1.0) > 1e-9:
            parts.append(f"倍率 {gain:g}")
        floor = row.color_floor.number()
        ceil = row.color_ceil.number()
        if floor is not None or ceil is not None:
            floor_text = "—" if floor is None else f"{floor:g}"
            ceil_text = "—" if ceil is None else f"{ceil:g}"
            parts.append(f"color scale {floor_text}–{ceil_text}")
        if not parts:
            return ""
        return f"{row.slot_us:g} us " + " ".join(parts)

    def _on_smooth_toggled(self, checked: bool) -> None:
        for box in (self.smooth_kx, self.smooth_ky, self.smooth_sigma):
            box.setEnabled(checked)
        self._on_preprocess_changed()

    def _on_preprocess_changed(self, *_args) -> None:
        self._sync_exposure_prefs_from_rows()
        if self.raw is None or self.batch_check.isChecked() or self._batch_running:
            return
        if len(self._selected()) >= 1:
            self._compute()

    def _on_log_toggled(self, _checked: bool) -> None:
        self._manual_x = False
        self._manual_y = False
        self._show_current()
        QTimer.singleShot(0, self._rescale_profile_intensity)

    def _rescale_profile_intensity(self) -> None:
        """Fit both profile intensity axes to the curves now on screen."""
        if self._x_values is not None:
            self._fit_intensity(self.plot_x, self._x_values, "y")
        if self._y_values is not None:
            self._fit_intensity(self.plot_y, self._y_values, "x")

    def eventFilter(self, watched, event):
        if watched is self.win and event.type() in (QEvent.Resize, QEvent.Show):
            self._queue_profile_sync()
        return super().eventFilter(watched, event)

    def _queue_profile_sync(self) -> None:
        if self._profile_sync_pending or self._profile_syncing:
            return
        self._profile_sync_pending = True
        QTimer.singleShot(0, self._sync_profile_thickness)

    def _sync_profile_thickness(self) -> None:
        """Make the Y profile curve as wide as the X profile curve is tall."""
        self._profile_sync_pending = False
        if self._profile_syncing:
            return
        self._profile_syncing = True
        try:
            layout = self.win.ci.layout
            for _ in range(4):
                target = float(self.plot_x.vb.sceneBoundingRect().height())
                current = float(self.plot_y.vb.sceneBoundingRect().width())
                if target < 48 or abs(current - target) < 2:
                    return
                width = float(self.plot_y.sceneBoundingRect().width()) + (target - current)
                if width < 80:
                    return
                layout.setColumnFixedWidth(0, width)
                layout.activate()
        finally:
            self._profile_syncing = False

    def _pin_axis_frame(self, plot, left: int, bottom: int) -> None:
        """Keep tick text from resizing the plot when the numbers get longer."""
        for name in ("left", "right", "top", "bottom"):
            plot.getAxis(name).setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        plot.getAxis("left").setWidth(left)
        plot.getAxis("bottom").setHeight(bottom)

    def _install_y_mode_bar(self, layout) -> None:
        frame, row = function_frame()
        tip = (
            "Y 剖面是十字所在那一行的強度。"
            "Y 累加是計算範圍（ROI）內每一列沿 X 加總。"
            "有勾 log 時，累加是先加總線性值再取 log10。一次只能選一個。"
        )
        style = (
            "QPushButton {"
            "background: #ffffff; color: #333333;"
            "border: 1px solid #d5dbe2; border-radius: 6px;"
            "padding: 2px 8px; min-height: 18px;"
            "}"
            "QPushButton:checked {"
            "background: #2c333a; color: #ffffff; border-color: #2c333a;"
            "}"
        )
        self.y_slice_btn = QPushButton("Y 剖面")
        self.y_sum_btn = QPushButton("Y 累加")
        button_metrics = QFontMetrics(self.y_slice_btn.font())
        button_w = max(
            button_metrics.horizontalAdvance("Y 剖面"),
            button_metrics.horizontalAdvance("Y 累加"),
        ) + 24
        self.y_mode_group = QButtonGroup(frame)
        self.y_mode_group.setExclusive(True)
        for button in (self.y_slice_btn, self.y_sum_btn):
            button.setCheckable(True)
            button.setFixedWidth(button_w)
            button.setStyleSheet(style)
            button.setToolTip(tip)
            button.setCursor(Qt.PointingHandCursor)
            self.y_mode_group.addButton(button)
            row.addWidget(button)
        self.y_slice_btn.setChecked(True)
        self.y_caption = QLabel("")
        caption_w = QFontMetrics(self.y_caption.font()).horizontalAdvance("ROI 範圍加總") + 8
        self.y_caption.setFixedWidth(caption_w)
        self.y_caption.setStyleSheet("color: #5c6570; background: transparent;")
        row.addWidget(self.y_caption)
        self.y_mode_group.buttonClicked.connect(self._on_y_mode_changed)
        layout.addWidget(frame)

    def _y_is_sum(self) -> bool:
        return bool(self.y_sum_btn.isChecked())

    def _on_y_mode_changed(self, _button=None) -> None:
        if self.shown is None:
            return
        self._manual_y = False
        self._update_crosshair(reset_intensity=False)
        if self._y_values is not None:
            self._fit_intensity(self.plot_y, self._y_values, "x")

    def _show_current(self) -> None:
        if self.raw is None:
            return
        self.preview = self.raw
        self._draw_preview(self.raw)

    def _as_plane(self, image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        data = np.asarray(image, dtype=np.float64)
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        return data

    def _linear_plane(self) -> np.ndarray | None:
        """Intensity before optional log."""
        if self.log_check.isChecked():
            return self._as_plane(self.raw if self.raw is not None else self.preview)
        return self._as_plane(self.preview if self.preview is not None else self.raw)

    def _display_plane(self, preview: np.ndarray) -> np.ndarray:
        """Build the heatmap / profile plane. Log uses the float result."""
        if self.log_check.isChecked():
            source = self._as_plane(self.raw if self.raw is not None else preview)
            assert source is not None
            return display_log10(source)
        shown = self._as_plane(preview)
        assert shown is not None
        return shown

    def _draw_preview(self, preview: np.ndarray) -> None:
        shown = self._display_plane(preview)
        previous_shape = None if self.shown is None else self.shown.shape
        self.shown = shown
        finite = shown[np.isfinite(shown)]
        if finite.size:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        self.image_item.setImage(shown.T, autoLevels=False)
        self._set_color_levels(lo, hi)
        intensity = "log10 強度" if self.log_check.isChecked() else "強度"
        self.plot_x.setLabel("left", intensity)
        self.plot_y.setLabel("bottom", intensity)
        height, width = shown.shape
        reset_view = (
            previous_shape != shown.shape
            or self.cross_xy is None
            or not (0 <= self.cross_xy[0] < width and 0 <= self.cross_xy[1] < height)
        )
        if reset_view:
            self.cross_xy = self._brightest(shown)
            self._manual_x = False
            self._manual_y = False
        self._update_crosshair(reset_intensity=True)
        if reset_view:
            self._reset_heatmap_view()
            self._reset_profile_view("x")
            self._reset_profile_view("y")

    def _brightest(self, shown: np.ndarray) -> tuple[int, int]:
        finite = np.isfinite(shown)
        if not finite.any():
            return 0, 0
        index = int(np.argmax(np.where(finite, shown, -np.inf)))
        y, x = np.unravel_index(index, shown.shape)
        return int(x), int(y)

    def _update_crosshair(self, reset_intensity: bool = False) -> None:
        if self.shown is None or self.cross_xy is None:
            return
        x, y = self.cross_xy
        height, width = self.shown.shape
        row = np.nan_to_num(self.shown[y, :], nan=0.0)
        self._x_values = row
        self.curve_x.setData(np.arange(width) + 0.5, row)
        self.marker_x.setData([x + 0.5], [float(row[x])])
        self.vline.setPos(x + 0.5)
        self.hline.setPos(y + 0.5)
        self.vline.show()
        self.hline.show()
        self.plot_x.setTitle(f"X 剖面  y={y}")

        y_sum = self._y_is_sum()
        if y_sum:
            linear = self._linear_plane()
            if linear is None:
                return
            # Result plane is already the crop ROI.
            summed = np.nan_to_num(linear.sum(axis=1), nan=0.0)
            col = display_log10(summed) if self.log_check.isChecked() else summed
            self.y_caption.setText("ROI 範圍加總")
            intensity = "log10 累加強度" if self.log_check.isChecked() else "累加強度"
        else:
            col = np.nan_to_num(self.shown[:, x], nan=0.0)
            self.y_caption.setText(f"x={x}")
            intensity = "log10 強度" if self.log_check.isChecked() else "強度"
        self._y_values = col
        self.curve_y.setData(col, np.arange(height) + 0.5)
        self.marker_y.setData([float(col[y])], [y + 0.5])
        self.plot_y.setLabel("bottom", intensity)
        self.plot_x.setLabel("left", "log10 強度" if self.log_check.isChecked() else "強度")

        if reset_intensity:
            self._manual_x = False
            if not y_sum:
                self._manual_y = False
        if reset_intensity or not self._manual_x:
            self._fit_intensity(self.plot_x, row, "y")
        if not y_sum and (reset_intensity or not self._manual_y):
            self._fit_intensity(self.plot_y, col, "x")

    def _fit_intensity(self, plot, values: np.ndarray, axis: str) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
            pad = (hi - lo) * 0.05
            lo -= pad
            hi += pad
        plot.enableAutoRange(x=False, y=False)
        if axis == "y":
            plot.setYRange(lo, hi, padding=0)
        else:
            plot.setXRange(lo, hi, padding=0)

    def _reset_heatmap_view(self) -> None:
        if self.shown is None:
            return
        height, width = self.shown.shape
        self.plot.getViewBox().setRange(xRange=(0, width), yRange=(0, height), padding=0)

    def _reset_profile_view(self, which: str) -> None:
        if self.shown is None:
            return
        height, width = self.shown.shape
        if which == "x":
            self._manual_x = False
            self.plot_x.setXRange(0, width, padding=0)
            if self._x_values is not None:
                self._fit_intensity(self.plot_x, self._x_values, "y")
        else:
            self._manual_y = False
            self.plot_y.setYRange(0, height, padding=0)
            if self._y_values is not None:
                self._fit_intensity(self.plot_y, self._y_values, "x")

    def _reset_color_levels(self) -> None:
        if self.shown is None:
            return
        shown = self.shown
        finite = shown[np.isfinite(shown)]
        if finite.size:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        self._set_color_levels(lo, hi)

    def _make_color_box(self, prefix: str, tooltip: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e15, 1e15)
        box.setDecimals(2)
        box.setKeyboardTracking(False)
        box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        box.setPrefix(prefix)
        box.setToolTip(tooltip)
        box.setSpecialValueText("--")
        box.setValue(box.minimum())
        box.setFixedWidth(148)
        return box

    def _make_stop_box(self, name: str, pos: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 1.0)
        box.setDecimals(2)
        box.setSingleStep(0.01)
        box.setKeyboardTracking(False)
        box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        box.setPrefix(f"{name}: ")
        box.setToolTip(f"色條上{name}色三角的位置。0 在下方，1 在上方。只改顏色分佈，不重算演算法。")
        box.setValue(float(pos))
        box.setFixedWidth(80)
        return box

    def _ordered_stops(self, values: list[float]) -> list[float]:
        gap = 0.01
        ordered = [min(1.0, max(0.0, float(value))) for value in values]
        for index in range(1, len(ordered)):
            if ordered[index] < ordered[index - 1] + gap:
                ordered[index] = min(1.0, ordered[index - 1] + gap)
        for index in range(len(ordered) - 2, -1, -1):
            if ordered[index] > ordered[index + 1] - gap:
                ordered[index] = max(0.0, ordered[index + 1] - gap)
        return ordered

    def _on_color_stop_changed(self, _value: float | None = None) -> None:
        if self._stops_updating:
            return
        self._apply_color_stops([box.value() for box in self.color_stop_boxes])

    def _apply_color_stops(self, values: list[float]) -> None:
        ordered = self._ordered_stops(values)
        self._stops_updating = True
        try:
            self.hist.gradient.setColorMap(colormap_from_stops(ordered))
            for box, pos in zip(self.color_stop_boxes, ordered):
                box.blockSignals(True)
                box.setValue(pos)
                box.blockSignals(False)
        finally:
            self._stops_updating = False

    def _on_gradient_ticks_moved(self, *_args) -> None:
        if self._stops_updating or not self.color_stop_boxes:
            return
        by_color = {
            color: box
            for box, (_name, _en, color, _pos) in zip(self.color_stop_boxes, COLOR_STOPS)
        }
        self._stops_updating = True
        try:
            for tick, value in self.hist.gradient.listTicks():
                color = tick.color
                key = (color.red(), color.green(), color.blue())
                box = by_color.get(key)
                if box is None:
                    continue
                box.blockSignals(True)
                box.setValue(float(value))
                box.blockSignals(False)
        finally:
            self._stops_updating = False

    def _set_color_levels(self, lo: float, hi: float) -> None:
        if lo >= hi:
            hi = lo + 1.0
        self._color_updating = True
        try:
            self.hist.setHistogramRange(lo, hi, padding=0)
            self.hist.setLevels(lo, hi)
            self.color_min.blockSignals(True)
            self.color_max.blockSignals(True)
            step = max((hi - lo) / 100.0, 1e-6)
            self.color_min.setSingleStep(step)
            self.color_max.setSingleStep(step)
            self.color_min.setValue(lo)
            self.color_max.setValue(hi)
        finally:
            self.color_min.blockSignals(False)
            self.color_max.blockSignals(False)
            self._color_updating = False

    def _on_color_limit_changed(self, _value: float | None = None) -> None:
        if self._color_updating or self.shown is None:
            return
        lo = float(self.color_min.value())
        hi = float(self.color_max.value())
        current = self.hist.getLevels()
        cur_lo, cur_hi = float(current[0]), float(current[1])
        span = max(cur_hi - cur_lo, 1.0)
        if lo >= hi:
            if abs(hi - cur_hi) >= abs(lo - cur_lo):
                lo = hi - span
            else:
                hi = lo + span
        self._set_color_levels(lo, hi)

    def _on_hist_levels_changed(self, *_args) -> None:
        if self._color_updating:
            return
        lo, hi = self.hist.getLevels()
        self._color_updating = True
        try:
            self.color_min.blockSignals(True)
            self.color_max.blockSignals(True)
            self.color_min.setValue(float(lo))
            self.color_max.setValue(float(hi))
        finally:
            self.color_min.blockSignals(False)
            self.color_max.blockSignals(False)
            self._color_updating = False

    def _write_limits(self, which: str, lo: float, hi: float) -> None:
        if lo >= hi:
            QMessageBox.warning(self, "上下限", "下限必須小於上限。")
            return
        if which == "x":
            self._manual_x = True
            self.plot_x.setYRange(lo, hi, padding=0)
        else:
            self._manual_y = True
            self.plot_y.setXRange(lo, hi, padding=0)

    def _intensity_axis_hit(self, pos) -> tuple[str, str] | None:
        hit = self._axis_end_hit(self.plot_x, "left", pos)
        if hit is not None:
            return "x", hit
        hit = self._axis_end_hit(self.plot_y, "bottom", pos)
        if hit is not None:
            return "y", hit
        return None

    @staticmethod
    def _axis_end_hit(plot, axis_name: str, pos) -> str | None:
        axis = plot.getAxis(axis_name)
        view_rect = plot.vb.sceneBoundingRect()
        axis_rect = axis.mapRectToScene(axis.boundingRect())
        if axis_name == "bottom":
            strip = QRectF(
                view_rect.left() - 24,
                view_rect.bottom() - 2,
                view_rect.width() + 48,
                max(12.0, axis_rect.bottom() - view_rect.bottom() + 8),
            )
        else:
            strip = QRectF(
                axis_rect.left() - 4,
                view_rect.top() - 16,
                max(12.0, view_rect.left() - axis_rect.left() + 8),
                view_rect.height() + 32,
            )
        if not strip.contains(pos):
            return None
        if axis_name == "bottom":
            data_range = plot.viewRange()[0]
            y_anchor = plot.viewRange()[1][0]
            min_scene = plot.vb.mapViewToScene(pg.Point(data_range[0], y_anchor))
            max_scene = plot.vb.mapViewToScene(pg.Point(data_range[1], y_anchor))
            d_min = abs(pos.x() - min_scene.x())
            d_max = abs(pos.x() - max_scene.x())
        else:
            data_range = plot.viewRange()[1]
            x_anchor = plot.viewRange()[0][0]
            min_scene = plot.vb.mapViewToScene(pg.Point(x_anchor, data_range[0]))
            max_scene = plot.vb.mapViewToScene(pg.Point(x_anchor, data_range[1]))
            d_min = abs(pos.y() - min_scene.y())
            d_max = abs(pos.y() - max_scene.y())
        if min(d_min, d_max) > 36:
            return None
        return "min" if d_min <= d_max else "max"

    def _prompt_axis_limit(self, which: str, end: str) -> None:
        if which == "x":
            lo, hi = self.plot_x.viewRange()[1]
            axis_name = "X 剖面強度"
        else:
            lo, hi = self.plot_y.viewRange()[0]
            axis_name = "Y 剖面強度"
        if end == "min":
            title = "下限"
            current = float(lo)
        else:
            title = "上限"
            current = float(hi)
        value, ok = ask_scientific(self, title, f"{axis_name}{title}", current)
        if not ok:
            return
        if end == "min":
            lo = float(value)
        else:
            hi = float(value)
        self._write_limits(which, float(lo), float(hi))

    def _on_double_click(self, pos) -> None:
        if self._item_contains(self.hist, pos):
            self._reset_color_levels()
            return
        if self.plot_x.sceneBoundingRect().contains(pos):
            self._reset_profile_view("x")
            return
        if self.plot_y.sceneBoundingRect().contains(pos):
            self._reset_profile_view("y")
            return
        if self.plot.sceneBoundingRect().contains(pos):
            self._reset_heatmap_view()

    @staticmethod
    def _item_contains(item, pos) -> bool:
        try:
            return item.sceneBoundingRect().contains(pos)
        except Exception:
            return False

    def _draw_response(self, response: np.ndarray | None) -> None:
        self.response_plot.clear()
        if response is None:
            if self.raw is not None and len(self._selected()) == 1:
                self.response_plot.setTitle("單張顯示沒有響應曲線")
            else:
                name = self.algorithm.currentText()
                self.response_plot.setTitle(f"{name} 沒有響應曲線")
            return
        curve = np.squeeze(response)
        pens = ["#555555", "#cc3333", "#33aa33", "#3366cc"]
        if curve.ndim == 1:
            self.response_plot.plot(np.arange(curve.shape[0]), curve, pen=pg.mkPen(pens[0], width=2))
        elif curve.ndim == 2:
            for channel in range(curve.shape[1]):
                self.response_plot.plot(
                    np.arange(curve.shape[0]),
                    curve[:, channel],
                    pen=pg.mkPen(pens[channel % len(pens)], width=2),
                )
        self.response_plot.setTitle("相機響應")

    @staticmethod
    def _make_profile_tip() -> pg.TextItem:
        tip = pg.TextItem(
            color="#333333",
            anchor=(1, 0),
            fill=pg.mkBrush(255, 255, 255, 230),
            border=pg.mkPen("#c5c5c5"),
        )
        tip.setZValue(1000)
        tip.hide()
        return tip

    def _hide_profile_tips(self) -> None:
        self.tip_x.hide()
        self.tip_y.hide()

    def _profile_value_name(self, which: str) -> str:
        if which == "y" and self._y_is_sum():
            return "log10 累加強度" if self.log_check.isChecked() else "累加強度"
        return "log10 強度" if self.log_check.isChecked() else "強度"

    def _show_profile_tip(self, which: str, pos) -> None:
        plot = self.plot_x if which == "x" else self.plot_y
        values = self._x_values if which == "x" else self._y_values
        if values is None or len(values) == 0:
            self._hide_profile_tips()
            return
        point = plot.vb.mapSceneToView(pos)
        coord = point.x() if which == "x" else point.y()
        index = int(np.floor(coord))
        if not (0 <= index < len(values)):
            self._hide_profile_tips()
            return
        value = float(values[index])
        axis = "x" if which == "x" else "y"
        name = self._profile_value_name(which)
        text = f"{axis} = {index} px\n{name}：{format_sci(value)}"
        tip = self.tip_x if which == "x" else self.tip_y
        other = self.tip_y if which == "x" else self.tip_x
        other.hide()
        tip.setText(text)
        tip.show()
        self._pin_profile_tip(which)
        self.hover.setText(f"剖面：{axis} = {index} px  {name}：{format_sci(value)}")

    def _pin_profile_tip(self, which: str) -> None:
        tip = self.tip_x if which == "x" else self.tip_y
        plot = self.plot_x if which == "x" else self.plot_y
        if not tip.isVisible():
            return
        views = self.win.scene().views()
        if not views:
            return
        rect = plot.vb.sceneBoundingRect()
        text_rect = tip.boundingRect()
        width = float(text_rect.width())
        height = float(text_rect.height())
        view_rect = views[0].mapToScene(views[0].viewport().rect()).boundingRect()
        margin = 6.0
        x = float(rect.right()) - margin
        y = float(rect.top()) + margin
        if x - width < view_rect.left() + margin:
            x = float(view_rect.left()) + margin + width
        if x > view_rect.right() - margin:
            x = float(view_rect.right()) - margin
        if y + height > view_rect.bottom() - margin:
            y = max(float(view_rect.top()) + margin, float(view_rect.bottom()) - margin - height)
        tip.setAnchor((1, 0))
        tip.setPos(x, y)

    def _on_mouse_moved(self, pos) -> None:
        if self.plot_x.sceneBoundingRect().contains(pos):
            self._show_profile_tip("x", pos)
            return
        if self.plot_y.sceneBoundingRect().contains(pos):
            self._show_profile_tip("y", pos)
            return
        showing_profile = self.tip_x.isVisible() or self.tip_y.isVisible()
        self._hide_profile_tips()
        if self.raw is None or not self.plot.sceneBoundingRect().contains(pos):
            if showing_profile:
                self.hover.setText("游標：—")
            return
        mouse = self.plot.getViewBox().mapSceneToView(pos)
        x = int(np.floor(mouse.x()))
        y = int(np.floor(mouse.y()))
        height = self.raw.shape[0]
        width = self.raw.shape[1]
        if not (0 <= x < width and 0 <= y < height):
            self.hover.setText("游標：影像外")
            return
        value = self.raw[y, x]
        if np.ndim(value) == 0:
            text = f"{float(value):.6g}"
        else:
            text = ", ".join(f"{float(channel):.6g}" for channel in np.ravel(value))
        self.hover.setText(f"游標：x={x}  y={y}  結果={text}")

    def _on_scene_clicked(self, event) -> None:
        if self.shown is None:
            return
        if event.button() not in (Qt.LeftButton, Qt.NoButton):
            return
        pos = event.scenePos()
        if event.double():
            self._on_double_click(pos)
            return
        hit = self._intensity_axis_hit(pos)
        if hit is not None:
            self._prompt_axis_limit(*hit)
            return
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        point = self.plot.vb.mapSceneToView(pos)
        x = int(np.floor(point.x()))
        y = int(np.floor(point.y()))
        height, width = self.shown.shape
        if not (0 <= x < width and 0 <= y < height):
            return
        self.cross_xy = (x, y)
        self._update_crosshair(reset_intensity=True)

    def _analysis_plane(self) -> np.ndarray | None:
        """Linear image on the scale after × 255. Mertens is already there."""
        plane = self._linear_plane()
        if plane is None:
            return None
        if self._export_full_scale:
            return plane * 255.0
        return plane

    def _refresh_source_label(self) -> None:
        if self.parent_root is not None:
            self.source_name.setText(self.parent_root.name)
            tip = str(self.parent_root)
            if self.source_root is not None:
                tip = f"{self.parent_root}\n目前點位：{self.source_root}"
            self.source_name.setToolTip(tip)
            return
        if self.source_root is None:
            self.source_name.setText("尚未載入")
            self.source_name.setToolTip("")
            return
        self.source_name.setText(self.source_root.name)
        self.source_name.setToolTip(str(self.source_root))

    def _on_shared_log_toggled(self, checked: bool) -> None:
        if self.log_check.isChecked() != bool(checked):
            self.log_check.blockSignals(True)
            self.log_check.setChecked(bool(checked))
            self.log_check.blockSignals(False)
        self._on_log_toggled(checked)

    def _on_shared_color_stop_changed(self, _value: float | None = None) -> None:
        if self.analysis is None:
            return
        self._apply_color_stops([float(box.value()) for box in self.analysis.color_stop_boxes])

    def _on_shared_gradient_moved(self, *_args) -> None:
        if self.analysis is None or self._stops_updating:
            return
        self._on_shared_color_stop_changed()

    def _on_shared_color_limit_changed(self, _value: float | None = None) -> None:
        if self.analysis is None or self._color_updating:
            return
        lo = float(self.analysis.color_min.value())
        hi = float(self.analysis.color_max.value())
        if hi < lo:
            lo, hi = hi, lo
        self._set_color_levels(lo, hi)

    def _push_analysis(self, *, switch_view: bool = False) -> bool:
        del switch_view
        if self.analysis is None:
            return False
        # Batch fusion leaves planes in batch_planes.
        if self.batch_planes:
            self.analysis.set_batch(self.batch_planes, self.batch_labels)
        else:
            plane = self._analysis_plane()
            if plane is None:
                return False
            self.analysis.set_image(np.array(plane, copy=True))
        self.analysis.set_source_folder(self.source_root)
        return True

    def _open_analysis(self) -> None:
        if not self._push_analysis():
            QMessageBox.warning(self, "尚未計算", "請先計算影像。")
            return

    def _export_bundle(self) -> None:
        if self.raw is None or self.shown is None:
            QMessageBox.warning(self, "尚未計算", "請先計算影像。")
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇匯出資料夾")
        if not folder:
            return
        directory = Path(folder)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            before = self._result_plane(smooth=False)
            after = self._result_plane(smooth=True)
            x_index, y_index = self._export_crosshair(after.shape)
            self._write_profile_csv(directory / f"profiles_{stamp}.csv", before, after, x_index, y_index)
            self._write_npy(directory / f"hdr_{stamp}.npy")
            self._write_params(directory / f"params_{stamp}.json", stamp, x_index, y_index)
            self._write_heatmap(directory / f"heatmap_{stamp}.png")
        except Exception as exc:
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.stats.setText(f"已匯出到 {directory}\n時間戳 {stamp}")
        QMessageBox.information(self, "匯出完成", f"已寫入 {directory}")

    def _result_plane(self, smooth: bool) -> np.ndarray:
        """Linear result. After uses the image on screen. Before drops only the Gaussian blur."""
        if smooth or not self.smooth_check.isChecked():
            plane = self._as_plane(self.raw)
            if plane is None:
                raise RuntimeError("沒有可匯出的結果。")
            return plane
        selected = self._selected()
        if not selected:
            raise RuntimeError("請至少勾選一張影像。")
        images = [self._preprocess_row(row, 1, 1, 0.0) for row in selected]
        if len(images) == 1:
            raw = np.asarray(images[0], dtype=np.float32)
        else:
            raw, _response = run_hdr(
                images,
                [float(row.time.value()) for row in selected],
                str(self.algorithm.currentData()),
                float(self.full_scale.value()),
                self._algorithm_settings(),
            )
        plane = self._as_plane(raw)
        if plane is None:
            raise RuntimeError("沒有可匯出的結果。")
        return plane

    def _export_crosshair(self, shape: tuple[int, ...]) -> tuple[int, int]:
        height, width = int(shape[0]), int(shape[1])
        if self.cross_xy is None:
            return 0, 0
        x, y = self.cross_xy
        return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))

    def _write_profile_csv(
        self,
        path: Path,
        before: np.ndarray,
        after: np.ndarray,
        x_index: int,
        y_index: int,
    ) -> None:
        columns: list[tuple[str, np.ndarray]] = []
        for label, plane in (("before", before), ("after", after)):
            data = np.nan_to_num(np.asarray(plane, dtype=np.float64), nan=0.0)
            x_profile = data[y_index, :]
            y_profile = data[:, x_index]
            y_sum = data.sum(axis=1)
            columns.extend([
                (f"{label}_x", x_profile),
                (f"{label}_x_log10", display_log10(x_profile)),
                (f"{label}_y", y_profile),
                (f"{label}_y_log10", display_log10(y_profile)),
                (f"{label}_ysum", y_sum),
                (f"{label}_ysum_log10", display_log10(y_sum)),
            ])
        length = max(values.size for _, values in columns)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pixel", *[name for name, _values in columns]])
            for index in range(length):
                row: list[object] = [index]
                for _name, values in columns:
                    if index < values.size:
                        row.append(f"{float(values[index]):.8g}")
                    else:
                        row.append("")
                writer.writerow(row)

    def _write_npy(self, path: Path) -> None:
        exported = np.asarray(self.raw, dtype=np.float32)
        if self._export_full_scale:
            scale = float(self.full_scale.value())
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError("滿量必須是大於 0 的數字。")
            exported = exported * np.float32(scale)
        np.save(path, exported)

    def _write_params(self, path: Path, stamp: str, x_index: int, y_index: int) -> None:
        width_x, width_y, sigma = self._smooth_widths()
        payload = {
            "timestamp": stamp,
            "algorithm": self.algorithm.currentText(),
            "algorithm_id": str(self.algorithm.currentData()),
            "full_scale": float(self.full_scale.value()),
            "npy_times_full_scale": bool(self._export_full_scale),
            "gaussian": {
                "enabled": bool(self.smooth_check.isChecked()),
                "width_x": width_x,
                "width_y": width_y,
                "sigma": sigma,
            },
            "log": bool(self.log_check.isChecked()),
            "y_profile": "ysum" if self._y_is_sum() else "y",
            "crosshair": {"x": x_index, "y": y_index},
            "color_levels": {
                "min": float(self.hist.getLevels()[0]),
                "max": float(self.hist.getLevels()[1]),
            },
            "color_stops": [
                {
                    "name": english,
                    "position": float(box.value()),
                }
                for box, (_name, english, _color, _pos) in zip(self.color_stop_boxes, COLOR_STOPS)
            ],
            "algorithm_settings": self._algorithm_settings(),
            "exposures": [
                {
                    "file": None if row.path is None else row.path.name,
                    "exposure_us": float(row.time.value()) if row.image is not None else row.slot_us,
                    "loaded": row.image is not None,
                    "used": row.image is not None and row.use.isChecked(),
                    "gain": float(row.gain.value()),
                    "color_scale_floor": row.color_floor.number(),
                    "color_scale_ceil": row.color_ceil.number(),
                }
                for row in self.rows
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_heatmap(self, path: Path) -> None:
        from pyqtgraph.exporters import ImageExporter

        shown = self.shown
        if shown is None or self._x_values is None or self._y_values is None or self.cross_xy is None:
            raise RuntimeError("沒有可匯出的熱圖。")
        widget = pg.GraphicsLayoutWidget()
        widget.setBackground(CANVAS)
        widget.resize(1200, 820)
        widget.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        try:
            height, width = shown.shape
            x_index, y_index = self.cross_xy
            curve_pen = pg.mkPen("#444444", width=1.5)
            curve_brush = pg.mkBrush("#444444")
            plot_y = widget.addPlot(row=0, col=0, title="Y 累加" if self._y_is_sum() else f"Y 剖面  x={x_index}")
            plot_y.setLabel("left", "Y (px)")
            plot_y.setLabel("bottom", self._profile_value_name("y"))
            plot_y.showGrid(x=True, y=True, alpha=0.35)
            plot_y.invertY(True)
            plot_y.enableAutoRange(x=False, y=False)
            positions_y = np.arange(self._y_values.size) + 0.5
            plot_y.plot(self._y_values, positions_y, pen=curve_pen)
            plot_y.plot(
                [float(self._y_values[int(np.clip(y_index, 0, self._y_values.size - 1))])],
                [y_index + 0.5],
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=curve_brush,
            )
            plot_y.setYRange(0, height, padding=0)
            self._fit_export_intensity(plot_y, self._y_values, "x")

            plot = widget.addPlot(row=0, col=1)
            plot.setAspectLocked(True)
            plot.invertY(True)
            plot.showGrid(x=True, y=True, alpha=0.45)
            plot.setLabel("bottom", "X")
            plot.setLabel("left", "Y")
            image_item = pg.ImageItem()
            plot.addItem(image_item)
            image_item.setImage(shown.T, autoLevels=False)
            lo, hi = self.hist.getLevels()
            image_item.setLevels([float(lo), float(hi)])
            image_item.setColorMap(self.hist.gradient.colorMap())
            plot.setXRange(0, width, padding=0)
            plot.setYRange(0, height, padding=0)

            plot_x = widget.addPlot(row=1, col=1, title=f"X 剖面  y={y_index}")
            plot_x.setLabel("bottom", "X (px)")
            plot_x.setLabel("left", "log10 強度" if self.log_check.isChecked() else "強度")
            plot_x.showGrid(x=True, y=True, alpha=0.35)
            plot_x.enableAutoRange(x=False, y=False)
            positions_x = np.arange(self._x_values.size) + 0.5
            plot_x.plot(positions_x, self._x_values, pen=curve_pen)
            plot_x.plot(
                [x_index + 0.5],
                [float(self._x_values[int(np.clip(x_index, 0, self._x_values.size - 1))])],
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=curve_brush,
            )
            plot_x.setXRange(0, width, padding=0)
            self._fit_export_intensity(plot_x, self._x_values, "y")
            widget.ci.layout.setRowStretchFactor(0, 5)
            widget.ci.layout.setRowStretchFactor(1, 2)
            widget.ci.layout.setColumnStretchFactor(0, 2)
            widget.ci.layout.setColumnStretchFactor(1, 5)
            widget.show()
            QApplication.processEvents()
            exporter = ImageExporter(widget.scene())
            exporter.parameters()["width"] = 1200
            if not exporter.export(str(path)):
                raise RuntimeError("熱圖沒有寫入。")
        finally:
            widget.hide()
            widget.deleteLater()

    def _fit_export_intensity(self, plot, values: np.ndarray, axis: str) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
            pad = (hi - lo) * 0.05
            lo -= pad
            hi += pad
        if axis == "x":
            plot.setXRange(lo, hi, padding=0)
        else:
            plot.setYRange(lo, hi, padding=0)


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HdrTestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
