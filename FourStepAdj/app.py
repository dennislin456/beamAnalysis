"""Simple UI for four-step exposure merge onto one intensity scale."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from image_canvas import ImageCanvas, OptionalNumberBox, linear_bound, log10_bound
from merge_exposures import (
    CAMERA_FULL_SCALE,
    EXPOSURE_COLOR_CEIL,
    EXPOSURE_COLOR_FLOOR,
    merge_four_exposures,
)
from fill_window import FillWindow
from spot_window import SpotWindow
from subtract_window import SubtractWindow

TOKENS = ("100000us", "50000us", "5000us", "500us", "50us", "5us")
BARE_TOKENS = {
    "100000": "100000us",
    "50000": "50000us",
    "5000": "5000us",
    "500": "500us",
    "50": "50us",
    "5": "5us",
}
SLOT_ORDER = ("50us", "500us", "5000us", "50000us")
IMAGE_EXTS = {".npy", ".png", ".bmp", ".tif", ".tiff", ".csv"}
FRAME_RE = re.compile(r"_(\d+)_\d{8}_\d{6}\.[^.]+$", re.IGNORECASE)


class _PathScroll(QScrollArea):
    """One folder name, scrolled sideways on its own."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta)
        event.accept()


def classify_exposure(name: str) -> str | None:
    lowered = name.lower().replace(" ", "")
    for token in TOKENS:
        if re.fullmatch(token, lowered) or re.search(rf"(?<!\d){token}(?!\d)", lowered):
            return token
    for bare, token in BARE_TOKENS.items():
        if re.fullmatch(bare, lowered) or re.search(rf"(?<!\d){bare}(?!\d)", lowered):
            return token
    return None


def frame_sort_key(path: Path) -> tuple:
    match = FRAME_RE.search(path.name)
    if match:
        return (int(match.group(1)), path.name.lower())
    parts = [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]
    return (10**9, parts)


def frame_label(path: Path) -> str:
    match = FRAME_RE.search(path.name)
    if match:
        return match.group(1)
    return path.stem


def list_images(folder: Path) -> list[Path]:
    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(files, key=frame_sort_key)


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    elif ext in {".png", ".bmp", ".tif", ".tiff"}:
        from PIL import Image

        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
    else:
        raise ValueError(f"不支援的檔案格式: {path.suffix}")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path.name} 不是二維影像，形狀為 {arr.shape}")
    return arr


def _panel_style() -> str:
    return """
    QWidget#sidePanel { background: #F7F8FA; border-radius: 12px; }
    QLabel#hint {
        color: #5c6570;
        background: #ffffff;
        border: 1px solid #e1e5ea;
        border-radius: 8px;
        padding: 8px 10px;
    }
    QLabel#section {
        color: #2c333a;
        font-weight: 600;
        padding: 10px 2px 0px 2px;
        border-top: 1px solid #e6eaee;
    }
    QWidget#slot {
        background: #ffffff;
        border: 1px solid #e1e5ea;
        border-radius: 8px;
    }
    QLabel#path { color: #66707a; }
    QLabel#card {
        background: #ffffff;
        color: #333;
        border: 1px solid #e1e5ea;
        border-radius: 8px;
        padding: 10px;
    }
    QPushButton {
        background: #ffffff;
        color: #333;
        border: 1px solid #d5dbe2;
        border-radius: 6px;
        padding: 6px 10px;
        min-height: 20px;
    }
    QPushButton:hover { background: #f3f5f7; }
    QPushButton:disabled { color: #a0a8b0; background: #f3f4f6; }
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
        background: #ffffff;
        color: #333;
        border: 1px solid #d5dbe2;
        border-radius: 6px;
        padding: 3px 6px;
        min-height: 22px;
    }
    QCheckBox { color: #333; spacing: 6px; }
    """


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


class FourStepWindow(QMainWindow):
    def __init__(self, title: str = "四段曝光強度校正", on_merged: Callable[[np.ndarray], None] | None = None):
        super().__init__()
        self.setWindowTitle(title)
        self.on_merged = on_merged
        self.resize(1400, 920)
        self.series: dict[str, list[Path]] = {token: [] for token in SLOT_ORDER}
        self.merged: np.ndarray | None = None
        self.source: np.ndarray | None = None
        self.analysis_window: SpotWindow | None = None
        self.subtract_window: SubtractWindow | None = None
        self.fill_window: FillWindow | None = None
        self.canvas = ImageCanvas(color_scale=True, clip_display=True, exposure_smooth=True)
        self.canvas.smooth_check.toggled.disconnect(self.canvas._on_smooth_toggled)
        for box in (self.canvas.smooth_kx, self.canvas.smooth_ky, self.canvas.smooth_sigma):
            box.valueChanged.disconnect(self.canvas._refresh_display)
            box.valueChanged.connect(self._on_merge_smooth_n)
        self.canvas.smooth_check.toggled.connect(self._on_merge_smooth)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root.setStyleSheet("QWidget#appRoot { background: #E8ECF0; }")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        left = QWidget()
        left.setObjectName("sidePanel")
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(_panel_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)
        hint = QLabel(
            "載入 50 / 500 / 5000 / 50000，或同名加上 us 的子資料夾後合成。\n"
            "點熱圖固定十字。雙擊熱圖、剖面或色條可各自復原。\n"
            "合成後可分析上下光斑，到相減視窗做 F1 − F2，或到補值視窗用 F2 補上 F1 沒通過門檻的像素。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        left_layout.addWidget(_section_label("資料"))
        btn_folder = QPushButton("載入主資料夾")
        btn_folder.clicked.connect(self.load_folder)
        left_layout.addWidget(btn_folder)

        self.record_combo = QComboBox()
        self.record_combo.currentIndexChanged.connect(self._on_record_changed)

        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.path_labels: dict[str, QLabel] = {}
        self.path_scrolls: dict[str, _PathScroll] = {}
        for token in SLOT_ORDER:
            row = QHBoxLayout()
            row.setSpacing(8)
            btn = QPushButton("資料夾")
            btn.setFixedWidth(72)
            btn.clicked.connect(lambda _checked=False, t=token: self.pick_folder(t))
            label = QLabel("尚未選擇")
            label.setObjectName("path")
            label.setWordWrap(False)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            path_scroll = _PathScroll()
            path_scroll.setObjectName("pathScroll")
            path_scroll.setWidget(label)
            path_scroll.setWidgetResizable(True)
            path_scroll.setFrameShape(QScrollArea.NoFrame)
            path_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            path_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            path_scroll.setFixedHeight(label.fontMetrics().height() + 18)
            path_scroll.setStyleSheet(
                "QScrollArea#pathScroll { background: transparent; border: none; }"
                "QLabel#path { color: #66707a; background: transparent; }"
            )
            self._fit_path_label(label)
            row.addWidget(btn)
            row.addWidget(path_scroll, 1)
            wrap = QWidget()
            wrap.setObjectName("slot")
            wrap.setAttribute(Qt.WA_StyledBackground, True)
            wrap.setLayout(row)
            row.setContentsMargins(8, 4, 8, 4)
            form.addRow(token, wrap)
            self.path_labels[token] = label
            self.path_scrolls[token] = path_scroll
        form.addRow("資料筆數", self.record_combo)
        left_layout.addLayout(form)

        left_layout.addWidget(_section_label("合成"))
        merge_grid = QGridLayout()
        merge_grid.setContentsMargins(0, 0, 0, 0)
        merge_grid.setHorizontalSpacing(8)
        merge_grid.setVerticalSpacing(8)
        for column in range(6):
            merge_grid.setColumnStretch(column, 0)
        self.merge_grid = merge_grid
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0, 1_000_000_000)
        self.threshold_input.setDecimals(1)
        self.threshold_input.setValue(CAMERA_FULL_SCALE)
        self.threshold_input.setSingleStep(1)
        self.threshold_input.setKeyboardTracking(False)
        self.threshold_input.setToolTip(
            "覆寫時，px × 10 大於等於此值就改用較短曝光的校正值。"
            "覆寫時，比較的是該段次方之後、有填的 color scale 之後、還沒做整張平滑的像素。"
            "線性相加和 log 相加不使用這個門檻；那兩種方式是先平滑，再套 color scale。"
            "線性相加和 log 相加不使用這個門檻。"
        )
        self.threshold_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.threshold_input.valueChanged.connect(self._on_threshold_changed)
        self.combine_mode = QComboBox()
        self.combine_mode.addItem("覆寫", "overwrite")
        self.combine_mode.addItem("線性相加", "linear")
        self.combine_mode.addItem("log 相加", "log")
        self.combine_mode.setToolTip(
            "覆寫是較短曝光蓋過較長曝光。"
            "線性相加和 log 相加會把每一段勾選且有載入的曝光都加進去。"
            "每一段先做自己的次方、平滑、color scale，再乘自己的倍率。"
            "log 相加是各段先取 log10(max(值, 1))，再把這些 log 加起來。"
            "小於 1 的值，包含負數，log 後是 0。"
            "切到 log 相加時，倍率和各段 color scale 會改成 log10，例如倍率 1000 變成 3、color scale 256 變成約 2.40824；切回覆寫或線性相加時再換回 10 的次方。"
        )
        self.combine_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combine_mode.currentIndexChanged.connect(self._on_combine_changed)
        merge_grid.addWidget(self._fixed_label("方式"), 0, 1)
        merge_grid.addWidget(self.combine_mode, 0, 2, 1, 4)
        merge_grid.addWidget(self._fixed_label("門檻"), 1, 1)
        merge_grid.addWidget(self.threshold_input, 1, 2, 1, 3)
        power_head = self._fixed_label("次方")
        power_head.setAlignment(Qt.AlignCenter)
        merge_grid.addWidget(power_head, 2, 2)
        color_head = self._fixed_label("color scale")
        color_head.setAlignment(Qt.AlignCenter)
        merge_grid.addWidget(color_head, 2, 3, 1, 2)
        self.scale_header = self._fixed_label("倍率")
        merge_grid.addWidget(self.scale_header, 2, 5)
        self.scale_inputs: dict[str, QDoubleSpinBox] = {}
        self.scale_labels: dict[str, QLabel] = {}
        self._scale_in_log = False
        self._exposure_color_in_log = False
        self.floor_inputs: dict[str, OptionalNumberBox] = {}
        self.ceil_inputs: dict[str, OptionalNumberBox] = {}
        self._floor_tips: dict[str, str] = {}
        self._ceil_tips: dict[str, str] = {}
        self.use_inputs: dict[str, QCheckBox] = {}
        self.power_inputs: dict[str, OptionalNumberBox] = {}
        self._power_applied: dict[str, float | None] = {}
        for row, (token, default) in enumerate(
            (("50us", 1000.0), ("500us", 100.0), ("5000us", 10.0), ("50000us", 1.0)),
            start=3,
        ):
            if token == "50000us":
                floor_tip = "覆寫時先套這個下限，整張覆寫完才平滑。相加時先平滑，再把平滑後小於等於這個值的像素設為 0，然後乘倍率。留空代表不採用。勾選多段且覆寫時，50000 us 當底圖，不做 px×10 留下判斷。"
                ceil_tip = "覆寫時先套這個上限，整張覆寫完才平滑。相加時先平滑，再把平滑後大於等於這個值的像素設成這個值，然後乘倍率。留空代表不採用。勾選多段且覆寫時，50000 us 當底圖，不做 px×10 留下判斷。"
            else:
                floor_tip = "覆寫時先套這個下限，再用 px×10 做留下判斷，整張覆寫完才平滑。相加時先平滑，再把小於等於這個值的像素設為 0。留空代表不採用。"
                ceil_tip = "覆寫時先套這個上限，再用 px×10 做留下判斷，整張覆寫完才平滑。相加時先平滑，再把大於等於這個值的像素設成這個值。留空代表不採用。"
            use_box = QCheckBox()
            use_box.setChecked(True)
            use_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            use_box.setToolTip("勾選才使用這段曝光。取消就整段排除。只勾一段時，整張圖是這段先做次方（若有填）、再套有填的 color scale、再乘倍率的值。")
            use_box.toggled.connect(self._on_threshold_changed)
            scale_box = self._merge_spin(default, 3, "這段曝光寫入合成圖時乘上的倍率。只勾這一段時也會乘上。", signed=True)
            self._floor_tips[token] = floor_tip
            self._ceil_tips[token] = ceil_tip
            floor_box = OptionalNumberBox(EXPOSURE_COLOR_FLOOR, floor_tip, width=80)
            ceil_box = OptionalNumberBox(EXPOSURE_COLOR_CEIL, ceil_tip, width=80)
            floor_box.numberChanged.connect(self._on_threshold_changed)
            ceil_box.numberChanged.connect(self._on_threshold_changed)
            power_box = OptionalNumberBox(
                None,
                "這段曝光每個像素先變成這個次方。覆寫時接著套 color scale，整張覆寫完才平滑。相加時先平滑再套 color scale。留空代表不採用。填 2 是平方。負數遇到非整數次方會變成 0。"
                "改次方時，這段有填而且大於 0 的 color scale 會跟著換成新次方下的數值。例如 256、次方 2 會變成 65536。清空次方會換回來。log 相加時，輸入框顯示的是換算後的 log10。",
                width=72,
            )
            power_box.numberChanged.connect(lambda token=token: self._on_power_edited(token))
            scale_label = self._fixed_label(f"{token[:-2]} us")
            merge_grid.addWidget(use_box, row, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
            merge_grid.addWidget(scale_label, row, 1)
            merge_grid.addWidget(power_box, row, 2)
            merge_grid.addWidget(floor_box, row, 3)
            merge_grid.addWidget(ceil_box, row, 4)
            merge_grid.addWidget(scale_box, row, 5)
            self.use_inputs[token] = use_box
            self.scale_labels[token] = scale_label
            self.scale_inputs[token] = scale_box
            self.floor_inputs[token] = floor_box
            self.ceil_inputs[token] = ceil_box
            self.power_inputs[token] = power_box
            self._power_applied[token] = None
        self.y_profile_mode = self.canvas.y_profile_mode
        self.y_profile_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        merge_grid.addWidget(self._fixed_label("左側剖面"), 7, 1)
        merge_grid.addWidget(self.y_profile_mode, 7, 2, 1, 4)
        left_layout.addLayout(merge_grid)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.btn_merge = QPushButton("合成")
        self.btn_merge.clicked.connect(self.run_merge)
        self.btn_analyze = QPushButton("分析光斑")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.open_analysis)
        self.btn_subtract = QPushButton("相減")
        self.btn_subtract.clicked.connect(self.open_subtract)
        self.btn_fill = QPushButton("補值")
        self.btn_fill.clicked.connect(self.open_fill)
        self.btn_export = QPushButton("匯出")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_result)
        for button in (self.btn_merge, self.btn_analyze, self.btn_subtract, self.btn_fill, self.btn_export):
            action_row.addWidget(button, 1)
        left_layout.addLayout(action_row)

        left_layout.addWidget(_section_label("結果"))
        self.stats = QLabel("尚未合成")
        self.stats.setObjectName("card")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left_layout.addWidget(self.stats)
        left_layout.addStretch(1)
        self._merge_host = left
        left.installEventFilter(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(left)
        scroll.setFixedWidth(640)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        layout.addWidget(scroll)
        layout.addWidget(self.canvas, 1)
        self._sync_smooth_tip()
        self._fit_merge_spacing()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_merge_spacing()

    def eventFilter(self, watched, event) -> bool:
        if watched is getattr(self, "_merge_host", None) and event.type() == QEvent.Resize:
            self._fit_merge_spacing()
        return super().eventFilter(watched, event)

    def _fit_merge_spacing(self) -> None:
        grid = getattr(self, "merge_grid", None)
        host = getattr(self, "_merge_host", None)
        if grid is None or host is None or not self.scale_labels:
            return
        margins = host.layout().contentsMargins()
        grid_margins = grid.contentsMargins()
        available = host.width() - margins.left() - margins.right() - grid_margins.left() - grid_margins.right()
        if available <= 0:
            return
        check_w = next(iter(self.use_inputs.values())).sizeHint().width()
        label_w = max(label.sizeHint().width() for label in self.scale_labels.values())
        label_w = max(label_w, next(iter(self.scale_labels.values())).fontMetrics().horizontalAdvance("左側剖面"))
        content = check_w + label_w + 92 + 80 + 80 + 72
        spacing = max(8, (available - content) // 5)
        if grid.horizontalSpacing() != spacing:
            grid.setHorizontalSpacing(int(spacing))

    def _fixed_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        return label

    def _fit_path_label(self, label: QLabel) -> None:
        width = label.fontMetrics().horizontalAdvance(label.text()) + 12
        label.setMinimumWidth(max(width, 1))

    def _set_path_label(self, token: str, text: str, tooltip: str = "") -> None:
        label = self.path_labels[token]
        label.setText(text)
        label.setToolTip(tooltip)
        self._fit_path_label(label)
        self.path_scrolls[token].horizontalScrollBar().setValue(0)

    def _merge_spin(self, value: float, decimals: int, tooltip: str, signed: bool = False) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        if signed:
            box.setRange(-1_000_000_000.0, 1_000_000_000.0)
        else:
            box.setRange(0.0, 1_000_000_000.0)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSingleStep(1.0)
        box.setKeyboardTracking(False)
        box.setFixedWidth(92)
        box.setToolTip(tooltip)
        box.valueChanged.connect(self._on_threshold_changed)
        return box

    def _on_merge_smooth(self, checked: bool) -> None:
        for box in (self.canvas.smooth_kx, self.canvas.smooth_ky, self.canvas.smooth_sigma):
            box.setEnabled(checked)
        self._on_threshold_changed()

    def _on_merge_smooth_n(self, _value: int) -> None:
        if self.canvas.smooth_check.isChecked():
            self._on_threshold_changed()

    def _power_value(self, token: str) -> float | None:
        return self.power_inputs[token].number()

    def load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇主資料夾")
        if not folder:
            return
        root = Path(folder)
        found = 0
        for token in SLOT_ORDER:
            self.series[token] = []
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir():
                continue
            token = classify_exposure(path.name)
            if token is None or token not in self.series or self.series[token]:
                continue
            self.series[token] = list_images(path)
            found += 1
        if found == 0:
            QMessageBox.warning(
                self,
                "找不到曝光資料夾",
                "主資料夾內沒有 50、500、5000、50000 或 50us、500us、5000us、50000us 子資料夾。",
            )
            return
        self._refresh_record_list(prefer_index=0)

    def pick_folder(self, token: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, f"選擇 {token} 資料夾")
        if not folder:
            return
        self.series[token] = list_images(Path(folder))
        if not self.series[token]:
            QMessageBox.warning(self, "沒有影像", f"{token} 資料夾內沒有可用的影像檔。")
        prefer = self.record_combo.currentIndex()
        self._refresh_record_list(prefer_index=max(prefer, 0))

    def _record_count(self) -> int:
        return max((len(files) for files in self.series.values()), default=0)

    def _refresh_record_list(self, prefer_index: int = 0) -> None:
        count = self._record_count()
        self.record_combo.blockSignals(True)
        self.record_combo.clear()
        for index in range(count):
            self.record_combo.addItem(self._record_item_label(index), index)
        if count:
            self.record_combo.setCurrentIndex(min(prefer_index, count - 1))
        self.record_combo.blockSignals(False)
        if count:
            self.run_merge()
        else:
            self._show_record_labels(-1)

    def _record_item_label(self, index: int) -> str:
        labels = []
        for token in SLOT_ORDER:
            files = self.series[token]
            if index < len(files):
                labels.append(frame_label(files[index]))
        if not labels:
            return f"第{index + 1}筆"
        if len(set(labels)) == 1:
            return labels[0]
        parts = []
        for token in SLOT_ORDER:
            files = self.series[token]
            if index < len(files):
                parts.append(f"{token}:{frame_label(files[index])}")
        return " / ".join(parts)

    def _show_record_labels(self, index: int) -> None:
        for token in SLOT_ORDER:
            files = self.series[token]
            if 0 <= index < len(files):
                path = files[index]
                self._set_path_label(token, path.name, str(path))
            elif files:
                self._set_path_label(token, "此筆沒有")
            else:
                self._set_path_label(token, "尚未選擇")

    def _on_record_changed(self, _index: int | None = None) -> None:
        if self.record_combo.currentIndex() < 0:
            return
        self.run_merge()

    def _images_for_record(self, index: int) -> dict[str, np.ndarray | None]:
        images: dict[str, np.ndarray | None] = {}
        for token in SLOT_ORDER:
            files = self.series[token]
            if 0 <= index < len(files):
                images[token] = load_image(files[index])
            else:
                images[token] = None
        return images

    def run_merge(self, keep_crosshair: bool = False) -> None:
        index = self.record_combo.currentIndex()
        self._show_record_labels(index)
        if index < 0 or not any(index < len(self.series[token]) for token in SLOT_ORDER):
            return
        if not any(box.isChecked() for box in self.use_inputs.values()):
            self.stats.setText("請至少勾選一段曝光")
            return
        try:
            images = self._images_for_record(index)
            threshold = float(self.threshold_input.value())
            smooth_x, smooth_y, sigma = self.canvas._smooth_params()
            result = merge_four_exposures(
                images["50us"],
                images["500us"],
                images["5000us"],
                images["50000us"],
                threshold=threshold,
                scale_50us=self._linear_scale("50us"),
                scale_500us=self._linear_scale("500us"),
                scale_5000us=self._linear_scale("5000us"),
                scale_50000us=self._linear_scale("50000us"),
                floor_50us=self._exposure_bound("50us", "floor"),
                ceil_50us=self._exposure_bound("50us", "ceil"),
                floor_500us=self._exposure_bound("500us", "floor"),
                ceil_500us=self._exposure_bound("500us", "ceil"),
                floor_5000us=self._exposure_bound("5000us", "floor"),
                ceil_5000us=self._exposure_bound("5000us", "ceil"),
                floor_50000us=self._exposure_bound("50000us", "floor"),
                ceil_50000us=self._exposure_bound("50000us", "ceil"),
                use_50us=self.use_inputs["50us"].isChecked(),
                use_500us=self.use_inputs["500us"].isChecked(),
                use_5000us=self.use_inputs["5000us"].isChecked(),
                use_50000us=self.use_inputs["50000us"].isChecked(),
                power_50us=self._power_value("50us"),
                power_500us=self._power_value("500us"),
                power_5000us=self._power_value("5000us"),
                power_50000us=self._power_value("50000us"),
                smooth_x=smooth_x,
                smooth_y=smooth_y,
                sigma=sigma,
                combine=str(self.combine_mode.currentData()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "合成失敗", str(exc))
            return

        self.canvas.set_image(result.image, result.source, keep_crosshair=keep_crosshair)
        self.merged = self.canvas.image
        self.source = self.canvas.source
        vmin = float(np.min(result.image))
        vmax = float(np.max(result.image))
        c = result.counts
        above_camera = int(np.sum(result.image > CAMERA_FULL_SCALE))
        if c.get("combine") in {"linear", "log"}:
            mode_text = "線性相加" if c["combine"] == "linear" else "log 相加（各段 log10(max(值, 1)) 後相加）"
            detail = (
                f"{mode_text}\n"
                f"  50 us {self._scale_mark('50us')}：{self._included_text('50us', images, c['used_50us'])}\n"
                f"  500 us {self._scale_mark('500us')}：{self._included_text('500us', images, c['used_500us'])}\n"
                f"  5000 us {self._scale_mark('5000us')}：{self._included_text('5000us', images, c['used_5000us'])}\n"
                f"  50000 us {self._scale_mark('50000us')}：{self._included_text('50000us', images, c['used_50000us'])}"
            )
        else:
            detail = (
                f"留下（px×10 ≥ {threshold:g}）\n"
                f"  50 us：{self._kept_text('50us', images, c)}\n"
                f"  500 us：{self._kept_text('500us', images, c)}\n"
                f"  5000 us：{self._kept_text('5000us', images, c)}\n\n"
                f"實際寫入（較短曝光優先）\n"
                f"  50 us {self._scale_mark('50us')}：{self._merge_count_text('50us', images, c['used_50us'])}\n"
                f"  500 us {self._scale_mark('500us')}：{self._merge_count_text('500us', images, c['used_500us'])}\n"
                f"  5000 us {self._scale_mark('5000us')}：{self._merge_count_text('5000us', images, c['used_5000us'])}\n"
                f"  50000 us {self._scale_mark('50000us')}：{self._merge_count_text('50000us', images, c['used_50000us'])}"
            )
        self.stats.setText(
            f"校正後最小值：{vmin:.6g}\n"
            f"校正後最大值：{vmax:.6g}\n"
            f"超過 {CAMERA_FULL_SCALE:g} 的像素：{above_camera}\n\n"
            f"{detail}"
        )
        self.btn_export.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self._emit_merged()

    def _emit_merged(self) -> None:
        if self.on_merged is None or self.merged is None:
            return
        self.on_merged(np.array(self.merged, copy=True))

    def _kept_text(self, token: str, images: dict, counts: dict) -> str:
        if counts.get("solo") == token:
            return "整段"
        return self._merge_count_text(token, images, int(counts[f"kept_{token}"]))

    def _on_combine_changed(self, _index: int = 0) -> None:
        self.threshold_input.setEnabled(self.combine_mode.currentData() == "overwrite")
        if not getattr(self, "scale_inputs", None):
            return
        log_mode = self.combine_mode.currentData() == "log"
        if log_mode and not self._scale_in_log:
            self._convert_scales_to_log()
            self._convert_exposure_colors(True)
        elif not log_mode and self._scale_in_log:
            self._convert_scales_to_linear()
            self._convert_exposure_colors(False)
        self._sync_smooth_tip()
        self._sync_display_log()
        self._on_threshold_changed()

    def _sync_smooth_tip(self) -> None:
        if self.combine_mode.currentData() == "overwrite":
            tip = "勾選後，先完成覆寫，再對整張合成圖做 Gaussian blur。水平、垂直是核寬，sigma 兩邊共用。某一邊為 1 時該方向不平滑。各段 color scale 用的是還沒平滑的數值。這次平滑會進合成結果和匯出。"
        else:
            tip = "勾選後，每一段先做 Gaussian blur，再用該段 color scale 處理平滑後的數值，然後相加。水平、垂直是核寬，sigma 兩邊共用。某一邊為 1 時該方向不平滑。合成圖不再平滑一次。這次平滑會進合成結果和匯出。"
        self.canvas.smooth_check.setToolTip(tip)

    def _sync_display_log(self) -> None:
        """Log addition is already in log space, so the display log is turned off."""
        check = self.canvas.log_check
        log_combine = self.combine_mode.currentData() == "log"
        if log_combine:
            if check.isEnabled():
                self._display_log_checked = check.isChecked()
                self._display_log_tip = check.toolTip()
            self.canvas.sync_color_bound_space(False)
            check.blockSignals(True)
            check.setChecked(False)
            check.setEnabled(False)
            check.blockSignals(False)
            check.setToolTip("log 相加的結果已經是 log，畫面不能再取一次 log。")
            if self.canvas.image is not None:
                self.canvas._refresh_display()
            return
        was_checked = getattr(self, "_display_log_checked", None)
        self._display_log_checked = None
        check.setEnabled(True)
        if getattr(self, "_display_log_tip", ""):
            check.setToolTip(self._display_log_tip)
        if was_checked:
            check.setChecked(True)

    def _on_power_edited(self, token: str) -> None:
        new = self.power_inputs[token].number()
        old = self._power_applied.get(token)
        self._power_applied[token] = new
        self._convert_color_scale(token, old, new)
        self._on_threshold_changed()

    def _convert_color_scale(self, token: str, old: float | None, new: float | None) -> None:
        old_n = 1.0 if old is None else float(old)
        new_n = 1.0 if new is None else float(new)
        if not np.isfinite(old_n) or not np.isfinite(new_n) or old_n == new_n or old_n == 0.0:
            return
        ratio = new_n / old_n
        for box in (self.floor_inputs[token], self.ceil_inputs[token]):
            value = box.number()
            if value is None or not np.isfinite(value):
                continue
            if self._exposure_color_in_log:
                linear = linear_bound(value)
                if linear is None or linear <= 0 or not np.isfinite(linear):
                    continue
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    converted = float(np.power(float(linear), ratio))
                if not np.isfinite(converted) or converted <= 0:
                    continue
                nearest = round(converted)
                if abs(converted - nearest) <= 1e-6 * max(1.0, abs(nearest)):
                    converted = float(nearest)
                box.set_number(log10_bound(converted), emit=False)
                continue
            if value <= 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                converted = float(np.power(float(value), ratio))
            if not np.isfinite(converted):
                continue
            nearest = round(converted)
            if abs(converted - nearest) <= 1e-6 * max(1.0, abs(nearest)):
                converted = float(nearest)
            box.set_number(converted, emit=False)

    def _exposure_bound(self, token: str, which: str) -> float | None:
        box = self.floor_inputs[token] if which == "floor" else self.ceil_inputs[token]
        value = box.number()
        if not self._exposure_color_in_log:
            return value
        return linear_bound(value)

    def _convert_exposure_colors(self, log_on: bool) -> None:
        if bool(log_on) == self._exposure_color_in_log:
            return
        convert = log10_bound if log_on else linear_bound
        note = (
            "現在顯示的是 log10。實際裁切用 10 的這個次方。切回覆寫或線性相加會換回線性門檻。"
            "小於等於 0 的線性門檻會改成 0。"
        )
        for token in self.floor_inputs:
            floor_box = self.floor_inputs[token]
            ceil_box = self.ceil_inputs[token]
            floor_box.set_number(convert(floor_box.number()), emit=False)
            ceil_box.set_number(convert(ceil_box.number()), emit=False)
            floor_box.setToolTip(note + self._floor_tips[token] if log_on else self._floor_tips[token])
            ceil_box.setToolTip(note + self._ceil_tips[token] if log_on else self._ceil_tips[token])
        self._exposure_color_in_log = bool(log_on)

    def _linear_scale(self, token: str) -> float:
        value = float(self.scale_inputs[token].value())
        if not self._scale_in_log:
            return value
        if not np.isfinite(value):
            return 1.0
        return float(10.0 ** value)

    def _scale_mark(self, token: str) -> str:
        shown = float(self.scale_inputs[token].value())
        if self._scale_in_log:
            return f"log×{shown:.6g}"
        return f"×{shown:.6g}"

    def _convert_scales_to_log(self) -> None:
        for token, box in self.scale_inputs.items():
            value = float(box.value())
            logged = float(np.log10(value)) if value > 0 and np.isfinite(value) else 0.0
            box.blockSignals(True)
            box.setRange(-12.0, 12.0)
            box.setDecimals(4)
            box.setSingleStep(0.1)
            box.setValue(logged)
            box.setToolTip("log10 倍率。3 代表線性 ×1000。切回覆寫或線性相加時會自動換成 10 的這個次方。小於等於 0 的線性倍率會改成 0。")
            box.blockSignals(False)
        self.scale_header.setText("log倍率")
        self._scale_in_log = True
        self._fit_merge_spacing()

    def _convert_scales_to_linear(self) -> None:
        for token, box in self.scale_inputs.items():
            logged = float(box.value())
            linear = float(10.0 ** logged) if np.isfinite(logged) else 1.0
            box.blockSignals(True)
            box.setRange(-1_000_000_000.0, 1_000_000_000.0)
            box.setDecimals(3)
            box.setSingleStep(1.0)
            box.setValue(linear)
            box.setToolTip("這段曝光寫入合成圖時乘上的倍率。只勾這一段時也會乘上。")
            box.blockSignals(False)
        self.scale_header.setText("倍率")
        self._scale_in_log = False
        self._fit_merge_spacing()

    def _included_text(self, token: str, images: dict, count: int) -> str:
        if not self.use_inputs[token].isChecked():
            return "未勾選"
        if images[token] is None:
            return "未載入"
        if count <= 0:
            return "0"
        return "整段"

    def _merge_count_text(self, token: str, images: dict, count: int) -> str:
        if not self.use_inputs[token].isChecked():
            return "未勾選"
        if images[token] is None:
            return "未載入"
        return str(count)

    def open_analysis(self) -> None:
        if self.merged is None:
            QMessageBox.warning(self, "尚未合成", "請先合成影像。")
            return
        if self.analysis_window is None:
            self.analysis_window = SpotWindow()
        self.analysis_window.set_image(np.array(self.merged, copy=True))
        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()

    def _on_threshold_changed(self, _value: float | None = None) -> None:
        if self.merged is None:
            return
        if self.record_combo.currentIndex() < 0:
            return
        self.run_merge(keep_crosshair=True)

    def open_subtract(self) -> None:
        if self.subtract_window is None:
            self.subtract_window = SubtractWindow()
        self.subtract_window.show()
        self.subtract_window.raise_()
        self.subtract_window.activateWindow()

    def open_fill(self) -> None:
        if self.fill_window is None:
            self.fill_window = FillWindow()
        self.fill_window.show()
        self.fill_window.raise_()
        self.fill_window.activateWindow()

    def export_result(self) -> None:
        self.canvas.export_result('merged_profiles.csv', '請先合成影像。')


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FourStepWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
