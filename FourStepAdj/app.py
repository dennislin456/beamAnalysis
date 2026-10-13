"""Simple UI for four-step exposure merge onto one intensity scale."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from merge_exposures import CAMERA_FULL_SCALE, merge_four_exposures
from sci_format import ask_scientific, format_sci
from spot_window import SpotWindow

TOKENS = ("50000us", "5000us", "500us", "50us")
SLOT_ORDER = ("50us", "500us", "5000us", "50000us")
IMAGE_EXTS = {".npy", ".png", ".bmp", ".tif", ".tiff", ".csv"}
FRAME_RE = re.compile(r"_(\d+)_\d{8}_\d{6}\.[^.]+$", re.IGNORECASE)


def classify_exposure(name: str) -> str | None:
    lowered = name.lower().replace(" ", "")
    for token in TOKENS:
        if re.fullmatch(token, lowered) or re.search(rf"(?<!\d){token}(?!\d)", lowered):
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


CANVAS = "#E8ECF0"


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
    QComboBox, QSpinBox, QDoubleSpinBox {
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("四段曝光強度校正")
        self.resize(1400, 920)
        self.series: dict[str, list[Path]] = {token: [] for token in SLOT_ORDER}
        self.merged: np.ndarray | None = None
        self.source: np.ndarray | None = None
        self.profile_xy: tuple[int, int] | None = None
        self.y_sum: np.ndarray | None = None
        self._cross_visible = False
        self._manual_x = False
        self._manual_y = False
        self._color_updating = False
        self._x_intensity: np.ndarray | None = None
        self._y_intensity: np.ndarray | None = None
        self.analysis_window: SpotWindow | None = None
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
        left.setFixedWidth(400)
        left.setStyleSheet(_panel_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)
        hint = QLabel(
            "載入 50 / 500 / 5000 / 50000 us 子資料夾後合成。\n"
            "點熱圖固定十字。雙擊熱圖、剖面或色條可各自復原。\n"
            "合成後可到另一視窗分析上下光斑。"
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
        for token in SLOT_ORDER:
            row = QHBoxLayout()
            row.setSpacing(8)
            btn = QPushButton("資料夾")
            btn.setFixedWidth(72)
            btn.clicked.connect(lambda _checked=False, t=token: self.pick_folder(t))
            label = QLabel("尚未選擇")
            label.setObjectName("path")
            label.setWordWrap(True)
            row.addWidget(btn)
            row.addWidget(label, 1)
            wrap = QWidget()
            wrap.setObjectName("slot")
            wrap.setAttribute(Qt.WA_StyledBackground, True)
            wrap.setLayout(row)
            row.setContentsMargins(8, 4, 8, 4)
            form.addRow(token, wrap)
            self.path_labels[token] = label
        form.addRow("資料筆數", self.record_combo)
        left_layout.addLayout(form)

        left_layout.addWidget(_section_label("合成"))
        options = QFormLayout()
        options.setHorizontalSpacing(10)
        options.setVerticalSpacing(8)
        options.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0, 1_000_000_000)
        self.threshold_input.setDecimals(1)
        self.threshold_input.setValue(CAMERA_FULL_SCALE)
        self.threshold_input.setSingleStep(1)
        self.threshold_input.setKeyboardTracking(False)
        self.threshold_input.setToolTip("px × 10 大於等於此值時，改用較短曝光的校正值")
        self.threshold_input.valueChanged.connect(self._on_threshold_changed)
        options.addRow("門檻", self.threshold_input)

        self.y_profile_mode = QComboBox()
        self.y_profile_mode.addItem("Y 剖面", "slice")
        self.y_profile_mode.addItem("Y 累加", "sum")
        self.y_profile_mode.currentIndexChanged.connect(self._on_y_mode_changed)
        options.addRow("左側剖面", self.y_profile_mode)
        left_layout.addLayout(options)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.btn_merge = QPushButton("合成")
        self.btn_merge.clicked.connect(self.run_merge)
        self.btn_analyze = QPushButton("分析光斑")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.open_analysis)
        self.btn_export = QPushButton("匯出")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_result)
        for button in (self.btn_merge, self.btn_analyze, self.btn_export):
            action_row.addWidget(button, 1)
        left_layout.addLayout(action_row)

        left_layout.addWidget(_section_label("結果"))
        self.stats = QLabel("尚未合成")
        self.stats.setObjectName("card")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left_layout.addWidget(self.stats)

        self.hover = QLabel("游標：—")
        self.hover.setObjectName("card")
        self.hover.setWordWrap(True)
        left_layout.addWidget(self.hover)
        left_layout.addStretch(1)
        layout.addWidget(left)

        canvas = "#E8ECF0"
        pg.setConfigOption("background", canvas)
        pg.setConfigOption("foreground", "#333333")
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground(canvas)
        grid = self.win.ci.layout
        curve_pen = pg.mkPen("#444444", width=1.5)
        curve_brush = pg.mkBrush("#444444")

        self.plot_y = self.win.addPlot(row=0, col=0, title="Y 剖面")
        self.plot_y.setLabel("left", "Y (px)")
        self.plot_y.setLabel("bottom", "強度")
        self.plot_y.getAxis("bottom").setToolTip("點最小值輸入下限，點最大值輸入上限")
        self.plot_y.showGrid(x=True, y=True, alpha=0.35)
        self.plot_y.invertY(True)
        self.plot_y.enableAutoRange(x=False, y=False)
        self.curve_y = self.plot_y.plot(pen=curve_pen)
        self.marker_y = self.plot_y.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=curve_brush,
        )

        self.plot = self.win.addPlot(row=0, col=1, title="校正後強度")
        self.plot.setLabel("bottom", "X (px)")
        self.plot.setLabel("left", "Y (px)")
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.35)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        cross_pen = pg.mkPen("#666666", width=1, style=Qt.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.plot.addItem(self.vline)
        self.plot.addItem(self.hline)
        self.vline.hide()
        self.hline.hide()

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(colors)), colors)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(cmap)
        self.hist.plot.setPen(pg.mkPen("#666666", width=1))
        for line in self.hist.region.lines:
            line.setPen(pg.mkPen("#444444", width=1))
        self.hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self.win.addItem(self.hist, row=0, col=2)
        self.color_max = self._make_color_box("色條上限")
        self.color_min = self._make_color_box("色條下限")
        self.color_max.setPrefix("上限: ")
        self.color_min.setPrefix("下限: ")
        box_font = self.color_max.font()
        box_font.setPointSize(9)
        box_width = QFontMetrics(box_font).horizontalAdvance("上限: 00000000.00") + 22
        box_style = (
            "QDoubleSpinBox {"
            "background: #f7f7f7; color: #555555;"
            "border: 1px solid #d0d0d0; border-radius: 8px; padding: 1px 8px;"
            "}"
            "QDoubleSpinBox QLineEdit {"
            "background: transparent; color: #555555; border: none; padding: 0;"
            "}"
        )
        for box in (self.color_max, self.color_min):
            box.setFont(box_font)
            box.setButtonSymbols(QAbstractSpinBox.NoButtons)
            box.setSpecialValueText("--")
            box.setValue(box.minimum())
            box.setFixedWidth(box_width)
            box.setFixedHeight(28)
            box.setStyleSheet(box_style)
        self.color_min.valueChanged.connect(self._on_color_limit_changed)
        self.color_max.valueChanged.connect(self._on_color_limit_changed)

        self.plot_x = self.win.addPlot(row=1, col=1, title="X 剖面")
        self.plot_x.setLabel("bottom", "X (px)")
        self.plot_x.setLabel("left", "強度")
        self.plot_x.getAxis("left").setToolTip("點最小值輸入下限，點最大值輸入上限")
        self.plot_x.showGrid(x=True, y=True, alpha=0.35)
        self.plot_x.enableAutoRange(x=False, y=False)
        self.curve_x = self.plot_x.plot(pen=curve_pen)
        self.marker_x = self.plot_x.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=curve_brush,
        )

        corner = self.win.addPlot(row=1, col=0)
        corner.hideAxis("left")
        corner.hideAxis("bottom")
        corner.hideAxis("top")
        corner.hideAxis("right")
        corner.setMouseEnabled(x=False, y=False)
        corner.hideButtons()

        for plot in (self.plot, self.plot_x, self.plot_y):
            plot.getViewBox().setMouseEnabled(x=True, y=True)
        grid.setColumnStretchFactor(0, 2)
        grid.setColumnStretchFactor(1, 5)
        grid.setColumnStretchFactor(2, 0)
        grid.setRowStretchFactor(0, 5)
        grid.setRowStretchFactor(1, 2)

        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self.on_scene_clicked)

        self.plot_host = QWidget()
        self.plot_host.setAutoFillBackground(True)
        host_palette = self.plot_host.palette()
        host_palette.setColor(self.plot_host.backgroundRole(), QColor(canvas))
        self.plot_host.setPalette(host_palette)
        host_layout = QVBoxLayout(self.plot_host)
        host_layout.setContentsMargins(0, 0, 8, 0)
        host_layout.setSpacing(6)
        self.log_check = QCheckBox("log")
        self.log_check.setToolTip("熱圖、X 剖面、Y 剖面與 Y 累加一起改為 log10。小於 1 的值以 1 計算。")
        self.log_check.toggled.connect(self._on_log_toggled)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(8, 0, 0, 0)
        color_row.setSpacing(8)
        color_row.addWidget(self.log_check)
        color_row.addStretch(1)
        color_row.addWidget(self.color_max)
        color_row.addWidget(self.color_min)
        host_layout.addLayout(color_row)
        host_layout.addWidget(self.win, 1)
        layout.addWidget(self.plot_host, 1)

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
            if token is None or self.series[token]:
                continue
            self.series[token] = list_images(path)
            found += 1
        if found == 0:
            QMessageBox.warning(self, "找不到曝光資料夾", "主資料夾內沒有 50us、500us、5000us、50000us 子資料夾。")
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
                self.path_labels[token].setText(path.name)
                self.path_labels[token].setToolTip(str(path))
            elif files:
                self.path_labels[token].setText("此筆沒有")
                self.path_labels[token].setToolTip("")
            else:
                self.path_labels[token].setText("尚未選擇")
                self.path_labels[token].setToolTip("")

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

    def run_merge(self) -> None:
        index = self.record_combo.currentIndex()
        self._show_record_labels(index)
        if index < 0 or not any(index < len(self.series[token]) for token in SLOT_ORDER):
            return
        try:
            images = self._images_for_record(index)
            threshold = float(self.threshold_input.value())
            result = merge_four_exposures(
                images["50us"],
                images["500us"],
                images["5000us"],
                images["50000us"],
                threshold=threshold,
            )
        except Exception as exc:
            QMessageBox.critical(self, "合成失敗", str(exc))
            return

        self.merged = result.image
        self.source = result.source
        self._show_image(result.image)
        vmin = float(np.min(result.image))
        vmax = float(np.max(result.image))
        c = result.counts
        above_camera = int(np.sum(result.image > CAMERA_FULL_SCALE))
        self.stats.setText(
            f"校正後最小值：{vmin:.6g}\n"
            f"校正後最大值：{vmax:.6g}\n"
            f"超過 {CAMERA_FULL_SCALE:g} 的像素：{above_camera}\n\n"
            f"留下（px×10 ≥ {threshold:g}）\n"
            f"  50 us：{c['kept_50us'] if images['50us'] is not None else '未載入'}\n"
            f"  500 us：{c['kept_500us'] if images['500us'] is not None else '未載入'}\n"
            f"  5000 us：{c['kept_5000us'] if images['5000us'] is not None else '未載入'}\n\n"
            f"實際寫入（較短曝光優先）\n"
            f"  50 us ×1000：{c['used_50us']}\n"
            f"  500 us ×100：{c['used_500us']}\n"
            f"  5000 us ×10：{c['used_5000us']}\n"
            f"  50000 us 原值：{c['used_50000us']}"
        )
        self.btn_export.setEnabled(True)
        self.btn_analyze.setEnabled(True)

    def _log_enabled(self) -> bool:
        return bool(self.log_check.isChecked())

    def _display_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if not self._log_enabled():
            return data
        return np.log10(np.maximum(data, 1.0))

    def _intensity_label(self, cumulative: bool = False) -> str:
        if cumulative:
            return "log10 累加強度" if self._log_enabled() else "累加強度"
        return "log10 強度" if self._log_enabled() else "強度"

    def _show_image(self, image: np.ndarray) -> None:
        shown = self._display_values(image)
        vmin = float(np.min(shown))
        vmax = float(np.max(shown))
        if vmin == vmax:
            vmax = vmin + 1.0
        self.image_item.setImage(shown.T, autoLevels=False)
        self._set_color_levels(vmin, vmax)
        self.y_sum = image.sum(axis=1)
        self._manual_x = False
        self._manual_y = False
        self._cross_visible = True
        peak_y, peak_x = np.unravel_index(int(np.argmax(image)), image.shape)
        self._update_profiles(int(peak_x), int(peak_y))
        self._reset_heatmap_view()
        self._reset_profile_view("x")
        self._reset_profile_view("y")

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
        self.run_merge()

    def _on_log_toggled(self, _checked: bool) -> None:
        if self.merged is None:
            return
        shown = self._display_values(self.merged)
        vmin = float(np.min(shown))
        vmax = float(np.max(shown))
        if vmin == vmax:
            vmax = vmin + 1.0
        self.image_item.setImage(shown.T, autoLevels=False)
        self._set_color_levels(vmin, vmax)
        self._manual_x = False
        self._manual_y = False
        if self.profile_xy is not None:
            self._update_profiles(*self.profile_xy)
            self._apply_auto_limits("y")

    def _on_y_mode_changed(self, _index: int | None = None) -> None:
        if self.profile_xy is None:
            return
        self._manual_y = False
        self._update_profiles(*self.profile_xy)
        self._apply_auto_limits("y")

    def on_mouse_moved(self, pos) -> None:
        if self.merged is None:
            return
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        point = self.plot.vb.mapSceneToView(pos)
        x = int(np.floor(point.x()))
        y = int(np.floor(point.y()))
        height, width = self.merged.shape
        if not (0 <= x < width and 0 <= y < height):
            self.hover.setText("游標：影像外")
            return
        value = float(self.merged[y, x])
        exposure = int(self.source[y, x]) if self.source is not None else 0
        self.hover.setText(
            f"游標：x={x}, y={y}\n"
            f"校正值：{format_sci(value)}\n"
            f"來源曝光：{exposure} us"
        )

    def on_scene_clicked(self, event) -> None:
        if self.merged is None:
            return
        if event.button() not in (Qt.LeftButton, Qt.NoButton):
            return
        pos = event.scenePos()
        if event.double():
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
            return
        hit = self._intensity_axis_hit(pos)
        if hit is not None:
            self._prompt_axis_limit(*hit)
            return
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        pixel = self._pixel_at(pos)
        if pixel is None:
            return
        self._cross_visible = True
        self._update_profiles(*pixel)

    def _pixel_at(self, pos) -> tuple[int, int] | None:
        if self.merged is None:
            return None
        point = self.plot.vb.mapSceneToView(pos)
        x = int(np.floor(point.x()))
        y = int(np.floor(point.y()))
        height, width = self.merged.shape
        if not (0 <= x < width and 0 <= y < height):
            return None
        return x, y

    def _set_cross_visible(self, visible: bool) -> None:
        self._cross_visible = visible
        self.vline.setVisible(visible)
        self.hline.setVisible(visible)
        if visible and self.profile_xy is not None:
            self._update_profiles(*self.profile_xy)
            return
        self.marker_x.setData([], [])
        self.marker_y.setData([], [])

    def _update_profiles(self, x: int, y: int) -> None:
        if self.merged is None:
            return
        height, width = self.merged.shape
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        self.profile_xy = (x, y)
        row = self._display_values(self.merged[y, :])
        col = self._display_values(self.merged[:, x])
        y_positions = np.arange(height) + 0.5
        self._x_intensity = row
        self.curve_x.setData(np.arange(width) + 0.5, row)
        value = float(row[x])
        if self.y_profile_mode.currentData() == "sum":
            linear_sum = self.merged.sum(axis=1) if self.y_sum is None else self.y_sum
            y_values = self._display_values(linear_sum)
            self._y_intensity = y_values
            self.curve_y.setData(y_values, y_positions)
            y_marker = float(y_values[y])
            self.plot_y.setTitle("Y 累加")
            self.plot_y.setLabel("bottom", self._intensity_label(cumulative=True))
            self.plot_y.getAxis("bottom").setToolTip("點最小值輸入下限，點最大值輸入上限")
        else:
            self._y_intensity = col
            self.curve_y.setData(col, y_positions)
            y_marker = value
            self.plot_y.setTitle(f"Y 剖面  x={x}")
            self.plot_y.setLabel("bottom", self._intensity_label())
            self.plot_y.getAxis("bottom").setToolTip("點最小值輸入下限，點最大值輸入上限")
        self.plot_x.setLabel("left", self._intensity_label())
        if self._cross_visible:
            self.marker_x.setData([x + 0.5], [value])
            self.marker_y.setData([y_marker], [y + 0.5])
            self.vline.setPos(x + 0.5)
            self.hline.setPos(y + 0.5)
            self.vline.show()
            self.hline.show()
        else:
            self.marker_x.setData([], [])
            self.marker_y.setData([], [])
            self.vline.hide()
            self.hline.hide()
        self.plot_x.setTitle(f"X 剖面  y={y}")
        self._manual_x = False
        self._apply_auto_limits("x")
        if self.y_profile_mode.currentData() != "sum":
            self._manual_y = False
            self._apply_auto_limits("y")

    def _padded_limits(self, values: np.ndarray) -> tuple[float, float]:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin == vmax:
            vmax = vmin + 1.0
        pad = (vmax - vmin) * 0.05
        return vmin - pad, vmax + pad

    def _apply_auto_limits(self, which: str) -> None:
        values = self._x_intensity if which == "x" else self._y_intensity
        if values is None or values.size == 0:
            return
        lo, hi = self._padded_limits(values)
        self._write_limits(which, lo, hi, manual=False)

    def _item_contains(self, item, pos) -> bool:
        try:
            return item.sceneBoundingRect().contains(pos)
        except Exception:
            return False

    def _reset_heatmap_view(self) -> None:
        if self.merged is None:
            return
        height, width = self.merged.shape
        self.plot.setXRange(0, width, padding=0)
        self.plot.setYRange(0, height, padding=0)

    def _reset_profile_view(self, which: str) -> None:
        if self.merged is None:
            return
        height, width = self.merged.shape
        if which == "x":
            self._manual_x = False
            self.plot_x.setXRange(0, width, padding=0)
            self._apply_auto_limits("x")
        else:
            self._manual_y = False
            self.plot_y.setYRange(0, height, padding=0)
            self._apply_auto_limits("y")

    def _make_color_box(self, tooltip: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e15, 1e15)
        box.setDecimals(2)
        box.setKeyboardTracking(False)
        box.setSingleStep(1)
        box.setToolTip(tooltip)
        return box

    def _set_color_levels(self, lo: float, hi: float) -> None:
        if lo >= hi:
            hi = lo + 1.0
        self._color_updating = True
        try:
            self.hist.setHistogramRange(lo, hi, padding=0)
            self.hist.setLevels(lo, hi)
            for box in (self.color_min, self.color_max):
                box.blockSignals(True)
            step = max((hi - lo) / 100.0, 1e-6)
            self.color_min.setSingleStep(step)
            self.color_max.setSingleStep(step)
            self.color_min.setValue(lo)
            self.color_max.setValue(hi)
        finally:
            self.color_min.blockSignals(False)
            self.color_max.blockSignals(False)
            self._color_updating = False

    def _reset_color_levels(self) -> None:
        if self.merged is None:
            return
        shown = self._display_values(self.merged)
        lo = float(np.min(shown))
        hi = float(np.max(shown))
        if lo >= hi:
            hi = lo + 1.0
        self._set_color_levels(lo, hi)

    def _on_color_limit_changed(self, _value: float | None = None) -> None:
        if self._color_updating or self.merged is None:
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

    def _write_limits(self, which: str, lo: float, hi: float, manual: bool = False) -> None:
        if lo >= hi:
            if manual:
                QMessageBox.warning(self, "上下限", "下限必須小於上限。")
            return
        if manual and which == "x":
            self._manual_x = True
        elif manual:
            self._manual_y = True
        if which == "x":
            self.plot_x.setYRange(lo, hi, padding=0)
        else:
            self.plot_y.setXRange(lo, hi, padding=0)

    def _intensity_axis_hit(self, pos) -> tuple[str, str] | None:
        """Return ('x'|'y', 'min'|'max') when the click is on an intensity-axis end."""
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
        self._write_limits(which, float(lo), float(hi), manual=True)

    def export_result(self) -> None:
        if self.merged is None or self.profile_xy is None:
            QMessageBox.warning(self, "尚未合成", "請先合成影像。")
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "匯出剖面與熱圖",
            str(Path(__file__).resolve().parent / "merged_profiles.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        csv_path = Path(path)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        cross_path = csv_path.with_name(f"{csv_path.stem}_cross.png")
        sum_path = csv_path.with_name(f"{csv_path.stem}_ysum.png")
        x, y = self.profile_xy
        y_sum = self.merged.sum(axis=1)
        y_cross = self.merged[:, x]
        x_cross = self.merged[y, :]
        saved_mode = self.y_profile_mode.currentIndex()
        saved_cross = self._cross_visible
        saved_manual_y = self._manual_y
        saved_y_range = tuple(self.plot_y.viewRange()[0])
        try:
            write_profile_csv(csv_path, x, y, y_sum, y_cross, x_cross)
            self._grab_ui(0).save(str(cross_path), "PNG")
            self._grab_ui(1).save(str(sum_path), "PNG")
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", str(exc))
            return
        finally:
            self.y_profile_mode.blockSignals(True)
            self.y_profile_mode.setCurrentIndex(saved_mode)
            self.y_profile_mode.blockSignals(False)
            self._manual_y = saved_manual_y
            self._cross_visible = saved_cross
            self._update_profiles(x, y)
            if saved_manual_y:
                self.plot_y.setXRange(saved_y_range[0], saved_y_range[1], padding=0)
            QApplication.processEvents()
        QMessageBox.information(
            self,
            "匯出完成",
            f"{csv_path}\n{cross_path}\n{sum_path}",
        )

    def _grab_ui(self, mode_index: int):
        """Grab the current plot UI. X profile stays on the current cross row."""
        self.y_profile_mode.blockSignals(True)
        self.y_profile_mode.setCurrentIndex(mode_index)
        self.y_profile_mode.blockSignals(False)
        self._cross_visible = False
        if mode_index == 1:
            self._manual_y = False
        x, y = self.profile_xy
        self._update_profiles(x, y)
        if mode_index == 1:
            self._apply_auto_limits("y")
        self.win.repaint()
        QApplication.processEvents()
        pixmap = self.plot_host.grab()
        if pixmap.isNull():
            raise RuntimeError("無法擷取目前畫面")
        return pixmap


def write_profile_csv(
    path: Path,
    cross_x: int,
    cross_y: int,
    y_sum: np.ndarray,
    y_cross: np.ndarray,
    x_cross: np.ndarray,
) -> None:
    """One CSV: Y cumulative profile and the current cross profiles."""
    import csv

    height = int(y_sum.shape[0])
    width = int(x_cross.shape[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cross_x", "cross_y", "y_px", "y_sum", "y_cross", "x_px", "x_cross"])
        for index in range(max(height, width)):
            writer.writerow([
                cross_x,
                cross_y,
                index if index < height else "",
                f"{float(y_sum[index]):.8f}" if index < height else "",
                f"{float(y_cross[index]):.8f}" if index < height else "",
                index if index < width else "",
                f"{float(x_cross[index]):.8f}" if index < width else "",
            ])


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
