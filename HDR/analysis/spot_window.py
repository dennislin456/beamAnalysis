"""Analysis window: place an ROI on the merged image and find both spots."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from analysis.display import (
    COLOR_STOPS,
    apply_abs_then_color_scale,
    colormap_from_stops,
    jet_bw_colormap,
)
from analysis.dual_spot import (
    find_upper_spot,
    locate_dual_spots,
    slanted_line_profile,
    slanted_strip_outline,
    slanted_strip_sum,
)
from common.sci_format import ask_scientific, format_sci
from common.widgets import add_gaussian_controls, function_frame, install_input_undo

CANVAS = "#E8ECF0"
ANALYSIS_PANEL_W = 420


def _panel_style() -> str:
    return """
    QWidget#sidePanel {
        background: #F4F6F8;
        border: 1px solid #e2e7ec;
        border-radius: 14px;
    }
    QScrollArea#analysisScroll { background: transparent; border: none; }
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
        text-transform: uppercase;
    }
    QFrame#groupCard, QFrame#resultCard {
        background: #ffffff;
        border: 1px solid #e4e9ee;
        border-radius: 10px;
    }
    QLabel#resultKey { color: #6b7580; padding: 0; font-size: 12px; }
    QLabel#resultValue { color: #1f2933; padding: 0; font-size: 13px; }
    QLabel#resultNote { color: #8a939c; padding: 4px 0 0 0; font-size: 11px; }
    QLabel#metaLabel { color: #5a646e; background: transparent; }
    QPushButton {
        background: #ffffff;
        color: #2c333a;
        border: 1px solid #d5dbe2;
        border-radius: 8px;
        padding: 7px 12px;
        min-height: 24px;
    }
    QPushButton:hover { background: #f2f5f8; border-color: #c5ced8; }
    QPushButton:disabled { color: #a0a8b0; background: #f3f4f6; }
    QPushButton#primary {
        background: #243039;
        color: #ffffff;
        border: 1px solid #243039;
        font-weight: 600;
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
    QTableWidget {
        background: #ffffff;
        color: #2c333a;
        border: 1px solid #e4e9ee;
        border-radius: 10px;
        gridline-color: #eef1f4;
        alternate-background-color: #fafbfc;
        outline: none;
    }
    QTableWidget::item { padding: 4px 8px; }
    QTableWidget::item:selected { background: #dce6f0; color: #1f2933; }
    QHeaderView::section {
        background: #f0f3f6;
        color: #3d4650;
        border: none;
        border-bottom: 1px solid #e1e5ea;
        padding: 6px 8px;
        font-weight: 600;
        text-align: center;
    }
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
        label.setStyleSheet("QLabel#section { border-top: none; padding-top: 2px; }")
    return label


def _group_card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("groupCard")
    frame.setAttribute(Qt.WA_StyledBackground, True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    return frame, layout


def _meta_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("metaLabel")
    return label


class DistanceTable(QWidget):
    """每一筆的筆數、像素距離和微米距離。"""

    rowClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("distanceTable")
        self.table.setHorizontalHeaderLabels(["筆數", "px", "μm"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setWordWrap(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setFixedHeight(28)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.cellClicked.connect(self._on_cell_clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        self.setMinimumHeight(160)
        self.setToolTip("每一筆的上下光斑距離。點一列會切到那一張。")

    def blockSignals(self, block: bool) -> bool:
        previous = super().blockSignals(block)
        self.table.blockSignals(block)
        return previous

    def count(self) -> int:
        return self.table.rowCount()

    def currentRow(self) -> int:
        return self.table.currentRow()

    def setCurrentRow(self, row: int) -> None:
        if row < 0:
            self.table.clearSelection()
            return
        self.table.selectRow(row)

    def fill(self, rows: list[tuple[str, str, str]]) -> None:
        self.table.setRowCount(len(rows))
        for index, (frame, px, um) in enumerate(rows):
            self._set_cell(index, 0, frame, Qt.AlignCenter)
            self._set_cell(index, 1, px, Qt.AlignCenter)
            self._set_cell(index, 2, um, Qt.AlignCenter)

    def _on_cell_clicked(self, row: int, _column: int) -> None:
        if row < 0 or self.signalsBlocked():
            return
        self.setCurrentRow(row)
        self.rowClicked.emit(row)

    def _set_cell(self, row: int, column: int, text: str, align: int) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        self.table.setItem(row, column, item)


class SpotPanel(QWidget):
    """Spot analysis controls + canvas. Embed into the HDR window or use alone."""

    def __init__(self, parent=None, *, embed: bool = False):
        super().__init__(parent)
        self._embed = bool(embed)
        self.image: np.ndarray | None = None
        self.spots: dict | None = None
        self._batch_images: list[np.ndarray] = []
        self._batch_labels: list[str] = []
        self._roi_sync = False
        self._roi_user_set = False
        self._roi_x = 0
        self._color_updating = False
        self._stops_updating = False
        self._hold_color_levels = False
        self._listing = False
        self._exporting = False
        self._source_folder: Path | None = None
        self.color_stop_boxes: list[QDoubleSpinBox] = []
        self._manual_profile = False
        self._manual_x = False
        self.cross_xy: tuple[int, int] | None = None
        self._x_values: np.ndarray | None = None
        self._distance_stats: tuple[float, float, int] | None = None
        install_input_undo()
        self._build_ui()

    def _build_ui(self) -> None:
        left = QWidget()
        left.setObjectName("sidePanel")
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(_panel_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)
        hint = QLabel("門檻質心擺條帶，加總找下峰與波谷。數值是 × 255 尺度。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        hint.setToolTip(
            "先在整張圖用門檻質心抓上光斑 xy，再以上光斑 X 擺條帶；寬度可調，位置不能手動調。\n"
            "條帶加總不扣中位數；用距離、下峰比與谷深找下峰，兩峰之間最低點當波谷。\n"
            "下光斑可切形狀中心、強度質心、條帶波峰或顯示用斜剖面。log／abs 只改顯示。"
        )
        left_layout.addWidget(hint)

        meta_card, meta_layout = _group_card()
        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        source_row.addWidget(_meta_label("主資料夾"))
        self.source_name = QLabel("尚未載入")
        self.source_name.setObjectName("metaLabel")
        self.source_name.setWordWrap(False)
        self.source_name.setMinimumWidth(0)
        self.source_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.source_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_row.addWidget(self.source_name, 1)
        meta_layout.addLayout(source_row)

        self.batch_host = QWidget()
        batch_row = QHBoxLayout(self.batch_host)
        batch_row.setContentsMargins(0, 0, 0, 0)
        batch_row.setSpacing(8)
        batch_row.addWidget(_meta_label("筆數"))
        self.batch_index = QSpinBox()
        self.batch_index.setRange(0, 0)
        self.batch_index.setToolTip(
            "批次融合的第幾筆，0 是第一張，和主資料夾筆數相同。"
            "張數較少的曝光用該曝光最後一張。"
            "切換時沿用目前的條帶寬度、上下門檻和下光斑演算法。"
        )
        self.batch_index.valueChanged.connect(self._on_batch_index_changed)
        batch_row.addWidget(self.batch_index)
        self.batch_name = QLabel("")
        self.batch_name.setObjectName("metaLabel")
        self.batch_name.setWordWrap(False)
        self.batch_name.setMinimumWidth(0)
        self.batch_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        batch_row.addWidget(self.batch_name, 1)
        self.batch_host.setVisible(False)
        meta_layout.addWidget(self.batch_host)
        left_layout.addWidget(meta_card)

        self.strip_section = _section_label("參數")
        left_layout.addWidget(self.strip_section)
        params_card, params_layout = _group_card()
        self.strip_form = QFormLayout()
        self.strip_form.setContentsMargins(0, 0, 0, 0)
        self.strip_form.setHorizontalSpacing(12)
        self.strip_form.setVerticalSpacing(8)
        self.strip_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.roi_w = self._make_roi_box()
        self.roi_w.setRange(1, 100000)
        self.roi_w.setValue(160)
        self.roi_w.setToolTip(
            "條帶寬度。直條帶時左右以上光斑的 X 為中心。"
            "顯示用斜剖面選 Y 加總時，這是斜向範圍的水平寬度。"
        )
        self.roi_w.valueChanged.connect(self._on_roi_boxes_changed)
        self.strip_form.addRow("寬", self.roi_w)
        self.peak_dist = self._make_roi_box()
        self.peak_dist.setRange(1, 100000)
        self.peak_dist.setValue(50)
        self.peak_dist.setSuffix(" px")
        self.peak_dist.setToolTip(
            "上光斑到下峰的最小距離。條帶加總找波谷時，候選下峰至少要再往下這麼多列；"
            "通過距離、下峰比、谷深條件後取最高的下峰，兩峰之間的最低點當波谷。預設 50。"
        )
        self.peak_dist.valueChanged.connect(self._on_roi_boxes_changed)
        self.strip_form.addRow("距離", self.peak_dist)
        self.peak_ratio = QSpinBox()
        self.peak_ratio.setRange(0, 100)
        self.peak_ratio.setValue(10)
        self.peak_ratio.setSuffix(" %")
        self.peak_ratio.setKeyboardTracking(False)
        self.peak_ratio.setToolTip(
            "下峰條帶加總至少要達到上峰加總的這個比例，否則視為沒找到下光斑。"
            "用來擋主峰後面貼底的雜訊凸起。預設 10%。設 0% 表示不檢查比例。"
        )
        self.peak_ratio.valueChanged.connect(self._on_roi_boxes_changed)
        self.strip_form.addRow("下峰比", self.peak_ratio)
        self.thresh = QSpinBox()
        self.thresh.setRange(5, 95)
        self.thresh.setValue(50)
        self.thresh.setSuffix(" %")
        self.thresh.setKeyboardTracking(False)
        self.thresh.setToolTip(
            "上光斑：相對條帶內最強訊號的百分比。"
            "40% 就是留下不低於峰值 40% 的那一塊核心。"
        )
        self.thresh.valueChanged.connect(self._on_roi_boxes_changed)
        self.show_thresh = QCheckBox("顯示")
        self.show_thresh.setChecked(True)
        self.show_thresh.setToolTip("在熱圖上以半透明白色顯示上光斑的門檻範圍。")
        self.show_thresh.toggled.connect(self._draw_threshold)
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(8)
        thresh_row.addWidget(self.thresh, 1)
        thresh_row.addWidget(self.show_thresh)
        thresh_wrap = QWidget()
        thresh_wrap.setLayout(thresh_row)
        self.strip_form.addRow("上門檻", thresh_wrap)
        self.lower_thresh = QSpinBox()
        self.lower_thresh.setRange(5, 95)
        self.lower_thresh.setValue(50)
        self.lower_thresh.setSuffix(" %")
        self.lower_thresh.setKeyboardTracking(False)
        self.lower_thresh.setToolTip(
            "下光斑：相對波谷以下範圍內最強訊號的百分比。"
            "形狀中心／強度質心／顯示用斜剖面會用到；條帶波峰不用。"
            "下光斑較暗時可調低，例如 30%。"
        )
        self.lower_thresh.valueChanged.connect(self._on_roi_boxes_changed)
        self.strip_form.addRow("下門檻", self.lower_thresh)
        self.center_mode = QComboBox()
        self.center_mode.addItem("形狀中心", "shape")
        self.center_mode.addItem("強度質心", "intensity")
        self.center_mode.addItem("條帶波峰", "peak")
        self.center_mode.addItem("顯示用斜剖面", "line")
        self.center_mode.addItem("斜向迭代（測試）", "slant_iter")
        self.center_mode.setToolTip(
            "形狀中心：範圍內每個像素一樣重。"
            "強度質心：亮的像素權重高，會偏向上面較亮的一側。"
            "條帶波峰：只取條帶加總的兩個 Y 峰；X 固定在條帶中心，距離用兩個 Y 的差。"
            "顯示用斜剖面：抓點和強度質心相同；Y 剖面／加總改沿連線斜向顯示。"
            "斜向迭代（測試）：先直條帶強度質心，再沿連線斜條帶重抓；"
            "直條帶已包住兩團時幾乎不變，下光斑被切邊時才會差。"
        )
        self.center_mode.currentIndexChanged.connect(self._on_center_mode_changed)
        self.strip_form.addRow("演算法", self.center_mode)
        params_layout.addLayout(self.strip_form)
        left_layout.addWidget(params_card)

        left_layout.addWidget(_section_label("結果"))
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_grid = QGridLayout(result_card)
        result_grid.setContentsMargins(10, 10, 10, 10)
        result_grid.setHorizontalSpacing(12)
        result_grid.setVerticalSpacing(6)
        result_grid.setColumnStretch(1, 1)

        def _result_key(text: str) -> QLabel:
            label = QLabel(text)
            label.setObjectName("resultKey")
            return label

        def _result_value() -> QLabel:
            label = QLabel("—")
            label.setObjectName("resultValue")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)
            return label

        self.result_frame_label = _result_value()
        self.result_upper = _result_value()
        self.result_lower = _result_value()
        self.result_distance = _result_value()
        self.result_average = _result_value()
        self.result_std = _result_value()
        self.result_note = QLabel("")
        self.result_note.setObjectName("resultNote")
        self.result_note.setWordWrap(True)

        result_grid.addWidget(self.result_frame_label, 0, 0, 1, 2)
        result_grid.addWidget(_result_key("上光斑"), 1, 0)
        result_grid.addWidget(self.result_upper, 1, 1)
        result_grid.addWidget(_result_key("下光斑"), 2, 0)
        result_grid.addWidget(self.result_lower, 2, 1)
        result_grid.addWidget(_result_key("距離"), 3, 0)
        result_grid.addWidget(self.result_distance, 3, 1)
        result_grid.addWidget(_result_key("平均"), 4, 0)
        result_grid.addWidget(self.result_average, 4, 1)
        result_grid.addWidget(_result_key("標準差"), 5, 0)
        result_grid.addWidget(self.result_std, 5, 1)
        result_grid.addWidget(self.result_note, 6, 0, 1, 2)
        left_layout.addWidget(result_card)

        left_layout.addWidget(_section_label("距離表"))
        self.distance_table = DistanceTable()
        self.distance_table.rowClicked.connect(self._on_distance_clicked)
        left_layout.addWidget(self.distance_table, 1)

        self.btn_export = QPushButton("匯出")
        self.btn_export.setObjectName("primary")
        self.btn_export.setToolTip(
            "選一個父資料夾後，會在裡面新建以目前點位命名的子資料夾，"
            "再寫入三個 CSV、參數 JSON 和熱圖。"
            "熱圖不含色條和 ROI。有批次資料時每一筆一張熱圖，抓法沿用目前設定。"
        )
        self.btn_export.clicked.connect(self._export_csv)
        left_layout.addWidget(self.btn_export)
        self.export_bar = QProgressBar()
        self.export_bar.setRange(0, 1)
        self.export_bar.setValue(0)
        self.export_bar.setFormat("尚未匯出")
        self.export_bar.setTextVisible(True)
        left_layout.addWidget(self.export_bar)
        self.controls_panel = left
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("analysisScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QScrollArea.NoFrame)
        self.controls_scroll.setWidget(left)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setFixedWidth(ANALYSIS_PANEL_W)
        self.controls_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.controls_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        pg.setConfigOption("background", CANVAS)
        pg.setConfigOption("foreground", "#333333")
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground(CANVAS)

        curve_pen = pg.mkPen("#444444", width=1.5)
        curve_brush = pg.mkBrush("#444444")
        self.plot_y = self.win.addPlot(row=0, col=0, title="Y 剖面")
        self.plot_y.setLabel("left", "Y (px)")
        self.plot_y.setLabel("bottom", "強度")
        self.plot_y.getAxis("bottom").setToolTip("點最小值輸入下限，點最大值輸入上限。雙擊剖面復原。")
        self.plot_y.showGrid(x=True, y=True, alpha=0.35)
        self.plot_y.invertY(True)
        self.plot_y.enableAutoRange(x=False, y=False)
        self.curve_y = self.plot_y.plot(pen=curve_pen)
        self.profile_values: np.ndarray | None = None
        self.raw_strip_sum: np.ndarray | None = None
        self.tip_y = pg.TextItem(
            color="#333333",
            anchor=(1, 0),
            fill=pg.mkBrush(255, 255, 255, 230),
            border=pg.mkPen("#c5c5c5"),
        )
        self.tip_y.setZValue(1000)
        self.tip_y.hide()
        self.win.scene().addItem(self.tip_y)
        self.plot_y.vb.sigRangeChanged.connect(lambda *_: self._pin_profile_tip("y"))
        self.marker_upper = self.plot_y.plot(
            pen=None, symbol="+", symbolSize=12, symbolPen=pg.mkPen("#222222", width=2)
        )
        self.marker_lower = self.plot_y.plot(
            pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen("#222222", width=2)
        )
        self.marker_y = self.plot_y.plot(
            pen=None, symbol="o", symbolSize=7, symbolBrush=curve_brush
        )

        self.plot = self.win.addPlot(row=0, col=1, title="合成影像")
        self.plot.setLabel("bottom", "X (px)")
        self.plot.setLabel("left", "Y (px)")
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.35)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.mask_item = pg.ImageItem()
        self.mask_item.setZValue(1)
        self.plot.addItem(self.mask_item)
        cross_pen = pg.mkPen("#666666", width=1, style=Qt.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.plot.addItem(self.vline)
        self.plot.addItem(self.hline)
        self.vline.hide()
        self.hline.hide()
        self.roi = pg.RectROI(
            [0, 0],
            [160, 512],
            pen=pg.mkPen("#FF4FD8", width=2.4, style=Qt.DashLine),
            hoverPen=pg.mkPen("#FFD6F4", width=2.8),
            handlePen=pg.mkPen("#FF4FD8"),
            handleHoverPen=pg.mkPen("#FFD6F4"),
        )
        for handle in [item["item"] for item in list(self.roi.handles)]:
            self.roi.removeHandle(handle)
        self.roi.setZValue(2)
        self.roi.translatable = False
        self.roi.setAcceptedMouseButtons(Qt.NoButton)
        self.roi.sigRegionChanged.connect(self._on_roi_dragged)
        self.plot.addItem(self.roi)
        self.roi.hide()
        self.slant_roi = self.plot.plot(
            pen=pg.mkPen("#FF4FD8", width=2.4, style=Qt.DashLine),
        )
        self.slant_roi.setZValue(2)
        self.slant_roi.hide()
        self.profile_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#f2f2f2", width=1.2, style=Qt.DashLine),
        )
        self.profile_line.setZValue(3)
        self.plot.addItem(self.profile_line)
        self.profile_line.hide()
        spot_pen = pg.mkPen("#f7f7f7", width=2)
        self.spot_upper = self.plot.plot(pen=None, symbol="+", symbolSize=16, symbolPen=spot_pen)
        self.spot_lower = self.plot.plot(pen=None, symbol="x", symbolSize=14, symbolPen=spot_pen)

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        self._jet_cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(colors)), colors)
        self._base_cmap = jet_bw_colormap()
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(self._base_cmap)
        self.hist.plot.setPen(pg.mkPen("#666666", width=1))
        for line in self.hist.region.lines:
            line.setPen(pg.mkPen("#444444", width=1))
        self.hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self.hist.gradient.sigGradientChangeFinished.connect(self._on_gradient_ticks_moved)
        self.win.addItem(self.hist, row=0, col=2)

        self.plot_x = self.win.addPlot(row=1, col=1, title="X 剖面")
        self.plot_x.setLabel("bottom", "X (px)")
        self.plot_x.setLabel("left", "強度")
        self.plot_x.getAxis("left").setToolTip("點最小值輸入下限，點最大值輸入上限。雙擊剖面復原。")
        self.plot_x.showGrid(x=True, y=True, alpha=0.35)
        self.plot_x.enableAutoRange(x=False, y=False)
        self.curve_x = self.plot_x.plot(pen=curve_pen)
        self.marker_x = self.plot_x.plot(pen=None, symbol="o", symbolSize=7, symbolBrush=curve_brush)
        self.tip_x = pg.TextItem(
            color="#333333",
            anchor=(1, 0),
            fill=pg.mkBrush(255, 255, 255, 230),
            border=pg.mkPen("#c5c5c5"),
        )
        self.tip_x.setZValue(1000)
        self.tip_x.hide()
        self.win.scene().addItem(self.tip_x)
        self.plot_x.vb.sigRangeChanged.connect(lambda *_: self._pin_profile_tip("x"))

        grid = self.win.ci.layout
        grid.setRowStretchFactor(0, 4)
        grid.setRowStretchFactor(1, 2)
        grid.setColumnStretchFactor(0, 2)
        grid.setColumnStretchFactor(1, 4)
        grid.setColumnStretchFactor(2, 0)
        self._pin_axis_frame(self.plot_y, 64, 48)
        self._pin_axis_frame(self.plot, 48, 42)
        self._pin_axis_frame(self.plot_x, 80, 48)

        host = QWidget()
        host.setObjectName("analysisViewHost")
        host.setAutoFillBackground(True)
        palette = host.palette()
        palette.setColor(host.backgroundRole(), QColor(CANVAS))
        host.setPalette(palette)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(14, 14, 12, 12)
        host_layout.setSpacing(10)

        self.color_max = self._make_color_box("色條上限")
        self.color_min = self._make_color_box("色條下限")
        self.color_max.setPrefix("上限: ")
        self.color_min.setPrefix("下限: ")
        box_font = self.color_max.font()
        box_font.setPointSize(9)
        box_width = QFontMetrics(box_font).horizontalAdvance("上限: 00000000.00") + 22
        box_style = (
            "QDoubleSpinBox {"
            "background: #ffffff; color: #555555;"
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

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        self._install_y_mode_bar(toolbar)
        self.log_check = QCheckBox("log")
        self.log_check.setToolTip("只改熱圖和條帶圖的顯示。找上下光斑仍用線性強度。")
        self.log_check.toggled.connect(self._on_log_toggled)
        self.abs_check = QCheckBox("abs")
        self.abs_check.setToolTip(
            "熱圖與條帶圖先取絕對值再顯示。找上下光斑仍用原本的線性強度。"
        )
        self.abs_check.toggled.connect(self._on_abs_toggled)
        log_frame, log_row = function_frame()
        log_row.addWidget(self.log_check)
        abs_frame, abs_row = function_frame()
        abs_row.addWidget(self.abs_check)
        toolbar.addWidget(log_frame)
        toolbar.addWidget(abs_frame)
        toolbar.addStretch(1)
        toolbar.addWidget(self.color_max)
        toolbar.addWidget(self.color_min)
        host_layout.addLayout(toolbar)
        # Fusion preprocess blur on its own row so it is not clipped by the color limits.
        gauss_frame, gauss_row = function_frame()
        self.smooth_check = QCheckBox("高斯模糊")
        self.smooth_check.setToolTip(
            "勾選後，每一張曝光在倍率和自己的 color scale 之後做 Gaussian blur，然後才跑演算法。"
            "水平、垂直是核寬，sigma 兩邊共用。某一邊為 1 時該方向不模糊。"
            "會進演算法輸入、浮點結果和匯出 npy。改參數後會重新計算。"
        )
        self.smooth_check.toggled.connect(self._on_gauss_enabled)
        gauss_row.addWidget(self.smooth_check)
        self.smooth_kx, self.smooth_ky, self.smooth_sigma = add_gaussian_controls(gauss_row)
        gauss_row.addStretch(1)
        if not self._embed:
            gauss_frame.hide()
        host_layout.addWidget(gauss_frame)
        host_layout.addWidget(self.win, 1)

        self.hover = QLabel("游標：—")
        self.hover.setStyleSheet(
            "color: #2c333a; background: #ffffff; border: 1px solid #e1e5ea;"
            "border-radius: 6px; padding: 6px 10px;"
        )
        self.hover.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        status_row.addWidget(self.hover, 0)
        status_row.addStretch(1)
        for group in (COLOR_STOPS[:4], COLOR_STOPS[4:]):
            frame, row = function_frame()
            for name, _english, _color, pos in group:
                box = self._make_stop_box(name, pos)
                box.valueChanged.connect(self._on_color_stop_changed)
                self.color_stop_boxes.append(box)
                row.addWidget(box)
            status_row.addWidget(frame, 0)
        host_layout.addLayout(status_row)
        self.view_host = host
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.plot.scene().sigMouseMoved.connect(self._on_profile_moved)

        if not self._embed:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)
            layout.addWidget(self.controls_scroll)
            layout.addWidget(self.view_host, 1)

    def set_source_folder(self, root: Path | None) -> None:
        if root is None:
            self._source_folder = None
            self.source_name.setText("尚未載入")
            self.source_name.setToolTip("")
            return
        self._source_folder = Path(root)
        self.source_name.setText(self._source_folder.name)
        self.source_name.setToolTip(str(self._source_folder))

    def set_image(self, image: np.ndarray) -> None:
        self._batch_images = []
        self._batch_labels = []
        self.batch_host.setVisible(False)
        self._apply_image(image, keep_roi=False)
        self._refresh_distance_list()

    def set_batch(self, images: list[np.ndarray], labels: list[str]) -> None:
        self._batch_images = [np.asarray(image) for image in images]
        self._batch_labels = list(labels)
        count = len(self._batch_images)
        if count < 1:
            self.batch_host.setVisible(False)
            return
        self.batch_host.setVisible(True)
        self.batch_index.blockSignals(True)
        self.batch_index.setRange(0, count - 1)
        self.batch_index.setValue(0)
        self.batch_index.blockSignals(False)
        self._apply_image(self._batch_images[0], keep_roi=True)
        self._update_batch_caption()
        self._refresh_distance_list()

    def _on_batch_index_changed(self, index: int) -> None:
        if not self._batch_images or not (0 <= index < len(self._batch_images)):
            return
        self._apply_image(self._batch_images[index], keep_roi=True)
        self._update_batch_caption()
        if not self._listing:
            self.distance_table.blockSignals(True)
            self.distance_table.setCurrentRow(index)
            self.distance_table.blockSignals(False)

    def _update_batch_caption(self) -> None:
        index = int(self.batch_index.value())
        total = len(self._batch_images)
        name = self._batch_labels[index] if 0 <= index < len(self._batch_labels) else ""
        self.batch_name.setText(f"{index + 1} / {total}  {name}")
        self.batch_name.setToolTip(name)

    def _apply_image(self, image: np.ndarray, keep_roi: bool) -> None:
        self.image = np.asarray(image, dtype=np.float64)
        height, width = self.image.shape
        if not keep_roi:
            self.cross_xy = None
            self._manual_x = False
            self._manual_profile = False
        self.roi_w.setMaximum(width)
        if self.roi_w.value() > width or (not keep_roi and not self._roi_user_set):
            self._roi_sync = True
            try:
                self.roi_w.setValue(min(self.roi_w.value() if keep_roi else 160, width))
            finally:
                self._roi_sync = False
            if not keep_roi:
                self._roi_user_set = False
        self.roi.maxBounds = QRectF(0, 0, width, height)
        self._sync_roi_rect()
        self._show_heatmap()
        self._refresh_spots()
        self.plot.setXRange(0, width, padding=0)
        self.plot.setYRange(0, height, padding=0)

    def _abs_enabled(self) -> bool:
        return bool(self.abs_check.isChecked())

    def _on_gauss_enabled(self, checked: bool) -> None:
        for box in (self.smooth_kx, self.smooth_ky, self.smooth_sigma):
            box.setEnabled(bool(checked))

    def _view_image(self) -> np.ndarray:
        return apply_abs_then_color_scale(
            self.image,
            self._abs_enabled(),
            None,
            None,
            False,
        )

    def _display_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if not self.log_check.isChecked():
            return data
        return np.log10(np.maximum(data, 1.0))

    def _level_span(self, shown: np.ndarray) -> tuple[float, float]:
        lo = float(np.min(shown))
        hi = float(np.max(shown))
        if lo >= hi:
            hi = lo + 1.0
        return lo, hi

    def _apply_colormap(self) -> None:
        self._stops_updating = True
        try:
            self.hist.gradient.setColorMap(self._base_cmap)
        finally:
            self._stops_updating = False

    def _show_heatmap(self, refit: bool = False) -> None:
        if self.image is None:
            return
        shown = self._display_values(self._view_image())
        self.image_item.setImage(shown.T, autoLevels=False)
        if refit or not self._hold_color_levels:
            self._set_color_levels(*self._level_span(shown))
        else:
            lo, hi = self.hist.getLevels()
            self._set_color_levels(float(lo), float(hi))

    def _make_roi_box(self) -> QSpinBox:
        box = QSpinBox()
        box.setRange(0, 100000)
        box.setKeyboardTracking(False)
        return box

    def _make_stop_box(self, name: str, pos: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 1.0)
        box.setDecimals(2)
        box.setSingleStep(0.01)
        box.setKeyboardTracking(False)
        box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        box.setPrefix(f"{name}: ")
        box.setToolTip(f"色條上{name}色三角的位置。0 在下方，1 在上方。只改顏色分佈，不重算找點。")
        box.setValue(float(pos))
        box.setFixedWidth(78)
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
        self._base_cmap = colormap_from_stops(ordered)
        self._stops_updating = True
        try:
            for box, pos in zip(self.color_stop_boxes, ordered):
                box.blockSignals(True)
                box.setValue(pos)
                box.blockSignals(False)
        finally:
            self._stops_updating = False
        self._apply_colormap()

    def _on_gradient_ticks_moved(self, *_args) -> None:
        if self._stops_updating or not self.color_stop_boxes:
            return
        by_color = {
            color: box
            for box, (_name, _english, color, _pos) in zip(self.color_stop_boxes, COLOR_STOPS)
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
        self._base_cmap = colormap_from_stops([box.value() for box in self.color_stop_boxes])

    def apply_color_display(self, stops: list[float], levels: tuple[float, float]) -> None:
        """Use the fusion window's current color-stop positions and level limits."""
        self._apply_color_stops(stops)
        self._hold_color_levels = True
        self._set_color_levels(float(levels[0]), float(levels[1]))

    def _make_color_box(self, tooltip: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e15, 1e15)
        box.setDecimals(2)
        box.setKeyboardTracking(False)
        box.setToolTip(tooltip)
        return box

    def _strip_origin(self, center: float, width: int) -> tuple[int, int]:
        image_width = 1 if self.image is None else int(self.image.shape[1])
        return self._strip_origin_for(image_width, center, width)

    def _strip_origin_for(self, image_width: int, center: float, width: int) -> tuple[int, int]:
        image_width = max(1, int(image_width))
        roi_w = int(np.clip(width, 1, image_width))
        roi_x = int(round(float(center) - roi_w / 2.0))
        roi_x = int(np.clip(roi_x, 0, image_width - roi_w))
        return roi_x, roi_w

    def _roi_x_span(self, width: int) -> tuple[int, int]:
        """Inclusive-exclusive X span of the pink strip ROI on the full image."""
        width = max(0, int(width))
        spots = self.spots or {}
        bounds = spots.get("bounds")
        if bounds is not None:
            x0 = int(np.clip(bounds[0], 0, width))
            x1 = int(np.clip(bounds[2], x0, width))
            if x1 > x0:
                return x0, x1
        x0 = int(np.clip(self._roi_x, 0, width))
        x1 = int(np.clip(x0 + max(1, int(self.roi_w.value())), x0, width))
        if x1 <= x0 and width > 0:
            return 0, width
        return x0, x1

    def _roi_column_sum(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Sum each row inside the pink ROI. Returns (column_sum, roi_width)."""
        data = np.asarray(image, dtype=np.float64)
        if data.ndim != 2 or data.size == 0:
            return np.zeros(0, dtype=np.float64), 0
        height, width = data.shape
        x0, x1 = self._roi_x_span(width)
        if x1 <= x0:
            return np.zeros(height, dtype=np.float64), 0
        return np.nan_to_num(data[:, x0:x1].sum(axis=1), nan=0.0), int(x1 - x0)

    def _sync_roi_rect(self) -> None:
        if self.image is None:
            return
        height = int(self.image.shape[0])
        self._roi_sync = True
        try:
            self.roi.setPos(float(self._roi_x), 0.0, update=False)
            self.roi.setSize((float(max(1, self.roi_w.value())), float(height)), update=False)
        finally:
            self._roi_sync = False

    def _on_roi_boxes_changed(self, _value: int | None = None) -> None:
        if self._roi_sync or self.image is None:
            return
        self._roi_user_set = True
        self._refresh_spots()
        self._refresh_distance_list()

    def _on_roi_dragged(self) -> None:
        if self._roi_sync or self.image is None:
            return
        self._sync_roi_rect()

    def _on_log_toggled(self, _checked: bool) -> None:
        self._manual_profile = False
        self._show_heatmap(refit=True)
        self._refresh_spots()

    def _on_abs_toggled(self, _checked: bool) -> None:
        self._manual_profile = False
        self._show_heatmap(refit=True)
        self._draw_profile()

    def _refresh_spots(self) -> None:
        if self.image is None:
            self.spots = None
            self._draw_markers()
            self._sync_profile_mark()
            return
        self.spots, self._roi_x = self._measure_image(self.image)
        self._snap_crosshair_to_upper()
        self._sync_roi_rect()
        self._draw_markers()
        self._draw_profile()
        self._sync_profile_mark()

    def _snap_crosshair_to_upper(self) -> None:
        """Put the profile crosshair on the upper spot so markers sit on the found peak."""
        if self.image is None:
            return
        spots = self.spots or {}
        upper = spots.get("upper")
        if upper is None:
            return
        height, width = self.image.shape
        self.cross_xy = (
            int(np.clip(round(float(upper[0])), 0, width - 1)),
            int(np.clip(round(float(upper[1])), 0, height - 1)),
        )
        self._manual_x = False
        self._manual_profile = False

    def _line_mode(self) -> bool:
        return str(self.center_mode.currentData()) == "line"

    def _slant_iter_mode(self) -> bool:
        return str(self.center_mode.currentData()) == "slant_iter"

    def _y_is_sum(self) -> bool:
        return bool(self.y_sum_btn.isChecked())

    def _line_sum_mode(self) -> bool:
        return self._line_mode() and self._y_is_sum()

    def _pin_axis_frame(self, plot, left: int, bottom: int) -> None:
        for name in ("left", "right", "top", "bottom"):
            plot.getAxis(name).setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        plot.getAxis("left").setWidth(left)
        plot.getAxis("bottom").setHeight(bottom)

    def _install_y_mode_bar(self, layout) -> None:
        frame, row = function_frame()
        tip = (
            "Y 剖面是十字所在那一欄的強度；顯示用斜剖面時改為沿上下光斑虛線取值。"
            "Y 加總是粉紅 ROI／條帶範圍內每一列沿 X 加總；顯示用斜剖面時改為斜向範圍加總。"
            "一次只能選一個。下方 X 剖面是十字所在那一列。"
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
        self.y_sum_btn = QPushButton("Y 加總")
        button_metrics = QFontMetrics(self.y_slice_btn.font())
        button_w = max(
            button_metrics.horizontalAdvance("Y 剖面"),
            button_metrics.horizontalAdvance("Y 加總"),
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
        self.y_mode_frame = frame
        self.y_mode_frame.setVisible(True)
        if layout is not None:
            layout.addWidget(frame)

    def _on_y_mode_changed(self, _button=None) -> None:
        self._manual_profile = False
        self._sync_profile_mark()
        self._draw_profile()

    def _sync_profile_mark(self) -> None:
        line_mode = self._line_mode()
        sum_mode = self._line_sum_mode()
        self.y_mode_frame.setVisible(True)
        self.roi_w.setVisible(True)
        width_label = self.strip_form.labelForField(self.roi_w)
        if width_label is not None:
            width_label.setVisible(True)
        self.peak_dist.setVisible(True)
        dist_label = self.strip_form.labelForField(self.peak_dist)
        if dist_label is not None:
            dist_label.setVisible(True)
        self.peak_ratio.setVisible(True)
        ratio_label = self.strip_form.labelForField(self.peak_ratio)
        if ratio_label is not None:
            ratio_label.setVisible(True)
        # Lower blob threshold is unused in peak mode (Y from strip-sum only).
        peak_mode = str(self.center_mode.currentData()) == "peak"
        self.lower_thresh.setEnabled(not peak_mode)
        self.strip_section.setVisible(True)
        # Keep vertical ROI in slant_iter so first-pass strip can be compared to slant outline.
        self.roi.setVisible(self.image is not None and not line_mode)
        upper = None if not self.spots else self.spots.get("upper")
        lower = None if not self.spots else self.spots.get("lower")
        draw_link = (line_mode or self._slant_iter_mode()) and self.image is not None and upper is not None
        if draw_link:
            x1, y1 = float(upper[0]), float(upper[1])
            if lower is None:
                self.profile_line.setAngle(90)
                self.profile_line.setPos(x1)
            else:
                x2, y2 = float(lower[0]), float(lower[1])
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    self.profile_line.setAngle(90)
                    self.profile_line.setPos(x1)
                else:
                    self.profile_line.setAngle(float(np.degrees(np.arctan2(dy, dx))))
                    self.profile_line.setPos((x1, y1))
            self.profile_line.show()
        else:
            self.profile_line.hide()
        show_slant = (sum_mode or self._slant_iter_mode()) and self.image is not None and upper is not None
        if show_slant:
            height, width = self.image.shape
            xs, ys = slanted_strip_outline(upper, lower, int(self.roi_w.value()), height, width)
            self.slant_roi.setData(xs, ys)
            self.slant_roi.show()
            if sum_mode:
                self.y_caption.setText("斜向範圍水平加總")
            elif self._slant_iter_mode():
                self.y_caption.setText("斜向迭代條帶（測試）")
            else:
                self.y_caption.setText("")
        else:
            self.slant_roi.setData([], [])
            self.slant_roi.hide()
            if line_mode and upper is not None:
                self.y_caption.setText("斜線剖面")
            else:
                self.y_caption.setText("")

    def _measure_image(self, image: np.ndarray) -> tuple[dict, int]:
        """Find both spots with the current width, thresholds, and center mode."""
        data = np.asarray(image, dtype=np.float64)
        height, width = data.shape
        thresh = float(self.thresh.value())
        lower_thresh = float(self.lower_thresh.value())
        seed, _seed_mask = find_upper_spot(data, thresh)
        if seed is None:
            return {
                "bounds": None,
                "profile": np.array([]),
                "smooth": np.array([]),
                "mask": None,
                "upper": None,
                "lower": None,
                "raw_sum": np.array([]),
            }, 0
        origin, roi_w = self._strip_origin_for(width, float(seed[0]), int(self.roi_w.value()))
        spots = locate_dual_spots(
            data,
            origin,
            0,
            roi_w,
            int(height),
            min_peak_distance=int(self.peak_dist.value()),
            min_peak_ratio=float(self.peak_ratio.value()) / 100.0,
            use_log=self.log_check.isChecked(),
            thresh_percent=thresh,
            lower_thresh_percent=lower_thresh,
            center_mode=str(self.center_mode.currentData()),
        )
        return spots, origin

    def _draw_markers(self) -> None:
        spots = self.spots or {}
        upper = spots.get("upper")
        lower = spots.get("lower")
        if upper is None:
            self.spot_upper.setData([], [])
        else:
            self.spot_upper.setData([upper[0]], [upper[1]])
        if lower is None:
            self.spot_lower.setData([], [])
        else:
            self.spot_lower.setData([lower[0]], [lower[1]])
        self._update_spot_label()
        self._draw_threshold()

    def _update_spot_label(self) -> None:
        if self.image is None:
            self.result_frame_label.setText("尚未載入合成影像")
            self.result_frame_label.show()
            for label in (
                self.result_upper,
                self.result_lower,
                self.result_distance,
                self.result_average,
                self.result_std,
            ):
                label.setText("—")
            self.result_note.setText("")
            return
        spots = self.spots or {}
        upper = spots.get("upper")
        lower = spots.get("lower")
        if self._batch_images:
            index = int(self.batch_index.value())
            self.result_frame_label.setText(f"第 {index} 筆")
            self.result_frame_label.show()
        else:
            self.result_frame_label.hide()
            self.result_frame_label.setText("")
        peak_mode = str(self.center_mode.currentData()) == "peak"
        if peak_mode:
            self.result_upper.setText(
                "未找到" if upper is None else f"+ Y = {upper[1]:.1f}"
            )
            self.result_lower.setText(
                "未找到" if lower is None else f"× Y = {lower[1]:.1f}"
            )
        else:
            self.result_upper.setText(
                "未找到" if upper is None else f"+ ({upper[0]:.1f}, {upper[1]:.1f})"
            )
            self.result_lower.setText(
                "未找到" if lower is None else f"× ({lower[0]:.1f}, {lower[1]:.1f})"
            )
        self.result_distance.setText(self._distance_value_text(upper, lower))
        average_text, std_text, note = self._distance_stats_values()
        self.result_average.setText(average_text)
        self.result_std.setText(std_text)
        self.result_note.setText(note)

    @staticmethod
    def _safe_folder_name(name: str) -> str:
        """Strip characters that Windows / common filesystems reject in folder names."""
        cleaned = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in str(name).strip())
        cleaned = cleaned.rstrip(" .")
        return cleaned or "未命名點位"

    def _export_spot_folder_name(self) -> str:
        if self._source_folder is not None:
            name = self._source_folder.name.strip()
            if name:
                return self._safe_folder_name(name)
        label = self.source_name.text().strip()
        if label and label != "尚未載入":
            return self._safe_folder_name(label)
        return "未命名點位"

    def _export_csv(self) -> None:
        if self.image is None:
            QMessageBox.warning(self, "尚未分析", "請先載入影像。")
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇匯出父資料夾")
        if not folder:
            return
        spot_name = self._export_spot_folder_name()
        directory = Path(folder) / spot_name
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spots_path = directory / f"spots_{stamp}.csv"
        raw_path = directory / f"strip_sum_raw_{stamp}.csv"
        summed_path = directory / f"strip_sum_{stamp}.csv"
        params_path = directory / f"params_{stamp}.json"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._exporting = True
        self.btn_export.setEnabled(False)
        try:
            items = self._export_items()
            total = max(len(items) * 2, 1)
            self._write_analysis_csvs(spots_path, raw_path, summed_path, items, total)
            self._write_params(params_path, stamp)
            heat_count = self._write_heatmaps(directory, stamp, items, total)
            self._set_progress(total, total, "匯出完成")
        except Exception as exc:
            self._set_progress(0, 1, "匯出失敗")
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        finally:
            self._exporting = False
            self.btn_export.setEnabled(True)
            QApplication.restoreOverrideCursor()
        if heat_count == 1:
            heat_text = f"heatmap_{stamp}.png"
        else:
            heat_text = f"heatmap_{stamp}_筆數.png 共 {heat_count} 張"
        QMessageBox.information(
            self,
            "匯出完成",
            f"已寫入資料夾\n{directory}\n\n"
            f"{spots_path.name}\n{raw_path.name}\n{summed_path.name}\n"
            f"{params_path.name}\n{heat_text}",
        )

    def _export_items(self) -> list[tuple[int, str, np.ndarray]]:
        if self._batch_images:
            labels = self._batch_labels
            return [
                (index, labels[index] if index < len(labels) else "", np.asarray(image))
                for index, image in enumerate(self._batch_images)
            ]
        return [(0, "", np.asarray(self.image))]

    def _write_analysis_csvs(
        self,
        spots_path: Path,
        raw_path: Path,
        summed_path: Path,
        items: list[tuple[int, str, np.ndarray]] | None = None,
        total: int | None = None,
    ) -> None:
        if items is None:
            items = self._export_items()
        count = len(items)
        if total is None:
            total = max(count, 1)
        spot_rows: list[list[str]] = []
        raw_columns: list[tuple[str, np.ndarray]] = []
        summed_columns: list[tuple[str, np.ndarray]] = []
        single = count == 1 and not self._batch_images
        for step, (index, name, image) in enumerate(items):
            self._set_progress(step, total, f"匯出 {step + 1} / {count}")
            spots, _origin = self._measure_image(image)
            header = "strip_sum" if single else str(index)
            raw_header = "strip_sum_raw" if single else str(index)
            raw_columns.append((raw_header, self._raw_strip(spots, int(image.shape[0]), image)))
            summed_columns.append((header, self._shown_strip(spots, int(image.shape[0]), image)))
            spot_rows.append(self._spot_row(index, name, spots))
        spots_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_table(
            spots_path,
            [
                "frame",
                "file",
                "strip_width",
                "min_peak_distance",
                "min_peak_ratio_percent",
                "threshold_percent",
                "lower_threshold_percent",
                "center_mode",
                "upper_x",
                "upper_y",
                "lower_x",
                "lower_y",
                "distance_px",
                "distance_um",
            ],
            spot_rows,
        )
        self._write_curves(raw_path, raw_columns)
        self._write_curves(summed_path, summed_columns)

    def _write_params(self, path: Path, stamp: str) -> None:
        items = self._export_items()
        lo, hi = self.hist.getLevels()
        payload = {
            "timestamp": stamp,
            "batch": bool(self._batch_images),
            "frame_count": len(items),
            "files": [name for _index, name, _image in items],
            "strip_width": int(self.roi_w.value()),
            "min_peak_distance": int(self.peak_dist.value()),
            "min_peak_ratio_percent": int(self.peak_ratio.value()),
            "threshold_percent": int(self.thresh.value()),
            "lower_threshold_percent": int(self.lower_thresh.value()),
            "center_mode": str(self.center_mode.currentData()),
            "line_y_mode": "sum" if self._line_sum_mode() else "profile",
            "distance_stats": None
            if self._distance_stats is None
            else {
                "average_px": float(self._distance_stats[0]),
                "std_px": float(self._distance_stats[1]),
                "count": int(self._distance_stats[2]),
                "average_um": float(self._distance_stats[0] * 3.45),
                "std_um": float(self._distance_stats[1] * 3.45),
            },
            "log": bool(self.log_check.isChecked()),
            "abs": self._abs_enabled(),
            "color_levels": {"min": float(lo), "max": float(hi)},
            "color_stops": [
                {"name": english, "position": float(box.value())}
                for box, (_zh, english, _color, _pos) in zip(self.color_stop_boxes, COLOR_STOPS)
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_heatmaps(
        self,
        directory: Path,
        stamp: str,
        items: list[tuple[int, str, np.ndarray]] | None = None,
        total: int | None = None,
    ) -> int:
        if items is None:
            items = self._export_items()
        count = len(items)
        if total is None:
            total = max(count, 1)
        single = count == 1 and not self._batch_images
        for step, (index, _name, image) in enumerate(items):
            done = count + step + 1 if total >= count * 2 else step + 1
            self._set_progress(done, total, f"熱圖 {step + 1} / {count}")
            filename = f"heatmap_{stamp}.png" if single else f"heatmap_{stamp}_{index}.png"
            self._write_heatmap_png(directory / filename, self._shown_image(image))
        return count

    def _set_progress(self, value: int, total: int, text: str) -> None:
        self.export_bar.setRange(0, max(int(total), 1))
        self.export_bar.setValue(int(value))
        self.export_bar.setFormat(text)
        QApplication.processEvents()

    def _on_center_mode_changed(self, _index: int) -> None:
        self._manual_profile = False
        self._refresh_spots()
        self._refresh_distance_list()

    def _on_distance_clicked(self, row: int) -> None:
        if self._listing or self._exporting or not self._batch_images:
            return
        if 0 <= row < len(self._batch_images) and int(self.batch_index.value()) != row:
            self.batch_index.setValue(row)

    def _refresh_distance_list(self) -> None:
        if self._listing or self._exporting or self.image is None:
            return
        self._listing = True
        try:
            items = self._export_items()
            count = len(items)
            rows: list[tuple[str, str, str]] = []
            distances: list[float] = []
            for step, (index, _name, image) in enumerate(items):
                self._set_progress(step, count, f"距離 {step + 1} / {count}")
                spots, _origin = self._measure_image(image)
                cells = self._distance_cells(index, spots)
                rows.append(cells)
                if cells[1] != "--":
                    distances.append(float(cells[1]))
            self._distance_stats = self._compute_distance_stats(distances)
            self.distance_table.blockSignals(True)
            self.distance_table.fill(rows)
            current = int(self.batch_index.value()) if self._batch_images else 0
            if 0 <= current < self.distance_table.count():
                self.distance_table.setCurrentRow(current)
            self.distance_table.blockSignals(False)
            self._update_spot_label()
            self._set_progress(count, count, "距離已更新" if count else "尚未分析")
        finally:
            self._listing = False

    @staticmethod
    def _compute_distance_stats(distances: list[float]) -> tuple[float, float, int] | None:
        if not distances:
            return None
        values = np.asarray(distances, dtype=np.float64)
        mean = float(np.mean(values))
        if values.size >= 2:
            std = float(np.std(values, ddof=1))
        else:
            std = 0.0
        return mean, std, int(values.size)

    def _distance_cells(self, index: int, spots: dict) -> tuple[str, str, str]:
        upper = spots.get("upper")
        lower = spots.get("lower")
        if upper is None or lower is None:
            return str(index), "--", "--"
        distance = float(np.hypot(float(lower[0]) - float(upper[0]), float(lower[1]) - float(upper[1])))
        return str(index), f"{distance:.2f}", f"{distance * 3.45:.2f}"

    def _shown_image(self, image: np.ndarray) -> np.ndarray:
        shown = apply_abs_then_color_scale(
            np.asarray(image, dtype=np.float64),
            self._abs_enabled(),
            None,
            None,
            False,
        )
        return self._display_values(shown)

    def _write_heatmap_png(self, path: Path, shown: np.ndarray) -> None:
        import cv2

        data = np.nan_to_num(np.asarray(shown, dtype=np.float64), nan=0.0)
        lo, hi = [float(value) for value in self.hist.getLevels()]
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        cmap = self.hist.gradient.colorMap() or self._base_cmap
        rgba = np.asarray(cmap.map(norm, mode="byte"))
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("熱圖沒有寫入。")
        path.write_bytes(encoded.tobytes())

    def _spot_row(self, frame: int, name: str, spots: dict) -> list[str]:
        upper = spots.get("upper")
        lower = spots.get("lower")
        upper_x, upper_y = self._point_cells(upper)
        lower_x, lower_y = self._point_cells(lower)
        if upper is None or lower is None:
            distance_px, distance_um = "", ""
        else:
            distance = float(np.hypot(float(lower[0]) - float(upper[0]), float(lower[1]) - float(upper[1])))
            distance_px = f"{distance:.8g}"
            distance_um = f"{distance * 3.45:.8g}"
        return [
            str(frame),
            name,
            str(int(self.roi_w.value())),
            str(int(self.peak_dist.value())),
            str(int(self.peak_ratio.value())),
            str(int(self.thresh.value())),
            str(int(self.lower_thresh.value())),
            str(self.center_mode.currentData()),
            upper_x,
            upper_y,
            lower_x,
            lower_y,
            distance_px,
            distance_um,
        ]

    def _raw_strip(self, spots: dict, height: int, image: np.ndarray | None = None) -> np.ndarray:
        if self._line_sum_mode() and image is not None:
            upper = spots.get("upper")
            lower = spots.get("lower")
            if upper is not None:
                raw = slanted_strip_sum(image, upper, lower, int(self.roi_w.value()))
                values = np.full(height, np.nan)
                count = min(len(raw), height)
                values[:count] = raw[:count]
                return values
        values = np.full(height, np.nan)
        raw_sum = spots.get("raw_sum")
        bounds = spots.get("bounds")
        if raw_sum is None or bounds is None or len(raw_sum) == 0:
            return values
        y0 = int(bounds[1])
        count = min(len(raw_sum), max(0, height - y0))
        values[y0 : y0 + count] = np.asarray(raw_sum[:count], dtype=float)
        return values

    def _shown_strip(self, spots: dict, height: int, image: np.ndarray | None = None) -> np.ndarray:
        line_curve = self._line_curve(image, spots, height) if image is not None else None
        if line_curve is not None:
            return line_curve
        values = np.zeros(height, dtype=float)
        smooth = spots.get("smooth")
        bounds = spots.get("bounds")
        if smooth is not None and bounds is not None and len(smooth):
            y0 = int(bounds[1])
            count = min(len(smooth), max(0, height - y0))
            values[y0 : y0 + count] = np.asarray(smooth[:count], dtype=float)
        if self._abs_enabled():
            values = np.abs(values)
        return values

    def _line_curve(
        self,
        image: np.ndarray | None,
        spots: dict,
        height: int | None = None,
    ) -> np.ndarray | None:
        """Y curve for line mode: slanted X-interpolated profile or strip sum."""
        if not self._line_mode() or image is None:
            return None
        upper = spots.get("upper")
        if upper is None:
            return None
        rows = int(image.shape[0] if height is None else height)
        if self._line_sum_mode():
            raw = slanted_strip_sum(image, upper, spots.get("lower"), int(self.roi_w.value()))
            if raw.size == 0:
                return np.zeros(rows, dtype=float)
            shown = np.abs(raw) if self._abs_enabled() else np.asarray(raw, dtype=float)
            if self.log_check.isChecked():
                shown = np.log10(np.maximum(shown, 1.0))
            values = np.zeros(rows, dtype=float)
            count = min(len(shown), rows)
            values[:count] = shown[:count]
            return values
        column = self._displayed_slant_profile(image, spots)
        if column is None:
            return None
        values = np.zeros(rows, dtype=float)
        count = min(len(column), rows)
        values[:count] = column[:count]
        return values

    def _displayed_slant_profile(self, image: np.ndarray | None, spots: dict) -> np.ndarray | None:
        """Profile along the two-spot line, using the same display plane as the heatmap."""
        if not self._line_mode() or image is None:
            return None
        upper = spots.get("upper")
        if upper is None:
            return None
        shown = self._display_values(
            apply_abs_then_color_scale(
                np.asarray(image, dtype=np.float64),
                self._abs_enabled(),
                None,
                None,
                False,
            )
        )
        return slanted_line_profile(shown, upper, spots.get("lower"))

    @staticmethod
    def _write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _write_curves(path: Path, columns: list[tuple[str, np.ndarray]]) -> None:
        height = max((len(values) for _name, values in columns), default=0)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["y_px", *[name for name, _values in columns]])
            for y in range(height):
                row: list[object] = [y]
                for _name, values in columns:
                    if y >= len(values) or not np.isfinite(values[y]):
                        row.append("")
                    else:
                        row.append(f"{float(values[y]):.8g}")
                writer.writerow(row)

    @staticmethod
    def _point_cells(point) -> tuple[str, str]:
        if point is None:
            return "", ""
        return f"{float(point[0]):.8g}", f"{float(point[1]):.8g}"

    @staticmethod
    def _format_px_um(px: float) -> str:
        return f"{px:.2f} px / {px * 3.45:.2f} μm"

    def _distance_value_text(self, upper, lower) -> str:
        if upper is None or lower is None:
            return "—"
        if str(self.center_mode.currentData()) == "peak":
            dist = abs(float(lower[1]) - float(upper[1]))
        else:
            dist = float(
                np.hypot(float(lower[0]) - float(upper[0]), float(lower[1]) - float(upper[1]))
            )
        return self._format_px_um(dist)

    def _distance_stats_values(self) -> tuple[str, str, str]:
        stats = self._distance_stats
        if stats is None:
            return "—", "—", ""
        mean, std, count = stats
        note = f"統計 {count} 筆有效距離"
        return self._format_px_um(mean), self._format_px_um(std), note

    def _draw_threshold(self, _checked: bool | None = None) -> None:
        spots = self.spots or {}
        mask = spots.get("mask")
        if (
            self.image is None
            or mask is None
            or not self.show_thresh.isChecked()
            or not np.any(mask)
        ):
            self.mask_item.clear()
            return
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask, 0] = 255
        rgba[mask, 1] = 255
        rgba[mask, 2] = 255
        rgba[mask, 3] = 110
        self.mask_item.setImage(rgba.transpose(1, 0, 2), autoLevels=False)

    def _ensure_crosshair(self) -> tuple[int, int] | None:
        if self.image is None:
            return None
        height, width = self.image.shape
        if self.cross_xy is not None:
            x, y = self.cross_xy
            if 0 <= x < width and 0 <= y < height:
                return x, y
        spots = self.spots or {}
        upper = spots.get("upper")
        if upper is not None:
            x = int(np.clip(round(float(upper[0])), 0, width - 1))
            y = int(np.clip(round(float(upper[1])), 0, height - 1))
            self.cross_xy = (x, y)
            return self.cross_xy
        shown = np.nan_to_num(self._view_image(), nan=0.0)
        index = int(np.argmax(shown))
        y, x = np.unravel_index(index, shown.shape)
        self.cross_xy = (int(x), int(y))
        return self.cross_xy

    def _draw_profile(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape
        positions = np.arange(height, dtype=float) + 0.5
        spots = self.spots or {}
        cross = self._ensure_crosshair()
        if cross is None:
            return
        cx, cy = cross
        display = self._display_values(self._view_image())
        row = np.nan_to_num(display[cy, :], nan=0.0)
        self._x_values = row
        self.curve_x.setData(np.arange(width, dtype=float) + 0.5, row)
        self.marker_x.setData([cx + 0.5], [float(row[cx])])
        self.vline.setPos(cx + 0.5)
        self.hline.setPos(cy + 0.5)
        self.vline.show()
        self.hline.show()
        self.plot_x.setTitle(f"X 剖面  y={cy}")
        self.plot_x.setLabel("left", "log10 強度" if self.log_check.isChecked() else "強度")

        line_curve = self._line_curve(self.image, spots, height)
        if line_curve is not None:
            shown = line_curve
            if self._y_is_sum():
                self.y_caption.setText("斜向範圍水平加總")
            else:
                self.y_caption.setText("斜線剖面")
        elif self._y_is_sum():
            linear = np.asarray(self.image, dtype=np.float64)
            if self._abs_enabled():
                linear = np.abs(linear)
            summed, roi_w = self._roi_column_sum(linear)
            shown = self._display_values(summed)
            self.y_caption.setText(f"ROI 寬={roi_w}")
        else:
            shown = np.nan_to_num(display[:, cx], nan=0.0)
            self.y_caption.setText(f"x={cx}")
        self.profile_values = shown
        self.raw_strip_sum = self._raw_strip(spots, height, self.image)
        self.curve_y.setData(shown, positions)
        self.marker_y.setData([float(shown[cy])], [cy + 0.5])
        self.plot_y.setLabel("bottom", self._strip_caption())
        self.plot_y.setTitle("Y 加總" if self._y_is_sum() else "Y 剖面")
        upper = spots.get("upper")
        lower = spots.get("lower")
        if upper is None:
            self.marker_upper.setData([], [])
        else:
            index = int(np.clip(round(upper[1]), 0, height - 1))
            self.marker_upper.setData([float(shown[index])], [index + 0.5])
        if lower is None:
            self.marker_lower.setData([], [])
        else:
            index = int(np.clip(round(lower[1]), 0, height - 1))
            self.marker_lower.setData([float(shown[index])], [index + 0.5])
        if not self._manual_x:
            self._fit_profile_intensity(self.plot_x, row, "y")
            self.plot_x.setXRange(0, width, padding=0)
        if not self._manual_profile:
            self._fit_profile_intensity(self.plot_y, shown, "x")
            self.plot_y.setYRange(0, height, padding=0)

    def _fit_profile_intensity(self, plot, values: np.ndarray, axis: str) -> None:
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

    def _set_color_levels(self, lo: float, hi: float) -> None:
        if lo >= hi:
            hi = lo + 1.0
        self._color_updating = True
        try:
            self.hist.setHistogramRange(lo, hi, padding=0)
            self.hist.setLevels(lo, hi)
            self.color_min.blockSignals(True)
            self.color_max.blockSignals(True)
            self.color_min.setValue(lo)
            self.color_max.setValue(hi)
        finally:
            self.color_min.blockSignals(False)
            self.color_max.blockSignals(False)
            self._color_updating = False
        self._apply_colormap()

    def _on_color_limit_changed(self, _value: float | None = None) -> None:
        if self._color_updating or self.image is None:
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
        self._apply_colormap()

    def _on_profile_moved(self, pos) -> None:
        if self.image is not None and self.plot.sceneBoundingRect().contains(pos):
            self.tip_y.hide()
            self.tip_x.hide()
            point = self.plot.vb.mapSceneToView(pos)
            x = int(np.floor(point.x()))
            y = int(np.floor(point.y()))
            height, width = self.image.shape
            if not (0 <= x < width and 0 <= y < height):
                self.hover.setText("游標：影像外")
                return
            shown = self._cursor_image()
            self.hover.setText(
                f"游標：x={x}  y={y}  {self._cursor_caption()}：{format_sci(float(shown[y, x]))}"
            )
            return
        if self._x_values is not None and self.plot_x.sceneBoundingRect().contains(pos):
            self.tip_y.hide()
            point = self.plot_x.vb.mapSceneToView(pos)
            index = int(np.floor(point.x()))
            if not (0 <= index < len(self._x_values)):
                self.tip_x.hide()
                self.hover.setText("游標：—")
                return
            value = float(self._x_values[index])
            label = "log10 強度" if self.log_check.isChecked() else "強度"
            self.tip_x.setText(f"x = {index} px\n{label}：{format_sci(value)}")
            self.tip_x.show()
            self._pin_profile_tip("x")
            self.hover.setText(f"游標：x={index} px  {label}：{format_sci(value)}")
            return
        if self.profile_values is None or not self.plot_y.sceneBoundingRect().contains(pos):
            self.tip_y.hide()
            self.tip_x.hide()
            self.hover.setText("游標：—")
            return
        self.tip_x.hide()
        point = self.plot_y.vb.mapSceneToView(pos)
        index = int(np.floor(point.y()))
        if not (0 <= index < len(self.profile_values)):
            self.tip_y.hide()
            self.hover.setText("游標：—")
            return
        value = float(self.profile_values[index])
        self.tip_y.setText(f"y = {index} px\n{self._strip_caption()}：{format_sci(value)}")
        self.tip_y.show()
        self._pin_profile_tip("y")
        self.hover.setText(f"游標：y={index} px  {self._strip_caption()}：{format_sci(value)}")

    def _pin_profile_tip(self, which: str = "y") -> None:
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

    def _cursor_image(self) -> np.ndarray:
        key = (
            id(self.image),
            self._abs_enabled(),
            self.log_check.isChecked(),
        )
        cached = getattr(self, "_cursor_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        shown = self._display_values(self._view_image())
        self._cursor_cache = (key, shown)
        return shown

    def _strip_caption(self) -> str:
        if self._line_sum_mode():
            if self.log_check.isChecked() and self._abs_enabled():
                return "log10 絕對值斜向加總"
            if self.log_check.isChecked():
                return "log10 斜向加總"
            if self._abs_enabled():
                return "絕對值斜向加總"
            return "斜向加總"
        if self._line_mode():
            if self.log_check.isChecked() and self._abs_enabled():
                return "log10 絕對值"
            if self.log_check.isChecked():
                return "log10 強度"
            if self._abs_enabled():
                return "絕對值"
            return "強度"
        if self._y_is_sum():
            if self.log_check.isChecked() and self._abs_enabled():
                return "log10 絕對值累加強度"
            if self.log_check.isChecked():
                return "log10 累加強度"
            if self._abs_enabled():
                return "絕對值累加強度"
            return "累加強度"
        if self.log_check.isChecked() and self._abs_enabled():
            return "log10 絕對值"
        if self.log_check.isChecked():
            return "log10 強度"
        if self._abs_enabled():
            return "絕對值"
        return "強度"

    def _cursor_caption(self) -> str:
        if self.log_check.isChecked() and self._abs_enabled():
            return "log10 絕對值"
        if self.log_check.isChecked():
            return "log10 強度"
        if self._abs_enabled():
            return "絕對值"
        return "強度"

    def _on_scene_clicked(self, event) -> None:
        if self.image is None:
            return
        pos = event.scenePos()
        if not event.double():
            if event.button() not in (Qt.LeftButton, Qt.NoButton):
                return
            hit = self._profile_axis_hit(pos)
            if hit is not None:
                which, end = hit
                self._prompt_profile_limit(which, end)
                return
            if self.plot.sceneBoundingRect().contains(pos):
                point = self.plot.vb.mapSceneToView(pos)
                x = int(np.floor(point.x()))
                y = int(np.floor(point.y()))
                height, width = self.image.shape
                if 0 <= x < width and 0 <= y < height:
                    self.cross_xy = (x, y)
                    self._manual_profile = False
                    self._manual_x = False
                    self._draw_profile()
            return
        if self.plot_y.sceneBoundingRect().contains(pos):
            self._manual_profile = False
            self._draw_profile()
            return
        if self.plot_x.sceneBoundingRect().contains(pos):
            self._manual_x = False
            self._draw_profile()
            return
        if self.hist.sceneBoundingRect().contains(pos):
            shown = self._display_values(self._view_image())
            self._set_color_levels(*self._level_span(shown))
            return
        if self.plot.sceneBoundingRect().contains(pos):
            height, width = self.image.shape
            self.plot.setXRange(0, width, padding=0)
            self.plot.setYRange(0, height, padding=0)

    def _profile_axis_hit(self, pos) -> tuple[str, str] | None:
        for which, plot, axis_name in (
            ("y", self.plot_y, "bottom"),
            ("x", self.plot_x, "left"),
        ):
            view_rect = plot.vb.sceneBoundingRect()
            axis_rect = plot.getAxis(axis_name).mapRectToScene(plot.getAxis(axis_name).boundingRect())
            if which == "y":
                strip = QRectF(
                    view_rect.left() - 24,
                    view_rect.bottom() - 2,
                    view_rect.width() + 48,
                    max(12.0, axis_rect.bottom() - view_rect.bottom() + 8),
                )
                if not strip.contains(pos):
                    continue
                data_range = plot.viewRange()[0]
                anchor = plot.viewRange()[1][0]
                min_scene = plot.vb.mapViewToScene(pg.Point(data_range[0], anchor))
                max_scene = plot.vb.mapViewToScene(pg.Point(data_range[1], anchor))
                d_min = abs(pos.x() - min_scene.x())
                d_max = abs(pos.x() - max_scene.x())
            else:
                strip = QRectF(
                    view_rect.left() - max(12.0, view_rect.left() - axis_rect.left() + 8),
                    view_rect.top() - 24,
                    max(12.0, view_rect.left() - axis_rect.left() + 8),
                    view_rect.height() + 48,
                )
                if not strip.contains(pos):
                    continue
                data_range = plot.viewRange()[1]
                anchor = plot.viewRange()[0][0]
                min_scene = plot.vb.mapViewToScene(pg.Point(anchor, data_range[0]))
                max_scene = plot.vb.mapViewToScene(pg.Point(anchor, data_range[1]))
                d_min = abs(pos.y() - min_scene.y())
                d_max = abs(pos.y() - max_scene.y())
            if min(d_min, d_max) > 36:
                continue
            return which, ("min" if d_min <= d_max else "max")
        return None

    def _prompt_profile_limit(self, which: str, end: str) -> None:
        plot = self.plot_x if which == "x" else self.plot_y
        axis = 1 if which == "x" else 0
        lo, hi = plot.viewRange()[axis]
        current = float(lo if end == "min" else hi)
        title = "下限" if end == "min" else "上限"
        label = "X 剖面強度" if which == "x" else self._strip_caption()
        value, ok = ask_scientific(self, title, f"{label}{title}", current)
        if not ok:
            return
        if end == "min":
            lo = float(value)
        else:
            hi = float(value)
        if lo >= hi:
            return
        if which == "x":
            self._manual_x = True
            plot.setYRange(lo, hi, padding=0)
        else:
            self._manual_profile = True
            plot.setXRange(lo, hi, padding=0)


class SpotWindow(QMainWindow):
    """Standalone window wrapping :class:`SpotPanel`."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("光斑分析")
        self.resize(1280, 860)
        self.panel = SpotPanel(embed=False)
        root = QWidget()
        root.setObjectName("appRoot")
        root.setStyleSheet("QWidget#appRoot { background: #E8ECF0; }")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panel)
        self.setCentralWidget(root)

    def set_source_folder(self, root: Path | None) -> None:
        self.panel.set_source_folder(root)

    def set_image(self, image: np.ndarray) -> None:
        self.panel.set_image(image)

    def set_batch(self, images: list[np.ndarray], labels: list[str]) -> None:
        self.panel.set_batch(images, labels)

    def apply_color_display(self, stops: list[float], levels: tuple[float, float]) -> None:
        self.panel.apply_color_display(stops, levels)
