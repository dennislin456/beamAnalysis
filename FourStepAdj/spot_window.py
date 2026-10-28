"""Analysis window: place an ROI on the merged image and find both spots."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dual_spot import locate_dual_spots
from image_canvas import (
    COLOR_CEIL,
    COLOR_FLOOR,
    OptionalNumberBox,
    add_gaussian_controls,
    apply_abs_then_color_scale,
    build_bw_clip_colormap,
    display_threshold,
    fix_readout_height,
    function_frame,
    linear_bound,
    log10_bound,
)
from sci_format import ask_scientific, format_sci

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
    QFrame#preview {
        background: #E8ECF0;
        border: 1px solid #e1e5ea;
        border-radius: 8px;
    }
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


class SpotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光斑分析")
        self.resize(1280, 860)
        self.image: np.ndarray | None = None
        self.spots: dict | None = None
        self._roi_sync = False
        self._roi_user_set = False
        self._roi_x = 0
        self._color_updating = False
        self._color_bounds_in_log = False
        self._manual_profile = False
        self._manual_smooth = False
        self._smooth_restoring = False
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
        left.setMinimumWidth(324)
        left.setStyleSheet(_panel_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)
        hint = QLabel(
            "條帶的寬度以上光斑 X 為中心，位置不能手動調，Y 固定為整張高度。\n"
            "上光斑是最強訊號的門檻質心。條帶找出波谷後，以波谷中心為界分開上下光斑。\n"
            "左側平滑圖是整張合成影像的圓形平均。下光斑可切換形狀中心或強度質心。\n"
            "log、abs 和平滑只改顯示。雙擊條帶圖可復原強度軸。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        left_layout.addWidget(_section_label("條帶"))
        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.roi_w = self._make_roi_box()
        self.roi_w.setRange(1, 100000)
        self.roi_w.setValue(160)
        self.roi_w.setToolTip("條帶寬度。左右以上光斑的 X 為中心，不能另調位置。")
        self.roi_w.valueChanged.connect(self._on_roi_boxes_changed)
        form.addRow("寬", self.roi_w)
        left_layout.addLayout(form)

        left_layout.addWidget(_section_label("上光斑"))
        upper = QFormLayout()
        upper.setHorizontalSpacing(10)
        upper.setVerticalSpacing(8)
        self.thresh = QSpinBox()
        self.thresh.setRange(5, 95)
        self.thresh.setValue(50)
        self.thresh.setSuffix(" %")
        self.thresh.setKeyboardTracking(False)
        self.thresh.setToolTip("相對條帶內最強訊號。40% 就是留下不低於峰值 40% 的那一塊核心。")
        self.thresh.valueChanged.connect(self._on_roi_boxes_changed)
        self.show_thresh = QCheckBox("顯示門檻")
        self.show_thresh.setChecked(True)
        self.show_thresh.setToolTip("在熱圖上以半透明白色顯示上光斑的門檻範圍。")
        self.show_thresh.toggled.connect(self._draw_threshold)
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(8)
        thresh_row.addWidget(self.thresh, 1)
        thresh_row.addWidget(self.show_thresh)
        thresh_wrap = QWidget()
        thresh_wrap.setLayout(thresh_row)
        upper.addRow("門檻", thresh_wrap)
        left_layout.addLayout(upper)

        left_layout.addWidget(_section_label("下光斑"))
        lower = QFormLayout()
        lower.setHorizontalSpacing(10)
        lower.setVerticalSpacing(8)
        self.smooth_win = QSpinBox()
        self.smooth_win.setRange(1, 301)
        self.smooth_win.setSingleStep(2)
        self.smooth_win.setValue(7)
        self.smooth_win.setKeyboardTracking(False)
        self.smooth_win.setToolTip(
            "先圈出波谷中心以下的光斑，再用這個直徑做圓形平均。偶數會自動加 1。窗變大不會把範圍往下抹。"
        )
        self.smooth_win.valueChanged.connect(self._on_roi_boxes_changed)
        lower.addRow("平滑窗", self.smooth_win)
        self.center_mode = QComboBox()
        self.center_mode.addItem("形狀中心", "shape")
        self.center_mode.addItem("強度質心", "intensity")
        self.center_mode.setToolTip("形狀中心：範圍內每個像素一樣重。強度質心：亮的像素權重高，會偏向上面較亮的一側。")
        self.center_mode.currentIndexChanged.connect(lambda _index: self._refresh_spots())
        lower.addRow("演算法", self.center_mode)
        left_layout.addLayout(lower)

        left_layout.addWidget(_section_label("平滑圖"))
        preview = QFrame()
        preview.setObjectName("preview")
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        self.smooth_view = pg.GraphicsLayoutWidget()
        self.smooth_view.setBackground(CANVAS)
        self.smooth_view.setFixedHeight(340)
        self.smooth_view.setToolTip(
            "整張合成影像的圓形平均。滾輪縮放，雙擊復原。色階跟合成影像一起變。"
        )
        self.smooth_plot = self.smooth_view.addPlot()
        self.smooth_plot.setAspectLocked(True)
        self.smooth_plot.invertY(True)
        self.smooth_plot.showGrid(x=True, y=True, alpha=0.35)
        self.smooth_plot.setMenuEnabled(False)
        self.smooth_plot.hideButtons()
        self.smooth_plot.setLabel("bottom", "X")
        self.smooth_plot.setLabel("left", "Y")
        self.smooth_image_item = pg.ImageItem()
        self.smooth_plot.addItem(self.smooth_image_item)
        self.smooth_upper = self.smooth_plot.plot(
            pen=None, symbol="+", symbolSize=12, symbolPen=pg.mkPen("#f7f7f7", width=2)
        )
        self.smooth_marker = self.smooth_plot.plot(
            pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen("#f7f7f7", width=2)
        )
        preview_layout.addWidget(self.smooth_view)
        self.smooth_plot.sigRangeChanged.connect(self._on_smooth_range_changed)
        self.smooth_view.scene().sigMouseClicked.connect(self._on_smooth_clicked)
        left_layout.addWidget(preview)

        left_layout.addWidget(_section_label("結果"))
        self.spot_label = QLabel("尚未載入合成影像")
        self.spot_label.setObjectName("card")
        self.spot_label.setWordWrap(True)
        self.spot_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left_layout.addWidget(self.spot_label)
        left_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(left)
        scroll.setFixedWidth(380)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        layout.addWidget(scroll)

        pg.setConfigOption("background", CANVAS)
        pg.setConfigOption("foreground", "#333333")
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground(CANVAS)

        self.plot_y = self.win.addPlot(row=0, col=0, title="ROI 條帶")
        self.plot_y.setLabel("left", "Y (px)")
        self.plot_y.setLabel("bottom", "條帶加總")
        self.plot_y.getAxis("bottom").setToolTip("點最小值輸入下限，點最大值輸入上限")
        self.plot_y.showGrid(x=True, y=True, alpha=0.35)
        self.plot_y.invertY(True)
        self.plot_y.enableAutoRange(x=False, y=False)
        self.curve_y = self.plot_y.plot(pen=pg.mkPen("#444444", width=1.5))
        self.profile_values: np.ndarray | None = None
        self.tip_y = pg.TextItem(
            color="#333333",
            anchor=(0, 1),
            fill=pg.mkBrush(255, 255, 255, 230),
            border=pg.mkPen("#c5c5c5"),
        )
        self.tip_y.setZValue(5)
        self.tip_y.hide()
        self.plot_y.addItem(self.tip_y)
        self.marker_upper = self.plot_y.plot(
            pen=None, symbol="+", symbolSize=12, symbolPen=pg.mkPen("#222222", width=2)
        )
        self.marker_lower = self.plot_y.plot(
            pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen("#222222", width=2)
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
        spot_pen = pg.mkPen("#f7f7f7", width=2)
        self.spot_upper = self.plot.plot(pen=None, symbol="+", symbolSize=16, symbolPen=spot_pen)
        self.spot_lower = self.plot.plot(pen=None, symbol="x", symbolSize=14, symbolPen=spot_pen)

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        self._base_cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(colors)), colors)
        self.smooth_image_item.setColorMap(self._base_cmap)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(self._base_cmap)
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

        grid = self.win.ci.layout
        grid.setColumnStretchFactor(0, 2)
        grid.setColumnStretchFactor(1, 5)
        grid.setColumnStretchFactor(2, 0)

        host = QWidget()
        host.setAutoFillBackground(True)
        palette = host.palette()
        palette.setColor(host.backgroundRole(), QColor(CANVAS))
        host.setPalette(palette)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 8, 0)
        host_layout.setSpacing(6)
        self.log_check = QCheckBox("log")
        self.log_check.setToolTip("只改熱圖、平滑圖和條帶圖的顯示。找上下光斑仍用線性強度。")
        self.log_check.toggled.connect(self._on_log_toggled)
        self.abs_check = QCheckBox("abs")
        self.abs_check.setToolTip(
            "熱圖和平滑圖先取絕對值，再套 color scale。條帶圖只取絕對值。"
            "找上下光斑仍用原本的線性強度。"
        )
        self.abs_check.toggled.connect(self._on_abs_toggled)
        self.smooth_check = QCheckBox("平滑")
        self.smooth_check.setToolTip(
            "熱圖和圓形平均預覽改為 Gaussian blur。水平、垂直是核寬，sigma 兩邊共用。某一邊為 1 時該方向不平滑。"
            "有勾 abs 時先取絕對值。有勾 color scale 時，先做這個平滑，再套門檻，最後才是 log。"
            "找上下光斑和條帶圖仍用原本的線性強度，也不套這個平滑。"
        )
        self.smooth_check.toggled.connect(self._on_smooth_toggled)
        self.color_scale_check = QCheckBox("color scale")
        self.color_scale_check.setToolTip(
            "勾選後，小於等於左邊數值的像素顯示為 0，色條底部是黑色；"
            "大於等於右邊數值的像素顯示成該數值，色條頂部是白色。"
            "同時勾選 abs 時先取絕對值。有勾平滑時先做 Gaussian blur，再套這兩個門檻，最後才是 log。"
            "預設 256 和 4096。留空的那一側不採用。勾選 log 時，有填的門檻改顯示 log10，取消後換回 10 的次方；熱圖仍用線性門檻再取 log。"
            "找上下光斑仍用沒裁過的線性強度。"
        )
        self.color_scale_check.toggled.connect(self._on_color_scale_toggled)
        self._color_floor_tip = "小於等於這個值的像素為 0，色條底部黑色。留空代表不採用這個下限。"
        self._color_ceil_tip = "大於等於這個值的像素等於這個值，色條頂部白色。留空代表不採用這個上限。"
        self._color_floor_log_tip = (
            "log10 下限。畫面用 10 的這個次方裁線性值，再取 log。取消 log 會換回線性下限。"
            "留空代表不採用。小於等於 0 的線性下限會改成 0。"
        )
        self._color_ceil_log_tip = (
            "log10 上限。畫面用 10 的這個次方裁線性值，再取 log。取消 log 會換回線性上限。"
            "留空代表不採用。小於等於 0 的線性上限會改成 0。"
        )
        self.color_floor = OptionalNumberBox(COLOR_FLOOR, self._color_floor_tip)
        self.color_ceil = OptionalNumberBox(COLOR_CEIL, self._color_ceil_tip)
        self.color_floor.setEnabled(False)
        self.color_ceil.setEnabled(False)
        self.color_floor.numberChanged.connect(self._on_color_scale_value)
        self.color_ceil.numberChanged.connect(self._on_color_scale_value)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(8, 4, 0, 0)
        color_row.setSpacing(8)
        log_frame, log_row = function_frame()
        log_row.addWidget(self.log_check)
        abs_frame, abs_row = function_frame()
        abs_row.addWidget(self.abs_check)
        smooth_frame, smooth_row = function_frame()
        smooth_row.addWidget(self.smooth_check)
        color_row.addWidget(log_frame)
        color_row.addWidget(abs_frame)
        color_row.addWidget(smooth_frame)
        self.smooth_kx, self.smooth_ky, self.smooth_sigma = add_gaussian_controls(smooth_row)
        for box in (self.smooth_kx, self.smooth_ky, self.smooth_sigma):
            box.valueChanged.connect(self._on_smooth_n_changed)
            box.setToolTip(box.toolTip() + "和左側的平滑窗不同。")
        scale_frame, scale_row = function_frame()
        scale_row.addWidget(self.color_scale_check)
        scale_row.addWidget(self.color_floor)
        scale_row.addWidget(self.color_ceil)
        color_row.addWidget(scale_frame)
        color_row.addStretch(1)
        color_row.addWidget(self.color_max)
        color_row.addWidget(self.color_min)
        host_layout.addLayout(color_row)
        host_layout.addWidget(self.win, 1)
        self.hover = QLabel("游標：—")
        self.hover.setStyleSheet("color: #333333; padding: 0 8px 4px 8px;")
        fix_readout_height(self.hover, 2)
        host_layout.addWidget(self.hover, 0, Qt.AlignLeft)
        layout.addWidget(host, 1)
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.plot.scene().sigMouseMoved.connect(self._on_profile_moved)

    def set_image(self, image: np.ndarray) -> None:
        self.image = np.asarray(image, dtype=np.float64)
        height, width = self.image.shape
        self.roi_w.setMaximum(width)
        if not self._roi_user_set or self.roi_w.value() > width:
            self._roi_sync = True
            try:
                self.roi_w.setValue(min(160, width))
            finally:
                self._roi_sync = False
            self._roi_user_set = False
        self.roi.maxBounds = QRectF(0, 0, width, height)
        self._manual_smooth = False
        self._sync_roi_rect()
        self.roi.show()
        self._show_heatmap()
        self._refresh_spots()
        self.plot.setXRange(0, width, padding=0)
        self.plot.setYRange(0, height, padding=0)

    def _color_scale_enabled(self) -> bool:
        return bool(self.color_scale_check.isChecked())

    def _abs_enabled(self) -> bool:
        return bool(self.abs_check.isChecked())

    def _color_bounds(self) -> tuple[float | None, float | None]:
        if not self._color_scale_enabled():
            return None, None
        lo = self.color_floor.number()
        hi = self.color_ceil.number()
        if self._color_bounds_in_log:
            lo = linear_bound(lo)
            hi = linear_bound(hi)
        return lo, hi

    def _smooth_params(self) -> tuple[int, int, float]:
        if not self.smooth_check.isChecked():
            return 1, 1, 0.0
        return (
            int(self.smooth_kx.value()),
            int(self.smooth_ky.value()),
            float(self.smooth_sigma.value()),
        )

    def _view_image(self) -> np.ndarray:
        clip_on = self._color_scale_enabled()
        lo, hi = self._color_bounds()
        return apply_abs_then_color_scale(
            self.image,
            self._abs_enabled(),
            lo,
            hi,
            clip_on,
            self._smooth_params(),
        )

    def _display_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if not self.log_check.isChecked():
            return data
        return np.log10(np.maximum(data, 1.0))

    def _display_bounds(self) -> tuple[float | None, float | None]:
        if not self._color_scale_enabled():
            return None, None
        lo, hi = self._color_bounds()
        log_on = self.log_check.isChecked()
        lo_d = display_threshold(lo, log_on) if lo is not None else None
        hi_d = display_threshold(hi, log_on) if hi is not None else None
        if lo_d is not None and hi_d is not None and hi_d < lo_d:
            lo_d, hi_d = hi_d, lo_d
        return lo_d, hi_d

    def _scale_limits(self, shown: np.ndarray | None = None) -> tuple[float, float]:
        lo_b, hi_b = self._display_bounds()
        if shown is not None and np.size(shown):
            data_lo = float(np.min(shown))
            data_hi = float(np.max(shown))
        else:
            data_lo, data_hi = 0.0, 1.0
        lo = data_lo if lo_b is None else lo_b
        hi = data_hi if hi_b is None else hi_b
        if hi < lo:
            lo, hi = hi, lo
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _level_span(self, shown: np.ndarray) -> tuple[float, float]:
        if self._color_scale_enabled():
            return self._scale_limits(shown)
        lo = float(np.min(shown))
        hi = float(np.max(shown))
        if lo >= hi:
            hi = lo + 1.0
        return lo, hi

    def _apply_colormap(self) -> None:
        try:
            vmin, vmax = self.hist.getLevels()
        except Exception:
            return
        vmin, vmax = float(vmin), float(vmax)
        if self._color_scale_enabled():
            lo_b, hi_b = self._display_bounds()
            span = max(vmax - vmin, 1e-12)
            cmap = build_bw_clip_colormap(
                self._base_cmap,
                vmin,
                vmax,
                vmin + span * 1e-9 if lo_b is not None else None,
                vmax - span * 1e-9 if hi_b is not None else None,
            )
        else:
            cmap = self._base_cmap
        self.hist.gradient.setColorMap(cmap)
        self.smooth_image_item.setColorMap(cmap)

    def _on_color_scale_toggled(self, checked: bool) -> None:
        self.color_floor.setEnabled(checked)
        self.color_ceil.setEnabled(checked)
        self._show_heatmap()
        self._draw_smooth()

    def _on_color_scale_value(self) -> None:
        if not self._color_scale_enabled():
            return
        self._show_heatmap()
        self._draw_smooth()

    def _show_heatmap(self) -> None:
        if self.image is None:
            return
        shown = self._display_values(self._view_image())
        self.image_item.setImage(shown.T, autoLevels=False)
        self._set_color_levels(*self._level_span(shown))

    def _make_roi_box(self) -> QSpinBox:
        box = QSpinBox()
        box.setRange(0, 100000)
        box.setKeyboardTracking(False)
        return box

    def _make_color_box(self, tooltip: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e15, 1e15)
        box.setDecimals(2)
        box.setKeyboardTracking(False)
        box.setToolTip(tooltip)
        return box

    def _strip_origin(self, center: float, width: int) -> tuple[int, int]:
        image_width = 1 if self.image is None else int(self.image.shape[1])
        roi_w = int(np.clip(width, 1, image_width))
        roi_x = int(round(float(center) - roi_w / 2.0))
        roi_x = int(np.clip(roi_x, 0, image_width - roi_w))
        return roi_x, roi_w

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

    def _on_roi_dragged(self) -> None:
        if self._roi_sync or self.image is None:
            return
        self._sync_roi_rect()

    def _on_log_toggled(self, checked: bool) -> None:
        self._sync_color_bound_space(checked)
        self._manual_profile = False
        self._show_heatmap()
        self._refresh_spots()

    def _sync_color_bound_space(self, log_on: bool) -> None:
        if bool(log_on) == self._color_bounds_in_log:
            return
        convert = log10_bound if log_on else linear_bound
        self.color_floor.set_number(convert(self.color_floor.number()), emit=False)
        self.color_ceil.set_number(convert(self.color_ceil.number()), emit=False)
        self.color_floor.setToolTip(self._color_floor_log_tip if log_on else self._color_floor_tip)
        self.color_ceil.setToolTip(self._color_ceil_log_tip if log_on else self._color_ceil_tip)
        self._color_bounds_in_log = bool(log_on)

    def _on_abs_toggled(self, _checked: bool) -> None:
        self._manual_profile = False
        self._manual_smooth = False
        self._show_heatmap()
        self._draw_profile()
        self._draw_smooth()

    def _on_smooth_toggled(self, checked: bool) -> None:
        for box in (self.smooth_kx, self.smooth_ky, self.smooth_sigma):
            box.setEnabled(checked)
        self._manual_smooth = False
        self._show_heatmap()
        self._draw_smooth()

    def _on_smooth_n_changed(self, _value: int) -> None:
        if not self.smooth_check.isChecked():
            return
        self._manual_smooth = False
        self._show_heatmap()
        self._draw_smooth()

    def _refresh_spots(self) -> None:
        if self.image is None:
            self.spots = None
            self._draw_markers()
            self._draw_smooth()
            return
        height, _image_width = self.image.shape
        _peak_y, peak_x = np.unravel_index(int(np.argmax(self.image)), self.image.shape)
        origin, roi_w = self._strip_origin(float(peak_x), self.roi_w.value())
        spots = None
        seen: set[int] = set()
        for _ in range(4):
            if origin in seen:
                break
            seen.add(origin)
            spots = locate_dual_spots(
                self.image,
                origin,
                0,
                roi_w,
                int(height),
                smooth_win=self.smooth_win.value(),
                use_log=self.log_check.isChecked(),
                thresh_percent=float(self.thresh.value()),
                center_mode=str(self.center_mode.currentData()),
            )
            self._roi_x = origin
            upper = spots.get("upper")
            if upper is None:
                break
            nxt, _roi_w = self._strip_origin(float(upper[0]), roi_w)
            if nxt == origin:
                break
            origin = nxt
        self.spots = spots
        self._sync_roi_rect()
        self._draw_markers()
        self._draw_profile()
        self._draw_smooth()

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
        if self.image is None:
            self.spot_label.setText("尚未載入合成影像")
            return
        upper_text = "未找到" if upper is None else f"({upper[0]:.1f}, {upper[1]:.1f})"
        lower_text = "雜訊中未找到" if lower is None else f"({lower[0]:.1f}, {lower[1]:.1f})"
        self.spot_label.setText(
            f"上光斑 + {upper_text}\n下光斑 × {lower_text}\n{self._distance_text(upper, lower)}"
        )
        self._draw_threshold()

    def _distance_text(self, upper, lower) -> str:
        if upper is None or lower is None:
            return "距離 -- px\n-- μm"
        dist = float(np.hypot(float(lower[0]) - float(upper[0]), float(lower[1]) - float(upper[1])))
        return f"距離 {dist:.2f} px\n{dist * 3.45:.2f} μm"

    def _draw_smooth(self) -> None:
        spots = self.spots or {}
        image = spots.get("smooth_image")
        if image is None or self.image is None or not np.any(image):
            self.smooth_image_item.clear()
            self.smooth_upper.setData([], [])
            self.smooth_marker.setData([], [])
            return
        clip_on = self._color_scale_enabled()
        lo, hi = self._color_bounds()
        preview = apply_abs_then_color_scale(
            image, self._abs_enabled(), lo, hi, clip_on, self._smooth_params()
        )
        shown = self._display_values(preview)
        self.smooth_image_item.setImage(shown.T, autoLevels=False)
        levels = self.hist.getLevels()
        self.smooth_image_item.setLevels([float(levels[0]), float(levels[1])])
        upper = spots.get("upper")
        lower = spots.get("lower")
        if upper is None:
            self.smooth_upper.setData([], [])
        else:
            self.smooth_upper.setData([upper[0]], [upper[1]])
        if lower is None:
            self.smooth_marker.setData([], [])
        else:
            self.smooth_marker.setData([lower[0]], [lower[1]])
        if not self._manual_smooth:
            self._restore_smooth_view()

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

    def _draw_profile(self) -> None:
        if self.image is None:
            return
        height = self.image.shape[0]
        positions = np.arange(height, dtype=float) + 0.5
        values = np.zeros(height, dtype=float)
        spots = self.spots or {}
        smooth = spots.get("smooth")
        bounds = spots.get("bounds")
        if smooth is not None and bounds is not None and len(smooth):
            y0 = int(bounds[1])
            count = min(len(smooth), height - y0)
            values[y0 : y0 + count] = np.asarray(smooth[:count], dtype=float)
        shown = np.abs(values) if self._abs_enabled() else values
        self.profile_values = shown
        self.curve_y.setData(shown, positions)
        self.plot_y.setLabel("bottom", self._strip_caption())
        upper = spots.get("upper")
        lower = spots.get("lower")
        if upper is None:
            self.marker_upper.setData([], [])
        else:
            index = int(np.clip(round(upper[1]), 0, height - 1))
            self.marker_upper.setData([float(shown[index])], [upper[1]])
        if lower is None:
            self.marker_lower.setData([], [])
        else:
            index = int(np.clip(round(lower[1]), 0, height - 1))
            self.marker_lower.setData([float(shown[index])], [lower[1]])
        lo = float(np.min(shown))
        hi = float(np.max(shown))
        if lo >= hi:
            hi = lo + 1.0
        if not self._manual_profile:
            pad = (hi - lo) * 0.05
            self.plot_y.setXRange(lo - pad, hi + pad, padding=0)
            self.plot_y.setYRange(0, height, padding=0)

    def _set_color_levels(self, lo: float, hi: float) -> None:
        if lo >= hi:
            hi = lo + 1.0
        self._color_updating = True
        try:
            self.hist.setHistogramRange(lo, hi, padding=0)
            self.hist.setLevels(lo, hi)
            self.smooth_image_item.setLevels([lo, hi])
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
            self.smooth_image_item.setLevels([float(lo), float(hi)])
        finally:
            self.color_min.blockSignals(False)
            self.color_max.blockSignals(False)
            self._color_updating = False
        self._apply_colormap()

    def _restore_smooth_view(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape
        self._smooth_restoring = True
        try:
            self.smooth_plot.setXRange(0, width, padding=0)
            self.smooth_plot.setYRange(0, height, padding=0)
        finally:
            self._smooth_restoring = False

    def _on_smooth_range_changed(self, *_args) -> None:
        if not self._smooth_restoring:
            self._manual_smooth = True

    def _on_smooth_clicked(self, event) -> None:
        if self.image is None or not event.double():
            return
        if not self.smooth_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        self._manual_smooth = False
        self._restore_smooth_view()

    def _on_profile_moved(self, pos) -> None:
        if self.image is not None and self.plot.sceneBoundingRect().contains(pos):
            self.tip_y.hide()
            point = self.plot.vb.mapSceneToView(pos)
            x = int(np.floor(point.x()))
            y = int(np.floor(point.y()))
            height, width = self.image.shape
            if not (0 <= x < width and 0 <= y < height):
                self.hover.setText("游標：影像外")
                return
            shown = self._cursor_image()
            self.hover.setText(f"游標：x={x}, y={y}\n{self._cursor_caption()}：{format_sci(float(shown[y, x]))}")
            return
        if self.profile_values is None or not self.plot_y.sceneBoundingRect().contains(pos):
            self.tip_y.hide()
            self.hover.setText("游標：—")
            return
        point = self.plot_y.vb.mapSceneToView(pos)
        index = int(np.floor(point.y()))
        if not (0 <= index < len(self.profile_values)):
            self.tip_y.hide()
            self.hover.setText("游標：—")
            return
        value = float(self.profile_values[index])
        span = self.plot_y.viewRange()[0]
        anchor_x = 0 if point.x() <= (span[0] + span[1]) / 2.0 else 1
        self.tip_y.setAnchor((anchor_x, 1))
        self.tip_y.setText(f"y = {index} px\n{format_sci(value)}")
        self.tip_y.setPos(value, index + 0.5)
        self.tip_y.show()
        self.hover.setText(f"游標：y={index} px\n{self._strip_caption()}：{format_sci(value)}")

    def _cursor_image(self) -> np.ndarray:
        key = (
            id(self.image),
            self._abs_enabled(),
            self._smooth_params(),
            self._color_bounds() if self._color_scale_enabled() else (None, None),
            self.log_check.isChecked(),
        )
        cached = getattr(self, "_cursor_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        shown = self._display_values(self._view_image())
        self._cursor_cache = (key, shown)
        return shown

    def _strip_caption(self) -> str:
        if self.log_check.isChecked() and self._abs_enabled():
            return "log10 絕對值條帶加總"
        if self.log_check.isChecked():
            return "log10 條帶加總"
        if self._abs_enabled():
            return "絕對值條帶加總"
        return "條帶加總"

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
                self._prompt_profile_limit(hit)
            return
        if self.plot_y.sceneBoundingRect().contains(pos):
            self._manual_profile = False
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

    def _profile_axis_hit(self, pos) -> str | None:
        view_rect = self.plot_y.vb.sceneBoundingRect()
        axis_rect = self.plot_y.getAxis("bottom").mapRectToScene(
            self.plot_y.getAxis("bottom").boundingRect()
        )
        strip = QRectF(
            view_rect.left() - 24,
            view_rect.bottom() - 2,
            view_rect.width() + 48,
            max(12.0, axis_rect.bottom() - view_rect.bottom() + 8),
        )
        if not strip.contains(pos):
            return None
        data_range = self.plot_y.viewRange()[0]
        anchor = self.plot_y.viewRange()[1][0]
        min_scene = self.plot_y.vb.mapViewToScene(pg.Point(data_range[0], anchor))
        max_scene = self.plot_y.vb.mapViewToScene(pg.Point(data_range[1], anchor))
        d_min = abs(pos.x() - min_scene.x())
        d_max = abs(pos.x() - max_scene.x())
        if min(d_min, d_max) > 36:
            return None
        return "min" if d_min <= d_max else "max"

    def _prompt_profile_limit(self, end: str) -> None:
        lo, hi = self.plot_y.viewRange()[0]
        current = float(lo if end == "min" else hi)
        title = "下限" if end == "min" else "上限"
        value, ok = ask_scientific(self, title, f"條帶加總{title}", current)
        if not ok:
            return
        if end == "min":
            lo = float(value)
        else:
            hi = float(value)
        if lo >= hi:
            return
        self._manual_profile = True
        self.plot_y.setXRange(lo, hi, padding=0)
