"""Basler 短／長曝光雙檔測試分頁。

- 直接選兩個檔案（不選資料夾／位置）
- 左右兩張相同版面的 heatmap（InteractiveHeatmapPanel）
- 各自獨立 log／color scale／平滑
- 抓光斑：短曝光定位 → 對應標到長曝光；長曝光可設範圍抓第二光斑
  （既有質心／門檻；另可試用軸向一維投影＋微分法，條帶預設 200×512）
- 拼貼視窗：按鍵開啟獨立視窗；長曝光搜尋範圍區用長曝光、其餘用短曝光
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import uniform_filter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from batch_data_loader import load_numeric_matrix
from shared_components import (
    InteractiveHeatmapPanel,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    clip_roi_to_matrix,
    compute_auto_spot_center,
    find_dual_spots_axial_1d_projection,
)
from tab_batch_basler import (
    BaslerBatchTab,
    _build_bw_clip_colormap,
    _DEFAULT_BLACK_BELOW,
    _DEFAULT_WHITE_ABOVE,
)

_FILE_FILTER = (
    "資料檔 (*.npy *.csv *.xlsx *.xls);;"
    "NumPy (*.npy);;CSV (*.csv);;Excel (*.xlsx *.xls);;所有檔案 (*.*)"
)


class BaslerBatchDualTab(BaslerBatchTab):
    """Basler 短／長曝光測試版：雙檔、雙 heatmap、獨立抓光斑。"""

    def __init__(self, parent=None):
        self.batch_short_path = ""
        self.batch_long_path = ""
        self.batch_long_files = []
        self.batch_long_cache = {}
        self.batch_long_matrix = None
        self._short_cmap_guard = False
        self._long_cmap_guard = False
        self._collage_cmap_guard = False
        self.short_spot = None  # (cx, cy) 像素
        self.long_spot2 = None  # (cx, cy) 第二光斑
        self._short_cross_items = []
        self._long_cross_items = []
        self.batch_collage_matrix = None
        self._collage_roi_bounds = None  # (x0, y0, x1, y1) 長曝光區
        super().__init__(parent)
        self._adapt_ui_for_dual_exposure()

    def _adapt_ui_for_dual_exposure(self):
        self.setWindowTitle("Basler 短／長曝光")
        self.lbl_batch_status.setText("狀態: 等待匯入短／長曝光檔案")
        self.lbl_batch_pair_info.setText("尚未選擇短／長曝光檔案")

        for w in (
            getattr(self, "btn_location_config", None),
            getattr(self, "lbl_batch_selected_info", None),
            getattr(self, "btn_batch_run", None),
            getattr(self, "btn_batch_prev_left", None),
            getattr(self, "btn_batch_next_left", None),
            getattr(self, "lbl_batch_group_nav", None),
            getattr(self, "btn_batch_export", None),
            getattr(self, "btn_batch_pause_export", None),
            getattr(self, "btn_batch_resume_export", None),
            getattr(self, "btn_batch_clear_export", None),
            getattr(self, "btn_batch_view_m1", None),
            getattr(self, "btn_batch_view_m2", None),
            getattr(self, "btn_batch_view_cross", None),
            getattr(self, "btn_batch_contour_popup", None),
            getattr(self, "btn_batch_save_params", None),
        ):
            if w is not None:
                w.hide()

        self._hide_inherited_locate_ui()
        self._place_export_on_top_bar()

        self.btn_batch_m1_dir.show()
        self.lbl_batch_m1_info.show()
        self.btn_batch_m1_dir.setText("I. 選擇短曝光檔案")
        self.lbl_batch_m1_info.setText("未選擇短曝光檔案\n支援 .npy / .csv / .xlsx")
        self.btn_batch_m2_dir.setText("II. 選擇長曝光檔案")
        self.lbl_batch_m2_info.setText("未選擇長曝光檔案\n支援 .npy / .csv / .xlsx")

        try:
            self.btn_batch_m1_dir.clicked.disconnect()
        except TypeError:
            pass
        try:
            self.btn_batch_m2_dir.clicked.disconnect()
        except TypeError:
            pass
        self.btn_batch_m1_dir.clicked.connect(self.load_short_file)
        self.btn_batch_m2_dir.clicked.connect(self.load_long_file)

        self._remove_distance_table()
        self._build_dual_heatmap_layout()
        self._build_spot_controls()
        self._compact_left_panel()
        self._update_short_heatmap_title()
        self._update_long_heatmap_title()

    def _place_export_on_top_bar(self):
        """匯出按鈕放到「選擇儲存資料夾／路徑」右側（同一列）。"""
        self.btn_export_dual = QPushButton("匯出圖片／參數／光斑")
        self.btn_export_dual.setMinimumHeight(38)
        self.btn_export_dual.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #C2185B; border: none; border-radius: 5px; "
            "padding: 6px 14px; }"
            "QPushButton:hover { background-color: #D81B60; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        self.btn_export_dual.setEnabled(False)
        self.btn_export_dual.setToolTip(
            "匯出短／長／拼貼 Heatmap、JSON／CSV 參數、光斑座標與 ΔX／ΔY／總距。"
        )
        self.btn_export_dual.clicked.connect(self.export_dual_results)

        path_lbl = getattr(self, "lbl_batch_dir_path", None)
        if path_lbl is None:
            return

        def _find_layout_index(layout, target):
            if layout is None:
                return None
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is None:
                    continue
                if item.widget() is target:
                    return layout, i
                nested = item.layout()
                if nested is not None:
                    found = _find_layout_index(nested, target)
                    if found is not None:
                        return found
            return None

        parent = path_lbl.parentWidget()
        found = _find_layout_index(
            parent.layout() if parent is not None else None, path_lbl
        )
        if found is None:
            # fallback：跟選擇資料夾同一父層
            sel = getattr(self, "btn_batch_select_dir", None)
            if sel is not None and sel.parentWidget() is not None:
                found = _find_layout_index(sel.parentWidget().layout(), path_lbl)
        if found is None:
            return
        layout, idx = found
        layout.insertWidget(idx + 1, self.btn_export_dual)

    def _hide_inherited_locate_ui(self):
        """隱藏 M2／Basler 遺留定位、暫存參數、contour 等。"""
        attrs = [
            "chk_batch_valley_roi",
            "lbl_batch_valley_roi_x", "spin_batch_valley_roi_x",
            "lbl_batch_valley_roi_y", "spin_batch_valley_roi_y",
            "lbl_batch_valley_roi_w", "spin_batch_valley_roi_w",
            "lbl_batch_valley_roi_h", "spin_batch_valley_roi_h",
            "spin_expected_dist_min_um", "spin_expected_dist_max_um",
            "radio_batch_p2_auto_min",
            "radio_batch_p2_m2_thresh_geom",
            "radio_batch_p2_m2_centroid",
            "radio_batch_p2_m2_power_centroid",
            "radio_batch_p2_m2_inscribed",
            "radio_batch_p2_manual",
            "chk_batch_p2_use_threshold",
            "spin_batch_p2_thresh_percent",
            "lbl_batch_p2_thresh_spin",
            "chk_batch_p2_show_thresh",
            "chk_batch_m1_use_threshold",
            "spin_batch_m1_thresh_percent",
            "lbl_batch_m1_thresh_spin",
            "chk_batch_m1_show_thresh",
            "chk_batch_show_cross",
            "chk_batch_pixel_profile",
            "chk_batch_heatmap_gray",
            "btn_basler_view_range",
            "chk_basler_circle1", "spin_basler_circle1_r",
            "lbl_basler_circle1_pos", "btn_basler_clear_circle1",
            "chk_basler_circle2", "spin_basler_circle2_r",
            "lbl_basler_circle2_pos", "btn_basler_clear_circle2",
            "btn_basler_clear_circles",
            "lbl_batch_split_info",
            "lbl_batch_ab_distance_xy", "lbl_batch_ab_distance_total",
            "lbl_batch_ab_real_distance_xy", "lbl_batch_ab_real_distance_total",
            "spin_basler_range_x", "spin_basler_range_y",
            "spin_basler_range_w", "spin_basler_range_h",
            "btn_batch_save_params",
            "btn_batch_contour_popup",
            "btn_batch_view_cross",
        ]
        for attr in attrs:
            w = getattr(self, attr, None)
            if w is not None:
                w.hide()

        # 匿名 QLabel（X/Y/W/H、半徑、%、演算法提示等）一併隱藏
        hide_exact = {"X", "Y", "W", "H", "半徑", "%:", "%", "Min", "Max"}
        hide_substr = (
            "光斑定位",
            "範圍檢視",
            "標記圓",
            "Expected Distance",
            "Grayscale",
            "像素剖面",
            "質心／門檻幾何",
            "背景扣除",
            "顯示門檻",
            "顯示 切分",
            "顯示 M1",
            "啟用波谷",
            "啟用點擊像素",
            "熱力圖改為",
            "第一點",
            "第二點",
        )
        for w in self.findChildren(QLabel):
            # 保留我們稍後會建的短／長曝光標題與抓光斑區
            t = (w.text() or "").strip()
            if not t:
                continue
            spot_box = getattr(self, "_spot_controls_box", None)
            if spot_box is not None and spot_box.isAncestorOf(w):
                continue
            if t in hide_exact or any(key in t for key in hide_substr):
                w.hide()

        # 隱藏左側殘留水平分隔線（避免大片空白線）
        left = getattr(self, "batch_left_settings_widget", None)
        if left is not None:
            for line in left.findChildren(QFrame):
                if line.frameShape() == QFrame.HLine:
                    line.hide()

    def _compact_left_panel(self):
        """縮小左側間距，隱藏殘留控件，內容頂对齐，多餘空白集中在底部。"""
        left = getattr(self, "batch_left_settings_layout", None)
        if left is None:
            return
        left.setSpacing(4)
        left.setContentsMargins(8, 6, 8, 6)
        left.setAlignment(Qt.AlignTop)

        # 拿掉既有 spacer，稍後在 spot 控件後再加一個底部 stretch
        for i in reversed(range(left.count())):
            item = left.itemAt(i)
            if item is not None and item.spacerItem() is not None:
                left.removeItem(item)

        content = getattr(self, "batch_left_settings_widget", None)
        if content is None:
            return

        # 隱藏繼承下來、此分頁不用的 radio／checkbox／按鈕（避免撐出大塊空白）
        keep_buttons = {
            getattr(self, "btn_grab_short_spot", None),
            getattr(self, "btn_grab_long_spot", None),
            getattr(self, "btn_grab_axial_1d", None),
            getattr(self, "btn_clear_spots", None),
            getattr(self, "btn_open_collage", None),
        }
        keep_buttons.discard(None)

        for w in content.findChildren(QRadioButton):
            w.hide()
            w.setMaximumHeight(0)

        for w in content.findChildren(QCheckBox):
            t = (w.text() or "")
            if t in ("log", "color scale", "平滑") or "長曝光搜尋範圍" in t:
                continue
            if w is getattr(self, "chk_long_search_roi", None):
                continue
            w.hide()
            w.setMaximumHeight(0)

        for w in content.findChildren(QPushButton):
            if w in keep_buttons:
                continue
            # 上方固定區的選檔按鈕不在 left_content 內；此處全藏
            w.hide()
            w.setMaximumHeight(0)

        for w in content.findChildren(QLabel):
            t = (w.text() or "").strip()
            if not t:
                w.hide()
                w.setMaximumHeight(0)
                continue
            keep_prefix = (
                "抓光斑",
                "質心／門檻",
                "軸向一維",
                "短曝光",
                "長曝光",
                "兩光斑",
                "X",
                "Y",
                "W",
                "H",
            )
            if any(t.startswith(p) or t == p for p in keep_prefix):
                continue
            if t in {"X", "Y", "W", "H"}:
                continue
            # 仍可能是 ROI 旁的 X/Y 標籤（在 spot box 內）— 上面已 keep
            if w.parentWidget() is getattr(self, "_spot_controls_box", None):
                continue
            # spot box 內的子標籤
            spot_box = getattr(self, "_spot_controls_box", None)
            if spot_box is not None and spot_box.isAncestorOf(w):
                continue
            w.hide()
            w.setMaximumHeight(0)

        scroll = getattr(self, "batch_left_scroll", None)
        if scroll is not None:
            scroll.setWidgetResizable(True)

    def show_batch_contour_window(self):
        """此分頁不使用 Contour 檢視。"""
        return

    def save_current_batch_params(self):
        """此分頁不使用暫存參數按鈕。"""
        return
    def _remove_distance_table(self):
        table = getattr(self, "batch_distance_table", None)
        if table is None:
            return
        table_panel = table.parentWidget()
        if table_panel is not None:
            table_panel.hide()
        btn = getattr(self, "btn_batch_refresh_distance", None)
        if btn is not None:
            btn.hide()
            btn.setEnabled(False)

    def _make_display_bar(
        self,
        log_chk,
        bw_chk,
        black_lbl,
        black_spin,
        white_lbl,
        white_spin,
        smooth_chk,
        smooth_spin,
        object_name,
    ):
        def _group(*widgets):
            box = QWidget()
            row = QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            for w in widgets:
                row.addWidget(w)
            return box

        def _vsep():
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet("color: #B0BEC5;")
            return line

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)
        toolbar.setSpacing(14)
        toolbar.addWidget(_group(log_chk))
        toolbar.addWidget(_vsep())
        toolbar.addWidget(
            _group(bw_chk, black_lbl, black_spin, white_lbl, white_spin)
        )
        toolbar.addWidget(_vsep())
        toolbar.addWidget(_group(smooth_chk, smooth_spin))
        toolbar.addStretch(1)

        bar_host = QWidget()
        bar_host.setObjectName(object_name)
        bar_host.setLayout(toolbar)
        bar_host.setStyleSheet(
            f"#{object_name} {{ background: #ECEFF1; border: 1px solid #CFD8DC; "
            "border-radius: 4px; }"
        )
        return bar_host

    def _hide_interactive_panel_chrome(self, panel):
        for attr in (
            "lbl_tip",
            "btn_export",
            "btn_toggle_cross",
            "lbl_level_max",
            "lbl_level_min",
        ):
            w = getattr(panel, attr, None)
            if w is not None:
                w.hide()

    def _make_heatmap_panel(self, title, on_levels):
        panel = InteractiveHeatmapPanel(
            title=title,
            x_label="X Pixels",
            y_label="Y Pixels",
            aspect_locked=True,
            with_profiles=True,
            show_colorbar=True,
            show_tip=False,
        )
        self._hide_interactive_panel_chrome(panel)
        panel.plot.getViewBox().invertY(True)
        if panel.plot_y_profile is not None:
            panel.plot_y_profile.getViewBox().invertY(True)
        if panel.hist is not None:
            panel.hist.gradient.setColorMap(self.batch_jet_map)
            panel.hist.sigLevelsChanged.connect(on_levels)
        return panel

    def _build_long_display_controls(self):
        self.chk_long_heatmap_log = QCheckBox("log")
        self.chk_long_heatmap_log.setChecked(False)
        self.chk_long_heatmap_log.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_long_heatmap_log.toggled.connect(self._on_long_display_option_changed)

        self.chk_long_bw_scale = QCheckBox("color scale")
        self.chk_long_bw_scale.setChecked(False)
        self.chk_long_bw_scale.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_long_bw_scale.toggled.connect(self._on_long_bw_scale_toggled)

        self.lbl_long_black_below = QLabel("≤黑")
        self.lbl_long_black_below.setStyleSheet("color: #455A64;")
        self.spin_long_black_below = NoWheelDoubleSpinBox()
        self.spin_long_black_below.setRange(0.0, 1e9)
        self.spin_long_black_below.setDecimals(1)
        self.spin_long_black_below.setValue(_DEFAULT_BLACK_BELOW)
        self.spin_long_black_below.setFixedWidth(72)
        self.spin_long_black_below.setEnabled(False)
        self.spin_long_black_below.valueChanged.connect(
            self._on_long_bw_threshold_changed
        )

        self.lbl_long_white_above = QLabel("≥白")
        self.lbl_long_white_above.setStyleSheet("color: #455A64;")
        self.spin_long_white_above = NoWheelDoubleSpinBox()
        self.spin_long_white_above.setRange(0.0, 1e9)
        self.spin_long_white_above.setDecimals(1)
        self.spin_long_white_above.setValue(_DEFAULT_WHITE_ABOVE)
        self.spin_long_white_above.setFixedWidth(72)
        self.spin_long_white_above.setEnabled(False)
        self.spin_long_white_above.valueChanged.connect(
            self._on_long_bw_threshold_changed
        )

        self.chk_long_heatmap_smooth = QCheckBox("平滑")
        self.chk_long_heatmap_smooth.setChecked(False)
        self.chk_long_heatmap_smooth.setStyleSheet(
            "font-weight: bold; color: #37474F;"
        )
        self.chk_long_heatmap_smooth.toggled.connect(self._on_long_smooth_toggled)

        self.spin_long_heatmap_smooth = NoWheelSpinBox()
        self.spin_long_heatmap_smooth.setRange(1, 201)
        self.spin_long_heatmap_smooth.setSingleStep(2)
        self.spin_long_heatmap_smooth.setValue(31)
        self.spin_long_heatmap_smooth.setFixedWidth(60)
        self.spin_long_heatmap_smooth.setEnabled(False)
        self.spin_long_heatmap_smooth.valueChanged.connect(
            self._on_long_display_option_changed
        )

    def _build_short_display_controls(self):
        """短曝光顯示選項：若原 widget 仍在舊欄位內，先建立獨立一套以免隨欄位銷毀。"""
        # 重新建立，避免 _basler_heat_column 被移除時連控件一起刪掉
        self.chk_batch_heatmap_log = QCheckBox("log")
        self.chk_batch_heatmap_log.setChecked(False)
        self.chk_batch_heatmap_log.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_batch_heatmap_log.toggled.connect(self._on_basler_display_option_changed)

        self.chk_basler_bw_scale = QCheckBox("color scale")
        self.chk_basler_bw_scale.setChecked(False)
        self.chk_basler_bw_scale.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_basler_bw_scale.toggled.connect(self._on_basler_bw_scale_toggled)

        self.lbl_basler_black_below = QLabel("≤黑")
        self.lbl_basler_black_below.setStyleSheet("color: #455A64;")
        self.spin_basler_black_below = NoWheelDoubleSpinBox()
        self.spin_basler_black_below.setRange(0.0, 1e9)
        self.spin_basler_black_below.setDecimals(1)
        self.spin_basler_black_below.setValue(_DEFAULT_BLACK_BELOW)
        self.spin_basler_black_below.setFixedWidth(72)
        self.spin_basler_black_below.setEnabled(False)
        self.spin_basler_black_below.valueChanged.connect(
            self._on_basler_bw_threshold_changed
        )

        self.lbl_basler_white_above = QLabel("≥白")
        self.lbl_basler_white_above.setStyleSheet("color: #455A64;")
        self.spin_basler_white_above = NoWheelDoubleSpinBox()
        self.spin_basler_white_above.setRange(0.0, 1e9)
        self.spin_basler_white_above.setDecimals(1)
        self.spin_basler_white_above.setValue(_DEFAULT_WHITE_ABOVE)
        self.spin_basler_white_above.setFixedWidth(72)
        self.spin_basler_white_above.setEnabled(False)
        self.spin_basler_white_above.valueChanged.connect(
            self._on_basler_bw_threshold_changed
        )

        self.chk_batch_heatmap_smooth = QCheckBox("平滑")
        self.chk_batch_heatmap_smooth.setChecked(False)
        self.chk_batch_heatmap_smooth.setStyleSheet(
            "font-weight: bold; color: #37474F;"
        )
        self.chk_batch_heatmap_smooth.toggled.connect(self._on_basler_smooth_toggled)

        self.spin_batch_heatmap_smooth = NoWheelSpinBox()
        self.spin_batch_heatmap_smooth.setRange(1, 201)
        self.spin_batch_heatmap_smooth.setSingleStep(2)
        self.spin_batch_heatmap_smooth.setValue(31)
        self.spin_batch_heatmap_smooth.setFixedWidth(60)
        self.spin_batch_heatmap_smooth.setEnabled(False)
        self.spin_batch_heatmap_smooth.valueChanged.connect(
            self._on_basler_display_option_changed
        )

    def _build_collage_display_controls(self):
        """拼貼視窗：Short 區／Long 區各自獨立 log、color scale、平滑。"""

        def _make_pair(prefix, on_display, on_bw_toggle, on_bw_thresh, on_smooth_toggle):
            chk_log = QCheckBox("log")
            chk_log.setChecked(False)
            chk_log.setStyleSheet("font-weight: bold; color: #37474F;")
            chk_log.toggled.connect(on_display)

            chk_bw = QCheckBox("color scale")
            chk_bw.setChecked(False)
            chk_bw.setStyleSheet("font-weight: bold; color: #37474F;")
            chk_bw.toggled.connect(on_bw_toggle)

            lbl_black = QLabel("≤黑")
            lbl_black.setStyleSheet("color: #455A64;")
            spin_black = NoWheelDoubleSpinBox()
            spin_black.setRange(0.0, 1e9)
            spin_black.setDecimals(1)
            spin_black.setValue(_DEFAULT_BLACK_BELOW)
            spin_black.setFixedWidth(72)
            spin_black.setEnabled(False)
            spin_black.valueChanged.connect(on_bw_thresh)

            lbl_white = QLabel("≥白")
            lbl_white.setStyleSheet("color: #455A64;")
            spin_white = NoWheelDoubleSpinBox()
            spin_white.setRange(0.0, 1e9)
            spin_white.setDecimals(1)
            spin_white.setValue(_DEFAULT_WHITE_ABOVE)
            spin_white.setFixedWidth(72)
            spin_white.setEnabled(False)
            spin_white.valueChanged.connect(on_bw_thresh)

            chk_smooth = QCheckBox("平滑")
            chk_smooth.setChecked(False)
            chk_smooth.setStyleSheet("font-weight: bold; color: #37474F;")
            chk_smooth.toggled.connect(on_smooth_toggle)

            spin_smooth = NoWheelSpinBox()
            spin_smooth.setRange(1, 201)
            spin_smooth.setSingleStep(2)
            spin_smooth.setValue(31)
            spin_smooth.setFixedWidth(60)
            spin_smooth.setEnabled(False)
            spin_smooth.valueChanged.connect(on_display)

            return {
                "chk_log": chk_log,
                "chk_bw": chk_bw,
                "lbl_black": lbl_black,
                "spin_black": spin_black,
                "lbl_white": lbl_white,
                "spin_white": spin_white,
                "chk_smooth": chk_smooth,
                "spin_smooth": spin_smooth,
            }

        short = _make_pair(
            "short",
            self._on_collage_display_option_changed,
            self._on_collage_short_bw_scale_toggled,
            self._on_collage_short_bw_threshold_changed,
            self._on_collage_short_smooth_toggled,
        )
        long = _make_pair(
            "long",
            self._on_collage_display_option_changed,
            self._on_collage_long_bw_scale_toggled,
            self._on_collage_long_bw_threshold_changed,
            self._on_collage_long_smooth_toggled,
        )

        self.chk_collage_short_log = short["chk_log"]
        self.chk_collage_short_bw_scale = short["chk_bw"]
        self.lbl_collage_short_black_below = short["lbl_black"]
        self.spin_collage_short_black_below = short["spin_black"]
        self.lbl_collage_short_white_above = short["lbl_white"]
        self.spin_collage_short_white_above = short["spin_white"]
        self.chk_collage_short_smooth = short["chk_smooth"]
        self.spin_collage_short_smooth = short["spin_smooth"]

        self.chk_collage_long_log = long["chk_log"]
        self.chk_collage_long_bw_scale = long["chk_bw"]
        self.lbl_collage_long_black_below = long["lbl_black"]
        self.spin_collage_long_black_below = long["spin_black"]
        self.lbl_collage_long_white_above = long["lbl_white"]
        self.spin_collage_long_white_above = long["spin_white"]
        self.chk_collage_long_smooth = long["chk_smooth"]
        self.spin_collage_long_smooth = long["spin_smooth"]

    def _make_collage_region_bar(self, title, title_color, controls, object_name):
        """標題 + log／color scale／平滑 工具列。"""
        host = QWidget()
        host.setObjectName(object_name)
        row = QHBoxLayout(host)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(8)
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"font-weight: bold; color: {title_color}; font-size: 11px; min-width: 64px;"
        )
        row.addWidget(lbl)
        bar = self._make_display_bar(
            controls["chk_log"],
            controls["chk_bw"],
            controls["lbl_black"],
            controls["spin_black"],
            controls["lbl_white"],
            controls["spin_white"],
            controls["chk_smooth"],
            controls["spin_smooth"],
            object_name + "Inner",
        )
        row.addWidget(bar, 1)
        host.setStyleSheet(
            f"#{object_name} {{ background: #ECEFF1; border: 1px solid #CFD8DC; "
            "border-radius: 4px; }"
        )
        return host

    def _build_dual_heatmap_layout(self):
        """左右短／長 InteractiveHeatmapPanel；拼貼改獨立視窗。"""
        old_col = getattr(self, "_basler_heat_column", None)
        host = None
        if old_col is not None:
            host = old_col.parentWidget()
        if host is None:
            heat = getattr(self, "win_batch_top", None)
            host = heat.parentWidget() if heat is not None else None
        if host is None:
            return

        host_layout = host.layout()
        if host_layout is None:
            host_layout = QVBoxLayout(host)
            host_layout.setContentsMargins(0, 0, 0, 0)
            host_layout.setSpacing(6)

        for attr in (
            "_short_heat_column",
            "_long_heat_column",
            "_collage_heat_column",
            "_dual_stack_column",
            "_basler_heat_column",
        ):
            col = getattr(self, attr, None)
            if col is None:
                continue
            host_layout.removeWidget(col)
            col.hide()
            col.deleteLater()
            setattr(self, attr, None)

        win_old = getattr(self, "win_batch_top", None)
        if win_old is not None:
            win_old.hide()

        self._build_short_display_controls()
        self._build_long_display_controls()

        short_wrap = QWidget()
        short_layout = QVBoxLayout(short_wrap)
        short_layout.setContentsMargins(0, 0, 0, 0)
        short_layout.setSpacing(4)

        short_lbl = QLabel("短曝光 (Short)")
        short_lbl.setStyleSheet("font-weight: bold; color: #1565C0; font-size: 12px;")
        short_layout.addWidget(short_lbl)
        self.lbl_short_heat = short_lbl

        short_bar = self._make_display_bar(
            self.chk_batch_heatmap_log,
            self.chk_basler_bw_scale,
            self.lbl_basler_black_below,
            self.spin_basler_black_below,
            self.lbl_basler_white_above,
            self.spin_basler_white_above,
            self.chk_batch_heatmap_smooth,
            self.spin_batch_heatmap_smooth,
            "baslerShortDisplayBar",
        )
        short_layout.addWidget(short_bar)

        self.short_heatmap_panel = self._make_heatmap_panel(
            "Short Exposure Heatmap", self._on_short_levels_changed
        )
        short_layout.addWidget(self.short_heatmap_panel, 1)

        long_wrap = QWidget()
        long_layout = QVBoxLayout(long_wrap)
        long_layout.setContentsMargins(0, 0, 0, 0)
        long_layout.setSpacing(4)

        long_lbl = QLabel("長曝光 (Long)")
        long_lbl.setStyleSheet("font-weight: bold; color: #6A1B9A; font-size: 12px;")
        long_layout.addWidget(long_lbl)
        self.lbl_long_heat = long_lbl

        long_bar = self._make_display_bar(
            self.chk_long_heatmap_log,
            self.chk_long_bw_scale,
            self.lbl_long_black_below,
            self.spin_long_black_below,
            self.lbl_long_white_above,
            self.spin_long_white_above,
            self.chk_long_heatmap_smooth,
            self.spin_long_heatmap_smooth,
            "baslerLongDisplayBar",
        )
        long_layout.addWidget(long_bar)

        self.long_heatmap_panel = self._make_heatmap_panel(
            "Long Exposure Heatmap", self._on_long_levels_changed
        )
        long_layout.addWidget(self.long_heatmap_panel, 1)

        dual_row = QWidget()
        dual_row_layout = QHBoxLayout(dual_row)
        dual_row_layout.setContentsMargins(0, 0, 0, 0)
        dual_row_layout.setSpacing(6)
        dual_row_layout.addWidget(short_wrap, 1)
        dual_row_layout.addWidget(long_wrap, 1)

        insert_at = 0
        for i in range(host_layout.count()):
            item = host_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is not None and w is getattr(self, "win_batch_top", None):
                insert_at = i
                break
            if item is not None and item.layout() is not None and i == 0:
                insert_at = 1
        host_layout.insertWidget(insert_at, dual_row, 1)

        self._dual_stack_column = dual_row
        self._basler_heat_column = dual_row
        self._short_heat_column = short_wrap
        self._long_heat_column = long_wrap
        self._collage_heat_column = None

    def _ensure_collage_dialog(self):
        """建立（或重用）拼貼獨立視窗；尚未按下按鈕時可不顯示。"""
        dlg = getattr(self, "_collage_dialog", None)
        if dlg is not None:
            return dlg

        if getattr(self, "chk_collage_short_log", None) is None:
            self._build_collage_display_controls()

        dlg = QDialog(self)
        dlg.setWindowTitle("拼貼 Heatmap（Short 補區 ＋ Long 搜尋範圍）")
        dlg.setWindowFlags(
            dlg.windowFlags() | Qt.Window | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        dlg.resize(980, 760)
        dlg.setAttribute(Qt.WA_DeleteOnClose, False)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        hdr = QHBoxLayout()
        collage_lbl = QLabel("拼貼 (Short 補區 ＋ Long 搜尋範圍)")
        collage_lbl.setStyleSheet("font-weight: bold; color: #00695C; font-size: 12px;")
        self.lbl_collage_heat = collage_lbl
        hdr.addWidget(collage_lbl)
        hdr.addStretch(1)
        self.lbl_collage_roi_info = QLabel(
            "尚未建立拼貼（請載入雙檔並設定長曝光搜尋範圍）"
        )
        self.lbl_collage_roi_info.setStyleSheet("color: #546E7A; font-size: 11px;")
        hdr.addWidget(self.lbl_collage_roi_info)
        root.addLayout(hdr)

        short_bar = self._make_collage_region_bar(
            "Short 區",
            "#1565C0",
            {
                "chk_log": self.chk_collage_short_log,
                "chk_bw": self.chk_collage_short_bw_scale,
                "lbl_black": self.lbl_collage_short_black_below,
                "spin_black": self.spin_collage_short_black_below,
                "lbl_white": self.lbl_collage_short_white_above,
                "spin_white": self.spin_collage_short_white_above,
                "chk_smooth": self.chk_collage_short_smooth,
                "spin_smooth": self.spin_collage_short_smooth,
            },
            "baslerCollageShortBar",
        )
        long_bar = self._make_collage_region_bar(
            "Long 區",
            "#6A1B9A",
            {
                "chk_log": self.chk_collage_long_log,
                "chk_bw": self.chk_collage_long_bw_scale,
                "lbl_black": self.lbl_collage_long_black_below,
                "spin_black": self.spin_collage_long_black_below,
                "lbl_white": self.lbl_collage_long_white_above,
                "spin_white": self.spin_collage_long_white_above,
                "chk_smooth": self.chk_collage_long_smooth,
                "spin_smooth": self.spin_collage_long_smooth,
            },
            "baslerCollageLongBar",
        )
        root.addWidget(short_bar)
        root.addWidget(long_bar)

        self.collage_heatmap_panel = self._make_heatmap_panel(
            "Collage Heatmap (Short+Long)", self._on_collage_levels_changed
        )
        self.collage_heatmap_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        root.addWidget(self.collage_heatmap_panel, 1)

        self._collage_dialog = dlg
        return dlg

    def open_collage_window(self):
        """按鍵開啟拼貼視窗（非模態，可一邊改 ROI 一邊看）。"""
        if (
            getattr(self, "batch_result_matrix", None) is None
            or self.batch_long_matrix is None
        ):
            QMessageBox.warning(self, "警告", "請先載入短／長曝光檔案。")
            return
        dlg = self._ensure_collage_dialog()
        self._refresh_collage_heatmap_display(reset_levels=True)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _build_spot_controls(self):
        left = getattr(self, "batch_left_settings_layout", None)
        if left is None:
            return

        box = QWidget()
        box.setObjectName("baslerDualSpotBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(4)
        lay.setAlignment(Qt.AlignTop)

        title = QLabel("抓光斑（短／長獨立）")
        title.setStyleSheet("font-weight: bold; color: #1565C0; font-size: 12px;")

        algo = QLabel(
            "既有：質心／門檻幾何（背景扣除＋最大連通區＋亞像素）\n"
            "試用：軸向一維投影＋微分（短曝光定位 → 長曝光 200×512 條帶水平加總）"
        )
        algo.setStyleSheet("color: #546E7A; font-size: 11px;")
        algo.setWordWrap(True)
        self.lbl_spot_algo = algo

        self.spin_short_spot_thresh = NoWheelDoubleSpinBox()
        self.spin_short_spot_thresh.setRange(1.0, 99.0)
        self.spin_short_spot_thresh.setDecimals(1)
        self.spin_short_spot_thresh.setValue(50.0)
        self.spin_short_spot_thresh.setFixedWidth(72)
        self.spin_short_spot_thresh.setToolTip(
            "短曝光門檻％（相對搜尋區內最大值，背景扣除後）。"
        )

        short_thresh_row = QHBoxLayout()
        short_thresh_row.setContentsMargins(0, 0, 0, 0)
        short_thresh_row.setSpacing(6)
        short_thresh_row.addWidget(QLabel("短曝光門檻 %"))
        short_thresh_row.addWidget(self.spin_short_spot_thresh)
        short_thresh_row.addStretch(1)

        self.btn_grab_short_spot = QPushButton("抓短曝光光斑")
        self.btn_grab_short_spot.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #1565C0; border: none; border-radius: 4px; "
            "padding: 6px 10px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.btn_grab_short_spot.setToolTip(
            "短曝光全圖：質心／門檻幾何；結果以十字標到長曝光同一像素。"
        )
        self.btn_grab_short_spot.clicked.connect(self.grab_short_spot)
        self.lbl_short_spot_pos = QLabel("短曝光光斑: --")
        self.lbl_short_spot_pos.setStyleSheet("color: #1565C0; font-weight: bold;")
        self.lbl_short_spot_pos.setWordWrap(True)

        self.chk_long_search_roi = QCheckBox(
            "啟用長曝光搜尋範圍（抓點限制；亦作為拼貼 Long 區）"
        )
        self.chk_long_search_roi.setChecked(True)
        self.chk_long_search_roi.setStyleSheet("font-weight: bold; color: #6A1B9A;")
        self.chk_long_search_roi.setToolTip(
            "勾選後長曝光只在範圍內抓第二光斑並顯示框線。\n"
            "無論是否勾選，下方 X/Y/W/H 都會用來拼貼："
            "範圍內＝長曝光，其餘＝短曝光。"
        )
        self.chk_long_search_roi.toggled.connect(self._on_long_roi_option_changed)

        roi_grid = QGridLayout()
        roi_grid.setContentsMargins(0, 0, 0, 0)
        roi_grid.setHorizontalSpacing(6)
        roi_grid.setVerticalSpacing(2)
        self.spin_long_roi_x = NoWheelSpinBox()
        self.spin_long_roi_y = NoWheelSpinBox()
        self.spin_long_roi_w = NoWheelSpinBox()
        self.spin_long_roi_h = NoWheelSpinBox()
        for spin, default in (
            (self.spin_long_roi_x, 0),
            (self.spin_long_roi_y, 0),
            (self.spin_long_roi_w, 200),
            (self.spin_long_roi_h, 200),
        ):
            spin.setRange(
                0 if spin in (self.spin_long_roi_x, self.spin_long_roi_y) else 1,
                100000,
            )
            spin.setValue(default)
            spin.setFixedWidth(72)
            spin.valueChanged.connect(self._on_long_roi_spin_changed)
        roi_grid.addWidget(QLabel("X"), 0, 0)
        roi_grid.addWidget(self.spin_long_roi_x, 0, 1)
        roi_grid.addWidget(QLabel("Y"), 0, 2)
        roi_grid.addWidget(self.spin_long_roi_y, 0, 3)
        roi_grid.addWidget(QLabel("W"), 1, 0)
        roi_grid.addWidget(self.spin_long_roi_w, 1, 1)
        roi_grid.addWidget(QLabel("H"), 1, 2)
        roi_grid.addWidget(self.spin_long_roi_h, 1, 3)

        self.spin_long_spot_thresh = NoWheelDoubleSpinBox()
        self.spin_long_spot_thresh.setRange(1.0, 99.0)
        self.spin_long_spot_thresh.setDecimals(1)
        self.spin_long_spot_thresh.setValue(50.0)
        self.spin_long_spot_thresh.setFixedWidth(72)
        self.spin_long_spot_thresh.setToolTip(
            "長曝光門檻％：搜尋範圍內最大值 → 背景扣除＋最大連通區＋亞像素質心。"
        )

        long_thresh_row = QHBoxLayout()
        long_thresh_row.setContentsMargins(0, 0, 0, 0)
        long_thresh_row.setSpacing(6)
        long_thresh_row.addWidget(QLabel("長曝光門檻 %"))
        long_thresh_row.addWidget(self.spin_long_spot_thresh)
        long_thresh_row.addStretch(1)

        self.btn_grab_long_spot = QPushButton("抓長曝光第二光斑（範圍內）")
        self.btn_grab_long_spot.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #6A1B9A; border: none; border-radius: 4px; "
            "padding: 6px 10px; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        self.btn_grab_long_spot.setToolTip(
            "長曝光搜尋範圍內：質心／門檻幾何；範圍外不參與。"
        )
        self.btn_grab_long_spot.clicked.connect(self.grab_long_spot2)
        self.lbl_long_spot_pos = QLabel("長曝光第二光斑: --")
        self.lbl_long_spot_pos.setStyleSheet("color: #6A1B9A; font-weight: bold;")
        self.lbl_long_spot_pos.setWordWrap(True)

        self.btn_grab_axial_1d = QPushButton("一維投影＋微分法抓雙光斑")
        self.btn_grab_axial_1d.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #00838F; border: none; border-radius: 4px; "
            "padding: 6px 10px; }"
            "QPushButton:hover { background-color: #0097A7; }"
        )
        self.btn_grab_axial_1d.setToolTip(
            "保留既有質心法；此為試用演算法：\n"
            "1) 先抓短曝光光斑（若尚未抓會自動抓）\n"
            "2) 以該點為寬中心，在長曝光框 寬200 × 長512\n"
            "3) 條帶內水平加總成 1D 剖面，微分零交越找雙峰\n"
            "4) 較遠離短曝光峰者作為長曝光第二光斑"
        )
        self.btn_grab_axial_1d.clicked.connect(self.grab_dual_spots_axial_1d)

        self.lbl_spot_delta = QLabel("兩光斑差距: --")
        self.lbl_spot_delta.setStyleSheet("color: #E65100; font-weight: bold;")
        self.lbl_spot_delta.setWordWrap(True)

        self.btn_clear_spots = QPushButton("清除光斑標記")
        self.btn_clear_spots.clicked.connect(self.clear_spots)

        self.btn_open_collage = QPushButton("開啟拼貼視窗")
        self.btn_open_collage.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #00695C; border: none; border-radius: 4px; "
            "padding: 6px 10px; }"
            "QPushButton:hover { background-color: #00897B; }"
        )
        self.btn_open_collage.setToolTip(
            "另開視窗顯示拼貼 heatmap（Long 搜尋範圍＋Short 其餘區），"
            "含剖面／colorbar／log，可與主畫面並排調整。"
        )
        self.btn_open_collage.clicked.connect(self.open_collage_window)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setMaximumHeight(2)

        lay.addWidget(sep)
        lay.addWidget(title)
        lay.addWidget(algo)
        lay.addLayout(short_thresh_row)
        lay.addWidget(self.btn_grab_short_spot)
        lay.addWidget(self.lbl_short_spot_pos)
        lay.addWidget(self.btn_grab_axial_1d)
        lay.addWidget(self.chk_long_search_roi)
        lay.addLayout(roi_grid)
        lay.addLayout(long_thresh_row)
        lay.addWidget(self.btn_grab_long_spot)
        lay.addWidget(self.lbl_long_spot_pos)
        lay.addWidget(self.lbl_spot_delta)
        lay.addWidget(self.btn_clear_spots)
        lay.addWidget(self.btn_open_collage)

        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left.addWidget(box)
        left.addStretch(1)
        self._spot_controls_box = box

    def _pick_data_file(self, title):
        start = self.save_dir_path or ""
        path, _ = QFileDialog.getOpenFileName(self, title, start, _FILE_FILTER)
        return path or ""

    def load_batch_m1_folder(self):
        self.load_short_file()

    def load_batch_m2_folder(self):
        self.load_long_file()

    def load_short_file(self):
        path = self._pick_data_file("選擇短曝光檔案")
        if not path:
            return
        self.batch_short_path = path
        self.batch_m2_root = os.path.dirname(path)
        self.batch_m1_root = ""
        if not self.save_dir_path:
            self.save_dir_path = os.path.dirname(path)
            self.lbl_batch_dir_path.setText(self.save_dir_path)
        self.lbl_batch_m1_info.setText(
            f"{os.path.basename(path)}\n{os.path.dirname(path)}"
        )
        self._update_pair_info()
        self._try_load_both_files()

    def load_long_file(self):
        path = self._pick_data_file("選擇長曝光檔案")
        if not path:
            return
        self.batch_long_path = path
        if not self.save_dir_path:
            self.save_dir_path = os.path.dirname(path)
            self.lbl_batch_dir_path.setText(self.save_dir_path)
        self.lbl_batch_m2_info.setText(
            f"{os.path.basename(path)}\n{os.path.dirname(path)}"
        )
        self._update_pair_info()
        self._try_load_both_files()

    def _update_pair_info(self):
        short_ok = bool(self.batch_short_path)
        long_ok = bool(self.batch_long_path)
        if short_ok and long_ok:
            self.lbl_batch_pair_info.setText(
                f"已選：短={os.path.basename(self.batch_short_path)}｜"
                f"長={os.path.basename(self.batch_long_path)}"
            )
        elif short_ok:
            self.lbl_batch_pair_info.setText("已選短曝光；請再選長曝光檔案")
        elif long_ok:
            self.lbl_batch_pair_info.setText("已選長曝光；請再選短曝光檔案")
        else:
            self.lbl_batch_pair_info.setText("尚未選擇短／長曝光檔案")

    def _try_load_both_files(self):
        if not self.batch_short_path or not self.batch_long_path:
            return
        try:
            short_mat = load_numeric_matrix(self.batch_short_path)
            long_mat = load_numeric_matrix(self.batch_long_path)
        except Exception as exc:
            QMessageBox.critical(self, "讀檔失敗", f"無法讀取檔案：\n{exc}")
            self.lbl_batch_status.setText("狀態: 讀檔失敗")
            self.lbl_batch_status.setStyleSheet(
                "color: #C62828; font-weight: bold; font-size: 12px;"
            )
            return

        short_name = os.path.basename(self.batch_short_path)
        long_name = os.path.basename(self.batch_long_path)
        pair = {
            "location": "file",
            "filename": short_name,
            "long_filename": long_name,
            "m1_path": None,
            "m2_path": self.batch_short_path,
            "long_path": self.batch_long_path,
        }

        self.batch_pairs = [pair]
        self.batch_m1_files = []
        self.batch_m2_files = [self.batch_short_path]
        self.batch_long_files = [self.batch_long_path]
        self.batch_total_count = 1
        self.batch_current_idx = 0
        self.batch_saved_params.clear()
        self.batch_matrix_cache = {0: (None, short_mat)}
        self.batch_result_cache.clear()
        self.batch_long_cache = {0: long_mat}
        self.batch_result_matrix = np.asarray(short_mat, dtype=float)
        self.matrix2 = self.batch_result_matrix
        self.batch_long_matrix = np.asarray(long_mat, dtype=float)
        self.batch_available_locations = ["file"]
        self.batch_all_pairs_by_loc = {"file": [pair]}
        self.batch_location_config = {
            "file": {"enabled": True, "cycles": [short_name]}
        }

        self.clear_spots(silent=True)
        self._sync_long_roi_limits()
        self._refresh_short_heatmap_display(reset_levels=True)
        self._refresh_long_heatmap_display(reset_levels=True)
        self._build_collage_matrix()
        if getattr(self, "collage_heatmap_panel", None) is not None:
            self._refresh_collage_heatmap_display(reset_levels=True)
        self._redraw_spot_overlays()

        if hasattr(self, "lbl_batch_current_files"):
            self.lbl_batch_current_files.setText(
                f"目前檔案:\n短曝光: {short_name}\n長曝光: {long_name}"
            )
        h, w = self.batch_result_matrix.shape
        hl, wl = self.batch_long_matrix.shape
        self.lbl_batch_size.setText(f"矩陣大小: 短 {w}×{h} ｜ 長 {wl}×{hl}")
        self.lbl_batch_status.setText(
            f"狀態: 已載入短／長曝光｜{short_name} ＋ {long_name}"
        )
        self.lbl_batch_status.setStyleSheet(
            "color: #2E7D32; font-weight: bold; font-size: 12px;"
        )
        if hasattr(self, "btn_export_dual"):
            self.btn_export_dual.setEnabled(True)

    def process_batch_data(self):
        self._try_load_both_files()

    def _rebuild_batch_pairs(self):
        self._update_pair_info()

    def load_batch_group(self, idx, scroll_list=False):
        if idx != 0 or self.batch_total_count < 1:
            return
        self._try_load_both_files()

    def batch_go_prev(self):
        return

    def batch_go_next(self):
        return

    def update_batch_calculations(self, silent=False):
        return

    def redraw_batch_crosses(self):
        self._redraw_spot_overlays()

    def on_batch_process_mouse_clicked(self, evt):
        return

    def _transform_display_matrix(self, matrix, log_enabled, smooth_enabled, smooth_size):
        if matrix is None:
            return None
        data = np.asarray(matrix, dtype=float)
        if log_enabled:
            data = np.log10(np.maximum(data, 1.0))
        if smooth_enabled:
            k = max(1, int(smooth_size))
            if k > 1:
                data = uniform_filter(data, size=k, mode="nearest")
        return data

    def _short_display_matrix(self):
        return self._transform_display_matrix(
            getattr(self, "batch_result_matrix", None),
            self._basler_log_enabled(),
            self._basler_smooth_enabled(),
            self._basler_smooth_size(),
        )

    def _long_log_enabled(self):
        chk = getattr(self, "chk_long_heatmap_log", None)
        return chk is not None and chk.isChecked()

    def _long_smooth_enabled(self):
        chk = getattr(self, "chk_long_heatmap_smooth", None)
        return chk is not None and chk.isChecked()

    def _long_smooth_size(self):
        spin = getattr(self, "spin_long_heatmap_smooth", None)
        if spin is None:
            return 1
        return max(1, int(spin.value()))

    def _long_bw_scale_enabled(self):
        chk = getattr(self, "chk_long_bw_scale", None)
        return chk is not None and chk.isChecked()

    def _to_long_display_threshold(self, value):
        value = float(value)
        if self._long_log_enabled():
            return np.log10(max(value, 1.0))
        return value

    def _long_bw_thresholds_display(self):
        if not self._long_bw_scale_enabled():
            return None, None
        return (
            self._to_long_display_threshold(self.spin_long_black_below.value()),
            self._to_long_display_threshold(self.spin_long_white_above.value()),
        )

    def _long_display_matrix(self):
        return self._transform_display_matrix(
            self.batch_long_matrix,
            self._long_log_enabled(),
            self._long_smooth_enabled(),
            self._long_smooth_size(),
        )

    def _build_collage_matrix(self):
        """長曝光搜尋範圍區取長曝光，其餘取短曝光，合併成同尺寸拼貼。"""
        short_mat = getattr(self, "batch_result_matrix", None)
        long_mat = self.batch_long_matrix
        if short_mat is None or long_mat is None:
            self.batch_collage_matrix = None
            self._collage_roi_bounds = None
            return None

        short = np.asarray(short_mat, dtype=float)
        long = np.asarray(long_mat, dtype=float)
        h = min(short.shape[0], long.shape[0])
        w = min(short.shape[1], long.shape[1])
        if h <= 0 or w <= 0:
            self.batch_collage_matrix = None
            self._collage_roi_bounds = None
            return None

        short = short[:h, :w]
        long = long[:h, :w]
        collage = np.array(short, copy=True)
        bounds = clip_roi_to_matrix(long, self._long_roi_tuple())
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            collage[y0:y1, x0:x1] = long[y0:y1, x0:x1]
            self._collage_roi_bounds = (int(x0), int(y0), int(x1), int(y1))
        else:
            self._collage_roi_bounds = None

        self.batch_collage_matrix = collage
        return collage

    def _update_collage_roi_info_label(self):
        lbl = getattr(self, "lbl_collage_roi_info", None)
        if lbl is None:
            return
        bounds = self._collage_roi_bounds
        mat = self.batch_collage_matrix
        if mat is None:
            lbl.setText("尚未建立拼貼（請載入雙檔並設定長曝光搜尋範圍）")
            return
        h, w = mat.shape
        if bounds is None:
            lbl.setText(f"拼貼 {w}×{h}：搜尋範圍無效，目前全為短曝光")
            return
        x0, y0, x1, y1 = bounds
        lbl.setText(
            f"Long 區 X={x0}–{x1 - 1}, Y={y0}–{y1 - 1}；其餘為 Short｜{w}×{h}"
        )

    def _collage_short_log_enabled(self):
        chk = getattr(self, "chk_collage_short_log", None)
        return chk is not None and chk.isChecked()

    def _collage_long_log_enabled(self):
        chk = getattr(self, "chk_collage_long_log", None)
        return chk is not None and chk.isChecked()

    def _collage_short_smooth_enabled(self):
        chk = getattr(self, "chk_collage_short_smooth", None)
        return chk is not None and chk.isChecked()

    def _collage_long_smooth_enabled(self):
        chk = getattr(self, "chk_collage_long_smooth", None)
        return chk is not None and chk.isChecked()

    def _collage_short_smooth_size(self):
        spin = getattr(self, "spin_collage_short_smooth", None)
        if spin is None:
            return 1
        return max(1, int(spin.value()))

    def _collage_long_smooth_size(self):
        spin = getattr(self, "spin_collage_long_smooth", None)
        if spin is None:
            return 1
        return max(1, int(spin.value()))

    def _collage_short_bw_scale_enabled(self):
        chk = getattr(self, "chk_collage_short_bw_scale", None)
        return chk is not None and chk.isChecked()

    def _collage_long_bw_scale_enabled(self):
        chk = getattr(self, "chk_collage_long_bw_scale", None)
        return chk is not None and chk.isChecked()

    def _region_display_thresholds(self, log_on, bw_on, black_spin, white_spin):
        if not bw_on or black_spin is None or white_spin is None:
            return None, None
        black = float(black_spin.value())
        white = float(white_spin.value())
        if log_on:
            black = np.log10(max(black, 1.0))
            white = np.log10(max(white, 1.0))
        return black, white

    def _normalize_region_display(
        self,
        matrix,
        log_on,
        smooth_on,
        smooth_size,
        bw_on,
        black_spin,
        white_spin,
        force_unit,
    ):
        """單區顯示變換；color scale 或 force_unit 時映射到 0–1。"""
        data = self._transform_display_matrix(
            matrix, log_on, smooth_on, smooth_size
        )
        if data is None:
            return None
        data = np.asarray(data, dtype=float)
        if bw_on:
            lo, hi = self._region_display_thresholds(
                log_on, True, black_spin, white_spin
            )
            if lo is None or hi is None:
                return data
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        if force_unit:
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return np.zeros_like(data)
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        return data

    def _collage_display_matrix(self):
        """Short／Long 各自 log／color scale 後再依 ROI 拼貼。"""
        short_mat = getattr(self, "batch_result_matrix", None)
        long_mat = self.batch_long_matrix
        if short_mat is None or long_mat is None:
            return None

        short = np.asarray(short_mat, dtype=float)
        long = np.asarray(long_mat, dtype=float)
        h = min(short.shape[0], long.shape[0])
        w = min(short.shape[1], long.shape[1])
        if h <= 0 or w <= 0:
            return None
        short = short[:h, :w]
        long = long[:h, :w]

        force_unit = (
            self._collage_short_bw_scale_enabled()
            or self._collage_long_bw_scale_enabled()
        )
        short_disp = self._normalize_region_display(
            short,
            self._collage_short_log_enabled(),
            self._collage_short_smooth_enabled(),
            self._collage_short_smooth_size(),
            self._collage_short_bw_scale_enabled(),
            getattr(self, "spin_collage_short_black_below", None),
            getattr(self, "spin_collage_short_white_above", None),
            force_unit,
        )
        long_disp = self._normalize_region_display(
            long,
            self._collage_long_log_enabled(),
            self._collage_long_smooth_enabled(),
            self._collage_long_smooth_size(),
            self._collage_long_bw_scale_enabled(),
            getattr(self, "spin_collage_long_black_below", None),
            getattr(self, "spin_collage_long_white_above", None),
            force_unit,
        )
        if short_disp is None or long_disp is None:
            return None

        collage = np.array(short_disp, copy=True)
        bounds = self._collage_roi_bounds
        if bounds is None:
            clipped = clip_roi_to_matrix(long, self._long_roi_tuple())
            bounds = clipped
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            collage[y0:y1, x0:x1] = long_disp[y0:y1, x0:x1]
        return collage

    def _title_suffix(self, log_on, smooth_on, smooth_size, bw_on, black, white):
        title = ""
        if log_on:
            title += " [log10]"
        if smooth_on and smooth_size > 1:
            title += f" [smooth {smooth_size}]"
        if bw_on:
            title += f" [BW ≤{black:.0f}/≥{white:.0f}]"
        return title

    def _update_short_heatmap_title(self):
        panel = getattr(self, "short_heatmap_panel", None)
        if panel is None:
            return
        title = "Short Exposure Heatmap" + self._title_suffix(
            self._basler_log_enabled(),
            self._basler_smooth_enabled(),
            self._basler_smooth_size(),
            self._basler_bw_scale_enabled(),
            self.spin_basler_black_below.value(),
            self.spin_basler_white_above.value(),
        )
        panel.set_plot_title(title)

    def _update_long_heatmap_title(self):
        panel = getattr(self, "long_heatmap_panel", None)
        if panel is None:
            return
        title = "Long Exposure Heatmap" + self._title_suffix(
            self._long_log_enabled(),
            self._long_smooth_enabled(),
            self._long_smooth_size(),
            self._long_bw_scale_enabled(),
            self.spin_long_black_below.value(),
            self.spin_long_white_above.value(),
        )
        panel.set_plot_title(title)

    def _update_collage_heatmap_title(self):
        panel = getattr(self, "collage_heatmap_panel", None)
        if panel is None:
            return
        short_sfx = self._title_suffix(
            self._collage_short_log_enabled(),
            self._collage_short_smooth_enabled(),
            self._collage_short_smooth_size(),
            self._collage_short_bw_scale_enabled(),
            float(self.spin_collage_short_black_below.value()),
            float(self.spin_collage_short_white_above.value()),
        )
        long_sfx = self._title_suffix(
            self._collage_long_log_enabled(),
            self._collage_long_smooth_enabled(),
            self._collage_long_smooth_size(),
            self._collage_long_bw_scale_enabled(),
            float(self.spin_collage_long_black_below.value()),
            float(self.spin_collage_long_white_above.value()),
        )
        title = f"Collage Heatmap  Short{short_sfx or ' [-]'} ｜ Long{long_sfx or ' [-]'}"
        panel.set_plot_title(title)

    def _update_basler_heatmap_title(self):
        self._update_short_heatmap_title()

    def _set_panel_image(self, panel, display, reset_levels, intensity_label):
        if panel is None:
            return
        if display is None:
            panel.set_image(None, reset_view=True)
            return
        levels = None
        if reset_levels:
            finite = display[np.isfinite(display)]
            if finite.size:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmin == vmax:
                    vmax = vmin + 1.0
                levels = (vmin, vmax)
        panel.set_image(
            display.T,
            levels=levels,
            reset_view=True,
            source_matrix=display,
        )
        try:
            if panel.plot_x_profile is not None:
                panel.plot_x_profile.setLabel("right", intensity_label)
            if panel.plot_y_profile is not None:
                panel.plot_y_profile.setLabel("top", intensity_label)
        except Exception:
            pass

    def _intensity_label(self, log_on, smooth_on, smooth_size):
        label = "log10" if log_on else "Value"
        if smooth_on and smooth_size > 1:
            label += " (smooth)"
        return label

    def _apply_panel_colorbar(self, panel, guard_attr, bw_enabled, bw_thresholds):
        if getattr(self, guard_attr, False):
            return
        if panel is None or panel.hist is None:
            return
        try:
            vmin, vmax = panel.hist.getLevels()
        except Exception:
            return
        vmin, vmax = float(vmin), float(vmax)
        base = self.batch_jet_map
        if bw_enabled:
            black, white = bw_thresholds
            cmap = _build_bw_clip_colormap(
                base, vmin, vmax, black_floor=black, white_ceil=white
            )
        else:
            cmap = base
        setattr(self, guard_attr, True)
        try:
            panel.hist.gradient.setColorMap(cmap)
            panel.hist.setLevels(vmin, vmax)
        finally:
            setattr(self, guard_attr, False)

    def _refresh_short_heatmap_display(self, reset_levels=True):
        display = self._short_display_matrix()
        self._update_short_heatmap_title()
        self._set_panel_image(
            getattr(self, "short_heatmap_panel", None),
            display,
            reset_levels,
            self._intensity_label(
                self._basler_log_enabled(),
                self._basler_smooth_enabled(),
                self._basler_smooth_size(),
            ),
        )
        self._apply_short_colorbar_floor()
        self._redraw_spot_overlays()

    def _refresh_long_heatmap_display(self, reset_levels=True):
        display = self._long_display_matrix()
        self._update_long_heatmap_title()
        self._set_panel_image(
            getattr(self, "long_heatmap_panel", None),
            display,
            reset_levels,
            self._intensity_label(
                self._long_log_enabled(),
                self._long_smooth_enabled(),
                self._long_smooth_size(),
            ),
        )
        self._apply_long_colorbar_floor()
        self._redraw_spot_overlays()

    def _refresh_collage_heatmap_display(self, reset_levels=True):
        self._build_collage_matrix()
        self._update_collage_roi_info_label()
        panel = getattr(self, "collage_heatmap_panel", None)
        if panel is None:
            return
        display = self._collage_display_matrix()
        self._update_collage_heatmap_title()
        unit = (
            self._collage_short_bw_scale_enabled()
            or self._collage_long_bw_scale_enabled()
        )
        if unit:
            intensity = "norm 0–1"
        elif self._collage_short_log_enabled() or self._collage_long_log_enabled():
            intensity = "log10 / Value"
        else:
            intensity = "Value"
        self._set_panel_image(panel, display, reset_levels, intensity)
        self._apply_collage_colorbar_floor()

    def _apply_short_colorbar_floor(self):
        self._apply_panel_colorbar(
            getattr(self, "short_heatmap_panel", None),
            "_short_cmap_guard",
            self._basler_bw_scale_enabled(),
            self._basler_bw_thresholds_display(),
        )

    def _apply_long_colorbar_floor(self):
        self._apply_panel_colorbar(
            getattr(self, "long_heatmap_panel", None),
            "_long_cmap_guard",
            self._long_bw_scale_enabled(),
            self._long_bw_thresholds_display(),
        )

    def _apply_collage_colorbar_floor(self):
        # color scale 已分別寫入 Short／Long 區資料，colorbar 用標準 jet
        self._apply_panel_colorbar(
            getattr(self, "collage_heatmap_panel", None),
            "_collage_cmap_guard",
            False,
            (None, None),
        )

    def _on_short_levels_changed(self, *_args):
        self._apply_short_colorbar_floor()

    def _on_long_levels_changed(self, *_args):
        self._apply_long_colorbar_floor()

    def _on_collage_levels_changed(self, *_args):
        self._apply_collage_colorbar_floor()

    def _refresh_basler_heatmap_display(self, reset_levels=True):
        self._refresh_short_heatmap_display(reset_levels=reset_levels)

    def _apply_basler_colorbar_floor(self):
        self._apply_short_colorbar_floor()

    def _on_basler_display_option_changed(self, *_args):
        self._refresh_short_heatmap_display(reset_levels=True)

    def _on_basler_bw_scale_toggled(self, checked=False):
        enabled = bool(checked)
        self.spin_basler_black_below.setEnabled(enabled)
        self.spin_basler_white_above.setEnabled(enabled)
        self._update_short_heatmap_title()
        self._apply_short_colorbar_floor()

    def _on_basler_bw_threshold_changed(self, *_args):
        if not self._basler_bw_scale_enabled():
            return
        self._update_short_heatmap_title()
        self._apply_short_colorbar_floor()

    def _on_basler_smooth_toggled(self, checked=False):
        self.spin_batch_heatmap_smooth.setEnabled(bool(checked))
        self._on_basler_display_option_changed()

    def _on_long_smooth_toggled(self, checked=False):
        self.spin_long_heatmap_smooth.setEnabled(bool(checked))
        self._on_long_display_option_changed()

    def _on_long_bw_scale_toggled(self, checked=False):
        enabled = bool(checked)
        self.spin_long_black_below.setEnabled(enabled)
        self.spin_long_white_above.setEnabled(enabled)
        self._update_long_heatmap_title()
        self._apply_long_colorbar_floor()

    def _on_long_bw_threshold_changed(self, *_args):
        if not self._long_bw_scale_enabled():
            return
        self._update_long_heatmap_title()
        self._apply_long_colorbar_floor()

    def _on_long_display_option_changed(self, *_args):
        self._refresh_long_heatmap_display(reset_levels=True)

    def _on_collage_display_option_changed(self, *_args):
        self._refresh_collage_heatmap_display(reset_levels=True)

    def _on_collage_short_smooth_toggled(self, checked=False):
        self.spin_collage_short_smooth.setEnabled(bool(checked))
        self._on_collage_display_option_changed()

    def _on_collage_long_smooth_toggled(self, checked=False):
        self.spin_collage_long_smooth.setEnabled(bool(checked))
        self._on_collage_display_option_changed()

    def _on_collage_short_bw_scale_toggled(self, checked=False):
        enabled = bool(checked)
        self.spin_collage_short_black_below.setEnabled(enabled)
        self.spin_collage_short_white_above.setEnabled(enabled)
        self._on_collage_display_option_changed()

    def _on_collage_long_bw_scale_toggled(self, checked=False):
        enabled = bool(checked)
        self.spin_collage_long_black_below.setEnabled(enabled)
        self.spin_collage_long_white_above.setEnabled(enabled)
        self._on_collage_display_option_changed()

    def _on_collage_short_bw_threshold_changed(self, *_args):
        if not self._collage_short_bw_scale_enabled():
            return
        self._on_collage_display_option_changed()

    def _on_collage_long_bw_threshold_changed(self, *_args):
        if not self._collage_long_bw_scale_enabled():
            return
        self._on_collage_display_option_changed()

    def on_batch_heatmap_colormap_toggled(self, checked=False):
        self._apply_short_colorbar_floor()
        self._apply_long_colorbar_floor()
        self._apply_collage_colorbar_floor()

    def on_colorbar_levels_changed(self):
        self._apply_short_colorbar_floor()

    def _short_spot_thresh_percent(self):
        spin = getattr(self, "spin_short_spot_thresh", None)
        if spin is None:
            return 50.0
        return float(spin.value())

    def _long_spot_thresh_percent(self):
        spin = getattr(self, "spin_long_spot_thresh", None)
        if spin is None:
            return 50.0
        return float(spin.value())

    def _find_spot(self, matrix, roi=None, thresh_percent=50.0):
        """質心／門檻幾何：背景扣除＋最大連通區＋亞像素。

        長曝光若有搜尋範圍：只在 ROI 內找最大值並做質心（範圍外不參與）。
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0:
            return None
        ox = oy = 0
        work = matrix
        if roi is not None:
            clipped = clip_roi_to_matrix(matrix, roi)
            if clipped is None:
                return None
            x0, y0, x1, y1 = clipped
            work = matrix[y0:y1, x0:x1]
            if work.size == 0:
                return None
            ox, oy = x0, y0

        if not np.isfinite(work).any():
            return None

        cx, cy = compute_auto_spot_center(
            work,
            "centroid",
            use_threshold=True,
            thresh_percent=float(thresh_percent),
            bg_subtract=True,
            largest_cc_only=True,
            subpixel=True,
            power=1.0,
        )
        return float(cx + ox), float(cy + oy)

    def _long_roi_tuple(self):
        return (
            self.spin_long_roi_x.value(),
            self.spin_long_roi_y.value(),
            self.spin_long_roi_w.value(),
            self.spin_long_roi_h.value(),
        )

    def _sync_long_roi_limits(self):
        mat = self.batch_long_matrix
        if mat is None:
            return
        h, w = mat.shape
        self.spin_long_roi_x.setMaximum(max(0, w - 1))
        self.spin_long_roi_y.setMaximum(max(0, h - 1))
        self.spin_long_roi_w.setMaximum(max(1, w))
        self.spin_long_roi_h.setMaximum(max(1, h))
        if self.spin_long_roi_w.value() > w:
            self.spin_long_roi_w.setValue(w)
        if self.spin_long_roi_h.value() > h:
            self.spin_long_roi_h.setValue(h)
        if (
            self.spin_long_roi_x.value() == 0
            and self.spin_long_roi_y.value() == 0
            and self.spin_long_roi_w.value() == 200
            and self.spin_long_roi_h.value() == 200
        ):
            self.spin_long_roi_w.setValue(w)
            self.spin_long_roi_h.setValue(h)

    def grab_short_spot(self):
        matrix = getattr(self, "batch_result_matrix", None)
        if matrix is None:
            QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
            return
        spot = self._find_spot(
            matrix,
            roi=None,
            thresh_percent=self._short_spot_thresh_percent(),
        )
        if spot is None:
            QMessageBox.warning(self, "警告", "無法在短曝光找到光斑。")
            return
        self.short_spot = spot
        self.lbl_short_spot_pos.setText(
            f"短曝光光斑: X={spot[0]:.2f}, Y={spot[1]:.2f}"
        )
        self._update_spot_delta_label()
        self._redraw_spot_overlays()
        self.lbl_batch_status.setText(
            f"狀態: 已抓短曝光光斑 ({spot[0]:.1f}, {spot[1]:.1f})，"
            "已對應標示到長曝光"
        )

    def grab_long_spot2(self):
        matrix = self.batch_long_matrix
        if matrix is None:
            QMessageBox.warning(self, "警告", "請先載入長曝光檔案。")
            return
        roi = None
        if (
            getattr(self, "chk_long_search_roi", None) is not None
            and self.chk_long_search_roi.isChecked()
        ):
            roi = self._long_roi_tuple()
        spot = self._find_spot(
            matrix,
            roi=roi,
            thresh_percent=self._long_spot_thresh_percent(),
        )
        if spot is None:
            QMessageBox.warning(
                self, "警告", "無法在指定範圍內找到長曝光第二光斑。"
            )
            return
        self.long_spot2 = spot
        self.lbl_long_spot_pos.setText(
            f"長曝光第二光斑: X={spot[0]:.2f}, Y={spot[1]:.2f}"
        )
        self._update_spot_delta_label()
        self._redraw_spot_overlays()
        self.lbl_batch_status.setText(
            f"狀態: 已抓長曝光第二光斑 ({spot[0]:.1f}, {spot[1]:.1f})"
        )

    def _apply_long_roi_xywh(self, xywh):
        """寫入長曝光搜尋範圍 spin（短暫阻斷訊號避免重複重繪）。"""
        if xywh is None:
            return
        x, y, w, h = xywh
        spins = (
            self.spin_long_roi_x,
            self.spin_long_roi_y,
            self.spin_long_roi_w,
            self.spin_long_roi_h,
        )
        vals = (int(x), int(y), int(w), int(h))
        for spin, val in zip(spins, vals):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        if getattr(self, "chk_long_search_roi", None) is not None:
            self.chk_long_search_roi.blockSignals(True)
            self.chk_long_search_roi.setChecked(True)
            self.chk_long_search_roi.blockSignals(False)

    def grab_dual_spots_axial_1d(self):
        """軸向一維投影剖面與微分法（試用，不取代既有質心法）。

        1. 先確保短曝光光斑存在（質心／門檻）
        2. 以短曝光對應點為中心，長曝光框 寬 200 × 長 512
        3. 條帶水平加總 → 1D 剖面 → 微分找雙峰
        4. 較遠峰作為長曝光第二光斑
        """
        if getattr(self, "batch_result_matrix", None) is None:
            QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
            return
        if self.batch_long_matrix is None:
            QMessageBox.warning(self, "警告", "請先載入長曝光檔案。")
            return

        if self.short_spot is None:
            self.grab_short_spot()
            if self.short_spot is None:
                return

        center = self._map_short_spot_to_long()
        if center is None:
            QMessageBox.warning(self, "警告", "無法將短曝光光斑對應到長曝光。")
            return

        result = find_dual_spots_axial_1d_projection(
            self.batch_long_matrix,
            center,
            roi_w=200,
            roi_h=512,
            smooth_win=7,
            min_peak_distance=8,
        )
        spots = result.get("spots")
        if not spots or len(spots) < 2:
            QMessageBox.warning(
                self,
                "警告",
                "一維投影＋微分法未能在長曝光條帶內找到兩個光斑峰。\n"
                "可先確認短曝光定位是否落在雙光斑軸上，或手動調整搜尋範圍後再用質心法。",
            )
            if result.get("roi_xywh") is not None:
                self._apply_long_roi_xywh(result["roi_xywh"])
                self._redraw_spot_overlays()
            return

        # 兩峰 Y 過近 → 視為未分開
        if abs(float(spots[0][1]) - float(spots[1][1])) < 2.0:
            QMessageBox.warning(
                self,
                "警告",
                "一維投影＋微分法只找到單一主峰（雙峰未分開）。",
            )
            if result.get("roi_xywh") is not None:
                self._apply_long_roi_xywh(result["roi_xywh"])
                self._redraw_spot_overlays()
            return

        self._apply_long_roi_xywh(result.get("roi_xywh"))

        primary_i = int(result.get("primary_index", 0))
        secondary_i = 1 - primary_i
        spot2 = spots[secondary_i]
        self.long_spot2 = (float(spot2[0]), float(spot2[1]))
        self.lbl_long_spot_pos.setText(
            f"長曝光第二光斑: X={self.long_spot2[0]:.2f}, "
            f"Y={self.long_spot2[1]:.2f}（一維投影＋微分）"
        )
        self._update_spot_delta_label()
        self._redraw_spot_overlays()

        y_lo, y_hi = result.get("peak_ys", (None, None))
        self.lbl_batch_status.setText(
            f"狀態: 一維投影＋微分法｜雙峰 Y={y_lo:.1f}/{y_hi:.1f}｜"
            f"第二光斑 ({self.long_spot2[0]:.1f}, {self.long_spot2[1]:.1f})"
        )

    def clear_spots(self, silent=False):
        self.short_spot = None
        self.long_spot2 = None
        if hasattr(self, "lbl_short_spot_pos"):
            self.lbl_short_spot_pos.setText("短曝光光斑: --")
        if hasattr(self, "lbl_long_spot_pos"):
            self.lbl_long_spot_pos.setText("長曝光第二光斑: --")
        if hasattr(self, "lbl_spot_delta"):
            self.lbl_spot_delta.setText("兩光斑差距: --")
        self._redraw_spot_overlays()
        if not silent:
            self.lbl_batch_status.setText("狀態: 已清除光斑標記")

    def _px_to_um(self, px):
        """實際距離：μm = px / 2.5 * 3.45。"""
        return float(px) / 2.5 * 3.45

    def _update_spot_delta_label(self):
        if self.short_spot is None or self.long_spot2 is None:
            self.lbl_spot_delta.setText("兩光斑差距: --")
            return
        dx = self.long_spot2[0] - self.short_spot[0]
        dy = self.long_spot2[1] - self.short_spot[1]
        dist_px = float(np.hypot(dx, dy))
        dist_um = self._px_to_um(dist_px)
        self.lbl_spot_delta.setText(
            f"兩光斑差距: ΔX={dx:.2f} px, ΔY={dy:.2f} px, "
            f"|d|={dist_px:.2f} px ({dist_um:.2f} μm)"
        )

    def _hide_spot_overlays_for_export(self):
        """暫時清除十字與搜尋範圍，供匯出乾淨圖檔。"""
        short_panel = getattr(self, "short_heatmap_panel", None)
        long_panel = getattr(self, "long_heatmap_panel", None)
        if short_panel is not None:
            self._clear_plot_items(short_panel.plot, "_short_cross_items")
        if long_panel is not None:
            self._clear_plot_items(long_panel.plot, "_long_cross_items")

    def _clear_plot_items(self, plot, items_attr):
        items = list(getattr(self, items_attr, []) or [])
        for item in items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        setattr(self, items_attr, [])

    def _add_cross(self, plot, items_attr, cx, cy, shape, pen):
        h, w = shape
        v = pg.PlotCurveItem(x=[cx, cx], y=[-0.5, h - 0.5], pen=pen)
        hz = pg.PlotCurveItem(x=[-0.5, w - 0.5], y=[cy, cy], pen=pen)
        plot.addItem(v)
        plot.addItem(hz)
        getattr(self, items_attr).extend([v, hz])

    def _add_roi_rect(self, plot, items_attr, bounds, pen):
        x0, y0, x1, y1 = bounds
        xs = [x0 - 0.5, x1 - 0.5, x1 - 0.5, x0 - 0.5, x0 - 0.5]
        ys = [y0 - 0.5, y0 - 0.5, y1 - 0.5, y1 - 0.5, y0 - 0.5]
        item = pg.PlotCurveItem(x=xs, y=ys, pen=pen)
        plot.addItem(item)
        getattr(self, items_attr).append(item)

    def _map_short_spot_to_long(self):
        if self.short_spot is None or self.batch_long_matrix is None:
            return None
        cx, cy = self.short_spot
        h, w = self.batch_long_matrix.shape
        if w <= 0 or h <= 0:
            return None
        return (
            float(np.clip(cx, 0.0, w - 1.0)),
            float(np.clip(cy, 0.0, h - 1.0)),
        )

    def _redraw_spot_overlays(self):
        short_panel = getattr(self, "short_heatmap_panel", None)
        long_panel = getattr(self, "long_heatmap_panel", None)

        if short_panel is not None:
            self._clear_plot_items(short_panel.plot, "_short_cross_items")
            if self.short_spot is not None and self.batch_result_matrix is not None:
                self._add_cross(
                    short_panel.plot,
                    "_short_cross_items",
                    self.short_spot[0],
                    self.short_spot[1],
                    self.batch_result_matrix.shape,
                    pg.mkPen("#D50000", width=2.5),
                )

        if long_panel is not None:
            self._clear_plot_items(long_panel.plot, "_long_cross_items")
            if self.batch_long_matrix is not None:
                if (
                    getattr(self, "chk_long_search_roi", None) is not None
                    and self.chk_long_search_roi.isChecked()
                ):
                    bounds = clip_roi_to_matrix(
                        self.batch_long_matrix, self._long_roi_tuple()
                    )
                    if bounds is not None:
                        self._add_roi_rect(
                            long_panel.plot,
                            "_long_cross_items",
                            bounds,
                            pg.mkPen("#EF6C00", width=2.0, style=Qt.DashLine),
                        )
                mapped = self._map_short_spot_to_long()
                if mapped is not None:
                    self._add_cross(
                        long_panel.plot,
                        "_long_cross_items",
                        mapped[0],
                        mapped[1],
                        self.batch_long_matrix.shape,
                        pg.mkPen("#2962FF", width=2.5),
                    )
                if self.long_spot2 is not None:
                    self._add_cross(
                        long_panel.plot,
                        "_long_cross_items",
                        self.long_spot2[0],
                        self.long_spot2[1],
                        self.batch_long_matrix.shape,
                        pg.mkPen("#D50000", width=2.5),
                    )

    def _on_long_roi_option_changed(self, *_args):
        # 搜尋範圍同時驅動拼貼，旋鈕一律可調（勾選只影響長曝光抓點／框線）
        for spin in (
            getattr(self, "spin_long_roi_x", None),
            getattr(self, "spin_long_roi_y", None),
            getattr(self, "spin_long_roi_w", None),
            getattr(self, "spin_long_roi_h", None),
        ):
            if spin is not None:
                spin.setEnabled(True)
        self._redraw_spot_overlays()
        self._build_collage_matrix()
        if getattr(self, "collage_heatmap_panel", None) is not None:
            self._refresh_collage_heatmap_display(reset_levels=False)

    def _on_long_roi_spin_changed(self, *_args):
        self._redraw_spot_overlays()
        self._build_collage_matrix()
        if getattr(self, "collage_heatmap_panel", None) is not None:
            self._refresh_collage_heatmap_display(reset_levels=False)

    # ------------------------------------------------------------------
    # 匯出：圖片＋參數＋光斑位置＋ΔX
    # ------------------------------------------------------------------
    def _spot_delta_values(self):
        if self.short_spot is None or self.long_spot2 is None:
            return {
                "dx_px": None,
                "dy_px": None,
                "dist_px": None,
                "dx_um": None,
                "dy_um": None,
                "dist_um": None,
                "pixel_pitch_um": 3.45,
                "distance_scale": 2.5,
                "um_formula": "|d_px| / 2.5 * 3.45",
            }
        dx = float(self.long_spot2[0] - self.short_spot[0])
        dy = float(self.long_spot2[1] - self.short_spot[1])
        dist = float(np.hypot(dx, dy))
        return {
            "dx_px": dx,
            "dy_px": dy,
            "dist_px": dist,
            "dx_um": self._px_to_um(dx),
            "dy_um": self._px_to_um(dy),
            "dist_um": self._px_to_um(dist),
            "pixel_pitch_um": 3.45,
            "distance_scale": 2.5,
            "um_formula": "|d_px| / 2.5 * 3.45",
        }

    def _build_export_payload(self):
        # 確保拼貼矩陣／範圍與目前 ROI、顯示控件同步後再寫入參數
        self._build_collage_matrix()
        if getattr(self, "chk_collage_short_log", None) is None:
            self._build_collage_display_controls()
        delta = self._spot_delta_values()
        short_mat = getattr(self, "batch_result_matrix", None)
        long_mat = self.batch_long_matrix
        short_shape = list(short_mat.shape) if short_mat is not None else None
        long_shape = list(long_mat.shape) if long_mat is not None else None
        roi_enabled = bool(
            getattr(self, "chk_long_search_roi", None)
            and self.chk_long_search_roi.isChecked()
        )
        return {
            "mode": "basler_short_long_dual",
            "algorithm": "centroid_threshold_geom",
            "algorithm_desc": "質心／門檻幾何：背景扣除＋最大連通區＋亞像素",
            "short_file": os.path.basename(self.batch_short_path or ""),
            "long_file": os.path.basename(self.batch_long_path or ""),
            "short_path": self.batch_short_path or "",
            "long_path": self.batch_long_path or "",
            "short_matrix_shape_yx": short_shape,
            "long_matrix_shape_yx": long_shape,
            "short_thresh_percent": self._short_spot_thresh_percent(),
            "long_thresh_percent": self._long_spot_thresh_percent(),
            "long_search_roi_enabled": roi_enabled,
            "long_search_roi_xywh": list(self._long_roi_tuple()) if roi_enabled else None,
            "short_spot_x_px": None if self.short_spot is None else float(self.short_spot[0]),
            "short_spot_y_px": None if self.short_spot is None else float(self.short_spot[1]),
            "long_spot_x_px": None if self.long_spot2 is None else float(self.long_spot2[0]),
            "long_spot_y_px": None if self.long_spot2 is None else float(self.long_spot2[1]),
            "short_to_long_delta_x_px": delta["dx_px"],
            "short_to_long_delta_y_px": delta["dy_px"],
            "short_to_long_distance_px": delta["dist_px"],
            "short_to_long_delta_x_um": delta["dx_um"],
            "short_to_long_delta_y_um": delta["dy_um"],
            "short_to_long_distance_um": delta["dist_um"],
            "pixel_pitch_um": delta["pixel_pitch_um"],
            "distance_scale": delta["distance_scale"],
            "um_formula": delta["um_formula"],
            "short_display": {
                "log": self._basler_log_enabled(),
                "smooth": self._basler_smooth_enabled(),
                "smooth_size": self._basler_smooth_size(),
                "color_scale": self._basler_bw_scale_enabled(),
                "black_below": float(self.spin_basler_black_below.value()),
                "white_above": float(self.spin_basler_white_above.value()),
            },
            "long_display": {
                "log": self._long_log_enabled(),
                "smooth": self._long_smooth_enabled(),
                "smooth_size": self._long_smooth_size(),
                "color_scale": self._long_bw_scale_enabled(),
                "black_below": float(self.spin_long_black_below.value()),
                "white_above": float(self.spin_long_white_above.value()),
            },
            "collage": {
                "rule": "long_roi_from_long_rest_from_short",
                "matrix_shape_yx": (
                    list(self.batch_collage_matrix.shape)
                    if self.batch_collage_matrix is not None
                    else None
                ),
                "long_region_x0y0x1y1": (
                    list(self._collage_roi_bounds)
                    if self._collage_roi_bounds is not None
                    else None
                ),
                "long_search_roi_xywh": list(self._long_roi_tuple()),
                "short_region_display": {
                    "log": self._collage_short_log_enabled(),
                    "smooth": self._collage_short_smooth_enabled(),
                    "smooth_size": self._collage_short_smooth_size(),
                    "color_scale": self._collage_short_bw_scale_enabled(),
                    "black_below": float(self.spin_collage_short_black_below.value()),
                    "white_above": float(self.spin_collage_short_white_above.value()),
                },
                "long_region_display": {
                    "log": self._collage_long_log_enabled(),
                    "smooth": self._collage_long_smooth_enabled(),
                    "smooth_size": self._collage_long_smooth_size(),
                    "color_scale": self._collage_long_bw_scale_enabled(),
                    "black_below": float(self.spin_collage_long_black_below.value()),
                    "white_above": float(self.spin_collage_long_white_above.value()),
                },
            },
        }

    def _export_payload_rows(self, payload):
        rows = [
            ["Mode", payload["mode"], ""],
            ["Algorithm", payload["algorithm_desc"], ""],
            ["Short File", payload["short_file"], ""],
            ["Long File", payload["long_file"], ""],
            ["Short Threshold", payload["short_thresh_percent"], "%"],
            ["Long Threshold", payload["long_thresh_percent"], "%"],
            [
                "Long Search ROI Enabled",
                "Yes" if payload["long_search_roi_enabled"] else "No",
                "",
            ],
        ]
        roi = payload.get("long_search_roi_xywh")
        if roi:
            rows.extend(
                [
                    ["Long Search ROI X", roi[0], "px"],
                    ["Long Search ROI Y", roi[1], "px"],
                    ["Long Search ROI W", roi[2], "px"],
                    ["Long Search ROI H", roi[3], "px"],
                ]
            )
        collage = payload.get("collage") or {}
        rows.append(["Collage Rule", collage.get("rule"), ""])
        c_bounds = collage.get("long_region_x0y0x1y1")
        if c_bounds:
            rows.extend(
                [
                    ["Collage Long Region X0", c_bounds[0], "px"],
                    ["Collage Long Region Y0", c_bounds[1], "px"],
                    ["Collage Long Region X1", c_bounds[2], "px"],
                    ["Collage Long Region Y1", c_bounds[3], "px"],
                ]
            )
        c_shape = collage.get("matrix_shape_yx")
        if c_shape:
            rows.append(["Collage Matrix Shape Yx", f"{c_shape[0]}x{c_shape[1]}", ""])

        def _append_region_disp(prefix, disp):
            if not disp:
                return
            rows.extend(
                [
                    [f"{prefix} Log", "Yes" if disp.get("log") else "No", ""],
                    [f"{prefix} Smooth", "Yes" if disp.get("smooth") else "No", ""],
                    [f"{prefix} Smooth Size", disp.get("smooth_size"), ""],
                    [
                        f"{prefix} Color Scale",
                        "Yes" if disp.get("color_scale") else "No",
                        "",
                    ],
                    [f"{prefix} Black Below", disp.get("black_below"), ""],
                    [f"{prefix} White Above", disp.get("white_above"), ""],
                ]
            )

        _append_region_disp("Collage Short Region", collage.get("short_region_display"))
        _append_region_disp("Collage Long Region", collage.get("long_region_display"))
        rows.extend(
            [
                ["Short Spot X", payload["short_spot_x_px"], "px"],
                ["Short Spot Y", payload["short_spot_y_px"], "px"],
                ["Long Spot X", payload["long_spot_x_px"], "px"],
                ["Long Spot Y", payload["long_spot_y_px"], "px"],
                ["Delta X (Long-Short)", payload["short_to_long_delta_x_px"], "px"],
                ["Delta Y (Long-Short)", payload["short_to_long_delta_y_px"], "px"],
                ["Total Distance", payload["short_to_long_distance_px"], "px"],
                ["Delta X (Real)", payload["short_to_long_delta_x_um"], "μm"],
                ["Delta Y (Real)", payload["short_to_long_delta_y_um"], "μm"],
                ["Total Distance (Real)", payload["short_to_long_distance_um"], "μm"],
                ["Pixel Pitch", payload["pixel_pitch_um"], "μm"],
                ["Distance Scale", payload["distance_scale"], ""],
                ["UM Formula", payload["um_formula"], ""],
            ]
        )
        return rows

    def export_dual_results(self):
        if self.batch_result_matrix is None or self.batch_long_matrix is None:
            QMessageBox.warning(self, "警告", "請先載入短／長曝光檔案。")
            return
        if self.short_spot is None or self.long_spot2 is None:
            QMessageBox.warning(
                self, "警告", "請先完成短曝光與長曝光光斑抓取，再匯出。"
            )
            return

        default_name = "Basler_ShortLong_Spot"
        short_stem = os.path.splitext(os.path.basename(self.batch_short_path or ""))[0]
        if short_stem:
            default_name = f"{short_stem}_ShortLong"
        initial = default_name
        if self.save_dir_path:
            initial = os.path.join(self.save_dir_path, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出短／長曝光結果（JSON 基準檔名）",
            initial,
            "JSON Files (*.json)",
        )
        if not path:
            return
        base_path, ext = os.path.splitext(path)
        if ext.lower() != ".json":
            path = base_path + ".json"
            base_path = os.path.splitext(path)[0]

        try:
            payload = self._build_export_payload()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            csv_path = f"{base_path}_Spot_Data.csv"
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Item", "Value", "Unit"])
                for row in self._export_payload_rows(payload):
                    writer.writerow(row)

            exported_imgs = []
            short_panel = getattr(self, "short_heatmap_panel", None)
            long_panel = getattr(self, "long_heatmap_panel", None)
            # 匯出前確保拼貼面板存在並有實際尺寸（不必讓使用者先開視窗）
            collage_dlg = self._ensure_collage_dialog()
            collage_was_visible = collage_dlg.isVisible()
            if not collage_was_visible:
                collage_dlg.show()
                QApplication.processEvents()
            self._refresh_collage_heatmap_display(reset_levels=False)
            collage_panel = getattr(self, "collage_heatmap_panel", None)
            # 匯出圖不含十字與搜尋範圍；畫面標記匯出後還原
            self._hide_spot_overlays_for_export()
            try:
                if short_panel is not None:
                    short_base = f"{base_path}_Short_Heatmap_With_Profiles"
                    outs = short_panel.export_plot_bundle(short_base) or []
                    exported_imgs.extend(outs)
                if long_panel is not None:
                    long_base = f"{base_path}_Long_Heatmap_With_Profiles"
                    outs = long_panel.export_plot_bundle(long_base) or []
                    exported_imgs.extend(outs)
                if collage_panel is not None and self.batch_collage_matrix is not None:
                    collage_base = f"{base_path}_Collage_Heatmap_With_Profiles"
                    outs = collage_panel.export_plot_bundle(collage_base) or []
                    exported_imgs.extend(outs)
            finally:
                self._redraw_spot_overlays()
                if not collage_was_visible:
                    collage_dlg.hide()

            msg = (
                f"已匯出：\n"
                f"JSON：{path}\n"
                f"CSV：{csv_path}\n"
            )
            if exported_imgs:
                msg += "圖片：\n" + "\n".join(exported_imgs)
            else:
                msg += "（Heatmap 圖片未產出，請確認顯示面板）"
            QMessageBox.information(self, "匯出完成", msg)
            self.lbl_batch_status.setText("狀態: 匯出完成")
            self.lbl_batch_status.setStyleSheet(
                "color: #2E7D32; font-weight: bold; font-size: 12px;"
            )
        except Exception as exc:
            self._redraw_spot_overlays()
            QMessageBox.critical(self, "匯出失敗", f"無法匯出：\n{exc}")
            self.lbl_batch_status.setText("狀態: 匯出失敗")
            self.lbl_batch_status.setStyleSheet(
                "color: #C62828; font-weight: bold; font-size: 12px;"
            )


if __name__ == "__main__":
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
    window.show()
    sys.exit(app.exec_())
