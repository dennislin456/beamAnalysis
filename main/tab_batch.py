import os
import json
import csv
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as pg_export
from scipy.ndimage import uniform_filter, label as ndi_label, distance_transform_edt

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QMessageBox, QFrame, 
                             QSplitter, QSizePolicy, QRadioButton, QButtonGroup, 
                             QScrollArea, QCheckBox, QComboBox, QApplication,
                             QShortcut, QDialog, QDialogButtonBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

import openpyxl

from shared_components import (NoWheelSpinBox, NoWheelDoubleSpinBox, 
                               HeatmapViewerWindow, ContourBatchViewerWindow, 
                               CrossProfileViewerWindow,
                               compute_auto_spot_center, build_robust_threshold_mask,
                               estimate_border_background, split_y_index)


def _natural_sort_key(name):
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(name))]


class LocationConfigDialog(QDialog):
    """位置與 Cycle 配置彈窗：可勾選位置資料夾，並指定各位置要用哪些 cycle。"""

    def __init__(self, locations, pairs_by_loc, current_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("位置配置")
        self.resize(560, 520)
        self.setModal(True)

        self.locations = list(locations)
        self.pairs_by_loc = pairs_by_loc or {}
        self.current_config = current_config or {}
        self.loc_checks = {}
        self.cycle_checks = {}  # loc -> {filename: QCheckBox}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        tip = QLabel("勾選要計算的位置，並指定該位置要使用的 cycle（對應資料夾內的檔名）。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #546E7A; font-size: 12px;")
        root.addWidget(tip)

        top_btns = QHBoxLayout()
        btn_all_loc = QPushButton("全選位置")
        btn_all_loc.clicked.connect(lambda: self._set_all_locations(True))
        top_btns.addWidget(btn_all_loc)
        btn_none_loc = QPushButton("全不選位置")
        btn_none_loc.clicked.connect(lambda: self._set_all_locations(False))
        top_btns.addWidget(btn_none_loc)
        btn_all_cyc = QPushButton("全部位置選滿 cycle")
        btn_all_cyc.clicked.connect(self._select_all_cycles_for_enabled)
        top_btns.addWidget(btn_all_cyc)
        top_btns.addStretch()
        root.addLayout(top_btns)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #d0d0d0; border-radius: 4px; background: #FAFAFA; }"
        )
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(8, 8, 8, 8)
        host_layout.setSpacing(10)

        if not self.locations:
            host_layout.addWidget(QLabel("尚無可配置的位置，請先匯入 M1／M2。"))
        else:
            for loc in self.locations:
                pairs = self.pairs_by_loc.get(loc, [])
                filenames = [p["filename"] for p in pairs]
                prev = self.current_config.get(loc, {})
                prev_enabled = bool(prev.get("enabled", False))
                prev_cycles = set(prev.get("cycles", []))

                box = QGroupBox()
                box_layout = QVBoxLayout(box)
                box_layout.setContentsMargins(8, 6, 8, 8)
                box_layout.setSpacing(6)

                loc_chk = QCheckBox(f"位置：{loc}　（共 {len(filenames)} 個 cycle）")
                loc_chk.setStyleSheet("font-weight: bold; font-size: 13px;")
                loc_chk.setChecked(prev_enabled)
                loc_chk.toggled.connect(lambda checked, L=loc: self._on_loc_toggled(L, checked))
                self.loc_checks[loc] = loc_chk
                box_layout.addWidget(loc_chk)

                cyc_row_wrap = QWidget()
                cyc_layout = QHBoxLayout(cyc_row_wrap)
                cyc_layout.setContentsMargins(18, 0, 0, 0)
                cyc_layout.setSpacing(4)

                self.cycle_checks[loc] = {}
                for fname in filenames:
                    stem = os.path.splitext(fname)[0]
                    cyc_chk = QCheckBox(stem)
                    cyc_chk.setToolTip(fname)
                    # 若位置曾啟用：沿用先前 cycle；若首次啟用預設全選
                    if prev_enabled:
                        cyc_chk.setChecked(fname in prev_cycles if prev_cycles else True)
                    else:
                        cyc_chk.setChecked(True)
                    cyc_chk.setEnabled(prev_enabled)
                    self.cycle_checks[loc][fname] = cyc_chk
                    cyc_layout.addWidget(cyc_chk)

                cyc_layout.addStretch()
                box_layout.addWidget(cyc_row_wrap)

                cyc_btn_row = QHBoxLayout()
                cyc_btn_row.setContentsMargins(18, 0, 0, 0)
                btn_c_all = QPushButton("此位置全選 cycle")
                btn_c_all.setFixedHeight(26)
                btn_c_all.clicked.connect(lambda _=False, L=loc: self._set_cycles(L, True))
                cyc_btn_row.addWidget(btn_c_all)
                btn_c_none = QPushButton("此位置清空 cycle")
                btn_c_none.setFixedHeight(26)
                btn_c_none.clicked.connect(lambda _=False, L=loc: self._set_cycles(L, False))
                cyc_btn_row.addWidget(btn_c_none)
                cyc_btn_row.addStretch()
                box_layout.addLayout(cyc_btn_row)

                host_layout.addWidget(box)

        host_layout.addStretch()
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("font-weight: bold; color: #E65100; font-size: 12px;")
        root.addWidget(self.lbl_summary)
        self._refresh_summary()

        # 勾選 cycle 變更時更新摘要
        for loc_map in self.cycle_checks.values():
            for chk in loc_map.values():
                chk.toggled.connect(self._refresh_summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("套用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_loc_toggled(self, loc, checked):
        for chk in self.cycle_checks.get(loc, {}).values():
            chk.setEnabled(checked)
        self._refresh_summary()

    def _set_all_locations(self, checked):
        for loc, chk in self.loc_checks.items():
            chk.setChecked(checked)

    def _set_cycles(self, loc, checked):
        for chk in self.cycle_checks.get(loc, {}).values():
            chk.setChecked(checked)
        self._refresh_summary()

    def _select_all_cycles_for_enabled(self):
        for loc, loc_chk in self.loc_checks.items():
            if loc_chk.isChecked():
                self._set_cycles(loc, True)

    def _refresh_summary(self, *_args):
        n_loc = 0
        n_cyc = 0
        for loc, loc_chk in self.loc_checks.items():
            if not loc_chk.isChecked():
                continue
            n_loc += 1
            n_cyc += sum(1 for c in self.cycle_checks.get(loc, {}).values() if c.isChecked())
        self.lbl_summary.setText(f"目前選擇：{n_loc} 個位置｜{n_cyc} 組 cycle")

    def get_config(self):
        """回傳 {location: {"enabled": bool, "cycles": [filename, ...]}}"""
        cfg = {}
        for loc in self.locations:
            enabled = self.loc_checks[loc].isChecked()
            cycles = [
                fname for fname, chk in self.cycle_checks.get(loc, {}).items()
                if chk.isChecked()
            ]
            # 依自然排序
            cycles = sorted(cycles, key=_natural_sort_key)
            cfg[loc] = {"enabled": enabled, "cycles": cycles}
        return cfg


class DataRayBatchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # --- Batch 專屬變數初始化 ---
        self.save_dir_path = ""
        self.matrix1 = None
        self.matrix2 = None
        self.batch_result_matrix = None
        
        self.batch_m1_files = []
        self.batch_m2_files = []
        self.batch_m1_root = ""
        self.batch_m2_root = ""
        self.batch_pairs = []  # 目前已勾選位置＋cycle 的量測組
        self.batch_all_pairs_by_loc = {}  # {location: [pair, ...]}
        self.batch_available_locations = []
        self.batch_location_config = {}  # {loc: {"enabled": bool, "cycles": [filename]}}
        self.batch_current_idx = 0
        self.batch_total_count = 0
        self.batch_saved_params = {}
        # 切換加速：預載矩陣與運算結果快取
        self.batch_matrix_cache = {}   # idx -> (matrix1, matrix2)
        self.batch_result_cache = {}   # (idx, mode, normalize) -> (result, scale_info)
        
        # 彈出視窗與十字標記預留變數
        self.viewer_batch_m1_win = None
        self.viewer_batch_m2_win = None
        self.contour_batch_win = None
        self.cross_batch_profile_win = None
        
        self.batch_m1_center_point = None
        self.batch_m2_center_point = None      # M2 below（M1 Y 以下）
        self.batch_m2_above_point = None       # M2 above（M1 Y 以上）
        self.batch_m2_below_circle_r = None    # 內切圓半徑（below），僅 inscribed 模式
        self.batch_m2_above_circle_r = None    # 內切圓半徑（above），僅 inscribed 模式
        self.batch_cross_items = [] # 用來存放畫在 Heatmap 上的十字線
        self.batch_scale_info = None  # {max1_val, match2_val, scale_ratio, max1_idx}

        # 建立 UI
        self.setup_ui()

    def setup_ui(self):
        btn_style_folder = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #EF6C00; border: none; border-radius: 5px; padding: 6px 12px; }
            QPushButton:hover { background-color: #F57C00; }
        """
        btn_style_view = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #7B1FA2; border: none; border-radius: 5px; padding: 6px 14px; }
            QPushButton:hover { background-color: #8E24AA; }
            QPushButton:disabled { background-color: #E0E0E0; color: #A0A0A0; }
        """
        btn_style_primary = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #2E7D32; border: none; border-radius: 5px; padding: 8px 12px; }
            QPushButton:hover { background-color: #388E3C; }
        """
        btn_style_export = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #0288D1; border: none; border-radius: 5px; padding: 8px 16px; }
            QPushButton:hover { background-color: #039BE5; }
            QPushButton:disabled { background-color: #B0BEC5; }
        """
        btn_style_cross = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #00897B; border: none; border-radius: 5px; padding: 6px 14px; }
            QPushButton:hover { background-color: #009688; }
            QPushButton:disabled { background-color: #E0E0E0; color: #A0A0A0; }
        """

        batch_layout = QHBoxLayout(self)
        batch_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #dcdcdc; width: 4px; }")
        batch_layout.addWidget(splitter)

        # ---------------------------------------------------------
        # 左側控制區
        # ---------------------------------------------------------
        left_container = QWidget()
        left_container.setMinimumWidth(320)
        left_container.setMaximumWidth(360)
        left_outer_layout = QVBoxLayout(left_container)
        left_outer_layout.setContentsMargins(0, 0, 0, 0)

        # 上方固定控制區
        top_fixed_widget = QWidget()
        top_fixed_layout = QVBoxLayout(top_fixed_widget)
        top_fixed_layout.setContentsMargins(12, 12, 12, 12)

        lbl_b_mode = QLabel("選擇工作模式 (Batch)：")
        lbl_b_mode.setStyleSheet("font-weight: bold;")
        top_fixed_layout.addWidget(lbl_b_mode)

        self.combo_batch_mode = QComboBox()
        self.combo_batch_mode.addItem("雙檔峰值校正 (R = M1 - k * M2)", "calc")
        self.combo_batch_mode.addItem("雙檔純相減畫圖 (M1 - M2)", "sub")
        self.combo_batch_mode.addItem("雙檔純相除畫圖 (M1 / M2)", "div")
        self.combo_batch_mode.currentIndexChanged.connect(self.on_batch_mode_or_norm_changed)
        top_fixed_layout.addWidget(self.combo_batch_mode)

        self.chk_batch_normalize = QCheckBox("Normalize")
        self.chk_batch_normalize.toggled.connect(self.on_batch_mode_or_norm_changed)
        top_fixed_layout.addWidget(self.chk_batch_normalize)

        top_fixed_layout.addWidget(self._create_hline())

        self.btn_batch_m1_dir = QPushButton("I. 選擇 M1 主資料夾（含位置子資料夾）")
        self.btn_batch_m1_dir.setStyleSheet(btn_style_folder)
        self.btn_batch_m1_dir.clicked.connect(self.load_batch_m1_folder)
        top_fixed_layout.addWidget(self.btn_batch_m1_dir)

        self.lbl_batch_m1_info = QLabel("未選擇 M1 主資料夾\n格式: M1/<位置>/<檔名>.xlsx")
        self.lbl_batch_m1_info.setStyleSheet("font-size: 11px; color: #546E7A;")
        self.lbl_batch_m1_info.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_m1_info)

        self.btn_batch_m2_dir = QPushButton("II. 選擇 M2 主資料夾（含位置子資料夾）")
        self.btn_batch_m2_dir.setStyleSheet(btn_style_folder)
        self.btn_batch_m2_dir.clicked.connect(self.load_batch_m2_folder)
        top_fixed_layout.addWidget(self.btn_batch_m2_dir)

        self.lbl_batch_m2_info = QLabel("未選擇 M2 主資料夾\n格式: M2/<位置>/<檔名>.xlsx")
        self.lbl_batch_m2_info.setStyleSheet("font-size: 11px; color: #546E7A;")
        self.lbl_batch_m2_info.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_m2_info)

        self.lbl_batch_pair_info = QLabel("配對結果: 尚未選取雙邊資料夾")
        self.lbl_batch_pair_info.setStyleSheet("font-size: 11px; color: #1565C0; font-weight: bold;")
        self.lbl_batch_pair_info.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_pair_info)

        self.btn_location_config = QPushButton("位置配置")
        self.btn_location_config.setStyleSheet("""
            QPushButton {
                font-size: 13px; font-weight: bold; color: white;
                background-color: #5E35B1; border: none; border-radius: 5px; padding: 8px 12px;
            }
            QPushButton:hover { background-color: #7E57C2; }
            QPushButton:disabled { background-color: #B0BEC5; }
        """)
        self.btn_location_config.setEnabled(False)
        self.btn_location_config.clicked.connect(self.open_location_config_dialog)
        top_fixed_layout.addWidget(self.btn_location_config)

        self.lbl_batch_selected_info = QLabel("已配置: 0 個位置｜0 組 cycle")
        self.lbl_batch_selected_info.setStyleSheet("font-size: 11px; color: #E65100; font-weight: bold;")
        self.lbl_batch_selected_info.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_selected_info)

        top_fixed_layout.addWidget(self._create_hline())

        self.btn_batch_run = QPushButton("載入已配置位置並運算")
        self.btn_batch_run.setStyleSheet(btn_style_primary)
        self.btn_batch_run.clicked.connect(self.process_batch_data)
        top_fixed_layout.addWidget(self.btn_batch_run)

        top_fixed_layout.addWidget(self._create_hline())

        lbl_status_title = QLabel("計算數據與狀態")
        lbl_status_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #37474F;")
        top_fixed_layout.addWidget(lbl_status_title)

        self.lbl_batch_status = QLabel("狀態: 等待匯入檔案")
        self.lbl_batch_status.setStyleSheet("color: #1565C0; font-weight: bold; font-size: 12px;")
        top_fixed_layout.addWidget(self.lbl_batch_status)

        self.lbl_batch_max1 = QLabel("M1 最大值(位置): --")
        self.lbl_batch_max1.setStyleSheet("font-size: 12px;")
        self.lbl_batch_max2 = QLabel("M2 同位置數值: --")
        self.lbl_batch_max2.setStyleSheet("font-size: 12px;")
        self.lbl_batch_ratio = QLabel("計算得出比例: --")
        self.lbl_batch_ratio.setStyleSheet("font-size: 12px;")
        self.lbl_batch_size = QLabel("矩陣大小: --")
        self.lbl_batch_size.setStyleSheet("font-size: 12px;")
        top_fixed_layout.addWidget(self.lbl_batch_max1)
        top_fixed_layout.addWidget(self.lbl_batch_max2)
        top_fixed_layout.addWidget(self.lbl_batch_ratio)
        top_fixed_layout.addWidget(self.lbl_batch_size)

        self.lbl_batch_distance_xy = QLabel("M1→M2(below) 差距: ΔX: -- px, ΔY: -- px")
        self.lbl_batch_distance_xy.setStyleSheet("font-weight: bold; color: #1565C0; font-size: 12px;")
        self.lbl_batch_distance_xy.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_distance_xy)

        self.lbl_batch_distance_total = QLabel("M1→M2(below) 總距離: -- px")
        self.lbl_batch_distance_total.setStyleSheet("font-weight: bold; color: #1565C0; font-size: 12px;")
        self.lbl_batch_distance_total.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_distance_total)

        self.lbl_batch_real_distance_xy = QLabel("M1→M2(below) 實際 (*5.5): ΔX: -- μm, ΔY: -- μm")
        self.lbl_batch_real_distance_xy.setStyleSheet("font-weight: bold; color: #2962FF; font-size: 12px;")
        self.lbl_batch_real_distance_xy.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_real_distance_xy)

        self.lbl_batch_real_distance_total = QLabel("M1→M2(below) 實際總距離: -- μm")
        self.lbl_batch_real_distance_total.setStyleSheet("font-weight: bold; color: #2962FF; font-size: 12px;")
        self.lbl_batch_real_distance_total.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_real_distance_total)

        self.lbl_batch_ab_distance_xy = QLabel("M2(above)→M2(below) 差距: ΔX: -- px, ΔY: -- px")
        self.lbl_batch_ab_distance_xy.setStyleSheet("font-weight: bold; color: #C62828; font-size: 12px;")
        self.lbl_batch_ab_distance_xy.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_ab_distance_xy)

        self.lbl_batch_ab_distance_total = QLabel("M2(above)→M2(below) 總距離: -- px")
        self.lbl_batch_ab_distance_total.setStyleSheet("font-weight: bold; color: #C62828; font-size: 12px;")
        self.lbl_batch_ab_distance_total.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_ab_distance_total)

        self.lbl_batch_ab_real_distance_xy = QLabel("M2(above)→M2(below) 實際 (*5.5): ΔX: -- μm, ΔY: -- μm")
        self.lbl_batch_ab_real_distance_xy.setStyleSheet("font-weight: bold; color: #D50000; font-size: 12px;")
        self.lbl_batch_ab_real_distance_xy.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_ab_real_distance_xy)

        self.lbl_batch_ab_real_distance_total = QLabel("M2(above)→M2(below) 實際總距離: -- μm")
        self.lbl_batch_ab_real_distance_total.setStyleSheet("font-weight: bold; color: #D50000; font-size: 12px;")
        self.lbl_batch_ab_real_distance_total.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_ab_real_distance_total)

        # 左側也可切換群組（與右側、左右鍵同步）
        nav_row = QHBoxLayout()
        self.btn_batch_prev_left = QPushButton("◀ 上一組")
        self.btn_batch_prev_left.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; background-color: #E0E0E0; "
            "border-radius: 4px; padding: 4px 8px; } QPushButton:hover { background-color: #BDBDBD; }"
        )
        self.btn_batch_prev_left.clicked.connect(self.batch_go_prev)
        nav_row.addWidget(self.btn_batch_prev_left)

        self.lbl_batch_group_nav = QLabel("0 / 0")
        self.lbl_batch_group_nav.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #D81B60; padding: 0 6px;")
        self.lbl_batch_group_nav.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_batch_group_nav)

        self.btn_batch_next_left = QPushButton("下一組 ▶")
        self.btn_batch_next_left.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; background-color: #E0E0E0; "
            "border-radius: 4px; padding: 4px 8px; } QPushButton:hover { background-color: #BDBDBD; }"
        )
        self.btn_batch_next_left.clicked.connect(self.batch_go_next)
        nav_row.addWidget(self.btn_batch_next_left)
        top_fixed_layout.addLayout(nav_row)

        self.lbl_batch_current_files = QLabel("目前檔案: --")
        self.lbl_batch_current_files.setStyleSheet("font-size: 11px; color: #546E7A;")
        self.lbl_batch_current_files.setWordWrap(True)
        top_fixed_layout.addWidget(self.lbl_batch_current_files)

        left_outer_layout.addWidget(top_fixed_widget)

        # 左右鍵切換群組（與單次自動計算資料面板同步）
        self.shortcut_batch_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_batch_prev.activated.connect(self.batch_go_prev)
        self.shortcut_batch_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_batch_next.activated.connect(self.batch_go_next)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_batch_focus_changed)

        # 下方捲動設定區
        left_scroll_panel = QScrollArea()
        left_scroll_panel.setWidgetResizable(True)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        left_layout.addWidget(self._create_hline())

        lbl_center_mode = QLabel("M1 / 雙點定位模式 (Batch)：")
        lbl_center_mode.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(lbl_center_mode)

        lbl_m1_pt = QLabel("第一點（M1 Heatmap）：")
        lbl_m1_pt.setStyleSheet("font-weight: bold; color: #2E7D32;")
        left_layout.addWidget(lbl_m1_pt)

        self.batch_m1_point_group = QButtonGroup(self)
        self.radio_batch_m1_centroid = QRadioButton("自動抓取 (質心中心)")
        self.radio_batch_m1_centroid.setChecked(True)
        self.radio_batch_m1_centroid.toggled.connect(self.on_batch_m1_point_mode_changed)
        self.batch_m1_point_group.addButton(self.radio_batch_m1_centroid)
        left_layout.addWidget(self.radio_batch_m1_centroid)

        self.radio_batch_m1_thresh_geom = QRadioButton("自動抓取 (門檻區域幾何中心)")
        self.radio_batch_m1_thresh_geom.toggled.connect(self.on_batch_m1_point_mode_changed)
        self.batch_m1_point_group.addButton(self.radio_batch_m1_thresh_geom)
        left_layout.addWidget(self.radio_batch_m1_thresh_geom)

        self.radio_batch_m1_peak_geom = QRadioButton("自動抓取 (最高值區域幾何中心)")
        self.radio_batch_m1_peak_geom.toggled.connect(self.on_batch_m1_point_mode_changed)
        self.batch_m1_point_group.addButton(self.radio_batch_m1_peak_geom)
        left_layout.addWidget(self.radio_batch_m1_peak_geom)

        self.radio_batch_m1_manual = QRadioButton("手動抓取 (點擊 M1 影像)")
        self.radio_batch_m1_manual.toggled.connect(self.on_batch_m1_point_mode_changed)
        self.batch_m1_point_group.addButton(self.radio_batch_m1_manual)
        left_layout.addWidget(self.radio_batch_m1_manual)

        self.chk_batch_m1_use_threshold = QCheckBox("使用門檻（第一點）")
        self.chk_batch_m1_use_threshold.setChecked(True)
        self.chk_batch_m1_use_threshold.setStyleSheet("color: #2E7D32;")
        self.chk_batch_m1_use_threshold.toggled.connect(self.on_batch_m1_threshold_toggled)
        left_layout.addWidget(self.chk_batch_m1_use_threshold)

        layout_m1_thresh = QHBoxLayout()
        self.lbl_batch_m1_thresh_spin = QLabel("第一點門檻比例 (%):")
        self.spin_batch_m1_thresh_percent = NoWheelDoubleSpinBox()
        self.spin_batch_m1_thresh_percent.setRange(0.1, 100.0)
        self.spin_batch_m1_thresh_percent.setValue(50.0)
        self.spin_batch_m1_thresh_percent.setSingleStep(1.0)
        self.spin_batch_m1_thresh_percent.setDecimals(1)
        self.spin_batch_m1_thresh_percent.valueChanged.connect(self.on_batch_m1_thresh_percent_changed)
        layout_m1_thresh.addWidget(self.lbl_batch_m1_thresh_spin)
        layout_m1_thresh.addWidget(self.spin_batch_m1_thresh_percent)
        left_layout.addLayout(layout_m1_thresh)

        self.chk_batch_m1_show_thresh = QCheckBox("顯示門檻區域於 M1 圖")
        self.chk_batch_m1_show_thresh.setChecked(True)
        self.chk_batch_m1_show_thresh.setStyleSheet("color: #2E7D32;")
        self.chk_batch_m1_show_thresh.toggled.connect(self.update_batch_calculations)
        left_layout.addWidget(self.chk_batch_m1_show_thresh)

        left_layout.addWidget(self._create_hline())

        lbl_m2_pt = QLabel("第二點（Process Result Heatmap）：")
        lbl_m2_pt.setStyleSheet("font-weight: bold; color: #1565C0;")
        left_layout.addWidget(lbl_m2_pt)

        self.batch_m2_point_group = QButtonGroup(self)

        self.radio_batch_p2_auto_min = QRadioButton("自動抓取 (第一點 Y 以下最小值)")
        self.radio_batch_p2_auto_min.setChecked(True)
        self.radio_batch_p2_auto_min.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_p2_auto_min)
        left_layout.addWidget(self.radio_batch_p2_auto_min)

        self.radio_batch_p2_m2_thresh_geom = QRadioButton("自動抓取 (M2 第一點 Y 以下門檻幾何中心)")
        self.radio_batch_p2_m2_thresh_geom.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_p2_m2_thresh_geom)
        left_layout.addWidget(self.radio_batch_p2_m2_thresh_geom)

        self.radio_batch_p2_m2_centroid = QRadioButton("自動抓取 (M2 第一點 Y 以下質心中心)")
        self.radio_batch_p2_m2_centroid.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_p2_m2_centroid)
        left_layout.addWidget(self.radio_batch_p2_m2_centroid)

        self.radio_batch_p2_m2_inscribed = QRadioButton(
            "自動抓取 (M2 門檻 contour 內切圓中心，Y 上／下)"
        )
        self.radio_batch_p2_m2_inscribed.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_p2_m2_inscribed)
        left_layout.addWidget(self.radio_batch_p2_m2_inscribed)

        self.radio_batch_p2_manual = QRadioButton("手動抓取 (點擊 Process 影像)")
        self.radio_batch_p2_manual.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_p2_manual)
        left_layout.addWidget(self.radio_batch_p2_manual)

        self.chk_batch_p2_use_threshold = QCheckBox("使用門檻（第二點／M2）")
        self.chk_batch_p2_use_threshold.setChecked(True)
        self.chk_batch_p2_use_threshold.setStyleSheet("color: #1565C0;")
        self.chk_batch_p2_use_threshold.toggled.connect(self.on_batch_p2_threshold_toggled)
        left_layout.addWidget(self.chk_batch_p2_use_threshold)

        layout_m2_thresh = QHBoxLayout()
        self.lbl_batch_p2_thresh_spin = QLabel("第二點門檻比例 (%):")
        self.spin_batch_p2_thresh_percent = NoWheelDoubleSpinBox()
        self.spin_batch_p2_thresh_percent.setRange(0.1, 100.0)
        self.spin_batch_p2_thresh_percent.setValue(50.0)
        self.spin_batch_p2_thresh_percent.setSingleStep(1.0)
        self.spin_batch_p2_thresh_percent.setDecimals(1)
        self.spin_batch_p2_thresh_percent.valueChanged.connect(self.on_batch_p2_thresh_percent_changed)
        layout_m2_thresh.addWidget(self.lbl_batch_p2_thresh_spin)
        layout_m2_thresh.addWidget(self.spin_batch_p2_thresh_percent)
        left_layout.addLayout(layout_m2_thresh)

        lbl_m2_algo_hint = QLabel("質心／門檻幾何：背景扣除＋最大連通區＋亞像素（預設門檻 50%）")
        lbl_m2_algo_hint.setStyleSheet("color: #546E7A; font-size: 11px;")
        lbl_m2_algo_hint.setWordWrap(True)
        left_layout.addWidget(lbl_m2_algo_hint)

        self.chk_batch_p2_show_thresh = QCheckBox("顯示門檻區域於 M2 圖（半透明，含 Y 上／下）")
        self.chk_batch_p2_show_thresh.setChecked(True)
        self.chk_batch_p2_show_thresh.setStyleSheet("color: #1565C0;")
        self.chk_batch_p2_show_thresh.toggled.connect(self.update_batch_calculations)
        left_layout.addWidget(self.chk_batch_p2_show_thresh)

        left_layout.addWidget(self._create_hline())

        # 加入 Batch 專屬十字開關
        self.chk_batch_show_cross = QCheckBox("顯示 M1(綠虛線) / M2-below(藍) / M2-above(紅) 十字")
        self.chk_batch_show_cross.setChecked(True)
        self.chk_batch_show_cross.setStyleSheet("font-weight: bold; color: #1565C0;")
        self.chk_batch_show_cross.toggled.connect(self.redraw_batch_crosses)
        left_layout.addWidget(self.chk_batch_show_cross)

        self.chk_batch_heatmap_gray = QCheckBox("熱力圖改為黑白（方便對照十字／圓）")
        self.chk_batch_heatmap_gray.setChecked(False)
        self.chk_batch_heatmap_gray.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_batch_heatmap_gray.toggled.connect(self.on_batch_heatmap_colormap_toggled)
        left_layout.addWidget(self.chk_batch_heatmap_gray)

        left_layout.addWidget(self._create_hline())

        self.btn_batch_save_params = QPushButton("💾 暫存此組參數與位置")
        self.btn_batch_save_params.setStyleSheet("""
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #D81B60; border: none; border-radius: 5px; padding: 8px 12px; }
            QPushButton:hover { background-color: #E91E63; }
        """)
        self.btn_batch_save_params.clicked.connect(self.save_current_batch_params)
        left_layout.addWidget(self.btn_batch_save_params)

        left_layout.addStretch()
        left_scroll_panel.setWidget(left_content)
        left_outer_layout.addWidget(left_scroll_panel)
        splitter.addWidget(left_container)

        # ---------------------------------------------------------
        # 右側主繪圖與控制區
        # ---------------------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 6)

        self.btn_batch_select_dir = QPushButton("選擇儲存資料夾")
        self.btn_batch_select_dir.setStyleSheet(btn_style_folder)
        self.btn_batch_select_dir.setMinimumHeight(38)
        self.btn_batch_select_dir.clicked.connect(self.select_save_directory)
        top_bar.addWidget(self.btn_batch_select_dir)

        self.lbl_batch_dir_path = QLabel("")
        self.lbl_batch_dir_path.setStyleSheet("color: #333333; background-color: #f5f5f5; border: 1px solid #d0d0d0; border-radius: 4px; padding: 6px 10px; font-size: 12px;")
        self.lbl_batch_dir_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lbl_batch_dir_path.setMinimumHeight(38)
        top_bar.addWidget(self.lbl_batch_dir_path)

        self.btn_batch_contour_popup = QPushButton("📈")
        self.btn_batch_contour_popup.setFixedSize(38, 38)
        self.btn_batch_contour_popup.setStyleSheet("QPushButton { font-size: 16px; background-color: #7B1FA2; color: white; border-radius: 4px; } QPushButton:hover { background-color: #8E24AA; }")
        self.btn_batch_contour_popup.setToolTip("點擊彈出 Contour 與 Waveform 視窗")
        self.btn_batch_contour_popup.clicked.connect(self.show_batch_contour_window)
        top_bar.addWidget(self.btn_batch_contour_popup)

        self.btn_batch_view_cross = QPushButton("查看十字波形視窗")
        self.btn_batch_view_cross.setStyleSheet(btn_style_cross)
        self.btn_batch_view_cross.setMinimumHeight(38)
        self.btn_batch_view_cross.setEnabled(False)
        self.btn_batch_view_cross.clicked.connect(self.show_batch_cross_profile)
        top_bar.addWidget(self.btn_batch_view_cross)

        self.btn_batch_view_m1 = QPushButton("查看 M1 Heatmap")
        self.btn_batch_view_m1.setStyleSheet(btn_style_view)
        self.btn_batch_view_m1.setMinimumHeight(38)
        self.btn_batch_view_m1.setEnabled(False)
        self.btn_batch_view_m1.clicked.connect(self.show_batch_m1_heatmap)
        top_bar.addWidget(self.btn_batch_view_m1)

        self.btn_batch_view_m2 = QPushButton("查看 M2 Heatmap")
        self.btn_batch_view_m2.setStyleSheet(btn_style_view)
        self.btn_batch_view_m2.setMinimumHeight(38)
        self.btn_batch_view_m2.setEnabled(False)
        self.btn_batch_view_m2.clicked.connect(self.show_batch_m2_heatmap)
        top_bar.addWidget(self.btn_batch_view_m2)

        self.btn_batch_export = QPushButton("一鍵匯出 (ZIP 壓縮包)")
        self.btn_batch_export.setStyleSheet(btn_style_export)
        self.btn_batch_export.setMinimumHeight(38)
        self.btn_batch_export.setFixedWidth(240)
        self.btn_batch_export.setEnabled(False)
        self.btn_batch_export.clicked.connect(self.export_batch_results_zip)
        top_bar.addWidget(self.btn_batch_export)

        right_layout.addLayout(top_bar)

        self.win_batch_top = pg.GraphicsLayoutWidget()
        self.win_batch_top.setStyleSheet("border: 1px solid #d0d0d0; background-color: black;")
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        pos = np.linspace(0.0, 1.0, len(colors))
        self.batch_jet_map = pg.ColorMap(pos, colors)
        self.batch_gray_map = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])

        self.plot_batch_heat = self.win_batch_top.addPlot(row=0, col=0, title='Processed Matrix Result Heatmap (Batch)')
        self.plot_batch_heat.getViewBox().invertY(False)
        self.plot_batch_heat.setAspectLocked(True)
        self.plot_batch_heat.setLabel('bottom', 'X Pixels')
        self.plot_batch_heat.setLabel('left', 'Y Pixels')

        self.batch_image_item = pg.ImageItem()
        self.plot_batch_heat.addItem(self.batch_image_item)
        self.plot_batch_heat.scene().sigMouseClicked.connect(self.on_batch_process_mouse_clicked)

        self.batch_hist = pg.HistogramLUTItem()
        self.batch_hist.setImageItem(self.batch_image_item)
        self.batch_hist.gradient.setColorMap(self.batch_jet_map)
        self.batch_hist.sigLevelsChanged.connect(self.on_colorbar_levels_changed)
        self.win_batch_top.addItem(self.batch_hist, row=0, col=1)

        right_layout.addWidget(self.win_batch_top, 3)

        bottom_layout_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_layout_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.win_batch_sub = pg.GraphicsLayoutWidget()
        self.win_batch_sub.setStyleSheet("border: 1px solid #d0d0d0; background-color: black;")
        
        # 子圖 1: 顏色分布
        self.plot_batch_hist = self.win_batch_sub.addPlot(row=0, col=0, title="Peak Color / Value Distribution")
        self.plot_batch_hist.setLabel('bottom', 'Intensity Value')
        self.plot_batch_hist.setLabel('left', 'Pixel Count')
        self.plot_batch_hist.showGrid(x=True, y=True, alpha=0.3)
        
        # 子圖 2: Peak Row 趨勢
        self.plot_batch_trend = self.win_batch_sub.addPlot(row=0, col=1, title="Peak Row Trend")
        self.plot_batch_trend.setLabel('bottom', 'X Position (px)')
        self.plot_batch_trend.setLabel('left', 'Intensity')
        self.plot_batch_trend.showGrid(x=True, y=True, alpha=0.3)
        
        bottom_layout.addWidget(self.win_batch_sub, 4)

        group_switch_widget = QWidget()
        group_switch_layout = QVBoxLayout(group_switch_widget)
        group_switch_layout.setAlignment(Qt.AlignCenter)

        lbl_switch_title = QLabel("資料群組切換")
        lbl_switch_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #37474F;")
        group_switch_layout.addWidget(lbl_switch_title, alignment=Qt.AlignCenter)

        h_switch = QHBoxLayout()
        self.btn_batch_prev = QPushButton("◀")
        self.btn_batch_prev.setFixedSize(40, 40)
        self.btn_batch_prev.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; background-color: #E0E0E0; border-radius: 4px; } QPushButton:hover { background-color: #BDBDBD; }")
        self.btn_batch_prev.clicked.connect(self.batch_go_prev)
        h_switch.addWidget(self.btn_batch_prev)

        self.lbl_batch_group_num = QLabel("0 / 0")
        self.lbl_batch_group_num.setStyleSheet("font-size: 16px; font-weight: bold; color: #D81B60; padding: 0 10px;")
        h_switch.addWidget(self.lbl_batch_group_num)

        self.btn_batch_next = QPushButton("▶")
        self.btn_batch_next.setFixedSize(40, 40)
        self.btn_batch_next.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; background-color: #E0E0E0; border-radius: 4px; } QPushButton:hover { background-color: #BDBDBD; }")
        self.btn_batch_next.clicked.connect(self.batch_go_next)
        h_switch.addWidget(self.btn_batch_next)

        group_switch_layout.addLayout(h_switch)
        bottom_layout.addWidget(group_switch_widget, 1)

        right_layout.addWidget(bottom_layout_container, 2)
        splitter.addWidget(right_panel)

    # ---------------------------------------------------------
    # 邏輯層功能與事件
    # ---------------------------------------------------------
    def _create_hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #e0e0e0; margin-top: 2px; margin-bottom: 2px;")
        return line

    def select_save_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇儲存資料夾", "")
        if dir_path:
            self.save_dir_path = dir_path
            self.lbl_batch_dir_path.setText(f"{dir_path}")

    def _natural_sort_key(self, name):
        return _natural_sort_key(name)

    def _scan_location_files(self, root_dir):
        """掃描主資料夾 → {位置名: {檔名: 完整路徑}}。"""
        result = {}
        if not root_dir or not os.path.isdir(root_dir):
            return result
        try:
            entries = os.listdir(root_dir)
        except OSError:
            return result
        for entry in entries:
            loc_path = os.path.join(root_dir, entry)
            if not os.path.isdir(loc_path):
                continue
            files = {}
            try:
                for fname in os.listdir(loc_path):
                    if fname.startswith("~$"):
                        continue
                    lower = fname.lower()
                    if lower.endswith(".xlsx") or lower.endswith(".xls"):
                        files[fname] = os.path.join(loc_path, fname)
            except OSError:
                continue
            if files:
                result[entry] = files
        return result

    def _rebuild_batch_pairs(self):
        """掃描並辨識共同位置；實際運算清單由位置配置決定。"""
        self.batch_pairs = []
        self.batch_m1_files = []
        self.batch_m2_files = []
        self.batch_all_pairs_by_loc = {}
        self.batch_available_locations = []
        self.batch_matrix_cache.clear()
        self.batch_result_cache.clear()

        if not self.batch_m1_root or not self.batch_m2_root:
            if hasattr(self, "lbl_batch_pair_info"):
                self.lbl_batch_pair_info.setText("配對結果: 請分別選擇 M1 與 M2 主資料夾")
            if hasattr(self, "btn_location_config"):
                self.btn_location_config.setEnabled(False)
            self.batch_location_config = {}
            self._update_selected_location_info()
            return

        m1_map = self._scan_location_files(self.batch_m1_root)
        m2_map = self._scan_location_files(self.batch_m2_root)
        common_locs = sorted(set(m1_map.keys()) & set(m2_map.keys()), key=self._natural_sort_key)

        pairs_by_loc = {}
        total_pairs = 0
        for loc in common_locs:
            common_files = sorted(
                set(m1_map[loc].keys()) & set(m2_map[loc].keys()),
                key=self._natural_sort_key,
            )
            loc_pairs = []
            for fname in common_files:
                loc_pairs.append({
                    "location": loc,
                    "filename": fname,
                    "m1_path": m1_map[loc][fname],
                    "m2_path": m2_map[loc][fname],
                })
            if loc_pairs:
                pairs_by_loc[loc] = loc_pairs
                total_pairs += len(loc_pairs)

        self.batch_all_pairs_by_loc = pairs_by_loc
        self.batch_available_locations = list(pairs_by_loc.keys())

        # 保留舊配置中仍存在的位置；新位置預設未啟用
        new_cfg = {}
        for loc in self.batch_available_locations:
            old = self.batch_location_config.get(loc, {})
            valid_names = {p["filename"] for p in pairs_by_loc[loc]}
            old_cycles = [c for c in old.get("cycles", []) if c in valid_names]
            new_cfg[loc] = {
                "enabled": bool(old.get("enabled", False)),
                "cycles": old_cycles if old_cycles else sorted(valid_names, key=self._natural_sort_key),
            }
        self.batch_location_config = new_cfg

        only_m1 = sorted(set(m1_map.keys()) - set(m2_map.keys()), key=self._natural_sort_key)
        only_m2 = sorted(set(m2_map.keys()) - set(m1_map.keys()), key=self._natural_sort_key)
        msg = (
            f"辨識到 {len(self.batch_available_locations)} 個共同位置，"
            f"共 {total_pairs} 組量測（請點「位置配置」選擇位置與 cycle）"
        )
        if only_m1:
            msg += f"\n僅 M1 有: {', '.join(only_m1[:8])}{'...' if len(only_m1) > 8 else ''}"
        if only_m2:
            msg += f"\n僅 M2 有: {', '.join(only_m2[:8])}{'...' if len(only_m2) > 8 else ''}"
        if hasattr(self, "lbl_batch_pair_info"):
            self.lbl_batch_pair_info.setText(msg)
        if hasattr(self, "btn_location_config"):
            self.btn_location_config.setEnabled(bool(self.batch_available_locations))
        self._update_selected_location_info()

    def open_location_config_dialog(self):
        if not self.batch_available_locations:
            QMessageBox.warning(self, "警告", "尚無可配置的位置，請先匯入 M1／M2 主資料夾。")
            return
        dlg = LocationConfigDialog(
            self.batch_available_locations,
            self.batch_all_pairs_by_loc,
            current_config=self.batch_location_config,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self.batch_location_config = dlg.get_config()
            self._update_selected_location_info()

    def _get_selected_locations(self):
        selected = []
        for loc in self.batch_available_locations:
            cfg = self.batch_location_config.get(loc, {})
            if cfg.get("enabled") and cfg.get("cycles"):
                selected.append(loc)
        return selected

    def _update_selected_location_info(self, *_args):
        selected = self._get_selected_locations()
        n_pairs = 0
        detail_parts = []
        for loc in selected:
            cycles = self.batch_location_config.get(loc, {}).get("cycles", [])
            n_pairs += len(cycles)
            stems = [os.path.splitext(c)[0] for c in cycles]
            if len(stems) <= 6:
                cyc_txt = ",".join(stems)
            else:
                cyc_txt = ",".join(stems[:4]) + f"...(+{len(stems) - 4})"
            detail_parts.append(f"{loc}[{cyc_txt}]")
        if hasattr(self, "lbl_batch_selected_info"):
            summary = f"已配置: {len(selected)} 個位置｜{n_pairs} 組 cycle"
            if detail_parts:
                summary += "\n" + "；".join(detail_parts[:6])
                if len(detail_parts) > 6:
                    summary += " ..."
            self.lbl_batch_selected_info.setText(summary)

    def _apply_selected_pairs(self):
        """依位置配置（位置＋指定 cycle）組成要運算的 pairs。"""
        selected = self._get_selected_locations()
        pairs = []
        for loc in selected:
            wanted = set(self.batch_location_config.get(loc, {}).get("cycles", []))
            for p in self.batch_all_pairs_by_loc.get(loc, []):
                if p["filename"] in wanted:
                    pairs.append(p)
        self.batch_pairs = pairs
        self.batch_m1_files = [p["m1_path"] for p in pairs]
        self.batch_m2_files = [p["m2_path"] for p in pairs]
        return selected, pairs

    def load_batch_m1_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇 M1 主資料夾（內含位置子資料夾）", "")
        if not dir_path:
            return
        loc_map = self._scan_location_files(dir_path)
        if not loc_map:
            QMessageBox.warning(
                self, "警告",
                "此資料夾下找不到「位置子資料夾／Excel」結構。\n"
                "預期格式：M1/<位置名>/<檔名>.xlsx"
            )
            return
        self.batch_m1_root = dir_path
        n_files = sum(len(v) for v in loc_map.values())
        locs = sorted(loc_map.keys(), key=self._natural_sort_key)
        self.lbl_batch_m1_info.setText(
            f"{os.path.basename(dir_path)}｜位置 {len(locs)} 個｜檔案 {n_files} 筆\n"
            f"位置: {', '.join(locs[:10])}{'...' if len(locs) > 10 else ''}"
        )
        if not self.save_dir_path:
            self.save_dir_path = dir_path
            self.lbl_batch_dir_path.setText(f"{dir_path}")
        self._rebuild_batch_pairs()

    def load_batch_m2_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇 M2 主資料夾（內含位置子資料夾）", "")
        if not dir_path:
            return
        loc_map = self._scan_location_files(dir_path)
        if not loc_map:
            QMessageBox.warning(
                self, "警告",
                "此資料夾下找不到「位置子資料夾／Excel」結構。\n"
                "預期格式：M2/<位置名>/<檔名>.xlsx"
            )
            return
        self.batch_m2_root = dir_path
        n_files = sum(len(v) for v in loc_map.values())
        locs = sorted(loc_map.keys(), key=self._natural_sort_key)
        self.lbl_batch_m2_info.setText(
            f"{os.path.basename(dir_path)}｜位置 {len(locs)} 個｜檔案 {n_files} 筆\n"
            f"位置: {', '.join(locs[:10])}{'...' if len(locs) > 10 else ''}"
        )
        self._rebuild_batch_pairs()

    def process_batch_data(self):
        if not self.batch_m1_root or not self.batch_m2_root:
            QMessageBox.warning(self, "警告", "請先選擇 M1 與 M2 主資料夾！")
            return
        if not self.batch_available_locations:
            self._rebuild_batch_pairs()
        if not self.batch_available_locations:
            QMessageBox.warning(
                self, "警告",
                "沒有可配對的位置！\n"
                "請確認 M1／M2 下有相同名稱的位置子資料夾。"
            )
            return

        selected, pairs = self._apply_selected_pairs()
        if not selected:
            QMessageBox.warning(
                self, "警告",
                "尚未配置任何位置！\n請先點「位置配置」，勾選位置並指定 cycle。"
            )
            return
        if not pairs:
            QMessageBox.warning(
                self, "警告",
                "已啟用的位置沒有勾選任何 cycle！\n請回到「位置配置」勾選 cycle。"
            )
            return

        self.batch_total_count = len(pairs)
        self.batch_current_idx = 0
        self.batch_saved_params.clear()
        self.batch_matrix_cache.clear()
        self.batch_result_cache.clear()

        try:
            self.lbl_batch_status.setText(
                f"狀態: 正在預載 {len(selected)} 個位置（{self.batch_total_count} 組）..."
            )
            self.lbl_batch_status.setStyleSheet("color: #F57C00; font-weight: bold; font-size: 12px;")
            self.btn_batch_run.setEnabled(False)
            QApplication.processEvents()

            for i in range(self.batch_total_count):
                self._ensure_matrix_cached(i)
                if (i + 1) % 2 == 0 or i == self.batch_total_count - 1:
                    self.lbl_batch_status.setText(
                        f"狀態: 預載中... {i + 1}/{self.batch_total_count}"
                    )
                    QApplication.processEvents()

            self.btn_batch_run.setEnabled(True)
            self.load_batch_group(0)
            self.btn_batch_export.setEnabled(True)
            self.btn_batch_view_m1.setEnabled(True)
            self.btn_batch_view_m2.setEnabled(True)
            self.btn_batch_view_cross.setEnabled(True)
            self.lbl_batch_status.setText(
                f"狀態: 已載入 {len(selected)} 個位置｜共 {self.batch_total_count} 組"
            )
            self.lbl_batch_status.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 12px;")
        except Exception as e:
            self.btn_batch_run.setEnabled(True)
            self.lbl_batch_status.setText("狀態: 預載失敗")
            self.lbl_batch_status.setStyleSheet("color: #C62828; font-weight: bold; font-size: 12px;")
            QMessageBox.critical(self, "錯誤", f"預載失敗: {str(e)}")

    def _read_excel_matrix(self, path):
        df = pd.read_excel(path, header=None, skiprows=4)
        return df.dropna(how='all').astype(float).values

    def _ensure_matrix_cached(self, idx):
        if idx in self.batch_matrix_cache:
            return self.batch_matrix_cache[idx]
        m1 = self._read_excel_matrix(self.batch_m1_files[idx])
        m2 = self._read_excel_matrix(self.batch_m2_files[idx])
        self.batch_matrix_cache[idx] = (m1, m2)
        return m1, m2

    def _result_cache_key(self, idx):
        return (
            idx,
            self.combo_batch_mode.currentData(),
            bool(self.chk_batch_normalize.isChecked()),
        )

    def _compute_group_result(self, matrix1, matrix2):
        mode_data = self.combo_batch_mode.currentData()
        if self.chk_batch_normalize.isChecked():
            max1, max2 = np.max(matrix1), np.max(matrix2)
            m2_proc = matrix2 * (max1 / max2) if max2 != 0 else matrix2
        else:
            m2_proc = matrix2

        max1_idx = np.unravel_index(np.argmax(matrix1, axis=None), matrix1.shape)
        max1_val = float(matrix1[max1_idx])
        match2_val = float(m2_proc[max1_idx]) if m2_proc[max1_idx] != 0 else 1e-9
        scale_ratio = max1_val / match2_val if match2_val != 0 else 0.0

        if mode_data == "sub":
            result = matrix1 - m2_proc
            scale_info = {
                "mode": "sub",
                "max1_val": max1_val,
                "match2_val": float(matrix2[max1_idx]),
                "scale_ratio": None,
                "max1_idx": max1_idx,
            }
        elif mode_data == "div":
            safe_m2 = np.where(m2_proc == 0, 1e-9, m2_proc)
            result = matrix1 / safe_m2
            scale_info = {
                "mode": "div",
                "max1_val": max1_val,
                "match2_val": float(matrix2[max1_idx]),
                "scale_ratio": None,
                "max1_idx": max1_idx,
            }
        else:
            result = matrix1 - (m2_proc * scale_ratio)
            scale_info = {
                "mode": "calc",
                "max1_val": max1_val,
                "match2_val": match2_val,
                "scale_ratio": scale_ratio,
                "max1_idx": max1_idx,
            }
        return result, scale_info

    def on_batch_mode_or_norm_changed(self, *_args):
        """模式／Normalize 變更時清結果快取並重算目前組。"""
        self.batch_result_cache.clear()
        if self.batch_total_count > 0 and self.batch_matrix_cache:
            self.load_batch_group(self.batch_current_idx)

    def load_batch_group(self, idx):
        if idx < 0 or idx >= self.batch_total_count:
            return
        try:
            self.batch_current_idx = idx
            group_text = f"{idx + 1} / {self.batch_total_count}"
            self.lbl_batch_group_num.setText(group_text)
            if hasattr(self, "lbl_batch_group_nav"):
                self.lbl_batch_group_nav.setText(group_text)

            f1 = self.batch_m1_files[idx]
            f2 = self.batch_m2_files[idx]
            pair = self.batch_pairs[idx] if idx < len(self.batch_pairs) else None
            if hasattr(self, "lbl_batch_current_files"):
                if pair:
                    self.lbl_batch_current_files.setText(
                        f"目前量測:\n"
                        f"位置: {pair['location']} ｜ 檔名: {pair['filename']}\n"
                        f"M1: .../{pair['location']}/{pair['filename']}\n"
                        f"M2: .../{pair['location']}/{pair['filename']}"
                    )
                else:
                    self.lbl_batch_current_files.setText(
                        f"目前檔案:\nM1: {os.path.basename(f1)}\nM2: {os.path.basename(f2)}"
                    )

            # 優先使用記憶體快取，避免每次切換重讀 Excel
            self.matrix1, self.matrix2 = self._ensure_matrix_cached(idx)

            cache_key = self._result_cache_key(idx)
            cached = self.batch_result_cache.get(cache_key)
            if cached is not None:
                self.batch_result_matrix, self.batch_scale_info = cached
            else:
                result, scale_info = self._compute_group_result(self.matrix1, self.matrix2)
                self.batch_result_matrix = result
                self.batch_scale_info = scale_info
                self.batch_result_cache[cache_key] = (result, scale_info)

            self.batch_image_item.setImage(self.batch_result_matrix.T, autoLevels=False)
            self.batch_hist.setLevels(
                float(np.min(self.batch_result_matrix)),
                float(np.max(self.batch_result_matrix)),
            )

            self.lbl_batch_status.setText(f"狀態: 已載入第 {idx + 1}/{self.batch_total_count} 組")
            self.lbl_batch_status.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 12px;")

            self._apply_saved_batch_params(idx)
            self.update_batch_calculations(silent=True)
            self.render_sub_plots_fast(self.batch_result_matrix)

            # 若 M1/M2 視窗已開，同步換成當前組資料
            self._refresh_open_batch_viewers()
        except Exception as e:
            self.lbl_batch_status.setText("狀態: 載入失敗")
            self.lbl_batch_status.setStyleSheet("color: #C62828; font-weight: bold; font-size: 12px;")
            QMessageBox.critical(self, "錯誤", f"載入失敗: {str(e)}")

    def _refresh_open_batch_viewers(self):
        if self.matrix1 is not None and getattr(self, "viewer_batch_m1_win", None) is not None:
            try:
                if self.viewer_batch_m1_win.isVisible():
                    self.viewer_batch_m1_win.matrix_data = self.matrix1
                    self.viewer_batch_m1_win.image_item.setImage(self.matrix1.T, autoLevels=False)
                    min_v, max_v = float(np.min(self.matrix1)), float(np.max(self.matrix1))
                    self.viewer_batch_m1_win.hist.setLevels(min_v, max_v)
            except RuntimeError:
                self.viewer_batch_m1_win = None
        if self.matrix2 is not None and getattr(self, "viewer_batch_m2_win", None) is not None:
            try:
                if self.viewer_batch_m2_win.isVisible():
                    self.viewer_batch_m2_win.matrix_data = self.matrix2
                    self.viewer_batch_m2_win.image_item.setImage(self.matrix2.T, autoLevels=False)
                    min_v, max_v = float(np.min(self.matrix2)), float(np.max(self.matrix2))
                    self.viewer_batch_m2_win.hist.setLevels(min_v, max_v)
            except RuntimeError:
                self.viewer_batch_m2_win = None
    def batch_go_prev(self):
        if self.batch_total_count > 0:
            self.load_batch_group((self.batch_current_idx - 1) % self.batch_total_count)

    def batch_go_next(self):
        if self.batch_total_count > 0:
            self.load_batch_group((self.batch_current_idx + 1) % self.batch_total_count)

    def _on_batch_focus_changed(self, _old, new_widget):
        """輸入框聚焦時暫時關閉左右鍵快捷鍵，避免與編輯衝突。"""
        from PyQt5.QtWidgets import QAbstractSpinBox, QLineEdit, QComboBox
        blocked = False
        fw = new_widget
        if fw is not None:
            if isinstance(fw, (QAbstractSpinBox, QLineEdit, QComboBox)):
                blocked = True
            else:
                parent = fw.parentWidget()
                if parent is not None and isinstance(parent, QAbstractSpinBox):
                    blocked = True
        if hasattr(self, "shortcut_batch_prev"):
            self.shortcut_batch_prev.setEnabled(not blocked)
        if hasattr(self, "shortcut_batch_next"):
            self.shortcut_batch_next.setEnabled(not blocked)

    def show_batch_m1_heatmap(self):
        if self.matrix1 is not None:
            if getattr(self, 'viewer_batch_m1_win', None) is not None:
                self.viewer_batch_m1_win.close()
            self.viewer_batch_m1_win = HeatmapViewerWindow("Batch M1 Heatmap", self.matrix1, app_parent=self, is_m1=True)
            self.viewer_batch_m1_win.setGeometry(50, 150, 700, 650)
            self.viewer_batch_m1_win.show()
            self.update_batch_calculations()

    def show_batch_m2_heatmap(self):
        if self.matrix2 is not None:
            if getattr(self, 'viewer_batch_m2_win', None) is not None:
                self.viewer_batch_m2_win.close()
            self.viewer_batch_m2_win = HeatmapViewerWindow("Batch M2 Heatmap", self.matrix2, app_parent=self, is_m1=False)
            self.viewer_batch_m2_win.setGeometry(800, 150, 700, 650)
            self.viewer_batch_m2_win.show()
            self.update_batch_calculations()

    def show_batch_contour_window(self):
        if self.batch_result_matrix is not None:
            if getattr(self, 'contour_batch_win', None) is not None:
                self.contour_batch_win.close()
            # 這裡因為原本的 Kernel 輸入移進 Contour 內部了，所以預設為 31
            smoothed = uniform_filter(self.batch_result_matrix, size=31, mode='nearest') 
            self.contour_batch_win = ContourBatchViewerWindow(smoothed, parent=self)
            self.contour_batch_win.setGeometry(250, 100, 1000, 650)
            self.contour_batch_win.show()

    def show_batch_cross_profile(self):
        if self.batch_result_matrix is not None:
            if getattr(self, 'cross_batch_profile_win', None) is not None:
                self.cross_batch_profile_win.close()
            self.cross_batch_profile_win = CrossProfileViewerWindow("Batch", parent=self)
            self.cross_batch_profile_win.setGeometry(300, 250, 900, 600)
            h, w = self.batch_result_matrix.shape
            self.cross_batch_profile_win.update_profiles(self.batch_result_matrix, w//2, h//2)
            self.cross_batch_profile_win.show()
            
    def export_batch_results_zip(self):
        if self.batch_total_count == 0 or self.batch_result_matrix is None:
            QMessageBox.warning(self, "警告", "目前無可匯出的數據！")
            return
        if not self.save_dir_path:
            QMessageBox.warning(self, "警告", "請先點擊「選擇儲存資料夾」按鈕以指定儲存路徑！")
            return

        zip_path = os.path.join(self.save_dir_path, "DataRay_Batch_Results.zip")
        summary_csv_path = os.path.join(self.save_dir_path, "Result_Spot_Analysis.csv")
        prev_idx = self.batch_current_idx
        tmp_root = None

        try:
            self.lbl_batch_status.setText("狀態: 正在匯出 ZIP...")
            self.lbl_batch_status.setStyleSheet("color: #F57C00; font-weight: bold; font-size: 12px;")
            self.btn_batch_export.setEnabled(False)
            QApplication.processEvents()

            tmp_root = tempfile.mkdtemp(prefix="dataray_batch_export_")
            summary_columns = []  # [(col_name, {item: value, ...}, {item: unit, ...})]
            item_order = []

            for idx in range(self.batch_total_count):
                self.load_batch_group(idx)
                QApplication.processEvents()

                group_name = f"Group_{idx + 1:02d}"
                if idx < len(self.batch_pairs):
                    loc = self.batch_pairs[idx]["location"]
                    fname = os.path.splitext(self.batch_pairs[idx]["filename"])[0]
                    safe_loc = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(loc))
                    safe_fname = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(fname))
                    group_name = f"Group_{idx + 1:02d}_{safe_loc}_{safe_fname}"
                group_dir = os.path.join(tmp_root, group_name)
                os.makedirs(group_dir, exist_ok=True)
                base_path = os.path.join(group_dir, "Result")

                spot_rows = self._export_single_group_like_dataray(base_path, idx)
                if spot_rows:
                    value_map = {}
                    unit_map = {}
                    for item, value, unit in spot_rows:
                        value_map[item] = value
                        unit_map[item] = unit
                        if item not in item_order:
                            item_order.append(item)
                    summary_columns.append((group_name, value_map, unit_map))

            # 主資料夾彙整 CSV：欄位 = Item / Unit / 各組名稱
            self._write_spot_analysis_summary_csv(
                summary_csv_path, item_order, summary_columns
            )
            # 同步放一份進 ZIP 根目錄，方便帶走
            shutil.copy2(summary_csv_path, os.path.join(tmp_root, "Result_Spot_Analysis.csv"))

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _dirs, files in os.walk(tmp_root):
                    for name in files:
                        abs_path = os.path.join(root, name)
                        arcname = os.path.relpath(abs_path, tmp_root)
                        zipf.write(abs_path, arcname)

            self.load_batch_group(prev_idx)
            self.lbl_batch_status.setText("狀態: 所有檔案匯出成功！")
            self.lbl_batch_status.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 12px;")
            self.btn_batch_export.setEnabled(True)
            QMessageBox.information(
                self, "成功",
                f"匯出完成！\n\n"
                f"ZIP：\n{zip_path}\n\n"
                f"彙整統計 CSV：\n{summary_csv_path}\n\n"
                f"每組內容：\n"
                f"• Result.json\n"
                f"• Result_Result.xlsx\n"
                f"• Result_Spot_Analysis.xlsx\n"
                f"• Result_Heatmap.png\n"
                f"• Result_Contour.png"
            )
        except Exception as e:
            try:
                self.load_batch_group(prev_idx)
            except Exception:
                pass
            self.lbl_batch_status.setText("狀態: 匯出失敗")
            self.lbl_batch_status.setStyleSheet("color: #C62828; font-weight: bold; font-size: 12px;")
            self.btn_batch_export.setEnabled(True)
            QMessageBox.critical(self, "匯出錯誤", f"匯出過程發生錯誤：\n{str(e)}")
        finally:
            if tmp_root and os.path.isdir(tmp_root):
                shutil.rmtree(tmp_root, ignore_errors=True)

    def _write_spot_analysis_summary_csv(self, csv_path, item_order, summary_columns):
        """
        彙整所有組的 Spot Analysis：
        header: Item, Unit, <Group_01_loc_cyc>, <Group_02_...>, ...
        每一列為同一統計項目，各組數值橫向對齊。
        """
        headers = ["Item", "Unit"] + [col_name for col_name, _v, _u in summary_columns]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in item_order:
                unit = ""
                for _name, _values, units in summary_columns:
                    if item in units and units[item] not in ("", None):
                        unit = units[item]
                        break
                row = [item, unit]
                for _name, values, _units in summary_columns:
                    row.append(values.get(item, ""))
                writer.writerow(row)

    def _build_group_column_name(self, idx):
        if idx < len(self.batch_pairs):
            loc = self.batch_pairs[idx]["location"]
            fname = os.path.splitext(self.batch_pairs[idx]["filename"])[0]
            safe_loc = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(loc))
            safe_fname = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(fname))
            return f"Group_{idx + 1:02d}_{safe_loc}_{safe_fname}"
        return f"Group_{idx + 1:02d}"

    def _export_single_group_like_dataray(self, base_path, idx):
        """對單一組輸出與 tab_dataray.export_results 同等檔案內容；回傳 spot_rows。"""
        if self.batch_result_matrix is None:
            return []

        c_min, c_max = self.batch_hist.getLevels()
        m1_pt = self.batch_m1_center_point
        m2_pt = self.batch_m2_center_point
        m2a_pt = self.batch_m2_above_point
        click_points = []
        if m1_pt is not None:
            click_points.append(list(m1_pt))
        if m2_pt is not None:
            click_points.append(list(m2_pt))
        if m2a_pt is not None:
            click_points.append(list(m2a_pt))

        pair = self.batch_pairs[idx] if idx < len(self.batch_pairs) else {}
        location = pair.get("location", "")
        filename = pair.get("filename", "")

        path = base_path + ".json"
        config_params = {
            "ma_kernel_size": 31,
            "colorbar_min": float(c_min),
            "colorbar_max": float(c_max),
            "cross_size": 40,
            "heatmap_click_points": click_points,
            "contour_click_points": [],
            "dr_enable_spot": False,
            "m1_center_x_px": m1_pt[0] if m1_pt else None,
            "m1_center_y_px": m1_pt[1] if m1_pt else None,
            "m2_below_x_px": m2_pt[0] if m2_pt else None,
            "m2_below_y_px": m2_pt[1] if m2_pt else None,
            "m2_above_x_px": m2a_pt[0] if m2a_pt else None,
            "m2_above_y_px": m2a_pt[1] if m2a_pt else None,
            "m2_below_inscribed_r_px": getattr(self, 'batch_m2_below_circle_r', None),
            "m2_above_inscribed_r_px": getattr(self, 'batch_m2_above_circle_r', None),
            "m1_point_mode": self._get_m1_auto_mode_name(),
            "p2_point_mode": self._get_p2_point_mode_name(),
            "dr_shape_type": "circle",
            "dr_center_x_px": m1_pt[0] if m1_pt else 0,
            "dr_center_y_px": m1_pt[1] if m1_pt else 0,
            "dr_use_threshold": False,
            "dr_threshold_percent": 50.0,
            "m1_use_threshold": self.chk_batch_m1_use_threshold.isChecked(),
            "m1_threshold_percent": self.spin_batch_m1_thresh_percent.value(),
            "p2_use_threshold": self.chk_batch_p2_use_threshold.isChecked(),
            "p2_threshold_percent": self.spin_batch_p2_thresh_percent.value(),
            "dr_circle_diameter_px": 100,
            "dr_ellipse_wx_px": 100,
            "dr_ellipse_wy_px": 100,
            "batch_group_index": idx + 1,
            "batch_location": location,
            "batch_filename": filename,
            "batch_m1_file": os.path.basename(self.batch_m1_files[idx]) if idx < len(self.batch_m1_files) else "",
            "batch_m2_file": os.path.basename(self.batch_m2_files[idx]) if idx < len(self.batch_m2_files) else "",
            "batch_m1_path": self.batch_m1_files[idx] if idx < len(self.batch_m1_files) else "",
            "batch_m2_path": self.batch_m2_files[idx] if idx < len(self.batch_m2_files) else "",
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_params, f, indent=4, ensure_ascii=False)

        excel_img_path = f"{base_path}_Result.xlsx"
        pd.DataFrame(self.batch_result_matrix).to_excel(excel_img_path, index=False, header=False)

        spot_analysis_excel_path = f"{base_path}_Spot_Analysis.xlsx"
        wb_spot = openpyxl.Workbook()
        ws_spot = wb_spot.active
        ws_spot.title = "Spot_and_Measurement"
        ws_spot.append(["Item", "Value", "Unit"])

        p1 = m1_pt if m1_pt is not None else ("--", "--")
        p2 = m2_pt if m2_pt is not None else ("--", "--")
        p2a = m2a_pt if m2a_pt is not None else ("--", "--")

        dx_px, dy_px, dist_px = "--", "--", "--"
        dx_um, dy_um, dist_um = "--", "--", "--"
        dx_ab_px, dy_ab_px, dist_ab_px = "--", "--", "--"
        dx_ab_um, dy_ab_um, dist_ab_um = "--", "--", "--"
        pixel_pitch_um = 5.5

        if m1_pt is not None and m2_pt is not None:
            dx_px = abs(m2_pt[0] - m1_pt[0])
            dy_px = abs(m2_pt[1] - m1_pt[1])
            dist_px = float(np.sqrt(dx_px**2 + dy_px**2))
            dx_um = dx_px * pixel_pitch_um
            dy_um = dy_px * pixel_pitch_um
            dist_um = dist_px * pixel_pitch_um

        if m2a_pt is not None and m2_pt is not None:
            dx_ab_px = abs(m2_pt[0] - m2a_pt[0])
            dy_ab_px = abs(m2_pt[1] - m2a_pt[1])
            dist_ab_px = float(np.sqrt(dx_ab_px**2 + dy_ab_px**2))
            dx_ab_um = dx_ab_px * pixel_pitch_um
            dy_ab_um = dy_ab_px * pixel_pitch_um
            dist_ab_um = dist_ab_px * pixel_pitch_um

        info = self.batch_scale_info or {}
        max1_idx = info.get("max1_idx")
        max1_val = info.get("max1_val", "")
        match2_val = info.get("match2_val", "")
        scale_ratio = info.get("scale_ratio", "")
        if max1_idx is not None:
            peak_row, peak_col = int(max1_idx[0]), int(max1_idx[1])
        else:
            peak_row, peak_col = "", ""

        spot_rows = [
            ["Group Name", self._build_group_column_name(idx), ""],
            ["Location", location, ""],
            ["Cycle / Filename", filename, ""],
            ["Group Index", idx + 1, ""],
            ["Matrix Height", self.batch_result_matrix.shape[0], "px"],
            ["Matrix Width", self.batch_result_matrix.shape[1], "px"],
            ["M1 Peak Value", max1_val, ""],
            ["M1 Peak Row", peak_row, ""],
            ["M1 Peak Col", peak_col, ""],
            ["M2 Value at M1 Peak", match2_val, ""],
            ["Scale Ratio", scale_ratio if scale_ratio is not None else "", ""],
            ["Shape Type", "N/A (Batch)", ""],
            ["Circle Size / Wx", "N/A", "px"],
            ["Ellipse Wy", "N/A", "px"],
            ["Use Spot Threshold", "No", ""],
            ["Spot Threshold Percent", 50.0, "%"],
            ["M1 Point Mode", self._get_m1_auto_mode_name(), ""],
            ["Use M1 Point Threshold", "Yes" if self.chk_batch_m1_use_threshold.isChecked() else "No", ""],
            ["M1 Point Threshold Percent", self.spin_batch_m1_thresh_percent.value(), "%"],
            ["P2 Point Mode", self._get_p2_point_mode_name(), ""],
            ["Use P2 Point Threshold", "Yes" if self.chk_batch_p2_use_threshold.isChecked() else "No", ""],
            ["P2 Point Threshold Percent", self.spin_batch_p2_thresh_percent.value(), "%"],
            ["Mouse Cursor X", "N/A (Realtime)", "px"],
            ["Mouse Cursor Y", "N/A (Realtime)", "px"],
            ["M1 Point (X)", p1[0] if p1 != ("--", "--") else "--", "px"],
            ["M1 Point (Y)", p1[1] if p1 != ("--", "--") else "--", "px"],
            ["M2 Below (X)", p2[0] if p2 != ("--", "--") else "--", "px"],
            ["M2 Below (Y)", p2[1] if p2 != ("--", "--") else "--", "px"],
            ["M2 Above (X)", p2a[0] if p2a != ("--", "--") else "--", "px"],
            ["M2 Above (Y)", p2a[1] if p2a != ("--", "--") else "--", "px"],
            ["M2 Below Inscribed Radius", getattr(self, 'batch_m2_below_circle_r', None) if getattr(self, 'batch_m2_below_circle_r', None) is not None else "--", "px"],
            ["M2 Above Inscribed Radius", getattr(self, 'batch_m2_above_circle_r', None) if getattr(self, 'batch_m2_above_circle_r', None) is not None else "--", "px"],
            ["M1 to M2Below Delta X (px)", dx_px, "px"],
            ["M1 to M2Below Delta Y (px)", dy_px, "px"],
            ["M1 to M2Below Total Distance (px)", dist_px, "px"],
            ["M1 to M2Below Delta X (Real)", dx_um, "μm"],
            ["M1 to M2Below Delta Y (Real)", dy_um, "μm"],
            ["M1 to M2Below Total Distance (Real)", dist_um, "μm"],
            ["M2Above to M2Below Delta X (px)", dx_ab_px, "px"],
            ["M2Above to M2Below Delta Y (px)", dy_ab_px, "px"],
            ["M2Above to M2Below Total Distance (px)", dist_ab_px, "px"],
            ["M2Above to M2Below Delta X (Real)", dx_ab_um, "μm"],
            ["M2Above to M2Below Delta Y (Real)", dy_ab_um, "μm"],
            ["M2Above to M2Below Total Distance (Real)", dist_ab_um, "μm"],
            ["Cross Marker Size", 40, "px"],
        ]
        for r in spot_rows:
            ws_spot.append(r)
        wb_spot.save(spot_analysis_excel_path)

        heatmap_img_path = f"{base_path}_Heatmap.png"
        pg_export.ImageExporter(self.plot_batch_heat).export(heatmap_img_path)

        contour_img_path = f"{base_path}_Contour.png"
        smoothed = uniform_filter(self.batch_result_matrix, size=31, mode='nearest')
        temp_win = pg.GraphicsLayoutWidget()
        plot_contour = temp_win.addPlot(title="Smoothed Contour Map")
        plot_contour.getViewBox().invertY(False)
        plot_contour.setAspectLocked(True)
        plot_contour.setLabel('bottom', 'X Pixels')
        plot_contour.setLabel('left', 'Y Pixels')
        contour_img = pg.ImageItem(smoothed.T)
        plot_contour.addItem(contour_img)
        min_v, max_v = float(np.min(smoothed)), float(np.max(smoothed))
        for level in np.linspace(min_v, max_v, 10):
            iso = pg.IsocurveItem(data=smoothed.T, level=level, pen=pg.mkPen('w', width=0.8))
            plot_contour.addItem(iso)
        QApplication.processEvents()
        pg_export.ImageExporter(plot_contour).export(contour_img_path)
        temp_win.close()
        temp_win.deleteLater()
        return spot_rows

    # ================== 子圖表與 UI 連動 ==================
    def on_colorbar_levels_changed(self):
        if self.batch_result_matrix is not None:
            min_lvl, max_lvl = self.batch_hist.getLevels()
            self.plot_batch_hist.setXRange(min_lvl, max_lvl, padding=0)

    def render_sub_plots_fast(self, matrix):
        self.plot_batch_hist.clear()
        self.plot_batch_trend.clear()
        if matrix is None: return

        total_pixels = matrix.size
        sample_data = matrix.ravel()[::10] if total_pixels > 1000000 else matrix.ravel()
        y_counts, x_edges = np.histogram(sample_data, bins=40)
        
        bar_item = pg.BarGraphItem(x0=x_edges[:-1], x1=x_edges[1:], height=y_counts, brush='#E91E63', pen=None)
        self.plot_batch_hist.addItem(bar_item)
        
        peak_val = np.max(matrix)
        v_line_peak = pg.InfiniteLine(pos=peak_val, angle=90, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        self.plot_batch_hist.addItem(v_line_peak)
        
        min_lvl, max_lvl = self.batch_hist.getLevels()
        self.plot_batch_hist.setXRange(min_lvl, max_lvl, padding=0)
        
        peak_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        peak_row = peak_idx[0]

        row_profile = matrix[peak_row, :]
        x_axis = np.arange(len(row_profile))
        
        trend_curve = pg.PlotCurveItem(x_axis, row_profile, pen=pg.mkPen('#00E5FF', width=1.5))
        peak_col = np.argmax(row_profile)
        peak_spot = pg.ScatterPlotItem(x=[peak_col], y=[row_profile[peak_col]], symbol='o', size=8, brush='y', pen='r')
        
        self.plot_batch_trend.addItem(trend_curve)
        self.plot_batch_trend.addItem(peak_spot)
        self.plot_batch_trend.setTitle(f"Peak Row Trend (Row Index: {peak_row})")

    def redraw_batch_crosses(self):
        for item in self.batch_cross_items:
            self.plot_batch_heat.removeItem(item)
        self.batch_cross_items.clear()

        if not self.chk_batch_show_cross.isChecked() or self.batch_result_matrix is None:
            return

        h, w = self.batch_result_matrix.shape

        if self.batch_m1_center_point:
            cx, cy = self.batch_m1_center_point
            pen = pg.mkPen('#00C853', width=2.5, style=Qt.DashLine)
            v_item = pg.PlotCurveItem(x=[cx, cx], y=[0, h], pen=pen)
            h_item = pg.PlotCurveItem(x=[0, w], y=[cy, cy], pen=pen)
            self.plot_batch_heat.addItem(v_item)
            self.plot_batch_heat.addItem(h_item)
            self.batch_cross_items.extend([v_item, h_item])

        if self.batch_m2_center_point:
            cx2, cy2 = self.batch_m2_center_point
            pen2 = pg.mkPen('#2962FF', width=2.5)
            v_item2 = pg.PlotCurveItem(x=[cx2, cx2], y=[0, h], pen=pen2)
            h_item2 = pg.PlotCurveItem(x=[0, w], y=[cy2, cy2], pen=pen2)
            self.plot_batch_heat.addItem(v_item2)
            self.plot_batch_heat.addItem(h_item2)
            self.batch_cross_items.extend([v_item2, h_item2])

        if self.batch_m2_above_point:
            cx3, cy3 = self.batch_m2_above_point
            pen3 = pg.mkPen('#D50000', width=2.5)
            v_item3 = pg.PlotCurveItem(x=[cx3, cx3], y=[0, h], pen=pen3)
            h_item3 = pg.PlotCurveItem(x=[0, w], y=[cy3, cy3], pen=pen3)
            self.plot_batch_heat.addItem(v_item3)
            self.plot_batch_heat.addItem(h_item3)
            self.batch_cross_items.extend([v_item3, h_item3])

        # 內切圓（僅 inscribed 模式有半徑）
        r2 = getattr(self, 'batch_m2_below_circle_r', None)
        if self.batch_m2_center_point and r2 is not None and r2 > 0:
            cx2, cy2 = self.batch_m2_center_point
            circle2 = self._make_circle_curve(cx2, cy2, r2, pg.mkPen('#2962FF', width=2))
            self.plot_batch_heat.addItem(circle2)
            self.batch_cross_items.append(circle2)

        r3 = getattr(self, 'batch_m2_above_circle_r', None)
        if self.batch_m2_above_point and r3 is not None and r3 > 0:
            cx3, cy3 = self.batch_m2_above_point
            circle3 = self._make_circle_curve(cx3, cy3, r3, pg.mkPen('#D50000', width=2))
            self.plot_batch_heat.addItem(circle3)
            self.batch_cross_items.append(circle3)

    def on_batch_heatmap_colormap_toggled(self, checked=False):
        """Process 熱力圖：彩色 ↔ 黑白；同步已開啟的 M1/M2 檢視窗。"""
        cmap = self.batch_gray_map if self.chk_batch_heatmap_gray.isChecked() else self.batch_jet_map
        levels = self.batch_hist.getLevels()
        self.batch_hist.gradient.setColorMap(cmap)
        self.batch_hist.setLevels(*levels)
        title = 'Processed Matrix Result Heatmap (Batch)'
        if self.chk_batch_heatmap_gray.isChecked():
            title += ' [Grayscale]'
        self.plot_batch_heat.setTitle(title)

        for win_attr in ('viewer_batch_m1_win', 'viewer_batch_m2_win'):
            win = getattr(self, win_attr, None)
            if win is not None:
                try:
                    if win.isVisible():
                        win.set_grayscale(self.chk_batch_heatmap_gray.isChecked())
                except RuntimeError:
                    setattr(self, win_attr, None)

    @staticmethod
    def _make_circle_curve(cx, cy, radius, pen, n=72):
        theta = np.linspace(0, 2 * np.pi, n)
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta)
        return pg.PlotCurveItem(x=xs, y=ys, pen=pen)
    def _compute_auto_spot_center(self, matrix, mode, use_threshold=False, thresh_percent=50.0):
        # 強化定位：背景扣除 + 最大連通區 + 亞像素（peak_geom 仍走峰值幾何）
        return compute_auto_spot_center(
            matrix, mode, use_threshold, thresh_percent,
            bg_subtract=(mode != "peak_geom"),
            largest_cc_only=(mode != "peak_geom"),
            subpixel=True,
        )

    def _build_threshold_mask(self, matrix, use_threshold, thresh_percent, y_below=None, y_above=None,
                              robust=True):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.size == 0:
            return None
        h, w = matrix.shape
        if y_below is not None:
            y_below = split_y_index(y_below)
            if y_below <= 0:
                return np.zeros((h, w), dtype=bool)
            region = matrix[:y_below, :]
            region_mask = build_robust_threshold_mask(
                region, use_threshold, thresh_percent,
                bg_subtract=robust, largest_cc_only=robust,
            )
            mask = np.zeros((h, w), dtype=bool)
            if region_mask is not None:
                mask[:y_below, :] = region_mask
            return mask
        if y_above is not None:
            y_above = split_y_index(y_above)
            if y_above >= h - 1:
                return np.zeros((h, w), dtype=bool)
            region = matrix[y_above + 1:, :]
            region_mask = build_robust_threshold_mask(
                region, use_threshold, thresh_percent,
                bg_subtract=robust, largest_cc_only=robust,
            )
            mask = np.zeros((h, w), dtype=bool)
            if region_mask is not None:
                mask[y_above + 1:, :] = region_mask
            return mask
        return build_robust_threshold_mask(
            matrix, use_threshold, thresh_percent,
            bg_subtract=robust, largest_cc_only=robust,
        )

    def on_batch_m1_point_mode_changed(self, checked=False):
        sender = self.sender()
        if sender is not None and hasattr(sender, "isChecked") and not sender.isChecked():
            return
        self.update_batch_calculations()

    def on_batch_m1_threshold_toggled(self, checked):
        self.spin_batch_m1_thresh_percent.setEnabled(checked)
        self.lbl_batch_m1_thresh_spin.setEnabled(checked)
        self.update_batch_calculations()

    def on_batch_m1_thresh_percent_changed(self):
        self.update_batch_calculations()

    def on_batch_p2_point_mode_changed(self, checked=False):
        sender = self.sender()
        if sender is not None and hasattr(sender, "isChecked") and not sender.isChecked():
            return
        self.update_batch_calculations()

    def on_batch_p2_threshold_toggled(self, checked):
        self.spin_batch_p2_thresh_percent.setEnabled(checked)
        self.lbl_batch_p2_thresh_spin.setEnabled(checked)
        self.update_batch_calculations()

    def on_batch_p2_thresh_percent_changed(self, _value=None):
        self.update_batch_calculations()

    def _get_m1_auto_mode_name(self):
        if self.radio_batch_m1_peak_geom.isChecked():
            return "peak_geom"
        if self.radio_batch_m1_centroid.isChecked():
            return "centroid"
        if self.radio_batch_m1_thresh_geom.isChecked():
            return "thresh_geom"
        return "manual"

    def _is_p2_auto_mode(self):
        return (
            self.radio_batch_p2_auto_min.isChecked()
            or self.radio_batch_p2_m2_thresh_geom.isChecked()
            or self.radio_batch_p2_m2_centroid.isChecked()
            or self.radio_batch_p2_m2_inscribed.isChecked()
        )

    def _get_p2_point_mode_name(self):
        if self.radio_batch_p2_auto_min.isChecked():
            return "auto_min"
        if self.radio_batch_p2_m2_thresh_geom.isChecked():
            return "m2_thresh_geom"
        if self.radio_batch_p2_m2_centroid.isChecked():
            return "m2_centroid"
        if self.radio_batch_p2_m2_inscribed.isChecked():
            return "m2_inscribed"
        return "manual"

    def _fit_inscribed_circle(self, matrix, use_threshold=True, thresh_percent=50.0):
        """門檻篩出最大連通 contour，以距離變換求最大內切圓中心與半徑。

        Returns:
            (cx, cy, radius) 或 None
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.size == 0:
            return None
        bg = estimate_border_background(matrix)
        work = np.clip(matrix - bg, 0.0, None)
        peak_val = float(np.max(work)) if work.size > 0 else 0.0
        if not np.isfinite(peak_val) or peak_val <= 0:
            return None
        thresh_val = peak_val * (thresh_percent / 100.0) if use_threshold else peak_val * 0.5
        mask = work >= thresh_val
        if not np.any(mask):
            return None

        labeled, num = ndi_label(mask)
        if num <= 0:
            return None
        counts = np.bincount(labeled.ravel())
        if counts.size <= 1:
            return None
        counts[0] = 0
        largest_id = int(np.argmax(counts))
        blob = labeled == largest_id

        dist = distance_transform_edt(blob)
        max_idx = np.argmax(dist)
        cy, cx = np.unravel_index(max_idx, dist.shape)
        radius = float(dist[cy, cx])
        # 距離變換峰值在像素格點上；回傳亞像素座標（格點中心）
        cx_f, cy_f = float(cx), float(cy)
        if radius <= 0:
            return (cx_f, cy_f, 0.0)
        return (cx_f, cy_f, radius)

    def _find_inscribed_circle_below_y(self, matrix, y1, use_thresh, thresh_percent):
        matrix = np.asarray(matrix)
        y1 = split_y_index(y1)
        if y1 <= 0:
            return None
        region = matrix[:y1, :]
        if region.size == 0:
            return None
        return self._fit_inscribed_circle(region, use_thresh, thresh_percent)

    def _find_inscribed_circle_above_y(self, matrix, y1, use_thresh, thresh_percent):
        matrix = np.asarray(matrix)
        y1 = split_y_index(y1)
        h = matrix.shape[0]
        if y1 >= h - 1:
            return None
        region = matrix[y1 + 1:, :]
        if region.size == 0:
            return None
        result = self._fit_inscribed_circle(region, use_thresh, thresh_percent)
        if result is None:
            return None
        cx, cy_local, radius = result
        return (cx, cy_local + y1 + 1, radius)

    def _find_min_below_y(self, matrix, y1):
        matrix = np.asarray(matrix)
        y1 = split_y_index(y1)
        if y1 <= 0:
            return None
        region = matrix[:y1, :]
        if region.size == 0:
            return None
        min_val = np.min(region)
        ys, xs = np.where(region == min_val)
        if len(xs) == 0:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _find_center_below_y(self, matrix, y1, mode, use_thresh, thresh_percent):
        matrix = np.asarray(matrix)
        y1 = split_y_index(y1)
        if y1 <= 0:
            return None
        region = matrix[:y1, :]
        if region.size == 0:
            return None
        return self._compute_auto_spot_center(region, mode, use_thresh, thresh_percent)

    def _find_min_above_y(self, matrix, y1):
        matrix = np.asarray(matrix)
        y1 = split_y_index(y1)
        h = matrix.shape[0]
        if y1 >= h - 1:
            return None
        region = matrix[y1 + 1:, :]
        if region.size == 0:
            return None
        min_val = np.min(region)
        ys, xs = np.where(region == min_val)
        if len(xs) == 0:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)) + y1 + 1)

    def _find_center_above_y(self, matrix, y1, mode, use_thresh, thresh_percent):
        matrix = np.asarray(matrix)
        y1 = split_y_index(y1)
        h = matrix.shape[0]
        if y1 >= h - 1:
            return None
        region = matrix[y1 + 1:, :]
        if region.size == 0:
            return None
        cx, cy_local = self._compute_auto_spot_center(region, mode, use_thresh, thresh_percent)
        return (cx, cy_local + y1 + 1)

    def _compute_m2_above_point(self, m1_y, p2_mode, silent=False):
        """以與 below 相同的 M2 方法／門檻，在 M1 Y 以上找 M2 above 質心。

        Returns:
            (cx, cy) 或 None。內切圓模式會一併寫入 batch_m2_above_circle_r。
        """
        use_thresh = self.chk_batch_p2_use_threshold.isChecked()
        thresh_percent = self.spin_batch_p2_thresh_percent.value()
        self.batch_m2_above_circle_r = None

        if p2_mode == "auto_min":
            # above 依需求在 M2 矩陣上找（同套用 auto_min = 找最小值）
            if self.matrix2 is None:
                return None
            if self.matrix1 is not None and self.matrix2.shape != self.matrix1.shape:
                if not silent:
                    QMessageBox.warning(self, "警告", "M1 與 M2 矩陣尺寸不一致，無法抓取 M2(above)。")
                return None
            return self._find_min_above_y(self.matrix2, m1_y)

        if self.matrix2 is None:
            return None
        if self.matrix1 is not None and self.matrix2.shape != self.matrix1.shape:
            if not silent:
                QMessageBox.warning(self, "警告", "M1 與 M2 矩陣尺寸不一致，無法抓取 M2(above)。")
            return None

        if p2_mode == "m2_inscribed":
            result = self._find_inscribed_circle_above_y(
                self.matrix2, m1_y, use_thresh, thresh_percent)
            if result is None:
                return None
            cx, cy, radius = result
            self.batch_m2_above_circle_r = radius
            return (cx, cy)

        # thresh_geom / centroid / manual：above 一律用 M2 + 目前門檻設定
        # manual 的 below 可手點；above 仍自動以質心＋門檻計算
        if p2_mode == "m2_thresh_geom":
            center_mode = "thresh_geom"
        elif p2_mode == "m2_centroid":
            center_mode = "centroid"
        else:
            # manual 或其他：沿用質心＋門檻
            center_mode = "centroid"
        return self._find_center_above_y(
            self.matrix2, m1_y, center_mode, use_thresh, thresh_percent
        )

    def update_batch_data_panel(self, m1_x=None, m1_y=None, m2_x=None, m2_y=None,
                                m2a_x=None, m2a_y=None):
        """同步左側「計算數據與狀態」：峰值／比例／矩陣大小／兩段距離。"""
        if self.matrix1 is None or self.batch_result_matrix is None:
            self.reset_batch_data_panel()
            return

        info = self.batch_scale_info or {}
        max1_idx = info.get("max1_idx")
        if max1_idx is None:
            max1_idx = np.unravel_index(np.argmax(self.matrix1, axis=None), self.matrix1.shape)
        max1_val = info.get("max1_val", float(self.matrix1[max1_idx]))
        match2_val = info.get("match2_val")
        if match2_val is None and self.matrix2 is not None:
            match2_val = float(self.matrix2[max1_idx])
        scale_ratio = info.get("scale_ratio")
        mode = info.get("mode", self.combo_batch_mode.currentData())

        row_i, col_i = int(max1_idx[0]), int(max1_idx[1])
        self.lbl_batch_max1.setText(
            f"M1 最大值(位置): {max1_val:.1f} (Row:{row_i}, Col:{col_i})"
        )
        if match2_val is not None:
            self.lbl_batch_max2.setText(f"M2 同位置數值: {match2_val:.1f}")
        else:
            self.lbl_batch_max2.setText("M2 同位置數值: --")

        if mode == "calc" and scale_ratio is not None:
            self.lbl_batch_ratio.setText(f"計算得出比例: {scale_ratio:.4f}")
        elif mode == "sub":
            self.lbl_batch_ratio.setText("計算得出比例: N/A (純相減)")
        elif mode == "div":
            self.lbl_batch_ratio.setText("計算得出比例: N/A (純相除)")
        else:
            self.lbl_batch_ratio.setText("計算得出比例: --")

        h, w = self.batch_result_matrix.shape
        self.lbl_batch_size.setText(f"矩陣大小: {h} × {w}")

        pixel_pitch_um = 5.5
        if m1_x is None or m1_y is None:
            if self.batch_m1_center_point:
                m1_x, m1_y = self.batch_m1_center_point
        if m2_x is None or m2_y is None:
            if self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
        if m2a_x is None or m2a_y is None:
            if self.batch_m2_above_point:
                m2a_x, m2a_y = self.batch_m2_above_point

        # --- M1 → M2(below) ---
        if m1_x is not None and m1_y is not None and m2_x is not None and m2_y is not None:
            dx = m2_x - m1_x
            dy = m2_y - m1_y
            distance_px = float(np.sqrt(dx**2 + dy**2))
            self.lbl_batch_distance_xy.setText(
                f"M1→M2(below) 差距: ΔX: {abs(dx):.2f} px, ΔY: {abs(dy):.2f} px"
            )
            self.lbl_batch_distance_total.setText(
                f"M1→M2(below) 總距離: {distance_px:.2f} px"
            )
            dx_real = abs(dx) * pixel_pitch_um
            dy_real = abs(dy) * pixel_pitch_um
            distance_real = distance_px * pixel_pitch_um
            self.lbl_batch_real_distance_xy.setText(
                f"M1→M2(below) 實際 (*5.5): ΔX: {dx_real:.2f} μm, ΔY: {dy_real:.2f} μm"
            )
            self.lbl_batch_real_distance_total.setText(
                f"M1→M2(below) 實際總距離: {distance_real:.2f} μm"
            )
        else:
            self.lbl_batch_distance_xy.setText("M1→M2(below) 差距: ΔX: -- px, ΔY: -- px")
            self.lbl_batch_distance_total.setText("M1→M2(below) 總距離: -- px")
            self.lbl_batch_real_distance_xy.setText(
                "M1→M2(below) 實際 (*5.5): ΔX: -- μm, ΔY: -- μm"
            )
            self.lbl_batch_real_distance_total.setText("M1→M2(below) 實際總距離: -- μm")

        # --- M2(above) → M2(below) ---
        if (m2a_x is not None and m2a_y is not None
                and m2_x is not None and m2_y is not None):
            dx_ab = m2_x - m2a_x
            dy_ab = m2_y - m2a_y
            dist_ab_px = float(np.sqrt(dx_ab**2 + dy_ab**2))
            self.lbl_batch_ab_distance_xy.setText(
                f"M2(above)→M2(below) 差距: ΔX: {abs(dx_ab):.2f} px, ΔY: {abs(dy_ab):.2f} px"
            )
            self.lbl_batch_ab_distance_total.setText(
                f"M2(above)→M2(below) 總距離: {dist_ab_px:.2f} px"
            )
            dx_ab_um = abs(dx_ab) * pixel_pitch_um
            dy_ab_um = abs(dy_ab) * pixel_pitch_um
            dist_ab_um = dist_ab_px * pixel_pitch_um
            self.lbl_batch_ab_real_distance_xy.setText(
                f"M2(above)→M2(below) 實際 (*5.5): ΔX: {dx_ab_um:.2f} μm, ΔY: {dy_ab_um:.2f} μm"
            )
            self.lbl_batch_ab_real_distance_total.setText(
                f"M2(above)→M2(below) 實際總距離: {dist_ab_um:.2f} μm"
            )
        else:
            self.lbl_batch_ab_distance_xy.setText(
                "M2(above)→M2(below) 差距: ΔX: -- px, ΔY: -- px"
            )
            self.lbl_batch_ab_distance_total.setText("M2(above)→M2(below) 總距離: -- px")
            self.lbl_batch_ab_real_distance_xy.setText(
                "M2(above)→M2(below) 實際 (*5.5): ΔX: -- μm, ΔY: -- μm"
            )
            self.lbl_batch_ab_real_distance_total.setText(
                "M2(above)→M2(below) 實際總距離: -- μm"
            )

    def reset_batch_data_panel(self):
        self.lbl_batch_max1.setText("M1 最大值(位置): --")
        self.lbl_batch_max2.setText("M2 同位置數值: --")
        self.lbl_batch_ratio.setText("計算得出比例: --")
        self.lbl_batch_size.setText("矩陣大小: --")
        self.lbl_batch_distance_xy.setText("M1→M2(below) 差距: ΔX: -- px, ΔY: -- px")
        self.lbl_batch_distance_total.setText("M1→M2(below) 總距離: -- px")
        self.lbl_batch_real_distance_xy.setText(
            "M1→M2(below) 實際 (*5.5): ΔX: -- μm, ΔY: -- μm"
        )
        self.lbl_batch_real_distance_total.setText("M1→M2(below) 實際總距離: -- μm")
        self.lbl_batch_ab_distance_xy.setText(
            "M2(above)→M2(below) 差距: ΔX: -- px, ΔY: -- px"
        )
        self.lbl_batch_ab_distance_total.setText("M2(above)→M2(below) 總距離: -- px")
        self.lbl_batch_ab_real_distance_xy.setText(
            "M2(above)→M2(below) 實際 (*5.5): ΔX: -- μm, ΔY: -- μm"
        )
        self.lbl_batch_ab_real_distance_total.setText(
            "M2(above)→M2(below) 實際總距離: -- μm"
        )
        if hasattr(self, "lbl_batch_current_files"):
            self.lbl_batch_current_files.setText("目前檔案: --")
        if hasattr(self, "lbl_batch_group_nav"):
            self.lbl_batch_group_nav.setText("0 / 0")

    def on_batch_process_mouse_clicked(self, evt):
        """手動第二點：對齊單次模式，點擊 Process Result Heatmap。"""
        if self.batch_result_matrix is None:
            return
        if not self.radio_batch_p2_manual.isChecked():
            return
        pos = evt.scenePos()
        if not self.plot_batch_heat.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_batch_heat.getViewBox().mapSceneToView(pos)
        cx = int(round(mouse_point.x()))
        cy = int(round(mouse_point.y()))
        h, w = self.batch_result_matrix.shape
        if not (0 <= cx < w and 0 <= cy < h):
            return
        if evt.double():
            self.batch_m2_center_point = None
        else:
            self.batch_m2_center_point = (cx, cy)
        self.update_batch_calculations()

    def update_batch_calculations(self, silent=False):
        if not hasattr(self, 'matrix1') or self.matrix1 is None:
            return
        if not hasattr(self, 'matrix2') or self.matrix2 is None:
            return
        if self.batch_result_matrix is None:
            return

        # 1. 運算 M1 座標（與單次相同）
        m1_mode = self._get_m1_auto_mode_name()
        if m1_mode != "manual":
            use_thresh = self.chk_batch_m1_use_threshold.isChecked()
            thresh_percent = self.spin_batch_m1_thresh_percent.value()
            m1_x, m1_y = self._compute_auto_spot_center(
                self.matrix1, m1_mode, use_thresh, thresh_percent)
            self.batch_m1_center_point = (m1_x, m1_y)
        else:
            if self.batch_m1_center_point:
                m1_x, m1_y = self.batch_m1_center_point
            else:
                m1_x, m1_y = self.matrix1.shape[1] // 2, self.matrix1.shape[0] // 2
                self.batch_m1_center_point = (m1_x, m1_y)

        m1_mask = self._build_threshold_mask(
            self.matrix1,
            self.chk_batch_m1_use_threshold.isChecked(),
            self.spin_batch_m1_thresh_percent.value(),
        )

        # 2. 運算第二點（嚴格對齊 tab_dataray.apply_p2_point_from_mode）
        p2_mode = self._get_p2_point_mode_name()
        m2_x, m2_y = None, None
        self.batch_m2_below_circle_r = None

        if p2_mode == "manual":
            if self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
            else:
                m2_x, m2_y = m1_x, m1_y
                self.batch_m2_center_point = (m2_x, m2_y)
        elif p2_mode == "auto_min":
            if self.matrix1 is not None and self.batch_result_matrix.shape != self.matrix1.shape:
                if not silent:
                    QMessageBox.warning(self, "警告", "M1 與 Process Result 矩陣尺寸不一致，無法自動抓取第二點。")
            else:
                p2 = self._find_min_below_y(self.batch_result_matrix, m1_y)
                if p2 is None:
                    if not silent:
                        QMessageBox.warning(self, "警告", "第一點 Y 以下沒有可搜尋區域，無法自動抓取第二點。")
                else:
                    m2_x, m2_y = p2
        elif p2_mode == "m2_inscribed":
            if self.matrix1 is not None and self.matrix2.shape != self.matrix1.shape:
                if not silent:
                    QMessageBox.warning(self, "警告", "M1 與 M2 矩陣尺寸不一致，無法自動抓取第二點。")
            else:
                use_thresh = self.chk_batch_p2_use_threshold.isChecked()
                thresh_percent = self.spin_batch_p2_thresh_percent.value()
                p2 = self._find_inscribed_circle_below_y(
                    self.matrix2, m1_y, use_thresh, thresh_percent)
                if p2 is None:
                    if not silent:
                        QMessageBox.warning(
                            self, "警告",
                            "第一點 Y 以下無法以門檻 contour 擬合內切圓，請調整門檻或確認光斑。")
                else:
                    m2_x, m2_y, self.batch_m2_below_circle_r = p2
        else:
            if self.matrix1 is not None and self.matrix2.shape != self.matrix1.shape:
                if not silent:
                    QMessageBox.warning(self, "警告", "M1 與 M2 矩陣尺寸不一致，無法自動抓取第二點。")
            else:
                use_thresh = self.chk_batch_p2_use_threshold.isChecked()
                thresh_percent = self.spin_batch_p2_thresh_percent.value()
                center_mode = "thresh_geom" if p2_mode == "m2_thresh_geom" else "centroid"
                p2 = self._find_center_below_y(
                    self.matrix2, m1_y, center_mode, use_thresh, thresh_percent)
                if p2 is None:
                    if not silent:
                        QMessageBox.warning(self, "警告", "第一點 Y 以下沒有可搜尋區域，無法自動抓取第二點。")
                else:
                    m2_x, m2_y = p2

        if m2_x is None or m2_y is None:
            if self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
            else:
                m2_x, m2_y = m1_x, m1_y

        self.batch_m2_center_point = (m2_x, m2_y)

        # 3. 運算 M2(above)：與 below 相同方法／門檻，搜尋 M1 Y 以上
        m2a = self._compute_m2_above_point(m1_y, p2_mode, silent=silent)
        if m2a is None:
            self.batch_m2_above_point = None
            m2a_x = m2a_y = None
            if not silent:
                # 僅在非 silent 且有可用 below 時提示一次即可；避免過度打擾
                pass
        else:
            m2a_x, m2a_y = m2a
            self.batch_m2_above_point = (m2a_x, m2a_y)

        # 門檻 overlay：below + above 區域（橘半透明）
        use_p2_thresh = self.chk_batch_p2_use_threshold.isChecked()
        p2_pct = self.spin_batch_p2_thresh_percent.value()
        m2_mask_below = self._build_threshold_mask(
            self.matrix2, use_p2_thresh, p2_pct, y_below=m1_y)
        m2_mask_above = self._build_threshold_mask(
            self.matrix2, use_p2_thresh, p2_pct, y_above=m1_y)
        if m2_mask_below is not None and m2_mask_above is not None:
            m2_mask = np.logical_or(m2_mask_below, m2_mask_above)
        else:
            m2_mask = m2_mask_below if m2_mask_below is not None else m2_mask_above

        distance = np.sqrt((m1_x - m2_x)**2 + (m1_y - m2_y)**2)
        if m2a_x is not None and m2a_y is not None:
            dist_ab = np.sqrt((m2a_x - m2_x)**2 + (m2a_y - m2_y)**2)
            print(
                f"[Batch 運算] M1({m1_x:.2f}, {m1_y:.2f}) | M2-below({m2_x:.2f}, {m2_y:.2f}) | "
                f"M2-above({m2a_x:.2f}, {m2a_y:.2f}) | M1→below: {distance:.2f} px | "
                f"above→below: {dist_ab:.2f} px"
            )
        else:
            print(
                f"[Batch 運算] M1({m1_x:.2f}, {m1_y:.2f}) | "
                f"M2-below({m2_x:.2f}, {m2_y:.2f}) | 距離: {distance:.2f} px"
            )

        self.update_batch_data_panel(m1_x, m1_y, m2_x, m2_y, m2a_x, m2a_y)
        self.redraw_batch_crosses()

        if getattr(self, 'viewer_batch_m1_win', None) is not None:
            self.viewer_batch_m1_win.draw_marker((m1_x, m1_y))
            if self.chk_batch_m1_show_thresh.isChecked():
                self.viewer_batch_m1_win.set_threshold_overlay(
                    m1_mask, visible=True, rgba_color=(0, 255, 0, 90))
            else:
                self.viewer_batch_m1_win.clear_threshold_overlay()

        if getattr(self, 'viewer_batch_m2_win', None) is not None:
            self.viewer_batch_m2_win.draw_marker(
                (m1_x, m1_y),
                pt2=(m2_x, m2_y),
                pt3=self.batch_m2_above_point,
                r2=getattr(self, 'batch_m2_below_circle_r', None),
                r3=getattr(self, 'batch_m2_above_circle_r', None),
            )
            if self.chk_batch_p2_show_thresh.isChecked():
                self.viewer_batch_m2_win.set_threshold_overlay(
                    m2_mask, visible=True, rgba_color=(41, 98, 255, 90))
            else:
                self.viewer_batch_m2_win.clear_threshold_overlay()

    def save_current_batch_params(self):
        if self.batch_total_count == 0:
            QMessageBox.warning(self, "警告", "目前沒有載入任何 Batch 資料！")
            return

        self.batch_saved_params[self.batch_current_idx] = {
            "m1_mode": self._get_m1_auto_mode_name(),
            "use_thresh": self.chk_batch_m1_use_threshold.isChecked(),
            "thresh_percent": self.spin_batch_m1_thresh_percent.value(),
            "m1_center_point": getattr(self, 'batch_m1_center_point', None),
            "m2_mode": self._get_p2_point_mode_name(),
            "p2_use_threshold": self.chk_batch_p2_use_threshold.isChecked(),
            "m2_thresh_percent": self.spin_batch_p2_thresh_percent.value(),
            "m2_center_point": getattr(self, 'batch_m2_center_point', None),
            "m2_above_point": getattr(self, 'batch_m2_above_point', None),
        }
        QMessageBox.information(self, "暫存成功", f"第 {self.batch_current_idx + 1} 組參數與位置已暫存！")

    def _apply_saved_batch_params(self, idx):
        """若該組有暫存參數，還原 UI 與座標後再計算。"""
        params = self.batch_saved_params.get(idx)
        if not params:
            return

        m1_mode = params.get("m1_mode", "centroid")
        radios_m1 = {
            "centroid": self.radio_batch_m1_centroid,
            "thresh_geom": self.radio_batch_m1_thresh_geom,
            "peak_geom": self.radio_batch_m1_peak_geom,
            "manual": self.radio_batch_m1_manual,
        }
        for r in radios_m1.values():
            r.blockSignals(True)
        radios_m1.get(m1_mode, self.radio_batch_m1_centroid).setChecked(True)
        for r in radios_m1.values():
            r.blockSignals(False)

        self.chk_batch_m1_use_threshold.blockSignals(True)
        self.chk_batch_m1_use_threshold.setChecked(bool(params.get("use_thresh", True)))
        self.chk_batch_m1_use_threshold.blockSignals(False)

        self.spin_batch_m1_thresh_percent.blockSignals(True)
        self.spin_batch_m1_thresh_percent.setValue(params.get("thresh_percent", 50.0))
        self.spin_batch_m1_thresh_percent.blockSignals(False)

        p2_mode = params.get("m2_mode", "auto_min")
        if p2_mode in ("auto_m1", "auto_global"):
            p2_mode = "auto_min"
        radios_p2 = {
            "auto_min": self.radio_batch_p2_auto_min,
            "m2_thresh_geom": self.radio_batch_p2_m2_thresh_geom,
            "m2_centroid": self.radio_batch_p2_m2_centroid,
            "m2_inscribed": self.radio_batch_p2_m2_inscribed,
            "manual": self.radio_batch_p2_manual,
        }
        for r in radios_p2.values():
            r.blockSignals(True)
        radios_p2.get(p2_mode, self.radio_batch_p2_auto_min).setChecked(True)
        for r in radios_p2.values():
            r.blockSignals(False)

        self.chk_batch_p2_use_threshold.blockSignals(True)
        self.chk_batch_p2_use_threshold.setChecked(bool(params.get("p2_use_threshold", True)))
        self.chk_batch_p2_use_threshold.blockSignals(False)

        self.spin_batch_p2_thresh_percent.blockSignals(True)
        self.spin_batch_p2_thresh_percent.setValue(params.get("m2_thresh_percent", 50.0))
        self.spin_batch_p2_thresh_percent.blockSignals(False)

        if m1_mode == "manual" and params.get("m1_center_point"):
            self.batch_m1_center_point = tuple(params["m1_center_point"])
        if p2_mode == "manual" and params.get("m2_center_point"):
            self.batch_m2_center_point = tuple(params["m2_center_point"])
