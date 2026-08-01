import os
import glob
import zipfile
import numpy as np
import pandas as pd
import pyqtgraph as pg
from scipy.ndimage import uniform_filter

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QMessageBox, QFrame, 
                             QSplitter, QSizePolicy, QRadioButton, QButtonGroup, 
                             QScrollArea, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt

from shared_components import (NoWheelSpinBox, NoWheelDoubleSpinBox, 
                               HeatmapViewerWindow, ContourBatchViewerWindow, 
                               CrossProfileViewerWindow)

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
        self.batch_current_idx = 0
        self.batch_total_count = 0
        self.batch_saved_params = {}
        
        # 彈出視窗與十字標記預留變數
        self.viewer_batch_m1_win = None
        self.viewer_batch_m2_win = None
        self.contour_batch_win = None
        self.cross_batch_profile_win = None
        
        self.batch_m1_center_point = None
        self.batch_m2_center_point = None
        self.batch_cross_items = [] # 用來存放畫在 Heatmap 上的十字線

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
        top_fixed_layout.addWidget(self.combo_batch_mode)

        self.chk_batch_normalize = QCheckBox("Normalize")
        top_fixed_layout.addWidget(self.chk_batch_normalize)

        top_fixed_layout.addWidget(self._create_hline())

        self.btn_batch_m1_dir = QPushButton("I. 選擇 M1 資料夾 (自動排序)")
        self.btn_batch_m1_dir.setStyleSheet(btn_style_folder)
        self.btn_batch_m1_dir.clicked.connect(self.load_batch_m1_folder)
        top_fixed_layout.addWidget(self.btn_batch_m1_dir)

        self.lbl_batch_m1_info = QLabel("未選擇 M1 資料夾")
        top_fixed_layout.addWidget(self.lbl_batch_m1_info)

        self.btn_batch_m2_dir = QPushButton("II. 選擇 M2 資料夾 (自動排序)")
        self.btn_batch_m2_dir.setStyleSheet(btn_style_folder)
        self.btn_batch_m2_dir.clicked.connect(self.load_batch_m2_folder)
        top_fixed_layout.addWidget(self.btn_batch_m2_dir)

        self.lbl_batch_m2_info = QLabel("未選擇 M2 資料夾")
        top_fixed_layout.addWidget(self.lbl_batch_m2_info)

        top_fixed_layout.addWidget(self._create_hline())

        self.btn_batch_run = QPushButton("開始批量載入與運算")
        self.btn_batch_run.setStyleSheet(btn_style_primary)
        self.btn_batch_run.clicked.connect(self.process_batch_data)
        top_fixed_layout.addWidget(self.btn_batch_run)

        self.lbl_batch_status = QLabel("狀態: 等待選擇資料夾")
        top_fixed_layout.addWidget(self.lbl_batch_status)

        left_outer_layout.addWidget(top_fixed_widget)

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

        lbl_m2_pt = QLabel("第二點（M2 Heatmap）：")
        lbl_m2_pt.setStyleSheet("font-weight: bold; color: #1565C0;")
        left_layout.addWidget(lbl_m2_pt)

        self.batch_m2_point_group = QButtonGroup(self)
        
        self.radio_batch_m2_auto_m1 = QRadioButton("自動抓取 (沿用 M1 座標尋找最小值)")
        self.radio_batch_m2_auto_m1.setChecked(True)
        self.radio_batch_m2_auto_m1.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_m2_auto_m1)
        left_layout.addWidget(self.radio_batch_m2_auto_m1)

        self.radio_batch_m2_auto_global = QRadioButton("自動抓取 (自動尋找全局最小值)")
        self.radio_batch_m2_auto_global.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_m2_auto_global)
        left_layout.addWidget(self.radio_batch_m2_auto_global)

        self.radio_batch_m2_manual = QRadioButton("手動抓取 (點擊 M2 影像)")
        self.radio_batch_m2_manual.toggled.connect(self.on_batch_p2_point_mode_changed)
        self.batch_m2_point_group.addButton(self.radio_batch_m2_manual)
        left_layout.addWidget(self.radio_batch_m2_manual)

        layout_m2_thresh = QHBoxLayout()
        self.lbl_batch_m2_thresh_spin = QLabel("第二點門檻比例 (%):")
        self.spin_batch_m2_thresh_percent = NoWheelDoubleSpinBox()
        self.spin_batch_m2_thresh_percent.setRange(0.1, 100.0)
        self.spin_batch_m2_thresh_percent.setValue(50.0)
        self.spin_batch_m2_thresh_percent.setSingleStep(1.0)
        self.spin_batch_m2_thresh_percent.setDecimals(1)
        self.spin_batch_m2_thresh_percent.valueChanged.connect(self.on_batch_m2_thresh_percent_changed)
        layout_m2_thresh.addWidget(self.lbl_batch_m2_thresh_spin)
        layout_m2_thresh.addWidget(self.spin_batch_m2_thresh_percent)
        left_layout.addLayout(layout_m2_thresh)

        self.chk_batch_p2_show_thresh = QCheckBox("顯示門檻區域於 M2 圖 (洋紅半透明)")
        self.chk_batch_p2_show_thresh.setChecked(True)
        self.chk_batch_p2_show_thresh.setStyleSheet("color: #1565C0;")
        self.chk_batch_p2_show_thresh.toggled.connect(self.update_batch_calculations)
        left_layout.addWidget(self.chk_batch_p2_show_thresh)

        left_layout.addWidget(self._create_hline())

        # 加入 Batch 專屬十字開關
        self.chk_batch_show_cross = QCheckBox("顯示 M1(黃) / M2(青) 十字標記於結果圖")
        self.chk_batch_show_cross.setChecked(True)
        self.chk_batch_show_cross.setStyleSheet("font-weight: bold; color: #E65100;")
        self.chk_batch_show_cross.toggled.connect(self.redraw_batch_crosses)
        left_layout.addWidget(self.chk_batch_show_cross)

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
        jet_map = pg.ColorMap(pos, colors)

        self.plot_batch_heat = self.win_batch_top.addPlot(row=0, col=0, title='Processed Matrix Result Heatmap (Batch)')
        self.plot_batch_heat.getViewBox().invertY(False)
        self.plot_batch_heat.setAspectLocked(True)
        self.plot_batch_heat.setLabel('bottom', 'X Pixels')
        self.plot_batch_heat.setLabel('left', 'Y Pixels')

        self.batch_image_item = pg.ImageItem()
        self.plot_batch_heat.addItem(self.batch_image_item)
        
        self.batch_hist = pg.HistogramLUTItem()
        self.batch_hist.setImageItem(self.batch_image_item)
        self.batch_hist.gradient.setColorMap(jet_map)
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

    def load_batch_m1_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇 M1 檔案資料夾", "")
        if dir_path:
            import re
            files = glob.glob(os.path.join(dir_path, "*.xlsx")) + glob.glob(os.path.join(dir_path, "*.xls"))
            files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(f))])
            self.batch_m1_files = files
            self.lbl_batch_m1_info.setText(f"{os.path.basename(dir_path)} (共 {len(files)} 筆)")
            if not self.save_dir_path:
                self.save_dir_path = dir_path
                self.lbl_batch_dir_path.setText(f"{dir_path}")

    def load_batch_m2_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇 M2 檔案資料夾", "")
        if dir_path:
            import re
            files = glob.glob(os.path.join(dir_path, "*.xlsx")) + glob.glob(os.path.join(dir_path, "*.xls"))
            files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(f))])
            self.batch_m2_files = files
            self.lbl_batch_m2_info.setText(f"{os.path.basename(dir_path)} (共 {len(files)} 筆)")

    def process_batch_data(self):
        if not self.batch_m1_files or not self.batch_m2_files:
            QMessageBox.warning(self, "警告", "請完整選擇 M1 與 M2 資料夾！")
            return
        self.batch_total_count = min(len(self.batch_m1_files), len(self.batch_m2_files))
        self.batch_current_idx = 0
        self.batch_saved_params.clear()
        self.load_batch_group(0)
        self.btn_batch_export.setEnabled(True)
        self.btn_batch_view_m1.setEnabled(True)
        self.btn_batch_view_m2.setEnabled(True)
        self.btn_batch_view_cross.setEnabled(True)

    def load_batch_group(self, idx):
        if idx < 0 or idx >= self.batch_total_count:
            return
        try:
            self.batch_current_idx = idx
            self.lbl_batch_group_num.setText(f"{idx + 1} / {self.batch_total_count}")
            f1 = self.batch_m1_files[idx]
            f2 = self.batch_m2_files[idx]

            df1 = pd.read_excel(f1, header=None, skiprows=4)
            df2 = pd.read_excel(f2, header=None, skiprows=4)
            self.matrix1 = df1.dropna(how='all').astype(float).values
            self.matrix2 = df2.dropna(how='all').astype(float).values

            mode_data = self.combo_batch_mode.currentData()
            if self.chk_batch_normalize.isChecked():
                max1, max2 = np.max(self.matrix1), np.max(self.matrix2)
                m2_proc = self.matrix2 * (max1 / max2) if max2 != 0 else self.matrix2
            else:
                m2_proc = self.matrix2

            if mode_data == "sub":
                self.batch_result_matrix = self.matrix1 - m2_proc
            elif mode_data == "div":
                safe_m2 = np.where(m2_proc == 0, 1e-9, m2_proc)
                self.batch_result_matrix = self.matrix1 / safe_m2
            else:
                max1_idx = np.unravel_index(np.argmax(self.matrix1, axis=None), self.matrix1.shape)
                max1_val = self.matrix1[max1_idx]
                match2_val = m2_proc[max1_idx] if m2_proc[max1_idx] != 0 else 1e-9
                scale_ratio = max1_val / match2_val
                self.batch_result_matrix = self.matrix1 - (m2_proc * scale_ratio)

            self.batch_image_item.setImage(self.batch_result_matrix.T)
            self.batch_hist.setLevels(np.min(self.batch_result_matrix), np.max(self.batch_result_matrix))
            
            self.lbl_batch_status.setText(f"狀態: 已載入第 {idx+1} 組")
            
            # 連動計算座標與距離
            self.update_batch_calculations()
            
            # 繪製下方的分布圖與趨勢圖
            self.render_sub_plots_fast(self.batch_result_matrix)
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入失敗: {str(e)}")

    def batch_go_prev(self):
        if self.batch_total_count > 0:
            self.load_batch_group((self.batch_current_idx - 1) % self.batch_total_count)

    def batch_go_next(self):
        if self.batch_total_count > 0:
            self.load_batch_group((self.batch_current_idx + 1) % self.batch_total_count)

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
        if self.batch_total_count == 0 or not self.save_dir_path:
            QMessageBox.warning(self, "警告", "請先確認資料載入且選擇儲存資料夾！")
            return
        zip_path = os.path.join(self.save_dir_path, "DataRay_Batch_Results.zip")
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx in range(self.batch_total_count):
                    f1 = self.batch_m1_files[idx]
                    zipf.writestr(f"Group_{idx+1:02d}_info.txt", f"M1: {os.path.basename(f1)}")
            QMessageBox.information(self, "成功", f"ZIP 壓縮檔已匯出至：\n{zip_path}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"匯出 ZIP 發生錯誤: {str(e)}")

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
            pen = pg.mkPen('y', width=2, style=Qt.DashLine)
            v_item = pg.PlotCurveItem(x=[cx, cx], y=[0, h], pen=pen)
            h_item = pg.PlotCurveItem(x=[0, w], y=[cy, cy], pen=pen)
            self.plot_batch_heat.addItem(v_item)
            self.plot_batch_heat.addItem(h_item)
            self.batch_cross_items.extend([v_item, h_item])

        if self.batch_m2_center_point:
            cx2, cy2 = self.batch_m2_center_point
            pen2 = pg.mkPen('c', width=2)
            v_item2 = pg.PlotCurveItem(x=[cx2, cx2], y=[0, h], pen=pen2)
            h_item2 = pg.PlotCurveItem(x=[0, w], y=[cy2, cy2], pen=pen2)
            self.plot_batch_heat.addItem(v_item2)
            self.plot_batch_heat.addItem(h_item2)
            self.batch_cross_items.extend([v_item2, h_item2])

    # ================== 核心計算與邏輯連動 ==================
    def _compute_auto_spot_center(self, matrix, mode, use_threshold=False, thresh_percent=50.0):
        matrix = np.asarray(matrix, dtype=np.float64)
        h, w = matrix.shape
        peak_val = float(np.max(matrix)) if matrix.size > 0 else 0.0

        if mode == "peak_geom":
            max_y_indices, max_x_indices = np.where(matrix == peak_val)
            if len(max_x_indices) > 0:
                return int(round(np.mean(max_x_indices))), int(round(np.mean(max_y_indices)))
            peak_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
            return int(peak_idx[1]), int(peak_idx[0])

        thresh_val = peak_val * (thresh_percent / 100.0) if use_threshold else peak_val * 0.5
        mask = matrix >= thresh_val
        
        if not np.any(mask):
            peak_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
            return int(peak_idx[1]), int(peak_idx[0])

        ys, xs = np.where(mask)
        if mode == "thresh_geom":
            return int(round(np.mean(xs))), int(round(np.mean(ys)))

        weights = matrix[mask]
        wsum = float(np.sum(weights))
        if wsum <= 0:
            return int(round(np.mean(xs))), int(round(np.mean(ys)))
        cx = int(round(np.sum(xs * weights) / wsum))
        cy = int(round(np.sum(ys * weights) / wsum))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        return cx, cy

    def _build_threshold_mask(self, matrix, use_threshold, thresh_percent, y_below=None):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.size == 0: return None
        h, w = matrix.shape
        if y_below is not None:
            y_below = int(y_below)
            if y_below <= 0: return np.zeros((h, w), dtype=bool)
            region = matrix[:y_below, :]
            peak_val = float(np.max(region)) if region.size > 0 else 0.0
            thresh_val = peak_val * (thresh_percent / 100.0) if use_threshold else peak_val * 0.5
            mask = np.zeros((h, w), dtype=bool)
            mask[:y_below, :] = region >= thresh_val
            return mask
        peak_val = float(np.max(matrix))
        thresh_val = peak_val * (thresh_percent / 100.0) if use_threshold else peak_val * 0.5
        return matrix >= thresh_val

    def on_batch_m1_point_mode_changed(self, checked=False):
        self.update_batch_calculations()

    def on_batch_m1_threshold_toggled(self, checked):
        self.spin_batch_m1_thresh_percent.setEnabled(checked)
        self.update_batch_calculations()

    def on_batch_m1_thresh_percent_changed(self):
        self.update_batch_calculations()

    def on_batch_p2_point_mode_changed(self, checked=False):
        self.update_batch_calculations()
        
    def on_batch_m2_thresh_percent_changed(self):
        self.update_batch_calculations()

    def update_batch_calculations(self):
        if not hasattr(self, 'matrix1') or self.matrix1 is None: return
        if not hasattr(self, 'matrix2') or self.matrix2 is None: return

        # 1. 運算 M1 座標
        m1_mode = "centroid"
        if self.radio_batch_m1_thresh_geom.isChecked(): m1_mode = "thresh_geom"
        elif self.radio_batch_m1_peak_geom.isChecked(): m1_mode = "peak_geom"
        elif self.radio_batch_m1_manual.isChecked(): m1_mode = "manual"

        if m1_mode != "manual":
            use_thresh = self.chk_batch_m1_use_threshold.isChecked()
            thresh_percent = self.spin_batch_m1_thresh_percent.value()
            m1_x, m1_y = self._compute_auto_spot_center(self.matrix1, m1_mode, use_thresh, thresh_percent)
            self.batch_m1_center_point = (m1_x, m1_y)
        else:
            if hasattr(self, 'batch_m1_center_point') and self.batch_m1_center_point:
                m1_x, m1_y = self.batch_m1_center_point
            else:
                m1_x, m1_y = self.matrix1.shape[1]//2, self.matrix1.shape[0]//2
                self.batch_m1_center_point = (m1_x, m1_y)

        m1_mask = self._build_threshold_mask(self.matrix1, self.chk_batch_m1_use_threshold.isChecked(), self.spin_batch_m1_thresh_percent.value())

        # 2. 運算 M2 座標
        if self.radio_batch_m2_auto_m1.isChecked():
            m2_x, m2_y = m1_x, m1_y 
        elif self.radio_batch_m2_auto_global.isChecked():
            m2_y, m2_x = np.unravel_index(np.argmax(self.matrix2, axis=None), self.matrix2.shape)
        elif self.radio_batch_m2_manual.isChecked():
            if hasattr(self, 'batch_m2_center_point') and self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
            else:
                m2_x, m2_y = m1_x, m1_y

        self.batch_m2_center_point = (m2_x, m2_y)
        m2_mask = self._build_threshold_mask(self.matrix2, True, self.spin_batch_m2_thresh_percent.value())

        # 3. 計算直線距離
        distance = np.sqrt((m1_x - m2_x)**2 + (m1_y - m2_y)**2)
        print(f"[Batch 運算] M1({m1_x}, {m1_y}) | M2({m2_x}, {m2_y}) | 距離: {distance:.2f} px")

        # 4. 更新主畫面十字線
        self.redraw_batch_crosses()

        # 5. 更新彈出視窗
        if getattr(self, 'viewer_batch_m1_win', None) is not None:
            self.viewer_batch_m1_win.draw_marker((m1_x, m1_y))
            if self.chk_batch_m1_show_thresh.isChecked():
                self.viewer_batch_m1_win.set_threshold_overlay(m1_mask, visible=True, rgba_color=(0, 255, 0, 90))
            else:
                self.viewer_batch_m1_win.clear_threshold_overlay()

        if getattr(self, 'viewer_batch_m2_win', None) is not None:
            self.viewer_batch_m2_win.draw_marker((m1_x, m1_y), pt2=(m2_x, m2_y))
            if self.chk_batch_p2_show_thresh.isChecked():
                self.viewer_batch_m2_win.set_threshold_overlay(m2_mask, visible=True, rgba_color=(255, 64, 255, 90))
            else:
                self.viewer_batch_m2_win.clear_threshold_overlay()

    def save_current_batch_params(self):
        if self.batch_total_count == 0:
            QMessageBox.warning(self, "警告", "目前沒有載入任何 Batch 資料！")
            return
        
        m1_mode = "centroid"
        if self.radio_batch_m1_thresh_geom.isChecked(): m1_mode = "thresh_geom"
        elif self.radio_batch_m1_peak_geom.isChecked(): m1_mode = "peak_geom"
        elif self.radio_batch_m1_manual.isChecked(): m1_mode = "manual"
        
        m2_mode = "auto_m1"
        if self.radio_batch_m2_auto_global.isChecked(): m2_mode = "auto_global"
        elif self.radio_batch_m2_manual.isChecked(): m2_mode = "manual"
        
        self.batch_saved_params[self.batch_current_idx] = {
            "m1_mode": m1_mode,
            "use_thresh": self.chk_batch_m1_use_threshold.isChecked(),
            "thresh_percent": self.spin_batch_m1_thresh_percent.value(),
            "m1_center_point": getattr(self, 'batch_m1_center_point', None),
            
            "m2_mode": m2_mode,
            "m2_thresh_percent": self.spin_batch_m2_thresh_percent.value(),
            "m2_center_point": getattr(self, 'batch_m2_center_point', None)
        }
        QMessageBox.information(self, "暫存成功", f"第 {self.batch_current_idx + 1} 組參數與位置已暫存！")