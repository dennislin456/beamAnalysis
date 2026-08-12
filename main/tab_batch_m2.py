"""DataRay M2-only Batch：僅載入 M2 Excel，以質心縱切雙峰波谷 Y 區分 above／below。

可選框選 ROI（X/Y/Width/Height）：僅在固定矩形內做縱切找波谷，避免散射雜點干擾。
"""
import os
import json
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as pg_export
from scipy.ndimage import uniform_filter

from PyQt5.QtWidgets import (
    QLabel, QFileDialog, QMessageBox, QApplication, QDialog,
    QHBoxLayout, QGridLayout, QCheckBox,
)
from PyQt5.QtCore import Qt

import openpyxl

from shared_components import (
    HeatmapViewerWindow, find_dual_peak_valley_y, split_y_index,
    NoWheelSpinBox, clip_roi_to_matrix,
)
from batch_data_loader import load_numeric_matrix
from tab_batch import DataRayBatchTab, LocationConfigDialog


class DataRayBatchM2Tab(DataRayBatchTab):
    """批量 M2：不需 M1；切分線 = 質心 X 縱切雙峰波谷 Y。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_split_y = None          # 波谷 Y（取代 M1 Y）
        self.batch_split_cx = None         # 縱切所用質心 X
        self.batch_split_peak_ys = None    # (y_lo, y_hi)
        self.batch_valley_roi_bounds = None  # 實際使用的 (x0,y0,x1,y1)
        self._adapt_ui_for_m2_only()

    # ------------------------------------------------------------------
    # UI 調整：隱藏 M1／雙檔運算，改寫文案
    # ------------------------------------------------------------------
    def _adapt_ui_for_m2_only(self):
        # 隱藏雙檔運算模式
        self.combo_batch_mode.hide()
        self.chk_batch_normalize.hide()
        for w in self.findChildren(QLabel):
            if w.text().startswith("選擇工作模式"):
                w.hide()
                break

        # 隱藏 M1 資料夾
        self.btn_batch_m1_dir.hide()
        self.lbl_batch_m1_info.hide()

        self.btn_batch_m2_dir.setText("I. 選擇 M2 主資料夾（含位置子資料夾）")
        self.lbl_batch_m2_info.setText("未選擇 M2 主資料夾\n格式: M2/<位置>/<檔名>.xlsx / .csv / .npy")
        self.lbl_batch_pair_info.setText("掃描結果: 尚未選擇 M2 主資料夾")

        # 隱藏 M1→below 距離列
        for attr in (
            "lbl_batch_max1", "lbl_batch_max2", "lbl_batch_ratio",
            "lbl_batch_distance_xy", "lbl_batch_distance_total",
            "lbl_batch_real_distance_xy", "lbl_batch_real_distance_total",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.hide()

        # 新增波谷切分狀態
        self.lbl_batch_split_info = QLabel("切分波谷 Y: -- ｜ 縱切 X: --")
        self.lbl_batch_split_info.setStyleSheet(
            "font-weight: bold; color: #6A1B9A; font-size: 12px;"
        )
        self.lbl_batch_split_info.setWordWrap(True)
        # 插在矩陣大小之後、above→below 距離之前
        size_lbl = self.lbl_batch_size
        parent_layout = size_lbl.parentWidget().layout()
        if parent_layout is not None:
            idx = parent_layout.indexOf(size_lbl)
            parent_layout.insertWidget(idx + 1, self.lbl_batch_split_info)

        # M1 定位區塊隱藏
        self.radio_batch_m1_centroid.hide()
        self.radio_batch_m1_thresh_geom.hide()
        self.radio_batch_m1_peak_geom.hide()
        self.radio_batch_m1_manual.hide()
        self.chk_batch_m1_use_threshold.hide()
        self.lbl_batch_m1_thresh_spin.hide()
        self.spin_batch_m1_thresh_percent.hide()
        self.chk_batch_m1_show_thresh.hide()
        for w in self.findChildren(QLabel):
            t = w.text()
            if "M1 / 雙點定位模式" in t or t.startswith("第一點（M1"):
                w.hide()

        # 第二點文案改為波谷切分
        for w in self.findChildren(QLabel):
            if "第二點（Process Result" in w.text():
                w.setText("光斑定位（M2 Heatmap，依波谷 Y 切分）：")
                w.setStyleSheet("font-weight: bold; color: #1565C0;")

        left_layout = getattr(self, "batch_left_settings_layout", None)
        if left_layout is None:
            left_layout = self.radio_batch_p2_auto_min.parentWidget().layout()
        insert_idx = left_layout.indexOf(self.radio_batch_p2_auto_min)

        # 波谷搜尋框選：僅調整 XYWH 輸入排成緊湊 2×2
        self.chk_batch_valley_roi = QCheckBox(
            "啟用波谷搜尋框選（僅在框內縱切找低谷，區分 above／below）"
        )
        self.chk_batch_valley_roi.setChecked(False)
        self.chk_batch_valley_roi.setStyleSheet("color: #6A1B9A; font-weight: bold;")
        self.chk_batch_valley_roi.setToolTip(
            "框住上下主光斑，避開上方散射雜點；切分後 above／below 仍對全圖計算。"
        )
        self.chk_batch_valley_roi.toggled.connect(self._on_valley_roi_toggled)
        left_layout.insertWidget(insert_idx, self.chk_batch_valley_roi)
        insert_idx += 1

        roi_grid = QGridLayout()
        roi_grid.setContentsMargins(0, 0, 0, 0)
        roi_grid.setHorizontalSpacing(6)
        roi_grid.setVerticalSpacing(4)
        self.lbl_batch_valley_roi_x = QLabel("X")
        self.spin_batch_valley_roi_x = NoWheelSpinBox()
        self.spin_batch_valley_roi_x.setRange(0, 100000)
        self.spin_batch_valley_roi_x.setValue(0)
        self.spin_batch_valley_roi_x.setFixedWidth(72)
        self.spin_batch_valley_roi_x.valueChanged.connect(self._on_valley_roi_changed)
        self.lbl_batch_valley_roi_y = QLabel("Y")
        self.spin_batch_valley_roi_y = NoWheelSpinBox()
        self.spin_batch_valley_roi_y.setRange(0, 100000)
        self.spin_batch_valley_roi_y.setValue(0)
        self.spin_batch_valley_roi_y.setFixedWidth(72)
        self.spin_batch_valley_roi_y.valueChanged.connect(self._on_valley_roi_changed)
        self.lbl_batch_valley_roi_w = QLabel("W")
        self.spin_batch_valley_roi_w = NoWheelSpinBox()
        self.spin_batch_valley_roi_w.setRange(1, 100000)
        self.spin_batch_valley_roi_w.setValue(200)
        self.spin_batch_valley_roi_w.setFixedWidth(72)
        self.spin_batch_valley_roi_w.valueChanged.connect(self._on_valley_roi_changed)
        self.lbl_batch_valley_roi_h = QLabel("H")
        self.spin_batch_valley_roi_h = NoWheelSpinBox()
        self.spin_batch_valley_roi_h.setRange(1, 100000)
        self.spin_batch_valley_roi_h.setValue(200)
        self.spin_batch_valley_roi_h.setFixedWidth(72)
        self.spin_batch_valley_roi_h.valueChanged.connect(self._on_valley_roi_changed)
        roi_grid.addWidget(self.lbl_batch_valley_roi_x, 0, 0)
        roi_grid.addWidget(self.spin_batch_valley_roi_x, 0, 1)
        roi_grid.addWidget(self.lbl_batch_valley_roi_y, 0, 2)
        roi_grid.addWidget(self.spin_batch_valley_roi_y, 0, 3)
        roi_grid.addWidget(self.lbl_batch_valley_roi_w, 1, 0)
        roi_grid.addWidget(self.spin_batch_valley_roi_w, 1, 1)
        roi_grid.addWidget(self.lbl_batch_valley_roi_h, 1, 2)
        roi_grid.addWidget(self.spin_batch_valley_roi_h, 1, 3)
        left_layout.insertLayout(insert_idx, roi_grid)

        self.lbl_batch_valley_roi_hint = QLabel("")
        self.lbl_batch_valley_roi_hint.hide()
        self._set_valley_roi_controls_enabled(False)

        # 門檻：checkbox + 比例同一列、縮小輸入框
        self._compact_threshold_row(left_layout)

        self.radio_batch_p2_auto_min.setText("自動抓取 (切分 Y 以下最小值)")
        self.radio_batch_p2_m2_thresh_geom.setText("自動抓取 (M2 切分 Y 以下門檻幾何中心)")
        self.radio_batch_p2_m2_centroid.setText("自動抓取 (M2 切分 Y 以下質心中心)")
        self.radio_batch_p2_m2_inscribed.setText(
            "自動抓取 (M2 門檻 contour 內切圓中心，Y 上／下)"
        )
        self.radio_batch_p2_manual.setText("手動抓取 (點擊 M2 影像)")
        self.radio_batch_p2_m2_centroid.setChecked(True)

        self.chk_batch_show_cross.setText(
            "顯示 切分線(紫) / 框選(橙) / M2-below(藍) / M2-above(紅) 十字"
        )

        self.btn_batch_view_m1.hide()
        self.btn_batch_view_m2.setText("查看 M2 Heatmap（彈窗）")
        self.plot_batch_heat.setTitle("M2 Heatmap (Batch M2-only)")
        self.btn_batch_run.setText("載入已配置位置並運算（僅 M2）")

    def _compact_threshold_row(self, left_layout):
        """把門檻 checkbox 與 % 輸入排成同一列，縮小 spin 寬度。"""
        # 移除原獨立列
        left_layout.removeWidget(self.chk_batch_p2_use_threshold)
        old_thresh_layout = None
        for i in range(left_layout.count()):
            item = left_layout.itemAt(i)
            lay = item.layout() if item is not None else None
            if lay is None:
                continue
            for j in range(lay.count()):
                child = lay.itemAt(j)
                w = child.widget() if child is not None else None
                if w is self.spin_batch_p2_thresh_percent:
                    old_thresh_layout = left_layout.takeAt(i)
                    break
            if old_thresh_layout is not None:
                break

        if old_thresh_layout is not None and old_thresh_layout.layout() is not None:
            old_lay = old_thresh_layout.layout()
            while old_lay.count():
                old_lay.takeAt(0)

        self.chk_batch_p2_use_threshold.setText("使用門檻（第二點／M2）")
        self.lbl_batch_p2_thresh_spin.setText("%:")
        self.spin_batch_p2_thresh_percent.setFixedWidth(72)

        thresh_row = QHBoxLayout()
        thresh_row.setContentsMargins(0, 0, 0, 0)
        thresh_row.setSpacing(6)
        thresh_row.addWidget(self.chk_batch_p2_use_threshold)
        thresh_row.addWidget(self.lbl_batch_p2_thresh_spin)
        thresh_row.addWidget(self.spin_batch_p2_thresh_percent)
        thresh_row.addStretch(1)

        show_idx = left_layout.indexOf(self.chk_batch_p2_show_thresh)
        if show_idx < 0:
            # 插在 manual radio 之後
            show_idx = left_layout.indexOf(self.radio_batch_p2_manual)
            if show_idx >= 0:
                show_idx += 1
            else:
                show_idx = left_layout.count()
        left_layout.insertLayout(show_idx, thresh_row)

    def _set_valley_roi_controls_enabled(self, enabled):
        for w in (
            self.lbl_batch_valley_roi_x, self.spin_batch_valley_roi_x,
            self.lbl_batch_valley_roi_y, self.spin_batch_valley_roi_y,
            self.lbl_batch_valley_roi_w, self.spin_batch_valley_roi_w,
            self.lbl_batch_valley_roi_h, self.spin_batch_valley_roi_h,
        ):
            w.setEnabled(enabled)
            w.setVisible(enabled)

    def _on_valley_roi_toggled(self, checked):
        self._set_valley_roi_controls_enabled(checked)
        if hasattr(self, "matrix2") and self.matrix2 is not None:
            self.update_batch_calculations(silent=True)

    def _on_valley_roi_changed(self, _value=None):
        if not getattr(self, "chk_batch_valley_roi", None):
            return
        if not self.chk_batch_valley_roi.isChecked():
            return
        if hasattr(self, "matrix2") and self.matrix2 is not None:
            self.update_batch_calculations(silent=True)

    def _get_valley_roi_tuple(self):
        """回傳使用者輸入的 (x, y, width, height)；未啟用則 None。"""
        if not getattr(self, "chk_batch_valley_roi", None):
            return None
        if not self.chk_batch_valley_roi.isChecked():
            return None
        return (
            self.spin_batch_valley_roi_x.value(),
            self.spin_batch_valley_roi_y.value(),
            self.spin_batch_valley_roi_w.value(),
            self.spin_batch_valley_roi_h.value(),
        )

    # ------------------------------------------------------------------
    # 資料掃描／載入（僅 M2）
    # ------------------------------------------------------------------
    def _rebuild_batch_pairs(self):
        self.batch_pairs = []
        self.batch_m1_files = []
        self.batch_m2_files = []
        self.batch_all_pairs_by_loc = {}
        self.batch_available_locations = []
        self.batch_matrix_cache.clear()
        self.batch_result_cache.clear()

        if not self.batch_m2_root:
            if hasattr(self, "lbl_batch_pair_info"):
                self.lbl_batch_pair_info.setText("掃描結果: 請選擇 M2 主資料夾")
            if hasattr(self, "btn_location_config"):
                self.btn_location_config.setEnabled(False)
            self.batch_location_config = {}
            self._update_selected_location_info()
            return

        m2_map = self._scan_location_files(self.batch_m2_root)
        locs = sorted(m2_map.keys(), key=self._natural_sort_key)

        pairs_by_loc = {}
        total_pairs = 0
        for loc in locs:
            files = sorted(m2_map[loc].keys(), key=self._natural_sort_key)
            loc_pairs = []
            for fname in files:
                loc_pairs.append({
                    "location": loc,
                    "filename": fname,
                    "m1_path": None,
                    "m2_path": m2_map[loc][fname],
                })
            if loc_pairs:
                pairs_by_loc[loc] = loc_pairs
                total_pairs += len(loc_pairs)

        self.batch_all_pairs_by_loc = pairs_by_loc
        self.batch_available_locations = list(pairs_by_loc.keys())

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

        msg = (
            f"辨識到 {len(self.batch_available_locations)} 個位置，"
            f"共 {total_pairs} 組 M2 量測（請點「位置配置」選擇位置與 cycle）"
        )
        if hasattr(self, "lbl_batch_pair_info"):
            self.lbl_batch_pair_info.setText(msg)
        if hasattr(self, "btn_location_config"):
            self.btn_location_config.setEnabled(bool(self.batch_available_locations))
        self._update_selected_location_info()

    def open_location_config_dialog(self):
        if not self.batch_available_locations:
            QMessageBox.warning(self, "警告", "尚無可配置的位置，請先匯入 M2 主資料夾。")
            return
        dlg = LocationConfigDialog(
            self.batch_available_locations,
            self.batch_all_pairs_by_loc,
            current_config=self.batch_location_config,
            parent=self,
        )
        dlg.setWindowTitle("位置配置（M2-only）")
        if dlg.exec_() == QDialog.Accepted:
            self.batch_location_config = dlg.get_config()
            self._update_selected_location_info()

    def _apply_selected_pairs(self):
        selected = self._get_selected_locations()
        pairs = []
        for loc in selected:
            wanted = set(self.batch_location_config.get(loc, {}).get("cycles", []))
            for p in self.batch_all_pairs_by_loc.get(loc, []):
                if p["filename"] in wanted:
                    pairs.append(p)
        self.batch_pairs = pairs
        self.batch_m1_files = []
        self.batch_m2_files = [p["m2_path"] for p in pairs]
        return selected, pairs

    def load_batch_m1_folder(self):
        """M2-only：不使用 M1。"""
        QMessageBox.information(self, "提示", "此分頁僅需載入 M2，不需選擇 M1。")

    def load_batch_m2_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "選擇 M2 主資料夾（內含位置子資料夾）", ""
        )
        if not dir_path:
            return
        loc_map = self._scan_location_files(dir_path)
        if not loc_map:
            QMessageBox.warning(
                self, "警告",
                "此資料夾下找不到「位置子資料夾／Excel或CSV」結構。\n"
                "預期格式：M2/<位置名>/<檔名>.xlsx 或 <檔名>.csv\n"
                "或如 beamImage/… 下各位置資料夾內的 .xlsx/.csv"
            )
            return
        self.batch_m2_root = dir_path
        self.batch_m1_root = ""  # 明確不使用
        n_files = sum(len(v) for v in loc_map.values())
        locs = sorted(loc_map.keys(), key=self._natural_sort_key)
        self.lbl_batch_m2_info.setText(
            f"{os.path.basename(dir_path)}｜位置 {len(locs)} 個｜檔案 {n_files} 筆\n"
            f"位置: {', '.join(locs[:10])}{'...' if len(locs) > 10 else ''}"
        )
        if not self.save_dir_path:
            self.save_dir_path = dir_path
            self.lbl_batch_dir_path.setText(f"{dir_path}")
        self._rebuild_batch_pairs()

    def process_batch_data(self):
        if not self.batch_m2_root:
            QMessageBox.warning(self, "警告", "請先選擇 M2 主資料夾！")
            return
        if not self.batch_available_locations:
            self._rebuild_batch_pairs()
        if not self.batch_available_locations:
            QMessageBox.warning(
                self, "警告",
                "找不到任何位置子資料夾內的 Excel/CSV！\n"
                "請確認結構為 M2/<位置>/<檔名>.xlsx 或 <檔名>.csv"
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
                f"狀態: 正在預載 {len(selected)} 個位置（{self.batch_total_count} 組 M2）..."
            )
            self.lbl_batch_status.setStyleSheet(
                "color: #F57C00; font-weight: bold; font-size: 12px;"
            )
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
            self.btn_batch_view_m1.setEnabled(False)
            self.btn_batch_view_m2.setEnabled(True)
            self.btn_batch_view_cross.setEnabled(True)
            self.lbl_batch_status.setText(
                f"狀態: 已載入 {len(selected)} 個位置｜共 {self.batch_total_count} 組（M2-only）"
            )
            self.lbl_batch_status.setStyleSheet(
                "color: #2E7D32; font-weight: bold; font-size: 12px;"
            )
        except Exception as e:
            self.btn_batch_run.setEnabled(True)
            self.lbl_batch_status.setText("狀態: 預載失敗")
            self.lbl_batch_status.setStyleSheet(
                "color: #C62828; font-weight: bold; font-size: 12px;"
            )
            QMessageBox.critical(self, "錯誤", f"預載失敗: {str(e)}")

    def _ensure_matrix_cached(self, idx):
        if idx in self.batch_matrix_cache:
            return self.batch_matrix_cache[idx]
        m2 = self._read_excel_matrix(self.batch_m2_files[idx])
        # 與父類相容：cache 仍為 (m1, m2)，此處 m1=None
        self.batch_matrix_cache[idx] = (None, m2)
        return None, m2

    def _read_excel_matrix(self, path):
        return load_numeric_matrix(path)

    def _result_cache_key(self, idx):
        return (idx, "m2_only")

    def _compute_group_result(self, matrix1, matrix2):
        """M2-only：結果矩陣即 M2 本身。"""
        max_idx = np.unravel_index(np.argmax(matrix2, axis=None), matrix2.shape)
        max_val = float(matrix2[max_idx])
        scale_info = {
            "mode": "m2_only",
            "max1_val": max_val,
            "match2_val": max_val,
            "scale_ratio": None,
            "max1_idx": max_idx,
        }
        return matrix2.copy(), scale_info

    def load_batch_group(self, idx):
        if idx < 0 or idx >= self.batch_total_count:
            return
        try:
            self.batch_current_idx = idx
            group_text = f"{idx + 1} / {self.batch_total_count}"
            self.lbl_batch_group_num.setText(group_text)
            if hasattr(self, "lbl_batch_group_nav"):
                self.lbl_batch_group_nav.setText(group_text)

            f2 = self.batch_m2_files[idx]
            pair = self.batch_pairs[idx] if idx < len(self.batch_pairs) else None
            if hasattr(self, "lbl_batch_current_files"):
                if pair:
                    self.lbl_batch_current_files.setText(
                        f"目前量測:\n"
                        f"位置: {pair['location']} ｜ 檔名: {pair['filename']}\n"
                        f"M2: .../{pair['location']}/{pair['filename']}"
                    )
                else:
                    self.lbl_batch_current_files.setText(
                        f"目前檔案:\nM2: {os.path.basename(f2)}"
                    )

            _m1, self.matrix2 = self._ensure_matrix_cached(idx)
            self.matrix1 = None

            cache_key = self._result_cache_key(idx)
            cached = self.batch_result_cache.get(cache_key)
            if cached is not None:
                self.batch_result_matrix, self.batch_scale_info = cached
            else:
                result, scale_info = self._compute_group_result(None, self.matrix2)
                self.batch_result_matrix = result
                self.batch_scale_info = scale_info
                self.batch_result_cache[cache_key] = (result, scale_info)

            self.batch_image_item.setImage(self.batch_result_matrix.T, autoLevels=False)
            self.batch_hist.setLevels(
                float(np.min(self.batch_result_matrix)),
                float(np.max(self.batch_result_matrix)),
            )

            # 載入後強制對齊檔案 XY 範圍（避免卡在 0,0 邊緣）
            self._batch_profile_view_ready = False
            if self.chk_batch_pixel_profile.isChecked():
                h, w = self.batch_result_matrix.shape
                if self.batch_profile_point is None:
                    self.set_batch_profile_point(w // 2, h // 2, reset_view=True)
                else:
                    cx, cy = self.batch_profile_point
                    self.set_batch_profile_point(
                        min(cx, w - 1), min(cy, h - 1), reset_view=True
                    )
            else:
                self.fit_batch_heatmap_to_data()

            self.lbl_batch_status.setText(
                f"狀態: 已載入第 {idx + 1}/{self.batch_total_count} 組（M2-only）"
            )
            self.lbl_batch_status.setStyleSheet(
                "color: #2E7D32; font-weight: bold; font-size: 12px;"
            )

            self._apply_saved_batch_params(idx)
            self.update_batch_calculations(silent=True)
            self.render_sub_plots_fast(self.batch_result_matrix)
            # 計算／標記完成後再對齊一次視野
            self.fit_batch_heatmap_to_data()
            if self.chk_batch_pixel_profile.isChecked() and self.batch_profile_point is not None:
                self.update_batch_inline_profiles(reset_view=True)
            self._refresh_open_batch_viewers()
        except Exception as e:
            self.lbl_batch_status.setText("狀態: 載入失敗")
            self.lbl_batch_status.setStyleSheet(
                "color: #C62828; font-weight: bold; font-size: 12px;"
            )
            QMessageBox.critical(self, "錯誤", f"載入失敗: {str(e)}")

    def show_batch_m1_heatmap(self):
        QMessageBox.information(self, "提示", "M2-only 分頁沒有 M1 資料。")

    def show_batch_m2_heatmap(self):
        if self.matrix2 is not None:
            if getattr(self, "viewer_batch_m2_win", None) is not None:
                self.viewer_batch_m2_win.close()
            self.viewer_batch_m2_win = HeatmapViewerWindow(
                "Batch M2 Heatmap (M2-only)", self.matrix2, app_parent=self, is_m1=False
            )
            self.viewer_batch_m2_win.setGeometry(800, 150, 700, 650)
            self.viewer_batch_m2_win.show()
            self.update_batch_calculations()

    def on_batch_mode_or_norm_changed(self, *_args):
        # M2-only 無運算模式切換
        pass

    def on_batch_heatmap_colormap_toggled(self, checked=False):
        cmap = self.batch_gray_map if self.chk_batch_heatmap_gray.isChecked() else self.batch_jet_map
        levels = self.batch_hist.getLevels()
        self.batch_hist.gradient.setColorMap(cmap)
        self.batch_hist.setLevels(*levels)
        title = "M2 Heatmap (Batch M2-only)"
        if self.chk_batch_heatmap_gray.isChecked():
            title += " [Grayscale]"
        self.plot_batch_heat.setTitle(title)

        win = getattr(self, "viewer_batch_m2_win", None)
        if win is not None:
            try:
                if win.isVisible():
                    win.set_grayscale(self.chk_batch_heatmap_gray.isChecked())
            except RuntimeError:
                self.viewer_batch_m2_win = None

    # ------------------------------------------------------------------
    # 核心：波谷切分 + M2 above／below
    # ------------------------------------------------------------------
    def _compute_split_y_from_m2(self):
        """以 M2 質心 X 縱切，找雙峰波谷 Y（可限於框選 ROI）。"""
        roi = self._get_valley_roi_tuple()
        info = find_dual_peak_valley_y(self.matrix2, roi=roi)
        self.batch_split_y = info["valley_y"]
        self.batch_split_cx = info["cx"]
        self.batch_split_peak_ys = info["peak_ys"]
        self.batch_valley_roi_bounds = info.get("roi")
        # 相容：把「切分點」放在 batch_m1_center_point，供匯出／十字複用欄位語意
        self.batch_m1_center_point = (self.batch_split_cx, self.batch_split_y)
        return self.batch_split_y

    def _compute_m2_above_point(self, split_y, p2_mode, silent=False):
        """以與 below 相同方法，在切分 Y 以上找 M2 above。"""
        use_thresh = self.chk_batch_p2_use_threshold.isChecked()
        thresh_percent = self.spin_batch_p2_thresh_percent.value()
        self.batch_m2_above_circle_r = None

        if self.matrix2 is None:
            return None

        if p2_mode == "auto_min":
            return self._find_min_above_y(self.matrix2, split_y)

        if p2_mode == "m2_inscribed":
            result = self._find_inscribed_circle_above_y(
                self.matrix2, split_y, use_thresh, thresh_percent
            )
            if result is None:
                return None
            cx, cy, radius = result
            self.batch_m2_above_circle_r = radius
            return (cx, cy)

        if p2_mode == "m2_thresh_geom":
            center_mode = "thresh_geom"
        elif p2_mode == "m2_centroid":
            center_mode = "centroid"
        else:
            center_mode = "centroid"
        return self._find_center_above_y(
            self.matrix2, split_y, center_mode, use_thresh, thresh_percent
        )

    def update_batch_calculations(self, silent=False):
        if not hasattr(self, "matrix2") or self.matrix2 is None:
            return
        if self.batch_result_matrix is None:
            return

        # 1. 波谷切分（取代 M1 Y）
        split_y = self._compute_split_y_from_m2()
        split_y_i = split_y_index(split_y)

        # 2. below／above（與 Batch 相同 M2 方法）
        p2_mode = self._get_p2_point_mode_name()
        m2_x, m2_y = None, None
        self.batch_m2_below_circle_r = None

        if p2_mode == "manual":
            if self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
            else:
                m2_x = self.batch_split_cx if self.batch_split_cx is not None else self.matrix2.shape[1] // 2
                m2_y = max(0, split_y_i - 1)
                self.batch_m2_center_point = (m2_x, m2_y)
        elif p2_mode == "auto_min":
            p2 = self._find_min_below_y(self.matrix2, split_y)
            if p2 is None:
                if not silent:
                    QMessageBox.warning(
                        self, "警告", "切分 Y 以下沒有可搜尋區域，無法自動抓取 below。"
                    )
            else:
                m2_x, m2_y = p2
        elif p2_mode == "m2_inscribed":
            use_thresh = self.chk_batch_p2_use_threshold.isChecked()
            thresh_percent = self.spin_batch_p2_thresh_percent.value()
            p2 = self._find_inscribed_circle_below_y(
                self.matrix2, split_y, use_thresh, thresh_percent
            )
            if p2 is None:
                if not silent:
                    QMessageBox.warning(
                        self, "警告",
                        "切分 Y 以下無法以門檻 contour 擬合內切圓，請調整門檻或確認光斑。"
                    )
            else:
                m2_x, m2_y, self.batch_m2_below_circle_r = p2
        else:
            use_thresh = self.chk_batch_p2_use_threshold.isChecked()
            thresh_percent = self.spin_batch_p2_thresh_percent.value()
            center_mode = "thresh_geom" if p2_mode == "m2_thresh_geom" else "centroid"
            p2 = self._find_center_below_y(
                self.matrix2, split_y, center_mode, use_thresh, thresh_percent
            )
            if p2 is None:
                if not silent:
                    QMessageBox.warning(
                        self, "警告", "切分 Y 以下沒有可搜尋區域，無法自動抓取 below。"
                    )
            else:
                m2_x, m2_y = p2

        if m2_x is None or m2_y is None:
            if self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
            else:
                m2_x = self.batch_split_cx if self.batch_split_cx is not None else 0
                m2_y = max(0, split_y_i - 1)

        self.batch_m2_center_point = (m2_x, m2_y)

        m2a = self._compute_m2_above_point(split_y, p2_mode, silent=silent)
        if m2a is None:
            self.batch_m2_above_point = None
            m2a_x = m2a_y = None
        else:
            m2a_x, m2a_y = m2a
            self.batch_m2_above_point = (m2a_x, m2a_y)

        use_p2_thresh = self.chk_batch_p2_use_threshold.isChecked()
        p2_pct = self.spin_batch_p2_thresh_percent.value()
        m2_mask_below = self._build_threshold_mask(
            self.matrix2, use_p2_thresh, p2_pct, y_below=split_y
        )
        m2_mask_above = self._build_threshold_mask(
            self.matrix2, use_p2_thresh, p2_pct, y_above=split_y
        )
        if m2_mask_below is not None and m2_mask_above is not None:
            m2_mask = np.logical_or(m2_mask_below, m2_mask_above)
        else:
            m2_mask = m2_mask_below if m2_mask_below is not None else m2_mask_above

        if m2a_x is not None and m2a_y is not None:
            dist_ab = np.sqrt((m2a_x - m2_x) ** 2 + (m2a_y - m2_y) ** 2)
            print(
                f"[Batch M2-only] splitY={split_y:.2f} cx={self.batch_split_cx:.2f} | "
                f"roi={self.batch_valley_roi_bounds} | "
                f"below({m2_x:.2f}, {m2_y:.2f}) | above({m2a_x:.2f}, {m2a_y:.2f}) | "
                f"above→below: {dist_ab:.2f} px"
            )
        else:
            print(
                f"[Batch M2-only] splitY={split_y:.2f} | "
                f"below({m2_x:.2f}, {m2_y:.2f})"
            )

        self.update_batch_data_panel(None, None, m2_x, m2_y, m2a_x, m2a_y)
        self.redraw_batch_crosses()

        if getattr(self, "viewer_batch_m2_win", None) is not None:
            self.viewer_batch_m2_win.draw_marker(
                (self.batch_split_cx, split_y),
                pt2=(m2_x, m2_y),
                pt3=self.batch_m2_above_point,
                r2=getattr(self, "batch_m2_below_circle_r", None),
                r3=getattr(self, "batch_m2_above_circle_r", None),
            )
            if self.chk_batch_p2_show_thresh.isChecked():
                self.viewer_batch_m2_win.set_threshold_overlay(
                    m2_mask, visible=True, rgba_color=(41, 98, 255, 90)
                )
            else:
                self.viewer_batch_m2_win.clear_threshold_overlay()

    def update_batch_data_panel(self, m1_x=None, m1_y=None, m2_x=None, m2_y=None,
                                m2a_x=None, m2a_y=None):
        if self.matrix2 is None or self.batch_result_matrix is None:
            self.reset_batch_data_panel()
            return

        h, w = self.batch_result_matrix.shape
        self.lbl_batch_size.setText(f"矩陣大小: {h} × {w}")

        sy = self.batch_split_y
        sx = self.batch_split_cx
        peaks = self.batch_split_peak_ys
        if sy is not None and sx is not None:
            peak_txt = ""
            if peaks is not None:
                peak_txt = f" ｜ 雙峰 Y: {peaks[0]:.1f}/{peaks[1]:.1f}"
            roi_txt = ""
            bounds = getattr(self, "batch_valley_roi_bounds", None)
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                roi_txt = f" ｜ 框選: ({x0},{y0})–({x1 - 1},{y1 - 1})"
            self.lbl_batch_split_info.setText(
                f"切分波谷 Y: {sy:.2f} ｜ 縱切 X: {sx:.2f}{peak_txt}{roi_txt}"
            )
        else:
            self.lbl_batch_split_info.setText("切分波谷 Y: -- ｜ 縱切 X: --")

        if m2_x is None or m2_y is None:
            if self.batch_m2_center_point:
                m2_x, m2_y = self.batch_m2_center_point
        if m2a_x is None or m2a_y is None:
            if self.batch_m2_above_point:
                m2a_x, m2a_y = self.batch_m2_above_point

        pixel_pitch_um = 5.5
        if (m2a_x is not None and m2a_y is not None
                and m2_x is not None and m2_y is not None):
            dx_ab = m2_x - m2a_x
            dy_ab = m2_y - m2a_y
            dist_ab_px = float(np.sqrt(dx_ab ** 2 + dy_ab ** 2))
            self.lbl_batch_ab_distance_xy.setText(
                f"M2(above)→M2(below) 差距: ΔX: {abs(dx_ab):.2f} px, ΔY: {abs(dy_ab):.2f} px"
            )
            self.lbl_batch_ab_distance_total.setText(
                f"M2(above)→M2(below) 總距離: {dist_ab_px:.2f} px"
            )
            self.lbl_batch_ab_real_distance_xy.setText(
                f"M2(above)→M2(below) 實際 (*5.5): "
                f"ΔX: {abs(dx_ab) * pixel_pitch_um:.2f} μm, "
                f"ΔY: {abs(dy_ab) * pixel_pitch_um:.2f} μm"
            )
            self.lbl_batch_ab_real_distance_total.setText(
                f"M2(above)→M2(below) 實際總距離: {dist_ab_px * pixel_pitch_um:.2f} μm"
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
        super().reset_batch_data_panel()
        if hasattr(self, "lbl_batch_split_info"):
            self.lbl_batch_split_info.setText("切分波谷 Y: -- ｜ 縱切 X: --")

    def redraw_batch_crosses(self):
        for item in self.batch_cross_items:
            self.plot_batch_heat.removeItem(item)
        self.batch_cross_items.clear()

        if not self.chk_batch_show_cross.isChecked() or self.batch_result_matrix is None:
            return

        h, w = self.batch_result_matrix.shape

        # 橘色：波谷搜尋框選 ROI
        bounds = getattr(self, "batch_valley_roi_bounds", None)
        if bounds is None and getattr(self, "chk_batch_valley_roi", None):
            if self.chk_batch_valley_roi.isChecked() and self.matrix2 is not None:
                bounds = clip_roi_to_matrix(self.matrix2, self._get_valley_roi_tuple())
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            # 畫到像素外緣（半開區間右下角用 x1-ε）
            xs = [x0 - 0.5, x1 - 0.5, x1 - 0.5, x0 - 0.5, x0 - 0.5]
            ys = [y0 - 0.5, y0 - 0.5, y1 - 0.5, y1 - 0.5, y0 - 0.5]
            pen_roi = pg.mkPen("#EF6C00", width=2.0, style=Qt.DashLine)
            roi_item = pg.PlotCurveItem(x=xs, y=ys, pen=pen_roi)
            self.plot_batch_heat.addItem(roi_item)
            self.batch_cross_items.append(roi_item)

        # 紫色：波谷切分水平線 + 質心縱切線
        if self.batch_split_y is not None:
            sy = self.batch_split_y
            pen_split = pg.mkPen("#6A1B9A", width=2.5, style=Qt.DashLine)
            # 縱切線／切分線若有框選，優先畫在框內較清楚
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                h_item = pg.PlotCurveItem(x=[x0, x1], y=[sy, sy], pen=pen_split)
                self.plot_batch_heat.addItem(h_item)
                self.batch_cross_items.append(h_item)
                if self.batch_split_cx is not None:
                    sx = self.batch_split_cx
                    v_item = pg.PlotCurveItem(x=[sx, sx], y=[y0, y1], pen=pen_split)
                    self.plot_batch_heat.addItem(v_item)
                    self.batch_cross_items.append(v_item)
            else:
                h_item = pg.PlotCurveItem(x=[0, w], y=[sy, sy], pen=pen_split)
                self.plot_batch_heat.addItem(h_item)
                self.batch_cross_items.append(h_item)
                if self.batch_split_cx is not None:
                    sx = self.batch_split_cx
                    v_item = pg.PlotCurveItem(x=[sx, sx], y=[0, h], pen=pen_split)
                    self.plot_batch_heat.addItem(v_item)
                    self.batch_cross_items.append(v_item)

        if self.batch_m2_center_point:
            cx2, cy2 = self.batch_m2_center_point
            pen2 = pg.mkPen("#2962FF", width=2.5)
            v_item2 = pg.PlotCurveItem(x=[cx2, cx2], y=[0, h], pen=pen2)
            h_item2 = pg.PlotCurveItem(x=[0, w], y=[cy2, cy2], pen=pen2)
            self.plot_batch_heat.addItem(v_item2)
            self.plot_batch_heat.addItem(h_item2)
            self.batch_cross_items.extend([v_item2, h_item2])

        if self.batch_m2_above_point:
            cx3, cy3 = self.batch_m2_above_point
            pen3 = pg.mkPen("#D50000", width=2.5)
            v_item3 = pg.PlotCurveItem(x=[cx3, cx3], y=[0, h], pen=pen3)
            h_item3 = pg.PlotCurveItem(x=[0, w], y=[cy3, cy3], pen=pen3)
            self.plot_batch_heat.addItem(v_item3)
            self.plot_batch_heat.addItem(h_item3)
            self.batch_cross_items.extend([v_item3, h_item3])

        r2 = getattr(self, "batch_m2_below_circle_r", None)
        if self.batch_m2_center_point and r2 is not None and r2 > 0:
            cx2, cy2 = self.batch_m2_center_point
            circle2 = self._make_circle_curve(cx2, cy2, r2, pg.mkPen("#2962FF", width=2))
            self.plot_batch_heat.addItem(circle2)
            self.batch_cross_items.append(circle2)

        r3 = getattr(self, "batch_m2_above_circle_r", None)
        if self.batch_m2_above_point and r3 is not None and r3 > 0:
            cx3, cy3 = self.batch_m2_above_point
            circle3 = self._make_circle_curve(cx3, cy3, r3, pg.mkPen("#D50000", width=2))
            self.plot_batch_heat.addItem(circle3)
            self.batch_cross_items.append(circle3)

    def on_batch_process_mouse_clicked(self, evt):
        """點擊主圖（M2）：更新像素剖面；雙擊座標圖／熱圖：還原。"""
        # 與父類相同的穩定互動邏輯（含剖面雙擊還原）
        super().on_batch_process_mouse_clicked(evt)

    def save_current_batch_params(self):
        if self.batch_total_count == 0:
            QMessageBox.warning(self, "警告", "目前沒有載入任何 Batch 資料！")
            return

        self.batch_saved_params[self.batch_current_idx] = {
            "m2_mode": self._get_p2_point_mode_name(),
            "p2_use_threshold": self.chk_batch_p2_use_threshold.isChecked(),
            "m2_thresh_percent": self.spin_batch_p2_thresh_percent.value(),
            "m2_center_point": getattr(self, "batch_m2_center_point", None),
            "m2_above_point": getattr(self, "batch_m2_above_point", None),
            "split_y": getattr(self, "batch_split_y", None),
            "split_cx": getattr(self, "batch_split_cx", None),
            "valley_roi_enabled": self.chk_batch_valley_roi.isChecked(),
            "valley_roi_x": self.spin_batch_valley_roi_x.value(),
            "valley_roi_y": self.spin_batch_valley_roi_y.value(),
            "valley_roi_w": self.spin_batch_valley_roi_w.value(),
            "valley_roi_h": self.spin_batch_valley_roi_h.value(),
        }
        QMessageBox.information(
            self, "暫存成功", f"第 {self.batch_current_idx + 1} 組參數與位置已暫存！"
        )

    def _apply_saved_batch_params(self, idx):
        params = self.batch_saved_params.get(idx)
        if not params:
            return

        p2_mode = params.get("m2_mode", "m2_centroid")
        radios_p2 = {
            "auto_min": self.radio_batch_p2_auto_min,
            "m2_thresh_geom": self.radio_batch_p2_m2_thresh_geom,
            "m2_centroid": self.radio_batch_p2_m2_centroid,
            "m2_inscribed": self.radio_batch_p2_m2_inscribed,
            "manual": self.radio_batch_p2_manual,
        }
        for r in radios_p2.values():
            r.blockSignals(True)
        radios_p2.get(p2_mode, self.radio_batch_p2_m2_centroid).setChecked(True)
        for r in radios_p2.values():
            r.blockSignals(False)

        self.chk_batch_p2_use_threshold.blockSignals(True)
        self.chk_batch_p2_use_threshold.setChecked(bool(params.get("p2_use_threshold", True)))
        self.chk_batch_p2_use_threshold.blockSignals(False)

        self.spin_batch_p2_thresh_percent.blockSignals(True)
        self.spin_batch_p2_thresh_percent.setValue(params.get("m2_thresh_percent", 70.0))
        self.spin_batch_p2_thresh_percent.blockSignals(False)

        if hasattr(self, "chk_batch_valley_roi"):
            self.chk_batch_valley_roi.blockSignals(True)
            self.chk_batch_valley_roi.setChecked(bool(params.get("valley_roi_enabled", False)))
            self.chk_batch_valley_roi.blockSignals(False)
            self._set_valley_roi_controls_enabled(self.chk_batch_valley_roi.isChecked())
            for spin, key, default in (
                (self.spin_batch_valley_roi_x, "valley_roi_x", 0),
                (self.spin_batch_valley_roi_y, "valley_roi_y", 0),
                (self.spin_batch_valley_roi_w, "valley_roi_w", 200),
                (self.spin_batch_valley_roi_h, "valley_roi_h", 200),
            ):
                spin.blockSignals(True)
                spin.setValue(int(params.get(key, default)))
                spin.blockSignals(False)

        self.batch_m2_center_point = params.get("m2_center_point")
        self.batch_m2_above_point = params.get("m2_above_point")

    def _get_m1_auto_mode_name(self):
        return "valley_split"

    def export_batch_results_zip(self):
        if self.batch_total_count == 0 or self.batch_result_matrix is None:
            QMessageBox.warning(self, "警告", "目前無可匯出的數據！")
            return
        if not self.save_dir_path:
            QMessageBox.warning(self, "警告", "請先點擊「選擇儲存資料夾」按鈕以指定儲存路徑！")
            return

        # 加上時間戳，避免同名覆蓋。
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(self.save_dir_path, f"DataRay_Batch_M2_Results_{ts}.zip")
        summary_csv_path = os.path.join(self.save_dir_path, f"Result_Spot_Analysis_M2_{ts}.csv")
        prev_idx = self.batch_current_idx
        tmp_root = None
        import tempfile
        import shutil
        import zipfile

        try:
            self.lbl_batch_status.setText("狀態: 正在匯出 ZIP...")
            self.lbl_batch_status.setStyleSheet(
                "color: #F57C00; font-weight: bold; font-size: 12px;"
            )
            self.btn_batch_export.setEnabled(False)
            QApplication.processEvents()

            tmp_root = tempfile.mkdtemp(prefix="dataray_batch_m2_export_")
            summary_columns = []
            item_order = []

            for idx in range(self.batch_total_count):
                self.load_batch_group(idx)
                QApplication.processEvents()

                group_name = f"Group_{idx + 1:02d}"
                if idx < len(self.batch_pairs):
                    loc = self.batch_pairs[idx]["location"]
                    fname = os.path.splitext(self.batch_pairs[idx]["filename"])[0]
                    safe_loc = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(loc))
                    safe_fname = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in str(fname)
                    )
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
                    src_fname = ""
                    if idx < len(self.batch_pairs):
                        src_fname = str(self.batch_pairs[idx].get("filename", ""))
                    summary_columns.append((group_name, value_map, unit_map, src_fname))

            self._write_spot_analysis_summary_csv(
                summary_csv_path, item_order, summary_columns
            )
            shutil.copy2(summary_csv_path, os.path.join(tmp_root, os.path.basename(summary_csv_path)))

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _dirs, files in os.walk(tmp_root):
                    for name in files:
                        abs_path = os.path.join(root, name)
                        arcname = os.path.relpath(abs_path, tmp_root)
                        zipf.write(abs_path, arcname)

            self.load_batch_group(prev_idx)
            self.lbl_batch_status.setText("狀態: 所有檔案匯出成功！")
            self.lbl_batch_status.setStyleSheet(
                "color: #2E7D32; font-weight: bold; font-size: 12px;"
            )
            self.btn_batch_export.setEnabled(True)
            QMessageBox.information(
                self, "成功",
                f"匯出完成！\n\nZIP：\n{zip_path}\n\n彙整統計 CSV：\n{summary_csv_path}\n\n"
                f"每組含：Heatmap／V_Profile／H_Profile／Heatmap_With_Profiles／Contour 等"
            )
        except Exception as e:
            try:
                self.load_batch_group(prev_idx)
            except Exception:
                pass
            self.lbl_batch_status.setText("狀態: 匯出失敗")
            self.lbl_batch_status.setStyleSheet(
                "color: #C62828; font-weight: bold; font-size: 12px;"
            )
            self.btn_batch_export.setEnabled(True)
            QMessageBox.critical(self, "匯出錯誤", f"匯出過程發生錯誤：\n{str(e)}")
        finally:
            if tmp_root and os.path.isdir(tmp_root):
                shutil.rmtree(tmp_root, ignore_errors=True)

    def _write_spot_analysis_summary_csv(self, csv_path, item_order, summary_columns):
        """M2 彙整：不同來源檔名之間插入一欄空白，提升閱讀性。"""

        def _src_filename(col):
            if len(col) >= 4:
                return str(col[3] or "")
            return ""

        headers = ["Item", "Unit"]
        prev_fname = None
        for col_name, _values, _units, *rest in summary_columns:
            cur_fname = _src_filename((col_name, _values, _units, *rest))
            if prev_fname is not None and cur_fname != prev_fname:
                headers.append("")
            headers.append(col_name)
            prev_fname = cur_fname

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for item in item_order:
                unit = ""
                for _name, _values, units, *_rest in summary_columns:
                    if item in units and units[item] not in ("", None):
                        unit = units[item]
                        break

                row = [item, unit]
                prev_fname = None
                for name, values, _units, *rest in summary_columns:
                    cur_fname = _src_filename((name, values, _units, *rest))
                    if prev_fname is not None and cur_fname != prev_fname:
                        row.append("")
                    row.append(values.get(item, ""))
                    prev_fname = cur_fname
                writer.writerow(row)

    def _export_single_group_like_dataray(self, base_path, idx):
        if self.batch_result_matrix is None:
            return []

        c_min, c_max = self.batch_hist.getLevels()
        split_pt = (
            (self.batch_split_cx, self.batch_split_y)
            if self.batch_split_y is not None else None
        )
        m2_pt = self.batch_m2_center_point
        m2a_pt = self.batch_m2_above_point
        click_points = []
        if split_pt is not None:
            click_points.append(list(split_pt))
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
            "mode": "m2_only",
            "split_valley_y_px": self.batch_split_y,
            "split_centroid_x_px": self.batch_split_cx,
            "split_peak_ys": list(self.batch_split_peak_ys) if self.batch_split_peak_ys else None,
            "valley_roi_enabled": self.chk_batch_valley_roi.isChecked(),
            "valley_roi_x": self.spin_batch_valley_roi_x.value() if self.chk_batch_valley_roi.isChecked() else None,
            "valley_roi_y": self.spin_batch_valley_roi_y.value() if self.chk_batch_valley_roi.isChecked() else None,
            "valley_roi_w": self.spin_batch_valley_roi_w.value() if self.chk_batch_valley_roi.isChecked() else None,
            "valley_roi_h": self.spin_batch_valley_roi_h.value() if self.chk_batch_valley_roi.isChecked() else None,
            "valley_roi_bounds": list(self.batch_valley_roi_bounds) if self.batch_valley_roi_bounds else None,
            "m2_below_x_px": m2_pt[0] if m2_pt else None,
            "m2_below_y_px": m2_pt[1] if m2_pt else None,
            "m2_above_x_px": m2a_pt[0] if m2a_pt else None,
            "m2_above_y_px": m2a_pt[1] if m2a_pt else None,
            "m2_below_inscribed_r_px": getattr(self, "batch_m2_below_circle_r", None),
            "m2_above_inscribed_r_px": getattr(self, "batch_m2_above_circle_r", None),
            "p2_point_mode": self._get_p2_point_mode_name(),
            "p2_use_threshold": self.chk_batch_p2_use_threshold.isChecked(),
            "p2_threshold_percent": self.spin_batch_p2_thresh_percent.value(),
            "batch_group_index": idx + 1,
            "batch_location": location,
            "batch_filename": filename,
            "batch_m2_file": os.path.basename(self.batch_m2_files[idx]) if idx < len(self.batch_m2_files) else "",
            "batch_m2_path": self.batch_m2_files[idx] if idx < len(self.batch_m2_files) else "",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_params, f, indent=4, ensure_ascii=False)

        excel_img_path = f"{base_path}_Result.xlsx"
        pd.DataFrame(self.batch_result_matrix).to_excel(excel_img_path, index=False, header=False)

        spot_analysis_excel_path = f"{base_path}_Spot_Analysis.xlsx"
        wb_spot = openpyxl.Workbook()
        ws_spot = wb_spot.active
        ws_spot.title = "Spot_and_Measurement"
        ws_spot.append(["Item", "Value", "Unit"])

        p2 = m2_pt if m2_pt is not None else ("--", "--")
        p2a = m2a_pt if m2a_pt is not None else ("--", "--")

        dx_ab_px = dy_ab_px = dist_ab_px = "--"
        dx_ab_um = dy_ab_um = dist_ab_um = "--"
        pixel_pitch_um = 5.5

        if m2a_pt is not None and m2_pt is not None:
            dx_ab_px = abs(m2_pt[0] - m2a_pt[0])
            dy_ab_px = abs(m2_pt[1] - m2a_pt[1])
            dist_ab_px = float(np.sqrt(dx_ab_px ** 2 + dy_ab_px ** 2))
            dx_ab_um = dx_ab_px * pixel_pitch_um
            dy_ab_um = dy_ab_px * pixel_pitch_um
            dist_ab_um = dist_ab_px * pixel_pitch_um

        spot_rows = [
            ["Group Name", self._build_group_column_name(idx), ""],
            ["Location", location, ""],
            ["Cycle / Filename", filename, ""],
            ["Group Index", idx + 1, ""],
            ["Mode", "M2-only (valley split)", ""],
            ["Matrix Height", self.batch_result_matrix.shape[0], "px"],
            ["Matrix Width", self.batch_result_matrix.shape[1], "px"],
            ["Split Valley Y", self.batch_split_y if self.batch_split_y is not None else "--", "px"],
            ["Split Centroid X", self.batch_split_cx if self.batch_split_cx is not None else "--", "px"],
            ["Split Peak Y Lo", self.batch_split_peak_ys[0] if self.batch_split_peak_ys else "--", "px"],
            ["Split Peak Y Hi", self.batch_split_peak_ys[1] if self.batch_split_peak_ys else "--", "px"],
            ["Valley ROI Enabled", "Yes" if self.chk_batch_valley_roi.isChecked() else "No", ""],
            ["Valley ROI X", self.spin_batch_valley_roi_x.value() if self.chk_batch_valley_roi.isChecked() else "--", "px"],
            ["Valley ROI Y", self.spin_batch_valley_roi_y.value() if self.chk_batch_valley_roi.isChecked() else "--", "px"],
            ["Valley ROI Width", self.spin_batch_valley_roi_w.value() if self.chk_batch_valley_roi.isChecked() else "--", "px"],
            ["Valley ROI Height", self.spin_batch_valley_roi_h.value() if self.chk_batch_valley_roi.isChecked() else "--", "px"],
            ["P2 Point Mode", self._get_p2_point_mode_name(), ""],
            ["Use P2 Point Threshold", "Yes" if self.chk_batch_p2_use_threshold.isChecked() else "No", ""],
            ["P2 Point Threshold Percent", self.spin_batch_p2_thresh_percent.value(), "%"],
            ["M2 Below (X)", p2[0] if p2 != ("--", "--") else "--", "px"],
            ["M2 Below (Y)", p2[1] if p2 != ("--", "--") else "--", "px"],
            ["M2 Above (X)", p2a[0] if p2a != ("--", "--") else "--", "px"],
            ["M2 Above (Y)", p2a[1] if p2a != ("--", "--") else "--", "px"],
            ["M2 Below Inscribed Radius", getattr(self, "batch_m2_below_circle_r", None) if getattr(self, "batch_m2_below_circle_r", None) is not None else "--", "px"],
            ["M2 Above Inscribed Radius", getattr(self, "batch_m2_above_circle_r", None) if getattr(self, "batch_m2_above_circle_r", None) is not None else "--", "px"],
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

        # 一併匯出當下點選的縱／橫剖面與合成圖
        self._export_batch_profile_images(base_path)

        contour_img_path = f"{base_path}_Contour.png"
        smoothed = uniform_filter(self.batch_result_matrix, size=31, mode="nearest")
        temp_win = pg.GraphicsLayoutWidget()
        plot_contour = temp_win.addPlot(title="Smoothed Contour Map")
        plot_contour.getViewBox().invertY(False)
        plot_contour.setAspectLocked(True)
        plot_contour.setLabel("bottom", "X Pixels")
        plot_contour.setLabel("left", "Y Pixels")
        contour_img = pg.ImageItem(smoothed.T)
        plot_contour.addItem(contour_img)
        min_v, max_v = float(np.min(smoothed)), float(np.max(smoothed))
        for level in np.linspace(min_v, max_v, 10):
            iso = pg.IsocurveItem(data=smoothed.T, level=level, pen=pg.mkPen("w", width=0.8))
            plot_contour.addItem(iso)
        QApplication.processEvents()
        pg_export.ImageExporter(plot_contour).export(contour_img_path)
        temp_win.close()
        temp_win.deleteLater()
        return spot_rows
