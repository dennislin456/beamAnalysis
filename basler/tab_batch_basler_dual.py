"""Basler 短／長曝光雙檔測試分頁。

- 資料夾匯入：檔名編號奇數＝短曝光、偶數＝長曝光（0001＋0002，或 001＋002）
- 演算法選單：質心／門檻、一維投影＋微分、水平加總、高斯濾波；參數依選擇顯示
- 一般演算法：左右兩張 heatmap；拼貼視窗另開
- 水平加總：主畫面為短／長波形（上列平滑／精修／次峰窗）；Heatmap 改由「查看」開啟
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import uniform_filter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from batch_data_loader import load_numeric_matrix
from shared_components import (
    InteractiveHeatmapPanel,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    attach_curve_hover_readout,
    attach_waveform_axis_scale,
    build_robust_threshold_mask,
    clip_roi_to_matrix,
    compute_auto_spot_center,
    compute_horizontal_sum_profile,
    export_timestamp_tag,
    find_dual_spots_axial_1d_projection,
    find_dual_spots_gaussian,
    find_dual_spots_horizontal_sum,
    fit_max_inscribed_circle,
    peak_y_max_in_window,
    refine_dual_peaks_quadratic,
    refine_hsum_dual_peaks,
    refine_peak_y_on_profile,
    refine_profile_together,
    with_export_timestamp,
)
from tab_batch_basler import (
    BaslerBatchTab,
    _build_bw_clip_colormap,
)

# dual 分頁 color scale 預設（原始強度）
_DEFAULT_BLACK_BELOW = 256.0
_DEFAULT_WHITE_ABOVE = 4096.0

_FILE_FILTER = (
    "資料檔 (*.npy *.csv *.xlsx *.xls);;"
    "NumPy (*.npy);;CSV (*.csv);;Excel (*.xlsx *.xls);;所有檔案 (*.*)"
)
_DATA_EXTS = (".npy", ".csv", ".xlsx", ".xls")
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")

_STYLE_BTN = (
    "QPushButton { font-size: 12px; color: #212121; "
    "background-color: #F5F5F5; border: 1px solid #BDBDBD; "
    "border-radius: 4px; padding: 7px 12px; }"
    "QPushButton:hover { background-color: #EEEEEE; }"
    "QPushButton:disabled { color: #9E9E9E; background-color: #FAFAFA; }"
)
_STYLE_BTN_PRIMARY = (
    "QPushButton { font-size: 12px; font-weight: bold; color: #FAFAFA; "
    "background-color: #455A64; border: 1px solid #37474F; "
    "border-radius: 4px; padding: 8px 12px; }"
    "QPushButton:hover { background-color: #546E7A; }"
    "QPushButton:disabled { color: #B0BEC5; background-color: #CFD8DC; "
    "border-color: #B0BEC5; }"
)
_STYLE_PANEL = (
    "QWidget#baslerDualSpotBox, QWidget#baslerDualDataBox {"
    " background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 6px; }"
    "QWidget#baslerDualSpotBox QLabel, QWidget#baslerDualDataBox QLabel {"
    " font-size: 12px; color: #424242; border: none; }"
)
_STYLE_SECTION = "font-weight: bold; color: #212121; font-size: 13px; border: none;"
_STYLE_MUTED = "color: #757575; font-size: 11px; border: none;"


def _extract_shot_index(filename):
    """檔名裡、日期前的拍攝編號。支援 _0001_ 與 _001_。"""
    idx, _group = _shot_index_and_group(filename)
    return idx


def _shot_index_and_group(filename):
    """回傳 (編號, 分組鍵)。分組鍵拿掉編號，避免 0001 與 001 撞號。"""
    stem = os.path.basename(filename)
    m = re.search(r"_(\d{3,})_(?=\d{8}(?:_|\.|$))", stem)
    if m:
        group = stem[: m.start(1)] + stem[m.end(1) :]
        return int(m.group(1)), group
    found = [int(x) for x in re.findall(r"_(\d{4,})_", stem)]
    for n in found:
        if 1 <= n <= 9999:
            return n, stem
    if found:
        return found[-1], stem
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", stem)
    if m:
        return int(m.group(1)), stem
    return None, stem


def _pair_indices(by_idx):
    """奇數＝短、偶數＝長；(1,2)、(3,4)…"""
    pairs = []
    used = set()
    for idx in sorted(by_idx.keys()):
        if idx in used:
            continue
        if idx % 2 == 1 and (idx + 1) in by_idx:
            s_idx, l_idx = idx, idx + 1
        elif idx % 2 == 0 and (idx - 1) in by_idx and (idx - 1) not in used:
            s_idx, l_idx = idx - 1, idx
        else:
            continue
        used.add(s_idx)
        used.add(l_idx)
        pairs.append((s_idx, l_idx))
    return pairs


def scan_short_long_pairs(folder):
    """Odd index = short, even = long; pairs (1,2), (3,4), ..."""
    if not folder or not os.path.isdir(folder):
        return []
    grouped = {}
    for name in os.listdir(folder):
        if not name.lower().endswith(_DATA_EXTS):
            continue
        idx, group = _shot_index_and_group(name)
        if idx is None:
            continue
        grouped.setdefault(group, {}).setdefault(idx, os.path.join(folder, name))

    pairs = []
    for by_idx in grouped.values():
        for s_idx, l_idx in _pair_indices(by_idx):
            pairs.append(
                {
                    "short_idx": s_idx,
                    "long_idx": l_idx,
                    "short_path": by_idx[s_idx],
                    "long_path": by_idx[l_idx],
                    "label": f"{s_idx:04d}（短）＋ {l_idx:04d}（長）",
                }
            )
    pairs.sort(key=lambda p: (os.path.basename(p["short_path"]), p["short_idx"]))
    return pairs


class BatchAnalysisResultsDialog(QDialog):
    """批次分析結果視窗：演算法資訊、可點選清單、短／長圖、匯出。

    短／長圖與主分頁同一套 InteractiveHeatmapPanel，並各自具備
    log／color scale／平滑／顯示門檻控制列。
    """

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批次分析資料")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)
        self.setMinimumSize(960, 640)
        self.resize(1280, 820)
        self._store = store or {}
        self._results = list(self._store.get("results") or [])
        self._tab = parent  # BaslerBatchDualTab
        self._current_result = None
        self._short_mat = None
        self._long_mat = None
        self._short_cross_items = []
        self._long_cross_items = []
        self._short_cmap_guard = False
        self._long_cmap_guard = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        algo = self._store.get("algorithm_desc") or self._store.get("algorithm") or "--"
        self.lbl_algo = QLabel(f"演算法：{algo}")
        self.lbl_algo.setWordWrap(True)
        self.lbl_algo.setStyleSheet("font-weight: bold; color: #37474F;")
        params = self._store.get("algorithm_params") or {}
        param_txt = ", ".join(f"{k}={v}" for k, v in list(params.items())[:8])
        self.lbl_params = QLabel(f"參數：{param_txt or '--'}")
        self.lbl_params.setWordWrap(True)
        self.lbl_params.setStyleSheet("color: #546E7A; font-size: 11px;")
        folder = self._store.get("data_folder") or "--"
        ok_n = sum(1 for r in self._results if r.get("ok"))
        self.lbl_summary = QLabel(
            f"資料夾：{folder}\n結果：{ok_n}/{len(self._results)} 成功"
        )
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("color: #546E7A; font-size: 11px;")
        root.addWidget(self.lbl_algo)
        root.addWidget(self.lbl_params)
        root.addWidget(self.lbl_summary)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("計算清單（點選載入圖）"))
        self.list_results = QListWidget()
        self.list_results.setMinimumWidth(260)
        for r in self._results:
            mark = "OK" if r.get("ok") else "FAIL"
            item = QListWidgetItem(f"{r.get('index', 0) + 1}. [{mark}] {r.get('label')}")
            tip = r.get("error") or ""
            if r.get("short_spot") and r.get("long_spot2"):
                tip = (
                    f"短 ({r['short_spot'][0]:.1f}, {r['short_spot'][1]:.1f})\n"
                    f"長2 ({r['long_spot2'][0]:.1f}, {r['long_spot2'][1]:.1f})"
                )
            item.setToolTip(tip)
            item.setData(Qt.UserRole, r.get("index", 0))
            self.list_results.addItem(item)
        self.list_results.currentRowChanged.connect(self._on_row_changed)
        left_lay.addWidget(self.list_results, 1)
        self.lbl_detail = QLabel("尚未選擇")
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #455A64; font-size: 11px;")
        left_lay.addWidget(self.lbl_detail)
        split.addWidget(left)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)
        self._build_display_controls()

        short_wrap = QWidget()
        short_lay = QVBoxLayout(short_wrap)
        short_lay.setContentsMargins(0, 0, 0, 0)
        short_lay.setSpacing(4)
        short_lbl = QLabel("短曝光 (Short)")
        short_lbl.setStyleSheet(
            "font-weight: bold; color: #424242; font-size: 13px; "
            "padding: 4px 8px; background: #E3F2FD; border-radius: 4px;"
        )
        short_lay.addWidget(short_lbl)
        short_lay.addWidget(self._short_display_bar)
        self.short_panel = self._make_panel("Short (batch)", self._on_short_levels)
        short_lay.addWidget(self.short_panel, 1)

        long_wrap = QWidget()
        long_lay = QVBoxLayout(long_wrap)
        long_lay.setContentsMargins(0, 0, 0, 0)
        long_lay.setSpacing(4)
        long_lbl = QLabel("長曝光 (Long)")
        long_lbl.setStyleSheet(
            "font-weight: bold; color: #424242; font-size: 13px; "
            "padding: 4px 8px; background: #F3E5F5; border-radius: 4px;"
        )
        long_lay.addWidget(long_lbl)
        long_lay.addWidget(self._long_display_bar)
        self.long_panel = self._make_panel("Long (batch)", self._on_long_levels)
        long_lay.addWidget(self.long_panel, 1)

        img_row = QHBoxLayout()
        img_row.setContentsMargins(0, 0, 0, 0)
        img_row.setSpacing(6)
        img_row.addWidget(short_wrap, 1)
        img_row.addWidget(long_wrap, 1)
        right_lay.addLayout(img_row, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("匯出結果（參數／圖片）")
        self.btn_export.setStyleSheet(_STYLE_BTN_PRIMARY)
        self.btn_export.setToolTip(
            "一次匯出整批摘要，以及每組成功配對的 JSON／CSV 參數與短／長 heatmap 圖"
        )
        self.btn_export.clicked.connect(self._export_results)
        self.btn_close = QPushButton("關閉")
        self.btn_close.setStyleSheet(_STYLE_BTN)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_export)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        if self._results:
            self.list_results.setCurrentRow(0)

    def _copy_chk(self, src_attr, default=False):
        src = getattr(self._tab, src_attr, None) if self._tab is not None else None
        if src is not None:
            return bool(src.isChecked())
        return bool(default)

    def _copy_spin(self, src_attr, default):
        src = getattr(self._tab, src_attr, None) if self._tab is not None else None
        if src is not None:
            return src.value()
        return default

    def _build_display_controls(self):
        """短／長各自一套顯示列（初始值沿用主分頁目前設定）。"""
        # Short
        self.chk_short_log = QCheckBox("log")
        self.chk_short_log.setChecked(self._copy_chk("chk_batch_heatmap_log"))
        self.chk_short_bw = QCheckBox("color scale")
        self.chk_short_bw.setChecked(self._copy_chk("chk_basler_bw_scale"))
        self.spin_short_black = NoWheelDoubleSpinBox()
        self.spin_short_black.setRange(0.0, 1e9)
        self.spin_short_black.setDecimals(1)
        self.spin_short_black.setFixedWidth(72)
        self.spin_short_black.setValue(
            float(self._copy_spin("spin_basler_black_below", _DEFAULT_BLACK_BELOW))
        )
        self.spin_short_white = NoWheelDoubleSpinBox()
        self.spin_short_white.setRange(0.0, 1e9)
        self.spin_short_white.setDecimals(1)
        self.spin_short_white.setFixedWidth(72)
        self.spin_short_white.setValue(
            float(self._copy_spin("spin_basler_white_above", _DEFAULT_WHITE_ABOVE))
        )
        self.chk_short_smooth = QCheckBox("平滑")
        self.chk_short_smooth.setChecked(self._copy_chk("chk_batch_heatmap_smooth"))
        self.spin_short_smooth = NoWheelSpinBox()
        self.spin_short_smooth.setRange(1, 201)
        self.spin_short_smooth.setSingleStep(2)
        self.spin_short_smooth.setFixedWidth(60)
        self.spin_short_smooth.setValue(
            int(self._copy_spin("spin_batch_heatmap_smooth", 31))
        )
        self.btn_short_thresh = QPushButton("顯示門檻")
        self.btn_short_thresh.setCheckable(True)
        self.btn_short_thresh.setCursor(Qt.PointingHandCursor)
        self.btn_short_thresh.setToolTip(
            "在熱圖上以半透明色顯示目前門檻％對應的光斑區域"
        )
        self.btn_short_hsum = QPushButton("查看水平加總圖")
        self.btn_short_hsum.setCursor(Qt.PointingHandCursor)
        self.btn_short_hsum.setToolTip(
            "一維投影專用：以目前條帶參數計算短曝光水平加總剖面"
        )
        self.btn_short_hsum.clicked.connect(
            lambda: self._show_horizontal_sum("short")
        )

        # Long
        self.chk_long_log = QCheckBox("log")
        self.chk_long_log.setChecked(self._copy_chk("chk_long_heatmap_log"))
        self.chk_long_bw = QCheckBox("color scale")
        self.chk_long_bw.setChecked(self._copy_chk("chk_long_bw_scale"))
        self.spin_long_black = NoWheelDoubleSpinBox()
        self.spin_long_black.setRange(0.0, 1e9)
        self.spin_long_black.setDecimals(1)
        self.spin_long_black.setFixedWidth(72)
        self.spin_long_black.setValue(
            float(self._copy_spin("spin_long_black_below", _DEFAULT_BLACK_BELOW))
        )
        self.spin_long_white = NoWheelDoubleSpinBox()
        self.spin_long_white.setRange(0.0, 1e9)
        self.spin_long_white.setDecimals(1)
        self.spin_long_white.setFixedWidth(72)
        self.spin_long_white.setValue(
            float(self._copy_spin("spin_long_white_above", _DEFAULT_WHITE_ABOVE))
        )
        self.chk_long_smooth = QCheckBox("平滑")
        self.chk_long_smooth.setChecked(self._copy_chk("chk_long_heatmap_smooth"))
        self.spin_long_smooth = NoWheelSpinBox()
        self.spin_long_smooth.setRange(1, 201)
        self.spin_long_smooth.setSingleStep(2)
        self.spin_long_smooth.setFixedWidth(60)
        self.spin_long_smooth.setValue(
            int(self._copy_spin("spin_long_heatmap_smooth", 31))
        )
        self.btn_long_thresh = QPushButton("顯示門檻")
        self.btn_long_thresh.setCheckable(True)
        self.btn_long_thresh.setCursor(Qt.PointingHandCursor)
        self.btn_long_thresh.setToolTip(
            "在熱圖上以半透明色顯示目前門檻％對應的光斑區域"
        )
        self.btn_long_hsum = QPushButton("查看水平加總圖")
        self.btn_long_hsum.setCursor(Qt.PointingHandCursor)
        self.btn_long_hsum.setToolTip(
            "一維投影專用：以目前條帶參數計算長曝光水平加總剖面"
        )
        self.btn_long_hsum.clicked.connect(
            lambda: self._show_horizontal_sum("long")
        )

        self.spin_short_black.setEnabled(self.chk_short_bw.isChecked())
        self.spin_short_white.setEnabled(self.chk_short_bw.isChecked())
        self.spin_short_smooth.setEnabled(self.chk_short_smooth.isChecked())
        self.spin_long_black.setEnabled(self.chk_long_bw.isChecked())
        self.spin_long_white.setEnabled(self.chk_long_bw.isChecked())
        self.spin_long_smooth.setEnabled(self.chk_long_smooth.isChecked())

        show_profile = BaslerBatchDualTab._is_axial_algorithm_id(
            self._store.get("algorithm")
        )
        self.btn_short_hsum.setVisible(show_profile)
        self.btn_long_hsum.setVisible(show_profile)

        tab = self._tab
        if tab is not None and hasattr(tab, "_make_display_bar"):
            self._short_display_bar = tab._make_display_bar(
                self.chk_short_log,
                self.chk_short_bw,
                self.spin_short_black,
                self.spin_short_white,
                self.chk_short_smooth,
                self.spin_short_smooth,
                "batchDlgShortDisplayBar",
                extra_widgets=(self.btn_short_thresh, self.btn_short_hsum),
            )
            self._long_display_bar = tab._make_display_bar(
                self.chk_long_log,
                self.chk_long_bw,
                self.spin_long_black,
                self.spin_long_white,
                self.chk_long_smooth,
                self.spin_long_smooth,
                "batchDlgLongDisplayBar",
                extra_widgets=(self.btn_long_thresh, self.btn_long_hsum),
            )
        else:
            self._short_display_bar = self._fallback_display_bar(
                self.chk_short_log,
                self.chk_short_bw,
                self.spin_short_black,
                self.spin_short_white,
                self.chk_short_smooth,
                self.spin_short_smooth,
                self.btn_short_thresh,
                self.btn_short_hsum,
            )
            self._long_display_bar = self._fallback_display_bar(
                self.chk_long_log,
                self.chk_long_bw,
                self.spin_long_black,
                self.spin_long_white,
                self.chk_long_smooth,
                self.spin_long_smooth,
                self.btn_long_thresh,
                self.btn_long_hsum,
            )

        self.chk_short_log.toggled.connect(self._on_short_log_toggled)
        self.chk_short_bw.toggled.connect(self._on_short_bw_toggled)
        self.spin_short_black.valueChanged.connect(self._on_short_display_changed)
        self.spin_short_white.valueChanged.connect(self._on_short_display_changed)
        self.chk_short_smooth.toggled.connect(self._on_short_smooth_toggled)
        self.spin_short_smooth.valueChanged.connect(self._on_short_display_changed)
        self.btn_short_thresh.toggled.connect(self._on_short_thresh_toggled)

        self.chk_long_log.toggled.connect(self._on_long_log_toggled)
        self.chk_long_bw.toggled.connect(self._on_long_bw_toggled)
        self.spin_long_black.valueChanged.connect(self._on_long_display_changed)
        self.spin_long_white.valueChanged.connect(self._on_long_display_changed)
        self.chk_long_smooth.toggled.connect(self._on_long_smooth_toggled)
        self.spin_long_smooth.valueChanged.connect(self._on_long_display_changed)
        self.btn_long_thresh.toggled.connect(self._on_long_thresh_toggled)

    @staticmethod
    def _fallback_display_bar(
        log_chk,
        bw_chk,
        black_spin,
        white_spin,
        smooth_chk,
        smooth_spin,
        *extra_btns,
    ):
        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(8)
        for w in (
            log_chk,
            bw_chk,
            black_spin,
            white_spin,
            smooth_chk,
            smooth_spin,
            *extra_btns,
        ):
            row.addWidget(w)
        row.addStretch(1)
        host = QWidget()
        host.setLayout(row)
        return host

    def _show_horizontal_sum(self, which="long"):
        """批次結果視窗：查看目前選中配對的水平加總圖。"""
        if not BaslerBatchDualTab._is_axial_algorithm_id(self._store.get("algorithm")):
            QMessageBox.information(
                self,
                "提示",
                "「查看水平加總圖」僅在一維投影＋微分批次結果可用。",
            )
            return
        result = self._current_result
        if result is None or not result.get("ok"):
            QMessageBox.warning(self, "警告", "請先選擇一組成功的配對結果。")
            return
        ss = result.get("short_spot")
        if ss is None:
            QMessageBox.warning(self, "警告", "此配對沒有短曝光光斑座標。")
            return

        is_short = str(which).lower() == "short"
        mat = self._short_mat if is_short else self._long_mat
        if mat is None:
            QMessageBox.warning(self, "警告", "尚未載入對應熱圖資料。")
            return
        analysis = self._analysis_matrix(mat, short=is_short)
        center = (float(ss[0]), float(ss[1]))
        if not is_short:
            h, w = np.asarray(analysis).shape[:2]
            center = (
                float(np.clip(center[0], 0.0, max(0.0, w - 1.0))),
                float(np.clip(center[1], 0.0, max(0.0, h - 1.0))),
            )
        strip = BaslerBatchDualTab._strip_params_from_algo_dict(
            self._store.get("algorithm_params")
        )
        tab = self._tab
        label = "短曝光" if is_short else "長曝光"
        if tab is not None and hasattr(tab, "show_horizontal_sum_profile_from_data"):
            tab.show_horizontal_sum_profile_from_data(
                analysis,
                center,
                label=label,
                strip_params=strip,
                parent=self,
                which="short" if is_short else "long",
            )
        else:
            QMessageBox.warning(self, "警告", "無法開啟水平加總圖。")

    def _on_short_bw_toggled(self, checked=False):
        on = bool(checked)
        self.spin_short_black.setEnabled(on)
        self.spin_short_white.setEnabled(on)
        self._on_short_display_changed()

    def _on_short_smooth_toggled(self, checked=False):
        self.spin_short_smooth.setEnabled(bool(checked))
        self._on_short_display_changed()

    def _on_long_bw_toggled(self, checked=False):
        on = bool(checked)
        self.spin_long_black.setEnabled(on)
        self.spin_long_white.setEnabled(on)
        self._on_long_display_changed()

    def _on_long_smooth_toggled(self, checked=False):
        self.spin_long_smooth.setEnabled(bool(checked))
        self._on_long_display_changed()

    def _on_short_log_toggled(self, checked=False):
        self._on_short_display_changed()

    def _on_long_log_toggled(self, checked=False):
        self._on_long_display_changed()

    def _on_short_display_changed(self, *_args):
        self._refresh_short_display(reset_levels=True)
        self._refresh_short_thresh_overlay()

    def _on_long_display_changed(self, *_args):
        self._refresh_long_display(reset_levels=True)
        self._refresh_long_thresh_overlay()

    def _on_short_thresh_toggled(self, checked=False):
        self.btn_short_thresh.setText("隱藏門檻" if checked else "顯示門檻")
        self._refresh_short_thresh_overlay()

    def _on_long_thresh_toggled(self, checked=False):
        self.btn_long_thresh.setText("隱藏門檻" if checked else "顯示門檻")
        self._refresh_long_thresh_overlay()

    def _log_on(self, short=True):
        chk = self.chk_short_log if short else self.chk_long_log
        return chk.isChecked()

    def _bw_on(self, short=True):
        chk = self.chk_short_bw if short else self.chk_long_bw
        return chk.isChecked()

    def _smooth_on(self, short=True):
        chk = self.chk_short_smooth if short else self.chk_long_smooth
        return chk.isChecked()

    def _smooth_size(self, short=True):
        spin = self.spin_short_smooth if short else self.spin_long_smooth
        return max(1, int(spin.value()))

    def _bw_raw_thresholds(self, short=True):
        if short:
            return float(self.spin_short_black.value()), float(
                self.spin_short_white.value()
            )
        return float(self.spin_long_black.value()), float(self.spin_long_white.value())

    def _bw_display_thresholds(self, short=True):
        if not self._bw_on(short):
            return None, None
        lo, hi = self._bw_raw_thresholds(short)
        if self._log_on(short):
            lo = float(np.log10(max(lo, 1.0)))
            hi = float(np.log10(max(hi, 1.0)))
        return lo, hi

    def _make_panel(self, title, on_levels):
        panel = InteractiveHeatmapPanel(
            title=title,
            x_label="X Pixels",
            y_label="Y Pixels",
            aspect_locked=True,
            with_profiles=True,
            show_colorbar=True,
            show_tip=False,
        )
        tab = self._tab
        if tab is not None and hasattr(tab, "_hide_interactive_panel_chrome"):
            tab._hide_interactive_panel_chrome(panel)
        else:
            for attr in ("lbl_tip", "btn_export"):
                w = getattr(panel, attr, None)
                if w is not None:
                    w.hide()
            for attr in ("lbl_level_max", "lbl_level_min"):
                w = getattr(panel, attr, None)
                if w is not None:
                    w.show()
        panel.plot.getViewBox().invertY(True)
        if panel.plot_y_profile is not None:
            panel.plot_y_profile.getViewBox().invertY(True)
        jet = getattr(tab, "batch_jet_map", None) if tab is not None else None
        if panel.hist is not None:
            if jet is not None:
                panel.hist.gradient.setColorMap(jet)
            panel.hist.sigLevelsChanged.connect(on_levels)
        return panel

    def _on_short_levels(self, *_args):
        self._apply_colorbar(self.short_panel, "_short_cmap_guard", short=True)

    def _on_long_levels(self, *_args):
        self._apply_colorbar(self.long_panel, "_long_cmap_guard", short=False)

    def _apply_colorbar(self, panel, guard_attr, short=True):
        tab = self._tab
        if panel is None or panel.hist is None:
            return
        if getattr(self, guard_attr, False):
            return
        try:
            vmin, vmax = panel.hist.getLevels()
        except Exception:
            return
        vmin, vmax = float(vmin), float(vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            vmax = vmin + 1.0
        base = getattr(tab, "batch_jet_map", None) if tab is not None else None
        if base is None:
            return
        bw_on = self._bw_on(short)
        if bw_on:
            span = max(vmax - vmin, 1e-12)
            cmap = _build_bw_clip_colormap(
                base,
                vmin,
                vmax,
                black_floor=vmin + span * 1e-9,
                white_ceil=vmax - span * 1e-9,
            )
        else:
            cmap = base
        setattr(self, guard_attr, True)
        try:
            panel.hist.gradient.setColorMap(cmap)
        finally:
            setattr(self, guard_attr, False)

    def _display_matrix(self, raw, short=True):
        """套用本視窗的 log／平滑／color scale。"""
        tab = self._tab
        mat = np.asarray(raw, dtype=float)
        if tab is not None:
            data = tab._transform_display_matrix(
                mat,
                self._log_on(short),
                self._smooth_on(short),
                self._smooth_size(short),
            )
        else:
            data = mat
            if self._log_on(short):
                data = np.log10(np.maximum(data, 1.0))
            if self._smooth_on(short):
                k = self._smooth_size(short)
                if k > 1:
                    data = uniform_filter(data, size=k, mode="nearest")
        if data is None or not self._bw_on(short):
            return data
        lo, hi = self._bw_display_thresholds(short)
        if tab is not None:
            return tab._clip_color_scale_values(data, lo, hi)
        out = np.asarray(data, dtype=float).copy()
        out = np.where(out < lo, 0.0, out)
        out = np.where(out > hi, hi, out)
        return out

    def _analysis_matrix(self, raw, short=True):
        """門檻 overlay 用：原始強度＋color scale 硬裁切（不含 log／平滑）。"""
        mat = np.asarray(raw, dtype=float)
        if not self._bw_on(short):
            return mat
        lo, hi = self._bw_raw_thresholds(short)
        tab = self._tab
        if tab is not None:
            return tab._clip_color_scale_values(mat, lo, hi)
        out = mat.copy()
        out = np.where(out < lo, 0.0, out)
        out = np.where(out > hi, hi, out)
        return out

    def _set_panel_image(self, panel, display, short=True):
        tab = self._tab
        if panel is None:
            return
        if display is None:
            panel.set_image(None, reset_view=True)
            return
        bw_lo = bw_hi = None
        if self._bw_on(short):
            bw_lo, bw_hi = self._bw_display_thresholds(short)
        if tab is not None:
            intensity = tab._intensity_label(
                self._log_on(short),
                self._smooth_on(short),
                self._smooth_size(short),
            )
            tab._set_panel_image(
                panel,
                display,
                True,
                intensity,
                bw_hi=bw_hi,
                bw_lo=bw_lo,
            )
        else:
            finite = display[np.isfinite(display)]
            levels = None
            if (
                bw_lo is not None
                and bw_hi is not None
                and np.isfinite(bw_lo)
                and np.isfinite(bw_hi)
            ):
                lo, hi = float(bw_lo), float(bw_hi)
                if hi < lo:
                    lo, hi = hi, lo
                if hi <= lo:
                    hi = lo + 1.0
                levels = (lo, hi)
            elif bw_hi is not None and np.isfinite(bw_hi) and float(bw_hi) > 0:
                levels = (0.0, float(bw_hi))
            elif finite.size:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmin == vmax:
                    vmax = vmin + 1.0
                levels = (vmin, vmax)
            panel.set_image(
                display.T, levels=levels, reset_view=True, source_matrix=display
            )
        self._apply_colorbar(
            panel,
            "_short_cmap_guard" if short else "_long_cmap_guard",
            short=short,
        )

    def _refresh_short_display(self, reset_levels=True):
        if self._short_mat is None:
            return
        self._set_panel_image(
            self.short_panel,
            self._display_matrix(self._short_mat, short=True),
            short=True,
        )
        if self._current_result is not None:
            self._redraw_overlays(self._current_result)

    def _refresh_long_display(self, reset_levels=True):
        if self._long_mat is None:
            return
        self._set_panel_image(
            self.long_panel,
            self._display_matrix(self._long_mat, short=False),
            short=False,
        )
        if self._current_result is not None:
            self._redraw_overlays(self._current_result)

    def _thresh_percent(self, short=True):
        params = self._store.get("algorithm_params") or {}
        if short:
            return float(params.get("short_thresh_percent", 50.0))
        if "long_thresh_percent" in params:
            return float(params.get("long_thresh_percent", 50.0))
        return float(params.get("short_thresh_percent", 50.0))

    def _refresh_short_thresh_overlay(self):
        panel = self.short_panel
        if panel is None:
            return
        show = bool(self.btn_short_thresh.isChecked())
        if (not show) or self._short_mat is None:
            panel.clear_threshold_overlay()
            return
        tab = self._tab
        mat = self._analysis_matrix(self._short_mat, short=True)
        if tab is None:
            panel.clear_threshold_overlay()
            return
        mask = tab._build_full_threshold_mask(
            mat, self._thresh_percent(short=True), roi=None
        )
        panel.set_threshold_overlay(
            mask, visible=True, rgba_color=(255, 64, 255, 90)
        )

    def _refresh_long_thresh_overlay(self):
        panel = self.long_panel
        if panel is None:
            return
        show = bool(self.btn_long_thresh.isChecked())
        if (not show) or self._long_mat is None:
            panel.clear_threshold_overlay()
            return
        tab = self._tab
        if tab is None:
            panel.clear_threshold_overlay()
            return
        mat = self._analysis_matrix(self._long_mat, short=False)
        roi = None
        r = self._current_result or {}
        roi_xywh = r.get("axial_roi_xywh") or r.get("long_search_roi_xywh")
        params = self._store.get("algorithm_params") or {}
        algo = self._store.get("algorithm") or ""
        if roi_xywh is None and "axial" in str(algo):
            roi_xywh = params.get("display_roi_xywh")
        if roi_xywh is None and params.get("long_search_roi_enabled"):
            roi_xywh = params.get("long_search_roi_xywh")
        if roi_xywh is not None:
            roi = tuple(roi_xywh)
        mask = tab._build_full_threshold_mask(
            mat, self._thresh_percent(short=False), roi=roi
        )
        panel.set_threshold_overlay(
            mask, visible=True, rgba_color=(255, 152, 0, 90)
        )

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

    def _redraw_overlays(self, result):
        short_mat = self._short_mat
        long_mat = self._long_mat
        ss = result.get("short_spot")
        ls = result.get("long_spot2")
        algo = self._store.get("algorithm") or ""
        params = self._store.get("algorithm_params") or {}

        if self.short_panel is not None:
            self._clear_plot_items(self.short_panel.plot, "_short_cross_items")
            if ss is not None and short_mat is not None:
                self._add_cross(
                    self.short_panel.plot,
                    "_short_cross_items",
                    float(ss[0]),
                    float(ss[1]),
                    short_mat.shape,
                    pg.mkPen("#D50000", width=2.5),
                )

        if self.long_panel is not None:
            self._clear_plot_items(self.long_panel.plot, "_long_cross_items")
            if long_mat is None:
                return
            # ROI 框：水平加總／一維投影條帶或質心搜尋範圍
            is_hsum = BaslerBatchDualTab._is_hsum_algorithm_id(algo)
            roi_xywh = (
                result.get("hsum_roi_xywh")
                or result.get("axial_roi_xywh")
                or result.get("long_search_roi_xywh")
            )
            if roi_xywh is None and is_hsum:
                roi_xywh = params.get("display_roi_xywh")
            if roi_xywh is None and "axial" in str(algo):
                roi_xywh = params.get("display_roi_xywh")
                if roi_xywh is None and ss is not None:
                    # 以短曝光座標＋條帶寬高重建顯示 ROI
                    rw = int(params.get("strip_roi_w") or 160)
                    rh = int(params.get("strip_roi_h") or 512)
                    cx, cy = float(ss[0]), float(ss[1])
                    roi_xywh = [
                        int(round(cx - rw / 2.0)),
                        int(round(cy - rh / 2.0)),
                        rw,
                        rh,
                    ]
            if roi_xywh is None and params.get("long_search_roi_enabled"):
                roi_xywh = params.get("long_search_roi_xywh")
            if roi_xywh is not None:
                bounds = clip_roi_to_matrix(long_mat, tuple(roi_xywh))
                if bounds is not None:
                    self._add_roi_rect(
                        self.long_panel.plot,
                        "_long_cross_items",
                        bounds,
                        pg.mkPen("#EF6C00", width=2.0, style=Qt.DashLine),
                    )
            if ss is not None:
                h, w = long_mat.shape
                mx = float(np.clip(ss[0], 0.0, w - 1.0))
                my = float(np.clip(ss[1], 0.0, h - 1.0))
                self._add_cross(
                    self.long_panel.plot,
                    "_long_cross_items",
                    mx,
                    my,
                    long_mat.shape,
                    pg.mkPen("#2962FF", width=2.5),
                )
            # 水平加總 Heatmap：不下光斑十字
            if ls is not None and not is_hsum:
                self._add_cross(
                    self.long_panel.plot,
                    "_long_cross_items",
                    float(ls[0]),
                    float(ls[1]),
                    long_mat.shape,
                    pg.mkPen("#D50000", width=2.5),
                )

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self._results):
            return
        r = self._results[row]
        self._current_result = r
        status = "成功" if r.get("ok") else f"失敗：{r.get('error') or '--'}"
        ss = r.get("short_spot")
        ls = r.get("long_spot2")
        delta = r.get("delta") or {}
        lines = [
            f"{r.get('label')}｜{status}",
            f"短: {os.path.basename(r.get('short_path') or '')}",
            f"長: {os.path.basename(r.get('long_path') or '')}",
        ]
        if ss:
            lines.append(f"短曝光光斑: ({ss[0]:.2f}, {ss[1]:.2f})")
        if ls:
            lines.append(f"長曝光第二光斑: ({ls[0]:.2f}, {ls[1]:.2f})")
        if delta:
            lines.append(
                f"Δ: dx={delta.get('dx_px', 0):.2f} px, "
                f"dy={delta.get('dy_px', 0):.2f} px, "
                f"dist={delta.get('dist_px', 0):.2f} px"
            )
        self.lbl_detail.setText("\n".join(lines))

        try:
            short_mat = load_numeric_matrix(r["short_path"])
            long_mat = load_numeric_matrix(r["long_path"])
        except Exception as exc:
            self._short_mat = None
            self._long_mat = None
            self._set_panel_image(self.short_panel, None, short=True)
            self._set_panel_image(self.long_panel, None, short=False)
            self.lbl_detail.setText(self.lbl_detail.text() + f"\n讀圖失敗: {exc}")
            return

        self._short_mat = np.asarray(short_mat, dtype=float)
        self._long_mat = np.asarray(long_mat, dtype=float)
        self._refresh_short_display(reset_levels=True)
        self._refresh_long_display(reset_levels=True)
        self._refresh_short_thresh_overlay()
        self._refresh_long_thresh_overlay()

    def _display_settings_dict(self, short=True):
        if short:
            return {
                "log": self._log_on(True),
                "smooth": self._smooth_on(True),
                "smooth_size": self._smooth_size(True),
                "color_scale": self._bw_on(True),
                "black_below": float(self.spin_short_black.value()),
                "white_above": float(self.spin_short_white.value()),
            }
        return {
            "log": self._log_on(False),
            "smooth": self._smooth_on(False),
            "smooth_size": self._smooth_size(False),
            "color_scale": self._bw_on(False),
            "black_below": float(self.spin_long_black.value()),
            "white_above": float(self.spin_long_white.value()),
        }

    @staticmethod
    def _safe_stem(label, index):
        raw = str(label or f"pair_{int(index) + 1}")
        cleaned = re.sub(r"[^\w\-.]+", "_", raw, flags=re.UNICODE).strip("._")
        return cleaned or f"pair_{int(index) + 1}"

    def _px_to_um(self, px):
        tab = self._tab
        if tab is not None and hasattr(tab, "_px_to_um"):
            return float(tab._px_to_um(px))
        return float(px) * 3.45

    def _delta_with_um(self, result):
        d = dict(result.get("delta") or {})
        ss = result.get("short_spot")
        ls = result.get("long_spot2")
        if ss is not None and ls is not None:
            dx = float(ls[0] - ss[0])
            dy = float(ls[1] - ss[1])
            dist = float(np.hypot(dx, dy))
            d.setdefault("dx_px", dx)
            d.setdefault("dy_px", dy)
            d.setdefault("dist_px", dist)
        for key_px, key_um in (
            ("dx_px", "dx_um"),
            ("dy_px", "dy_um"),
            ("dist_px", "dist_um"),
        ):
            if d.get(key_um) is None and d.get(key_px) is not None:
                d[key_um] = self._px_to_um(d[key_px])
        d.setdefault("pixel_pitch_um", 3.45)
        d.setdefault("distance_scale", 1.0)
        d.setdefault("um_formula", "|d_px| * 3.45")
        return d

    def _build_pair_payload(self, result, short_mat=None, long_mat=None):
        """單組配對匯出參數（對齊主分頁 JSON 欄位）。"""
        params = dict(self._store.get("algorithm_params") or {})
        if result.get("axial_roi_xywh") is not None:
            params["display_roi_xywh"] = list(result["axial_roi_xywh"])
        if result.get("long_search_roi_xywh") is not None:
            params["long_search_roi_xywh"] = list(result["long_search_roi_xywh"])
            params["long_search_roi_enabled"] = True

        ss = result.get("short_spot")
        ls = result.get("long_spot2")
        delta = self._delta_with_um(result)
        short_shape = list(short_mat.shape) if short_mat is not None else None
        long_shape = list(long_mat.shape) if long_mat is not None else None
        roi_enabled = bool(params.get("long_search_roi_enabled"))
        payload = {
            "mode": "basler_short_long_dual_batch",
            "batch_index": result.get("index"),
            "batch_label": result.get("label"),
            "algorithm": self._store.get("algorithm"),
            "algorithm_desc": self._store.get("algorithm_desc"),
            "algorithm_params": params,
            "short_file": os.path.basename(result.get("short_path") or ""),
            "long_file": os.path.basename(result.get("long_path") or ""),
            "short_path": result.get("short_path") or "",
            "long_path": result.get("long_path") or "",
            "short_matrix_shape_yx": short_shape,
            "long_matrix_shape_yx": long_shape,
            "short_thresh_percent": float(
                params.get("short_thresh_percent", 50.0)
            ),
            "long_thresh_percent": float(
                params.get(
                    "long_thresh_percent",
                    params.get("short_thresh_percent", 50.0),
                )
            ),
            "long_search_roi_enabled": roi_enabled,
            "long_search_roi_xywh": (
                list(params.get("long_search_roi_xywh"))
                if params.get("long_search_roi_xywh") is not None
                else None
            ),
            "short_spot_x_px": None if ss is None else float(ss[0]),
            "short_spot_y_px": None if ss is None else float(ss[1]),
            "long_spot_x_px": None if ls is None else float(ls[0]),
            "long_spot_y_px": None if ls is None else float(ls[1]),
            "short_to_long_delta_x_px": delta.get("dx_px"),
            "short_to_long_delta_y_px": delta.get("dy_px"),
            "short_to_long_distance_px": delta.get("dist_px"),
            "short_to_long_delta_x_um": delta.get("dx_um"),
            "short_to_long_delta_y_um": delta.get("dy_um"),
            "short_to_long_distance_um": delta.get("dist_um"),
            "pixel_pitch_um": delta.get("pixel_pitch_um"),
            "distance_scale": delta.get("distance_scale"),
            "um_formula": delta.get("um_formula"),
            "short_display": self._display_settings_dict(short=True),
            "long_display": self._display_settings_dict(short=False),
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
        }
        if (
            BaslerBatchDualTab._is_hsum_algorithm_id(self._store.get("algorithm"))
            and ss is not None
            and short_mat is not None
            and long_mat is not None
        ):
            short_an = self._analysis_matrix(short_mat, short=True)
            long_an = self._analysis_matrix(long_mat, short=False)
            strip = BaslerBatchDualTab._strip_params_from_algo_dict(params)
            try:
                n_ref = int(params.get("refine_iters", 1) or 0)
            except Exception:
                n_ref = 0
            sec_win = params.get("secondary_search_window_y")
            tab = self._tab
            if tab is not None and hasattr(tab, "_compute_hsum_profiles_for_mats"):
                payload["horizontal_sum"] = tab._compute_hsum_profiles_for_mats(
                    short_an,
                    long_an,
                    ss,
                    strip_params=strip,
                    n_refine=n_ref,
                    secondary_window_y=sec_win,
                )
            else:
                roi_w, roi_h, smooth_win, min_sep = strip
                hsum = {
                    "strip_roi_w": roi_w,
                    "strip_roi_h": roi_h,
                    "profile_smooth_win": smooth_win,
                    "min_peak_distance_px": min_sep,
                    "short": None,
                    "long": None,
                }
                lh, lw = np.asarray(long_an).shape[:2]
                long_center = (
                    float(np.clip(ss[0], 0.0, max(0.0, lw - 1.0))),
                    float(np.clip(ss[1], 0.0, max(0.0, lh - 1.0))),
                )
                for key, mat, center in (
                    ("short", short_an, ss),
                    ("long", long_an, long_center),
                ):
                    p = compute_horizontal_sum_profile(
                        mat,
                        center,
                        roi_w=roi_w,
                        roi_h=roi_h,
                        smooth_win=smooth_win,
                        min_peak_distance=min_sep,
                        find_secondary=True,
                    )
                    hsum[key] = BaslerBatchDualTab._serialize_hsum_profile(p)
                payload["horizontal_sum"] = hsum
        return payload

    def _hide_overlays_for_export(self):
        if self.short_panel is not None:
            self._clear_plot_items(self.short_panel.plot, "_short_cross_items")
            self.short_panel.clear_threshold_overlay()
        if self.long_panel is not None:
            self._clear_plot_items(self.long_panel.plot, "_long_cross_items")
            self.long_panel.clear_threshold_overlay()

    def _restore_overlays_after_export(self):
        if self._current_result is not None:
            self._redraw_overlays(self._current_result)
        self._refresh_short_thresh_overlay()
        self._refresh_long_thresh_overlay()

    def _load_pair_into_panels(self, result):
        short_mat = np.asarray(load_numeric_matrix(result["short_path"]), dtype=float)
        long_mat = np.asarray(load_numeric_matrix(result["long_path"]), dtype=float)
        self._current_result = result
        self._short_mat = short_mat
        self._long_mat = long_mat
        self._refresh_short_display(reset_levels=True)
        self._refresh_long_display(reset_levels=True)
        QApplication.processEvents()
        return short_mat, long_mat

    def _export_one_pair(
        self, folder, result, exported_files, stamp=None, hsum_collector=None
    ):
        """匯出單組：JSON／CSV 參數＋短／長 heatmap 圖（檔名含時間戳）。"""
        if not result.get("ok"):
            return
        stamp = stamp or export_timestamp_tag()
        short_mat, long_mat = self._load_pair_into_panels(result)
        stem = self._safe_stem(result.get("label"), result.get("index", 0))
        # 邏輯前綴不加戳；各輸出檔在副檔名前統一加同一 stamp
        base_path = os.path.join(folder, f"{stem}_ShortLong")

        payload = self._build_pair_payload(result, short_mat, long_mat)
        json_path = with_export_timestamp(base_path + ".json", stamp=stamp)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        exported_files.append(json_path)

        csv_path = with_export_timestamp(
            f"{base_path}_Spot_Data.csv", stamp=stamp
        )
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Item", "Value", "Unit"])
            tab = self._tab
            if tab is not None and hasattr(tab, "_export_payload_rows"):
                for row in tab._export_payload_rows(payload):
                    writer.writerow(row)
                if hasattr(tab, "_write_hsum_csv_sections"):
                    tab._write_hsum_csv_sections(
                        writer, payload.get("horizontal_sum")
                    )
            else:
                for k, v in payload.items():
                    if isinstance(v, (dict, list)):
                        continue
                    writer.writerow([k, v, ""])
                BaslerBatchDualTab._write_hsum_csv_sections(
                    writer, payload.get("horizontal_sum")
                )
        exported_files.append(csv_path)

        # 一維投影：額外匯出水平加總雙峰 CSV（峰位／平滑前後／間距）
        hsum = payload.get("horizontal_sum")
        if hsum:
            BaslerBatchDualTab._export_hsum_peaks_files(
                base_path, hsum, stamp, exported=exported_files
            )
            if hsum_collector is not None:
                hsum_collector.append(
                    {
                        "index": result.get("index"),
                        "label": result.get("label") or stem,
                        "hsum": hsum,
                    }
                )

        self._hide_overlays_for_export()
        try:
            QApplication.processEvents()
            if self.short_panel is not None:
                outs = (
                    self.short_panel.export_plot_bundle(
                        f"{base_path}_Short_Heatmap_With_Profiles",
                        stamp=stamp,
                    )
                    or []
                )
                exported_files.extend(outs)
            if self.long_panel is not None:
                outs = (
                    self.long_panel.export_plot_bundle(
                        f"{base_path}_Long_Heatmap_With_Profiles",
                        stamp=stamp,
                    )
                    or []
                )
                exported_files.extend(outs)
        finally:
            self._restore_overlays_after_export()

    def _export_results(self):
        """一次匯出：整批摘要＋每組成功配對的參數／短長圖（同一時間戳）。"""
        start = ""
        parent = self.parent()
        if parent is not None and getattr(parent, "save_dir_path", None):
            start = parent.save_dir_path
        folder = QFileDialog.getExistingDirectory(self, "選擇匯出資料夾", start)
        if not folder:
            return

        ok_results = [r for r in self._results if r.get("ok")]
        if not ok_results:
            QMessageBox.warning(self, "警告", "沒有成功的配對可匯出圖片／參數。")
            return

        stamp = export_timestamp_tag()
        prev_row = self.list_results.currentRow()
        exported = []
        hsum_collector = []
        progress = QProgressDialog(
            "批次匯出中…", "取消", 0, len(ok_results) + 1, self
        )
        progress.setWindowTitle("匯出結果")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            # 1) 整批摘要（含目前顯示設定）
            summary = dict(self._store)
            summary["short_display"] = self._display_settings_dict(short=True)
            summary["long_display"] = self._display_settings_dict(short=False)
            summary["exported_at"] = stamp
            summary_base = os.path.join(folder, "batch_analysis_results")
            json_path = with_export_timestamp(summary_base + ".json", stamp=stamp)
            csv_path = with_export_timestamp(summary_base + ".csv", stamp=stamp)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            exported.append(json_path)
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "index",
                        "label",
                        "ok",
                        "error",
                        "short_path",
                        "long_path",
                        "short_x",
                        "short_y",
                        "long2_x",
                        "long2_y",
                        "dx_px",
                        "dy_px",
                        "dist_px",
                        "dx_um",
                        "dy_um",
                        "dist_um",
                        "algorithm",
                    ]
                )
                algo = self._store.get("algorithm")
                for r in self._results:
                    ss = r.get("short_spot") or (None, None)
                    ls = r.get("long_spot2") or (None, None)
                    d = self._delta_with_um(r) if r.get("ok") else (r.get("delta") or {})
                    writer.writerow(
                        [
                            r.get("index"),
                            r.get("label"),
                            r.get("ok"),
                            r.get("error") or "",
                            r.get("short_path"),
                            r.get("long_path"),
                            ss[0],
                            ss[1],
                            ls[0],
                            ls[1],
                            d.get("dx_px"),
                            d.get("dy_px"),
                            d.get("dist_px"),
                            d.get("dx_um"),
                            d.get("dy_um"),
                            d.get("dist_um"),
                            algo,
                        ]
                    )
            exported.append(csv_path)
            progress.setValue(1)
            QApplication.processEvents()

            # 2) 每組成功配對：參數 JSON／CSV＋短／長圖
            for i, r in enumerate(ok_results):
                if progress.wasCanceled():
                    break
                progress.setLabelText(
                    f"匯出 {i + 1}/{len(ok_results)}：{r.get('label', '')}"
                )
                progress.setValue(i + 1)
                QApplication.processEvents()
                self._export_one_pair(
                    folder, r, exported, stamp=stamp, hsum_collector=hsum_collector
                )

            # 3) 全部水平加總剖面彙整（方便一次疊圖）
            if hsum_collector and not progress.wasCanceled():
                all_hsum_path = with_export_timestamp(
                    os.path.join(folder, "batch_hsum_all_profiles.csv"),
                    stamp=stamp,
                )
                BaslerBatchDualTab._write_batch_hsum_all_csv(
                    all_hsum_path, hsum_collector
                )
                exported.append(all_hsum_path)

            progress.setValue(len(ok_results) + 1)
        except Exception as exc:
            self._restore_overlays_after_export()
            progress.close()
            QMessageBox.critical(self, "匯出失敗", f"無法匯出：\n{exc}")
            return
        finally:
            progress.close()
            if 0 <= prev_row < len(self._results):
                self.list_results.setCurrentRow(prev_row)
                self._on_row_changed(prev_row)

        n_img = sum(
            1
            for p in exported
            if str(p).lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )
        preview = "\n".join(exported[:12])
        if len(exported) > 12:
            preview += f"\n…共 {len(exported)} 個檔案（含 {n_img} 張圖）"
        else:
            preview += f"\n（含 {n_img} 張圖）"
        QMessageBox.information(self, "匯出完成", f"已寫入：\n{preview}")


class BatchHSumResultsDialog(QDialog):
    """批次水平加總檢視：短＋長剖面、可調平滑、可拖雙峰、匯出峰位／間距。"""

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批次水平加總")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)
        self.setMinimumSize(960, 640)
        self.resize(1180, 820)
        self._store = store or {}
        self._results = list(self._store.get("results") or [])
        self._tab = parent
        self._current_result = None
        self._short_mat = None
        self._long_mat = None
        self._loading = False
        self._peak_guard = False
        self._region_guard = False
        # 全域次峰搜尋窗 (y_lo, y_hi)：批次每個點共用
        self._sec_window = None
        self._last_refine_meta = {"n_done": 0}
        # which -> (x0, x1, y0, y1) 雙擊復原用
        self._default_ranges = {"short": None, "long": None}
        self._scale_ctrls = {}
        self._profiles = {"short": None, "long": None}

        roi_w, roi_h, smooth_win, min_sep = (
            BaslerBatchDualTab._strip_params_from_algo_dict(
                self._store.get("algorithm_params")
            )
        )
        self._roi_w = int(roi_w)
        self._roi_h = int(roi_h)
        self._min_sep = int(min_sep)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        algo = self._store.get("algorithm_desc") or self._store.get("algorithm") or "--"
        self.lbl_algo = QLabel(f"演算法：{algo}")
        self.lbl_algo.setWordWrap(True)
        self.lbl_algo.setStyleSheet("font-weight: bold; color: #37474F;")
        root.addWidget(self.lbl_algo)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("剖面平滑窗"))
        self.spin_smooth = NoWheelSpinBox()
        self.spin_smooth.setRange(1, 201)
        self.spin_smooth.setSingleStep(2)
        self.spin_smooth.setValue(max(1, int(smooth_win)))
        self.spin_smooth.setFixedWidth(72)
        self.spin_smooth.setToolTip(
            "調整平滑曲線；主峰固定為平滑前最高峰；次峰在紅色搜尋窗內取最高點"
        )
        ctrl.addWidget(self.spin_smooth)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("次峰窗 Y"))
        self.spin_sec_lo = NoWheelDoubleSpinBox()
        self.spin_sec_lo.setRange(0.0, 1e6)
        self.spin_sec_lo.setDecimals(1)
        self.spin_sec_lo.setFixedWidth(80)
        self.spin_sec_lo.setToolTip(
            "次峰搜尋窗下限（Y pixels）；批次全部點共用（僅能用數值調整，不可拖圖）"
        )
        self.spin_sec_hi = NoWheelDoubleSpinBox()
        self.spin_sec_hi.setRange(0.0, 1e6)
        self.spin_sec_hi.setDecimals(1)
        self.spin_sec_hi.setFixedWidth(80)
        self.spin_sec_hi.setToolTip(
            "次峰搜尋窗上限（Y pixels）；批次全部點共用（僅能用數值調整，不可拖圖）"
        )
        ctrl.addWidget(self.spin_sec_lo)
        ctrl.addWidget(QLabel("～"))
        ctrl.addWidget(self.spin_sec_hi)
        self.btn_auto_sec_win = QPushButton("自動次峰窗")
        self.btn_auto_sec_win.setStyleSheet(_STYLE_BTN)
        self.btn_auto_sec_win.setCursor(Qt.PointingHandCursor)
        self.btn_auto_sec_win.setToolTip(
            "依目前長曝光剖面自動建議次峰搜尋窗，並套用到全部點"
        )
        self.btn_auto_sec_win.clicked.connect(self._auto_sec_window)
        ctrl.addWidget(self.btn_auto_sec_win)
        self.btn_auto_peaks = QPushButton("重新找次峰")
        self.btn_auto_peaks.setStyleSheet(_STYLE_BTN)
        self.btn_auto_peaks.setCursor(Qt.PointingHandCursor)
        self.btn_auto_peaks.setToolTip("用目前次峰窗，對目前點重新抓次峰")
        self.btn_auto_peaks.clicked.connect(self._auto_peaks_current)
        ctrl.addWidget(self.btn_auto_peaks)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("精修次數"))
        self.spin_refine = NoWheelSpinBox()
        self.spin_refine.setRange(0, 20)
        _ref_default = 1
        try:
            _ref_default = int(
                (self._store.get("algorithm_params") or {}).get("refine_iters", 1)
                or 1
            )
        except Exception:
            _ref_default = 1
        self.spin_refine.setValue(max(0, min(20, _ref_default)))
        self.spin_refine.setFixedWidth(56)
        self.spin_refine.setToolTip(
            "精修 N：0＝整點峰位；N≥1＝分段二次加密 after refine，"
            "虛線對齊該曲線強度最大點（與主畫面水平加總相同）"
        )
        ctrl.addWidget(self.spin_refine)
        self.btn_refine = QPushButton("套用精修")
        self.btn_refine.setStyleSheet(_STYLE_BTN)
        self.btn_refine.setCursor(Qt.PointingHandCursor)
        self.btn_refine.setToolTip(
            "依目前精修次數重算兩峰（對齊 after refine 峰頂）並更新 ΔY／波形"
        )
        self.btn_refine.clicked.connect(self._apply_refine_current)
        ctrl.addWidget(self.btn_refine)
        ctrl.addStretch(1)
        self.lbl_peaks = QLabel("峰位：--")
        self.lbl_peaks.setWordWrap(True)
        self.lbl_peaks.setStyleSheet(
            "color: #1B5E20; font-size: 12px; font-weight: bold; "
            "padding: 4px 8px; background: #E8F5E9; border-radius: 4px;"
        )
        ctrl.addWidget(self.lbl_peaks, 2)
        root.addLayout(ctrl)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("計算清單（點選載入剖面）"))
        self.list_results = QListWidget()
        self.list_results.setMinimumWidth(260)
        for r in self._results:
            mark = "OK" if r.get("ok") else "FAIL"
            item = QListWidgetItem(
                f"{r.get('index', 0) + 1}. [{mark}] {r.get('label')}"
            )
            tip = r.get("error") or ""
            if r.get("short_spot") and r.get("long_spot2"):
                tip = (
                    f"短 ({r['short_spot'][0]:.1f}, {r['short_spot'][1]:.1f})\n"
                    f"長2 ({r['long_spot2'][0]:.1f}, {r['long_spot2'][1]:.1f})"
                )
            item.setToolTip(tip)
            item.setData(Qt.UserRole, r.get("index", 0))
            self.list_results.addItem(item)
        self.list_results.currentRowChanged.connect(self._on_row_changed)
        left_lay.addWidget(self.list_results, 1)
        self.lbl_detail = QLabel("尚未選擇")
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #455A64; font-size: 11px;")
        left_lay.addWidget(self.lbl_detail)
        split.addWidget(left)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)

        self.plot_short, self.line_s1, self.line_s2 = self._make_profile_plot(
            "短曝光 Horizontal Sum", which="short", dual_peaks=False
        )
        self.plot_long, self.line_l1, self.line_l2 = self._make_profile_plot(
            "長曝光 Horizontal Sum", which="long", dual_peaks=True
        )
        # 主峰／次峰皆不可拖；次峰由搜尋窗決定
        self.line_s1.setMovable(False)
        self.line_l1.setMovable(False)
        self.line_l2.setMovable(False)
        self.line_s2.hide()
        self.sec_region = pg.LinearRegionItem(
            values=(0.0, 100.0),
            orientation="vertical",
            brush=pg.mkBrush(198, 40, 40, 35),
            pen=pg.mkPen("#C62828", width=1.2),
            movable=False,
        )
        self.sec_region.setZValue(-10)
        try:
            # 僅作視覺範圍；上下限改由上方數值框調整，禁止拖移／縮放
            self.sec_region.setAcceptedMouseButtons(Qt.NoButton)
        except Exception:
            pass
        self.plot_long.addItem(self.sec_region)

        right_lay.addWidget(self.plot_short, 1)
        right_lay.addWidget(self.plot_long, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        btn_row = QHBoxLayout()
        self.btn_export_current = QPushButton("匯出目前配對")
        self.btn_export_current.setStyleSheet(_STYLE_BTN_PRIMARY)
        self.btn_export_current.setToolTip(
            "依目前平滑窗、精修次數與峰線位置，匯出此配對短／長峰位、間距與剖面 CSV"
        )
        self.btn_export_current.clicked.connect(self._export_current)
        self.btn_export_all = QPushButton("匯出全部配對")
        self.btn_export_all.setStyleSheet(_STYLE_BTN_PRIMARY)
        self.btn_export_all.setToolTip(
            "以目前平滑窗、次峰搜尋窗與精修次數，重算全部成功配對的峰位／間距"
        )
        self.btn_export_all.clicked.connect(self._export_all)
        self.btn_close = QPushButton("關閉")
        self.btn_close.setStyleSheet(_STYLE_BTN)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_export_current)
        btn_row.addWidget(self.btn_export_all)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        self.spin_smooth.valueChanged.connect(self._on_smooth_changed)
        self.spin_sec_lo.valueChanged.connect(self._on_sec_spin_changed)
        self.spin_sec_hi.valueChanged.connect(self._on_sec_spin_changed)
        self.spin_refine.valueChanged.connect(self._on_refine_changed)

        # 若批次參數已有次峰窗，先套用（波形修正用，與 heatmap 無關）
        try:
            sw = (self._store.get("algorithm_params") or {}).get(
                "secondary_search_window_y"
            )
            if sw is not None and len(sw) >= 2:
                self._set_sec_window(float(sw[0]), float(sw[1]), refresh=False)
        except Exception:
            pass

        if self._results:
            self.list_results.setCurrentRow(0)

    def _px_to_um(self, px):
        tab = self._tab
        if tab is not None and hasattr(tab, "_px_to_um"):
            return float(tab._px_to_um(px))
        return float(px) * 3.45

    def _make_profile_plot(self, title, *, which, dual_peaks=True):
        plot = pg.PlotWidget(title=title)
        plot.setBackground("#FAFBFC")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "Y (pixels)")
        plot.setLabel("left", "Horizontal sum")
        plot.addLegend(offset=(10, 10))
        line1 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#2E7D32", width=2, style=Qt.DashLine),
        )
        line2 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#C62828", width=2, style=Qt.DashLine),
        )
        plot.addItem(line1)
        plot.addItem(line2)
        if not dual_peaks:
            line2.hide()
        self._scale_ctrls[which] = attach_waveform_axis_scale(
            plot,
            self,
            x_label="Y (pixels)",
            y_label="Horizontal sum",
        )
        attach_curve_hover_readout(plot, pixel_name="Y")
        return plot, line1, line2

    def _data_range_from_block(self, block):
        if not block:
            return None
        ys = np.asarray(block.get("y_px") or [], dtype=float)
        before = np.asarray(
            block.get("sum_before_smooth")
            or block.get("sum_minus_median")
            or [],
            dtype=float,
        )
        after = np.asarray(
            block.get("sum_after_smooth") or block.get("sum_smooth") or [],
            dtype=float,
        )
        if ys.size == 0:
            return None
        x0 = float(np.nanmin(ys))
        x1 = float(np.nanmax(ys))
        vals = []
        if before.size:
            vals.append(before)
        if after.size:
            vals.append(after)
        if not vals:
            return None
        stacked = np.concatenate(vals)
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            return None
        y0 = float(min(0.0, np.nanmin(finite)))
        y1 = float(np.nanmax(finite))
        if y1 <= y0:
            y1 = y0 + 1.0
        pad = (y1 - y0) * 0.05
        return (x0, x1, y0, y1 + pad)

    def _apply_default_range(self, which, plot):
        rng = self._default_ranges.get(which)
        ctrl = self._scale_ctrls.get(which)
        if rng is not None and ctrl is not None:
            ctrl.set_default_range(*rng)
        if rng is None:
            plot.enableAutoRange()
            return
        x0, x1, y0, y1 = rng
        plot.setXRange(x0, x1, padding=0.02)
        plot.setYRange(y0, y1, padding=0)

    def _update_default_ranges_from_profiles(self, *, apply=False):
        self._default_ranges["short"] = self._data_range_from_block(
            self._profiles.get("short")
        )
        self._default_ranges["long"] = self._data_range_from_block(
            self._profiles.get("long")
        )
        for which in ("short", "long"):
            rng = self._default_ranges.get(which)
            ctrl = self._scale_ctrls.get(which)
            if rng is not None and ctrl is not None:
                ctrl.set_default_range(*rng)
        if apply:
            self._apply_default_range("short", self.plot_short)
            self._apply_default_range("long", self.plot_long)

    def _reset_plot_range(self, which):
        plot = self.plot_short if which == "short" else self.plot_long
        if self._default_ranges.get(which) is None:
            self._default_ranges[which] = self._data_range_from_block(
                self._profiles.get(which)
            )
        self._apply_default_range(which, plot)

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self._results):
            return
        self._load_result(self._results[row])

    def _on_smooth_changed(self, *_args):
        if self._loading or self._current_result is None:
            return
        # 平滑改變：保留次峰搜尋窗，重算曲線與峰
        self._recompute_profiles(reset_view=False)

    def _auto_peaks_current(self):
        if self._current_result is None:
            return
        self._refresh_peaks_from_data()

    def _on_refine_changed(self, *_args):
        if self._loading or self._current_result is None:
            return
        self._refresh_peaks_from_data()

    def _apply_refine_current(self):
        if self._current_result is None:
            QMessageBox.information(self, "提示", "請先載入一組成功的配對。")
            return
        self._refresh_peaks_from_data()

    def _refine_count(self):
        return max(0, int(self.spin_refine.value()))

    def _profile_local_sum(self, block):
        """ROI 內扣背景剖面（與條帶列數對齊）。"""
        if not block:
            return None
        return np.asarray(
            block.get("sum_before_smooth")
            or block.get("sum_minus_median")
            or [],
            dtype=float,
        )

    def _apply_refine_to_long_peaks(
        self, long_mat, center_l, primary_y, secondary_y, long_block
    ):
        """先 smooth；N≥1 分段二次加密，峰位＝after refine 強度最大點（非整點偏移）。"""
        n = self._refine_count()
        meta = {"n_done": 0, "primary_xy": None, "secondary_xy": None}
        if primary_y is None:
            return primary_y, secondary_y, meta
        p1 = float(primary_y)
        p2 = None if secondary_y is None else float(secondary_y)
        if n <= 0 or not long_block:
            return p1, p2, meta

        ys = np.asarray(long_block.get("y_px") or [], dtype=float)
        before = np.asarray(
            long_block.get("sum_before_smooth")
            or long_block.get("sum_minus_median")
            or [],
            dtype=float,
        )
        after = np.asarray(
            long_block.get("sum_after_smooth")
            or long_block.get("sum_smooth")
            or [],
            dtype=float,
        )
        if ys.size == 0 or before.size != ys.size:
            return p1, p2, meta

        win = int(long_block.get("smooth_win") or self.spin_smooth.value() or 7)
        min_sep = float(getattr(self, "_min_sep", 12) or 12)
        refine_src = after if after.size == ys.size else before
        refine_pack = refine_profile_together(ys, refine_src, n, base_win=win)
        ys_r = np.asarray(refine_pack.get("y_px"), dtype=float)
        refined = np.asarray(refine_pack.get("sum_refine"), dtype=float)
        long_block["sum_refine"] = [float(v) for v in refined]
        long_block["y_refine_px"] = [float(v) for v in ys_r]
        lo, hi = self._get_sec_window()
        # 加密曲線上取強度最大，虛線對齊 after refine 峰頂
        p_new, s_new = refine_dual_peaks_quadratic(
            ys_r if ys_r.size and refined.size == ys_r.size else ys,
            refined if ys_r.size and refined.size == ys_r.size else refine_src,
            secondary_lo=lo,
            secondary_hi=hi,
            min_sep=min_sep,
            subpixel=False,
        )
        if p_new is not None:
            p1 = float(p_new)
        if p2 is not None and s_new is not None:
            p2 = float(s_new)
        meta = {
            "n_done": int(n),
            "primary_xy": (
                None
                if center_l is None
                else (float(center_l[0]), float(p1))
            ),
            "secondary_xy": (
                None
                if p2 is None or center_l is None
                else (float(center_l[0]), float(p2))
            ),
        }
        return float(p1), (None if p2 is None else float(p2)), meta

    def _auto_sec_window(self):
        """依目前長曝光剖面自動建議次峰窗，並套用為全域範圍。"""
        block = self._profiles.get("long")
        if not block:
            QMessageBox.information(self, "提示", "請先載入一組成功的配對。")
            return
        primary = self._argmax_peak_y(block, prefer="before")
        lo, hi = self._default_sec_window(block, primary)
        self._set_sec_window(lo, hi, refresh=True)

    def _load_result(self, result):
        self._loading = True
        try:
            self._current_result = result
            self._short_mat = None
            self._long_mat = None
            if not result.get("ok"):
                self.lbl_detail.setText(
                    f"{result.get('label')}\n失敗：{result.get('error') or '--'}"
                )
                self._clear_plots()
                self.lbl_peaks.setText("峰位：--（此配對失敗）")
                return
            try:
                short_mat = np.asarray(
                    load_numeric_matrix(result["short_path"]), dtype=float
                )
                long_mat = np.asarray(
                    load_numeric_matrix(result["long_path"]), dtype=float
                )
            except Exception as exc:
                self.lbl_detail.setText(
                    f"{result.get('label')}\n讀檔失敗：{exc}"
                )
                self._clear_plots()
                return
            self._short_mat = short_mat
            self._long_mat = long_mat
            ss = result.get("short_spot")
            ls = result.get("long_spot2")
            detail = f"{result.get('label')}"
            if ss is not None:
                detail += f"\n短斑 ({ss[0]:.1f}, {ss[1]:.1f})"
            if ls is not None:
                detail += f"\n長斑2 ({ls[0]:.1f}, {ls[1]:.1f})"
            self.lbl_detail.setText(detail)
            self._recompute_profiles(reset_view=True)
        finally:
            self._loading = False

    def _clear_plots(self):
        self._peak_guard = True
        try:
            for plot in (self.plot_short, self.plot_long):
                for item in list(plot.listDataItems()):
                    plot.removeItem(item)
            self._profiles = {"short": None, "long": None}
        finally:
            self._peak_guard = False

    def _analysis_pair(self):
        """回傳短／長分析矩陣（沿用主分頁 color scale 裁切若可用）。"""
        short = self._short_mat
        long = self._long_mat
        if short is None or long is None:
            return None, None
        tab = self._tab
        if tab is None:
            return short, long
        short_out = np.asarray(short, dtype=float)
        long_out = np.asarray(long, dtype=float)
        if hasattr(tab, "_basler_bw_scale_enabled") and tab._basler_bw_scale_enabled():
            lo = float(tab.spin_basler_black_below.value())
            hi = float(tab.spin_basler_white_above.value())
            short_out = tab._clip_color_scale_values(short_out, lo, hi)
        if hasattr(tab, "_long_bw_scale_enabled") and tab._long_bw_scale_enabled():
            lo = float(tab.spin_long_black_below.value())
            hi = float(tab.spin_long_white_above.value())
            long_out = tab._clip_color_scale_values(long_out, lo, hi)
        return short_out, long_out

    def _centers(self, result, long_mat):
        ss = result.get("short_spot")
        if ss is None:
            return None, None
        center_s = (float(ss[0]), float(ss[1]))
        h, w = np.asarray(long_mat).shape[:2]
        center_l = (
            float(np.clip(center_s[0], 0.0, max(0.0, w - 1.0))),
            float(np.clip(center_s[1], 0.0, max(0.0, h - 1.0))),
        )
        return center_s, center_l

    def _recompute_profiles(self, reset_view=False):
        result = self._current_result
        if result is None or not result.get("ok"):
            return
        short_an, long_an = self._analysis_pair()
        if short_an is None or long_an is None:
            return
        center_s, center_l = self._centers(result, long_an)
        if center_s is None:
            self.lbl_peaks.setText("峰位：--（無短曝光光斑）")
            return
        smooth_win = max(1, int(self.spin_smooth.value()))
        min_sep = max(1, int(self._min_sep))
        p_s = compute_horizontal_sum_profile(
            short_an,
            center_s,
            roi_w=self._roi_w,
            roi_h=self._roi_h,
            smooth_win=smooth_win,
            min_peak_distance=min_sep,
            find_secondary=False,
        )
        p_l = compute_horizontal_sum_profile(
            long_an,
            center_l,
            roi_w=self._roi_w,
            roi_h=self._roi_h,
            smooth_win=smooth_win,
            min_peak_distance=min_sep,
            find_secondary=True,
        )
        self._profiles["short"] = self._normalize_short_block(
            BaslerBatchDualTab._serialize_hsum_profile(p_s)
        )
        self._profiles["long"] = BaslerBatchDualTab._serialize_hsum_profile(p_l)
        self._redraw_curves("short", self.plot_short, self._profiles["short"])
        self._redraw_curves("long", self.plot_long, self._profiles["long"])
        self._update_default_ranges_from_profiles(apply=reset_view)
        self._ensure_sec_window()
        self._refresh_peaks_from_data()

    def _redraw_curves(self, which, plot, block):
        for item in list(plot.listDataItems()):
            plot.removeItem(item)
        legend = plot.plotItem.legend
        if legend is not None:
            legend.clear()
        if not block:
            return
        ys = np.asarray(block.get("y_px") or [], dtype=float)
        before = np.asarray(
            block.get("sum_before_smooth")
            or block.get("sum_minus_median")
            or [],
            dtype=float,
        )
        after = np.asarray(
            block.get("sum_after_smooth") or block.get("sum_smooth") or [],
            dtype=float,
        )
        if ys.size == 0:
            return
        win = block.get("smooth_win", self.spin_smooth.value())
        plot.plot(
            ys,
            before,
            pen=pg.mkPen("#1E88E5", width=1.4),
            name="raw",
        )
        plot.plot(
            ys,
            after,
            pen=pg.mkPen("#E65100", width=2.0),
            name=f"after smooth (win={win})",
        )
        # 以目前 spin 為準，避免 block 內舊 refine_iters 不同步
        n_ref = int(self._refine_count())
        if n_ref > 0:
            ys_r = np.asarray(block.get("y_refine_px") or [], dtype=float)
            refined = np.asarray(block.get("sum_refine") or [], dtype=float)
            if refined.size == 0 or ys_r.size != refined.size:
                src = after if after.size == ys.size else before
                pack = refine_profile_together(ys, src, n_ref, base_win=int(win))
                ys_r = np.asarray(pack.get("y_px"), dtype=float)
                refined = np.asarray(pack.get("sum_refine"), dtype=float)
            if ys_r.size and refined.size == ys_r.size:
                plot.plot(
                    ys_r,
                    refined,
                    pen=pg.mkPen("#6A1B9A", width=2.2),
                    name=f"after refine (quad×{n_ref})",
                )

    @staticmethod
    def _subpixel_peak_y(ys, vals, j):
        """在索引 j 附近做三点抛物线亚像素；失敗則回傳 ys[j]。"""
        ys = np.asarray(ys, dtype=float)
        vals = np.asarray(vals, dtype=float)
        j = int(j)
        if ys.size == 0 or j < 0 or j >= len(ys):
            return None
        if 0 < j < len(vals) - 1:
            y0v, y1v, y2v = float(vals[j - 1]), float(vals[j]), float(vals[j + 1])
            denom = y0v - 2.0 * y1v + y2v
            if abs(denom) > 1e-12 and y1v >= y0v and y1v >= y2v:
                delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
                return float(ys[j]) + delta
        return float(ys[j])

    @staticmethod
    def _argmax_peak_y(block, prefer="before", *, subpixel=True):
        """取剖面最高峰 Y（預設用平滑前；subpixel 控制是否三点抛物线）。"""
        if not block:
            return None
        ys = np.asarray(block.get("y_px") or [], dtype=float)
        if prefer == "after":
            vals = np.asarray(
                block.get("sum_after_smooth") or block.get("sum_smooth") or [],
                dtype=float,
            )
        else:
            vals = np.asarray(
                block.get("sum_before_smooth")
                or block.get("sum_minus_median")
                or [],
                dtype=float,
            )
        if ys.size == 0 or vals.size != ys.size:
            return None
        finite = np.isfinite(vals)
        if not np.any(finite):
            return None
        i = int(np.nanargmax(np.where(finite, vals, -np.inf)))
        if not subpixel:
            return float(ys[i])
        return BatchHSumResultsDialog._subpixel_peak_y(ys, vals, i)

    def _peak_in_window(
        self, block, y_lo, y_hi, *, exclude_y=None, prefer="after", subpixel=True
    ):
        """在 [y_lo, y_hi] 內找最高峰。"""
        if not block:
            return None
        ys = np.asarray(block.get("y_px") or [], dtype=float)
        if prefer == "before":
            vals = np.asarray(
                block.get("sum_before_smooth")
                or block.get("sum_minus_median")
                or [],
                dtype=float,
            )
        else:
            vals = np.asarray(
                block.get("sum_after_smooth") or block.get("sum_smooth") or [],
                dtype=float,
            )
        if ys.size == 0 or vals.size != ys.size:
            return None
        lo, hi = float(min(y_lo, y_hi)), float(max(y_lo, y_hi))
        mask = (ys >= lo) & (ys <= hi) & np.isfinite(vals)
        min_sep = max(1, int(self._min_sep))
        if exclude_y is not None:
            mask &= np.abs(ys - float(exclude_y)) >= float(min_sep)
        if not np.any(mask):
            return None
        idx = np.flatnonzero(mask)
        j = int(idx[int(np.argmax(vals[idx]))])
        if not subpixel:
            return float(np.clip(float(ys[j]), lo, hi))
        y_peak = self._subpixel_peak_y(ys, vals, j)
        if y_peak is None:
            return None
        return float(np.clip(y_peak, lo, hi))

    def _normalize_short_block(self, block):
        """短曝光只保留單一主峰（平滑前最高峰）。"""
        if not block:
            return None
        out = dict(block)
        p = self._argmax_peak_y(out, prefer="before")
        out["primary_y_px"] = p
        out["secondary_y_px"] = None
        out["peak1_y_px"] = p
        out["peak2_y_px"] = None
        out["peak_delta_y_px"] = None
        out["peak_delta_y_um"] = None
        return out

    def _default_sec_window(self, block, primary_y):
        ys = np.asarray((block or {}).get("y_px") or [], dtype=float)
        if ys.size == 0:
            return (0.0, 100.0)
        y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
        span = max(y1 - y0, 1.0)
        min_sep = max(1.0, float(self._min_sep))
        sec = (block or {}).get("secondary_y_px")
        if sec is not None:
            half = max(40.0, 0.12 * span)
            return (float(sec) - half, float(sec) + half)
        if primary_y is None:
            mid = 0.5 * (y0 + y1)
            return (mid, y1)
        if float(primary_y) < 0.5 * (y0 + y1):
            return (float(primary_y) + min_sep, y1)
        return (y0, float(primary_y) - min_sep)

    def _get_sec_window(self):
        """回傳目前全域次峰窗 (lo, hi)。"""
        if self._sec_window is not None:
            lo, hi = self._sec_window
            return float(min(lo, hi)), float(max(lo, hi))
        lo = float(self.spin_sec_lo.value())
        hi = float(self.spin_sec_hi.value())
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _set_sec_window(self, lo, hi, *, refresh=True):
        """設定全域次峰窗，同步 spin／圖上紅區。"""
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        if hi <= lo:
            hi = lo + 1.0
        self._sec_window = (lo, hi)
        self._region_guard = True
        try:
            self.spin_sec_lo.blockSignals(True)
            self.spin_sec_hi.blockSignals(True)
            for spin in (self.spin_sec_lo, self.spin_sec_hi):
                if hi > spin.maximum():
                    spin.setMaximum(max(hi * 1.2, spin.maximum()))
                if lo < spin.minimum():
                    spin.setMinimum(min(lo, spin.minimum()))
            self.spin_sec_lo.setValue(lo)
            self.spin_sec_hi.setValue(hi)
            self.spin_sec_lo.blockSignals(False)
            self.spin_sec_hi.blockSignals(False)
            self.sec_region.setRegion((lo, hi))
            try:
                self.sec_region.setMovable(False)
            except Exception:
                pass
        finally:
            self._region_guard = False
        if refresh and not self._loading:
            self._refresh_peaks_from_data()

    def _ensure_sec_window(self):
        """若尚未設定全域窗，依目前剖面自動初始化一次。"""
        if self._sec_window is not None:
            lo, hi = self._get_sec_window()
            self._set_sec_window(lo, hi, refresh=False)
            return
        block = self._profiles.get("long")
        primary = self._argmax_peak_y(block, prefer="before")
        lo, hi = self._default_sec_window(block, primary)
        self._set_sec_window(lo, hi, refresh=False)

    def _on_sec_spin_changed(self, *_args):
        if self._region_guard or self._loading:
            return
        lo = float(self.spin_sec_lo.value())
        hi = float(self.spin_sec_hi.value())
        self._set_sec_window(lo, hi, refresh=True)

    def _refresh_peaks_from_data(self):
        """抓兩個峰後，依精修次數更新峰線與 ΔY。"""
        short_b = self._profiles.get("short")
        long_b = self._profiles.get("long")
        use_sub = self._refine_count() > 0
        p_s = self._argmax_peak_y(short_b, prefer="before", subpixel=use_sub)
        p_l = self._argmax_peak_y(long_b, prefer="before", subpixel=use_sub)
        lo, hi = self._get_sec_window()
        # 次峰種子＝搜尋窗內曲線最高
        p2 = self._peak_in_window(
            long_b, lo, hi, exclude_y=p_l, prefer="after", subpixel=use_sub
        )
        refine_meta = {"n_done": 0}
        result = self._current_result
        long_an = None
        center_l = None
        if result is not None and result.get("ok"):
            _short_an, long_an = self._analysis_pair()
            if long_an is not None:
                _cs, center_l = self._centers(result, long_an)
        if p_l is not None and p2 is not None and self._refine_count() > 0:
            p_l, p2, refine_meta = self._apply_refine_to_long_peaks(
                long_an, center_l, p_l, p2, long_b
            )
        elif long_b is not None:
            long_b.pop("sum_refine", None)
            long_b.pop("y_refine_px", None)
        self._last_refine_meta = refine_meta
        self._peak_guard = True
        try:
            self.line_s2.hide()
            self.line_s1.setMovable(False)
            self.line_l1.setMovable(False)
            self.line_l2.setMovable(False)
            if p_s is not None:
                self.line_s1.setValue(float(p_s))
            if p_l is not None:
                self.line_l1.setValue(float(p_l))
            if p2 is not None:
                self.line_l2.setValue(float(p2))
                self.line_l2.show()
            elif p_l is not None:
                self.line_l2.setValue(float(p_l))
        finally:
            self._peak_guard = False
        if short_b is not None:
            self._profiles["short"] = self._block_with_manual_peaks(
                short_b, p_s, int(self.spin_smooth.value()), single=True
            )
        if long_b is not None:
            peaks = None if p_l is None else (
                (float(p_l), float(p2)) if p2 is not None else (float(p_l), float(p_l))
            )
            long_out = self._block_with_manual_peaks(
                long_b, peaks, int(self.spin_smooth.value()), single=False
            )
            if long_out is not None:
                long_out["refine_iters"] = int(self._refine_count())
                long_out["refine_done"] = int(refine_meta.get("n_done") or 0)
                if refine_meta.get("primary_xy") is not None:
                    long_out["primary_xy"] = list(refine_meta["primary_xy"])
                if refine_meta.get("secondary_xy") is not None:
                    long_out["secondary_xy"] = list(refine_meta["secondary_xy"])
                if self._refine_count() <= 0:
                    long_out.pop("sum_refine", None)
                    long_out.pop("y_refine_px", None)
            self._profiles["long"] = long_out
        self._redraw_curves("short", self.plot_short, self._profiles.get("short"))
        self._redraw_curves("long", self.plot_long, self._profiles.get("long"))
        self._update_peak_label()

    def _format_peak_bits(self, tag, peaks, *, single=False):
        if peaks is None:
            return f"{tag}: --"
        if single:
            p1 = peaks if not isinstance(peaks, (list, tuple)) else peaks[0]
            return f"{tag}: 峰={float(p1):.2f}"
        p1, p2 = peaks
        dy = abs(float(p2) - float(p1))
        return (
            f"{tag}: 峰1={p1:.3f} 峰2={p2:.3f} "
            f"ΔY={dy:.3f} px（{self._px_to_um(dy):.3f} μm）"
        )

    def _update_peak_label(self):
        s = float(self.line_s1.value())
        p1 = float(self.line_l1.value())
        p2 = float(self.line_l2.value())
        lo, hi = self._get_sec_window()
        win = int(self.spin_smooth.value())
        n_req = self._refine_count()
        n_done = int(getattr(self, "_last_refine_meta", {}).get("n_done") or 0)
        if n_req <= 0:
            refine_txt = "精修=關"
        else:
            refine_txt = f"精修={n_done}/{n_req}"
        self.lbl_peaks.setText(
            f"平滑窗={win}｜{refine_txt}｜"
            + self._format_peak_bits("短", s, single=True)
            + "｜"
            + self._format_peak_bits("長", (p1, p2))
            + f"｜次峰窗(共用) [{lo:.0f}, {hi:.0f}]"
        )

    def _block_with_manual_peaks(self, block, peaks, smooth_win, *, single=False):
        """複製剖面 block，覆寫峰位／間距／平滑窗。"""
        if not block:
            return None
        out = dict(block)
        out["smooth_win"] = int(smooth_win)
        if peaks is None:
            out["peak1_y_px"] = None
            out["peak2_y_px"] = None
            out["peak_delta_y_px"] = None
            out["peak_delta_y_um"] = None
            out["primary_y_px"] = None
            out["secondary_y_px"] = None
            return out
        if single:
            p1 = float(peaks if not isinstance(peaks, (list, tuple)) else peaks[0])
            out["peak1_y_px"] = p1
            out["peak2_y_px"] = None
            out["peak_delta_y_px"] = None
            out["peak_delta_y_um"] = None
            out["primary_y_px"] = p1
            out["secondary_y_px"] = None
            return out
        p1, p2 = float(peaks[0]), float(peaks[1])
        lo, hi = (p1, p2) if p1 <= p2 else (p2, p1)
        dy = float(hi - lo)
        out["peak1_y_px"] = lo
        out["peak2_y_px"] = hi
        out["peak_delta_y_px"] = dy
        out["peak_delta_y_um"] = self._px_to_um(dy)
        out["primary_y_px"] = p1
        out["secondary_y_px"] = p2
        return out

    def _current_hsum_payload(self):
        """目前配對的 short/long 剖面（含目前峰位）。"""
        if self._current_result is None:
            return None
        win = int(self.spin_smooth.value())
        self._refresh_peaks_from_data()
        lo, hi = self._get_sec_window()
        return {
            "strip_roi_w": self._roi_w,
            "strip_roi_h": self._roi_h,
            "profile_smooth_win": win,
            "min_peak_distance_px": self._min_sep,
            "refine_iters": self._refine_count(),
            "secondary_search_window_y": [float(lo), float(hi)],
            "short": self._profiles.get("short"),
            "long": self._profiles.get("long"),
        }

    def _compute_hsum_for_result(self, result, smooth_win, use_overrides=True):
        """為任意配對計算剖面；套用已存次峰搜尋窗與精修次數。"""
        if not result.get("ok") or result.get("short_spot") is None:
            return None
        try:
            short_mat = np.asarray(
                load_numeric_matrix(result["short_path"]), dtype=float
            )
            long_mat = np.asarray(
                load_numeric_matrix(result["long_path"]), dtype=float
            )
        except Exception:
            return None
        tab = self._tab
        short_an = short_mat
        long_an = long_mat
        if tab is not None:
            if hasattr(tab, "_basler_bw_scale_enabled") and tab._basler_bw_scale_enabled():
                short_an = tab._clip_color_scale_values(
                    short_mat,
                    float(tab.spin_basler_black_below.value()),
                    float(tab.spin_basler_white_above.value()),
                )
            if hasattr(tab, "_long_bw_scale_enabled") and tab._long_bw_scale_enabled():
                long_an = tab._clip_color_scale_values(
                    long_mat,
                    float(tab.spin_long_black_below.value()),
                    float(tab.spin_long_white_above.value()),
                )
        center_s, center_l = self._centers(result, long_an)
        if center_s is None:
            return None
        p_s = compute_horizontal_sum_profile(
            short_an,
            center_s,
            roi_w=self._roi_w,
            roi_h=self._roi_h,
            smooth_win=smooth_win,
            min_peak_distance=self._min_sep,
            find_secondary=False,
        )
        p_l = compute_horizontal_sum_profile(
            long_an,
            center_l,
            roi_w=self._roi_w,
            roi_h=self._roi_h,
            smooth_win=smooth_win,
            min_peak_distance=self._min_sep,
            find_secondary=True,
        )
        block_s = self._normalize_short_block(
            BaslerBatchDualTab._serialize_hsum_profile(p_s)
        )
        block_l = BaslerBatchDualTab._serialize_hsum_profile(p_l)
        use_sub = self._refine_count() > 0
        p_short = self._argmax_peak_y(
            block_s, prefer="before", subpixel=use_sub
        )
        p_long = self._argmax_peak_y(
            block_l, prefer="before", subpixel=use_sub
        )
        win_reg = self._get_sec_window()
        if win_reg[1] <= win_reg[0]:
            win_reg = self._default_sec_window(block_l, p_long)
        p2 = self._peak_in_window(
            block_l,
            win_reg[0],
            win_reg[1],
            exclude_y=p_long,
            prefer="after",
            subpixel=use_sub,
        )
        refine_meta = {"n_done": 0}
        if p_long is not None and p2 is not None and self._refine_count() > 0:
            p_long, p2, refine_meta = self._apply_refine_to_long_peaks(
                long_an, center_l, p_long, p2, block_l
            )
        elif block_l is not None:
            block_l.pop("sum_refine", None)
            block_l.pop("y_refine_px", None)
        block_s = self._block_with_manual_peaks(
            block_s, p_short, smooth_win, single=True
        )
        long_peaks = None
        if p_long is not None:
            long_peaks = (
                (float(p_long), float(p2))
                if p2 is not None
                else (float(p_long), float(p_long))
            )
        block_l = self._block_with_manual_peaks(
            block_l, long_peaks, smooth_win, single=False
        )
        if block_l is not None:
            block_l["refine_iters"] = int(self._refine_count())
            block_l["refine_done"] = int(refine_meta.get("n_done") or 0)
            if refine_meta.get("primary_xy") is not None:
                block_l["primary_xy"] = list(refine_meta["primary_xy"])
            if refine_meta.get("secondary_xy") is not None:
                block_l["secondary_xy"] = list(refine_meta["secondary_xy"])
        return {
            "strip_roi_w": self._roi_w,
            "strip_roi_h": self._roi_h,
            "profile_smooth_win": int(smooth_win),
            "min_peak_distance_px": self._min_sep,
            "refine_iters": self._refine_count(),
            "secondary_search_window_y": [
                float(min(win_reg[0], win_reg[1])),
                float(max(win_reg[0], win_reg[1])),
            ],
            "short": block_s,
            "long": block_l,
        }

    def _export_dir_start(self):
        tab = self._tab
        if tab is not None and getattr(tab, "save_dir_path", None):
            return tab.save_dir_path
        return self._store.get("data_folder") or ""

    def _export_current(self):
        hsum = self._current_hsum_payload()
        if not hsum or (not hsum.get("short") and not hsum.get("long")):
            QMessageBox.warning(self, "警告", "目前沒有可匯出的水平加總資料。")
            return
        result = self._current_result
        stem = BatchAnalysisResultsDialog._safe_stem(
            result.get("label"), result.get("index", 0)
        )
        stamp = export_timestamp_tag()
        start = self._export_dir_start()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出目前配對水平加總",
            os.path.join(start, f"{stem}_HSumPeaks.csv")
            if start
            else f"{stem}_HSumPeaks.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        base = os.path.splitext(path)[0]
        # 若使用者選的是 .csv，拆成 Short/Long 兩個檔寫在同目錄
        folder = os.path.dirname(path)
        base_nostamp = os.path.join(folder, f"{stem}_ShortLong")
        outs = BaslerBatchDualTab._export_hsum_peaks_files(
            base_nostamp, hsum, stamp
        )
        # 另寫一份摘要（峰位＋間距）到使用者選的檔名
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Item", "Value", "Unit"])
            w.writerow(["Label", result.get("label"), ""])
            w.writerow(["Index", result.get("index"), ""])
            w.writerow(["Smooth Win", hsum.get("profile_smooth_win"), ""])
            w.writerow(["Refine Iters", hsum.get("refine_iters"), ""])
            w.writerow(["Strip ROI W", hsum.get("strip_roi_w"), "px"])
            w.writerow(["Strip ROI H", hsum.get("strip_roi_h"), "px"])
            for which, key in (("Short", "short"), ("Long", "long")):
                b = hsum.get(key) or {}
                w.writerow([f"{which} Peak1 Y", b.get("peak1_y_px"), "px"])
                w.writerow([f"{which} Peak2 Y", b.get("peak2_y_px"), "px"])
                w.writerow([f"{which} Peak Delta Y", b.get("peak_delta_y_px"), "px"])
                w.writerow(
                    [f"{which} Peak Delta Y", b.get("peak_delta_y_um"), "μm"]
                )
                if b.get("refine_done") is not None:
                    w.writerow([f"{which} Refine Done", b.get("refine_done"), ""])
                if b.get("primary_xy"):
                    w.writerow(
                        [f"{which} Primary XY", b.get("primary_xy"), "px"]
                    )
                if b.get("secondary_xy"):
                    w.writerow(
                        [f"{which} Secondary XY", b.get("secondary_xy"), "px"]
                    )
        msg = f"已匯出：\n{path}"
        if outs:
            msg += "\n" + "\n".join(outs)
        QMessageBox.information(self, "匯出完成", msg)

    def _export_all(self):
        ok_results = [r for r in self._results if r.get("ok")]
        if not ok_results:
            QMessageBox.warning(self, "警告", "沒有成功的配對可匯出。")
            return
        # 先確認全域次峰窗
        self._ensure_sec_window()
        stamp = export_timestamp_tag()
        start = self._export_dir_start()
        folder = QFileDialog.getExistingDirectory(
            self, "選擇匯出資料夾", start
        )
        if not folder:
            return
        smooth_win = max(1, int(self.spin_smooth.value()))
        collector = []
        progress = QProgressDialog(
            "匯出水平加總…", "取消", 0, len(ok_results), self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        try:
            for i, r in enumerate(ok_results):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                progress.setLabelText(
                    f"{i + 1}/{len(ok_results)}：{r.get('label', '')}"
                )
                QApplication.processEvents()
                hsum = self._compute_hsum_for_result(
                    r, smooth_win, use_overrides=True
                )
                if not hsum:
                    continue
                stem = BatchAnalysisResultsDialog._safe_stem(
                    r.get("label"), r.get("index", 0)
                )
                base = os.path.join(folder, f"{stem}_ShortLong")
                BaslerBatchDualTab._export_hsum_peaks_files(
                    base, hsum, stamp
                )
                collector.append(
                    {
                        "index": r.get("index"),
                        "label": r.get("label") or stem,
                        "hsum": hsum,
                    }
                )
            progress.setValue(len(ok_results))
        finally:
            progress.close()
        if not collector:
            QMessageBox.warning(self, "警告", "沒有可寫入的水平加總資料。")
            return
        all_path = with_export_timestamp(
            os.path.join(folder, "batch_hsum_all_profiles.csv"), stamp=stamp
        )
        BaslerBatchDualTab._write_batch_hsum_all_csv(all_path, collector)
        QMessageBox.information(
            self,
            "匯出完成",
            f"已匯出 {len(collector)} 組配對\n彙整：{all_path}",
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
        self._collage_roi_bounds = None  # (x0, y0, x1, y1) long ROI
        self.data_folder = ""
        self._data_pairs = []
        self._axial_roi_xywh = None  # 一維投影專用顯示 ROI（不寫入質心 ROI）
        self._hsum_roi_xywh = None  # 水平加總條帶顯示 ROI
        self._hsum_sec_window = None
        self._gaussian_roi_xywh = None  # 高斯下光斑搜尋帶顯示 ROI
        self._spot_algo = "centroid"
        self._batch_analysis_store = None  # 批次分析結果（供檢視視窗）
        self._batch_results_dialog = None
        self._batch_hsum_dialog = None
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
        self._place_batch_view_on_top_bar()
        self._place_export_on_top_bar()

        self.btn_batch_m1_dir.show()
        self.btn_batch_m1_dir.setText("選擇資料夾")
        self.btn_batch_m1_dir.setStyleSheet(_STYLE_BTN)
        # 路徑改顯示在下方「資料」區；此兩段敘述隱藏
        if getattr(self, "lbl_batch_m1_info", None) is not None:
            self.lbl_batch_m1_info.hide()
            self.lbl_batch_m1_info.setMaximumHeight(0)
        if getattr(self, "lbl_batch_current_files", None) is not None:
            self.lbl_batch_current_files.hide()
            self.lbl_batch_current_files.setMaximumHeight(0)
        self.btn_batch_m2_dir.hide()
        self.lbl_batch_m2_info.hide()

        # 儲存路徑與資料夾分開：未手動選儲存時保持空白
        self.save_dir_path = ""
        path_lbl = getattr(self, "lbl_batch_dir_path", None)
        if path_lbl is not None:
            path_lbl.setText("")
            path_lbl.setToolTip("尚未選擇儲存資料夾")

        # 此分頁不用的狀態列項目（避免左上角擠壓重疊）
        for w in (
            getattr(self, "combo_batch_mode", None),
            getattr(self, "chk_batch_normalize", None),
            getattr(self, "lbl_batch_max1", None),
            getattr(self, "lbl_batch_max2", None),
            getattr(self, "lbl_batch_ratio", None),
            getattr(self, "lbl_batch_distance_xy", None),
            getattr(self, "lbl_batch_distance_total", None),
            getattr(self, "lbl_batch_real_distance_xy", None),
            getattr(self, "lbl_batch_real_distance_total", None),
            getattr(self, "lbl_batch_ab_distance_xy", None),
            getattr(self, "lbl_batch_ab_distance_total", None),
            getattr(self, "lbl_batch_ab_real_distance_xy", None),
            getattr(self, "lbl_batch_ab_real_distance_total", None),
        ):
            if w is not None:
                w.hide()
                w.setMaximumHeight(0)

        try:
            self.btn_batch_m1_dir.clicked.disconnect()
        except TypeError:
            pass
        try:
            self.btn_batch_m2_dir.clicked.disconnect()
        except TypeError:
            pass
        self.btn_batch_m1_dir.clicked.connect(self.select_data_folder)

        self._remove_distance_table()
        self._build_dual_heatmap_layout()
        self._build_data_pair_controls()
        self._build_spot_controls()
        self._wire_hsum_focus_param_signals()
        self._sync_main_view_for_algo()
        self._compact_left_panel()
        self._relax_top_status_area()
        self._update_short_heatmap_title()
        self._update_long_heatmap_title()
        # 不自動載入 test；等使用者選資料夾／配對後再顯示 heatmap


    def _relax_top_status_area(self):
        """左上固定區：只保留必要狀態列，避免與下方捲動區互擠造成文字重疊。"""
        status = getattr(self, "lbl_batch_status", None)
        if status is None:
            return
        top_fixed = status.parentWidget()
        if top_fixed is None:
            return

        # 隱藏「選擇工作模式」等殘留標題
        for w in top_fixed.findChildren(QLabel):
            t = (w.text() or "").strip()
            if t.startswith("選擇工作模式") or t.startswith("Batch"):
                w.hide()
                w.setMaximumHeight(0)

        keep = {
            getattr(self, "btn_batch_m1_dir", None),
            getattr(self, "lbl_batch_pair_info", None),
            getattr(self, "lbl_batch_status", None),
            getattr(self, "lbl_batch_size", None),
        }
        keep.discard(None)

        for w in (
            getattr(self, "lbl_batch_status", None),
            getattr(self, "lbl_batch_pair_info", None),
            getattr(self, "lbl_batch_size", None),
        ):
            if w is None:
                continue
            w.setWordWrap(True)
            w.setMaximumHeight(16777215)
            w.setMinimumHeight(0)
            w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            w.setStyleSheet("font-size: 12px; color: #424242;")

        # 明確隱藏路徑敘述
        for w in (
            getattr(self, "lbl_batch_m1_info", None),
            getattr(self, "lbl_batch_current_files", None),
        ):
            if w is not None:
                w.hide()
                w.setMaximumHeight(0)
                w.clear()

        title = None
        for w in top_fixed.findChildren(QLabel):
            if (w.text() or "").strip() == "計算數據與狀態":
                title = w
                break
        if title is not None:
            title.setStyleSheet("font-weight: bold; font-size: 12px; color: #424242;")
            title.show()
            title.setMaximumHeight(16777215)

        top_fixed.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = top_fixed.layout()
        if lay is not None:
            lay.setSpacing(6)
            lay.setContentsMargins(10, 8, 10, 8)

        # 下方捲動區吃剩餘高度；上方依內容高度，不被壓扁
        scroll = getattr(self, "batch_left_scroll", None)
        outer = top_fixed.parentWidget()
        if outer is not None and scroll is not None:
            outer_lay = outer.layout()
            if outer_lay is not None:
                outer_lay.setStretchFactor(top_fixed, 0)
                outer_lay.setStretchFactor(scroll, 1)

        if getattr(self, "lbl_batch_size", None) is not None:
            self.lbl_batch_size.setText("矩陣大小: --")

    def _place_batch_view_on_top_bar(self):
        """「查看批次分析資料」放到「選擇儲存資料夾」左側。"""
        self.btn_view_batch_results = QPushButton("查看批次分析資料")
        self.btn_view_batch_results.setMinimumHeight(38)
        self.btn_view_batch_results.setStyleSheet(_STYLE_BTN)
        self.btn_view_batch_results.setEnabled(False)
        self.btn_view_batch_results.setToolTip(
            "選擇檢視：Heatmap（短／長熱圖）或水平加總（短＋長剖面）"
        )
        self.btn_view_batch_results.clicked.connect(
            self._on_view_batch_results_clicked
        )

        sel = getattr(self, "btn_batch_select_dir", None)
        if sel is None:
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

        parent = sel.parentWidget()
        found = _find_layout_index(
            parent.layout() if parent is not None else None, sel
        )
        if found is None:
            return
        layout, idx = found
        layout.insertWidget(idx, self.btn_view_batch_results)

    def _place_export_on_top_bar(self):
        """匯出按鈕放到「選擇儲存資料夾／路徑」右側（同一列）。"""
        self.btn_export_dual = QPushButton("匯出圖片／參數／光斑")
        self.btn_export_dual.setMinimumHeight(38)
        self.btn_export_dual.setStyleSheet(_STYLE_BTN_PRIMARY)
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
            data_box = getattr(self, "_data_pair_box", None)
            if data_box is not None and data_box.isAncestorOf(w):
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
        spot_box = getattr(self, "_spot_controls_box", None)
        data_box = getattr(self, "_data_pair_box", None)
        keep_buttons = {
            getattr(self, "btn_apply_spot", None),
            getattr(self, "btn_clear_spots", None),
            getattr(self, "btn_open_collage", None),
            getattr(self, "btn_rescan_pairs", None),
        }
        keep_buttons.discard(None)

        for w in content.findChildren(QRadioButton):
            if spot_box is not None and spot_box.isAncestorOf(w):
                continue
            w.hide()
            w.setMaximumHeight(0)

        for w in content.findChildren(QCheckBox):
            if spot_box is not None and spot_box.isAncestorOf(w):
                continue
            if data_box is not None and data_box.isAncestorOf(w):
                continue
            t = (w.text() or "")
            if t in ("log", "color scale", "平滑") or "長曝光搜尋範圍" in t:
                continue
            if w is getattr(self, "chk_long_search_roi", None):
                continue
            if w is getattr(self, "chk_hide_profile_cross", None):
                continue
            w.hide()
            w.setMaximumHeight(0)

        for w in content.findChildren(QPushButton):
            if spot_box is not None and spot_box.isAncestorOf(w):
                continue
            if data_box is not None and data_box.isAncestorOf(w):
                continue
            if w in keep_buttons:
                continue
            # 上方固定區的選檔按鈕不在 left_content 內；此處全藏
            w.hide()
            w.setMaximumHeight(0)

        for w in content.findChildren(QLabel):
            if spot_box is not None and spot_box.isAncestorOf(w):
                continue
            if data_box is not None and data_box.isAncestorOf(w):
                continue
            t = (w.text() or "").strip()
            if not t:
                w.hide()
                w.setMaximumHeight(0)
                continue
            keep_prefix = (
                "分析",
                "資料",
                "演算法",
                "質心",
                "一維",
                "短曝光",
                "長曝光",
                "兩光斑",
                "隱藏剖面",
                "配對",
                "平滑",
                "最小峰距",
                "條帶",
                "批次",
                "查看",
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
        black_spin,
        white_spin,
        smooth_chk,
        smooth_spin,
        object_name,
        extra_widgets=None,
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
        toolbar.addWidget(_group(bw_chk, black_spin, white_spin))
        toolbar.addWidget(_vsep())
        toolbar.addWidget(_group(smooth_chk, smooth_spin))
        if extra_widgets:
            toolbar.addWidget(_vsep())
            toolbar.addWidget(_group(*extra_widgets))
        toolbar.addStretch(1)

        bar_host = QWidget()
        bar_host.setObjectName(object_name)
        bar_host.setLayout(toolbar)
        bar_host.setStyleSheet(
            f"#{object_name} {{"
            " background: #F5F7FA; border: 1px solid #CFD8DC;"
            " border-radius: 6px; }"
            f"#{object_name} QCheckBox {{ color: #37474F; font-weight: bold; }}"
            f"#{object_name} QLabel {{ color: #546E7A; }}"
            f"#{object_name} QPushButton {{"
            " color: #424242; background: #FAFAFA; border: 1px solid #BDBDBD;"
            " border-radius: 4px; padding: 4px 10px; font-size: 12px; }"
            f"#{object_name} QPushButton:checked {{"
            " color: #FAFAFA; background: #5E35B1; border-color: #4527A0;"
            " font-weight: bold; }"
            f"#{object_name} QPushButton:hover:!checked {{ background: #EEEEEE; }}"
        )
        return bar_host

    def _make_thresh_overlay_button(self, which):
        """heatmap 上方：切換半透明門檻分布。which = 'short' | 'long'"""
        btn = QPushButton("顯示門檻")
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(
            "在熱圖上以半透明色顯示目前門檻％對應的光斑區域（背景扣除＋最大連通區）"
        )
        if which == "short":
            btn.toggled.connect(self._on_short_thresh_overlay_toggled)
            self.btn_short_show_thresh = btn
        else:
            btn.toggled.connect(self._on_long_thresh_overlay_toggled)
            self.btn_long_show_thresh = btn
        return btn

    def _make_horizontal_sum_button(self, which):
        """heatmap 上方：開啟一維投影水平加總圖。which = 'short' | 'long'"""
        btn = QPushButton("查看水平加總圖")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(
            "一維投影：以目前條帶參數開啟水平加總剖面（raw／扣背景／平滑）"
        )
        if which == "short":
            btn.clicked.connect(lambda: self.show_horizontal_sum_profile("short"))
            self.btn_short_hsum = btn
        else:
            btn.clicked.connect(lambda: self.show_horizontal_sum_profile("long"))
            self.btn_long_hsum = btn
        btn.hide()
        return btn

    def _update_horizontal_sum_buttons_visibility(self):
        """單檔一維投影才顯示「查看水平加總圖」（在顯示門檻右邊）。"""
        show = self._current_spot_algo() == "axial_1d"
        for attr in ("btn_short_hsum", "btn_long_hsum"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setVisible(show)

    def _axial_strip_params(self):
        """一維投影條帶參數（寬／高／平滑／最小峰距）。"""
        roi_w = (
            int(self.spin_axial_roi_w.value())
            if getattr(self, "spin_axial_roi_w", None)
            else 160
        )
        roi_h = (
            int(self.spin_axial_roi_h.value())
            if getattr(self, "spin_axial_roi_h", None)
            else 512
        )
        smooth_win = (
            int(self.spin_axial_smooth.value())
            if getattr(self, "spin_axial_smooth", None)
            else 7
        )
        min_sep = (
            int(self.spin_axial_min_sep.value())
            if getattr(self, "spin_axial_min_sep", None)
            else 12
        )
        return roi_w, roi_h, smooth_win, min_sep

    def _active_hsum_strip_params(self):
        """水平加總／波形檢視用條帶參數（優先水平加總演算法 UI）。"""
        if self._current_spot_algo() == "hsum":
            kw = self._hsum_algo_kwargs()
            return (
                int(kw["roi_w"]),
                int(kw["roi_h"]),
                int(kw["smooth_win"]),
                int(kw["min_peak_distance"]),
            )
        return self._axial_strip_params()

    @staticmethod
    def _is_hsum_algorithm_id(algo_id):
        return algo_id in (
            "horizontal_sum_dual_peak",
            "hsum",
        )

    @staticmethod
    def _is_axial_algorithm_id(algo_id):
        return algo_id in (
            "axial_1d_projection_diff",
            "axial_1d",
        )

    @staticmethod
    def _strip_params_from_algo_dict(params):
        """自 algorithm_params（或等同 dict）取出條帶參數。"""
        params = params or {}
        return (
            int(params.get("strip_roi_w", 160) or 160),
            int(params.get("strip_roi_h", 512) or 512),
            int(params.get("profile_smooth_win", 7) or 7),
            int(params.get("min_peak_distance_px", 12) or 12),
        )

    @staticmethod
    def _serialize_hsum_profile(profile):
        """水平加總剖面 → JSON／CSV 可序列化 dict；失敗回傳 None。"""
        if not profile:
            return None
        y = np.asarray(profile.get("y_full_px"), dtype=float)
        raw = np.asarray(profile.get("sum_raw"), dtype=float)
        bg = np.asarray(profile.get("sum_minus_median"), dtype=float)
        sm = np.asarray(profile.get("sum_smooth"), dtype=float)
        if y.size == 0 or raw.size != y.size:
            return None
        roi = profile.get("roi_xywh")
        center = profile.get("center_xy")
        p1 = profile.get("primary_y_full")
        p2 = profile.get("secondary_y_full")
        p1 = None if p1 is None else float(p1)
        p2 = None if p2 is None else float(p2)
        # 依 Y 排序：峰1＝較小 Y、峰2＝較大 Y
        peak_lo = peak_hi = None
        dy_px = None
        if p1 is not None and p2 is not None:
            peak_lo, peak_hi = (p1, p2) if p1 <= p2 else (p2, p1)
            dy_px = float(peak_hi - peak_lo)
        elif p1 is not None:
            peak_lo = p1
        elif p2 is not None:
            peak_lo = p2
        return {
            "y_px": [float(v) for v in y],
            "sum_raw": [float(v) for v in raw],
            "sum_before_smooth": [float(v) for v in bg],
            "sum_minus_median": [float(v) for v in bg],
            "sum_after_smooth": [float(v) for v in sm],
            "sum_smooth": [float(v) for v in sm],
            "primary_y_px": p1,
            "secondary_y_px": p2,
            "peak1_y_px": peak_lo,
            "peak2_y_px": peak_hi,
            "peak_delta_y_px": dy_px,
            "peak_delta_y_um": (
                None if dy_px is None else float(dy_px) * 3.45
            ),
            "roi_xywh": list(roi) if roi is not None else None,
            "center_xy": list(center) if center is not None else None,
            "smooth_win": int(profile.get("smooth_win", 7) or 7),
            "n_points": int(y.size),
        }

    def _compute_hsum_profiles_for_mats(
        self,
        short_mat,
        long_mat,
        short_spot,
        *,
        strip_params=None,
        long_center=None,
        n_refine=None,
        secondary_window_y=None,
    ):
        """依短曝光光斑與條帶參數計算短／長水平加總（匯出／檢視共用）。

        長曝光峰位與主畫面一致：N=0 整點；N≥1 分段二次加密後取強度最大。
        """
        if short_spot is None:
            return None
        if strip_params is None:
            roi_w, roi_h, smooth_win, min_sep = self._active_hsum_strip_params()
        else:
            roi_w, roi_h, smooth_win, min_sep = strip_params
        if n_refine is None:
            if (
                self._current_spot_algo() == "hsum"
                and getattr(self, "spin_hsum_refine", None) is not None
            ):
                n_refine = int(self.spin_hsum_refine.value())
            else:
                n_refine = 0
        n_refine = max(0, int(n_refine))
        if secondary_window_y is None and self._current_spot_algo() == "hsum":
            sec = getattr(self, "_hsum_sec_window", None)
            if sec is not None and len(sec) >= 2:
                secondary_window_y = (float(sec[0]), float(sec[1]))
            else:
                hkw = self._hsum_algo_kwargs()
                secondary_window_y = hkw.get("secondary_window_y")

        out = {
            "strip_roi_w": int(roi_w),
            "strip_roi_h": int(roi_h),
            "profile_smooth_win": int(smooth_win),
            "min_peak_distance_px": int(min_sep),
            "refine_iters": int(n_refine),
            "short": None,
            "long": None,
        }
        center_s = (float(short_spot[0]), float(short_spot[1]))
        if short_mat is not None:
            p_s = compute_horizontal_sum_profile(
                short_mat,
                center_s,
                roi_w=roi_w,
                roi_h=roi_h,
                smooth_win=smooth_win,
                min_peak_distance=min_sep,
                find_secondary=False,
            )
            out["short"] = self._serialize_hsum_profile(p_s)
            if out["short"] is not None and n_refine <= 0:
                # N=0：短峰改整點最高（與主畫面一致）
                ys = np.asarray(out["short"].get("y_px") or [], dtype=float)
                before = np.asarray(
                    out["short"].get("sum_before_smooth")
                    or out["short"].get("sum_minus_median")
                    or [],
                    dtype=float,
                )
                if ys.size and before.size == ys.size:
                    j = int(np.nanargmax(np.where(np.isfinite(before), before, -np.inf)))
                    p_int = float(ys[j])
                    out["short"]["primary_y_px"] = p_int
                    out["short"]["peak1_y_px"] = p_int
                    out["short"]["peak2_y_px"] = None
                    out["short"]["secondary_y_px"] = None
                    out["short"]["peak_delta_y_px"] = None
                    out["short"]["peak_delta_y_um"] = None

        if long_mat is not None:
            if long_center is not None:
                center_l = (float(long_center[0]), float(long_center[1]))
            else:
                h, w = np.asarray(long_mat).shape[:2]
                center_l = (
                    float(np.clip(center_s[0], 0.0, max(0.0, w - 1.0))),
                    float(np.clip(center_s[1], 0.0, max(0.0, h - 1.0))),
                )
            p_l = compute_horizontal_sum_profile(
                long_mat,
                center_l,
                roi_w=roi_w,
                roi_h=roi_h,
                smooth_win=smooth_win,
                min_peak_distance=min_sep,
                find_secondary=True,
            )
            block_l = self._serialize_hsum_profile(p_l)
            # 以 find_dual_spots_horizontal_sum 對齊主畫面精修峰位／曲線
            result = find_dual_spots_horizontal_sum(
                long_mat,
                center_l,
                roi_w=roi_w,
                roi_h=roi_h,
                smooth_win=smooth_win,
                min_peak_distance=min_sep,
                n_refine=n_refine,
                secondary_window_y=secondary_window_y,
            )
            if block_l is not None and result.get("primary_y") is not None:
                p1 = float(result["primary_y"])
                p2 = result.get("secondary_y")
                p2 = None if p2 is None else float(p2)
                block_l["primary_y_px"] = p1
                block_l["secondary_y_px"] = p2
                if p2 is not None:
                    lo_y, hi_y = (p1, p2) if p1 <= p2 else (p2, p1)
                    dy = float(hi_y - lo_y)
                    block_l["peak1_y_px"] = lo_y
                    block_l["peak2_y_px"] = hi_y
                    block_l["peak_delta_y_px"] = dy
                    block_l["peak_delta_y_um"] = dy * 3.45
                    block_l["primary_xy"] = [float(center_l[0]), p1]
                    block_l["secondary_xy"] = [float(center_l[0]), p2]
                else:
                    block_l["peak1_y_px"] = p1
                    block_l["peak2_y_px"] = None
                    block_l["peak_delta_y_px"] = None
                    block_l["peak_delta_y_um"] = None
                block_l["refine_iters"] = int(n_refine)
                block_l["refine_done"] = int(result.get("refine_done") or 0)
                ys_r = np.asarray(result.get("y_refine_px"), dtype=float)
                refined = np.asarray(result.get("profile_refine"), dtype=float)
                if n_refine > 0 and ys_r.size and refined.size == ys_r.size:
                    block_l["y_refine_px"] = [float(v) for v in ys_r]
                    block_l["sum_refine"] = [float(v) for v in refined]
                else:
                    block_l.pop("y_refine_px", None)
                    block_l.pop("sum_refine", None)
                win = result.get("secondary_window_y")
                if win is not None and len(win) >= 2:
                    out["secondary_search_window_y"] = [
                        float(win[0]),
                        float(win[1]),
                    ]
            out["long"] = block_l
        return out

    @staticmethod
    def _write_batch_hsum_all_csv(csv_path, hsum_collector):
        """批次全部水平加總：峰位摘要＋長表剖面（方便一次疊圖）。

        結構：
          1) PEAKS 區塊：每組 short/long 的峰1／峰2／間距
          2) 空行
          3) PROFILES 區塊：index,label,exposure,y_px,sum_before,sum_after,sum_raw
        """
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# PEAKS"])
            writer.writerow(
                [
                    "index",
                    "label",
                    "exposure",
                    "peak1_y_px",
                    "peak2_y_px",
                    "peak_delta_y_px",
                    "peak_delta_y_um",
                    "primary_y_px",
                    "secondary_y_px",
                    "smooth_win",
                    "n_points",
                ]
            )
            for item in hsum_collector:
                idx = item.get("index")
                label = item.get("label")
                hsum = item.get("hsum") or {}
                for which, key in (("short", "short"), ("long", "long")):
                    block = hsum.get(key)
                    if not block:
                        continue
                    writer.writerow(
                        [
                            idx,
                            label,
                            which,
                            block.get("peak1_y_px"),
                            block.get("peak2_y_px"),
                            block.get("peak_delta_y_px"),
                            block.get("peak_delta_y_um"),
                            block.get("primary_y_px"),
                            block.get("secondary_y_px"),
                            block.get("smooth_win"),
                            block.get("n_points"),
                        ]
                    )

            writer.writerow([])
            writer.writerow(["# PROFILES"])
            writer.writerow(
                [
                    "index",
                    "label",
                    "exposure",
                    "y_px",
                    "sum_before_smooth",
                    "sum_after_smooth",
                    "sum_raw",
                ]
            )
            for item in hsum_collector:
                idx = item.get("index")
                label = item.get("label")
                hsum = item.get("hsum") or {}
                for which, key in (("short", "short"), ("long", "long")):
                    block = hsum.get(key)
                    if not block:
                        continue
                    ys = block.get("y_px") or []
                    before = (
                        block.get("sum_before_smooth")
                        or block.get("sum_minus_median")
                        or []
                    )
                    after = (
                        block.get("sum_after_smooth")
                        or block.get("sum_smooth")
                        or []
                    )
                    raw = block.get("sum_raw") or []
                    n = min(len(ys), len(before), len(after), len(raw))
                    for i in range(n):
                        writer.writerow(
                            [
                                idx,
                                label,
                                which,
                                ys[i],
                                before[i],
                                after[i],
                                raw[i],
                            ]
                        )

    @staticmethod
    def _write_hsum_csv_sections(writer, hsum_block):
        """在 Spot_Data CSV 末尾追加短／長水平加總數值表。"""
        if not hsum_block:
            return
        for which, key in (("short", "short"), ("long", "long")):
            block = hsum_block.get(key)
            if not block:
                continue
            writer.writerow([])
            writer.writerow(
                [
                    f"Horizontal Sum Profile ({which})",
                    "",
                    "",
                ]
            )
            writer.writerow(
                [
                    "y_px",
                    "sum_raw",
                    "sum_before_smooth",
                    "sum_after_smooth",
                ]
            )
            ys = block.get("y_px") or []
            raw = block.get("sum_raw") or []
            bg = block.get("sum_before_smooth") or block.get("sum_minus_median") or []
            sm = block.get("sum_after_smooth") or block.get("sum_smooth") or []
            n = min(len(ys), len(raw), len(bg), len(sm))
            for i in range(n):
                writer.writerow([ys[i], raw[i], bg[i], sm[i]])

    @staticmethod
    def _export_hsum_peaks_files(base_nostamp, hsum_block, stamp, exported=None):
        """額外匯出水平加總雙峰 CSV（峰位／平滑前後／間距）。

        檔名：{base}_HSumPeaks_{Short|Long}_{stamp}.csv
        """
        if not hsum_block:
            return []
        out_paths = [] if exported is None else exported
        for which, key in (("Short", "short"), ("Long", "long")):
            block = hsum_block.get(key)
            if not block:
                continue
            csv_path = with_export_timestamp(
                f"{base_nostamp}_HSumPeaks_{which}.csv", stamp=stamp
            )
            BaslerBatchDualTab._write_hsum_peaks_csv(
                csv_path, block, which=which.lower()
            )
            out_paths.append(csv_path)
        return out_paths

    @staticmethod
    def _write_hsum_peaks_csv(csv_path, block, *, which="long"):
        """單一曝光：雙峰摘要 + 平滑前／後剖面。"""
        p1 = block.get("peak1_y_px")
        p2 = block.get("peak2_y_px")
        dy = block.get("peak_delta_y_px")
        dy_um = block.get("peak_delta_y_um")
        if dy is None and p1 is not None and p2 is not None:
            dy = abs(float(p2) - float(p1))
            dy_um = float(dy) * 3.45
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Item", "Value", "Unit"])
            writer.writerow(["Exposure", which, ""])
            writer.writerow(["Smooth Win", block.get("smooth_win"), ""])
            writer.writerow(["Refine Iters", block.get("refine_iters"), ""])
            writer.writerow(["Refine Done", block.get("refine_done"), ""])
            writer.writerow(["Peak1 Y (smaller)", p1, "px"])
            writer.writerow(["Peak2 Y (larger)", p2, "px"])
            writer.writerow(["Peak Delta Y", dy, "px"])
            writer.writerow(["Peak Delta Y", dy_um, "μm"])
            writer.writerow(
                ["Primary Y (algo)", block.get("primary_y_px"), "px"]
            )
            writer.writerow(
                ["Secondary Y (algo)", block.get("secondary_y_px"), "px"]
            )
            if block.get("primary_xy"):
                writer.writerow(["Primary XY", block.get("primary_xy"), "px"])
            if block.get("secondary_xy"):
                writer.writerow(
                    ["Secondary XY", block.get("secondary_xy"), "px"]
                )
            roi = block.get("roi_xywh")
            if roi and len(roi) == 4:
                writer.writerow(["Strip ROI X", roi[0], "px"])
                writer.writerow(["Strip ROI Y", roi[1], "px"])
                writer.writerow(["Strip ROI W", roi[2], "px"])
                writer.writerow(["Strip ROI H", roi[3], "px"])
            writer.writerow([])
            ys_r = block.get("y_refine_px") or []
            refined = block.get("sum_refine") or []
            has_refine = bool(ys_r) and len(ys_r) == len(refined) and len(ys_r) > 0
            if has_refine:
                writer.writerow(
                    [
                        "y_px",
                        "sum_before_smooth",
                        "sum_after_smooth",
                        "sum_raw",
                        "y_refine_px",
                        "sum_refine",
                    ]
                )
            else:
                writer.writerow(
                    [
                        "y_px",
                        "sum_before_smooth",
                        "sum_after_smooth",
                        "sum_raw",
                    ]
                )
            ys = block.get("y_px") or []
            before = (
                block.get("sum_before_smooth")
                or block.get("sum_minus_median")
                or []
            )
            after = block.get("sum_after_smooth") or block.get("sum_smooth") or []
            raw = block.get("sum_raw") or []
            n = min(len(ys), len(before), len(after), len(raw))
            n_ref = len(ys_r) if has_refine else 0
            n_rows = max(n, n_ref)
            for i in range(n_rows):
                row = [
                    ys[i] if i < n else "",
                    before[i] if i < n else "",
                    after[i] if i < n else "",
                    raw[i] if i < n else "",
                ]
                if has_refine:
                    row.extend(
                        [
                            ys_r[i] if i < n_ref else "",
                            refined[i] if i < n_ref else "",
                        ]
                    )
                writer.writerow(row)

    def show_horizontal_sum_profile(self, which="long"):
        """開啟短／長曝光的一維投影水平加總圖（僅 axial_1d）。"""
        if self._current_spot_algo() != "axial_1d":
            QMessageBox.information(
                self,
                "提示",
                "「查看水平加總圖」僅在演算法為「一維投影＋微分」時可用。",
            )
            return

        is_short = str(which).lower() == "short"
        label = "短曝光" if is_short else "長曝光"

        if is_short:
            matrix = self._short_analysis_matrix()
            if matrix is None:
                QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
                return
            if self.short_spot is None:
                self.grab_short_spot()
            if self.short_spot is None:
                return
            center = (float(self.short_spot[0]), float(self.short_spot[1]))
        else:
            matrix = self._long_analysis_matrix()
            if matrix is None:
                QMessageBox.warning(self, "警告", "請先載入長曝光檔案。")
                return
            if getattr(self, "batch_result_matrix", None) is None:
                QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
                return
            if self.short_spot is None:
                self.grab_short_spot()
            if self.short_spot is None:
                return
            center = self._map_short_spot_to_long()
            if center is None:
                QMessageBox.warning(self, "警告", "無法將短曝光光斑對應到長曝光。")
                return

        roi_w, roi_h, smooth_win, min_sep = self._axial_strip_params()
        profile = compute_horizontal_sum_profile(
            matrix,
            center,
            roi_w=roi_w,
            roi_h=roi_h,
            smooth_win=smooth_win,
            min_peak_distance=min_sep,
            find_secondary=not is_short,
        )
        if profile is None:
            QMessageBox.warning(
                self,
                "警告",
                f"無法計算{label}水平加總剖面（請確認光斑位置與條帶參數）。",
            )
            return
        self._open_hsum_profile_dialog(
            profile,
            label=label,
            center=center,
            smooth_win=smooth_win,
            min_sep=min_sep,
            which="short" if is_short else "long",
        )

    def show_horizontal_sum_profile_from_data(
        self,
        matrix,
        center_xy,
        *,
        label="水平加總",
        strip_params=None,
        parent=None,
        which=None,
    ):
        """以外部矩陣／中心開啟水平加總圖（供批次結果視窗呼叫）。"""
        if matrix is None or center_xy is None:
            QMessageBox.warning(parent or self, "警告", "缺少矩陣或光斑中心。")
            return
        if strip_params is None:
            roi_w, roi_h, smooth_win, min_sep = self._active_hsum_strip_params()
        else:
            roi_w, roi_h, smooth_win, min_sep = strip_params
        # 短曝光只找主峰；長曝光找雙峰
        key = which
        if key is None:
            key = "short" if "短" in str(label) else "long"
        find_sec = str(key).lower() != "short"
        profile = compute_horizontal_sum_profile(
            matrix,
            center_xy,
            roi_w=roi_w,
            roi_h=roi_h,
            smooth_win=smooth_win,
            min_peak_distance=min_sep,
            find_secondary=find_sec,
        )
        if profile is None:
            QMessageBox.warning(
                parent or self,
                "警告",
                f"無法計算{label}水平加總剖面。",
            )
            return
        self._open_hsum_profile_dialog(
            profile,
            label=label,
            center=center_xy,
            smooth_win=smooth_win,
            min_sep=min_sep,
            parent=parent,
            which=key,
        )

    def _open_hsum_profile_dialog(
        self,
        profile,
        *,
        label,
        center,
        smooth_win,
        min_sep,
        parent=None,
        which="long",
    ):
        y = np.asarray(profile["y_full_px"], dtype=float)
        sum_raw = np.asarray(profile["sum_raw"], dtype=float)
        sum_bg = np.asarray(profile["sum_minus_median"], dtype=float)
        sum_sm = np.asarray(profile["sum_smooth"], dtype=float)
        primary_y = profile.get("primary_y_full")
        secondary_y = profile.get("secondary_y_full")
        # 主峰固定為平滑前最高峰（與批次水平加總視窗一致）
        if y.size and sum_bg.size == y.size:
            finite = np.isfinite(sum_bg)
            if np.any(finite):
                primary_y = float(y[int(np.nanargmax(np.where(finite, sum_bg, -np.inf)))])
        if str(which).lower() == "short":
            secondary_y = None
        rx, ry, rw, rh = profile.get("roi_xywh", (0, 0, 0, 0))

        host = parent if parent is not None else self
        key = "short" if str(which).lower() == "short" else "long"
        if not hasattr(self, "_hsum_dialogs") or self._hsum_dialogs is None:
            self._hsum_dialogs = {}
        # 關閉同類型舊視窗，避免看起來像「沒更新」
        old = self._hsum_dialogs.get(key)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
            self._hsum_dialogs[key] = None

        dlg = QDialog(host)
        dlg.setWindowTitle(f"查看水平加總圖 · {label}")
        dlg.setWindowFlags(
            dlg.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        dlg.resize(860, 760)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        info = QLabel(
            f"{label}｜中心 ({float(center[0]):.1f}, {float(center[1]):.1f})｜"
            f"條帶 ROI x={rx}, y={ry}, w={rw}, h={rh}｜"
            f"平滑窗={smooth_win}｜最小峰距={min_sep}"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #455A64; font-size: 12px;")
        root.addWidget(info)

        plot = pg.PlotWidget()
        plot.setBackground("#FAFBFC")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "Y (pixels)")
        plot.setLabel("left", "Horizontal sum")
        plot.addLegend(offset=(10, 10))
        attach_curve_hover_readout(plot, pixel_name="Y")

        plot.plot(
            y,
            sum_raw,
            pen=pg.mkPen("#90A4AE", width=1.2),
            name="sum raw",
        )
        plot.plot(
            y,
            sum_bg,
            pen=pg.mkPen("#1E88E5", width=1.5),
            name="sum − median",
        )
        plot.plot(
            y,
            sum_sm,
            pen=pg.mkPen("#E65100", width=2.0),
            name=f"smooth (win={smooth_win})",
        )

        if primary_y is not None:
            plot.addItem(
                pg.InfiniteLine(
                    pos=float(primary_y),
                    angle=90,
                    pen=pg.mkPen("#2E7D32", width=2, style=Qt.DashLine),
                )
            )
        if secondary_y is not None:
            plot.addItem(
                pg.InfiniteLine(
                    pos=float(secondary_y),
                    angle=90,
                    pen=pg.mkPen("#C62828", width=2, style=Qt.DashLine),
                )
            )

        # 雙峰依 Y 排序＋間距（px／μm）
        peak_lo = peak_hi = None
        if primary_y is not None and secondary_y is not None:
            y1, y2 = float(primary_y), float(secondary_y)
            peak_lo, peak_hi = (y1, y2) if y1 <= y2 else (y2, y1)
        elif primary_y is not None:
            peak_lo = float(primary_y)
        elif secondary_y is not None:
            peak_lo = float(secondary_y)

        peak_bits = []
        if peak_lo is not None and peak_hi is not None:
            dy_px = float(peak_hi - peak_lo)
            dy_um = self._px_to_um(dy_px)
            peak_bits.append(f"峰1 Y={peak_lo:.1f}")
            peak_bits.append(f"峰2 Y={peak_hi:.1f}")
            peak_bits.append(f"雙峰距 ΔY={dy_px:.1f} px（{dy_um:.1f} μm）")
        elif peak_lo is not None:
            peak_bits.append(f"僅單一峰 Y={peak_lo:.1f}")
        if peak_bits:
            peak_lbl = QLabel("｜".join(peak_bits))
            peak_lbl.setWordWrap(True)
            peak_lbl.setStyleSheet(
                "color: #1B5E20; font-size: 13px; font-weight: bold; "
                "padding: 6px 8px; background: #E8F5E9; border-radius: 4px;"
            )
            root.addWidget(peak_lbl)

        root.addWidget(plot, 3)

        # 找峰用的就是這條平滑曲線的微分（win=1 時等於 sum−median）
        dy = (
            np.gradient(sum_sm)
            if sum_sm.size == y.size and y.size
            else np.asarray([], dtype=float)
        )
        plot_d = pg.PlotWidget()
        plot_d.setBackground("#FAFBFC")
        plot_d.showGrid(x=True, y=True, alpha=0.25)
        plot_d.setLabel("bottom", "Y (pixels)")
        plot_d.setLabel("left", "d(smooth) / dY")
        plot_d.addLegend(offset=(10, 10))
        plot_d.setXLink(plot)
        attach_curve_hover_readout(plot_d, pixel_name="Y")
        if dy.size == y.size:
            plot_d.plot(
                y,
                dy,
                pen=pg.mkPen("#6A1B9A", width=1.6),
                name=f"d(smooth win={smooth_win})/dY",
            )
            plot_d.addItem(
                pg.InfiniteLine(
                    pos=0.0,
                    angle=0,
                    pen=pg.mkPen("#90A4AE", width=1, style=Qt.DashLine),
                )
            )
        if primary_y is not None:
            plot_d.addItem(
                pg.InfiniteLine(
                    pos=float(primary_y),
                    angle=90,
                    pen=pg.mkPen("#2E7D32", width=2, style=Qt.DashLine),
                )
            )
        if secondary_y is not None:
            plot_d.addItem(
                pg.InfiniteLine(
                    pos=float(secondary_y),
                    angle=90,
                    pen=pg.mkPen("#C62828", width=2, style=Qt.DashLine),
                )
            )
        root.addWidget(plot_d, 2)

        # 波型圖：雙擊復原；軸外側輸入上下限
        scale_ctrl = attach_waveform_axis_scale(
            plot,
            dlg,
            x_label="Y (pixels)",
            y_label="Horizontal sum",
        )
        scale_d = attach_waveform_axis_scale(
            plot_d,
            dlg,
            x_label="Y (pixels)",
            y_label="d(smooth) / dY",
        )
        if y.size:
            x0, x1 = float(np.nanmin(y)), float(np.nanmax(y))
            vals = []
            for arr in (sum_raw, sum_bg, sum_sm):
                a = np.asarray(arr, dtype=float)
                if a.size:
                    vals.append(a)
            if vals:
                stacked = np.concatenate(vals)
                finite = stacked[np.isfinite(stacked)]
                if finite.size:
                    y0 = float(min(0.0, np.nanmin(finite)))
                    y1 = float(np.nanmax(finite))
                    if y1 <= y0:
                        y1 = y0 + 1.0
                    pad = 0.05 * (y1 - y0)
                    scale_ctrl.set_default_range(x0, x1, y0, y1 + pad)
                    scale_ctrl.reset_view()
            if dy.size:
                d_finite = dy[np.isfinite(dy)]
                if d_finite.size:
                    d0 = float(np.nanmin(d_finite))
                    d1 = float(np.nanmax(d_finite))
                    if d1 <= d0:
                        d1 = d0 + 1.0
                    dpad = 0.08 * (d1 - d0)
                    scale_d.set_default_range(x0, x1, d0 - dpad, d1 + dpad)
                    scale_d.reset_view()

        btn_close = QPushButton("關閉")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        def _on_destroyed(*_args, k=key):
            dialogs = getattr(self, "_hsum_dialogs", None) or {}
            if dialogs.get(k) is dlg:
                dialogs[k] = None

        dlg.destroyed.connect(_on_destroyed)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        QApplication.processEvents()
        self._hsum_dialogs[key] = dlg
        self._hsum_dialog = dlg

    def _hide_interactive_panel_chrome(self, panel):
        # 保留「隱藏剖面十字」與色階上下限標籤（可點擊設定 colorbar）
        for attr in (
            "lbl_tip",
            "btn_export",
        ):
            w = getattr(panel, attr, None)
            if w is not None:
                w.hide()
        for attr in ("lbl_level_max", "lbl_level_min"):
            w = getattr(panel, attr, None)
            if w is not None:
                w.show()
                w.setToolTip("點擊可輸入 colorbar 上下限")
        btn = getattr(panel, "btn_toggle_cross", None)
        if btn is not None:
            btn.show()
            btn.setText("隱藏剖面十字")
            btn.setToolTip("隱藏此熱圖的剖面十字與剖面位置線（青色）")

    def _heatmap_panels(self):
        panels = []
        for attr in (
            "short_heatmap_panel",
            "long_heatmap_panel",
            "collage_heatmap_panel",
        ):
            panel = getattr(self, attr, None)
            if panel is not None:
                panels.append(panel)
        return panels

    def _set_all_profile_cross_visible(self, visible):
        for panel in self._heatmap_panels():
            setter = getattr(panel, "set_profile_cross_visible", None)
            if callable(setter):
                setter(bool(visible))
            else:
                btn = getattr(panel, "btn_toggle_cross", None)
                if btn is not None:
                    btn.setChecked(not bool(visible))

    def _on_hide_profile_cross_toggled(self, checked):
        self._set_all_profile_cross_visible(not bool(checked))

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
        # 與左側總開關同步
        hide = getattr(self, "chk_hide_profile_cross", None)
        if hide is not None and hide.isChecked():
            panel.set_profile_cross_visible(False)
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
        self.chk_long_heatmap_log.toggled.connect(self._on_long_log_toggled)

        self.chk_long_bw_scale = QCheckBox("color scale")
        self.chk_long_bw_scale.setChecked(False)
        self.chk_long_bw_scale.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_long_bw_scale.toggled.connect(self._on_long_bw_scale_toggled)

        self.spin_long_black_below = NoWheelDoubleSpinBox()
        self.spin_long_black_below.setRange(0.0, 1e9)
        self.spin_long_black_below.setDecimals(1)
        self.spin_long_black_below.setValue(_DEFAULT_BLACK_BELOW)
        self.spin_long_black_below.setFixedWidth(72)
        self.spin_long_black_below.setEnabled(False)
        self.spin_long_black_below.valueChanged.connect(
            self._on_long_bw_threshold_changed
        )

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
        self.chk_batch_heatmap_log.toggled.connect(self._on_short_log_toggled)

        self.chk_basler_bw_scale = QCheckBox("color scale")
        self.chk_basler_bw_scale.setChecked(False)
        self.chk_basler_bw_scale.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_basler_bw_scale.toggled.connect(self._on_basler_bw_scale_toggled)

        self.spin_basler_black_below = NoWheelDoubleSpinBox()
        self.spin_basler_black_below.setRange(0.0, 1e9)
        self.spin_basler_black_below.setDecimals(1)
        self.spin_basler_black_below.setValue(_DEFAULT_BLACK_BELOW)
        self.spin_basler_black_below.setFixedWidth(72)
        self.spin_basler_black_below.setEnabled(False)
        self.spin_basler_black_below.valueChanged.connect(
            self._on_basler_bw_threshold_changed
        )

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

        def _make_pair(
            prefix, on_log, on_display, on_bw_toggle, on_bw_thresh, on_smooth_toggle
        ):
            chk_log = QCheckBox("log")
            chk_log.setChecked(False)
            chk_log.setStyleSheet("font-weight: bold; color: #37474F;")
            chk_log.toggled.connect(on_log)

            chk_bw = QCheckBox("color scale")
            chk_bw.setChecked(False)
            chk_bw.setStyleSheet("font-weight: bold; color: #37474F;")
            chk_bw.toggled.connect(on_bw_toggle)

            spin_black = NoWheelDoubleSpinBox()
            spin_black.setRange(0.0, 1e9)
            spin_black.setDecimals(1)
            spin_black.setValue(_DEFAULT_BLACK_BELOW)
            spin_black.setFixedWidth(72)
            spin_black.setEnabled(False)
            spin_black.valueChanged.connect(on_bw_thresh)

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
                "spin_black": spin_black,
                "spin_white": spin_white,
                "chk_smooth": chk_smooth,
                "spin_smooth": spin_smooth,
            }

        short = _make_pair(
            "short",
            self._on_collage_short_log_toggled,
            self._on_collage_display_option_changed,
            self._on_collage_short_bw_scale_toggled,
            self._on_collage_short_bw_threshold_changed,
            self._on_collage_short_smooth_toggled,
        )
        long = _make_pair(
            "long",
            self._on_collage_long_log_toggled,
            self._on_collage_display_option_changed,
            self._on_collage_long_bw_scale_toggled,
            self._on_collage_long_bw_threshold_changed,
            self._on_collage_long_smooth_toggled,
        )

        self.chk_collage_short_log = short["chk_log"]
        self.chk_collage_short_bw_scale = short["chk_bw"]
        self.spin_collage_short_black_below = short["spin_black"]
        self.spin_collage_short_white_above = short["spin_white"]
        self.chk_collage_short_smooth = short["chk_smooth"]
        self.spin_collage_short_smooth = short["spin_smooth"]

        self.chk_collage_long_log = long["chk_log"]
        self.chk_collage_long_bw_scale = long["chk_bw"]
        self.spin_collage_long_black_below = long["spin_black"]
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
            controls["spin_black"],
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
        """主視覺：非 hsum＝雙 heatmap；hsum＝波形焦點（Heatmap 改由查看開啟）。"""
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
            "_main_algo_view_stack",
            "_hsum_focus_column",
            "_heatmap_host_page",
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
        short_lbl.setStyleSheet(
            "font-weight: bold; color: #424242; font-size: 13px; "
            "padding: 4px 8px; background: #E3F2FD; border-radius: 4px;"
        )
        short_layout.addWidget(short_lbl)
        self.lbl_short_heat = short_lbl

        short_bar = self._make_display_bar(
            self.chk_batch_heatmap_log,
            self.chk_basler_bw_scale,
            self.spin_basler_black_below,
            self.spin_basler_white_above,
            self.chk_batch_heatmap_smooth,
            self.spin_batch_heatmap_smooth,
            "baslerShortDisplayBar",
            extra_widgets=(
                self._make_thresh_overlay_button("short"),
                self._make_horizontal_sum_button("short"),
            ),
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
        long_lbl.setStyleSheet(
            "font-weight: bold; color: #424242; font-size: 13px; "
            "padding: 4px 8px; background: #F3E5F5; border-radius: 4px;"
        )
        long_layout.addWidget(long_lbl)
        self.lbl_long_heat = long_lbl

        long_bar = self._make_display_bar(
            self.chk_long_heatmap_log,
            self.chk_long_bw_scale,
            self.spin_long_black_below,
            self.spin_long_white_above,
            self.chk_long_heatmap_smooth,
            self.spin_long_heatmap_smooth,
            "baslerLongDisplayBar",
            extra_widgets=(
                self._make_thresh_overlay_button("long"),
                self._make_horizontal_sum_button("long"),
            ),
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
        self._dual_heatmap_row = dual_row

        heatmap_host = QWidget()
        heatmap_host_lay = QVBoxLayout(heatmap_host)
        heatmap_host_lay.setContentsMargins(0, 0, 0, 0)
        heatmap_host_lay.setSpacing(0)
        heatmap_host_lay.addWidget(dual_row, 1)
        self._heatmap_host_page = heatmap_host
        self._heatmap_host_layout = heatmap_host_lay

        hsum_focus = self._build_hsum_focus_pane()

        stack = QStackedWidget()
        stack.addWidget(heatmap_host)  # 0：雙 heatmap
        stack.addWidget(hsum_focus)  # 1：水平加總波形
        self._main_algo_view_stack = stack

        insert_at = 0
        for i in range(host_layout.count()):
            item = host_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is not None and w is getattr(self, "win_batch_top", None):
                insert_at = i
                break
            if item is not None and item.layout() is not None and i == 0:
                insert_at = 1
        host_layout.insertWidget(insert_at, stack, 1)

        self._dual_stack_column = stack
        self._basler_heat_column = stack
        self._short_heat_column = short_wrap
        self._long_heat_column = long_wrap
        self._hsum_focus_column = hsum_focus
        self._collage_heat_column = None

    def _build_hsum_focus_pane(self):
        """水平加總主畫面：上方參數列 + log/color scale + 短／長波形。"""
        wrap = QWidget()
        root = QVBoxLayout(wrap)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)

        self.spin_hsum_smooth = NoWheelSpinBox()
        self.spin_hsum_smooth.setRange(1, 201)
        self.spin_hsum_smooth.setSingleStep(2)
        self.spin_hsum_smooth.setValue(7)
        self.spin_hsum_smooth.setFixedWidth(64)
        self.spin_hsum_smooth.setToolTip(
            "剖面平滑窗：只影響水平加總波形與次峰抓取"
        )

        self.spin_hsum_refine = NoWheelSpinBox()
        self.spin_hsum_refine.setRange(0, 20)
        self.spin_hsum_refine.setValue(1)
        self.spin_hsum_refine.setFixedWidth(56)
        self.spin_hsum_refine.setToolTip(
            "精修 N：0＝整點峰位；N≥1＝分段二次加密波形，"
            "虛線對齊 after refine 曲線的強度最大點"
        )

        self.spin_hsum_sec_lo = NoWheelDoubleSpinBox()
        self.spin_hsum_sec_hi = NoWheelDoubleSpinBox()
        for spin in (self.spin_hsum_sec_lo, self.spin_hsum_sec_hi):
            spin.setRange(0.0, 1e6)
            spin.setDecimals(1)
            spin.setFixedWidth(72)
        self.spin_hsum_sec_lo.setValue(0.0)
        self.spin_hsum_sec_hi.setValue(0.0)
        self.spin_hsum_sec_lo.setToolTip(
            "次峰搜尋窗下限（Y px）；上下皆 0＝Apply 時自動"
        )
        self.spin_hsum_sec_hi.setToolTip(
            "次峰搜尋窗上限（Y px）；上下皆 0＝Apply 時自動"
        )

        bar.addWidget(QLabel("平滑窗"))
        bar.addWidget(self.spin_hsum_smooth)
        bar.addWidget(QLabel("精修"))
        bar.addWidget(self.spin_hsum_refine)
        bar.addWidget(QLabel("次峰窗 Y"))
        bar.addWidget(self.spin_hsum_sec_lo)
        bar.addWidget(QLabel("–"))
        bar.addWidget(self.spin_hsum_sec_hi)

        self.btn_hsum_view_heatmap = QPushButton("查看 Heatmap")
        self.btn_hsum_view_heatmap.setCursor(Qt.PointingHandCursor)
        self.btn_hsum_view_heatmap.setStyleSheet(_STYLE_BTN)
        self.btn_hsum_view_heatmap.setToolTip(
            "開啟短／長 Heatmap（保留 log／color scale、ROI、上光斑；不下光斑）"
        )
        self.btn_hsum_view_heatmap.clicked.connect(self._open_hsum_heatmap_view)
        bar.addWidget(self.btn_hsum_view_heatmap)
        bar.addStretch(1)
        root.addLayout(bar)

        self.lbl_hsum_focus_peaks = QLabel("峰位：--（Apply 後顯示）")
        self.lbl_hsum_focus_peaks.setWordWrap(True)
        self.lbl_hsum_focus_peaks.setStyleSheet(
            "color: #1B5E20; font-size: 12px; font-weight: bold; "
            "padding: 4px 8px; background: #E8F5E9; border-radius: 4px;"
        )
        root.addWidget(self.lbl_hsum_focus_peaks)

        self._hsum_focus_scale = {}
        self._hsum_focus_curve_items = {"short": [], "long": []}
        self._hsum_refine_regions = {}
        self.plot_hsum_short, self.line_hsum_s1, self.line_hsum_s2 = (
            self._make_hsum_focus_plot("短曝光水平加總", dual_peaks=False)
        )
        self.plot_hsum_long, self.line_hsum_l1, self.line_hsum_l2 = (
            self._make_hsum_focus_plot("長曝光水平加總", dual_peaks=True)
        )
        self._hsum_focus_scale["short"] = attach_waveform_axis_scale(
            self.plot_hsum_short,
            self,
            x_label="Y (pixels)",
            y_label="Horizontal sum",
        )
        self._hsum_focus_scale["long"] = attach_waveform_axis_scale(
            self.plot_hsum_long,
            self,
            x_label="Y (pixels)",
            y_label="Horizontal sum",
        )

        self.region_hsum_sec = pg.LinearRegionItem(
            values=(0, 1),
            orientation="vertical",
            brush=pg.mkBrush(255, 193, 7, 40),
            pen=pg.mkPen("#FF8F00", width=1.5),
            movable=False,
        )
        self.region_hsum_sec.setZValue(-10)
        self.plot_hsum_long.addItem(self.region_hsum_sec)
        self.region_hsum_sec.hide()

        short_block = QWidget()
        short_block_lay = QVBoxLayout(short_block)
        short_block_lay.setContentsMargins(0, 0, 0, 0)
        short_block_lay.setSpacing(4)
        short_block_lay.addWidget(
            self._make_hsum_focus_display_bar("short"),
        )
        short_block_lay.addWidget(self.plot_hsum_short, 1)

        long_block = QWidget()
        long_block_lay = QVBoxLayout(long_block)
        long_block_lay.setContentsMargins(0, 0, 0, 0)
        long_block_lay.setSpacing(4)
        long_block_lay.addWidget(
            self._make_hsum_focus_display_bar("long"),
        )
        long_block_lay.addWidget(self.plot_hsum_long, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(short_block)
        split.addWidget(long_block)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)
        return wrap

    def _make_hsum_focus_display_bar(self, which):
        """波形主畫面專用 log／color scale（與 heatmap 控件雙向同步）。"""
        is_short = which == "short"
        log_chk = QCheckBox("log")
        log_chk.setStyleSheet("font-weight: bold; color: #37474F;")
        bw_chk = QCheckBox("color scale")
        bw_chk.setStyleSheet("font-weight: bold; color: #37474F;")
        black = NoWheelDoubleSpinBox()
        white = NoWheelDoubleSpinBox()
        for spin in (black, white):
            spin.setRange(0.0, 1e9)
            spin.setDecimals(1)
            spin.setFixedWidth(72)
        black.setValue(
            float(self.spin_basler_black_below.value())
            if is_short
            else float(self.spin_long_black_below.value())
        )
        white.setValue(
            float(self.spin_basler_white_above.value())
            if is_short
            else float(self.spin_long_white_above.value())
        )
        src_log = (
            self.chk_batch_heatmap_log if is_short else self.chk_long_heatmap_log
        )
        src_bw = self.chk_basler_bw_scale if is_short else self.chk_long_bw_scale
        log_chk.setChecked(bool(src_log.isChecked()))
        bw_chk.setChecked(bool(src_bw.isChecked()))
        black.setEnabled(bool(src_bw.isChecked()))
        white.setEnabled(bool(src_bw.isChecked()))

        if is_short:
            self.chk_hsum_short_log = log_chk
            self.chk_hsum_short_bw = bw_chk
            self.spin_hsum_short_black = black
            self.spin_hsum_short_white = white
            log_chk.toggled.connect(self._on_hsum_short_log_toggled)
            bw_chk.toggled.connect(self._on_hsum_short_bw_toggled)
            black.valueChanged.connect(self._on_hsum_short_bw_value_changed)
            white.valueChanged.connect(self._on_hsum_short_bw_value_changed)
        else:
            self.chk_hsum_long_log = log_chk
            self.chk_hsum_long_bw = bw_chk
            self.spin_hsum_long_black = black
            self.spin_hsum_long_white = white
            log_chk.toggled.connect(self._on_hsum_long_log_toggled)
            bw_chk.toggled.connect(self._on_hsum_long_bw_toggled)
            black.valueChanged.connect(self._on_hsum_long_bw_value_changed)
            white.valueChanged.connect(self._on_hsum_long_bw_value_changed)

        tag = QLabel("短曝光" if is_short else "長曝光")
        tag.setStyleSheet("color: #546E7A; font-weight: bold;")
        host = QWidget()
        host.setObjectName(f"hsumFocus{which.title()}Disp")
        row = QHBoxLayout(host)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(10)
        row.addWidget(tag)
        row.addWidget(log_chk)
        row.addWidget(bw_chk)
        row.addWidget(black)
        row.addWidget(white)
        row.addStretch(1)
        host.setStyleSheet(
            f"#hsumFocus{which.title()}Disp {{"
            " background: #F5F7FA; border: 1px solid #CFD8DC;"
            " border-radius: 6px; }"
            f"#hsumFocus{which.title()}Disp QCheckBox {{"
            " color: #37474F; font-weight: bold; }"
        )
        return host

    def _sync_hsum_log_to_plot(self, which, checked):
        plot = self.plot_hsum_short if which == "short" else self.plot_hsum_long
        if plot is None:
            return
        try:
            plot.setLogMode(x=False, y=bool(checked))
        except Exception:
            pass

    def _on_hsum_short_log_toggled(self, checked=False):
        self._sync_checkbox(self.chk_batch_heatmap_log, checked)
        self._refresh_short_heatmap_display(reset_levels=True)
        self._sync_hsum_log_to_plot("short", checked)
        self._update_hsum_focus_plots()

    def _on_hsum_long_log_toggled(self, checked=False):
        self._sync_checkbox(self.chk_long_heatmap_log, checked)
        self._refresh_long_heatmap_display(reset_levels=True)
        self._sync_hsum_log_to_plot("long", checked)
        self._update_hsum_focus_plots()

    def _on_hsum_short_bw_toggled(self, checked=False):
        self._sync_checkbox(self.chk_basler_bw_scale, checked)
        for spin in (
            self.spin_basler_black_below,
            self.spin_basler_white_above,
            self.spin_hsum_short_black,
            self.spin_hsum_short_white,
        ):
            spin.setEnabled(bool(checked))
        self._refresh_short_heatmap_display(reset_levels=True)
        self._update_hsum_focus_plots(recompute_peaks=True)

    def _on_hsum_long_bw_toggled(self, checked=False):
        self._sync_checkbox(self.chk_long_bw_scale, checked)
        for spin in (
            self.spin_long_black_below,
            self.spin_long_white_above,
            self.spin_hsum_long_black,
            self.spin_hsum_long_white,
        ):
            spin.setEnabled(bool(checked))
        self._refresh_long_heatmap_display(reset_levels=True)
        self._update_hsum_focus_plots(recompute_peaks=True)

    def _on_hsum_short_bw_value_changed(self, *_args):
        self._sync_spin(self.spin_basler_black_below, self.spin_hsum_short_black.value())
        self._sync_spin(self.spin_basler_white_above, self.spin_hsum_short_white.value())
        if self._basler_bw_scale_enabled():
            self._refresh_short_heatmap_display(reset_levels=True)
            self._update_hsum_focus_plots(recompute_peaks=True)

    def _on_hsum_long_bw_value_changed(self, *_args):
        self._sync_spin(self.spin_long_black_below, self.spin_hsum_long_black.value())
        self._sync_spin(self.spin_long_white_above, self.spin_hsum_long_white.value())
        if self._long_bw_scale_enabled():
            self._refresh_long_heatmap_display(reset_levels=True)
            self._update_hsum_focus_plots(recompute_peaks=True)

    @staticmethod
    def _sync_checkbox(chk, checked):
        if chk is None:
            return
        if bool(chk.isChecked()) == bool(checked):
            return
        chk.blockSignals(True)
        chk.setChecked(bool(checked))
        chk.blockSignals(False)

    @staticmethod
    def _sync_spin(spin, value):
        if spin is None:
            return
        try:
            if abs(float(spin.value()) - float(value)) < 1e-9:
                return
        except Exception:
            return
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def _make_hsum_focus_plot(self, title, *, dual_peaks=True):
        plot = pg.PlotWidget(title=title)
        plot.setBackground("#FAFBFC")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "Y (pixels)")
        plot.setLabel("left", "Horizontal sum")
        plot.addLegend(offset=(10, 10))
        line1 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#2E7D32", width=2, style=Qt.DashLine),
        )
        line2 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#C62828", width=2, style=Qt.DashLine),
        )
        plot.addItem(line1)
        plot.addItem(line2)
        if not dual_peaks:
            line2.hide()
        attach_curve_hover_readout(plot, pixel_name="Y")
        return plot, line1, line2

    def _wire_hsum_focus_param_signals(self):
        for spin in (
            getattr(self, "spin_hsum_smooth", None),
            getattr(self, "spin_hsum_refine", None),
            getattr(self, "spin_hsum_sec_lo", None),
            getattr(self, "spin_hsum_sec_hi", None),
            getattr(self, "spin_hsum_roi_w", None),
            getattr(self, "spin_hsum_roi_h", None),
            getattr(self, "spin_hsum_min_sep", None),
        ):
            if spin is None:
                continue
            try:
                spin.valueChanged.disconnect(self._on_hsum_focus_params_changed)
            except Exception:
                pass
            spin.valueChanged.connect(self._on_hsum_focus_params_changed)

    def _sync_main_view_for_algo(self):
        stack = getattr(self, "_main_algo_view_stack", None)
        if stack is None:
            return
        is_hsum = self._current_spot_algo() == "hsum"
        if not is_hsum:
            dlg = getattr(self, "_hsum_heatmap_dialog", None)
            if dlg is not None and dlg.isVisible():
                dlg.close()
        stack.setCurrentIndex(1 if is_hsum else 0)
        if is_hsum:
            self._update_hsum_focus_plots()
        self._update_horizontal_sum_buttons_visibility()

    def _on_hsum_focus_params_changed(self, *_args):
        if self._current_spot_algo() != "hsum":
            return
        self._update_hsum_focus_plots(recompute_peaks=True)

    def _clear_hsum_focus_curves(self, which):
        plot = self.plot_hsum_short if which == "short" else self.plot_hsum_long
        items = self._hsum_focus_curve_items.get(which) or []
        for it in items:
            try:
                plot.removeItem(it)
            except Exception:
                pass
        self._hsum_focus_curve_items[which] = []
        legend = plot.plotItem.legend
        if legend is not None:
            legend.clear()

    def _update_hsum_focus_plots(self, *, recompute_peaks=False):
        """依目前矩陣與參數重畫主畫面短／長水平加總波形。"""
        if getattr(self, "plot_hsum_short", None) is None:
            return
        if self._current_spot_algo() != "hsum":
            return

        short_mat = self._short_analysis_matrix()
        long_mat = self._long_analysis_matrix()
        hkw = self._hsum_algo_kwargs()
        roi_w = hkw["roi_w"]
        roi_h = hkw["roi_h"]
        smooth_win = hkw["smooth_win"]
        min_sep = hkw["min_peak_distance"]
        n_ref = hkw["n_refine"]
        sec_win = hkw.get("secondary_window_y")

        p_short = None
        p_long = None
        if short_mat is not None and self.short_spot is not None:
            p_short = compute_horizontal_sum_profile(
                short_mat,
                (float(self.short_spot[0]), float(self.short_spot[1])),
                roi_w=roi_w,
                roi_h=roi_h,
                smooth_win=smooth_win,
                min_peak_distance=min_sep,
                find_secondary=False,
            )
        center_l = self._map_short_spot_to_long()
        if long_mat is not None and center_l is not None:
            p_long = compute_horizontal_sum_profile(
                long_mat,
                center_l,
                roi_w=roi_w,
                roi_h=roi_h,
                smooth_win=smooth_win,
                min_peak_distance=min_sep,
                find_secondary=True,
            )

        peak_s = None
        peak_l1 = None
        peak_l2 = None
        if recompute_peaks and long_mat is not None and center_l is not None:
            result = find_dual_spots_horizontal_sum(
                long_mat,
                center_l,
                **hkw,
            )
            peaks = result.get("peak_ys")
            if peaks is not None and len(peaks) >= 2:
                peak_l1, peak_l2 = float(peaks[0]), float(peaks[1])
                # peak_ys 已依 Y 排序；主／次以 primary_index 區分顯示色
                pri = result.get("primary_y")
                sec = result.get("secondary_y")
                if pri is not None:
                    peak_l1 = float(pri)
                if sec is not None:
                    peak_l2 = float(sec)
                win = result.get("secondary_window_y")
                if win is not None and len(win) >= 2:
                    self._hsum_sec_window = (float(win[0]), float(win[1]))
                    if sec_win is None:
                        for spin, val in (
                            (self.spin_hsum_sec_lo, float(win[0])),
                            (self.spin_hsum_sec_hi, float(win[1])),
                        ):
                            spin.blockSignals(True)
                            if val > spin.maximum():
                                spin.setMaximum(max(val * 1.2, spin.maximum()))
                            spin.setValue(val)
                            spin.blockSignals(False)
                if result.get("roi_xywh") is not None:
                    self._set_hsum_display_roi(result["roi_xywh"])
                spots = result.get("spots")
                if spots and len(spots) >= 2:
                    primary_i = int(result.get("primary_index", 0))
                    spot2 = spots[1 - primary_i]
                    self.long_spot2 = (float(spot2[0]), float(spot2[1]))
                    if hasattr(self, "lbl_long_spot_pos"):
                        self.lbl_long_spot_pos.setText(
                            f"長曝光第二光斑: X={self.long_spot2[0]:.2f}, "
                            f"Y={self.long_spot2[1]:.2f}（水平加總）"
                        )
                    self._update_spot_delta_label()
                    self._redraw_spot_overlays()
            elif self.long_spot2 is not None:
                peak_l2 = float(self.long_spot2[1])
                mapped = center_l
                if mapped is not None:
                    peak_l1 = float(mapped[1])
        else:
            if p_short is not None:
                peak_s = p_short.get("primary_y_full")
            if self.long_spot2 is not None:
                peak_l2 = float(self.long_spot2[1])
            mapped = center_l
            if mapped is not None:
                peak_l1 = float(mapped[1])
            if p_long is not None:
                if peak_l1 is None:
                    peak_l1 = p_long.get("primary_y_full")
                if peak_l2 is None:
                    peak_l2 = p_long.get("secondary_y_full")
            # Apply 後若有結果峰，仍可用 find 一次對齊精修
            if (
                long_mat is not None
                and center_l is not None
                and self.long_spot2 is not None
            ):
                result = find_dual_spots_horizontal_sum(
                    long_mat, center_l, **hkw
                )
                pri = result.get("primary_y")
                sec = result.get("secondary_y")
                if pri is not None:
                    peak_l1 = float(pri)
                if sec is not None:
                    peak_l2 = float(sec)

        if p_short is not None and peak_s is None:
            peak_s = p_short.get("primary_y_full")

        self._draw_hsum_focus_profile(
            "short", p_short, peak_s, None, n_refine=n_ref
        )
        self._draw_hsum_focus_profile(
            "long", p_long, peak_l1, peak_l2, n_refine=n_ref
        )

        lo = hi = None
        if getattr(self, "_hsum_sec_window", None) is not None:
            lo, hi = self._hsum_sec_window
        elif sec_win is not None:
            lo, hi = sec_win
        elif (
            self.spin_hsum_sec_lo.value() > 0 or self.spin_hsum_sec_hi.value() > 0
        ):
            lo = float(self.spin_hsum_sec_lo.value())
            hi = float(self.spin_hsum_sec_hi.value())
        if lo is not None and hi is not None and hi > lo:
            self.region_hsum_sec.setRegion((lo, hi))
            self.region_hsum_sec.show()
        else:
            self.region_hsum_sec.hide()

        if peak_l1 is not None and peak_l2 is not None:
            dy = abs(float(peak_l2) - float(peak_l1))
            self.lbl_hsum_focus_peaks.setText(
                f"平滑={smooth_win}｜精修={n_ref}｜"
                f"主峰 Y={float(peak_l1):.3f}｜次峰 Y={float(peak_l2):.3f}｜"
                f"ΔY={dy:.3f} px（{self._px_to_um(dy):.3f} μm）"
            )
        elif peak_s is not None:
            self.lbl_hsum_focus_peaks.setText(
                f"平滑={smooth_win}｜短峰 Y={float(peak_s):.3f}｜長峰：--"
            )
        else:
            self.lbl_hsum_focus_peaks.setText(
                "峰位：--（請先 Apply 水平加總）"
            )

    def _draw_hsum_focus_profile(
        self, which, profile, peak1, peak2, *, n_refine=0
    ):
        plot = self.plot_hsum_short if which == "short" else self.plot_hsum_long
        line1 = self.line_hsum_s1 if which == "short" else self.line_hsum_l1
        line2 = self.line_hsum_s2 if which == "short" else self.line_hsum_l2
        self._clear_hsum_focus_curves(which)
        if not profile:
            line1.hide()
            if which == "long":
                line2.hide()
            return
        y = np.asarray(profile.get("y_full_px"), dtype=float)
        # raw＝平滑前剖面（扣列中位數後）；after smooth 由此平滑而來
        # 若誤畫 sum_raw，會因未扣背景而與 win=1 的 after smooth 看起來不同
        raw = np.asarray(profile.get("sum_minus_median"), dtype=float)
        before = raw
        after = np.asarray(profile.get("sum_smooth"), dtype=float)
        if y.size == 0 or raw.size != y.size:
            return
        win = int(profile.get("smooth_win") or self.spin_hsum_smooth.value())
        items = []
        items.append(
            plot.plot(
                y, raw, pen=pg.mkPen("#1E88E5", width=1.4), name="raw"
            )
        )
        if after.size == y.size:
            items.append(
                plot.plot(
                    y,
                    after,
                    pen=pg.mkPen("#E65100", width=2.0),
                    name=f"after smooth (win={win})",
                )
            )
        n_ref = max(0, int(n_refine))
        refined = None
        ys_r = None
        if n_ref > 0:
            src = after if after.size == y.size else before
            if src.size == y.size:
                pack = refine_profile_together(y, src, n_ref, base_win=win)
                ys_r = np.asarray(pack.get("y_px"), dtype=float)
                refined = np.asarray(pack.get("sum_refine"), dtype=float)
                if ys_r.size and refined.size == ys_r.size:
                    items.append(
                        plot.plot(
                            ys_r,
                            refined,
                            pen=pg.mkPen("#6A1B9A", width=2.2),
                            name=f"after refine (quad×{n_ref})",
                        )
                    )
        self._hsum_focus_curve_items[which] = items
        if peak1 is not None:
            line1.setValue(float(peak1))
            line1.show()
        else:
            line1.hide()
        if which == "long":
            if peak2 is not None:
                line2.setValue(float(peak2))
                line2.show()
            else:
                line2.hide()
        # 預設範圍
        series = [raw]
        if after.size == y.size:
            series.append(after)
        if refined is not None:
            series.append(refined)
        finite_parts = [s[np.isfinite(s)] for s in series if s.size]
        finite = (
            np.concatenate(finite_parts) if finite_parts else np.array([])
        )
        if finite.size and y.size:
            x0, x1 = float(np.nanmin(y)), float(np.nanmax(y))
            y0 = float(min(0.0, np.nanmin(finite)))
            y1 = float(np.nanmax(finite))
            if y1 <= y0:
                y1 = y0 + 1.0
            pad = (y1 - y0) * 0.05
            ctrl = self._hsum_focus_scale.get(which)
            if ctrl is not None:
                ctrl.set_default_range(x0, x1, y0, y1 + pad)
            log_on = False
            if which == "short":
                log_on = bool(
                    getattr(self, "chk_hsum_short_log", None)
                    and self.chk_hsum_short_log.isChecked()
                )
            else:
                log_on = bool(
                    getattr(self, "chk_hsum_long_log", None)
                    and self.chk_hsum_long_log.isChecked()
                )
            self._sync_hsum_log_to_plot(which, log_on)
            if log_on:
                y0 = max(y0, 1e-3)
            # 使用者已設軸上下限時，調參數只更新曲線，不復原視窗
            if ctrl is None or not ctrl.is_user_locked():
                plot.setXRange(x0, x1, padding=0.02)
                plot.setYRange(y0, y1 + pad, padding=0)

    def _open_hsum_heatmap_view(self):
        """水平加總：將雙 Heatmap 放到查看視窗（log／color scale 保留）。"""
        dual = getattr(self, "_dual_heatmap_row", None)
        if dual is None:
            QMessageBox.information(self, "提示", "Heatmap 尚未建立。")
            return

        dlg = getattr(self, "_hsum_heatmap_dialog", None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("查看 Heatmap（水平加總）")
            dlg.setWindowFlags(
                dlg.windowFlags()
                | Qt.Window
                | Qt.WindowMinimizeButtonHint
                | Qt.WindowMaximizeButtonHint
            )
            dlg.resize(1100, 720)
            dlg.setAttribute(Qt.WA_DeleteOnClose, False)
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            tip = QLabel(
                "標示：條帶 ROI、上光斑（短→長對應）。不下光斑十字。"
                "Log／Color scale 在各圖上方列。"
            )
            tip.setWordWrap(True)
            tip.setStyleSheet(_STYLE_MUTED)
            lay.addWidget(tip)
            self._hsum_heatmap_dialog_layout = lay
            dlg.finished.connect(self._on_hsum_heatmap_dialog_closed)
            self._hsum_heatmap_dialog = dlg

        host_lay = getattr(self, "_heatmap_host_layout", None)
        dlg_lay = self._hsum_heatmap_dialog_layout
        if dual.parentWidget() is not dlg:
            if host_lay is not None:
                host_lay.removeWidget(dual)
            dlg_lay.addWidget(dual, 1)
            dual.show()
        self._redraw_spot_overlays()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_hsum_heatmap_dialog_closed(self, *_args):
        dual = getattr(self, "_dual_heatmap_row", None)
        host_lay = getattr(self, "_heatmap_host_layout", None)
        dlg_lay = getattr(self, "_hsum_heatmap_dialog_layout", None)
        if dual is None or host_lay is None:
            return
        if dlg_lay is not None:
            dlg_lay.removeWidget(dual)
        host_lay.addWidget(dual, 1)
        dual.show()

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
        collage_lbl.setStyleSheet(
            "font-weight: bold; color: #424242; font-size: 13px; "
            "padding: 4px 8px; background: #E0F2F1; border-radius: 4px;"
        )
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
                "spin_black": self.spin_collage_short_black_below,
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
                "spin_black": self.spin_collage_long_black_below,
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

    def _build_data_pair_controls(self):
        left = getattr(self, "batch_left_settings_layout", None)
        if left is None:
            return

        box = QWidget()
        box.setObjectName("baslerDualDataBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        box.setStyleSheet(_STYLE_PANEL)

        title = QLabel("資料")
        title.setStyleSheet(_STYLE_SECTION)

        hint = QLabel("資料夾內奇數編號＝短曝光、偶數＝長曝光；清單為配對列。")
        hint.setStyleSheet(_STYLE_MUTED)
        hint.setWordWrap(True)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.btn_rescan_pairs = QPushButton("重新掃描")
        self.btn_rescan_pairs.setStyleSheet(_STYLE_BTN)
        self.btn_rescan_pairs.clicked.connect(self.rescan_data_folder)
        self.lbl_data_folder = QLabel("資料夾: --")
        self.lbl_data_folder.setStyleSheet(_STYLE_MUTED)
        self.lbl_data_folder.setWordWrap(True)
        row.addWidget(self.btn_rescan_pairs)
        row.addWidget(self.lbl_data_folder, 1)

        self.list_data_pairs = QListWidget()
        self.list_data_pairs.setMinimumHeight(120)
        self.list_data_pairs.setMaximumHeight(180)
        self.list_data_pairs.setStyleSheet(
            "QListWidget { background: #FFFFFF; border: 1px solid #E0E0E0; "
            "color: #212121; }"
            "QListWidget::item:selected { background: #ECEFF1; color: #212121; }"
        )
        self.list_data_pairs.currentRowChanged.connect(self._on_data_pair_selected)

        self.lbl_pair_detail = QLabel("尚未選擇配對")
        self.lbl_pair_detail.setStyleSheet(_STYLE_MUTED)
        self.lbl_pair_detail.setWordWrap(True)

        lay.addWidget(title)
        lay.addWidget(hint)
        lay.addLayout(row)
        lay.addWidget(self.list_data_pairs)
        lay.addWidget(self.lbl_pair_detail)

        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left.addWidget(box)
        self._data_pair_box = box

    def select_data_folder(self):
        start = self.data_folder or _DEFAULT_DATA_DIR or self.save_dir_path or ""
        folder = QFileDialog.getExistingDirectory(self, "選擇短／長曝光資料夾", start)
        if not folder:
            return
        self._load_data_folder(folder)

    def rescan_data_folder(self):
        folder = self.data_folder or _DEFAULT_DATA_DIR
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "警告", "請先選擇資料夾。")
            return
        self._load_data_folder(folder)

    def _load_data_folder(self, folder):
        self.data_folder = folder
        # 不連動儲存路徑；儲存資料夾需另外「選擇儲存資料夾」
        if getattr(self, "lbl_data_folder", None) is not None:
            self.lbl_data_folder.setText(f"資料夾: {folder}")

        self._data_pairs = scan_short_long_pairs(folder)
        self._refresh_data_pair_list()
        n = len(self._data_pairs)
        self.lbl_batch_status.setText(
            f"狀態: 已掃描 {n} 組短／長曝光配對（資料夾內檔案成對列）"
        )
        self.lbl_batch_pair_info.setText(
            f"資料夾配對: {n} 組" if n else "資料夾內找不到可配對檔案"
        )
        if n:
            self.list_data_pairs.setCurrentRow(0)
        else:
            self.lbl_pair_detail.setText("找不到配對（需有 0001＋0002、0003＋0004…）")

    def _refresh_data_pair_list(self):
        lst = getattr(self, "list_data_pairs", None)
        if lst is None:
            return
        lst.blockSignals(True)
        lst.clear()
        for i, pair in enumerate(self._data_pairs):
            item = QListWidgetItem(f"{i + 1}. {pair['label']}")
            item.setData(Qt.UserRole, i)
            item.setToolTip(
                f"短: {os.path.basename(pair['short_path'])}\n"
                f"長: {os.path.basename(pair['long_path'])}"
            )
            lst.addItem(item)
        lst.blockSignals(False)

    def _on_data_pair_selected(self, row):
        if row < 0 or row >= len(self._data_pairs):
            return
        pair = self._data_pairs[row]
        self.lbl_pair_detail.setText(
            f"短 {pair['short_idx']:04d}: {os.path.basename(pair['short_path'])}\n"
            f"長 {pair['long_idx']:04d}: {os.path.basename(pair['long_path'])}"
        )
        self._load_pair_paths(pair["short_path"], pair["long_path"])

    def _load_pair_paths(self, short_path, long_path):
        self.batch_short_path = short_path
        self.batch_long_path = long_path
        self.batch_m2_root = os.path.dirname(short_path)
        self.batch_m1_root = ""
        self._update_pair_info()
        self.clear_spots()
        self._try_load_both_files()

    def _build_spot_controls(self):
        left = getattr(self, "batch_left_settings_layout", None)
        if left is None:
            return

        box = QWidget()
        box.setObjectName("baslerDualSpotBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignTop)
        box.setStyleSheet(_STYLE_PANEL)

        title = QLabel("分析")
        title.setStyleSheet(_STYLE_SECTION)

        algo_lbl = QLabel("演算法")
        algo_lbl.setStyleSheet(_STYLE_MUTED)

        self.combo_spot_algo = QComboBox()
        self.combo_spot_algo.setMinimumHeight(32)
        self.combo_spot_algo.setCursor(Qt.PointingHandCursor)
        self.combo_spot_algo.setStyleSheet(
            "QComboBox { font-size: 12px; color: #212121; background: #FAFAFA;"
            " border: 1px solid #BDBDBD; border-radius: 4px; padding: 4px 8px; }"
            "QComboBox:hover { background: #EEEEEE; }"
            "QComboBox::drop-down { border: none; width: 22px; }"
        )
        # index 對齊 stack_algo_params：0 質心／1 一維／2 水平加總／3 高斯
        self._spot_algo_items = (
            ("centroid", "質心／門檻"),
            ("axial_1d", "一維投影＋微分"),
            ("hsum", "水平加總"),
            ("gaussian", "高斯濾波"),
        )
        for _key, label in self._spot_algo_items:
            self.combo_spot_algo.addItem(label)
        self.combo_spot_algo.setCurrentIndex(0)
        self.combo_spot_algo.currentIndexChanged.connect(self._on_spot_algo_changed)

        # 相容舊分段按鈕參照（已改為選單）
        self.btn_algo_centroid = None
        self.btn_algo_axial = None
        self.btn_algo_hsum = None
        self.btn_algo_gaussian = None
        self._spot_algo_group = None

        self.chk_hide_profile_cross = QCheckBox("隱藏剖面十字")
        self.chk_hide_profile_cross.setChecked(False)
        self.chk_hide_profile_cross.setToolTip(
            "隱藏短／長曝光與拼貼熱圖上的剖面十字與剖面位置線"
        )
        self.chk_hide_profile_cross.toggled.connect(self._on_hide_profile_cross_toggled)

        self.chk_batch_analysis = QCheckBox("批次分析")
        self.chk_batch_analysis.setChecked(False)
        self.chk_batch_analysis.setToolTip(
            "勾選後按 Apply：用目前參數對資料夾內全部配對依序分析，"
            "結果以「查看批次分析資料」開啟"
        )

        self.stack_algo_params = QStackedWidget()

        def _param_label(text, width=112):
            lbl = QLabel(text)
            lbl.setFixedWidth(width)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            return lbl

        # --- centroid params ---
        centroid_page = QWidget()
        c_lay = QVBoxLayout(centroid_page)
        c_lay.setContentsMargins(0, 4, 0, 0)
        c_lay.setSpacing(6)

        self.spin_short_spot_thresh = NoWheelDoubleSpinBox()
        self.spin_short_spot_thresh.setRange(1.0, 99.0)
        self.spin_short_spot_thresh.setDecimals(1)
        self.spin_short_spot_thresh.setValue(50.0)
        self.spin_short_spot_thresh.setFixedWidth(72)
        self.spin_short_spot_thresh.setToolTip("短曝光門檻％（相對搜尋區最大值）")
        self.spin_short_spot_thresh.valueChanged.connect(
            self._on_analysis_thresh_changed
        )

        self.chk_long_search_roi = QCheckBox("啟用長曝光搜尋範圍（亦作拼貼 Long 區）")
        self.chk_long_search_roi.setChecked(True)
        self.chk_long_search_roi.toggled.connect(self._on_long_roi_option_changed)

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

        self.spin_long_spot_thresh = NoWheelDoubleSpinBox()
        self.spin_long_spot_thresh.setRange(1.0, 99.0)
        self.spin_long_spot_thresh.setDecimals(1)
        self.spin_long_spot_thresh.setValue(50.0)
        self.spin_long_spot_thresh.setFixedWidth(72)
        self.spin_long_spot_thresh.valueChanged.connect(
            self._on_analysis_thresh_changed
        )

        c_grid = QGridLayout()
        c_grid.setContentsMargins(0, 0, 0, 0)
        c_grid.setHorizontalSpacing(8)
        c_grid.setVerticalSpacing(6)
        c_grid.addWidget(_param_label("短曝光門檻 %"), 0, 0)
        c_grid.addWidget(self.spin_short_spot_thresh, 0, 1)
        c_grid.addWidget(_param_label("ROI X"), 1, 0)
        c_grid.addWidget(self.spin_long_roi_x, 1, 1)
        c_grid.addWidget(_param_label("ROI Y"), 2, 0)
        c_grid.addWidget(self.spin_long_roi_y, 2, 1)
        c_grid.addWidget(_param_label("ROI W"), 3, 0)
        c_grid.addWidget(self.spin_long_roi_w, 3, 1)
        c_grid.addWidget(_param_label("ROI H"), 4, 0)
        c_grid.addWidget(self.spin_long_roi_h, 4, 1)
        c_grid.addWidget(_param_label("長曝光門檻 %"), 5, 0)
        c_grid.addWidget(self.spin_long_spot_thresh, 5, 1)
        c_grid.setColumnStretch(2, 1)

        c_lay.addLayout(c_grid)
        c_lay.addWidget(self.chk_long_search_roi)
        c_lay.addStretch(1)

        # --- axial 1d params ---
        axial_page = QWidget()
        a_lay = QVBoxLayout(axial_page)
        a_lay.setContentsMargins(0, 4, 0, 0)
        a_lay.setSpacing(6)

        self.spin_axial_short_thresh = NoWheelDoubleSpinBox()
        self.spin_axial_short_thresh.setRange(1.0, 99.0)
        self.spin_axial_short_thresh.setDecimals(1)
        self.spin_axial_short_thresh.setValue(50.0)
        self.spin_axial_short_thresh.setFixedWidth(72)
        self.spin_axial_short_thresh.valueChanged.connect(
            self._on_analysis_thresh_changed
        )

        self.spin_axial_roi_w = NoWheelSpinBox()
        self.spin_axial_roi_h = NoWheelSpinBox()
        self.spin_axial_roi_w.setRange(20, 2000)
        self.spin_axial_roi_h.setRange(20, 4000)
        self.spin_axial_roi_w.setValue(160)
        self.spin_axial_roi_h.setValue(512)
        self.spin_axial_roi_w.setFixedWidth(72)
        self.spin_axial_roi_h.setFixedWidth(72)

        self.spin_axial_smooth = NoWheelSpinBox()
        self.spin_axial_smooth.setRange(1, 51)
        self.spin_axial_smooth.setSingleStep(2)
        self.spin_axial_smooth.setValue(7)
        self.spin_axial_smooth.setFixedWidth(72)

        self.spin_axial_min_sep = NoWheelSpinBox()
        self.spin_axial_min_sep.setRange(4, 200)
        self.spin_axial_min_sep.setValue(12)
        self.spin_axial_min_sep.setFixedWidth(72)

        a_grid = QGridLayout()
        a_grid.setContentsMargins(0, 0, 0, 0)
        a_grid.setHorizontalSpacing(8)
        a_grid.setVerticalSpacing(6)
        a_grid.addWidget(_param_label("短曝光門檻 %"), 0, 0)
        a_grid.addWidget(self.spin_axial_short_thresh, 0, 1)
        a_grid.addWidget(_param_label("條帶寬 W"), 1, 0)
        a_grid.addWidget(self.spin_axial_roi_w, 1, 1)
        a_grid.addWidget(_param_label("條帶高 H"), 2, 0)
        a_grid.addWidget(self.spin_axial_roi_h, 2, 1)
        a_grid.addWidget(_param_label("剖面平滑窗"), 3, 0)
        a_grid.addWidget(self.spin_axial_smooth, 3, 1)
        a_grid.addWidget(_param_label("最小峰距 (px)"), 4, 0)
        a_grid.addWidget(self.spin_axial_min_sep, 4, 1)
        a_grid.setColumnStretch(2, 1)

        a_lay.addLayout(a_grid)
        a_lay.addStretch(1)

        # --- horizontal sum params（左欄：門檻／ROI／峰距；平滑／精修／次峰窗在波形上方）---
        hsum_page = QWidget()
        h_lay = QVBoxLayout(hsum_page)
        h_lay.setContentsMargins(0, 4, 0, 0)
        h_lay.setSpacing(6)

        self.spin_hsum_short_thresh = NoWheelDoubleSpinBox()
        self.spin_hsum_short_thresh.setRange(1.0, 99.0)
        self.spin_hsum_short_thresh.setDecimals(1)
        self.spin_hsum_short_thresh.setValue(50.0)
        self.spin_hsum_short_thresh.setFixedWidth(72)
        self.spin_hsum_short_thresh.valueChanged.connect(
            self._on_analysis_thresh_changed
        )

        self.spin_hsum_roi_w = NoWheelSpinBox()
        self.spin_hsum_roi_h = NoWheelSpinBox()
        self.spin_hsum_roi_w.setRange(20, 2000)
        self.spin_hsum_roi_h.setRange(20, 4000)
        self.spin_hsum_roi_w.setValue(160)
        self.spin_hsum_roi_h.setValue(512)
        self.spin_hsum_roi_w.setFixedWidth(72)
        self.spin_hsum_roi_h.setFixedWidth(72)
        self.spin_hsum_roi_w.setToolTip("條帶寬（水平加總 ROI）")
        self.spin_hsum_roi_h.setToolTip("條帶高（水平加總 ROI）")

        # 平滑／精修／次峰窗已在主畫面波形上方建立
        if getattr(self, "spin_hsum_smooth", None) is None:
            self.spin_hsum_smooth = NoWheelSpinBox()
            self.spin_hsum_smooth.setRange(1, 201)
            self.spin_hsum_smooth.setSingleStep(2)
            self.spin_hsum_smooth.setValue(7)
            self.spin_hsum_smooth.setFixedWidth(72)
        if getattr(self, "spin_hsum_refine", None) is None:
            self.spin_hsum_refine = NoWheelSpinBox()
            self.spin_hsum_refine.setRange(0, 20)
            self.spin_hsum_refine.setValue(1)
            self.spin_hsum_refine.setFixedWidth(72)
        if getattr(self, "spin_hsum_sec_lo", None) is None:
            self.spin_hsum_sec_lo = NoWheelDoubleSpinBox()
            self.spin_hsum_sec_lo.setRange(0.0, 1e6)
            self.spin_hsum_sec_lo.setDecimals(1)
            self.spin_hsum_sec_lo.setFixedWidth(72)
            self.spin_hsum_sec_lo.setValue(0.0)
        if getattr(self, "spin_hsum_sec_hi", None) is None:
            self.spin_hsum_sec_hi = NoWheelDoubleSpinBox()
            self.spin_hsum_sec_hi.setRange(0.0, 1e6)
            self.spin_hsum_sec_hi.setDecimals(1)
            self.spin_hsum_sec_hi.setFixedWidth(72)
            self.spin_hsum_sec_hi.setValue(0.0)

        self.spin_hsum_min_sep = NoWheelSpinBox()
        self.spin_hsum_min_sep.setRange(4, 200)
        self.spin_hsum_min_sep.setValue(12)
        self.spin_hsum_min_sep.setFixedWidth(72)

        h_grid = QGridLayout()
        h_grid.setContentsMargins(0, 0, 0, 0)
        h_grid.setHorizontalSpacing(8)
        h_grid.setVerticalSpacing(6)
        h_grid.addWidget(_param_label("短曝光門檻 %"), 0, 0)
        h_grid.addWidget(self.spin_hsum_short_thresh, 0, 1)
        h_grid.addWidget(_param_label("條帶寬 W"), 1, 0)
        h_grid.addWidget(self.spin_hsum_roi_w, 1, 1)
        h_grid.addWidget(_param_label("條帶高 H"), 2, 0)
        h_grid.addWidget(self.spin_hsum_roi_h, 2, 1)
        h_grid.addWidget(_param_label("最小峰距 (px)"), 3, 0)
        h_grid.addWidget(self.spin_hsum_min_sep, 3, 1)
        h_grid.setColumnStretch(2, 1)
        h_lay.addLayout(h_grid)
        h_hint = QLabel(
            "主畫面為波形；平滑／精修／次峰窗在上方。"
            "Heatmap 請按「查看 Heatmap」（ROI＋上光斑，不下光斑）。"
        )
        h_hint.setWordWrap(True)
        h_hint.setStyleSheet(_STYLE_MUTED)
        h_lay.addWidget(h_hint)
        h_lay.addStretch(1)

        # --- gaussian params ---
        gaussian_page = QWidget()
        g_lay = QVBoxLayout(gaussian_page)
        g_lay.setContentsMargins(0, 4, 0, 0)
        g_lay.setSpacing(6)

        self.spin_gauss_short_thresh = NoWheelDoubleSpinBox()
        self.spin_gauss_short_thresh.setRange(1.0, 99.0)
        self.spin_gauss_short_thresh.setDecimals(1)
        self.spin_gauss_short_thresh.setValue(50.0)
        self.spin_gauss_short_thresh.setFixedWidth(72)
        self.spin_gauss_short_thresh.valueChanged.connect(
            self._on_analysis_thresh_changed
        )

        self.spin_gauss_sigma = NoWheelDoubleSpinBox()
        self.spin_gauss_sigma.setRange(0.5, 32.0)
        self.spin_gauss_sigma.setDecimals(1)
        self.spin_gauss_sigma.setSingleStep(0.5)
        self.spin_gauss_sigma.setValue(4.0)
        self.spin_gauss_sigma.setFixedWidth(72)
        self.spin_gauss_sigma.setToolTip("2D 高斯平滑 σ（px）；愈大愈穩但質心易偏")

        self.spin_gauss_sec_thresh = NoWheelDoubleSpinBox()
        self.spin_gauss_sec_thresh.setRange(0.05, 0.95)
        self.spin_gauss_sec_thresh.setDecimals(2)
        self.spin_gauss_sec_thresh.setSingleStep(0.05)
        self.spin_gauss_sec_thresh.setValue(0.35)
        self.spin_gauss_sec_thresh.setFixedWidth(72)
        self.spin_gauss_sec_thresh.setToolTip("下光斑相對峰門檻（峰值比例）")

        self.spin_gauss_half_x = NoWheelSpinBox()
        self.spin_gauss_half_x.setRange(10, 400)
        self.spin_gauss_half_x.setValue(80)
        self.spin_gauss_half_x.setFixedWidth(72)
        self.spin_gauss_half_x.setToolTip("下光斑搜尋帶 X 半寬（px）")

        self.spin_gauss_min_sep = NoWheelSpinBox()
        self.spin_gauss_min_sep.setRange(10, 400)
        self.spin_gauss_min_sep.setValue(120)
        self.spin_gauss_min_sep.setFixedWidth(72)
        self.spin_gauss_min_sep.setToolTip("上光斑下方排除帶起點（px）")

        self.spin_gauss_blank_r = NoWheelDoubleSpinBox()
        self.spin_gauss_blank_r.setRange(0.0, 400.0)
        self.spin_gauss_blank_r.setDecimals(0)
        self.spin_gauss_blank_r.setValue(90.0)
        self.spin_gauss_blank_r.setFixedWidth(72)
        self.spin_gauss_blank_r.setToolTip("抹掉上光斑鄰域半徑（px），避免抓到主峰下緣")

        g_grid = QGridLayout()
        g_grid.setContentsMargins(0, 0, 0, 0)
        g_grid.setHorizontalSpacing(8)
        g_grid.setVerticalSpacing(6)
        g_grid.addWidget(_param_label("短曝光門檻 %"), 0, 0)
        g_grid.addWidget(self.spin_gauss_short_thresh, 0, 1)
        g_grid.addWidget(_param_label("高斯 σ (px)"), 1, 0)
        g_grid.addWidget(self.spin_gauss_sigma, 1, 1)
        g_grid.addWidget(_param_label("下斑相對峰"), 2, 0)
        g_grid.addWidget(self.spin_gauss_sec_thresh, 2, 1)
        g_grid.addWidget(_param_label("搜尋半寬 X"), 3, 0)
        g_grid.addWidget(self.spin_gauss_half_x, 3, 1)
        g_grid.addWidget(_param_label("最小分離 Y"), 4, 0)
        g_grid.addWidget(self.spin_gauss_min_sep, 4, 1)
        g_grid.addWidget(_param_label("主斑抹除半徑"), 5, 0)
        g_grid.addWidget(self.spin_gauss_blank_r, 5, 1)
        g_grid.setColumnStretch(2, 1)

        g_lay.addLayout(g_grid)
        g_lay.addStretch(1)

        self.stack_algo_params.addWidget(centroid_page)
        self.stack_algo_params.addWidget(axial_page)
        self.stack_algo_params.addWidget(hsum_page)
        self.stack_algo_params.addWidget(gaussian_page)

        self.btn_apply_spot = QPushButton("Apply")
        self.btn_apply_spot.setStyleSheet(_STYLE_BTN_PRIMARY)
        self.btn_apply_spot.setToolTip(
            "單組：分析目前載入配對｜勾選「批次分析」：分析資料夾全部配對"
        )
        self.btn_apply_spot.clicked.connect(self.apply_selected_spot_algorithm)

        self.lbl_short_spot_pos = QLabel("短曝光光斑: --")
        self.lbl_short_spot_pos.setWordWrap(True)
        self.lbl_long_spot_pos = QLabel("長曝光第二光斑: --")
        self.lbl_long_spot_pos.setWordWrap(True)
        self.lbl_spot_delta = QLabel("兩光斑差距: --")
        self.lbl_spot_delta.setWordWrap(True)

        self.btn_clear_spots = QPushButton("清除光斑標記")
        self.btn_clear_spots.setStyleSheet(_STYLE_BTN)
        self.btn_clear_spots.clicked.connect(self.clear_spots)

        self.btn_open_collage = QPushButton("開啟拼貼視窗")
        self.btn_open_collage.setStyleSheet(_STYLE_BTN)
        self.btn_open_collage.setToolTip(
            "另開視窗顯示拼貼 heatmap（Long 搜尋範圍＋Short 其餘區）"
        )
        self.btn_open_collage.clicked.connect(self.open_collage_window)

        # compatibility aliases (旧 API / 其他方法仍可能引用)
        self.btn_grab_short_spot = self.btn_apply_spot
        self.btn_grab_long_spot = self.btn_apply_spot
        self.btn_grab_axial_1d = self.btn_apply_spot
        self.btn_grab_gaussian = self.btn_apply_spot

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #E0E0E0;")
        sep.setMaximumHeight(2)

        # --- Tab 1：分析 ---
        analysis_page = QWidget()
        a_page_lay = QVBoxLayout(analysis_page)
        a_page_lay.setContentsMargins(4, 6, 4, 4)
        a_page_lay.setSpacing(8)
        a_page_lay.setAlignment(Qt.AlignTop)
        a_page_lay.addWidget(algo_lbl)
        a_page_lay.addWidget(self.combo_spot_algo)
        a_page_lay.addWidget(self.chk_hide_profile_cross)
        a_page_lay.addWidget(self.chk_batch_analysis)
        a_page_lay.addWidget(self.stack_algo_params)
        a_page_lay.addWidget(self.btn_apply_spot)
        a_page_lay.addWidget(sep)
        a_page_lay.addWidget(self.lbl_short_spot_pos)
        a_page_lay.addWidget(self.lbl_long_spot_pos)
        a_page_lay.addWidget(self.lbl_spot_delta)
        a_page_lay.addWidget(self.btn_clear_spots)
        a_page_lay.addWidget(self.btn_open_collage)
        a_page_lay.addStretch(1)

        # --- Tab 2：Beam size ---
        beam_page = self._build_beam_size_tab()

        self.tabs_spot_tools = QTabWidget()
        self.tabs_spot_tools.setDocumentMode(True)
        self.tabs_spot_tools.addTab(analysis_page, "分析")
        self.tabs_spot_tools.addTab(beam_page, "Beam size")
        self.tabs_spot_tools.currentChanged.connect(self._on_spot_tools_tab_changed)

        title.setText("分析工具")
        lay.addWidget(title)
        lay.addWidget(self.tabs_spot_tools)

        self._short_beam_circle_items = []
        self._long_beam_circle_items = []
        self._beam_size_short = None  # (cx, cy, r) full-matrix coords
        self._beam_size_long = None

        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left.addWidget(box)
        left.addStretch(1)
        self._spot_controls_box = box
        self._on_spot_algo_changed(0)

    def _build_beam_size_tab(self):
        """Beam size：依分析門檻區求真正最大內切圓（不鎖峰值／質心）。"""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignTop)

        hint = QLabel(
            "與「顯示門檻」同一套分析矩陣＋門檻％："
            "對門檻最大連通區做距離變換，取可內切的最大圓（圓心可偏離峰值）。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_STYLE_MUTED)

        self.lbl_beam_thresh_info = QLabel("目前門檻: 短 --% ／ 長 --%")
        self.lbl_beam_thresh_info.setWordWrap(True)

        self.chk_beam_size_enable = QCheckBox("啟用 Beam size（計算並顯示內切圓）")
        self.chk_beam_size_enable.setChecked(False)
        self.chk_beam_size_enable.setToolTip(
            "勾選後依門檻區幾何求最大內切圓（非峰值鎖定），並畫在短／長曝光 heatmap 上"
        )
        self.chk_beam_size_enable.toggled.connect(self._on_beam_size_enable_toggled)

        self.btn_beam_size_recalc = QPushButton("重新計算")
        self.btn_beam_size_recalc.setStyleSheet(_STYLE_BTN)
        self.btn_beam_size_recalc.setEnabled(False)
        self.btn_beam_size_recalc.clicked.connect(self._compute_and_show_beam_size)

        self.lbl_beam_short = QLabel("短曝光 Beam size: --")
        self.lbl_beam_short.setWordWrap(True)
        self.lbl_beam_long = QLabel("長曝光 Beam size: --")
        self.lbl_beam_long.setWordWrap(True)

        note = QLabel(
            "直徑 = 2×內切圓半徑（px／μm）。"
            "圓心 = 門檻區（填洞後）距離變換最大值，對齊半透明門檻外輪廓。"
        )
        note.setWordWrap(True)
        note.setStyleSheet(_STYLE_MUTED)

        lay.addWidget(hint)
        lay.addWidget(self.lbl_beam_thresh_info)
        lay.addWidget(self.chk_beam_size_enable)
        lay.addWidget(self.btn_beam_size_recalc)
        lay.addWidget(self.lbl_beam_short)
        lay.addWidget(self.lbl_beam_long)
        lay.addWidget(note)
        lay.addStretch(1)
        return page

    def _on_spot_tools_tab_changed(self, index):
        if int(index) == 1:
            self._update_beam_thresh_info_label()
            if (
                getattr(self, "chk_beam_size_enable", None) is not None
                and self.chk_beam_size_enable.isChecked()
            ):
                self._compute_and_show_beam_size()

    def _on_beam_size_enable_toggled(self, checked):
        btn = getattr(self, "btn_beam_size_recalc", None)
        if btn is not None:
            btn.setEnabled(bool(checked))
        if checked:
            self._compute_and_show_beam_size()
        else:
            self._clear_beam_size_overlays()
            self._beam_size_short = None
            self._beam_size_long = None
            if getattr(self, "lbl_beam_short", None) is not None:
                self.lbl_beam_short.setText("短曝光 Beam size: --")
            if getattr(self, "lbl_beam_long", None) is not None:
                self.lbl_beam_long.setText("長曝光 Beam size: --")

    def _update_beam_thresh_info_label(self):
        lbl = getattr(self, "lbl_beam_thresh_info", None)
        if lbl is None:
            return
        short_t = self._short_spot_thresh_percent()
        long_t = self._long_spot_thresh_percent()
        lbl.setText(
            f"目前門檻: 短 {short_t:.1f}% ／ 長 {long_t:.1f}%（與分析分頁同步）"
        )

    def _fit_beam_circle_on_matrix(self, matrix, thresh_percent, roi=None):
        """對與「顯示門檻」相同的 mask 求最大內切圓，回傳全圖 (cx, cy, r)。

        圓心 = 門檻最大連通區的 EDT argmax（不鎖峰值／質心）。
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0 or not np.isfinite(matrix).any():
            return None
        mask = self._build_full_threshold_mask(
            matrix, float(thresh_percent), roi=roi
        )
        if mask is None or not np.any(mask):
            return None
        return fit_max_inscribed_circle(mask)

    def _beam_size_label_text(self, prefix, circle):
        if circle is None:
            return f"{prefix}: --（門檻區無效或無影像）"
        cx, cy, r = circle
        diam = 2.0 * r
        um = self._px_to_um(diam)
        return (
            f"{prefix}: 直徑={diam:.2f} px ({um:.2f} μm)\n"
            f"  圓心=({cx:.2f}, {cy:.2f})"
        )

    def _long_beam_size_roi(self):
        """長曝光 Beam size／門檻 overlay 共用的 ROI。"""
        if self._current_spot_algo() == "axial_1d":
            axial = getattr(self, "_axial_roi_xywh", None)
            if axial is not None:
                return axial
        if self._current_spot_algo() == "hsum":
            hsum = getattr(self, "_hsum_roi_xywh", None)
            if hsum is not None:
                return hsum
        if self._current_spot_algo() == "gaussian":
            g_roi = getattr(self, "_gaussian_roi_xywh", None)
            if g_roi is not None:
                return g_roi
        if (
            getattr(self, "chk_long_search_roi", None) is not None
            and self.chk_long_search_roi.isChecked()
        ):
            return self._long_roi_tuple()
        return None

    def _gaussian_algo_kwargs(self):
        """目前 UI 高斯演算法參數。"""
        return {
            "sigma": float(
                self.spin_gauss_sigma.value()
                if getattr(self, "spin_gauss_sigma", None)
                else 4.0
            ),
            "thresh_frac": float(
                self.spin_gauss_sec_thresh.value()
                if getattr(self, "spin_gauss_sec_thresh", None)
                else 0.35
            ),
            "search_half_x": int(
                self.spin_gauss_half_x.value()
                if getattr(self, "spin_gauss_half_x", None)
                else 80
            ),
            "min_sep_y": int(
                self.spin_gauss_min_sep.value()
                if getattr(self, "spin_gauss_min_sep", None)
                else 120
            ),
            "primary_blank_r": float(
                self.spin_gauss_blank_r.value()
                if getattr(self, "spin_gauss_blank_r", None)
                else 90.0
            ),
        }

    def _compute_and_show_beam_size(self):
        self._update_beam_thresh_info_label()
        # 必須與「顯示門檻」同一分析矩陣（含 color scale 硬裁切），否則同％會得到不同 mask
        short_mat = self._short_analysis_matrix()
        long_mat = self._long_analysis_matrix()

        self._beam_size_short = None
        self._beam_size_long = None

        if short_mat is not None:
            self._beam_size_short = self._fit_beam_circle_on_matrix(
                short_mat, self._short_spot_thresh_percent(), roi=None
            )
        if long_mat is not None:
            self._beam_size_long = self._fit_beam_circle_on_matrix(
                long_mat,
                self._long_spot_thresh_percent(),
                roi=self._long_beam_size_roi(),
            )

        if getattr(self, "lbl_beam_short", None) is not None:
            self.lbl_beam_short.setText(
                self._beam_size_label_text("短曝光 Beam size", self._beam_size_short)
            )
        if getattr(self, "lbl_beam_long", None) is not None:
            self.lbl_beam_long.setText(
                self._beam_size_label_text("長曝光 Beam size", self._beam_size_long)
            )
        self._redraw_beam_size_overlays()

    def _clear_beam_size_overlays(self):
        short_panel = getattr(self, "short_heatmap_panel", None)
        long_panel = getattr(self, "long_heatmap_panel", None)
        if short_panel is not None:
            self._clear_plot_items(short_panel.plot, "_short_beam_circle_items")
        if long_panel is not None:
            self._clear_plot_items(long_panel.plot, "_long_beam_circle_items")

    def _add_circle_overlay(self, plot, items_attr, cx, cy, radius, pen):
        if radius is None or float(radius) <= 0:
            return
        n = 96
        theta = np.linspace(0.0, 2.0 * np.pi, n)
        xs = float(cx) + float(radius) * np.cos(theta)
        ys = float(cy) + float(radius) * np.sin(theta)
        circle = pg.PlotCurveItem(x=xs, y=ys, pen=pen)
        # 十字標示圓心
        arm = max(2.0, 0.15 * float(radius))
        cross_v = pg.PlotCurveItem(
            x=[cx, cx], y=[cy - arm, cy + arm], pen=pen
        )
        cross_h = pg.PlotCurveItem(
            x=[cx - arm, cx + arm], y=[cy, cy], pen=pen
        )
        plot.addItem(circle)
        plot.addItem(cross_v)
        plot.addItem(cross_h)
        items = getattr(self, items_attr)
        items.extend([circle, cross_v, cross_h])

    def _redraw_beam_size_overlays(self):
        self._clear_beam_size_overlays()
        enabled = (
            getattr(self, "chk_beam_size_enable", None) is not None
            and self.chk_beam_size_enable.isChecked()
        )
        if not enabled:
            return
        short_panel = getattr(self, "short_heatmap_panel", None)
        long_panel = getattr(self, "long_heatmap_panel", None)
        if short_panel is not None and self._beam_size_short is not None:
            cx, cy, r = self._beam_size_short
            self._add_circle_overlay(
                short_panel.plot,
                "_short_beam_circle_items",
                cx,
                cy,
                r,
                pg.mkPen("#00C853", width=2.5),
            )
        if long_panel is not None and self._beam_size_long is not None:
            cx, cy, r = self._beam_size_long
            self._add_circle_overlay(
                long_panel.plot,
                "_long_beam_circle_items",
                cx,
                cy,
                r,
                pg.mkPen("#FF6D00", width=2.5),
            )

    def _on_analysis_thresh_changed(self, *_args):
        self._refresh_short_thresh_overlay()
        self._refresh_long_thresh_overlay()
        self._update_beam_thresh_info_label()
        if (
            getattr(self, "chk_beam_size_enable", None) is not None
            and self.chk_beam_size_enable.isChecked()
        ):
            self._compute_and_show_beam_size()

    def _on_algo_button_clicked(self, _btn=None):
        """相容舊呼叫：改走選單目前索引。"""
        combo = getattr(self, "combo_spot_algo", None)
        idx = int(combo.currentIndex()) if combo is not None else 0
        if idx < 0:
            idx = 0
        self._on_spot_algo_changed(idx)

    def _style_algo_segment_buttons(self):
        """舊分段按鈕樣式；改為選單後不再需要。"""
        return

    def _current_spot_algo(self):
        combo = getattr(self, "combo_spot_algo", None)
        if combo is None:
            return getattr(self, "_spot_algo", "centroid") or "centroid"
        idx = int(combo.currentIndex())
        items = getattr(self, "_spot_algo_items", None)
        if items and 0 <= idx < len(items):
            return items[idx][0]
        if idx == 3:
            return "gaussian"
        if idx == 2:
            return "hsum"
        if idx == 1:
            return "axial_1d"
        return "centroid"

    def _reset_axial_roi_defaults(self):
        """切換到一維投影時，條帶 ROI 回到預設（不沿用質心 ROI）。"""
        if getattr(self, "spin_axial_roi_w", None) is not None:
            self.spin_axial_roi_w.blockSignals(True)
            self.spin_axial_roi_w.setValue(160)
            self.spin_axial_roi_w.blockSignals(False)
        if getattr(self, "spin_axial_roi_h", None) is not None:
            self.spin_axial_roi_h.blockSignals(True)
            self.spin_axial_roi_h.setValue(512)
            self.spin_axial_roi_h.blockSignals(False)
        self._axial_roi_xywh = None

    def _reset_hsum_roi_defaults(self):
        """切換到水平加總時，條帶 ROI／次峰窗回到預設。"""
        for spin, val in (
            (getattr(self, "spin_hsum_roi_w", None), 160),
            (getattr(self, "spin_hsum_roi_h", None), 512),
        ):
            if spin is None:
                continue
            spin.blockSignals(True)
            spin.setValue(int(val))
            spin.blockSignals(False)
        for spin in (
            getattr(self, "spin_hsum_sec_lo", None),
            getattr(self, "spin_hsum_sec_hi", None),
        ):
            if spin is None:
                continue
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._hsum_roi_xywh = None
        self._hsum_sec_window = None

    def _set_axial_display_roi(self, xywh):
        """一維投影結果的條帶框，只供顯示，不寫入質心 ROI spin。"""
        self._axial_roi_xywh = tuple(int(v) for v in xywh) if xywh is not None else None

    def _set_hsum_display_roi(self, xywh):
        """水平加總條帶框，只供顯示。"""
        self._hsum_roi_xywh = tuple(int(v) for v in xywh) if xywh is not None else None

    def _set_gaussian_display_roi(self, xywh):
        """高斯下光斑搜尋帶，只供顯示。"""
        self._gaussian_roi_xywh = (
            tuple(int(v) for v in xywh) if xywh is not None else None
        )

    def _hsum_algo_kwargs(self):
        """水平加總演算法參數。"""
        sec_lo = float(
            self.spin_hsum_sec_lo.value()
            if getattr(self, "spin_hsum_sec_lo", None)
            else 0.0
        )
        sec_hi = float(
            self.spin_hsum_sec_hi.value()
            if getattr(self, "spin_hsum_sec_hi", None)
            else 0.0
        )
        sec_win = None
        if abs(sec_lo) > 1e-9 or abs(sec_hi) > 1e-9:
            sec_win = (sec_lo, sec_hi)
        return {
            "roi_w": int(
                self.spin_hsum_roi_w.value()
                if getattr(self, "spin_hsum_roi_w", None)
                else 160
            ),
            "roi_h": int(
                self.spin_hsum_roi_h.value()
                if getattr(self, "spin_hsum_roi_h", None)
                else 512
            ),
            "smooth_win": int(
                self.spin_hsum_smooth.value()
                if getattr(self, "spin_hsum_smooth", None)
                else 7
            ),
            "min_peak_distance": int(
                self.spin_hsum_min_sep.value()
                if getattr(self, "spin_hsum_min_sep", None)
                else 12
            ),
            "n_refine": int(
                self.spin_hsum_refine.value()
                if getattr(self, "spin_hsum_refine", None)
                else 1
            ),
            "secondary_window_y": sec_win,
        }

    def _on_spot_algo_changed(self, index):
        stack = getattr(self, "stack_algo_params", None)
        if stack is None:
            return
        idx = max(0, int(index))
        stack.setCurrentIndex(idx)
        prev = getattr(self, "_spot_algo", "centroid")
        if idx == 3:
            self._spot_algo = "gaussian"
        elif idx == 2:
            self._spot_algo = "hsum"
        elif idx == 1:
            self._spot_algo = "axial_1d"
        else:
            self._spot_algo = "centroid"
        if self._spot_algo == "axial_1d" and prev != "axial_1d":
            self._reset_axial_roi_defaults()
        if self._spot_algo == "hsum" and prev != "hsum":
            self._reset_hsum_roi_defaults()
        if self._spot_algo != "gaussian":
            self._gaussian_roi_xywh = None
        self._sync_main_view_for_algo()
        self._redraw_spot_overlays()
        self._refresh_all_thresh_overlays()
        self._update_horizontal_sum_buttons_visibility()

    def apply_selected_spot_algorithm(self):
        if (
            getattr(self, "chk_batch_analysis", None) is not None
            and self.chk_batch_analysis.isChecked()
        ):
            self.run_batch_analysis()
            return
        algo = self._current_spot_algo()
        if algo == "axial_1d":
            self.grab_dual_spots_axial_1d()
        elif algo == "hsum":
            self.grab_dual_spots_hsum()
        elif algo == "gaussian":
            self.grab_dual_spots_gaussian()
        else:
            self.grab_short_spot()
            if self.short_spot is None:
                return
            self.grab_long_spot2()

    def _compute_spots_quiet(self):
        """對目前已載入矩陣跑演算法；不跳對話框。回傳結果 dict。"""
        algo = self._current_spot_algo()
        algo_info = self._collect_algorithm_export_info()
        out = {
            "ok": False,
            "error": None,
            "algorithm": algo_info.get("algorithm"),
            "short_spot": None,
            "long_spot2": None,
            "delta": None,
        }
        short_mat = self._short_analysis_matrix()
        long_mat = self._long_analysis_matrix()
        if short_mat is None:
            out["error"] = "短曝光資料缺失"
            return out
        if long_mat is None:
            out["error"] = "長曝光資料缺失"
            return out

        try:
            if algo == "axial_1d":
                short_spot = self._find_spot(
                    short_mat,
                    roi=None,
                    thresh_percent=self._short_spot_thresh_percent(),
                )
                if short_spot is None:
                    out["error"] = "短曝光找不到光斑"
                    return out
                self.short_spot = short_spot
                center = self._map_short_spot_to_long()
                if center is None:
                    out["error"] = "無法對應短曝光到長曝光"
                    return out
                roi_w = (
                    int(self.spin_axial_roi_w.value())
                    if getattr(self, "spin_axial_roi_w", None)
                    else 160
                )
                roi_h = (
                    int(self.spin_axial_roi_h.value())
                    if getattr(self, "spin_axial_roi_h", None)
                    else 512
                )
                smooth_win = (
                    int(self.spin_axial_smooth.value())
                    if getattr(self, "spin_axial_smooth", None)
                    else 7
                )
                min_sep = (
                    int(self.spin_axial_min_sep.value())
                    if getattr(self, "spin_axial_min_sep", None)
                    else 12
                )
                result = find_dual_spots_axial_1d_projection(
                    long_mat,
                    center,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    smooth_win=smooth_win,
                    min_peak_distance=min_sep,
                )
                spots = result.get("spots")
                if not spots or len(spots) < 2:
                    out["error"] = "一維投影未找到雙峰"
                    out["short_spot"] = short_spot
                    return out
                if abs(float(spots[0][1]) - float(spots[1][1])) < 2.0:
                    out["error"] = "雙峰未分開"
                    out["short_spot"] = short_spot
                    return out
                if result.get("roi_xywh") is not None:
                    self._set_axial_display_roi(result["roi_xywh"])
                primary_i = int(result.get("primary_index", 0))
                secondary_i = 1 - primary_i
                spot2 = spots[secondary_i]
                self.long_spot2 = (float(spot2[0]), float(spot2[1]))
            elif algo == "hsum":
                short_spot = self._find_spot(
                    short_mat,
                    roi=None,
                    thresh_percent=self._short_spot_thresh_percent(),
                )
                if short_spot is None:
                    out["error"] = "短曝光找不到光斑"
                    return out
                self.short_spot = short_spot
                center = self._map_short_spot_to_long()
                if center is None:
                    out["error"] = "無法對應短曝光到長曝光"
                    return out
                hkw = self._hsum_algo_kwargs()
                result = find_dual_spots_horizontal_sum(
                    long_mat,
                    center,
                    **hkw,
                )
                spots = result.get("spots")
                if not spots or len(spots) < 2:
                    out["error"] = "水平加總未找到雙峰"
                    out["short_spot"] = short_spot
                    return out
                if abs(float(spots[0][1]) - float(spots[1][1])) < 2.0:
                    out["error"] = "雙峰未分開"
                    out["short_spot"] = short_spot
                    return out
                if result.get("roi_xywh") is not None:
                    self._set_hsum_display_roi(result["roi_xywh"])
                win = result.get("secondary_window_y")
                if win is not None:
                    self._hsum_sec_window = (float(win[0]), float(win[1]))
                primary_i = int(result.get("primary_index", 0))
                secondary_i = 1 - primary_i
                spot2 = spots[secondary_i]
                self.long_spot2 = (float(spot2[0]), float(spot2[1]))
            elif algo == "gaussian":
                short_spot = self._find_spot(
                    short_mat,
                    roi=None,
                    thresh_percent=self._short_spot_thresh_percent(),
                )
                if short_spot is None:
                    out["error"] = "短曝光找不到光斑"
                    return out
                self.short_spot = short_spot
                center = self._map_short_spot_to_long()
                if center is None:
                    out["error"] = "無法對應短曝光到長曝光"
                    return out
                result = find_dual_spots_gaussian(
                    long_mat,
                    center,
                    **self._gaussian_algo_kwargs(),
                )
                secondary = result.get("secondary")
                if result.get("search_roi_xywh") is not None:
                    self._set_gaussian_display_roi(result["search_roi_xywh"])
                if secondary is None:
                    out["error"] = f"高斯濾波未找到下光斑（{result.get('reason', '')}）"
                    out["short_spot"] = short_spot
                    return out
                self.long_spot2 = (float(secondary[0]), float(secondary[1]))
            else:
                short_spot = self._find_spot(
                    short_mat,
                    roi=None,
                    thresh_percent=self._short_spot_thresh_percent(),
                )
                if short_spot is None:
                    out["error"] = "短曝光找不到光斑"
                    return out
                self.short_spot = short_spot
                roi = None
                if (
                    getattr(self, "chk_long_search_roi", None) is not None
                    and self.chk_long_search_roi.isChecked()
                ):
                    roi = self._long_roi_tuple()
                long_spot = self._find_spot(
                    long_mat,
                    roi=roi,
                    thresh_percent=self._long_spot_thresh_percent(),
                )
                if long_spot is None:
                    out["error"] = "長曝光找不到第二光斑"
                    out["short_spot"] = short_spot
                    return out
                self.long_spot2 = long_spot

            out["ok"] = True
            out["short_spot"] = (
                float(self.short_spot[0]),
                float(self.short_spot[1]),
            )
            out["long_spot2"] = (
                float(self.long_spot2[0]),
                float(self.long_spot2[1]),
            )
            out["delta"] = self._spot_delta_values()
            if algo == "axial_1d" and getattr(self, "_axial_roi_xywh", None) is not None:
                out["axial_roi_xywh"] = list(self._axial_roi_xywh)
            elif algo == "hsum" and getattr(self, "_hsum_roi_xywh", None) is not None:
                out["hsum_roi_xywh"] = list(self._hsum_roi_xywh)
                if getattr(self, "_hsum_sec_window", None) is not None:
                    out["secondary_search_window_y"] = list(self._hsum_sec_window)
            elif (
                algo == "gaussian"
                and getattr(self, "_gaussian_roi_xywh", None) is not None
            ):
                out["gaussian_search_roi_xywh"] = list(self._gaussian_roi_xywh)
            elif (
                getattr(self, "chk_long_search_roi", None) is not None
                and self.chk_long_search_roi.isChecked()
            ):
                out["long_search_roi_xywh"] = list(self._long_roi_tuple())
            self.lbl_short_spot_pos.setText(
                f"短曝光光斑: X={self.short_spot[0]:.2f}, Y={self.short_spot[1]:.2f}"
            )
            self.lbl_long_spot_pos.setText(
                f"長曝光第二光斑: X={self.long_spot2[0]:.2f}, "
                f"Y={self.long_spot2[1]:.2f}"
            )
            self._update_spot_delta_label()
            self._redraw_spot_overlays()
        except Exception as exc:
            out["error"] = str(exc)
        return out

    def run_batch_analysis(self):
        """用目前參數對資料夾全部配對做分析，結果存入批次檢視。"""
        pairs = list(getattr(self, "_data_pairs", None) or [])
        if not pairs:
            QMessageBox.warning(
                self, "警告", "請先選擇資料夾並確認有短／長曝光配對。"
            )
            return

        algo_info = self._collect_algorithm_export_info()
        n = len(pairs)
        progress = QProgressDialog("批次分析中…", "取消", 0, n, self)
        progress.setWindowTitle("批次分析")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        results = []
        cancelled = False
        for i, pair in enumerate(pairs):
            if progress.wasCanceled():
                cancelled = True
                break
            progress.setLabelText(
                f"分析 {i + 1}/{n}：{pair.get('label', '')}"
            )
            progress.setValue(i)
            QApplication.processEvents()

            row = {
                "index": i,
                "label": pair.get("label", f"pair_{i + 1}"),
                "short_path": pair.get("short_path"),
                "long_path": pair.get("long_path"),
                "short_idx": pair.get("short_idx"),
                "long_idx": pair.get("long_idx"),
                "ok": False,
                "error": None,
                "short_spot": None,
                "long_spot2": None,
                "delta": None,
            }
            try:
                self._load_pair_paths(pair["short_path"], pair["long_path"])
                QApplication.processEvents()
                computed = self._compute_spots_quiet()
                row["ok"] = bool(computed.get("ok"))
                row["error"] = computed.get("error")
                row["short_spot"] = computed.get("short_spot")
                row["long_spot2"] = computed.get("long_spot2")
                row["delta"] = computed.get("delta")
                if computed.get("axial_roi_xywh") is not None:
                    row["axial_roi_xywh"] = computed.get("axial_roi_xywh")
                if computed.get("long_search_roi_xywh") is not None:
                    row["long_search_roi_xywh"] = computed.get(
                        "long_search_roi_xywh"
                    )
            except Exception as exc:
                row["error"] = str(exc)
            results.append(row)

        progress.setValue(n)
        progress.close()

        ok_n = sum(1 for r in results if r.get("ok"))
        self._batch_analysis_store = {
            "algorithm": algo_info.get("algorithm"),
            "algorithm_desc": algo_info.get("algorithm_desc"),
            "algorithm_params": algo_info.get("algorithm_params") or {},
            "data_folder": self.data_folder,
            "results": results,
            "cancelled": cancelled,
        }
        if getattr(self, "btn_view_batch_results", None) is not None:
            self.btn_view_batch_results.setEnabled(True)

        status = (
            f"狀態: 批次分析完成 {ok_n}/{len(results)} 成功"
            + ("（已取消剩餘）" if cancelled else "")
        )
        self.lbl_batch_status.setText(status)
        is_hsum = BaslerBatchDualTab._is_hsum_algorithm_id(
            algo_info.get("algorithm")
        )
        QMessageBox.information(
            self,
            "批次分析",
            f"{status}\n\n"
            + (
                "將開啟批次水平加總。"
                if is_hsum
                else "請按「查看批次分析資料」開啟結果視窗。"
            ),
        )
        if is_hsum:
            self.open_batch_hsum_window()
        else:
            self.open_batch_results_window()

    def _on_view_batch_results_clicked(self):
        store = getattr(self, "_batch_analysis_store", None)
        if not store or not store.get("results"):
            QMessageBox.information(
                self,
                "批次分析",
                "尚無批次分析資料，請先勾選「批次分析」並按 Apply。",
            )
            return
        is_hsum = BaslerBatchDualTab._is_hsum_algorithm_id(store.get("algorithm"))
        is_axial = BaslerBatchDualTab._is_axial_algorithm_id(store.get("algorithm"))
        # 一維投影只有 Heatmap 檢視，直接開啟，不放反白的水平加總選項
        if is_axial:
            self.open_batch_results_window()
            return
        menu = QMenu(self)
        act_heatmap = menu.addAction("Heatmap（短／長熱圖）")
        act_hsum = menu.addAction("水平加總（短＋長剖面）")
        act_hsum.setEnabled(is_hsum)
        if not is_hsum:
            act_hsum.setToolTip("僅水平加總演算法批次結果可用")
        chosen = menu.exec_(
            self.btn_view_batch_results.mapToGlobal(
                self.btn_view_batch_results.rect().bottomLeft()
            )
        )
        if chosen is act_heatmap:
            self.open_batch_results_window()
        elif chosen is act_hsum:
            self.open_batch_hsum_window()

    def open_batch_results_window(self):
        store = getattr(self, "_batch_analysis_store", None)
        if not store or not store.get("results"):
            QMessageBox.information(
                self,
                "批次分析",
                "尚無批次分析資料，請先勾選「批次分析」並按 Apply。",
            )
            return
        dlg = getattr(self, "_batch_results_dialog", None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
        dlg = BatchAnalysisResultsDialog(store, parent=self)
        self._batch_results_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def open_batch_hsum_window(self):
        store = getattr(self, "_batch_analysis_store", None)
        if not store or not store.get("results"):
            QMessageBox.information(
                self,
                "批次分析",
                "尚無批次分析資料，請先勾選「批次分析」並按 Apply。",
            )
            return
        if not BaslerBatchDualTab._is_hsum_algorithm_id(store.get("algorithm")):
            QMessageBox.information(
                self,
                "提示",
                "水平加總檢視僅在演算法為「水平加總」時可用。",
            )
            return
        dlg = getattr(self, "_batch_hsum_dialog", None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
        dlg = BatchHSumResultsDialog(store, parent=self)
        self._batch_hsum_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

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
        self._update_pair_info()
        self._try_load_both_files()

    def load_long_file(self):
        path = self._pick_data_file("選擇長曝光檔案")
        if not path:
            return
        self.batch_long_path = path
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
        self._refresh_all_thresh_overlays()

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

    @staticmethod
    def _clip_color_scale_values(data, lo, hi):
        """color scale：< lo → 0（黑），> hi → hi（白），中間保留原值。"""
        if data is None or lo is None or hi is None:
            return data
        out = np.asarray(data, dtype=float).copy()
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return out
        if hi < lo:
            lo, hi = hi, lo
        out = np.where(out < lo, 0.0, out)
        out = np.where(out > hi, hi, out)
        return out

    def _short_analysis_matrix(self):
        """短曝光抓光斑／門檻用矩陣：僅在 color scale 啟用時硬裁切。"""
        mat = getattr(self, "batch_result_matrix", None)
        if mat is None:
            return None
        mat = np.asarray(mat, dtype=float)
        if not self._basler_bw_scale_enabled():
            return mat
        spin_lo = getattr(self, "spin_basler_black_below", None)
        spin_hi = getattr(self, "spin_basler_white_above", None)
        if spin_lo is None or spin_hi is None:
            return mat
        return self._clip_color_scale_values(
            mat, float(spin_lo.value()), float(spin_hi.value())
        )

    def _long_analysis_matrix(self):
        """長曝光抓光斑／門檻用矩陣：僅在 color scale 啟用時硬裁切。"""
        mat = self.batch_long_matrix
        if mat is None:
            return None
        mat = np.asarray(mat, dtype=float)
        if not self._long_bw_scale_enabled():
            return mat
        spin_lo = getattr(self, "spin_long_black_below", None)
        spin_hi = getattr(self, "spin_long_white_above", None)
        if spin_lo is None or spin_hi is None:
            return mat
        return self._clip_color_scale_values(
            mat, float(spin_lo.value()), float(spin_hi.value())
        )

    def _short_display_matrix(self):
        data = self._transform_display_matrix(
            getattr(self, "batch_result_matrix", None),
            self._basler_log_enabled(),
            self._basler_smooth_enabled(),
            self._basler_smooth_size(),
        )
        if data is None or not self._basler_bw_scale_enabled():
            return data
        lo, hi = self._basler_bw_thresholds_display()
        return self._clip_color_scale_values(data, lo, hi)

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

    def _basler_bw_thresholds_display(self):
        """回傳顯示座標下的 (black_floor, white_ceil)；spin 恆為原始強度。"""
        if not self._basler_bw_scale_enabled():
            return None, None
        black = float(self.spin_basler_black_below.value())
        white = float(self.spin_basler_white_above.value())
        if self._basler_log_enabled():
            black = float(np.log10(max(black, 1.0)))
            white = float(np.log10(max(white, 1.0)))
        return black, white

    def _long_bw_thresholds_display(self):
        if not self._long_bw_scale_enabled():
            return None, None
        return (
            self._to_long_display_threshold(self.spin_long_black_below.value()),
            self._to_long_display_threshold(self.spin_long_white_above.value()),
        )

    def _long_display_matrix(self):
        data = self._transform_display_matrix(
            self.batch_long_matrix,
            self._long_log_enabled(),
            self._long_smooth_enabled(),
            self._long_smooth_size(),
        )
        if data is None or not self._long_bw_scale_enabled():
            return data
        lo, hi = self._long_bw_thresholds_display()
        return self._clip_color_scale_values(data, lo, hi)

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
        force_unit=False,
    ):
        """單區顯示變換；color scale：<lo→0、>hi→hi。"""
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
            return self._clip_color_scale_values(data, lo, hi)
        # force_unit 保留參數相容；不再做 0–1 正規化
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

        short_disp = self._normalize_region_display(
            short,
            self._collage_short_log_enabled(),
            self._collage_short_smooth_enabled(),
            self._collage_short_smooth_size(),
            self._collage_short_bw_scale_enabled(),
            getattr(self, "spin_collage_short_black_below", None),
            getattr(self, "spin_collage_short_white_above", None),
        )
        long_disp = self._normalize_region_display(
            long,
            self._collage_long_log_enabled(),
            self._collage_long_smooth_enabled(),
            self._collage_long_smooth_size(),
            self._collage_long_bw_scale_enabled(),
            getattr(self, "spin_collage_long_black_below", None),
            getattr(self, "spin_collage_long_white_above", None),
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

    def _set_panel_image(
        self, panel, display, reset_levels, intensity_label, bw_hi=None, bw_lo=None
    ):
        if panel is None:
            return
        if display is None:
            panel.set_image(None, reset_view=True)
            return
        levels = None
        if reset_levels:
            if (
                bw_hi is not None
                and bw_lo is not None
                and np.isfinite(bw_lo)
                and np.isfinite(bw_hi)
            ):
                lo = float(bw_lo)
                hi = float(bw_hi)
                if hi < lo:
                    lo, hi = hi, lo
                if hi <= lo:
                    hi = lo + 1.0
                levels = (lo, hi)
            elif bw_hi is not None and np.isfinite(bw_hi) and float(bw_hi) > 0:
                levels = (0.0, float(bw_hi))
            else:
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
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            vmax = vmin + 1.0
        base = self.batch_jet_map
        if bw_enabled:
            # 不鎖死 levels（可拖曳調整）；僅讓目前範圍低端黑、高端白
            span = max(vmax - vmin, 1e-12)
            cmap = _build_bw_clip_colormap(
                base,
                vmin,
                vmax,
                black_floor=vmin + span * 1e-9,
                white_ceil=vmax - span * 1e-9,
            )
        else:
            cmap = base
        setattr(self, guard_attr, True)
        try:
            panel.hist.gradient.setColorMap(cmap)
        finally:
            setattr(self, guard_attr, False)

    def _refresh_short_heatmap_display(self, reset_levels=True):
        display = self._short_display_matrix()
        self._update_short_heatmap_title()
        bw_lo = bw_hi = None
        if self._basler_bw_scale_enabled():
            bw_lo, bw_hi = self._basler_bw_thresholds_display()
        self._set_panel_image(
            getattr(self, "short_heatmap_panel", None),
            display,
            reset_levels,
            self._intensity_label(
                self._basler_log_enabled(),
                self._basler_smooth_enabled(),
                self._basler_smooth_size(),
            ),
            bw_hi=bw_hi,
            bw_lo=bw_lo,
        )
        self._apply_short_colorbar_floor()
        self._redraw_spot_overlays()
        self._refresh_short_thresh_overlay()
        if (
            getattr(self, "chk_beam_size_enable", None) is not None
            and self.chk_beam_size_enable.isChecked()
        ):
            self._compute_and_show_beam_size()
        else:
            self._redraw_beam_size_overlays()

    def _refresh_long_heatmap_display(self, reset_levels=True):
        display = self._long_display_matrix()
        self._update_long_heatmap_title()
        bw_lo = bw_hi = None
        if self._long_bw_scale_enabled():
            bw_lo, bw_hi = self._long_bw_thresholds_display()
        self._set_panel_image(
            getattr(self, "long_heatmap_panel", None),
            display,
            reset_levels,
            self._intensity_label(
                self._long_log_enabled(),
                self._long_smooth_enabled(),
                self._long_smooth_size(),
            ),
            bw_hi=bw_hi,
            bw_lo=bw_lo,
        )
        self._apply_long_colorbar_floor()
        self._redraw_spot_overlays()
        self._refresh_long_thresh_overlay()
        if (
            getattr(self, "chk_beam_size_enable", None) is not None
            and self.chk_beam_size_enable.isChecked()
        ):
            self._compute_and_show_beam_size()
        else:
            self._redraw_beam_size_overlays()

    def _refresh_collage_heatmap_display(self, reset_levels=True):
        self._build_collage_matrix()
        self._update_collage_roi_info_label()
        panel = getattr(self, "collage_heatmap_panel", None)
        if panel is None:
            return
        display = self._collage_display_matrix()
        self._update_collage_heatmap_title()
        bw_on = (
            self._collage_short_bw_scale_enabled()
            or self._collage_long_bw_scale_enabled()
        )
        if bw_on:
            intensity = "Value (clip)"
        elif self._collage_short_log_enabled() or self._collage_long_log_enabled():
            intensity = "log10 / Value"
        else:
            intensity = "Value"
        bw_lo = bw_hi = None
        if bw_on:
            los = []
            his = []
            if self._collage_short_bw_scale_enabled():
                lo, hi = self._region_display_thresholds(
                    self._collage_short_log_enabled(),
                    True,
                    self.spin_collage_short_black_below,
                    self.spin_collage_short_white_above,
                )
                if lo is not None:
                    los.append(lo)
                if hi is not None:
                    his.append(hi)
            if self._collage_long_bw_scale_enabled():
                lo, hi = self._region_display_thresholds(
                    self._collage_long_log_enabled(),
                    True,
                    self.spin_collage_long_black_below,
                    self.spin_collage_long_white_above,
                )
                if lo is not None:
                    los.append(lo)
                if hi is not None:
                    his.append(hi)
            if los and his:
                bw_lo = min(los)
                bw_hi = max(his)
        self._set_panel_image(
            panel, display, reset_levels, intensity, bw_hi=bw_hi, bw_lo=bw_lo
        )
        self._apply_collage_colorbar_floor(bw_enabled=bw_on, bw_hi=bw_hi)

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

    def _apply_collage_colorbar_floor(self, bw_enabled=None, bw_hi=None):
        # 拼貼：各區已硬裁切；色階低端黑、高端白
        if bw_enabled is None:
            bw_enabled = (
                self._collage_short_bw_scale_enabled()
                or self._collage_long_bw_scale_enabled()
            )
        if bw_enabled and bw_hi is None:
            his = []
            for log_on, enabled, black_spin, white_spin in (
                (
                    self._collage_short_log_enabled(),
                    self._collage_short_bw_scale_enabled(),
                    getattr(self, "spin_collage_short_black_below", None),
                    getattr(self, "spin_collage_short_white_above", None),
                ),
                (
                    self._collage_long_log_enabled(),
                    self._collage_long_bw_scale_enabled(),
                    getattr(self, "spin_collage_long_black_below", None),
                    getattr(self, "spin_collage_long_white_above", None),
                ),
            ):
                if not enabled:
                    continue
                _, hi = self._region_display_thresholds(
                    log_on, True, black_spin, white_spin
                )
                if hi is not None:
                    his.append(hi)
            if his:
                bw_hi = max(his)
        white = float(bw_hi) if bw_hi is not None else None
        self._apply_panel_colorbar(
            getattr(self, "collage_heatmap_panel", None),
            "_collage_cmap_guard",
            bool(bw_enabled),
            (0.0, white) if bw_enabled else (None, None),
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
        self._sync_checkbox(getattr(self, "chk_hsum_short_bw", None), checked)
        for spin in (
            getattr(self, "spin_hsum_short_black", None),
            getattr(self, "spin_hsum_short_white", None),
        ):
            if spin is not None:
                spin.setEnabled(enabled)
        self._refresh_short_heatmap_display(reset_levels=True)
        if self._current_spot_algo() == "hsum":
            self._update_hsum_focus_plots(recompute_peaks=True)

    def _on_basler_bw_threshold_changed(self, *_args):
        if not self._basler_bw_scale_enabled():
            return
        self._sync_spin(
            getattr(self, "spin_hsum_short_black", None),
            self.spin_basler_black_below.value(),
        )
        self._sync_spin(
            getattr(self, "spin_hsum_short_white", None),
            self.spin_basler_white_above.value(),
        )
        self._refresh_short_heatmap_display(reset_levels=True)
        if self._current_spot_algo() == "hsum":
            self._update_hsum_focus_plots(recompute_peaks=True)

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
        self._sync_checkbox(getattr(self, "chk_hsum_long_bw", None), checked)
        for spin in (
            getattr(self, "spin_hsum_long_black", None),
            getattr(self, "spin_hsum_long_white", None),
        ):
            if spin is not None:
                spin.setEnabled(enabled)
        self._refresh_long_heatmap_display(reset_levels=True)
        if self._current_spot_algo() == "hsum":
            self._update_hsum_focus_plots(recompute_peaks=True)

    def _on_long_bw_threshold_changed(self, *_args):
        if not self._long_bw_scale_enabled():
            return
        self._sync_spin(
            getattr(self, "spin_hsum_long_black", None),
            self.spin_long_black_below.value(),
        )
        self._sync_spin(
            getattr(self, "spin_hsum_long_white", None),
            self.spin_long_white_above.value(),
        )
        self._refresh_long_heatmap_display(reset_levels=True)
        if self._current_spot_algo() == "hsum":
            self._update_hsum_focus_plots(recompute_peaks=True)

    def _on_long_display_option_changed(self, *_args):
        self._refresh_long_heatmap_display(reset_levels=True)

    def _on_long_log_toggled(self, checked=False):
        self._sync_checkbox(getattr(self, "chk_hsum_long_log", None), checked)
        self._sync_hsum_log_to_plot("long", checked)
        self._refresh_long_heatmap_display(reset_levels=True)
        if self._current_spot_algo() == "hsum":
            self._update_hsum_focus_plots()

    def _on_short_log_toggled(self, checked=False):
        self._sync_checkbox(getattr(self, "chk_hsum_short_log", None), checked)
        self._sync_hsum_log_to_plot("short", checked)
        self._refresh_short_heatmap_display(reset_levels=True)
        if self._current_spot_algo() == "hsum":
            self._update_hsum_focus_plots()

    def _on_collage_display_option_changed(self, *_args):
        self._refresh_collage_heatmap_display(reset_levels=True)

    def _on_collage_short_log_toggled(self, checked=False):
        self._refresh_collage_heatmap_display(reset_levels=True)

    def _on_collage_long_log_toggled(self, checked=False):
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
        algo = self._current_spot_algo()
        if algo == "axial_1d" and getattr(self, "spin_axial_short_thresh", None):
            return float(self.spin_axial_short_thresh.value())
        if algo == "hsum" and getattr(self, "spin_hsum_short_thresh", None):
            return float(self.spin_hsum_short_thresh.value())
        if algo == "gaussian" and getattr(self, "spin_gauss_short_thresh", None):
            return float(self.spin_gauss_short_thresh.value())
        spin = getattr(self, "spin_short_spot_thresh", None)
        if spin is None:
            return 50.0
        return float(spin.value())

    def _long_spot_thresh_percent(self):
        # 一維／水平加總／高斯沒有獨立長曝光門檻時，沿用短曝光門檻
        if self._current_spot_algo() in ("axial_1d", "hsum", "gaussian"):
            return self._short_spot_thresh_percent()
        spin = getattr(self, "spin_long_spot_thresh", None)
        if spin is None:
            return 50.0
        return float(spin.value())

    def _build_full_threshold_mask(self, matrix, thresh_percent, roi=None):
        """建立與抓光斑相同的門檻 mask（可限 ROI），回傳與 matrix 同尺寸 bool。"""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0 or not np.isfinite(matrix).any():
            return None
        h, w = matrix.shape
        full = np.zeros((h, w), dtype=bool)
        work = matrix
        x0 = y0 = 0
        if roi is not None:
            clipped = clip_roi_to_matrix(matrix, roi)
            if clipped is None:
                return full
            x0, y0, x1, y1 = clipped
            work = matrix[y0:y1, x0:x1]
            if work.size == 0:
                return full
        mask = build_robust_threshold_mask(
            work,
            use_threshold=True,
            thresh_percent=float(thresh_percent),
            bg_subtract=True,
            largest_cc_only=True,
        )
        if mask is None:
            return full
        if roi is None:
            return np.asarray(mask, dtype=bool)
        full[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
        return full

    def _on_short_thresh_overlay_toggled(self, checked):
        btn = getattr(self, "btn_short_show_thresh", None)
        if btn is not None:
            btn.setText("隱藏門檻" if checked else "顯示門檻")
        self._refresh_short_thresh_overlay()

    def _on_long_thresh_overlay_toggled(self, checked):
        btn = getattr(self, "btn_long_show_thresh", None)
        if btn is not None:
            btn.setText("隱藏門檻" if checked else "顯示門檻")
        self._refresh_long_thresh_overlay()

    def _refresh_short_thresh_overlay(self, *_args):
        panel = getattr(self, "short_heatmap_panel", None)
        if panel is None:
            return
        btn = getattr(self, "btn_short_show_thresh", None)
        show = bool(btn is not None and btn.isChecked())
        mat = self._short_analysis_matrix()
        if (not show) or mat is None:
            panel.clear_threshold_overlay()
            return
        mask = self._build_full_threshold_mask(
            mat, self._short_spot_thresh_percent(), roi=None
        )
        panel.set_threshold_overlay(
            mask, visible=True, rgba_color=(255, 64, 255, 90)
        )

    def _refresh_long_thresh_overlay(self, *_args):
        panel = getattr(self, "long_heatmap_panel", None)
        if panel is None:
            return
        btn = getattr(self, "btn_long_show_thresh", None)
        show = bool(btn is not None and btn.isChecked())
        mat = self._long_analysis_matrix()
        if (not show) or mat is None:
            panel.clear_threshold_overlay()
            return
        mask = self._build_full_threshold_mask(
            mat, self._long_spot_thresh_percent(), roi=self._long_beam_size_roi()
        )
        panel.set_threshold_overlay(
            mask, visible=True, rgba_color=(255, 152, 0, 90)
        )

    def _refresh_all_thresh_overlays(self):
        self._refresh_short_thresh_overlay()
        self._refresh_long_thresh_overlay()

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
        matrix = self._short_analysis_matrix()
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
        matrix = self._long_analysis_matrix()
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

        1. 依目前短曝光門檻重抓短曝光光斑（質心／門檻）
        2. 以短曝光對應點為中心，長曝光框 寬 160 × 長 512
        3. 條帶水平加總 → 1D 剖面 → 微分找雙峰
        4. 較遠峰作為長曝光第二光斑
        """
        if getattr(self, "batch_result_matrix", None) is None:
            QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
            return
        if self.batch_long_matrix is None:
            QMessageBox.warning(self, "警告", "請先載入長曝光檔案。")
            return

        # 每次 Apply 都依目前短曝光門檻重抓（勿沿用舊 short_spot）
        self.grab_short_spot()
        if self.short_spot is None:
            return

        center = self._map_short_spot_to_long()
        if center is None:
            QMessageBox.warning(self, "警告", "無法將短曝光光斑對應到長曝光。")
            return

        roi_w = (
            int(self.spin_axial_roi_w.value())
            if getattr(self, "spin_axial_roi_w", None)
            else 160
        )
        roi_h = (
            int(self.spin_axial_roi_h.value())
            if getattr(self, "spin_axial_roi_h", None)
            else 512
        )
        smooth_win = (
            int(self.spin_axial_smooth.value())
            if getattr(self, "spin_axial_smooth", None)
            else 7
        )
        min_sep = (
            int(self.spin_axial_min_sep.value())
            if getattr(self, "spin_axial_min_sep", None)
            else 12
        )
        result = find_dual_spots_axial_1d_projection(
            self._long_analysis_matrix(),
            center,
            roi_w=roi_w,
            roi_h=roi_h,
            smooth_win=smooth_win,
            min_peak_distance=min_sep,
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
                self._set_axial_display_roi(result["roi_xywh"])
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
                self._set_axial_display_roi(result["roi_xywh"])
                self._redraw_spot_overlays()
            return

        # 條帶框只更新一維投影專用顯示，不覆寫質心 ROI
        self._set_axial_display_roi(result.get("roi_xywh"))

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

    def grab_dual_spots_hsum(self):
        """水平加總雙峰：ROI 水平加總波形找兩峰；平滑／精修只改波形。

        1. 短曝光門檻定位
        2. 長曝光條帶水平加總
        3. 主峰＝平滑前最高；次峰＝搜尋窗內平滑後最高
        4. 精修＝兩峰峰頂二次 + 分段二次 after refine 波形；heatmap 不被改寫
        """
        if getattr(self, "batch_result_matrix", None) is None:
            QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
            return
        if self.batch_long_matrix is None:
            QMessageBox.warning(self, "警告", "請先載入長曝光檔案。")
            return

        self.grab_short_spot()
        if self.short_spot is None:
            return

        center = self._map_short_spot_to_long()
        if center is None:
            QMessageBox.warning(self, "警告", "無法將短曝光光斑對應到長曝光。")
            return

        hkw = self._hsum_algo_kwargs()
        result = find_dual_spots_horizontal_sum(
            self._long_analysis_matrix(),
            center,
            **hkw,
        )
        spots = result.get("spots")
        if not spots or len(spots) < 2:
            QMessageBox.warning(
                self,
                "警告",
                "水平加總未能在條帶波形內找到兩個峰。\n"
                "請調整次峰窗／平滑窗，或確認短曝光定位。",
            )
            if result.get("roi_xywh") is not None:
                self._set_hsum_display_roi(result["roi_xywh"])
                self._redraw_spot_overlays()
            return

        if abs(float(spots[0][1]) - float(spots[1][1])) < 2.0:
            QMessageBox.warning(self, "警告", "水平加總雙峰未分開。")
            if result.get("roi_xywh") is not None:
                self._set_hsum_display_roi(result["roi_xywh"])
                self._redraw_spot_overlays()
            return

        self._set_hsum_display_roi(result.get("roi_xywh"))
        win = result.get("secondary_window_y")
        if win is not None:
            self._hsum_sec_window = (float(win[0]), float(win[1]))
            # 回寫自動窗到 UI（僅當原本為 0／0 自動模式）
            if hkw.get("secondary_window_y") is None:
                for spin, val in (
                    (self.spin_hsum_sec_lo, float(win[0])),
                    (self.spin_hsum_sec_hi, float(win[1])),
                ):
                    spin.blockSignals(True)
                    if val > spin.maximum():
                        spin.setMaximum(max(val * 1.2, spin.maximum()))
                    spin.setValue(val)
                    spin.blockSignals(False)

        primary_i = int(result.get("primary_index", 0))
        secondary_i = 1 - primary_i
        spot2 = spots[secondary_i]
        # 主峰座標也更新 short 顯示用（短曝光仍用門檻點）；長曝光第二峰用波形次峰
        self.long_spot2 = (float(spot2[0]), float(spot2[1]))
        self.lbl_long_spot_pos.setText(
            f"長曝光第二光斑: X={self.long_spot2[0]:.2f}, "
            f"Y={self.long_spot2[1]:.2f}（水平加總）"
        )
        self._update_spot_delta_label()
        self._redraw_spot_overlays()
        self._update_hsum_focus_plots(recompute_peaks=False)

        y_lo, y_hi = result.get("peak_ys", (None, None))
        n_ref = int(result.get("refine_done") or 0)
        self.lbl_batch_status.setText(
            f"狀態: 水平加總｜雙峰 Y={y_lo:.2f}/{y_hi:.2f}｜"
            f"精修={n_ref}｜第二光斑 ({self.long_spot2[0]:.1f}, {self.long_spot2[1]:.1f})"
        )

    def grab_dual_spots_gaussian(self):
        """高斯濾波 2D 連通區抓下光斑。

        1. 依目前短曝光門檻重抓短曝光光斑（即上光斑）
        2. 長曝光：gaussian_filter(σ) → 以上光斑為錨抹鄰域 → 下方帶抓下光斑
        """
        if getattr(self, "batch_result_matrix", None) is None:
            QMessageBox.warning(self, "警告", "請先載入短曝光檔案。")
            return
        if self.batch_long_matrix is None:
            QMessageBox.warning(self, "警告", "請先載入長曝光檔案。")
            return

        self.grab_short_spot()
        if self.short_spot is None:
            return

        center = self._map_short_spot_to_long()
        if center is None:
            QMessageBox.warning(self, "警告", "無法將短曝光光斑對應到長曝光。")
            return

        result = find_dual_spots_gaussian(
            self._long_analysis_matrix(),
            center,
            **self._gaussian_algo_kwargs(),
        )
        if result.get("search_roi_xywh") is not None:
            self._set_gaussian_display_roi(result["search_roi_xywh"])

        secondary = result.get("secondary")
        if secondary is None:
            QMessageBox.warning(
                self,
                "警告",
                "高斯濾波未能找到下光斑。\n"
                f"原因：{result.get('reason', 'unknown')}\n"
                "可調大 σ、縮小主斑抹除半徑，或確認短曝光定位是否在軸上。",
            )
            self._redraw_spot_overlays()
            return

        self.long_spot2 = (float(secondary[0]), float(secondary[1]))
        self.lbl_long_spot_pos.setText(
            f"長曝光第二光斑: X={self.long_spot2[0]:.2f}, "
            f"Y={self.long_spot2[1]:.2f}（高斯濾波）"
        )
        self._update_spot_delta_label()
        self._redraw_spot_overlays()

        sigma = self._gaussian_algo_kwargs()["sigma"]
        self.lbl_batch_status.setText(
            f"狀態: 高斯濾波 σ={sigma:g}｜"
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
        """實際距離：μm = px * 3.45。"""
        return float(px) * 3.45

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
        """暫時清除十字、搜尋範圍與門檻分布，供匯出乾淨圖檔。"""
        short_panel = getattr(self, "short_heatmap_panel", None)
        long_panel = getattr(self, "long_heatmap_panel", None)
        if short_panel is not None:
            self._clear_plot_items(short_panel.plot, "_short_cross_items")
            self._clear_plot_items(short_panel.plot, "_short_beam_circle_items")
            short_panel.clear_threshold_overlay()
        if long_panel is not None:
            self._clear_plot_items(long_panel.plot, "_long_cross_items")
            self._clear_plot_items(long_panel.plot, "_long_beam_circle_items")
            long_panel.clear_threshold_overlay()

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
                algo = self._current_spot_algo()
                if algo == "axial_1d":
                    axial_roi = getattr(self, "_axial_roi_xywh", None)
                    if axial_roi is not None:
                        bounds = clip_roi_to_matrix(
                            self.batch_long_matrix, axial_roi
                        )
                        if bounds is not None:
                            self._add_roi_rect(
                                long_panel.plot,
                                "_long_cross_items",
                                bounds,
                                pg.mkPen("#EF6C00", width=2.0, style=Qt.DashLine),
                            )
                elif algo == "hsum":
                    hsum_roi = getattr(self, "_hsum_roi_xywh", None)
                    if hsum_roi is not None:
                        bounds = clip_roi_to_matrix(
                            self.batch_long_matrix, hsum_roi
                        )
                        if bounds is not None:
                            self._add_roi_rect(
                                long_panel.plot,
                                "_long_cross_items",
                                bounds,
                                pg.mkPen("#EF6C00", width=2.0, style=Qt.DashLine),
                            )
                elif algo == "gaussian":
                    g_roi = getattr(self, "_gaussian_roi_xywh", None)
                    if g_roi is not None:
                        bounds = clip_roi_to_matrix(
                            self.batch_long_matrix, g_roi
                        )
                        if bounds is not None:
                            self._add_roi_rect(
                                long_panel.plot,
                                "_long_cross_items",
                                bounds,
                                pg.mkPen("#EF6C00", width=2.0, style=Qt.DashLine),
                            )
                elif (
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
                # 水平加總：Heatmap 只標 ROI＋上光斑，不下光斑
                if algo != "hsum" and self.long_spot2 is not None:
                    self._add_cross(
                        long_panel.plot,
                        "_long_cross_items",
                        self.long_spot2[0],
                        self.long_spot2[1],
                        self.batch_long_matrix.shape,
                        pg.mkPen("#D50000", width=2.5),
                    )

        # 光斑十字重畫後，若 Beam size 啟用則一併重畫內切圓
        self._redraw_beam_size_overlays()

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
        self._refresh_long_thresh_overlay()
        if (
            getattr(self, "chk_beam_size_enable", None) is not None
            and self.chk_beam_size_enable.isChecked()
        ):
            self._compute_and_show_beam_size()

    def _on_long_roi_spin_changed(self, *_args):
        self._redraw_spot_overlays()
        self._build_collage_matrix()
        if getattr(self, "collage_heatmap_panel", None) is not None:
            self._refresh_collage_heatmap_display(reset_levels=False)
        self._refresh_long_thresh_overlay()
        if (
            getattr(self, "chk_beam_size_enable", None) is not None
            and self.chk_beam_size_enable.isChecked()
        ):
            self._compute_and_show_beam_size()

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
                "distance_scale": 1.0,
                "um_formula": "|d_px| * 3.45",
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
            "distance_scale": 1.0,
            "um_formula": "|d_px| * 3.45",
        }

    def _collect_algorithm_export_info(self):
        """匯出用：目前演算法 id、說明與參數。"""
        algo = self._current_spot_algo()
        if algo == "axial_1d":
            params = {
                "short_thresh_percent": float(
                    getattr(self, "spin_axial_short_thresh", None).value()
                    if getattr(self, "spin_axial_short_thresh", None)
                    else self._short_spot_thresh_percent()
                ),
                "strip_roi_w": int(
                    getattr(self, "spin_axial_roi_w", None).value()
                    if getattr(self, "spin_axial_roi_w", None)
                    else 160
                ),
                "strip_roi_h": int(
                    getattr(self, "spin_axial_roi_h", None).value()
                    if getattr(self, "spin_axial_roi_h", None)
                    else 512
                ),
                "profile_smooth_win": int(
                    getattr(self, "spin_axial_smooth", None).value()
                    if getattr(self, "spin_axial_smooth", None)
                    else 7
                ),
                "min_peak_distance_px": int(
                    getattr(self, "spin_axial_min_sep", None).value()
                    if getattr(self, "spin_axial_min_sep", None)
                    else 12
                ),
                "display_roi_xywh": (
                    list(self._axial_roi_xywh)
                    if getattr(self, "_axial_roi_xywh", None) is not None
                    else None
                ),
            }
            return {
                "algorithm": "axial_1d_projection_diff",
                "algorithm_desc": "一維投影＋微分：短曝光定位 → 長曝光條帶水平加總找雙峰",
                "algorithm_params": params,
            }

        if algo == "hsum":
            hkw = self._hsum_algo_kwargs()
            sec = hkw.get("secondary_window_y")
            params = {
                "short_thresh_percent": float(self._short_spot_thresh_percent()),
                "strip_roi_w": int(hkw["roi_w"]),
                "strip_roi_h": int(hkw["roi_h"]),
                "profile_smooth_win": int(hkw["smooth_win"]),
                "min_peak_distance_px": int(hkw["min_peak_distance"]),
                "refine_iters": int(hkw["n_refine"]),
                "secondary_search_window_y": (
                    list(sec) if sec is not None else (
                        list(self._hsum_sec_window)
                        if getattr(self, "_hsum_sec_window", None) is not None
                        else None
                    )
                ),
                "display_roi_xywh": (
                    list(self._hsum_roi_xywh)
                    if getattr(self, "_hsum_roi_xywh", None) is not None
                    else None
                ),
            }
            return {
                "algorithm": "horizontal_sum_dual_peak",
                "algorithm_desc": (
                    "水平加總：短曝光定位 → ROI 水平加總波形抓雙峰；"
                    "平滑／精修只改波形，不改 heatmap"
                ),
                "algorithm_params": params,
            }

        if algo == "gaussian":
            gkw = self._gaussian_algo_kwargs()
            params = {
                "short_thresh_percent": float(self._short_spot_thresh_percent()),
                "gaussian_sigma_px": float(gkw["sigma"]),
                "secondary_thresh_frac": float(gkw["thresh_frac"]),
                "search_half_x_px": int(gkw["search_half_x"]),
                "min_sep_y_px": int(gkw["min_sep_y"]),
                "primary_blank_r_px": float(gkw["primary_blank_r"]),
                "search_roi_xywh": (
                    list(self._gaussian_roi_xywh)
                    if getattr(self, "_gaussian_roi_xywh", None) is not None
                    else None
                ),
            }
            return {
                "algorithm": "gaussian_2d_cc",
                "algorithm_desc": (
                    "高斯濾波：上光斑＝短曝光 → 長曝光 2D 平滑 → "
                    "抹主斑鄰域 → 下方帶最大連通區質心"
                ),
                "algorithm_params": params,
            }

        roi_enabled = bool(
            getattr(self, "chk_long_search_roi", None)
            and self.chk_long_search_roi.isChecked()
        )
        params = {
            "short_thresh_percent": float(self._short_spot_thresh_percent()),
            "long_thresh_percent": float(self._long_spot_thresh_percent()),
            "long_search_roi_enabled": roi_enabled,
            "long_search_roi_xywh": list(self._long_roi_tuple()),
        }
        return {
            "algorithm": "centroid_threshold_geom",
            "algorithm_desc": "質心／門檻幾何：背景扣除＋最大連通區＋亞像素",
            "algorithm_params": params,
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
        algo_info = self._collect_algorithm_export_info()
        roi_enabled = bool(
            getattr(self, "chk_long_search_roi", None)
            and self.chk_long_search_roi.isChecked()
        )
        payload = {
            "mode": "basler_short_long_dual",
            "algorithm": algo_info["algorithm"],
            "algorithm_desc": algo_info["algorithm_desc"],
            "algorithm_params": algo_info["algorithm_params"],
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
        if algo_info.get("algorithm") == "axial_1d_projection_diff" and self.short_spot is not None:
            payload["horizontal_sum"] = self._compute_hsum_profiles_for_mats(
                self._short_analysis_matrix(),
                self._long_analysis_matrix(),
                self.short_spot,
                strip_params=self._axial_strip_params(),
                long_center=self._map_short_spot_to_long(),
            )
        if (
            BaslerBatchDualTab._is_hsum_algorithm_id(algo_info.get("algorithm"))
            and self.short_spot is not None
        ):
            hkw = self._hsum_algo_kwargs()
            payload["horizontal_sum"] = self._compute_hsum_profiles_for_mats(
                self._short_analysis_matrix(),
                self._long_analysis_matrix(),
                self.short_spot,
                strip_params=self._active_hsum_strip_params(),
                long_center=self._map_short_spot_to_long(),
                n_refine=int(hkw.get("n_refine", 0) or 0),
                secondary_window_y=hkw.get("secondary_window_y"),
            )
        return payload

    def _export_payload_rows(self, payload):
        rows = [
            ["Mode", payload["mode"], ""],
            ["Algorithm Id", payload.get("algorithm"), ""],
            ["Algorithm", payload["algorithm_desc"], ""],
            ["Short File", payload["short_file"], ""],
            ["Long File", payload["long_file"], ""],
        ]
        params = payload.get("algorithm_params") or {}
        algo = payload.get("algorithm")
        if algo == "axial_1d_projection_diff":
            rows.extend(
                [
                    ["Param Short Threshold", params.get("short_thresh_percent"), "%"],
                    ["Param Strip ROI W", params.get("strip_roi_w"), "px"],
                    ["Param Strip ROI H", params.get("strip_roi_h"), "px"],
                    ["Param Profile Smooth Win", params.get("profile_smooth_win"), ""],
                    ["Param Min Peak Distance", params.get("min_peak_distance_px"), "px"],
                ]
            )
            disp_roi = params.get("display_roi_xywh")
            if disp_roi:
                rows.extend(
                    [
                        ["Param Applied Strip X", disp_roi[0], "px"],
                        ["Param Applied Strip Y", disp_roi[1], "px"],
                        ["Param Applied Strip W", disp_roi[2], "px"],
                        ["Param Applied Strip H", disp_roi[3], "px"],
                    ]
                )
        elif algo == "horizontal_sum_dual_peak":
            rows.extend(
                [
                    ["Param Short Threshold", params.get("short_thresh_percent"), "%"],
                    ["Param Strip ROI W", params.get("strip_roi_w"), "px"],
                    ["Param Strip ROI H", params.get("strip_roi_h"), "px"],
                    ["Param Profile Smooth Win", params.get("profile_smooth_win"), ""],
                    ["Param Min Peak Distance", params.get("min_peak_distance_px"), "px"],
                    ["Param Refine Iters", params.get("refine_iters"), ""],
                ]
            )
            sec = params.get("secondary_search_window_y")
            if sec and len(sec) >= 2:
                rows.extend(
                    [
                        ["Param Sec Window Y Lo", sec[0], "px"],
                        ["Param Sec Window Y Hi", sec[1], "px"],
                    ]
                )
        elif algo == "gaussian_2d_cc":
            rows.extend(
                [
                    ["Param Short Threshold", params.get("short_thresh_percent"), "%"],
                    ["Param Gaussian Sigma", params.get("gaussian_sigma_px"), "px"],
                    [
                        "Param Secondary Thresh Frac",
                        params.get("secondary_thresh_frac"),
                        "",
                    ],
                    ["Param Search Half X", params.get("search_half_x_px"), "px"],
                    ["Param Min Sep Y", params.get("min_sep_y_px"), "px"],
                    ["Param Primary Blank R", params.get("primary_blank_r_px"), "px"],
                ]
            )
            g_roi = params.get("search_roi_xywh")
            if g_roi:
                rows.extend(
                    [
                        ["Param Search ROI X", g_roi[0], "px"],
                        ["Param Search ROI Y", g_roi[1], "px"],
                        ["Param Search ROI W", g_roi[2], "px"],
                        ["Param Search ROI H", g_roi[3], "px"],
                    ]
                )
        else:
            rows.extend(
                [
                    ["Param Short Threshold", params.get("short_thresh_percent"), "%"],
                    ["Param Long Threshold", params.get("long_thresh_percent"), "%"],
                    [
                        "Param Long Search ROI Enabled",
                        "Yes" if params.get("long_search_roi_enabled") else "No",
                        "",
                    ],
                ]
            )
            roi = params.get("long_search_roi_xywh")
            if roi:
                rows.extend(
                    [
                        ["Param Long Search ROI X", roi[0], "px"],
                        ["Param Long Search ROI Y", roi[1], "px"],
                        ["Param Long Search ROI W", roi[2], "px"],
                        ["Param Long Search ROI H", roi[3], "px"],
                    ]
                )

        # 相容舊欄位（質心 ROI 摘要）
        rows.extend(
            [
                ["Short Threshold", payload["short_thresh_percent"], "%"],
                ["Long Threshold", payload["long_thresh_percent"], "%"],
                [
                    "Long Search ROI Enabled",
                    "Yes" if payload["long_search_roi_enabled"] else "No",
                    "",
                ],
            ]
        )
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
        hsum = payload.get("horizontal_sum") or {}
        for which, key in (("Short", "short"), ("Long", "long")):
            block = hsum.get(key) or {}
            if not block:
                continue
            rows.extend(
                [
                    [f"HSum {which} N Points", block.get("n_points"), ""],
                    [f"HSum {which} Primary Y", block.get("primary_y_px"), "px"],
                    [f"HSum {which} Secondary Y", block.get("secondary_y_px"), "px"],
                    [f"HSum {which} Peak1 Y", block.get("peak1_y_px"), "px"],
                    [f"HSum {which} Peak2 Y", block.get("peak2_y_px"), "px"],
                    [f"HSum {which} Peak Delta Y", block.get("peak_delta_y_px"), "px"],
                    [f"HSum {which} Peak Delta Y", block.get("peak_delta_y_um"), "μm"],
                    [f"HSum {which} Smooth Win", block.get("smooth_win"), ""],
                ]
            )
            roi = block.get("roi_xywh")
            if roi and len(roi) == 4:
                rows.extend(
                    [
                        [f"HSum {which} ROI X", roi[0], "px"],
                        [f"HSum {which} ROI Y", roi[1], "px"],
                        [f"HSum {which} ROI W", roi[2], "px"],
                        [f"HSum {which} ROI H", roi[3], "px"],
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

        stamp = export_timestamp_tag()
        default_name = "Basler_ShortLong_Spot"
        short_stem = os.path.splitext(os.path.basename(self.batch_short_path or ""))[0]
        if short_stem:
            default_name = f"{short_stem}_ShortLong"
        initial = with_export_timestamp(default_name, stamp=stamp)
        if self.save_dir_path:
            initial = os.path.join(self.save_dir_path, initial)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出短／長曝光結果（JSON 基準檔名）",
            initial + ".json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        base_path, ext = os.path.splitext(path)
        if ext.lower() != ".json":
            base_path = os.path.splitext(path)[0]
        # 去掉使用者選檔時可能已帶的本次 stamp，後面各檔再統一加在副檔名前
        stamp_suffix = f"_{stamp}"
        if base_path.endswith(stamp_suffix):
            base_path = base_path[: -len(stamp_suffix)]
        path = with_export_timestamp(base_path + ".json", stamp=stamp)
        base_nostamp = os.path.splitext(path)[0]
        if base_nostamp.endswith(stamp_suffix):
            base_nostamp = base_nostamp[: -len(stamp_suffix)]

        try:
            payload = self._build_export_payload()
            payload["exported_at"] = stamp
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            csv_path = with_export_timestamp(
                f"{base_nostamp}_Spot_Data.csv", stamp=stamp
            )
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Item", "Value", "Unit"])
                for row in self._export_payload_rows(payload):
                    writer.writerow(row)
                self._write_hsum_csv_sections(writer, payload.get("horizontal_sum"))

            hsum_extra = []
            if payload.get("horizontal_sum"):
                hsum_extra = self._export_hsum_peaks_files(
                    base_nostamp, payload.get("horizontal_sum"), stamp
                )

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
                    short_base = f"{base_nostamp}_Short_Heatmap_With_Profiles"
                    outs = (
                        short_panel.export_plot_bundle(short_base, stamp=stamp)
                        or []
                    )
                    exported_imgs.extend(outs)
                if long_panel is not None:
                    long_base = f"{base_nostamp}_Long_Heatmap_With_Profiles"
                    outs = (
                        long_panel.export_plot_bundle(long_base, stamp=stamp) or []
                    )
                    exported_imgs.extend(outs)
                if collage_panel is not None and self.batch_collage_matrix is not None:
                    collage_base = f"{base_nostamp}_Collage_Heatmap_With_Profiles"
                    outs = (
                        collage_panel.export_plot_bundle(collage_base, stamp=stamp)
                        or []
                    )
                    exported_imgs.extend(outs)
            finally:
                self._redraw_spot_overlays()
                self._refresh_all_thresh_overlays()
                if not collage_was_visible:
                    collage_dlg.hide()

            msg = (
                f"已匯出：\n"
                f"JSON：{path}\n"
                f"CSV：{csv_path}\n"
            )
            if hsum_extra:
                msg += "水平加總雙峰：\n" + "\n".join(hsum_extra) + "\n"
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
            self._refresh_all_thresh_overlays()
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
