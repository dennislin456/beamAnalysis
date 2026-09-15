"""Basler JSON Batch：位置資料夾（L01…）＋單一 NPY Heatmap＋JSON 距離清單。"""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared_components import (
    EXPORT_IMAGE_EXT,
    InteractiveHeatmapPanel,
    export_timestamp_tag,
    normalize_export_image_path,
)

try:
    import orjson as _fast_json
except ImportError:  # pragma: no cover
    _fast_json = None


_JSON_FIELDS = ("x_mm", "y_mm", "r_deg", "exposure_ms", "distance_um")
_IMPORT_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 2))
_COORD_RE = re.compile(
    r"X(?P<x_sign>-?)(?P<x_int>\d+(?:p\d+)?)_Y(?P<y_sign>-?)(?P<y_int>\d+(?:p\d+)?)",
    re.IGNORECASE,
)
_XY_TOL = 1e-6


def _natural_sort_key(name):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(name))]


def _parse_xy_from_name(name):
    """從檔名解析 X/Y（如 X120p00_Y60p00 → 120.0, 60.0）。"""
    match = _COORD_RE.search(os.path.basename(name))
    if match is None:
        return None

    def parse_part(sign, value):
        number = float(value.replace("p", ".").replace("P", "."))
        return -number if sign == "-" else number

    return (
        parse_part(match.group("x_sign"), match.group("x_int")),
        parse_part(match.group("y_sign"), match.group("y_int")),
    )


def _xy_key(x, y):
    if x is None or y is None:
        return None
    try:
        return (round(float(x), 6), round(float(y), 6))
    except (TypeError, ValueError):
        return None


def _xy_match(a, b, tol=_XY_TOL):
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _load_json_obj(file_path):
    if _fast_json is not None:
        with open(file_path, "rb") as fh:
            return _fast_json.loads(fh.read())
    with open(file_path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _format_json_value(value):
    """null / 缺失顯示為 null；數值固定小數點三位。"""
    if value is None:
        return "null"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "null"
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):.3f}"
    return str(value)


def _format_xy_label(x, y):
    return f"X={_format_json_value(x)} Y={_format_json_value(y)}"


def _pick_json_field(data, key):
    """優先頂層欄位，其次 value 內同名欄位。"""
    if isinstance(data, dict) and key in data:
        return data.get(key)
    value = data.get("value") if isinstance(data, dict) else None
    if isinstance(value, dict) and key in value:
        return value.get(key)
    return None


def _read_one_distance_json(file_path):
    try:
        data = _load_json_obj(file_path)
        row = {"file": os.path.basename(file_path)}
        for key in _JSON_FIELDS:
            row[key] = _pick_json_field(data, key)
        return file_path, row, None
    except Exception as exc:
        return file_path, None, exc


class _NoFocusDelegate(QStyledItemDelegate):
    """去掉點位列表選取時的虛線／透明焦點框。"""

    def paint(self, painter, option, index):
        option.state &= ~QStyle.State_HasFocus
        super().paint(painter, option, index)


class LocationOnlyConfigDialog(QDialog):
    """僅勾選位置資料夾（每點一個 NPY，不需 cycle）。"""

    def __init__(self, locations, current_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("位置配置")
        self.resize(420, 480)
        self.setModal(True)
        self.locations = list(locations)
        self.current_config = current_config or {}
        self.loc_checks = {}

        root = QVBoxLayout(self)
        tip = QLabel(
            "勾選要載入的位置資料夾（L01、L02…）。"
            "載入後會依 JSON 的 x_mm / y_mm 拆成點位；點選點位只顯示該點 Heatmap 與距離清單。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #546E7A; font-size: 12px;")
        root.addWidget(tip)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("全選")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QPushButton("全不選")
        btn_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        root.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        for loc in self.locations:
            enabled = bool(self.current_config.get(loc, {}).get("enabled", False))
            chk = QCheckBox(f"位置：{loc}")
            chk.setChecked(enabled)
            chk.setStyleSheet("font-weight: bold; font-size: 13px;")
            self.loc_checks[loc] = chk
            host_layout.addWidget(chk)
        host_layout.addStretch()
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("font-weight: bold; color: #E65100; font-size: 12px;")
        root.addWidget(self.lbl_summary)
        for chk in self.loc_checks.values():
            chk.toggled.connect(self._refresh_summary)
        self._refresh_summary()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("套用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _set_all(self, checked):
        for chk in self.loc_checks.values():
            chk.setChecked(checked)

    def _refresh_summary(self, *_args):
        n = sum(1 for chk in self.loc_checks.values() if chk.isChecked())
        self.lbl_summary.setText(f"目前選擇：{n} 個位置")

    def get_config(self):
        return {
            loc: {"enabled": chk.isChecked()}
            for loc, chk in self.loc_checks.items()
        }


class BaslerJsonBatchTab(QWidget):
    """匯入含 L01/L02… 的主資料夾：JSON 距離清單 + NPY Heatmap（不抓光斑）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_dir = ""
        self.export_dir = ""
        self.available_locations = []
        self.location_meta = {}  # loc -> {npy_paths, json_paths}
        self.location_config = {}
        # 每個點位 = 同一位置資料夾內同一組 (x_mm, y_mm)
        self.points = []  # [{location, x_mm, y_mm, npy_path, rows, matrix, label, cross}, ...]
        self.current_idx = -1
        self._suppress_cross_save = False
        self._export_in_progress = False
        self._export_pause_requested = False
        self._export_paused = False
        self._export_session = None
        self._setup_ui()

    def _setup_ui(self):
        # 按鈕樣式對齊 Batch / Mapping（不加 min-height，避免左側鍵過大）
        btn_default = (
            "QPushButton { font-size: 13px; font-weight: bold; background-color: #f0f0f0; "
            "border: 1px solid #cccccc; border-radius: 5px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
            "QPushButton:disabled { background-color: #B0BEC5; color: #eceff1; }"
        )
        btn_primary = (
            "QPushButton { font-size: 13px; font-weight: bold; color: white; "
            "background-color: #2E7D32; border: none; border-radius: 5px; padding: 8px 12px; }"
            "QPushButton:hover { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        btn_export = (
            "QPushButton { font-size: 13px; font-weight: bold; color: white; "
            "background-color: #0288D1; border: none; border-radius: 5px; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #039BE5; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        btn_pause = (
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #EF6C00; border: none; border-radius: 5px; padding: 6px 10px; }"
            "QPushButton:hover { background-color: #F57C00; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        btn_resume = (
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #2E7D32; border: none; border-radius: 5px; padding: 6px 10px; }"
            "QPushButton:hover { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        btn_clear = (
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #546E7A; border: none; border-radius: 5px; padding: 6px 10px; }"
            "QPushButton:hover { background-color: #607D8B; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        section_lbl = "font-weight: bold; font-size: 12px; color: #455A64;"
        info_lbl = "color: #757575; font-size: 11px;"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #dcdcdc; width: 4px; }")
        layout.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        title = QLabel("JSON Batch")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #37474F;")
        left_layout.addWidget(title)

        tip = QLabel("匯入主資料夾 → 位置配置 → 載入點位")
        tip.setStyleSheet("color: #90A4AE; font-size: 11px;")
        left_layout.addWidget(tip)
        left_layout.addWidget(self._hline())

        lbl_import = QLabel("資料匯入")
        lbl_import.setStyleSheet(section_lbl)
        left_layout.addWidget(lbl_import)

        self.btn_import = QPushButton("選擇主資料夾")
        self.btn_import.setStyleSheet(btn_default)
        self.btn_import.clicked.connect(self.import_root_folder)
        left_layout.addWidget(self.btn_import)

        self.lbl_root = QLabel("未選擇資料夾")
        self.lbl_root.setWordWrap(True)
        self.lbl_root.setStyleSheet(info_lbl)
        left_layout.addWidget(self.lbl_root)

        self.btn_location_config = QPushButton("位置配置")
        self.btn_location_config.setStyleSheet(btn_default)
        self.btn_location_config.setEnabled(False)
        self.btn_location_config.clicked.connect(self.open_location_config)
        left_layout.addWidget(self.btn_location_config)

        self.lbl_selected = QLabel("已選位置: --")
        self.lbl_selected.setWordWrap(True)
        self.lbl_selected.setStyleSheet("color: #546E7A; font-size: 11px;")
        left_layout.addWidget(self.lbl_selected)

        self.btn_load = QPushButton("載入已配置位置")
        self.btn_load.setStyleSheet(btn_primary)
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self.load_selected_points)
        left_layout.addWidget(self.btn_load)

        left_layout.addWidget(self._hline())

        self.lbl_status = QLabel("狀態: 等待匯入主資料夾")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "color: #1565C0; font-weight: bold; font-size: 12px;"
        )
        left_layout.addWidget(self.lbl_status)

        left_layout.addWidget(self._hline())

        lbl_points = QLabel("點位列表")
        lbl_points.setStyleSheet(section_lbl)
        left_layout.addWidget(lbl_points)

        self.list_points = QListWidget()
        self.list_points.setMinimumHeight(160)
        self.list_points.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.list_points.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.list_points.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_points.setMovement(QListWidget.Static)
        self.list_points.setUniformItemSizes(True)
        self.list_points.setFocusPolicy(Qt.ClickFocus)
        self.list_points.setStyleSheet(
            "QListWidget { background: white; border: 1px solid #d0d0d0; font-size: 12px; "
            "outline: none; }"
            "QListWidget::item { padding: 4px 6px; border: none; outline: none; }"
            "QListWidget::item:selected { background: #BBDEFB; color: #0D47A1; "
            "border: none; outline: none; }"
            "QListWidget::item:selected:active { background: #BBDEFB; border: none; }"
            "QListWidget::item:selected:!active { background: #BBDEFB; border: none; }"
            "QListWidget::item:focus { border: none; outline: none; }"
        )
        self.list_points.setItemDelegate(_NoFocusDelegate(self.list_points))
        self.list_points.currentRowChanged.connect(self._on_point_selected)
        left_layout.addWidget(self.list_points, 1)

        self.lbl_point_stats = QLabel("distance_um  平均值: --　STD: --")
        self.lbl_point_stats.setWordWrap(True)
        self.lbl_point_stats.setStyleSheet(
            "color: #37474F; font-size: 12px; background: #FAFAFA; "
            "border: 1px solid #E0E0E0; border-radius: 4px; padding: 6px 8px;"
        )
        left_layout.addWidget(self.lbl_point_stats)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(6)

        self.btn_export_dir = QPushButton("選擇儲存資料夾")
        self.btn_export_dir.setStyleSheet(btn_default)
        self.btn_export_dir.setMinimumHeight(34)
        self.btn_export_dir.clicked.connect(self.select_export_directory)
        top_bar.addWidget(self.btn_export_dir)

        self.lbl_export_dir = QLabel("")
        self.lbl_export_dir.setStyleSheet(
            "color: #333333; background-color: #f5f5f5; border: 1px solid #d0d0d0; "
            "border-radius: 4px; padding: 4px 8px; font-size: 12px;"
        )
        self.lbl_export_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lbl_export_dir.setMinimumHeight(34)
        top_bar.addWidget(self.lbl_export_dir, 1)

        self.btn_export = QPushButton("一鍵匯出到資料夾")
        self.btn_export.setStyleSheet(btn_export)
        self.btn_export.setMinimumHeight(34)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.start_export)
        top_bar.addWidget(self.btn_export)

        self.btn_pause_export = QPushButton("暫停匯出")
        self.btn_pause_export.setStyleSheet(btn_pause)
        self.btn_pause_export.setMinimumHeight(34)
        self.btn_pause_export.setEnabled(False)
        self.btn_pause_export.clicked.connect(self._pause_export)
        top_bar.addWidget(self.btn_pause_export)

        self.btn_resume_export = QPushButton("續跑匯出")
        self.btn_resume_export.setStyleSheet(btn_resume)
        self.btn_resume_export.setMinimumHeight(34)
        self.btn_resume_export.setEnabled(False)
        self.btn_resume_export.clicked.connect(self._resume_export)
        top_bar.addWidget(self.btn_resume_export)

        self.btn_clear_export = QPushButton("清除匯出")
        self.btn_clear_export.setStyleSheet(btn_clear)
        self.btn_clear_export.setMinimumHeight(34)
        self.btn_clear_export.setEnabled(False)
        self.btn_clear_export.clicked.connect(self._clear_export_state)
        top_bar.addWidget(self.btn_clear_export)

        right_layout.addLayout(top_bar)

        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.setChildrenCollapsible(False)
        view_splitter.setHandleWidth(10)
        view_splitter.setStyleSheet(
            "QSplitter::handle { background-color: transparent; width: 10px; }"
        )
        right_layout.addWidget(view_splitter, 1)
        self._view_splitter = view_splitter

        heat_host = QWidget()
        heat_layout = QVBoxLayout(heat_host)
        heat_layout.setContentsMargins(0, 0, 6, 0)
        heat_layout.setSpacing(2)
        self.heatmap_panel = InteractiveHeatmapPanel(
            title="JSON Heatmap",
            x_label="X Pixels",
            y_label="Y Pixels",
            aspect_locked=True,
            with_profiles=True,
        )
        self.heatmap_panel.mouseMoved.connect(self._on_heatmap_mouse_moved)
        self.heatmap_panel.profilePointChanged.connect(self._on_profile_point_changed)
        self._reserve_colorbar_column_space()
        self._apply_heatmap_axis_tweaks()
        heat_layout.addWidget(self.heatmap_panel, 1)
        self.lbl_mouse = QLabel("滑鼠位置: X=--, Y=--, Value=--")
        self.lbl_mouse.setStyleSheet("color: #546E7A; font-size: 11px;")
        heat_layout.addWidget(self.lbl_mouse)
        view_splitter.addWidget(heat_host)

        table_panel = QWidget()
        table_panel.setMinimumWidth(380)
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(6, 0, 0, 0)
        table_layout.setSpacing(4)
        self.lbl_distance_title = QLabel("距離清單（目前點位）")
        self.lbl_distance_title.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #37474F;"
        )
        self.lbl_distance_title.setWordWrap(True)
        table_layout.addWidget(self.lbl_distance_title)
        self.distance_table = QTableWidget(0, 6)
        self.distance_table.setHorizontalHeaderLabels(
            ["#", "x_mm", "y_mm", "r_deg", "exposure_ms", "distance_um"]
        )
        self.distance_table.setAlternatingRowColors(True)
        self.distance_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.distance_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.distance_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.distance_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.distance_table.verticalHeader().setVisible(False)
        self.distance_table.setStyleSheet(
            "QTableWidget { background: white; border: 1px solid #d0d0d0; font-size: 12px; }"
            "QHeaderView::section { background: #f0f0f0; font-weight: bold; padding: 3px; "
            "border: 1px solid #d0d0d0; }"
        )
        header = self.distance_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setMinimumSectionSize(36)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.resizeSection(0, 36)
        header.resizeSection(1, 68)
        header.resizeSection(2, 68)
        header.resizeSection(3, 58)
        header.resizeSection(4, 88)
        header.resizeSection(5, 88)
        header.setStretchLastSection(True)
        self.distance_table.verticalHeader().setDefaultSectionSize(24)
        table_layout.addWidget(self.distance_table, 1)
        view_splitter.addWidget(table_panel)

        # Heatmap : 距離清單 = 5:3（stretch）、5:2（初始寬度）
        view_splitter.setStretchFactor(0, 5)
        view_splitter.setStretchFactor(1, 3)
        view_splitter.setSizes([1000, 400])

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 1100])

        self._refresh_action_buttons()


    @staticmethod
    def _hline():
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Plain)
        frame.setStyleSheet("background-color: #CFD8DC; max-height: 1px; border: none;")
        frame.setFixedHeight(1)
        return frame

    def _set_status(self, text, color="#1565C0"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 12px;"
        )

    def import_root_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "選擇主資料夾（內含 L01、L02… 位置子資料夾）", ""
        )
        if not dir_path:
            return

        meta, missing = self._scan_root(dir_path)
        if not meta:
            QMessageBox.warning(
                self,
                "警告",
                "找不到可用的位置資料夾。\n"
                "預期格式：<主資料夾>/<位置>/（一個 .npy + 多個 .json）",
            )
            return

        self.root_dir = dir_path
        self.location_meta = meta
        self.available_locations = sorted(meta.keys(), key=_natural_sort_key)
        self.location_config = {
            loc: {"enabled": bool(self.location_config.get(loc, {}).get("enabled", False))}
            for loc in self.available_locations
        }
        self.points = []
        self.current_idx = -1
        self.list_points.clear()
        self.distance_table.setRowCount(0)
        self.lbl_point_stats.setText("distance_um  平均值: --　STD: --")
        self.lbl_distance_title.setText("距離清單（目前點位）")

        n_json = sum(len(v["json_paths"]) for v in meta.values())
        n_npy = sum(len(v["npy_paths"]) for v in meta.values())
        info = (
            f"{os.path.basename(dir_path)}｜位置 {len(meta)} 個｜"
            f"NPY {n_npy}｜JSON {n_json}\n"
            f"位置: {', '.join(self.available_locations)}"
        )
        self.lbl_root.setText(info)

        msg = f"狀態: 已掃描 {len(meta)} 個位置，請進行位置配置後載入"
        if missing:
            msg += f"（略過無 NPY/JSON：{', '.join(missing[:8])}）"
        self._set_status(msg)
        self._update_selected_label()
        self._refresh_action_buttons()

    def _scan_root(self, root_dir):
        meta = {}
        missing = []
        try:
            entries = list(os.scandir(root_dir))
        except OSError:
            return {}, []

        for entry in entries:
            if not entry.is_dir():
                continue
            loc = entry.name
            npy_files = []
            json_files = []
            try:
                for child in os.scandir(entry.path):
                    if not child.is_file():
                        continue
                    lower = child.name.lower()
                    if lower.endswith(".npy"):
                        npy_files.append(child.path)
                    elif lower.endswith(".json"):
                        json_files.append(child.path)
            except OSError:
                continue

            npy_files.sort(key=lambda p: _natural_sort_key(os.path.basename(p)))
            json_files.sort(key=lambda p: _natural_sort_key(os.path.basename(p)))
            if not npy_files or not json_files:
                missing.append(loc)
                continue

            meta[loc] = {
                "npy_paths": npy_files,
                "json_paths": json_files,
            }
        return meta, missing

    def open_location_config(self):
        if not self.available_locations:
            QMessageBox.warning(self, "警告", "請先匯入主資料夾。")
            return
        dlg = LocationOnlyConfigDialog(
            self.available_locations,
            current_config=self.location_config,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self.location_config = dlg.get_config()
            self._update_selected_label()
            self._refresh_action_buttons()

    def _selected_locations(self):
        return [
            loc
            for loc in self.available_locations
            if self.location_config.get(loc, {}).get("enabled", False)
        ]

    def _update_selected_label(self):
        selected = self._selected_locations()
        if not selected:
            self.lbl_selected.setText("已選位置: （尚未勾選）")
        else:
            self.lbl_selected.setText(
                f"已選位置: {len(selected)} 個 — {', '.join(selected)}"
            )

    def _refresh_action_buttons(self):
        """依流程啟用／反白按鈕：匯入 → 位置配置 → 載入 → 匯出。"""
        has_root = bool(self.available_locations)
        has_selected = bool(self._selected_locations())
        has_points = bool(self.points)
        has_export_dir = bool(self.export_dir)
        exporting = bool(self._export_in_progress)
        paused = bool(self._export_paused and self._export_session)

        self.btn_location_config.setEnabled(has_root and not exporting)
        self.btn_load.setEnabled(has_root and has_selected and not exporting)
        self.btn_export_dir.setEnabled(not exporting)
        self.btn_export.setEnabled(
            has_points and has_export_dir and not exporting and not paused
        )
        # pause / resume / clear 由 _set_export_controls 另行控制；此處只在非匯出態維持一致
        if not exporting and not paused:
            self.btn_pause_export.setEnabled(False)
            self.btn_resume_export.setEnabled(False)
            self.btn_clear_export.setEnabled(False)

    def load_selected_points(self):
        selected = self._selected_locations()
        if not selected:
            QMessageBox.warning(
                self, "警告", "尚未配置任何位置！\n請先點「位置配置」勾選位置。"
            )
            return

        self.btn_load.setEnabled(False)
        self.points = []
        self.list_points.clear()
        self.distance_table.setRowCount(0)
        self.lbl_distance_title.setText("距離清單（目前點位）")
        warnings = []

        try:
            for i, loc in enumerate(selected):
                info = self.location_meta[loc]
                self._set_status(
                    f"狀態: 載入中 {i + 1}/{len(selected)} — {loc}",
                    "#F57C00",
                )
                QApplication.processEvents()

                rows, skipped = self._load_distance_rows(info["json_paths"])
                if skipped:
                    warnings.append(f"{loc} 略過 {len(skipped)} 筆壞 JSON")

                groups = {}
                missing_xy = 0
                for row in rows:
                    key = _xy_key(row.get("x_mm"), row.get("y_mm"))
                    if key is None:
                        missing_xy += 1
                        continue
                    groups.setdefault(key, []).append(row)
                if missing_xy:
                    warnings.append(f"{loc} 有 {missing_xy} 筆 JSON 缺少 x_mm/y_mm")

                npy_by_xy = []
                for npy_path in info["npy_paths"]:
                    xy = _parse_xy_from_name(npy_path)
                    if xy is None:
                        warnings.append(
                            f"{loc} 無法從 NPY 檔名解析座標：{os.path.basename(npy_path)}"
                        )
                        continue
                    npy_by_xy.append((xy, npy_path))

                for key in sorted(groups.keys()):
                    group_rows = groups[key]
                    npy_path = None
                    for xy, path in npy_by_xy:
                        if _xy_match(key, xy):
                            npy_path = path
                            break
                    if npy_path is None and len(npy_by_xy) == 1 and len(groups) == 1:
                        npy_path = npy_by_xy[0][1]
                    if npy_path is None:
                        warnings.append(
                            f"{loc} {_format_xy_label(key[0], key[1])} 找不到對應 NPY，已略過"
                        )
                        continue

                    try:
                        matrix = np.load(npy_path)
                        matrix = np.asarray(matrix, dtype=float)
                        if matrix.ndim != 2:
                            raise ValueError(f"NPY 維度不是 2D：{matrix.shape}")
                    except Exception as exc:
                        warnings.append(
                            f"{loc} {_format_xy_label(key[0], key[1])} NPY 讀取失敗：{exc}"
                        )
                        continue

                    label = (
                        f"{loc} | {_format_xy_label(key[0], key[1])} "
                        f"({len(group_rows)} JSON)"
                    )
                    self.points.append(
                        {
                            "location": loc,
                            "x_mm": key[0],
                            "y_mm": key[1],
                            "npy_path": npy_path,
                            "rows": group_rows,
                            "matrix": matrix,
                            "label": label,
                            "cross": None,  # (ix, iy) 各點獨立記住十字位置
                        }
                    )

            if not self.points:
                self._set_status("狀態: 載入失敗（無可用點位）", "#C62828")
                if warnings:
                    QMessageBox.warning(
                        self, "載入失敗", "\n".join(warnings[:12])
                    )
                return

            # 點位排序：先 X、再 Y、再 L01/L02…
            self.points.sort(
                key=lambda p: (
                    float(p["x_mm"]),
                    float(p["y_mm"]),
                    _natural_sort_key(p["location"]),
                )
            )
            self.list_points.clear()
            for idx, point in enumerate(self.points):
                item = QListWidgetItem(point["label"])
                item.setData(Qt.UserRole, idx)
                self.list_points.addItem(item)

            self.list_points.setCurrentRow(0)
            msg = f"狀態: 已載入 {len(self.points)} 個點位（依 X→Y→L）"
            if warnings:
                msg += "｜" + "；".join(warnings[:2])
            self._set_status(msg, "#2E7D32")
        finally:
            self._refresh_action_buttons()

    def _load_distance_rows(self, json_paths):
        rows = []
        skipped = []
        total = len(json_paths)
        workers = min(_IMPORT_WORKERS, max(1, total))
        results = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_read_one_distance_json, path): path for path in json_paths
            }
            done = 0
            for future in as_completed(futures):
                path, row, err = future.result()
                done += 1
                if done == 1 or done == total or done % 200 == 0:
                    QApplication.processEvents()
                if err is not None:
                    skipped.append(f"{os.path.basename(path)}: {err}")
                    continue
                results[path] = row

        for path in json_paths:
            row = results.get(path)
            if row is not None:
                rows.append(row)
        return rows, skipped

    def _on_point_selected(self, row):
        # 切換前先記住目前點的十字位置
        self._store_current_cross()
        if row < 0 or row >= len(self.points):
            self.current_idx = -1
            self.lbl_point_stats.setText("distance_um  平均值: --　STD: --")
            return
        self.current_idx = row
        point = self.points[row]
        self._fill_distance_table(point["rows"])
        self._update_point_stats(point)
        self._draw_heatmap(point)

    def _store_current_cross(self):
        if not (0 <= self.current_idx < len(self.points)):
            return
        cross = getattr(self.heatmap_panel, "_profile_point", None)
        if cross is None:
            return
        ix, iy = int(cross[0]), int(cross[1])
        self.points[self.current_idx]["cross"] = (ix, iy)

    def _on_profile_point_changed(self, ix, iy):
        if self._suppress_cross_save:
            return
        if not (0 <= self.current_idx < len(self.points)):
            return
        self.points[self.current_idx]["cross"] = (int(ix), int(iy))

    def _finite_values(self, rows, key):
        vals = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(num):
                vals.append(num)
        return np.asarray(vals, dtype=float)

    def _stat_text(self, values):
        if values.size == 0:
            return "null", "null"
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        return f"{mean:.3f}", f"{std:.3f}"

    def _update_point_stats(self, point):
        rows = point.get("rows") or []
        dist_mean, dist_std = self._stat_text(self._finite_values(rows, "distance_um"))
        self.lbl_point_stats.setText(
            f"distance_um  平均值: {dist_mean}\nSTD: {dist_std}"
        )

    def _fill_distance_table(self, rows):
        self.distance_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                str(i + 1),
                _format_json_value(row.get("x_mm")),
                _format_json_value(row.get("y_mm")),
                _format_json_value(row.get("r_deg")),
                _format_json_value(row.get("exposure_ms")),
                _format_json_value(row.get("distance_um")),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.distance_table.setItem(i, col, item)

    def _draw_heatmap(self, point):
        matrix = point["matrix"]
        ny, nx = matrix.shape
        title = (
            f"{point['location']} | "
            f"{_format_xy_label(point['x_mm'], point['y_mm'])}"
        )
        self.heatmap_panel.set_plot_title(title)
        self.lbl_distance_title.setText(f"距離清單 — {title}")
        self.heatmap_panel.set_axis_labels("X Pixels", "Y Pixels")

        # 先取出已記住的十字，避免 set_image 觸發的 signal 覆寫成中心點
        cross = point.get("cross")
        if cross is None:
            cross = (nx // 2, ny // 2)
        ix = int(np.clip(int(cross[0]), 0, nx - 1))
        iy = int(np.clip(int(cross[1]), 0, ny - 1))

        self._suppress_cross_save = True
        try:
            self.heatmap_panel._profile_point = None
            self.heatmap_panel.set_image(
                matrix.T,
                reset_view=True,
                source_matrix=matrix,
                x_coords=np.arange(nx, dtype=float),
                y_coords=np.arange(ny, dtype=float),
            )
            self.heatmap_panel.set_profile_point(ix, iy, reset_view=False)
            point["cross"] = (ix, iy)
            self._apply_heatmap_axis_tweaks()
        finally:
            self._suppress_cross_save = False

        self._set_status(
            f"狀態: 顯示 {title}\n"
            f"JSON {len(point['rows'])} 筆",
            "#2E7D32",
        )

    def _reserve_colorbar_column_space(self):
        """在 colorbar 下方佔位，避免 X 剖面右側 Value 軸被裁切。"""
        panel = self.heatmap_panel
        if panel.plot_x_profile is None or panel.hist is None:
            return
        try:
            if getattr(self, "_profile_hist_spacer", None) is not None:
                return
            spacer = panel.win.addPlot(row=1, col=2)
            spacer.hideAxis("left")
            spacer.hideAxis("right")
            spacer.hideAxis("top")
            spacer.hideAxis("bottom")
            spacer.setMouseEnabled(x=False, y=False)
            spacer.hideButtons()
            if hasattr(spacer, "titleLabel") and spacer.titleLabel is not None:
                spacer.titleLabel.hide()
            panel.win.ci.layout.setColumnFixedWidth(2, 110)
            panel.win.ci.layout.setColumnStretchFactor(2, 0)
            self._profile_hist_spacer = spacer
        except Exception:
            self._profile_hist_spacer = None

    def _apply_heatmap_axis_tweaks(self):
        """熱圖底軸不顯示 X Pixels（改由下方剖面顯示），避免標題被擋。"""
        try:
            # 熱圖底部：清空標籤、不顯示刻度，只留極小佔位對齊剖面
            bottom = self.heatmap_panel.plot.getAxis("bottom")
            self.heatmap_panel.plot.setLabel("bottom", "")
            bottom.setStyle(showValues=False)
            bottom.setHeight(6)
            # 下方 X 剖面保留完整「X Pixels」標籤與刻度
            x_prof = self.heatmap_panel.plot_x_profile
            if x_prof is not None:
                x_prof.setLabel("bottom", "X Pixels")
                right = x_prof.getAxis("right")
                right.setWidth(72)
                right.setStyle(tickTextOffset=2)
                try:
                    right.enableAutoSIPrefix(True)
                except Exception:
                    pass
            if self.heatmap_panel.hist is not None:
                self.heatmap_panel.win.ci.layout.setColumnFixedWidth(2, 110)
        except Exception:
            pass

    def _on_heatmap_mouse_moved(self, mouse_point):
        if mouse_point is None:
            return
        if self.current_idx < 0 or self.current_idx >= len(self.points):
            return
        matrix = self.points[self.current_idx]["matrix"]
        ny, nx = matrix.shape
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        ix = int(np.clip(round(x), 0, nx - 1))
        iy = int(np.clip(round(y), 0, ny - 1))
        if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
            return
        val = matrix[iy, ix]
        self.lbl_mouse.setText(
            f"滑鼠位置: X={ix}, Y={iy}, Value={val:.6g}"
        )

    def select_export_directory(self):
        path = QFileDialog.getExistingDirectory(self, "選擇儲存資料夾", "")
        if not path:
            return
        self.export_dir = path
        self.lbl_export_dir.setText(path)
        self._set_status("狀態: 已選擇匯出資料夾")
        self._refresh_action_buttons()

    def _set_export_controls(self, running=False, paused=False):
        self.btn_pause_export.setEnabled(running and not paused)
        self.btn_resume_export.setEnabled((not running) and paused)
        self.btn_clear_export.setEnabled((not running) and paused)

    def _pause_export(self):
        if not self._export_in_progress:
            return
        self._export_pause_requested = True
        self._set_export_controls(running=True, paused=False)
        self._set_status("狀態: 正在暫停並輸出目前進度...", "#EF6C00")

    def _resume_export(self):
        if self._export_in_progress:
            return
        if (not self._export_paused) or (not self._export_session):
            QMessageBox.information(self, "提示", "目前沒有可續跑的暫停匯出。")
            return
        self._export_pause_requested = False
        self._export_paused = False
        self._set_status("狀態: 匯出續跑中...", "#F57C00")
        self._run_export_loop()

    def _clear_export_state(self, notify=True):
        if self._export_in_progress:
            QMessageBox.warning(self, "提示", "匯出進行中，請先暫停後再清除。")
            return
        if (not self._export_paused) and (not self._export_session):
            if notify:
                QMessageBox.information(self, "提示", "目前沒有可清除的暫停匯出。")
            return
        self._export_session = None
        self._export_pause_requested = False
        self._export_paused = False
        self._export_in_progress = False
        self._set_export_controls(running=False, paused=False)
        self._refresh_action_buttons()
        self._set_status("狀態: 已清除暫停匯出，可重新一鍵匯出")
        if notify:
            QMessageBox.information(
                self, "已清除", "已清除暫停匯出狀態，現在可重新一鍵匯出。"
            )

    def start_export(self):
        if not self.points:
            QMessageBox.warning(self, "警告", "目前無可匯出的數據！")
            return
        if not self.export_dir:
            QMessageBox.warning(self, "警告", "請先選擇儲存資料夾！")
            return
        if self._export_in_progress:
            QMessageBox.information(self, "提示", "目前已有匯出在進行中。")
            return
        if self._export_paused and self._export_session:
            QMessageBox.information(self, "提示", "目前為暫停狀態，請按『續跑匯出』。")
            return

        ts = export_timestamp_tag()
        export_root = os.path.join(self.export_dir, f"Basler_JSON_Batch_Results_{ts}")
        os.makedirs(export_root, exist_ok=True)
        self._export_session = {
            "export_root": export_root,
            "prev_idx": self.current_idx if self.current_idx >= 0 else 0,
            "next_idx": 0,
        }
        self._export_in_progress = False
        self._export_pause_requested = False
        self._export_paused = False
        self._set_status("狀態: 正在匯出到資料夾...", "#F57C00")
        self._run_export_loop()

    def _run_export_loop(self):
        session = self._export_session
        if not session:
            return

        try:
            self._export_in_progress = True
            self._set_export_controls(running=True, paused=False)
            self.btn_export.setEnabled(False)
            QApplication.processEvents()

            total = len(self.points)
            for idx in range(session["next_idx"], total):
                if self._export_pause_requested:
                    break

                self.list_points.setCurrentRow(idx)
                QApplication.processEvents()
                point = self.points[idx]
                loc = point["location"]
                safe_loc = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in str(loc)
                )
                x_tag = str(point["x_mm"]).replace(".", "p").replace("-", "m")
                y_tag = str(point["y_mm"]).replace(".", "p").replace("-", "m")
                base_name = f"{safe_loc}_X{x_tag}_Y{y_tag}"
                loc_dir = os.path.join(session["export_root"], safe_loc)
                os.makedirs(loc_dir, exist_ok=True)
                base = os.path.join(loc_dir, base_name)
                csv_path = f"{base}.csv"
                heatmap_base = f"{base}_Heatmap_With_Profiles"
                self._write_point_csv(csv_path, point)
                self._export_heatmap_image(heatmap_base)

                session["next_idx"] = idx + 1
                self._set_status(
                    f"狀態: 匯出中... {session['next_idx']}/{total}",
                    "#F57C00",
                )
                QApplication.processEvents()

            if self._export_pause_requested and session["next_idx"] < total:
                self._export_in_progress = False
                self._export_pause_requested = False
                self._export_paused = True
                self._set_export_controls(running=False, paused=True)
                self._set_status(
                    f"狀態: 已暫停並輸出進度（{session['next_idx']}/{total}）",
                    "#EF6C00",
                )
                QMessageBox.information(
                    self,
                    "已暫停並輸出",
                    f"目前進度已匯出：\n\n資料夾：\n{session['export_root']}\n\n"
                    f"進度：{session['next_idx']}/{total}",
                )
                return

            prev = session.get("prev_idx", 0)
            if 0 <= prev < len(self.points):
                self.list_points.setCurrentRow(prev)
            self._set_export_controls(running=False, paused=False)
            self._export_session = None
            self._export_paused = False
            self._export_in_progress = False
            self._export_pause_requested = False
            self._refresh_action_buttons()
            self._set_status("狀態: 所有點位匯出成功！", "#2E7D32")
            QMessageBox.information(
                self,
                "成功",
                f"匯出完成！\n\n資料夾：\n{session['export_root']}\n\n"
                f"每個點位各一個 CSV（距離清單）與 Heatmap 圖"
                f"（{EXPORT_IMAGE_EXT}，含剖面、匯出時不含 colorbar）",
            )
        except Exception as exc:
            self._set_status("狀態: 匯出失敗", "#C62828")
            self._set_export_controls(running=False, paused=False)
            self._export_in_progress = False
            self._export_pause_requested = False
            self._refresh_action_buttons()
            QMessageBox.critical(self, "匯出錯誤", f"匯出過程發生錯誤：\n{exc}")

    def _write_point_csv(self, csv_path, point):
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["index", "json_file", "x_mm", "y_mm", "r_deg", "exposure_ms", "distance_um"]
            )
            for i, row in enumerate(point["rows"], start=1):
                writer.writerow(
                    [
                        i,
                        row.get("file", ""),
                        _format_json_value(row.get("x_mm")),
                        _format_json_value(row.get("y_mm")),
                        _format_json_value(row.get("r_deg")),
                        _format_json_value(row.get("exposure_ms")),
                        _format_json_value(row.get("distance_um")),
                    ]
                )

    def _export_heatmap_image(self, base_path):
        """匯出含剖面的 Heatmap（不含 colorbar／十字），與前面 Batch 風格一致。"""
        paths = self.heatmap_panel.export_plot_bundle(base_path)
        if not paths:
            raise RuntimeError("無法匯出 Heatmap 圖檔")
        return [normalize_export_image_path(p) for p in paths]
