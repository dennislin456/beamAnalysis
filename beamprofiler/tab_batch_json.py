"""Basler JSON Batch：位置資料夾（L01…）＋ JSON 距離清單；NPY Heatmap 可選。"""

import csv
import json
import math
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
    PROFILE_EDGE_AXIS_PX_COMPACT,
    PROFILE_SIDE_AXIS_PX_COMPACT,
    export_timestamp_tag,
    normalize_export_image_path,
)

try:
    import orjson as _fast_json
except ImportError:  # pragma: no cover
    _fast_json = None


_JSON_FIELDS = (
    "x_mm",
    "y_mm",
    "r1_deg",
    "r2_deg",
    "exposure_ms",
    "distance_um",
    "dist_ifc",
)
_POSITION_FIELDS = ("x_mm", "y_mm", "r1_deg", "r2_deg")
_ABOVE_BELOW_PX_FIELDS = (
    ("above_x_px", "above", "x_px"),
    ("above_y_px", "above", "y_px"),
    ("below_x_px", "below", "x_px"),
    ("below_y_px", "below", "y_px"),
)
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


def _nice_tick_step(span, target_labels=8):
    if span <= 0:
        return 1.0
    raw = span / max(float(target_labels), 1.0)
    magnitude = 10 ** math.floor(math.log10(raw))
    for nice in (1, 2, 5, 10):
        step = nice * magnitude
        if step >= raw:
            return float(step)
    return float(10 * magnitude)


def _pixel_tick_spacing(count, target_labels=8):
    span = max(float(count) - 1.0, 1.0)
    return _pixel_tick_spacing_for_span(span, target_labels)


def _pixel_tick_spacing_for_span(span, target_labels=8):
    span = max(float(span), 1.0)
    major = _nice_tick_step(span, target_labels)
    return major, major / 5.0


def _pick_json_field(data, key):
    """優先 position（座標／角度）、其次頂層，再其次 value。"""
    if not isinstance(data, dict):
        return None
    if key in _POSITION_FIELDS:
        position = data.get("position")
        if isinstance(position, dict) and key in position:
            return position.get(key)
    if key in data:
        return data.get(key)
    value = data.get("value")
    if isinstance(value, dict) and key in value:
        return value.get(key)
    return None


def _pick_value_nested(data, *path):
    """讀取 value 內巢狀欄位（如 value.above.x_px）；也相容頂層同路徑。"""
    if not isinstance(data, dict):
        return None
    for root_key in ("value", None):
        cur = data.get("value") if root_key == "value" else data
        if not isinstance(cur, dict):
            continue
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur.get(key)
        if ok:
            return cur
    return None


def _read_one_distance_json(file_path):
    try:
        data = _load_json_obj(file_path)
        row = {"file": os.path.basename(file_path)}
        for key in _JSON_FIELDS:
            row[key] = _pick_json_field(data, key)
        for out_key, section, field in _ABOVE_BELOW_PX_FIELDS:
            row[out_key] = _pick_value_nested(data, section, field)
        return file_path, row, None
    except Exception as exc:
        return file_path, None, exc


class _NoFocusDelegate(QStyledItemDelegate):
    """去掉點位列表選取時的虛線／透明焦點框。"""

    def paint(self, painter, option, index):
        option.state &= ~QStyle.State_HasFocus
        super().paint(painter, option, index)


class LocationOnlyConfigDialog(QDialog):
    """僅勾選位置資料夾（JSON 必要；NPY 可選）。"""

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
            "載入後會依 JSON 的 x_mm / y_mm 拆成點位；"
            "NPY 可選；同資料夾有多個 NPY 時只取第一筆畫 Heatmap。"
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
    """匯入含 L01/L02… 的主資料夾：JSON 距離清單（必要）＋ NPY Heatmap（可選）。"""

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
        self._heatmap_range_sync_guard = False
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

        self.lbl_point_stats = QLabel("平均值: --\nSTD: --")
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
        # realign 後再套用軸調整，避免剖面／熱圖對齊被覆寫
        _orig_realign = self.heatmap_panel._realign_profile_geometry

        def _realign_then_tweak(*args, **kwargs):
            _orig_realign(*args, **kwargs)
            self._apply_heatmap_axis_tweaks()
            self._sync_profile_pixel_axes()

        self.heatmap_panel._realign_profile_geometry = _realign_then_tweak
        self._hook_heatmap_view_sync()
        self._reserve_colorbar_column_space()
        self._apply_heatmap_axis_tweaks()
        heat_layout.addWidget(self.heatmap_panel, 1)
        self.lbl_mouse = QLabel("滑鼠位置: X=--, Y=--, Value=--")
        self.lbl_mouse.setStyleSheet("color: #546E7A; font-size: 11px;")
        heat_layout.addWidget(self.lbl_mouse)
        view_splitter.addWidget(heat_host)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(6, 0, 0, 0)
        table_layout.setSpacing(4)
        self.lbl_distance_title = QLabel("距離清單（目前點位）")
        self.lbl_distance_title.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #37474F;"
        )
        self.lbl_distance_title.setWordWrap(True)
        table_layout.addWidget(self.lbl_distance_title)
        self.distance_table = QTableWidget(0, 8)
        self.distance_table.setHorizontalHeaderLabels(
            [
                "#",
                "x_mm",
                "y_mm",
                "r1_deg",
                "r2_deg",
                "exposure_ms",
                "distance_um",
                "dist_mm",
            ]
        )
        self.distance_table.setAlternatingRowColors(True)
        self.distance_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.distance_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.distance_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.distance_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.distance_table.verticalHeader().setVisible(False)
        self.distance_table.setStyleSheet(
            "QTableWidget { background: white; border: 1px solid #d0d0d0; font-size: 12px; }"
            "QHeaderView::section { background: #f0f0f0; font-weight: bold; padding: 3px; "
            "border: 1px solid #d0d0d0; }"
        )
        col_widths = (34, 62, 62, 60, 60, 90, 88, 70)
        header = self.distance_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setMinimumSectionSize(36)
        header.setStretchLastSection(False)
        for col, width in enumerate(col_widths):
            header.setSectionResizeMode(col, QHeaderView.Fixed)
            header.resizeSection(col, width)
        self.distance_table.verticalHeader().setDefaultSectionSize(24)
        # 清單寬度 = 欄寬合計 + 直向捲軸預留 + 邊框／左邊距（不再用 5:3）
        table_w = (
            sum(col_widths)
            + self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
            + max(2, self.distance_table.frameWidth() * 2)
            + table_layout.contentsMargins().left()
            + 4
        )
        table_panel.setMinimumWidth(table_w)
        table_panel.setMaximumWidth(table_w)
        table_layout.addWidget(self.distance_table, 1)
        view_splitter.addWidget(table_panel)

        # Heatmap 吃剩餘寬度；清單依欄寬固定
        view_splitter.setStretchFactor(0, 1)
        view_splitter.setStretchFactor(1, 0)
        view_splitter.setSizes([1200, table_w])

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
                "預期格式：<主資料夾>/<位置>/（至少一個 .json；"
                ".npy 可選，同資料夾多檔時只取第一筆）",
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
        self.lbl_point_stats.setText("平均值: --\nSTD: --")
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
            msg += f"（略過無 JSON：{', '.join(missing[:8])}）"
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
            # JSON 必要；NPY 可選（僅 JSON 亦可匯入）
            if not json_files:
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

                # 同資料夾多個 NPY：只取排序後第一筆畫圖，其餘忽略
                first_npy = info["npy_paths"][0] if info["npy_paths"] else None
                shared_matrix = None
                shared_npy_path = None
                if first_npy is not None:
                    try:
                        shared_matrix = np.asarray(np.load(first_npy), dtype=float)
                        if shared_matrix.ndim != 2:
                            raise ValueError(f"NPY 維度不是 2D：{shared_matrix.shape}")
                        shared_npy_path = first_npy
                    except Exception as exc:
                        warnings.append(
                            f"{loc} NPY 讀取失敗（仍載入 JSON）："
                            f"{os.path.basename(first_npy)} — {exc}"
                        )
                        shared_matrix = None
                        shared_npy_path = None
                    if len(info["npy_paths"]) > 1:
                        warnings.append(
                            f"{loc} 有 {len(info['npy_paths'])} 個 NPY，"
                            f"僅使用第一筆：{os.path.basename(first_npy)}"
                        )

                for key in sorted(groups.keys()):
                    group_rows = groups[key]
                    npy_path = shared_npy_path
                    matrix = shared_matrix

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
            self.lbl_point_stats.setText("平均值: --\nSTD: --")
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
        self._sync_profile_pixel_axes()

    def _hook_heatmap_view_sync(self):
        """熱圖縮放／平移時，剖面 Pixels 軸跟著同步。"""
        try:
            heat_vb = self.heatmap_panel.plot.getViewBox()
        except Exception:
            heat_vb = None
        if heat_vb is not None:
            heat_vb.sigRangeChanged.connect(self._on_heatmap_view_range_changed)

    def _on_heatmap_view_range_changed(self, *args):
        if self._heatmap_range_sync_guard:
            return
        self._sync_profile_pixel_axes()

    def _sync_profile_pixel_axes(self):
        """讓剖面 Pixels 軸與熱圖使用相同刻度間距與資料範圍。"""
        if not (0 <= self.current_idx < len(self.points)):
            return
        matrix = self.points[self.current_idx].get("matrix")
        if matrix is None:
            return

        ny, nx = matrix.shape
        panel = self.heatmap_panel
        heat = panel.plot
        x_prof = panel.plot_x_profile
        y_prof = panel.plot_y_profile
        if heat is None or x_prof is None or y_prof is None:
            return

        try:
            x_range, y_range = heat.getViewBox().viewRange()
            x_range = (float(x_range[0]), float(x_range[1]))
            y_range = (float(y_range[0]), float(y_range[1]))
        except Exception:
            x_range = (0.0, float(max(nx - 1, 0)))
            y_range = (0.0, float(max(ny - 1, 0)))

        x_span = max(x_range[1] - x_range[0], 1.0)
        y_span = max(y_range[1] - y_range[0], 1.0)
        x_major, x_minor = _pixel_tick_spacing_for_span(x_span)
        y_major, y_minor = _pixel_tick_spacing_for_span(y_span)
        panel.set_tick_spacing(x_major, x_minor, y_major, y_minor, grid_alpha=0.25)

        self._heatmap_range_sync_guard = True
        try:
            try:
                _, y_intensity = x_prof.viewRange()[1]
                x_prof.setRange(xRange=x_range, yRange=y_intensity, padding=0)
            except Exception:
                x_prof.setXRange(x_range[0], x_range[1], padding=0)

            try:
                x_intensity = y_prof.viewRange()[0]
                y_prof.setRange(xRange=x_intensity, yRange=y_range, padding=0)
            except Exception:
                y_prof.setYRange(y_range[0], y_range[1], padding=0)
        finally:
            self._heatmap_range_sync_guard = False

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
            f"平均值: {dist_mean}\nSTD: {dist_std}"
        )

    def _fill_distance_table(self, rows):
        self.distance_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                str(i + 1),
                _format_json_value(row.get("x_mm")),
                _format_json_value(row.get("y_mm")),
                _format_json_value(row.get("r1_deg")),
                _format_json_value(row.get("r2_deg")),
                _format_json_value(row.get("exposure_ms")),
                _format_json_value(row.get("distance_um")),
                _format_json_value(row.get("dist_ifc")),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.distance_table.setItem(i, col, item)

    def _draw_heatmap(self, point):
        title = (
            f"{point['location']} | "
            f"{_format_xy_label(point['x_mm'], point['y_mm'])}"
        )
        self.heatmap_panel.set_plot_title(title)
        self.lbl_distance_title.setText(f"距離清單 — {title}")
        self.heatmap_panel.set_axis_labels("X Pixels", "Y Pixels")

        matrix = point.get("matrix")
        if matrix is None:
            self._suppress_cross_save = True
            try:
                self.heatmap_panel._profile_point = None
                self.heatmap_panel.clear_profile_cross()
                self.heatmap_panel.set_image(None, reset_view=True)
                self._apply_heatmap_axis_tweaks()
            finally:
                self._suppress_cross_save = False
            self.lbl_mouse.setText("滑鼠位置: （無 Heatmap）")
            self._set_status(
                f"狀態: 顯示 {title}\n"
                f"JSON {len(point['rows'])} 筆｜無 NPY",
                "#2E7D32",
            )
            return

        ny, nx = matrix.shape

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
            self._sync_profile_pixel_axes()
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
        """熱圖拿掉軸標題；軸佔位與剖面一致，讓剖面 Pixels 貼齊熱圖 ViewBox。"""
        try:
            side = PROFILE_SIDE_AXIS_PX_COMPACT
            edge = PROFILE_EDGE_AXIS_PX_COMPACT
            value_w = max(side, 72)
            heat = self.heatmap_panel.plot
            x_prof = self.heatmap_panel.plot_x_profile
            y_prof = self.heatmap_panel.plot_y_profile

            # 縮小格線間距，剖面更貼熱圖
            try:
                self.heatmap_panel.win.ci.layout.setSpacing(0)
                self.heatmap_panel.win.ci.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass

            # 熱圖：不顯示 Y/X 軸標題；刻度數字保留（Pixels 標籤改由剖面顯示）
            heat.setLabel("left", "")
            heat.setLabel("bottom", "")
            left = heat.getAxis("left")
            left.setStyle(showValues=True)
            left.setWidth(side)
            bottom = heat.getAxis("bottom")
            bottom.setStyle(showValues=True)
            bottom.setHeight(edge)
            # 右／上佔位與剖面同尺寸 → ViewBox 邊緣貼齊（對應紅線）
            heat.showAxis("right", show=True)
            heat.getAxis("right").setWidth(value_w)
            heat.getAxis("right").setStyle(showValues=False, tickLength=0)
            heat.showAxis("top", show=True)
            heat.getAxis("top").setHeight(edge)
            heat.getAxis("top").setStyle(showValues=False, tickLength=0)

            if x_prof is not None:
                x_prof.setLabel("bottom", "X Pixels")
                x_prof.setLabel("right", "Value")
                x_prof.getAxis("bottom").setHeight(edge)
                right = x_prof.getAxis("right")
                right.setWidth(value_w)
                right.setStyle(tickTextOffset=2)
                try:
                    right.enableAutoSIPrefix(True)
                except Exception:
                    pass
                x_prof.showAxis("left", show=True)
                x_prof.getAxis("left").setWidth(side)
                x_prof.getAxis("left").setStyle(showValues=False, tickLength=0)
                x_prof.setLabel("left", "")

            if y_prof is not None:
                y_prof.setLabel("left", "Y Pixels")
                y_prof.setLabel("top", "Value")
                y_prof.getAxis("left").setWidth(side)
                y_prof.getAxis("top").setHeight(edge)
                y_prof.showAxis("bottom", show=True)
                y_prof.getAxis("bottom").setHeight(edge)
                y_prof.getAxis("bottom").setStyle(showValues=False, tickLength=0)
                y_prof.setLabel("bottom", "")

            if self.heatmap_panel.hist is not None:
                self.heatmap_panel.win.ci.layout.setColumnFixedWidth(2, 110)
        except Exception:
            pass

    def _on_heatmap_mouse_moved(self, mouse_point):
        if mouse_point is None:
            return
        if self.current_idx < 0 or self.current_idx >= len(self.points):
            return
        matrix = self.points[self.current_idx].get("matrix")
        if matrix is None:
            self.lbl_mouse.setText("滑鼠位置: （無 Heatmap）")
            return
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
                if point.get("matrix") is not None:
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
                [
                    "index",
                    "json_file",
                    "x_mm",
                    "y_mm",
                    "r1_deg",
                    "r2_deg",
                    "exposure_ms",
                    "distance_um",
                    "dist_mm",
                    "above_x_px",
                    "above_y_px",
                    "below_x_px",
                    "below_y_px",
                ]
            )
            for i, row in enumerate(point["rows"], start=1):
                writer.writerow(
                    [
                        i,
                        row.get("file", ""),
                        _format_json_value(row.get("x_mm")),
                        _format_json_value(row.get("y_mm")),
                        _format_json_value(row.get("r1_deg")),
                        _format_json_value(row.get("r2_deg")),
                        _format_json_value(row.get("exposure_ms")),
                        _format_json_value(row.get("distance_um")),
                        _format_json_value(row.get("dist_ifc")),
                        _format_json_value(row.get("above_x_px")),
                        _format_json_value(row.get("above_y_px")),
                        _format_json_value(row.get("below_x_px")),
                        _format_json_value(row.get("below_y_px")),
                    ]
                )

    def _export_heatmap_image(self, base_path):
        """匯出含剖面的 Heatmap（不含 colorbar／十字），與前面 Batch 風格一致。"""
        paths = self.heatmap_panel.export_plot_bundle(base_path)
        if not paths:
            raise RuntimeError("無法匯出 Heatmap 圖檔")
        return [normalize_export_image_path(p) for p in paths]
