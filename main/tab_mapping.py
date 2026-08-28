import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as pg_export

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QSplitter, QFrame, QLineEdit,
    QDoubleSpinBox, QCheckBox, QMainWindow, QGridLayout, QComboBox,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, QRectF

from shared_components import InteractiveHeatmapPanel


class MappingRoiWindow(QMainWindow):
    """Mapping ROI 獨立檢視與匯出視窗。"""

    def __init__(self, matrix, x_coords, y_coords, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Mapping 範圍檢視｜X={float(x_coords[0]):.3f}~{float(x_coords[-1]):.3f} "
            f"｜Y={float(y_coords[0]):.3f}~{float(y_coords[-1]):.3f}"
        )
        self.resize(900, 760)
        self.matrix = np.asarray(matrix, dtype=float)
        self.x_coords = np.asarray(x_coords, dtype=float)
        self.y_coords = np.asarray(y_coords, dtype=float)
        self.selected_point_item = None
        self.selected_point = None
        self.extrema_items = []

        host = QWidget(self)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        self.panel = InteractiveHeatmapPanel(
            title="Mapping ROI",
            x_label="X (mm)",
            y_label="Y (mm)",
            aspect_locked=True,
            with_profiles=True,
        )
        layout.addWidget(self.panel, 1)

        toolbar = QHBoxLayout()
        self.btn_export = QPushButton("匯出範圍圖片與數值")
        self.btn_export.clicked.connect(self.export_roi)
        toolbar.addWidget(self.btn_export)
        self.lbl_point_info = QLabel("滑鼠位置: X=--, Y=--, Value=--")
        toolbar.addWidget(self.lbl_point_info, 1)
        self.lbl_extrema_info = QLabel("最大/最小: --")
        toolbar.addWidget(self.lbl_extrema_info, 1)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        self.setCentralWidget(host)

        rect = self._image_rect()
        vmin, vmax = self._finite_minmax()
        self.panel.set_image(
            self.matrix.T,
            rect=rect,
            levels=(vmin, vmax),
            reset_view=True,
            source_matrix=self.matrix,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
        )
        self._configure_grid()
        self._add_extrema_markers()
        self.panel.mouseMoved.connect(self._on_mouse_moved)
        self.panel.plot.scene().sigMouseClicked.connect(self._on_clicked)
        self.panel.crossVisibilityChanged.connect(self._on_cross_visibility_changed)

    def _finite_minmax(self):
        finite = self.matrix[np.isfinite(self.matrix)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _image_rect(self):
        dx = float(np.median(np.diff(self.x_coords))) if self.x_coords.size > 1 else 1.0
        dy = float(np.median(np.diff(self.y_coords))) if self.y_coords.size > 1 else 1.0
        return QRectF(
            float(self.x_coords[0]) - dx / 2,
            float(self.y_coords[0]) - dy / 2,
            dx * self.matrix.shape[1],
            dy * self.matrix.shape[0],
        )

    def _configure_grid(self):
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_tick_spacing"):
            x_label, x_grid = parent._tick_spacing(self.x_coords)
            y_label, y_grid = parent._tick_spacing(self.y_coords)
        else:
            x_label, x_grid = axis_tick_spacing(self.x_coords)
            y_label, y_grid = axis_tick_spacing(self.y_coords)
        self.panel.set_tick_spacing(x_label, x_grid, y_label, y_grid, grid_alpha=0.25)

    def _add_extrema_markers(self):
        """在 ROI 內以 X 標示最大值與最小值。"""
        finite_mask = np.isfinite(self.matrix)
        if not np.any(finite_mask):
            self.lbl_extrema_info.setText("最大/最小: 無有效數值")
            return

        finite_values = np.where(finite_mask, self.matrix, np.nan)
        max_index = np.unravel_index(np.nanargmax(finite_values), self.matrix.shape)
        min_index = np.unravel_index(np.nanargmin(finite_values), self.matrix.shape)
        max_y, max_x = max_index
        min_y, min_x = min_index
        max_value = float(self.matrix[max_y, max_x])
        min_value = float(self.matrix[min_y, min_x])

        max_item = pg.ScatterPlotItem(
            x=[self.x_coords[max_x]], y=[self.y_coords[max_y]],
            symbol="x", size=26,
            pen=pg.mkPen("#7F0000", width=5),
            brush=None,
        )
        min_item = pg.ScatterPlotItem(
            x=[self.x_coords[min_x]], y=[self.y_coords[min_y]],
            symbol="x", size=26,
            pen=pg.mkPen("#001B5E", width=5),
            brush=None,
        )
        self.panel.plot.addItem(max_item, ignoreBounds=True)
        self.panel.plot.addItem(min_item, ignoreBounds=True)
        self.extrema_items.extend([max_item, min_item])
        self.lbl_extrema_info.setText(
            f"最大 X: ({self.x_coords[max_x]:.3f}, {self.y_coords[max_y]:.3f}) "
            f"= {max_value:.6f} ｜ 最小 X: ({self.x_coords[min_x]:.3f}, "
            f"{self.y_coords[min_y]:.3f}) = {min_value:.6f}"
        )

    def _on_cross_visibility_changed(self, visible):
        """隱藏十字時隱藏最大／最小 X 標記；文字說明仍保留。"""
        for item in self.extrema_items:
            try:
                item.setVisible(visible)
            except Exception:
                pass

    def _on_mouse_moved(self, point):
        if self.selected_point is not None:
            return
        if point is None:
            self.lbl_point_info.setText("滑鼠位置: X=--, Y=--, Value=--")
            return
        ix = int(np.abs(self.x_coords - point.x()).argmin())
        iy = int(np.abs(self.y_coords - point.y()).argmin())
        value = self.matrix[iy, ix]
        value_text = "NaN" if np.isnan(value) else f"{value:.6f}"
        self.lbl_point_info.setText(
            f"滑鼠位置: X={self.x_coords[ix]:.3f}, "
            f"Y={self.y_coords[iy]:.3f}, Value={value_text}"
        )

    def _on_clicked(self, event):
        if event.double():
            self._reset_roi_view()
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        if not self.panel.plot.sceneBoundingRect().contains(pos):
            return
        point = self.panel.plot.getViewBox().mapSceneToView(pos)
        ix = int(np.abs(self.x_coords - point.x()).argmin())
        iy = int(np.abs(self.y_coords - point.y()).argmin())
        value = self.matrix[iy, ix]
        value_text = "NaN" if np.isnan(value) else f"{value:.6f}"
        self.selected_point = (ix, iy)
        self.lbl_point_info.setText(
            f"已固定: X={self.x_coords[ix]:.3f}, "
            f"Y={self.y_coords[iy]:.3f}, Value={value_text}"
        )
        self.panel.set_profile_point(ix, iy, reset_view=False)

        self._clear_selected_point()
        x, y = float(self.x_coords[ix]), float(self.y_coords[iy])
        x0, x1 = self._cell_bounds(self.x_coords, ix)
        y0, y1 = self._cell_bounds(self.y_coords, iy)
        self.selected_point_item = pg.PlotDataItem(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            pen=pg.mkPen("#FF0000", width=3),
        )
        self.panel.plot.addItem(self.selected_point_item, ignoreBounds=True)

    def _reset_roi_view(self):
        """雙擊 ROI 圖面時回到完整資料範圍（與主圖相同：解鎖比例鋪滿）。"""
        self.panel.reset_view()

    @staticmethod
    def _cell_bounds(coords, index):
        values = np.asarray(coords, dtype=float)
        if values.size < 2:
            value = float(values[0]) if values.size else 0.0
            return value - 0.5, value + 0.5
        left = values[index - 1] if index > 0 else values[index] - (values[index + 1] - values[index])
        right = values[index + 1] if index < values.size - 1 else values[index] + (values[index] - values[index - 1])
        return float((left + values[index]) / 2), float((values[index] + right) / 2)

    def _clear_selected_point(self):
        if self.selected_point_item is not None:
            self.panel.plot.removeItem(self.selected_point_item)
            self.selected_point_item = None

    def export_roi(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出 Mapping 範圍",
            "mapping_roi.png",
            "PNG 圖片 (*.png);;JPEG 圖片 (*.jpg);;所有檔案 (*)",
        )
        if not save_path:
            return
        try:
            base = os.path.splitext(save_path)[0]
            exporter = pg_export.ImageExporter(self.panel.plot)
            exporter.parameters()["width"] = max(int(self.panel.plot.width()), 400)
            exporter.parameters()["height"] = max(int(self.panel.plot.height()), 300)
            selected_item = self.selected_point_item
            if selected_item is not None:
                selected_item.hide()
            try:
                exporter.export(save_path)
            finally:
                if selected_item is not None:
                    selected_item.show()

            csv_path = f"{base}.csv"
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
                fh.write("x,y,value\n")
                for iy, y in enumerate(self.y_coords):
                    for ix, x in enumerate(self.x_coords):
                        fh.write(f"{x:.6f},{y:.6f},{self.matrix[iy, ix]:.6f}\n")
            np.save(f"{base}.npy", self.matrix)
            QMessageBox.information(
                self,
                "匯出完成",
                f"圖片：\n{save_path}\n\nCSV：\n{csv_path}\n\nNPY：\n{base}.npy",
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出範圍資料：\n{exc}")


def _grid_step_from_coords(coords, fallback=1.0):
    values = np.asarray(coords, dtype=float)
    if values.size < 2:
        return float(fallback)
    diffs = np.abs(np.diff(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float(fallback)
    return float(np.median(diffs))


def _nice_number(x):
    """取 1/2/5×10^n，讓軸刻度在不同螢幕／縮放下都好讀。"""
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    frac = x / (10.0 ** exp)
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.0:
        nice = 2.0
    elif frac < 7.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0 ** exp))


def axis_tick_spacing(coords, target_major_labels=8):
    """回傳 (major, minor)。

    不再把次格線設成資料 step(如 0.2mm)：視野一大就會畫出上百條線，
    在另一台高 DPI／不同解析度電腦上特別容易糊成怪格線。
    """
    values = np.asarray(coords, dtype=float)
    if values.size == 0:
        return 1.0, 0.2
    span = float(np.nanmax(values) - np.nanmin(values))
    if not np.isfinite(span) or span <= 0:
        step = _grid_step_from_coords(coords)
        return step, step
    major = _nice_number(span / max(float(target_major_labels), 1.0))
    minor = major / 5.0
    # 次格線也不要比資料步長細太多（避免無意義過密）
    data_step = _grid_step_from_coords(coords)
    if data_step > 0 and minor < data_step * 0.5:
        minor = data_step
        if major < minor:
            major = _nice_number(minor * 5.0)
    return float(major), float(minor)


def compute_mapping_gradient_maps(matrix, x_coords, y_coords):
    """計算四張梯度圖。

    1) gx = [(x+1,y)-(x,y)] / step_x，省略每列最後一點
    2) gy = [(x,y+1)-(x,y)] / step_y，省略每欄最後一點
    3) (gx+gy)/2，兩者皆可算的格子（同時省略最後列與最後欄）
    4) arctan(gy/gx) * 180/π
    """
    matrix = np.asarray(matrix, dtype=float)
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("矩陣必須是二維。")
    ny, nx = matrix.shape
    if nx < 2 or ny < 2:
        raise ValueError("至少需要 2×2 個點才能計算梯度。")
    if x_coords.size != nx or y_coords.size != ny:
        raise ValueError("座標長度與矩陣尺寸不一致。")

    step_x = _grid_step_from_coords(x_coords)
    step_y = _grid_step_from_coords(y_coords)
    if step_x <= 0 or step_y <= 0:
        raise ValueError("無法從座標推算有效 step。")

    left = matrix[:, :-1]
    right = matrix[:, 1:]
    gx = (right - left) / step_x

    below = matrix[:-1, :]
    above = matrix[1:, :]
    gy = (above - below) / step_y

    gx_c = gx[:-1, :]
    gy_c = gy[:, :-1]
    g_avg = (gx_c + gy_c) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g_ang = np.degrees(np.arctan2(gy_c, gx_c))

    return {
        "step_x": step_x,
        "step_y": step_y,
        "gx": gx,
        "gy": gy,
        "g_avg": g_avg,
        "g_ang": g_ang,
        "x_gx": x_coords[:-1],
        "y_gx": y_coords,
        "x_gy": x_coords,
        "y_gy": y_coords[:-1],
        "x_c": x_coords[:-1],
        "y_c": y_coords[:-1],
    }


class MappingGradientWindow(QMainWindow):
    """顯示 Mapping 四張梯度／角度分析圖（可調平均半寬度、點選、分開匯出）。"""

    _PLOT_SPECS = (
        ("gx", "① X 向梯度"),
        ("gy", "② Y 向梯度"),
        ("g_avg", "③ 平均梯度"),
        ("g_ang", "④ 梯度角度 [deg]"),
    )

    def __init__(self, matrix, x_coords, y_coords, parent=None, avg_size=1):
        super().__init__(parent)
        self.source_matrix = np.asarray(matrix, dtype=float)
        self.x_coords = np.asarray(x_coords, dtype=float)
        self.y_coords = np.asarray(y_coords, dtype=float)
        self.maps = None
        self.plot_slots = []  # per-panel state
        self._exporting = False

        self.setWindowTitle("Mapping 梯度分析")
        self.resize(1400, 920)

        host = QWidget(self)
        root = QVBoxLayout(host)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("平均半寬度 (點數)"))
        self.edit_avg_size = QLineEdit()
        self.edit_avg_size.setFixedWidth(72)
        self.edit_avg_size.setValidator(QIntValidator(1, 999, self))
        self.edit_avg_size.setText(str(max(1, int(avg_size))))
        self.edit_avg_size.setToolTip("1 = 不平滑；越大越平滑。變更後立即重算四張圖。")
        self.edit_avg_size.editingFinished.connect(self._on_avg_changed)
        toolbar.addWidget(self.edit_avg_size)

        self.btn_apply_avg = QPushButton("套用平均")
        self.btn_apply_avg.clicked.connect(self._on_avg_changed)
        toolbar.addWidget(self.btn_apply_avg)

        self.btn_export_all = QPushButton("匯出四張圖（分開檔案）")
        self.btn_export_all.setStyleSheet(
            "QPushButton { font-weight: bold; color: white; background-color: #0288D1; "
            "border: none; border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #039BE5; }"
        )
        self.btn_export_all.setToolTip(
            "各存一張 PNG；匯出時不含十字、X 標示、colorbar 與紅色選取框。"
        )
        self.btn_export_all.clicked.connect(self.export_four_plots)
        toolbar.addWidget(self.btn_export_all)

        self.lbl_step_info = QLabel("")
        self.lbl_step_info.setStyleSheet("color: #546E7A; font-size: 12px;")
        toolbar.addWidget(self.lbl_step_info, 1)
        root.addLayout(toolbar)

        # 操作提示只顯示一次（紅色方框區域），不在四個畫布各自重複
        tip = QLabel(
            "滾輪縮放主圖／剖面（剖面不跟主圖連動）· 點擊更新剖面 · "
            "雙擊主圖依比例復原 · 雙擊剖面復原　｜　"
            "隱藏十字及X標示時一併隱藏最大／最小 X 標記"
        )
        tip.setStyleSheet("color: #607D8B; font-size: 11px;")
        tip.setWordWrap(True)
        root.addWidget(tip)

        grid = QGridLayout()
        grid.setSpacing(6)
        root.addLayout(grid, 1)

        positions = ((0, 0), (0, 1), (1, 0), (1, 1))
        for (row, col), (key, title) in zip(positions, self._PLOT_SPECS):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)

            # 色條保留；操作說明改放視窗上方共用
            panel = InteractiveHeatmapPanel(
                title=title,
                x_label="X (mm)",
                y_label="Y (mm)",
                aspect_locked=True,
                with_profiles=True,
                show_colorbar=True,
                show_tip=False,
            )
            # 個別「匯出當前圖檔」時也隱藏紅框
            try:
                panel.btn_export.clicked.disconnect()
            except TypeError:
                pass
            panel.btn_export.clicked.connect(
                lambda _checked=False, k=key: self._export_one_plot(k)
            )

            lbl_info = QLabel("點選: X=--, Y=--, Value=--")
            lbl_info.setStyleSheet("color: #37474F; font-size: 11px;")
            lbl_extrema = QLabel("最大/最小: --")
            lbl_extrema.setStyleSheet("color: #37474F; font-size: 11px;")

            cell_layout.addWidget(panel, 1)
            cell_layout.addWidget(lbl_info)
            cell_layout.addWidget(lbl_extrema)
            grid.addWidget(cell, row, col)

            slot = {
                "key": key,
                "title": title,
                "panel": panel,
                "lbl_info": lbl_info,
                "lbl_extrema": lbl_extrema,
                "matrix": None,
                "x_coords": None,
                "y_coords": None,
                "selected_point": None,
                "selected_item": None,
                "extrema_items": [],
            }
            self.plot_slots.append(slot)
            panel.mouseMoved.connect(
                lambda pt, s=slot: self._on_panel_mouse_moved(s, pt)
            )
            panel.plot.scene().sigMouseClicked.connect(
                lambda evt, s=slot: self._on_panel_clicked(s, evt)
            )
            panel.crossVisibilityChanged.connect(
                lambda visible, s=slot: self._on_cross_visibility_changed(s, visible)
            )

        self.setCentralWidget(host)
        self._rebuild_plots(reset_view=True)

    def _current_avg_size(self):
        text = self.edit_avg_size.text().strip() or "1"
        try:
            return max(1, int(text))
        except ValueError:
            return 1

    def _smooth_matrix(self, matrix, kernel_size):
        matrix = np.asarray(matrix, dtype=float)
        if kernel_size <= 1:
            return matrix
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_smooth_matrix"):
            return parent._smooth_matrix(matrix, kernel_size)
        padded = np.pad(matrix, kernel_size // 2, mode="reflect")
        smoothed = np.full(matrix.shape, np.nan, dtype=float)
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                block = padded[yi:yi + kernel_size, xi:xi + kernel_size]
                if np.isnan(block).all():
                    smoothed[yi, xi] = np.nan
                else:
                    smoothed[yi, xi] = np.nanmean(block)
        return smoothed

    def _on_avg_changed(self):
        size = self._current_avg_size()
        self.edit_avg_size.setText(str(size))
        self._rebuild_plots(reset_view=False)

    def _rebuild_plots(self, reset_view=False):
        avg = self._current_avg_size()
        smoothed = self._smooth_matrix(self.source_matrix, avg)
        maps = compute_mapping_gradient_maps(smoothed, self.x_coords, self.y_coords)
        self.maps = maps
        self.setWindowTitle(
            f"Mapping 梯度分析  |  avg={avg}  |  "
            f"step_x={maps['step_x']:.4g}  step_y={maps['step_y']:.4g}"
        )
        self.lbl_step_info.setText(
            f"平均半寬度={avg}　｜　"
            f"step_x={maps['step_x']:.6g} mm　｜　step_y={maps['step_y']:.6g} mm　｜　"
            f"① X差分 ② Y差分 ③ 平均 ④ 角度"
        )

        data_by_key = {
            "gx": (maps["gx"], maps["x_gx"], maps["y_gx"]),
            "gy": (maps["gy"], maps["x_gy"], maps["y_gy"]),
            "g_avg": (maps["g_avg"], maps["x_c"], maps["y_c"]),
            "g_ang": (maps["g_ang"], maps["x_c"], maps["y_c"]),
        }
        for slot in self.plot_slots:
            data, xs, ys = data_by_key[slot["key"]]
            slot["matrix"] = data
            slot["x_coords"] = np.asarray(xs, dtype=float)
            slot["y_coords"] = np.asarray(ys, dtype=float)
            self._clear_selected_point(slot)
            self._clear_extrema_markers(slot)
            slot["selected_point"] = None
            slot["lbl_info"].setText("點選: X=--, Y=--, Value=--")
            rect = self._image_rect(data, slot["x_coords"], slot["y_coords"])
            vmin, vmax = self._finite_minmax(data)
            slot["panel"].set_image(
                data.T,
                rect=rect,
                levels=(vmin, vmax),
                reset_view=reset_view,
                source_matrix=data,
                x_coords=slot["x_coords"],
                y_coords=slot["y_coords"],
            )
            self._configure_grid(slot["panel"], slot["x_coords"], slot["y_coords"])
            self._add_extrema_markers(slot)

    def _on_panel_mouse_moved(self, slot, point):
        if slot["selected_point"] is not None:
            return
        if point is None or slot["matrix"] is None:
            slot["lbl_info"].setText("點選: X=--, Y=--, Value=--")
            return
        ix, iy = self._cell_index(slot, point.x(), point.y())
        self._set_info_text(slot, ix, iy, prefix="滑鼠")

    def _on_panel_clicked(self, slot, event):
        if event.double():
            return  # 雙擊交給 panel 復原視野
        if event.button() not in (Qt.LeftButton, Qt.NoButton):
            return
        if slot["matrix"] is None:
            return
        pos = event.scenePos()
        panel = slot["panel"]
        try:
            if panel.hist.sceneBoundingRect().contains(pos):
                return
        except Exception:
            pass
        if not panel.plot.sceneBoundingRect().contains(pos):
            return
        point = panel.plot.getViewBox().mapSceneToView(pos)
        ix, iy = self._cell_index(slot, point.x(), point.y())
        slot["selected_point"] = (ix, iy)
        self._set_info_text(slot, ix, iy, prefix="已固定")
        self._draw_selected_box(slot, ix, iy)
        panel.set_profile_point(ix, iy, reset_view=False)

    def _cell_index(self, slot, x, y):
        xs = slot["x_coords"]
        ys = slot["y_coords"]
        ix = int(np.clip(np.abs(xs - float(x)).argmin(), 0, xs.size - 1))
        iy = int(np.clip(np.abs(ys - float(y)).argmin(), 0, ys.size - 1))
        return ix, iy

    def _set_info_text(self, slot, ix, iy, prefix="點選"):
        value = slot["matrix"][iy, ix]
        value_text = "NaN" if not np.isfinite(value) else f"{float(value):.6g}"
        slot["lbl_info"].setText(
            f"{prefix}: X={slot['x_coords'][ix]:.3f}, "
            f"Y={slot['y_coords'][iy]:.3f}, Value={value_text}"
        )

    def _draw_selected_box(self, slot, ix, iy):
        self._clear_selected_point(slot)
        x0, x1 = self._cell_bounds(slot["x_coords"], ix)
        y0, y1 = self._cell_bounds(slot["y_coords"], iy)
        item = pg.PlotDataItem(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            pen=pg.mkPen("#FF0000", width=3),
        )
        slot["panel"].plot.addItem(item, ignoreBounds=True)
        slot["selected_item"] = item

    def _clear_selected_point(self, slot):
        item = slot.get("selected_item")
        if item is not None:
            try:
                slot["panel"].plot.removeItem(item)
            except Exception:
                pass
            slot["selected_item"] = None

    def _clear_extrema_markers(self, slot):
        for item in slot.get("extrema_items") or []:
            try:
                slot["panel"].plot.removeItem(item)
            except Exception:
                pass
        slot["extrema_items"] = []
        lbl = slot.get("lbl_extrema")
        if lbl is not None:
            lbl.setText("最大/最小: --")

    def _add_extrema_markers(self, slot):
        """與範圍檢視相同：以 X 標示最大值與最小值位置。"""
        self._clear_extrema_markers(slot)
        matrix = slot.get("matrix")
        xs = slot.get("x_coords")
        ys = slot.get("y_coords")
        lbl = slot.get("lbl_extrema")
        if matrix is None or xs is None or ys is None:
            return

        finite_mask = np.isfinite(matrix)
        if not np.any(finite_mask):
            if lbl is not None:
                lbl.setText("最大/最小: 無有效數值")
            return

        finite_values = np.where(finite_mask, matrix, np.nan)
        max_y, max_x = np.unravel_index(np.nanargmax(finite_values), matrix.shape)
        min_y, min_x = np.unravel_index(np.nanargmin(finite_values), matrix.shape)
        max_value = float(matrix[max_y, max_x])
        min_value = float(matrix[min_y, min_x])

        max_item = pg.ScatterPlotItem(
            x=[float(xs[max_x])], y=[float(ys[max_y])],
            symbol="x", size=26,
            pen=pg.mkPen("#7F0000", width=5),
            brush=None,
        )
        min_item = pg.ScatterPlotItem(
            x=[float(xs[min_x])], y=[float(ys[min_y])],
            symbol="x", size=26,
            pen=pg.mkPen("#001B5E", width=5),
            brush=None,
        )
        slot["panel"].plot.addItem(max_item, ignoreBounds=True)
        slot["panel"].plot.addItem(min_item, ignoreBounds=True)
        slot["extrema_items"] = [max_item, min_item]
        if lbl is not None:
            lbl.setText(
                f"最大 X: ({float(xs[max_x]):.3f}, {float(ys[max_y]):.3f}) "
                f"= {max_value:.6g} ｜ 最小 X: ({float(xs[min_x]):.3f}, "
                f"{float(ys[min_y]):.3f}) = {min_value:.6g}"
            )
        # 若目前為「隱藏十字」，極值標記一併隱藏
        self._apply_extrema_visibility(slot)

    def _on_cross_visibility_changed(self, slot, visible):
        """隱藏十字時，最大／最小 X 標記一併隱藏；下方文字說明仍保留。"""
        self._apply_extrema_visibility(slot, visible=visible)

    def _apply_extrema_visibility(self, slot, visible=None):
        if visible is None:
            panel = slot.get("panel")
            visible = True if panel is None else bool(
                getattr(panel, "_show_profile_cross", True)
            )
        for item in slot.get("extrema_items") or []:
            try:
                item.setVisible(visible)
            except Exception:
                pass
        # 下方「最大/最小」文字說明不隱藏

    def _hide_all_selection_boxes(self):
        hidden = []
        for slot in self.plot_slots:
            item = slot.get("selected_item")
            if item is not None:
                item.hide()
                hidden.append(item)
        return hidden

    def _prepare_clean_export_overlays(self):
        """四張一起匯出時暫時隱藏十字、X 標示、色條與選取框；回傳還原函式。"""
        restore_fns = []

        # 紅色選取框
        for item in self._hide_all_selection_boxes():
            restore_fns.append(lambda it=item: it.show())

        for slot in self.plot_slots:
            panel = slot.get("panel")
            if panel is None:
                continue

            # 剖面十字
            cross_items = list(getattr(panel, "_profile_cross_items", []) or [])
            for item in cross_items:
                try:
                    was_visible = item.isVisible()
                    item.hide()
                    restore_fns.append(
                        lambda it=item, vis=was_visible: it.setVisible(vis)
                    )
                except Exception:
                    pass

            # 最大／最小 X 標示
            for item in slot.get("extrema_items") or []:
                try:
                    was_visible = item.isVisible()
                    item.hide()
                    restore_fns.append(
                        lambda it=item, vis=was_visible: it.setVisible(vis)
                    )
                except Exception:
                    pass

            # colorbar
            hist = getattr(panel, "hist", None)
            if hist is not None:
                try:
                    was_visible = hist.isVisible()
                    hist.hide()
                    restore_fns.append(
                        lambda h=hist, vis=was_visible: h.setVisible(vis)
                    )
                except Exception:
                    pass

        def _restore():
            for fn in restore_fns:
                try:
                    fn()
                except Exception:
                    pass

        return _restore

    @staticmethod
    def _show_items(items):
        for item in items:
            try:
                item.show()
            except Exception:
                pass

    def _export_one_plot(self, key):
        slot = next((s for s in self.plot_slots if s["key"] == key), None)
        if slot is None:
            return
        safe = slot["title"].replace(" ", "_").replace("/", "-")
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"匯出 {slot['title']}",
            f"{safe}.png",
            "PNG 圖片 (*.png);;JPEG 圖片 (*.jpg);;所有檔案 (*)",
        )
        if not path:
            return
        hidden = self._hide_all_selection_boxes()
        try:
            target = getattr(slot["panel"].win, "ci", None) or slot["panel"].plot
            exporter = pg_export.ImageExporter(target)
            exporter.parameters()["width"] = max(int(slot["panel"].win.width()), 400)
            exporter.parameters()["height"] = max(int(slot["panel"].win.height()), 300)
            exporter.export(path)
            QMessageBox.information(self, "匯出完成", f"已匯出：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出圖檔：\n{exc}")
        finally:
            self._show_items(hidden)

    def export_four_plots(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇四張圖的儲存資料夾", "")
        if not folder:
            return
        avg = self._current_avg_size()
        names = {
            "gx": f"01_X_gradient_avg{avg}.png",
            "gy": f"02_Y_gradient_avg{avg}.png",
            "g_avg": f"03_mean_gradient_avg{avg}.png",
            "g_ang": f"04_gradient_angle_avg{avg}.png",
        }
        restore = self._prepare_clean_export_overlays()
        # 等 layout 收掉 colorbar 再匯出，避免空白邊
        QApplication.processEvents()
        saved = []
        try:
            for slot in self.plot_slots:
                path = os.path.join(folder, names[slot["key"]])
                target = getattr(slot["panel"].win, "ci", None) or slot["panel"].plot
                exporter = pg_export.ImageExporter(target)
                exporter.parameters()["width"] = max(int(slot["panel"].win.width()), 400)
                exporter.parameters()["height"] = max(int(slot["panel"].win.height()), 300)
                exporter.export(path)
                saved.append(path)
            QMessageBox.information(
                self,
                "匯出完成",
                "已分開匯出四張圖（不含十字、X 標示、colorbar）：\n"
                + "\n".join(saved),
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出圖檔：\n{exc}")
        finally:
            restore()
            QApplication.processEvents()

    @staticmethod
    def _cell_bounds(coords, index):
        values = np.asarray(coords, dtype=float)
        if values.size < 2:
            value = float(values[0]) if values.size else 0.0
            return value - 0.5, value + 0.5
        left = (
            values[index - 1]
            if index > 0
            else values[index] - (values[index + 1] - values[index])
        )
        right = (
            values[index + 1]
            if index < values.size - 1
            else values[index] + (values[index] - values[index - 1])
        )
        return float((left + values[index]) / 2), float((values[index] + right) / 2)

    @staticmethod
    def _finite_minmax(matrix):
        finite = np.asarray(matrix, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    @staticmethod
    def _image_rect(matrix, x_coords, y_coords):
        dx = _grid_step_from_coords(x_coords)
        dy = _grid_step_from_coords(y_coords)
        return QRectF(
            float(np.min(x_coords)) - dx / 2.0,
            float(np.min(y_coords)) - dy / 2.0,
            dx * matrix.shape[1],
            dy * matrix.shape[0],
        )

    @staticmethod
    def _configure_grid(panel, x_coords, y_coords):
        x_major, x_minor = axis_tick_spacing(x_coords)
        y_major, y_minor = axis_tick_spacing(y_coords)
        panel.set_tick_spacing(x_major, x_minor, y_major, y_minor, grid_alpha=0.25)


class MappingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.save_dir_path = ""
        self.export_dir = ""
        self.mapping_path = ""
        self.mapping_matrix = None
        self.mapping_matrix_f1 = None
        self.mapping_matrix_f2 = None
        self.x_coords = None
        self.y_coords = None
        self.x_coords_f2 = None
        self.y_coords_f2 = None
        self.mapping_f1_path = ""
        self.mapping_f2_path = ""
        self.plot_drawn = False
        self.contour_line_items = []
        self.roi_rect_item = None
        self.roi_window = None
        self.gradient_window = None
        self.selected_point_item = None
        self.selected_point = None
        self._setup_ui()

    def _setup_ui(self):
        btn_style_default = """
            QPushButton { font-size: 13px; font-weight: bold; background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; padding: 6px 12px; }
            QPushButton:hover { background-color: #e0e0e0; border-color: #b0b0b0; }
            QPushButton:pressed { background-color: #d0d0d0; }
        """
        btn_style_primary = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #2E7D32; border: none; border-radius: 5px; padding: 8px 12px; }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:pressed { background-color: #1B5E20; }
        """
        btn_style_export = """
            QPushButton { font-size: 13px; font-weight: bold; color: white; background-color: #0288D1; border: none; border-radius: 5px; padding: 8px 16px; }
            QPushButton:hover { background-color: #039BE5; }
            QPushButton:pressed { background-color: #01579B; }
            QPushButton:disabled { background-color: #B0BEC5; }
        """

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #dcdcdc; width: 4px; }")
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(360)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        lbl_title = QLabel("Mapping 匯入 / 畫圖 / 匯出")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #37474F;")
        left_layout.addWidget(lbl_title)

        self.combo_mapping_mode = QComboBox()
        self.combo_mapping_mode.addItem("單檔 Mapping", "single")
        self.combo_mapping_mode.addItem("F1 - F2（Chuck - Wafer）", "subtract")
        self.combo_mapping_mode.currentIndexChanged.connect(self._on_mapping_mode_changed)
        left_layout.addWidget(self.combo_mapping_mode)

        self.btn_load_mapping = QPushButton("I. 匯入 Mapping 檔案")
        self.btn_load_mapping.setStyleSheet(btn_style_default)
        self.btn_load_mapping.clicked.connect(self.load_mapping_file)
        left_layout.addWidget(self.btn_load_mapping)

        self.lbl_mapping_path = QLabel("未選擇 Mapping 檔案")
        self.lbl_mapping_path.setStyleSheet("color: #757575; font-size: 11px;")
        self.lbl_mapping_path.setWordWrap(True)
        left_layout.addWidget(self.lbl_mapping_path)

        self.btn_load_mapping_f2 = QPushButton("II. 匯入 F2（Wafer）檔案")
        self.btn_load_mapping_f2.setStyleSheet(btn_style_default)
        self.btn_load_mapping_f2.clicked.connect(self.load_mapping_file_f2)
        self.btn_load_mapping_f2.setVisible(False)
        left_layout.addWidget(self.btn_load_mapping_f2)

        self.lbl_mapping_f2_path = QLabel("未選擇 F2（Wafer）檔案")
        self.lbl_mapping_f2_path.setStyleSheet("color: #757575; font-size: 11px;")
        self.lbl_mapping_f2_path.setWordWrap(True)
        self.lbl_mapping_f2_path.setVisible(False)
        left_layout.addWidget(self.lbl_mapping_f2_path)

        self.btn_plot_mapping = QPushButton("II. 畫出 Mapping 圖")
        self.btn_plot_mapping.setStyleSheet(btn_style_primary)
        self.btn_plot_mapping.clicked.connect(self.plot_mapping)
        self.btn_plot_mapping.setEnabled(False)
        left_layout.addWidget(self.btn_plot_mapping)

        self.btn_select_export_dir = QPushButton("選擇儲存資料夾")
        self.btn_select_export_dir.setStyleSheet(btn_style_default)
        self.btn_select_export_dir.clicked.connect(self.select_export_directory)
        left_layout.addWidget(self.btn_select_export_dir)

        self.lbl_export_dir = QLabel("未選擇儲存資料夾")
        self.lbl_export_dir.setStyleSheet("color: #757575; font-size: 11px;")
        self.lbl_export_dir.setWordWrap(True)
        left_layout.addWidget(self.lbl_export_dir)

        self.btn_export_mapping = QPushButton("III. 匯出 Mapping 圖與資料")
        self.btn_export_mapping.setStyleSheet(btn_style_export)
        self.btn_export_mapping.clicked.connect(self.export_mapping)
        self.btn_export_mapping.setEnabled(False)
        left_layout.addWidget(self.btn_export_mapping)

        roi_title = QLabel("範圍框選（可隨時取消）")
        roi_title.setStyleSheet("font-weight: bold; color: #263238; font-size: 12px;")
        left_layout.addWidget(roi_title)
        self.chk_roi_enabled = QCheckBox("啟用黑框範圍")
        self.chk_roi_enabled.toggled.connect(self._on_roi_enabled)
        self.chk_roi_enabled.setEnabled(False)
        left_layout.addWidget(self.chk_roi_enabled)

        roi_grid = QGridLayout()
        self.roi_spin_x0 = self._make_roi_spin()
        self.roi_spin_x1 = self._make_roi_spin()
        self.roi_spin_y0 = self._make_roi_spin()
        self.roi_spin_y1 = self._make_roi_spin()
        for label, spin, row, col in (
            ("X 起點", self.roi_spin_x0, 0, 0),
            ("X 終點", self.roi_spin_x1, 0, 2),
            ("Y 起點", self.roi_spin_y0, 1, 0),
            ("Y 終點", self.roi_spin_y1, 1, 2),
        ):
            roi_grid.addWidget(QLabel(label), row, col)
            roi_grid.addWidget(spin, row, col + 1)
        left_layout.addLayout(roi_grid)
        self.btn_view_roi = QPushButton("查看範圍圖")
        self.btn_view_roi.clicked.connect(self.show_roi_window)
        self.btn_view_roi.setEnabled(False)
        left_layout.addWidget(self.btn_view_roi)

        self.btn_view_gradient = QPushButton("查看梯度／角度分析圖")
        self.btn_view_gradient.setStyleSheet(btn_style_default)
        self.btn_view_gradient.setToolTip(
            "① X 向梯度  ② Y 向梯度  ③ 平均  ④ 角度(度)"
        )
        self.btn_view_gradient.clicked.connect(self.show_gradient_window)
        self.btn_view_gradient.setEnabled(False)
        left_layout.addWidget(self.btn_view_gradient)

        avg_label = QLabel("平均半寬度 (點數)")
        avg_label.setStyleSheet("color: #424242; font-size: 12px;")
        left_layout.addWidget(avg_label)

        self.edit_avg_size = QLineEdit()
        self.edit_avg_size.setPlaceholderText("1")
        self.edit_avg_size.setText("1")
        self.edit_avg_size.setValidator(QIntValidator(1, 999, self))
        self.edit_avg_size.setToolTip("設定均值濾波窗口大小。1 表示不做平滑。可手動輸入多位數值。")
        self.edit_avg_size.textChanged.connect(self._on_average_changed)
        left_layout.addWidget(self.edit_avg_size)

        left_layout.addWidget(self._create_hline())

        self.lbl_status = QLabel("狀態: 等待匯入 Mapping 檔案")
        self.lbl_status.setStyleSheet("color: #1565C0; font-weight: bold; font-size: 12px;")
        left_layout.addWidget(self.lbl_status)

        self.lbl_point_count = QLabel("總點數: --")
        self.lbl_point_count.setStyleSheet("color: #37474F; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(self.lbl_point_count)

        self.lbl_mouse_info = QLabel("滑鼠位置: X=--, Y=--, Value=--")
        self.lbl_mouse_info.setStyleSheet("color: #424242; font-size: 11px;")
        self.lbl_mouse_info.setWordWrap(True)
        left_layout.addWidget(self.lbl_mouse_info)

        left_layout.addStretch(1)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(6)

        self.heatmap_panel = InteractiveHeatmapPanel(
            title="Mapping Heatmap",
            x_label="X Pixels",
            y_label="Y Pixels",
            aspect_locked=True,
            with_profiles=True,
        )
        self.contour_panel = InteractiveHeatmapPanel(
            title="Contour Plot",
            x_label="X (mm)",
            y_label="Y (mm)",
            aspect_locked=True,
            with_profiles=True,
        )

        # 相容舊屬性名稱，供匯出與等高線邏輯使用
        self.plot = self.heatmap_panel.plot
        self.image_item = self.heatmap_panel.image_item
        self.hist = self.heatmap_panel.hist
        self.contour_plot = self.contour_panel.plot
        self.contour_image_item = self.contour_panel.image_item
        self.contour_hist = self.contour_panel.hist

        self.heatmap_panel.mouseMoved.connect(self._on_heatmap_mouse_moved)
        self.heatmap_panel.plot.scene().sigMouseClicked.connect(self._on_heatmap_clicked)

        right_layout.addWidget(self.heatmap_panel, 1)
        right_layout.addWidget(self.contour_panel, 1)
        self.right_panel = right_panel
        splitter.addWidget(right_panel)

        self.setLayout(layout)

    def _make_roi_spin(self):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e9, 1e9)
        spin.setSingleStep(1.0)
        spin.setEnabled(False)
        spin.valueChanged.connect(self._on_roi_value_changed)
        return spin

    def _create_hline(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Sunken)
        frame.setStyleSheet("color: #c0c0c0;")
        frame.setFixedHeight(1)
        return frame

    def _build_image_rect(self, matrix):
        if self.x_coords is None or self.y_coords is None:
            return None
        x0, x1 = float(self.x_coords[0]), float(self.x_coords[-1])
        y0, y1 = float(self.y_coords[0]), float(self.y_coords[-1])
        dx = (x1 - x0) / max(1, len(self.x_coords) - 1)
        dy = (y1 - y0) / max(1, len(self.y_coords) - 1)
        return QRectF(x0 - dx / 2, y0 - dy / 2, dx * matrix.shape[1], dy * matrix.shape[0])

    def _finite_minmax(self, matrix):
        finite = np.asarray(matrix, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        return vmin, vmax

    def _grid_spacing(self, coords):
        """直接使用座標資料 step 畫主要格線。"""
        values = np.asarray(coords, dtype=float)
        if values.size < 2:
            return 1.0

        diffs = np.abs(np.diff(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 1.0

        data_step = float(np.median(diffs))
        return data_step

    def _tick_spacing(self, coords):
        """回傳 (文字標籤間距, 格線間距)，跨螢幕／DPI 保持可讀。"""
        return axis_tick_spacing(coords)

    def _configure_mapping_grid(self):
        """設定主要／次要格線，避免依 0.2mm 畫出過密怪格。"""
        if self.x_coords is None or self.y_coords is None:
            return

        x_label_spacing, x_grid_spacing = self._tick_spacing(self.x_coords)
        y_label_spacing, y_grid_spacing = self._tick_spacing(self.y_coords)
        self.heatmap_panel.set_tick_spacing(
            x_label_spacing, x_grid_spacing, y_label_spacing, y_grid_spacing, grid_alpha=0.25
        )

    def _on_mapping_mode_changed(self):
        subtract = self.combo_mapping_mode.currentData() == "subtract"
        self.btn_load_mapping.setText(
            "I. 匯入 F1（Chuck）檔案" if subtract else "I. 匯入 Mapping 檔案"
        )
        self.btn_load_mapping_f2.setVisible(subtract)
        self.lbl_mapping_f2_path.setVisible(subtract)
        if subtract:
            self.lbl_mapping_path.setText("未選擇 F1（Chuck）檔案")
        else:
            self.lbl_mapping_path.setText("未選擇 Mapping 檔案")
        self.btn_plot_mapping.setEnabled(False)
        self.btn_export_mapping.setEnabled(False)
        self.btn_view_gradient.setEnabled(False)
        self.mapping_matrix = None
        self.mapping_matrix_f1 = None
        self.mapping_matrix_f2 = None
        self.plot_drawn = False

    def _read_mapping_file(self, file_path):
        if file_path.lower().endswith(".npy"):
            matrix = np.asarray(np.load(file_path), dtype=float)
            if matrix.ndim != 2:
                raise ValueError("NPY 必須是二維數值矩陣。")
            return matrix, np.arange(matrix.shape[1], dtype=float), np.arange(matrix.shape[0], dtype=float)

        df = pd.read_csv(file_path)
        normalized_columns = {
            str(column).replace("\ufeff", "").strip().lower(): column
            for column in df.columns
        }
        required = ("x_rel_mm", "y_rel_mm", "value")
        missing = [name for name in required if name not in normalized_columns]
        if missing:
            raise ValueError(
                "CSV 檔案缺少欄位："
                + ", ".join(missing)
                + "；需要 x_rel_mm、y_rel_mm、value。"
            )
        df = df[[normalized_columns[name] for name in required]].copy()
        if df.empty:
            raise ValueError("CSV 檔案不含有效的 x_rel_mm / y_rel_mm / value 資料。")
        x_vals = np.asarray(df["x_rel_mm"], dtype=float)
        y_vals = np.asarray(df["y_rel_mm"], dtype=float)
        z_vals = np.asarray(df["value"], dtype=float)
        x_unique = np.sort(np.unique(x_vals))
        y_unique = np.sort(np.unique(y_vals))
        if x_unique.size < 2 or y_unique.size < 2:
            raise ValueError("Mapping 檔案需有至少 2 個不同的 X/Y 座標。")
        matrix = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
        matrix[
            np.searchsorted(y_unique, y_vals),
            np.searchsorted(x_unique, x_vals),
        ] = z_vals
        return matrix, x_unique, y_unique

    def _update_mapping_import_state(self):
        subtract = self.combo_mapping_mode.currentData() == "subtract"
        ready = self.mapping_matrix_f1 is not None
        if subtract:
            ready = ready and self.mapping_matrix_f2 is not None
        self.btn_plot_mapping.setEnabled(ready)

    def load_mapping_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 Mapping 檔案",
            "",
            "CSV 檔 (*.csv);;NumPy 檔 (*.npy);;文字檔 (*.txt);;所有檔案 (*)"
        )
        if not file_path:
            return

        try:
            matrix, x_coords, y_coords = self._read_mapping_file(file_path)
            self.mapping_matrix_f1 = matrix
            self.x_coords = x_coords
            self.y_coords = y_coords
            self.mapping_f1_path = file_path
            if self.combo_mapping_mode.currentData() == "subtract":
                self.lbl_mapping_path.setText(f"F1（Chuck）: {os.path.basename(file_path)}")
                self.lbl_status.setText(f"狀態: 已匯入 F1（Chuck）{os.path.basename(file_path)}")
            else:
                self.lbl_mapping_path.setText(os.path.basename(file_path))
                self.lbl_status.setText(f"狀態: 已匯入 {os.path.basename(file_path)}")
            self._update_mapping_import_state()
            self.btn_export_mapping.setEnabled(False)
            self.chk_roi_enabled.setEnabled(True)
            self._set_roi_spin_ranges()
            self.plot_drawn = False
            self._update_point_count_label()
        except Exception as exc:
            QMessageBox.critical(self, "匯入失敗", f"無法讀取檔案：\n{str(exc)}")
            self.lbl_status.setText("狀態: 匯入失敗，請選擇有效的 Mapping 檔案")
            self.mapping_matrix = None
            self.mapping_matrix_f1 = None
            self.mapping_matrix_f2 = None
            self.x_coords = None
            self.y_coords = None
            self.btn_plot_mapping.setEnabled(False)
            self.btn_export_mapping.setEnabled(False)
            self.chk_roi_enabled.setEnabled(False)
            self._update_point_count_label()
            self.btn_view_roi.setEnabled(False)
            self.btn_view_gradient.setEnabled(False)
            self._clear_roi_overlay()

    def load_mapping_file_f2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 F2（Wafer）Mapping 檔案", "",
            "CSV 檔 (*.csv);;NumPy 檔 (*.npy);;所有檔案 (*)"
        )
        if not file_path:
            return
        try:
            matrix, x_coords, y_coords = self._read_mapping_file(file_path)
            if self.mapping_matrix_f1 is not None:
                if not np.array_equal(self.x_coords, x_coords) or not np.array_equal(self.y_coords, y_coords):
                    raise ValueError("F1（Chuck）與 F2（Wafer）的 X/Y 座標網格不一致。")
                if self.mapping_matrix_f1.shape != matrix.shape:
                    raise ValueError("F1（Chuck）與 F2（Wafer）的矩陣尺寸不一致。")
            self.mapping_matrix_f2 = matrix
            self.x_coords_f2 = x_coords
            self.y_coords_f2 = y_coords
            self.mapping_f2_path = file_path
            self.lbl_mapping_f2_path.setText(f"F2（Wafer）: {os.path.basename(file_path)}")
            self.lbl_status.setText(f"狀態: 已匯入 F2（Wafer）{os.path.basename(file_path)}")
            self._update_mapping_import_state()
        except Exception as exc:
            QMessageBox.critical(self, "F2 匯入失敗", f"無法讀取 F2（Wafer）：\n{exc}")

    def plot_mapping(self):
        if self.mapping_matrix_f1 is None:
            QMessageBox.warning(self, "提醒", "請先匯入 F1（Chuck）或 Mapping 檔案。")
            return

        if self.combo_mapping_mode.currentData() == "subtract":
            if self.mapping_matrix_f2 is None:
                QMessageBox.warning(self, "提醒", "請先匯入 F2（Wafer）檔案。")
                return
            self.mapping_matrix = self.mapping_matrix_f1 - self.mapping_matrix_f2
            plot_title = "Mapping Heatmap（F1 Chuck - F2 Wafer）"
        else:
            self.mapping_matrix = self.mapping_matrix_f1.copy()
            plot_title = "Mapping Heatmap (mm)"

        raw_matrix = self.mapping_matrix
        processed_matrix = self._get_processed_matrix()
        rect = self._build_image_rect(raw_matrix)
        raw_min, raw_max = self._finite_minmax(raw_matrix)

        if rect is not None:
            self.heatmap_panel.set_axis_labels("X (mm)", "Y (mm)")
            self.heatmap_panel.set_plot_title(plot_title)
        else:
            self.heatmap_panel.set_axis_labels("X Pixels", "Y Pixels")
            self.heatmap_panel.set_plot_title("Mapping Heatmap")

        self.heatmap_panel.set_image(
            raw_matrix.T,
            rect=rect,
            levels=(raw_min, raw_max),
            reset_view=True,
            source_matrix=raw_matrix,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
        )
        self._configure_mapping_grid()

        self._update_contour_plot(processed_matrix, reset_view=True)

        self.lbl_status.setText("狀態: Mapping 圖已繪製")
        self._update_point_count_label()
        self.plot_drawn = True
        self.btn_export_mapping.setEnabled(bool(self.export_dir))
        self.btn_view_gradient.setEnabled(True)
        self._update_roi_overlay()

    def _update_point_count_label(self):
        """狀態下方顯示有效點數（非 NaN）。"""
        matrix = self.mapping_matrix
        if matrix is None:
            matrix = self.mapping_matrix_f1
        if matrix is None:
            self.lbl_point_count.setText("總點數: --")
            return
        finite_n = int(np.sum(np.isfinite(matrix)))
        total_n = int(np.asarray(matrix).size)
        if finite_n == total_n:
            self.lbl_point_count.setText(f"總點數: {finite_n}")
        else:
            self.lbl_point_count.setText(f"總點數: {finite_n}（網格 {total_n}）")

    def _set_roi_spin_ranges(self):
        if self.x_coords is None or self.y_coords is None:
            return
        for spin, coords in (
            (self.roi_spin_x0, self.x_coords),
            (self.roi_spin_x1, self.x_coords),
            (self.roi_spin_y0, self.y_coords),
            (self.roi_spin_y1, self.y_coords),
        ):
            spin.blockSignals(True)
            spin.setRange(float(np.min(coords)), float(np.max(coords)))
            spin.setValue(float(coords[0] if spin in (self.roi_spin_x0, self.roi_spin_y0) else coords[-1]))
            spin.blockSignals(False)

    def _on_roi_enabled(self, enabled):
        for spin in (self.roi_spin_x0, self.roi_spin_x1, self.roi_spin_y0, self.roi_spin_y1):
            spin.setEnabled(enabled)
        self.btn_view_roi.setEnabled(enabled and self.plot_drawn)
        if enabled:
            self._update_roi_overlay()
        else:
            self._clear_roi_overlay()

    def _on_roi_value_changed(self, _value):
        if self.chk_roi_enabled.isChecked():
            self._update_roi_overlay()

    def _clear_roi_overlay(self):
        if self.roi_rect_item is not None:
            self.plot.removeItem(self.roi_rect_item)
            self.roi_rect_item = None

    def _roi_indices(self):
        if self.mapping_matrix is None:
            return None
        x0, x1 = sorted((self.roi_spin_x0.value(), self.roi_spin_x1.value()))
        y0, y1 = sorted((self.roi_spin_y0.value(), self.roi_spin_y1.value()))
        x_mask = (self.x_coords >= x0) & (self.x_coords <= x1)
        y_mask = (self.y_coords >= y0) & (self.y_coords <= y1)
        x_idx = np.flatnonzero(x_mask)
        y_idx = np.flatnonzero(y_mask)
        if x_idx.size == 0 or y_idx.size == 0:
            return None
        return x_idx, y_idx

    def _update_roi_overlay(self):
        if not self.chk_roi_enabled.isChecked():
            self._clear_roi_overlay()
            self.btn_view_roi.setEnabled(False)
            return
        indices = self._roi_indices()
        if indices is None or not self.plot_drawn:
            self.btn_view_roi.setEnabled(False)
            return
        x_idx, y_idx = indices
        x0, x1 = float(self.x_coords[x_idx[0]]), float(self.x_coords[x_idx[-1]])
        y0, y1 = float(self.y_coords[y_idx[0]]), float(self.y_coords[y_idx[-1]])
        pen = pg.mkPen("#000000", width=3)
        if self.roi_rect_item is not None:
            self.plot.removeItem(self.roi_rect_item)
        self.roi_rect_item = pg.PlotDataItem(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], pen=pen
        )
        self.plot.addItem(self.roi_rect_item, ignoreBounds=True)
        self.btn_view_roi.setEnabled(True)

    def show_roi_window(self):
        indices = self._roi_indices()
        if indices is None:
            QMessageBox.warning(self, "範圍無效", "請確認 X/Y 起點與終點涵蓋有效資料。")
            return
        x_idx, y_idx = indices
        roi_matrix = self._get_processed_matrix()[np.ix_(y_idx, x_idx)]
        self.roi_window = MappingRoiWindow(
            roi_matrix, self.x_coords[x_idx], self.y_coords[y_idx], parent=self
        )
        self.roi_window.show()

    def show_gradient_window(self):
        if self.mapping_matrix is None or self.x_coords is None or self.y_coords is None:
            QMessageBox.warning(self, "提醒", "請先畫出 Mapping 圖。")
            return
        if not self.plot_drawn:
            QMessageBox.warning(self, "提醒", "請先點「畫出 Mapping 圖」。")
            return
        try:
            avg_text = self.edit_avg_size.text().strip() or "1"
            avg_size = max(1, int(avg_text))
            if getattr(self, "gradient_window", None) is not None:
                try:
                    self.gradient_window.close()
                except Exception:
                    pass
            # 傳入原始矩陣；平均半寬度在分析視窗內即時調整
            self.gradient_window = MappingGradientWindow(
                self.mapping_matrix,
                self.x_coords,
                self.y_coords,
                parent=self,
                avg_size=avg_size,
            )
            self.gradient_window.show()
            self.lbl_status.setText("狀態: 已開啟梯度／角度分析視窗")
        except Exception as exc:
            QMessageBox.critical(self, "無法計算梯度", f"{exc}")

    def select_export_directory(self):
        export_dir = QFileDialog.getExistingDirectory(self, "選擇儲存資料夾", "")
        if export_dir:
            self.export_dir = export_dir
            self.lbl_export_dir.setText(os.path.basename(export_dir) or export_dir)
            self.btn_export_mapping.setEnabled(self.plot_drawn)
            self.lbl_status.setText("狀態: 已選擇匯出資料夾")

    def _on_average_changed(self, value):
        if self.mapping_matrix is None or not self.plot_drawn:
            return
        avg_text = self.edit_avg_size.text().strip()
        if not avg_text:
            avg_text = '1'
        avg_value = max(1, int(avg_text))
        self.edit_avg_size.setText(str(avg_value))
        self._update_contour_plot(self._get_processed_matrix(), reset_view=False)
        self.lbl_status.setText(f"狀態: 已套用 {avg_value} 點平均至 contour")

    def _get_processed_matrix(self):
        if self.mapping_matrix is None:
            return None
        avg_text = self.edit_avg_size.text().strip()
        if not avg_text:
            avg_text = '1'
        kernel_size = max(1, int(avg_text))
        if kernel_size <= 1:
            return self.mapping_matrix
        return self._smooth_matrix(self.mapping_matrix, kernel_size)

    def _smooth_matrix(self, matrix, kernel_size):
        if kernel_size <= 1:
            return matrix
        padded = np.pad(matrix, kernel_size // 2, mode='reflect')
        smoothed = np.full(matrix.shape, np.nan, dtype=float)
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                block = padded[yi:yi + kernel_size, xi:xi + kernel_size]
                if np.isnan(block).all():
                    smoothed[yi, xi] = np.nan
                else:
                    smoothed[yi, xi] = np.nanmean(block)
        return smoothed

    def _clear_contour_lines(self):
        for line in getattr(self, 'contour_line_items', []):
            self.contour_plot.removeItem(line)
        self.contour_line_items = []

    def _update_contour_plot(self, matrix, reset_view=False):
        if matrix is None:
            return

        rect = self._build_image_rect(matrix)
        contour_min, contour_max = self._finite_minmax(matrix)
        avg_text = self.edit_avg_size.text().strip() or '1'
        avg_size = int(avg_text)

        self.contour_panel.set_axis_labels("X (mm)", "Y (mm)")
        self.contour_panel.set_plot_title(f"Contour Plot ({avg_size} x {avg_size} average)")
        self.contour_panel.set_image(
            matrix.T,
            rect=rect,
            levels=(contour_min, contour_max),
            reset_view=reset_view,
            source_matrix=matrix,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
        )
        contour_x_label_spacing, contour_x_grid_spacing = self._tick_spacing(self.x_coords)
        contour_y_label_spacing, contour_y_grid_spacing = self._tick_spacing(self.y_coords)
        self.contour_panel.set_tick_spacing(
            contour_x_label_spacing,
            contour_x_grid_spacing,
            contour_y_label_spacing,
            contour_y_grid_spacing,
            grid_alpha=0.25,
        )

        self._clear_contour_lines()
        # 讓 contour 更連續：從 8 個 level 提升為 16~20 個，
        # 也避免因為某些格子含 NaN / 未跨 level 而直接中斷整條輪廓。
        level_count = 18 if contour_max > contour_min else 2
        levels = np.linspace(contour_min, contour_max, level_count)
        for level in levels:
            segments = self._marching_squares(matrix, level)
            if not segments:
                continue
            line_x = []
            line_y = []
            for segment in segments:
                if segment.shape[0] < 2:
                    continue
                line_x.extend([segment[0, 0], segment[1, 0], np.nan])
                line_y.extend([segment[0, 1], segment[1, 1], np.nan])
            if not line_x:
                continue
            line = pg.PlotDataItem(
                np.asarray(line_x, dtype=float),
                np.asarray(line_y, dtype=float),
                pen=pg.mkPen(color=(50, 50, 50), width=1),
            )
            self.contour_plot.addItem(line)
            self.contour_line_items.append(line)

        if reset_view:
            self.contour_panel.reset_view()

    def _marching_squares(self, matrix, level):
        ny, nx = matrix.shape
        segments = []

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                cell = matrix[iy:iy + 2, ix:ix + 2]
                vals = np.asarray(cell, dtype=float).reshape(-1)
                finite_mask = np.isfinite(vals)
                if not np.any(finite_mask):
                    continue

                # 若 cell 有 NaN，先用該 cell 內已有的有效值做局部補值，
                # 避免因為單點缺失直接使該格完全跳過，導致 contour 斷裂。
                if np.any(~finite_mask):
                    fill_value = float(np.nanmean(vals[finite_mask])) if np.any(finite_mask) else 0.0
                    vals = vals.copy()
                    vals[~finite_mask] = fill_value

                v0, v1, v2, v3 = vals
               
                x0, x1 = self.x_coords[ix], self.x_coords[ix + 1]
                y0, y1 = self.y_coords[iy], self.y_coords[iy + 1]
                points = []

                def interp(p1, p2, vv1, vv2):
                    if vv2 == vv1:
                        return p1
                    t = (level - vv1) / (vv2 - vv1)
                    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

                b0 = v0 >= level
                b1 = v1 >= level
                b2 = v2 >= level
                b3 = v3 >= level

                if b0 != b1:
                    points.append(interp((x0, y0), (x1, y0), v0, v1))
                if b1 != b2:
                    points.append(interp((x1, y0), (x1, y1), v1, v2))
                if b2 != b3:
                    points.append(interp((x1, y1), (x0, y1), v2, v3))
                if b3 != b0:
                    points.append(interp((x0, y1), (x0, y0), v3, v0))

                if len(points) == 2:
                    segments.append(np.array(points))
                elif len(points) == 4:
                    if (v0 + v2) > (v1 + v3):
                        segments.append(np.array([points[0], points[1]]))
                        segments.append(np.array([points[2], points[3]]))
                    else:
                        segments.append(np.array([points[0], points[3]]))
                        segments.append(np.array([points[1], points[2]]))
        return segments

    def _on_heatmap_mouse_moved(self, mouse_point):
        if self.selected_point is not None:
            return
        if mouse_point is None:
            self.lbl_mouse_info.setText("滑鼠位置: X=--, Y=--, Value=--")
            return
        x = mouse_point.x()
        y = mouse_point.y()
        if (
            self.mapping_matrix is not None
            and self.x_coords is not None
            and self.y_coords is not None
        ):
            ix = int(np.clip(np.abs(self.x_coords - x).argmin(), 0, self.mapping_matrix.shape[1] - 1))
            iy = int(np.clip(np.abs(self.y_coords - y).argmin(), 0, self.mapping_matrix.shape[0] - 1))
            value = self.mapping_matrix[iy, ix]
            value_text = 'NaN' if np.isnan(value) else f'{value:.6f}'
            self.lbl_mouse_info.setText(
                f"滑鼠位置: X={x:.3f} mm, Y={y:.3f} mm, Value={value_text}"
            )
        else:
            self.lbl_mouse_info.setText(f"滑鼠位置: X={x:.3f} mm, Y={y:.3f} mm, Value=--")

    def _on_heatmap_clicked(self, event):
        if self.mapping_matrix is None or not self.plot_drawn:
            return
        if event.double():
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return

        point = self.plot.getViewBox().mapSceneToView(pos)
        ix = int(np.abs(self.x_coords - point.x()).argmin())
        iy = int(np.abs(self.y_coords - point.y()).argmin())
        x0, x1 = MappingRoiWindow._cell_bounds(self.x_coords, ix)
        y0, y1 = MappingRoiWindow._cell_bounds(self.y_coords, iy)
        value = self.mapping_matrix[iy, ix]
        value_text = "NaN" if np.isnan(value) else f"{value:.6f}"
        self.selected_point = (ix, iy)
        self.lbl_mouse_info.setText(
            f"已固定範圍: X={x0:.3f}~{x1:.3f} mm, "
            f"Y={y0:.3f}~{y1:.3f} mm, Value={value_text}"
        )
        self.heatmap_panel.set_profile_point(ix, iy, reset_view=False)

        self._clear_selected_point()
        self.selected_point_item = pg.PlotDataItem(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            pen=pg.mkPen("#FF0000", width=3),
        )
        self.plot.addItem(self.selected_point_item, ignoreBounds=True)

    def _clear_selected_point(self):
        if self.selected_point_item is not None:
            self.plot.removeItem(self.selected_point_item)
            self.selected_point_item = None

    def export_mapping(self):
        if self.mapping_matrix is None:
            QMessageBox.warning(self, "提醒", "請先匯入並繪製 Mapping。")
            return
        if not self.export_dir:
            QMessageBox.warning(self, "提醒", "請先選擇儲存資料夾。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出 Heatmap 檔名",
            os.path.join(self.export_dir, "mapping_heatmap.png"),
            "PNG 圖片 (*.png);;JPEG 圖片 (*.jpg);;所有檔案 (*)"
        )
        if not save_path:
            return

        avg_matrix = self._get_processed_matrix()
        try:
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            ext = os.path.splitext(save_path)[1] or ".png"
            heatmap_path = save_path
            contour_path = os.path.join(self.export_dir, f"{base_name}_contour{ext}")

            # 選點紅框只供畫面判讀，匯出圖片時暫時移除。
            selected_item = self.selected_point_item
            if selected_item is not None:
                selected_item.hide()
            for panel, out_path in (
                (self.heatmap_panel, heatmap_path),
                (self.contour_panel, contour_path),
            ):
                # 匯出主圖區域（含座標與格線），不包含右側 colorbar。
                target = panel.plot
                exporter = pg_export.ImageExporter(target)
                exporter.parameters()["width"] = max(int(panel.plot.width()), 400)
                exporter.parameters()["height"] = max(int(panel.plot.height()), 300)
                exporter.export(out_path)

            csv_path = os.path.join(self.export_dir, f"{base_name}.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('x,y,value\n')
                if self.x_coords is not None and self.y_coords is not None:
                    for iy, y in enumerate(self.y_coords):
                        for ix, x in enumerate(self.x_coords):
                            value = avg_matrix[iy, ix]
                            f.write(f"{x:.6f},{y:.6f},{value:.6f}\n")

            self.lbl_status.setText("狀態: 已匯出 heatmap、contour 兩張圖與平均後 CSV")
            QMessageBox.information(
                self,
                "匯出完成",
                f"已匯出：\n{heatmap_path}\n{contour_path}\n\nCSV：\n{csv_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出檔案：\n{str(exc)}")
        finally:
            if selected_item is not None:
                selected_item.show()
