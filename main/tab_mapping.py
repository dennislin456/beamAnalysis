import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as pg_export

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QSplitter, QFrame, QLineEdit,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, QRectF

from shared_components import apply_readable_plot_theme


class MappingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.save_dir_path = ""
        self.export_dir = ""
        self.mapping_path = ""
        self.mapping_matrix = None
        self.plot_drawn = False
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

        self.btn_load_mapping = QPushButton("I. 匯入 Mapping 檔案")
        self.btn_load_mapping.setStyleSheet(btn_style_default)
        self.btn_load_mapping.clicked.connect(self.load_mapping_file)
        left_layout.addWidget(self.btn_load_mapping)

        self.lbl_mapping_path = QLabel("未選擇 Mapping 檔案")
        self.lbl_mapping_path.setStyleSheet("color: #757575; font-size: 11px;")
        self.lbl_mapping_path.setWordWrap(True)
        left_layout.addWidget(self.lbl_mapping_path)

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

        self.win = pg.GraphicsLayoutWidget()
        self.win.setStyleSheet("background-color: #E8EEF2;")
        self.plot = self.win.addPlot(row=0, col=0, title='Mapping Heatmap')
        self.plot.getViewBox().invertY(False)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.4)
        self.plot.setLabel('bottom', 'X Pixels')
        self.plot.setLabel('left', 'Y Pixels')
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        self.proxy = pg.SignalProxy(self.win.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        pos = np.linspace(0.0, 1.0, 6)
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255)]
        self.hist.gradient.setColorMap(pg.ColorMap(pos, colors))
        self.win.addItem(self.hist, row=0, col=1)

        self.contour_win = pg.GraphicsLayoutWidget()
        self.contour_win.setStyleSheet("background-color: #E8EEF2;")
        self.contour_plot = self.contour_win.addPlot(row=0, col=0, title='Contour Plot')
        vb = self.contour_plot.getViewBox()
        vb.invertY(False)
        vb.setAspectLocked(True)
        vb.enableAutoRange(True)
        self.contour_plot.showGrid(x=True, y=True, alpha=0.4)
        self.contour_plot.setLabel('bottom', 'X (mm)')
        self.contour_plot.setLabel('left', 'Y (mm)')
        self.contour_image_item = pg.ImageItem()
        self.contour_plot.addItem(self.contour_image_item)
        self.contour_line_items = []

        self.contour_hist = pg.HistogramLUTItem()
        self.contour_hist.setImageItem(self.contour_image_item)
        self.contour_hist.gradient.setColorMap(pg.ColorMap(pos, colors))
        self.contour_win.addItem(self.contour_hist, row=0, col=1)

        right_layout.addWidget(self.win, 1)
        right_layout.addWidget(self.contour_win, 1)
        self.right_panel = right_panel
        splitter.addWidget(right_panel)

        self.setLayout(layout)
        apply_readable_plot_theme(self.win, [self.plot])
        apply_readable_plot_theme(self.contour_win, [self.contour_plot])

    def _create_hline(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Sunken)
        frame.setStyleSheet("color: #c0c0c0;")
        frame.setFixedHeight(1)
        return frame

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
            if file_path.lower().endswith('.npy'):
                matrix = np.load(file_path)
                self.x_coords = np.arange(matrix.shape[1])
                self.y_coords = np.arange(matrix.shape[0])
            else:
                df = pd.read_csv(file_path, usecols=['x_rel_mm', 'y_rel_mm', 'value'])
                if df.empty:
                    raise ValueError("CSV 檔案不含 x_rel_mm / y_rel_mm / value 資料。")

                x_vals = np.asarray(df['x_rel_mm'], dtype=float)
                y_vals = np.asarray(df['y_rel_mm'], dtype=float)
                z_vals = np.asarray(df['value'], dtype=float)

                x_unique = np.unique(x_vals)
                y_unique = np.unique(y_vals)
                if x_unique.size < 2 or y_unique.size < 2:
                    raise ValueError("Mapping 檔案需有至少 2 個不同的 x_rel_mm 與 y_rel_mm 值。")

                x_unique.sort()
                y_unique.sort()
                self.x_coords = x_unique
                self.y_coords = y_unique
                matrix = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
                x_idx = np.searchsorted(x_unique, x_vals)
                y_idx = np.searchsorted(y_unique, y_vals)
                matrix[y_idx, x_idx] = z_vals

            self.mapping_matrix = matrix.astype(float)
            self.mapping_path = file_path
            self.lbl_mapping_path.setText(os.path.basename(file_path))
            self.lbl_status.setText(f"狀態: 已匯入 {os.path.basename(file_path)}，請點「畫圖」")
            self.btn_plot_mapping.setEnabled(True)
            self.btn_export_mapping.setEnabled(False)
            self.plot_drawn = False
        except Exception as exc:
            QMessageBox.critical(self, "匯入失敗", f"無法讀取檔案：\n{str(exc)}")
            self.lbl_status.setText("狀態: 匯入失敗，請選擇有效的 Mapping 檔案")
            self.mapping_matrix = None
            self.x_coords = None
            self.y_coords = None
            self.btn_plot_mapping.setEnabled(False)
            self.btn_export_mapping.setEnabled(False)

    def plot_mapping(self):
        if self.mapping_matrix is None:
            QMessageBox.warning(self, "提醒", "請先匯入 Mapping 檔案。")
            return

        raw_matrix = self.mapping_matrix
        processed_matrix = self._get_processed_matrix()
        
        # 1. 先設定 Heatmap 影像
        self.image_item.setImage(raw_matrix.T, autoLevels=False)
        if hasattr(self, 'x_coords') and hasattr(self, 'y_coords') and self.x_coords is not None and self.y_coords is not None:
            x0, x1 = float(self.x_coords[0]), float(self.x_coords[-1])
            y0, y1 = float(self.y_coords[0]), float(self.y_coords[-1])
            dx = (x1 - x0) / max(1, len(self.x_coords) - 1)
            dy = (y1 - y0) / max(1, len(self.y_coords) - 1)
            self.image_item.setRect(QRectF(x0 - dx / 2, y0 - dy / 2, dx * raw_matrix.shape[1], dy * raw_matrix.shape[0]))
            self.plot.setLabel('bottom', 'X (mm)')
            self.plot.setLabel('left', 'Y (mm)')
            self.plot.setTitle('Mapping Heatmap (mm)')
        else:
            self.plot.setTitle('Mapping Heatmap')

        # 2. 設定 Heatmap 色階與直方圖範圍
        raw_min = float(np.nanmin(raw_matrix))
        raw_max = float(np.nanmax(raw_matrix))
        if raw_min == raw_max:
            raw_min -= 1
            raw_max += 1
        self.hist.setLevels(raw_min, raw_max)
        self.hist.setHistogramRange(raw_min, raw_max)

        # 3. 更新等高線圖（內部會對 contour_image_item 執行 setImage）
        self._update_contour_plot(processed_matrix)

        # 4. 設定等高線圖（右圖）的色階與直方圖範圍
        contour_min = float(np.nanmin(processed_matrix))
        contour_max = float(np.nanmax(processed_matrix))
        if contour_min == contour_max:
            contour_min -= 1
            contour_max += 1
        self.contour_hist.setLevels(contour_min, contour_max)
        self.contour_hist.setHistogramRange(contour_min, contour_max)

        self.lbl_status.setText("狀態: Mapping 圖已繪製")
        self.plot_drawn = True
        self.btn_export_mapping.setEnabled(bool(self.export_dir))

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
        self._update_contour_plot(self._get_processed_matrix())
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
                # 檢查區塊是否全部都是 NaN，避免發出 RuntimeWarning
                if np.isnan(block).all():
                    smoothed[yi, xi] = np.nan
                else:
                    smoothed[yi, xi] = np.nanmean(block)
        return smoothed

    def _clear_contour_lines(self):
        for line in getattr(self, 'contour_line_items', []):
            self.contour_plot.removeItem(line)
        self.contour_line_items = []

    def _update_contour_plot(self, matrix):
        if matrix is None:
            return

        if hasattr(self, 'x_coords') and hasattr(self, 'y_coords') and self.x_coords is not None and self.y_coords is not None:
            x0, x1 = float(self.x_coords[0]), float(self.x_coords[-1])
            y0, y1 = float(self.y_coords[0]), float(self.y_coords[-1])
            dx = (x1 - x0) / max(1, len(self.x_coords) - 1)
            dy = (y1 - y0) / max(1, len(self.y_coords) - 1)
            
            # 確保圖片對齊真實 mm 座標（與左圖完全相同的做法）
            self.contour_image_item.setImage(matrix.T, autoLevels=False)
            self.contour_image_item.setRect(QRectF(x0 - dx / 2, y0 - dy / 2, dx * matrix.shape[1], dy * matrix.shape[0]))
            
            # 同步更新 Contour Plot 的軸標籤與顯示範圍
            self.contour_plot.setLabel('bottom', 'X (mm)')
            self.contour_plot.setLabel('left', 'Y (mm)')
        else:
            self.contour_image_item.setImage(matrix.T, autoLevels=False)
            
        avg_text = self.edit_avg_size.text().strip() or '1'
        avg_size = int(avg_text)
        self.contour_plot.setTitle(f"Contour Plot ({avg_size} x {avg_size} average)")

        self._clear_contour_lines()
        levels = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 8)
        for level in levels:
            segments = self._marching_squares(matrix, level)
            for seg in segments:
                if seg.shape[0] < 2:
                    continue
                line = pg.PlotDataItem(seg[:, 0], seg[:, 1], pen=pg.mkPen(color=(50, 50, 50), width=1))
                self.contour_plot.addItem(line)
                self.contour_line_items.append(line)

    def _marching_squares(self, matrix, level):
        ny, nx = matrix.shape
        segments = []
        if np.isnan(matrix).any():
            valid = ~np.isnan(matrix)
        else:
            valid = np.ones_like(matrix, dtype=bool)

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                if not valid[iy:iy + 2, ix:ix + 2].all():
                    continue
                v0 = matrix[iy, ix]
                v1 = matrix[iy, ix + 1]
                v2 = matrix[iy + 1, ix + 1]
                v3 = matrix[iy + 1, ix]
                if np.any(np.isnan([v0, v1, v2, v3])):
                    continue

                x0, x1 = self.x_coords[ix], self.x_coords[ix + 1]
                y0, y1 = self.y_coords[iy], self.y_coords[iy + 1]
                points = []

                def interp(p1, p2, v1, v2):
                    if v2 == v1:
                        return p1
                    t = (level - v1) / (v2 - v1)
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

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mouse_point = self.plot.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()

            if self.mapping_matrix is not None and hasattr(self, 'x_coords') and hasattr(self, 'y_coords') and self.x_coords is not None and self.y_coords is not None:
                ix = int(np.clip(np.abs(self.x_coords - x).argmin(), 0, self.mapping_matrix.shape[1] - 1))
                iy = int(np.clip(np.abs(self.y_coords - y).argmin(), 0, self.mapping_matrix.shape[0] - 1))
                value = self.mapping_matrix[iy, ix]
                value_text = 'NaN' if np.isnan(value) else f'{value:.6f}'
                self.lbl_mouse_info.setText(f"滑鼠位置: X={x:.3f} mm, Y={y:.3f} mm, Value={value_text}")
            else:
                self.lbl_mouse_info.setText(f"滑鼠位置: X={x:.3f} mm, Y={y:.3f} mm, Value=--")
        else:
            self.lbl_mouse_info.setText("滑鼠位置: X=--, Y=--, Value=--")

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

            heatmap_exporter = pg_export.ImageExporter(self.plot)
            heatmap_exporter.parameters()["width"] = int(self.plot.width())
            heatmap_exporter.parameters()["height"] = int(self.plot.height())
            heatmap_exporter.export(heatmap_path)

            contour_exporter = pg_export.ImageExporter(self.contour_plot)
            contour_exporter.parameters()["width"] = int(self.contour_plot.width())
            contour_exporter.parameters()["height"] = int(self.contour_plot.height())
            contour_exporter.export(contour_path)

            csv_path = os.path.join(self.export_dir, f"{base_name}.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('x,y,value\n')
                if hasattr(self, 'x_coords') and hasattr(self, 'y_coords') and self.x_coords is not None and self.y_coords is not None:
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