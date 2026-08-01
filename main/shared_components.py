import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

# =========================================================================
# 基礎 UI 元件
# =========================================================================
class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

# =========================================================================
# 共用彈出視窗：M1 / M2 熱圖檢視
# =========================================================================
class HeatmapViewerWindow(QMainWindow):
    def __init__(self, title, matrix_data, app_parent=None, is_m1=False):
        super().__init__(app_parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 800, 700)
        self.app_parent = app_parent
        self.is_m1 = is_m1
        self.matrix_data = matrix_data

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.win = pg.GraphicsLayoutWidget()
        self.win.setStyleSheet("border: 1px solid #d0d0d0; background-color: black;")
        layout.addWidget(self.win)

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        pos = np.linspace(0.0, 1.0, len(colors))
        jet_map = pg.ColorMap(pos, colors)

        self.plot = self.win.addPlot(row=0, col=0, title=title)
        self.plot.getViewBox().invertY(False)
        self.plot.setAspectLocked(True)
        self.plot.setLabel('bottom', 'X Pixels')
        self.plot.setLabel('left', 'Y Pixels')

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        self.thresh_overlay_item = pg.ImageItem()
        self.thresh_overlay_item.setZValue(5)
        self.thresh_overlay_item.setOpacity(1.0)
        self.plot.addItem(self.thresh_overlay_item)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(jet_map)
        self.win.addItem(self.hist, row=0, col=1)

        self.marker_items = []

        if matrix_data is not None:
            self.image_item.setImage(matrix_data.T)
            min_v, max_v = float(np.min(matrix_data)), float(np.max(matrix_data))
            self.hist.setHistogramRange(min_v, max_v)
            self.hist.setLevels(min_v, max_v)

        if self.is_m1:
            self.plot.scene().sigMouseClicked.connect(self.on_m1_mouse_clicked)

        # 這裡會安全地向 parent 請求座標，即使屬性不存在也不會崩潰
        p1 = getattr(self.app_parent, "m1_center_point", None) if self.app_parent else None
        
        click_pts = getattr(self.app_parent, "click_points", [])
        p2 = click_pts[1] if (self.app_parent and len(click_pts) >= 2) else None
        
        if not self.is_m1:
            self.draw_marker(p1, pt2=p2)
        elif p1 is not None:
            self.draw_marker(p1)

    def on_m1_mouse_clicked(self, evt):
        if self.matrix_data is None or self.app_parent is None:
            return
        
        # 兼容 DataRay 單檔與 Batch 兩種命名
        radio_manual = getattr(self.app_parent, "radio_m1_manual", None)
        if not radio_manual:
            radio_manual = getattr(self.app_parent, "radio_batch_m1_manual", None)
            
        if not radio_manual or not radio_manual.isChecked():
            return
        
        pos = evt.scenePos()
        if self.plot.sceneBoundingRect().contains(pos):
            mouse_point = self.plot.getViewBox().mapSceneToView(pos)
            cx = int(round(mouse_point.x()))
            cy = int(round(mouse_point.y()))
            h, w = self.matrix_data.shape
            
            if 0 <= cx < w and 0 <= cy < h:
                is_batch = hasattr(self.app_parent, "batch_m1_center_point")
                
                if evt.double():
                    if is_batch:
                        self.app_parent.batch_m1_center_point = None
                        self.app_parent.update_batch_calculations()
                    else:
                        self.app_parent.m1_center_point = None
                        self.app_parent.update_all_m1_markers()
                        self.app_parent.sync_dual_points_after_m1_change()
                else:
                    if is_batch:
                        self.app_parent.batch_m1_center_point = (cx, cy)
                        self.app_parent.update_batch_calculations()
                    else:
                        self.app_parent.m1_center_point = (cx, cy)
                        self.app_parent.update_all_m1_markers()
                        self.app_parent.sync_dual_points_after_m1_change()

    def draw_marker(self, pt, pt2=None):
        self.clear_marker()
        if self.matrix_data is None:
            return
        h, w = self.matrix_data.shape
        if pt is not None:
            cx, cy = pt
            pen = pg.mkPen('#76FF03', width=1.5, style=Qt.DashLine)
            v_item = pg.PlotCurveItem(x=[cx, cx], y=[0, h], pen=pen)
            h_item = pg.PlotCurveItem(x=[0, w], y=[cy, cy], pen=pen)
            self.plot.addItem(v_item)
            self.plot.addItem(h_item)
            self.marker_items.extend([v_item, h_item])
        if pt2 is not None:
            cx2, cy2 = pt2
            pen2 = pg.mkPen('#00E5FF', width=2)
            v2 = pg.PlotCurveItem(x=[cx2, cx2], y=[0, h], pen=pen2)
            h2 = pg.PlotCurveItem(x=[0, w], y=[cy2, cy2], pen=pen2)
            self.plot.addItem(v2)
            self.plot.addItem(h2)
            self.marker_items.extend([v2, h2])

    def clear_marker(self):
        for item in self.marker_items:
            self.plot.removeItem(item)
        self.marker_items.clear()

    def set_threshold_overlay(self, mask, visible=True, rgba_color=(255, 64, 255, 90)):
        if self.thresh_overlay_item is None:
            return
        if (not visible) or mask is None or self.matrix_data is None:
            self.thresh_overlay_item.clear()
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.matrix_data.shape:
            self.thresh_overlay_item.clear()
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        if np.any(mask):
            rgba[mask, 0] = rgba_color[0]
            rgba[mask, 1] = rgba_color[1]
            rgba[mask, 2] = rgba_color[2]
            rgba[mask, 3] = rgba_color[3]
        self.thresh_overlay_item.setImage(np.transpose(rgba, (1, 0, 2)), levels=(0, 255))

    def clear_threshold_overlay(self):
        if self.thresh_overlay_item is not None:
            self.thresh_overlay_item.clear()

# =========================================================================
# 共用彈出視窗：十字點擊波形
# =========================================================================
class CrossProfileViewerWindow(QMainWindow):
    def __init__(self, title_suffix="DataRay", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title_suffix} 十字即時波形檢視器")
        self.setGeometry(250, 250, 900, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.win = pg.GraphicsLayoutWidget()
        self.win.setStyleSheet("border: 1px solid #d0d0d0; background-color: black;")
        layout.addWidget(self.win)

        self.plot_x = self.win.addPlot(row=0, col=0, title="X-Axis Cross Profile (Row Intensity)")
        self.plot_x.setLabel('bottom', 'X Position (px)')
        self.plot_x.setLabel('left', 'Intensity')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)

        self.plot_y = self.win.addPlot(row=1, col=0, title="Y-Axis Cross Profile (Column Intensity)")
        self.plot_y.setLabel('bottom', 'Y Position (px)')
        self.plot_y.setLabel('left', 'Intensity')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)

    def update_profiles(self, matrix, cx, cy, y_range=None):
        if matrix is None:
            return
        h, w = matrix.shape
        cy_clamped = max(0, min(h - 1, cy))
        cx_clamped = max(0, min(w - 1, cx))

        x_profile = matrix[cy_clamped, :]
        y_profile = matrix[:, cx_clamped]
        x_axis = np.arange(w)
        y_axis = np.arange(h)

        self.plot_x.clear()
        self.plot_y.clear()

        curve_x = pg.PlotCurveItem(x_axis, x_profile, pen=pg.mkPen('#00E5FF', width=1.5))
        curve_y = pg.PlotCurveItem(y_axis, y_profile, pen=pg.mkPen('#FF5722', width=1.5))

        self.plot_x.addItem(curve_x)
        self.plot_y.addItem(curve_y)

        self.plot_x.setTitle(f"X-Axis Cross Profile (at Y = {cy})")
        self.plot_y.setTitle(f"Y-Axis Cross Profile (at X = {cx})")
        self.plot_x.setXRange(0, w, padding=0)
        self.plot_y.setXRange(0, h, padding=0)

        if y_range is not None:
            self.plot_x.setYRange(y_range[0], y_range[1], padding=0)
            self.plot_y.setYRange(y_range[0], y_range[1], padding=0)

# =========================================================================
# 共用彈出視窗：Batch 專用的 Contour 視窗
# =========================================================================
class ContourBatchViewerWindow(QMainWindow):
    def __init__(self, matrix_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Contour 檢視器")
        self.setGeometry(250, 100, 1000, 650)
        self.matrix_data = matrix_data
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        
        self.plot = self.win.addPlot(title="Smoothed Contour Map")
        self.plot.getViewBox().invertY(False)
        self.plot.setAspectLocked(True)
        
        self.img_item = pg.ImageItem(self.matrix_data.T)
        self.plot.addItem(self.img_item)
        
        # 產生等高線
        min_v, max_v = float(np.min(matrix_data)), float(np.max(matrix_data))
        levels = np.linspace(min_v, max_v, 10)
        for level in levels:
            iso = pg.IsocurveItem(data=self.matrix_data.T, level=level, pen=pg.mkPen('w', width=0.8))
            self.plot.addItem(iso)