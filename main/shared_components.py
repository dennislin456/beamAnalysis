import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pg_export
from scipy.ndimage import label as ndi_label
from PyQt5.QtWidgets import (
    QSpinBox, QDoubleSpinBox, QMainWindow, QWidget, QVBoxLayout, QCheckBox,
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QCursor


# =========================================================================
# 座標／繪圖可讀性主題（淺底＋深色軸，避免黑底看不清）
# =========================================================================
PLOT_WIDGET_BG = "#E8EEF2"
PLOT_VIEW_BG = "#FAFBFC"
PLOT_AXIS_COLOR = "#263238"
PLOT_WIDGET_STYLE = (
    f"border: 1px solid #90A4AE; background-color: {PLOT_WIDGET_BG};"
)
# 剖面訊號區目標邊長（縱剖面寬 ≈ 橫剖面高）
PROFILE_VIEW_PX = 130
PROFILE_VIEW_PX_COMPACT = 88  # Mapping 等含剖面的面板，略縮以減少擁擠
PROFILE_SIDE_AXIS_PX = 48   # V 左軸／H 右軸
PROFILE_EDGE_AXIS_PX = 28   # V 上軸／H 下軸
PROFILE_SIDE_AXIS_PX_COMPACT = 40
PROFILE_EDGE_AXIS_PX_COMPACT = 24


def apply_readable_plot_theme(widget, plots=None, transparent_view_plots=None):
    """套用淺色背景與深色座標軸；可指定透明 ViewBox（去除白底）。"""
    if widget is not None:
        widget.setStyleSheet(PLOT_WIDGET_STYLE)
        if hasattr(widget, "setBackground"):
            widget.setBackground(PLOT_WIDGET_BG)

    pen = pg.mkPen(PLOT_AXIS_COLOR, width=1)
    transparent = set(transparent_view_plots or [])
    for plot in (plots or []):
        if plot is None:
            continue
        for axis_name in ("left", "bottom", "top", "right"):
            try:
                ax = plot.getAxis(axis_name)
            except Exception:
                continue
            ax.setPen(pen)
            ax.setTextPen(pen)
        vb = plot.getViewBox()
        if vb is not None:
            if plot in transparent:
                vb.setBackgroundColor(None)
            else:
                vb.setBackgroundColor(PLOT_VIEW_BG)


def set_heatmap_view_transparent(plot):
    """完全移除熱圖座標區白色背景（ViewBox 透明，不留白邊）。"""
    if plot is None:
        return
    vb = plot.getViewBox()
    if vb is not None:
        vb.setBackgroundColor(None)
    # 避免 PlotItem 本身再鋪一層底色
    try:
        plot.setContentsMargins(0, 0, 0, 0)
    except Exception:
        pass


def configure_equal_profile_strips(
    layout_widget,
    profile_col=0,
    profile_row=1,
    view_px=None,
    side_axis_px=None,
    edge_axis_px=None,
):
    """讓縱剖面欄寬與橫剖面列高一致，訊號 ViewBox 視覺尺寸相同。"""
    if layout_widget is None or not hasattr(layout_widget, "ci"):
        return
    view = PROFILE_VIEW_PX if view_px is None else int(view_px)
    side = PROFILE_SIDE_AXIS_PX if side_axis_px is None else int(side_axis_px)
    edge = PROFILE_EDGE_AXIS_PX if edge_axis_px is None else int(edge_axis_px)
    col_w = view + side
    row_h = view + edge
    layout = layout_widget.ci.layout
    layout.setColumnFixedWidth(profile_col, col_w)
    layout.setRowFixedHeight(profile_row, row_h)
    layout.setColumnStretchFactor(profile_col, 0)
    layout.setRowStretchFactor(profile_row, 0)


def enforce_square_heatmap_cell(
    layout_widget,
    heat_plot,
    heat_col=1,
    heat_row=0,
    profile_col=0,
    profile_row=1,
    hist_col=2,
    spacer_col=3,
    hist_width=110,
    margin_px=8,
):
    """
    將熱圖格子強制為正方形（寬＝高），多餘寬度留給右側 spacer。
    若有標題列，列高會加上標題高度，讓實際 ViewBox 接近 1:1。
    """
    if layout_widget is None or heat_plot is None or not hasattr(layout_widget, "ci"):
        return 0
    layout = layout_widget.ci.layout
    profile_col_w = PROFILE_VIEW_PX + PROFILE_SIDE_AXIS_PX
    profile_row_h = PROFILE_VIEW_PX + PROFILE_EDGE_AXIS_PX

    title_h = 0
    try:
        if hasattr(heat_plot, "titleLabel") and heat_plot.titleLabel is not None:
            if heat_plot.titleLabel.isVisible():
                title_h = int(max(0.0, heat_plot.titleLabel.boundingRect().height()))
    except Exception:
        title_h = 0

    avail_w = int(layout_widget.width()) - profile_col_w - int(hist_width) - int(margin_px) * 2
    avail_h = int(layout_widget.height()) - profile_row_h - int(margin_px) * 2 - title_h
    side = max(80, min(avail_w, avail_h))

    # 熱圖欄／列固定為正方形（列含標題）
    layout.setColumnFixedWidth(heat_col, side)
    layout.setRowFixedHeight(heat_row, side + title_h)
    layout.setColumnStretchFactor(heat_col, 0)
    layout.setRowStretchFactor(heat_row, 0)

    # 色條欄固定寬，避免把熱圖拉開
    try:
        layout.setColumnFixedWidth(hist_col, int(hist_width))
        layout.setColumnStretchFactor(hist_col, 0)
    except Exception:
        pass

    # 右側剩餘空間給 spacer 欄（可無 item，僅佔伸縮）
    try:
        layout.setColumnStretchFactor(spacer_col, 1)
        layout.setColumnMinimumWidth(spacer_col, 0)
    except Exception:
        pass

    # 剖面條維持等寬等高
    configure_equal_profile_strips(
        layout_widget, profile_col=profile_col, profile_row=profile_row
    )
    return side


def configure_stable_plot_item(plot, mouse_enabled=False):
    """穩定座標圖：隱藏 Auto(A) 鈕與右鍵選單；可開啟滾輪／拖曳縮放。"""
    if plot is None:
        return
    try:
        plot.hideButtons()
    except Exception:
        pass
    try:
        plot.setMenuEnabled(False)
    except Exception:
        pass
    plot.setDefaultPadding(0)
    vb = plot.getViewBox()
    if vb is not None:
        vb.setMouseEnabled(x=bool(mouse_enabled), y=bool(mouse_enabled))
        vb.setMenuEnabled(False)
        # 關閉自動範圍，避免縮放後外框被拉動；仍可用滾輪手動縮放
        try:
            vb.enableAutoRange(x=False, y=False)
        except Exception:
            pass
        if hasattr(vb, "autoBtn"):
            try:
                vb.autoBtn.hide()
            except Exception:
                pass


def lock_plot_ranges(plot, x_range=None, y_range=None):
    """設定範圍並關閉 auto-range，避免外框被自動縮放拉動。"""
    if plot is None:
        return
    vb = plot.getViewBox()
    if x_range is not None:
        plot.setXRange(x_range[0], x_range[1], padding=0)
    if y_range is not None:
        plot.setYRange(y_range[0], y_range[1], padding=0)
    if vb is not None:
        try:
            vb.enableAutoRange(x=False, y=False)
        except Exception:
            pass
    try:
        plot.hideButtons()
    except Exception:
        pass


# =========================================================================
# 光斑定位演算法（M1／M2 共用）
# =========================================================================
def estimate_border_background(matrix):
    """以邊界像素中位數估計背景。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    h, w = matrix.shape
    if h < 2 or w < 2:
        return float(np.median(matrix))
    border = np.concatenate([matrix[0, :], matrix[-1, :], matrix[1:-1, 0], matrix[1:-1, -1]])
    return float(np.median(border))


def build_robust_threshold_mask(matrix, use_threshold=True, thresh_percent=50.0,
                                bg_subtract=True, largest_cc_only=True):
    """背景扣除後依門檻建 mask，可選擇只保留最大連通區。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return None

    if bg_subtract:
        bg = estimate_border_background(matrix)
        work = np.clip(matrix - bg, 0.0, None)
    else:
        work = matrix

    peak_val = float(np.max(work)) if work.size > 0 else 0.0
    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.zeros(matrix.shape, dtype=bool)

    frac = (thresh_percent / 100.0) if use_threshold else 0.5
    mask = work >= (peak_val * frac)
    if not np.any(mask):
        return mask

    if largest_cc_only:
        labeled, num = ndi_label(mask)
        if num > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            mask = labeled == int(np.argmax(counts))
        elif num == 0:
            return np.zeros(matrix.shape, dtype=bool)
    return mask


def compute_auto_spot_center(matrix, mode, use_threshold=False, thresh_percent=50.0,
                             bg_subtract=True, largest_cc_only=True, subpixel=True,
                             power=1.0):
    """計算光斑中心。centroid／thresh_geom 預設：背景扣除 + 最大連通區 + 亞像素。

    power: 僅用於質心加權（mode 非 thresh_geom／peak_geom）。1=一般質心，2=I² 加權。

    Returns:
        (cx, cy) — subpixel=True 時為 float，否則為 int
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return 0, 0
    h, w = matrix.shape
    peak_val = float(np.max(matrix))

    if mode == "peak_geom":
        ys, xs = np.where(matrix == peak_val)
        if len(xs) > 0:
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
        else:
            yi, xi = np.unravel_index(np.argmax(matrix), matrix.shape)
            cx, cy = float(xi), float(yi)
        if subpixel:
            return cx, cy
        return int(round(cx)), int(round(cy))

    if bg_subtract:
        bg = estimate_border_background(matrix)
        work = np.clip(matrix - bg, 0.0, None)
    else:
        work = matrix

    peak_work = float(np.max(work)) if work.size > 0 else 0.0
    if not np.isfinite(peak_work) or peak_work <= 0:
        yi, xi = np.unravel_index(np.argmax(matrix), matrix.shape)
        return (float(xi), float(yi)) if subpixel else (int(xi), int(yi))

    frac = (thresh_percent / 100.0) if use_threshold else 0.5
    mask = work >= (peak_work * frac)
    if not np.any(mask):
        yi, xi = np.unravel_index(np.argmax(work), work.shape)
        return (float(xi), float(yi)) if subpixel else (int(xi), int(yi))

    if largest_cc_only:
        labeled, num = ndi_label(mask)
        if num > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            mask = labeled == int(np.argmax(counts))
        elif num == 0:
            yi, xi = np.unravel_index(np.argmax(work), work.shape)
            return (float(xi), float(yi)) if subpixel else (int(xi), int(yi))

    ys, xs = np.where(mask)
    if mode == "thresh_geom":
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    else:
        p = float(power) if np.isfinite(power) and power > 0 else 1.0
        if abs(p - 1.0) < 1e-12:
            weights = work[mask]
        else:
            weights = np.power(np.clip(work[mask], 0.0, None), p)
        wsum = float(np.sum(weights))
        if wsum <= 0:
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
        else:
            cx = float(np.sum(xs * weights) / wsum)
            cy = float(np.sum(ys * weights) / wsum)

    cx = min(max(cx, 0.0), w - 1.0)
    cy = min(max(cy, 0.0), h - 1.0)
    if subpixel:
        return cx, cy
    return int(round(cx)), int(round(cy))


def split_y_index(y1):
    """將（可能為亞像素的）Y 轉成整數切割列。"""
    return int(round(float(y1)))


def clip_roi_to_matrix(matrix, roi):
    """將 (x, y, width, height) 裁切到矩陣範圍，回傳 (x0, y0, x1, y1) 半開區間。

    x1/y1 保證至少比 x0/y0 大 1（若矩陣非空）。無效 roi 回傳 None。
    """
    matrix = np.asarray(matrix)
    if matrix.size == 0 or roi is None:
        return None
    h, w = matrix.shape
    try:
        x, y, rw, rh = roi
        x0 = int(round(float(x)))
        y0 = int(round(float(y)))
        rw = int(round(float(rw)))
        rh = int(round(float(rh)))
    except (TypeError, ValueError):
        return None
    if rw <= 0 or rh <= 0:
        return None
    x0 = int(np.clip(x0, 0, max(0, w - 1)))
    y0 = int(np.clip(y0, 0, max(0, h - 1)))
    x1 = int(np.clip(x0 + rw, x0 + 1, w))
    y1 = int(np.clip(y0 + rh, y0 + 1, h))
    return (x0, y0, x1, y1)


def find_dual_peak_valley_y(matrix, cx=None, col_half_width=2, smooth_win=7,
                            min_peak_distance=5, roi=None):
    """沿質心 X 縱切取 1D profile，找雙峰之間波谷的 Y（用來區分 above／below）。

    流程：
      1. 若指定 roi=(x, y, width, height)，僅在該矩形內搜尋（可排除散射雜點）
      2. 若未指定 cx，以（ROI 或全圖）質心（門檻 50%）估 X
      3. 取 cx ± col_half_width 欄平均成垂直 profile
      4. 背景扣除後平滑，找兩個主峰，取峰間最小值為波谷 Y

    Returns:
        dict: {
            "valley_y": float,   # 全圖座標
            "cx": float,         # 全圖座標
            "peak_ys": (y_lo, y_hi) 或 None,  # 全圖座標
            "profile": 1d ndarray,  # ROI／全圖高度的原始縱切
            "roi": (x0, y0, x1, y1) 或 None,  # 實際使用的半開裁切
        }
        失敗時 valley_y 回退為搜尋區中線。
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return {
            "valley_y": 0.0, "cx": 0.0, "peak_ys": None,
            "profile": np.array([]), "roi": None,
        }

    full_h, full_w = matrix.shape
    roi_bounds = clip_roi_to_matrix(matrix, roi)
    if roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        work_mat = matrix[y0:y1, x0:x1]
        y_offset = float(y0)
        x_offset = float(x0)
        if cx is not None:
            cx = float(cx) - x_offset
    else:
        work_mat = matrix
        y_offset = 0.0
        x_offset = 0.0
        roi_bounds = None

    h, w = work_mat.shape
    if h < 1 or w < 1:
        mid_y = (full_h - 1) / 2.0
        mid_x = (full_w - 1) / 2.0
        return {
            "valley_y": mid_y, "cx": mid_x, "peak_ys": None,
            "profile": np.array([]), "roi": roi_bounds,
        }

    if cx is None:
        try:
            cx, _cy = compute_auto_spot_center(
                work_mat, "centroid", use_threshold=True, thresh_percent=50.0,
                bg_subtract=True, largest_cc_only=True, subpixel=True,
            )
        except Exception:
            cx = (w - 1) / 2.0
    cx = float(np.clip(cx, 0.0, w - 1.0))
    ci = int(round(cx))
    c0 = max(0, ci - int(col_half_width))
    c1 = min(w, ci + int(col_half_width) + 1)
    profile = np.mean(work_mat[:, c0:c1], axis=1)

    bg = float(np.median(profile)) if profile.size else 0.0
    work = np.clip(profile - bg, 0.0, None)
    win = max(1, int(smooth_win))
    if win % 2 == 0:
        win += 1
    if work.size >= win:
        smooth = uniform_filter1d(work, size=win, mode="nearest")
    else:
        smooth = work

    fallback_y = (h - 1) / 2.0
    peak_ys = None
    valley_y = fallback_y

    if smooth.size >= 3 and float(np.max(smooth)) > 0:
        prominence = max(float(np.max(smooth)) * 0.05, 1e-9)
        peaks, props = find_peaks(
            smooth,
            distance=max(1, int(min_peak_distance)),
            prominence=prominence,
        )
        if len(peaks) < 2:
            # 放寬條件再試一次
            peaks, props = find_peaks(
                smooth,
                distance=max(1, int(min_peak_distance)),
            )
        if len(peaks) >= 2:
            # 取 prominence 最高的兩個峰（若無 prominence 則取高度最高）
            if "prominences" in props and props["prominences"] is not None:
                order = np.argsort(props["prominences"])[::-1]
            else:
                order = np.argsort(smooth[peaks])[::-1]
            top2 = sorted(int(peaks[i]) for i in order[:2])
            y_lo, y_hi = top2[0], top2[1]
            if y_hi > y_lo:
                seg = smooth[y_lo:y_hi + 1]
                min_val = float(np.min(seg))
                # 平坦波谷取最低平台中點，避免偏到單側
                tol = max(abs(min_val) * 0.02, float(np.max(seg)) * 0.005, 1e-12)
                cands = np.where(seg <= min_val + tol)[0]
                valley_local = int(np.round(np.median(cands))) if cands.size else int(np.argmin(seg))
                valley_y = float(y_lo + valley_local)
                peak_ys = (float(y_lo), float(y_hi))

    valley_y = float(np.clip(valley_y, 0.0, h - 1.0))
    # 轉回全圖座標
    valley_y = float(np.clip(valley_y + y_offset, 0.0, full_h - 1.0))
    cx_full = float(np.clip(cx + x_offset, 0.0, full_w - 1.0))
    if peak_ys is not None:
        peak_ys = (float(peak_ys[0] + y_offset), float(peak_ys[1] + y_offset))

    return {
        "valley_y": valley_y,
        "cx": cx_full,
        "peak_ys": peak_ys,
        "profile": profile,
        "roi": roi_bounds,
    }


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
# 共用彈出視窗：M1 / M2 熱圖檢視（純熱圖，不含剖面）
# =========================================================================
class HeatmapViewerWindow(QMainWindow):
    def __init__(self, title, matrix_data, app_parent=None, is_m1=False):
        super().__init__(app_parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 800, 700)
        self.app_parent = app_parent
        self.is_m1 = is_m1
        self.matrix_data = matrix_data
        self._base_title = title

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)

        toolbar = QHBoxLayout()
        self.chk_grayscale = QCheckBox("熱力圖改為黑白（方便對照十字／圓）")
        self.chk_grayscale.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_grayscale.toggled.connect(self._on_grayscale_toggled)
        toolbar.addWidget(self.chk_grayscale)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        pos = np.linspace(0.0, 1.0, len(colors))
        self.jet_map = pg.ColorMap(pos, colors)
        self.gray_map = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])

        self.plot = self.win.addPlot(row=0, col=0, title=title)
        self.plot.getViewBox().invertY(False)
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "X Pixels")
        self.plot.setLabel("left", "Y Pixels")
        configure_stable_plot_item(self.plot, mouse_enabled=True)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        self.thresh_overlay_item = pg.ImageItem()
        self.thresh_overlay_item.setZValue(5)
        self.thresh_overlay_item.setOpacity(1.0)
        self.plot.addItem(self.thresh_overlay_item)

        self.hist = LevelAlignedHistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(self.jet_map)
        self.win.addItem(self.hist, row=0, col=1)

        self.marker_items = []

        apply_readable_plot_theme(
            self.win, [self.plot], transparent_view_plots=[self.plot]
        )
        set_heatmap_view_transparent(self.plot)

        if matrix_data is not None:
            self.image_item.setImage(matrix_data.T)
            min_v, max_v = float(np.min(matrix_data)), float(np.max(matrix_data))
            self.hist.setHistogramRange(min_v, max_v, padding=0)
            self.hist.setLevels(min_v, max_v)
            self.reset_view_to_data()

        # 雙擊復原視野；M1 手動模式另處理單擊
        self.plot.scene().sigMouseClicked.connect(self.on_viewer_mouse_clicked)

        # 若 parent 已勾選黑白，開啟時同步
        parent_gray = False
        if self.app_parent is not None:
            chk = getattr(self.app_parent, "chk_batch_heatmap_gray", None)
            if chk is not None:
                parent_gray = chk.isChecked()
        if parent_gray:
            self.chk_grayscale.blockSignals(True)
            self.chk_grayscale.setChecked(True)
            self.chk_grayscale.blockSignals(False)
            self.set_grayscale(True)

        p1 = getattr(self.app_parent, "m1_center_point", None) if self.app_parent else None
        if p1 is None and self.app_parent is not None:
            p1 = getattr(self.app_parent, "batch_m1_center_point", None)

        click_pts = getattr(self.app_parent, "click_points", []) if self.app_parent else []
        p2 = click_pts[1] if (self.app_parent and len(click_pts) >= 2) else None
        if p2 is None and self.app_parent is not None:
            p2 = getattr(self.app_parent, "batch_m2_center_point", None)

        pt3 = None
        r2 = r3 = None
        if self.app_parent is not None and not self.is_m1:
            pt3 = getattr(self.app_parent, "batch_m2_above_point", None)
            r2 = getattr(self.app_parent, "batch_m2_below_circle_r", None)
            r3 = getattr(self.app_parent, "batch_m2_above_circle_r", None)

        if not self.is_m1:
            self.draw_marker(p1, pt2=p2, pt3=pt3, r2=r2, r3=r3)
        elif p1 is not None:
            self.draw_marker(p1)

        # 標記線加完後再鎖一次範圍，避免被線段拉大
        self.reset_view_to_data()

    def showEvent(self, event):
        """視窗真正顯示後再套一次資料範圍（此時才有正確長寬比）。"""
        super().showEvent(event)
        self.reset_view_to_data()

    def reset_view_to_data(self):
        """將視野復原到影像完整 XY 範圍（對應檔案矩陣尺寸）。"""
        if self.matrix_data is None:
            return
        h, w = self.matrix_data.shape
        x1, y1 = float(max(w, 1)), float(max(h, 1))
        vb = self.plot.getViewBox()
        if vb is not None:
            try:
                # 以 ImageItem 實際邊界為準（避免十字線把範圍拉大）
                vb.enableAutoRange(x=False, y=False)
                if self.image_item is not None:
                    vb.setAspectLocked(True)
                    vb.autoRange(items=[self.image_item], padding=0.0)
                else:
                    vb.setAspectLocked(False)
                    vb.setRange(xRange=(0.0, x1), yRange=(0.0, y1), padding=0.0)
                    vb.setAspectLocked(True)
            except Exception:
                lock_plot_ranges(self.plot, x_range=(0.0, x1), y_range=(0.0, y1))
            try:
                vb.setLimits(xMin=-x1 * 0.05, xMax=x1 * 1.05, yMin=-y1 * 0.05, yMax=y1 * 1.05)
            except Exception:
                pass
            try:
                vb.enableAutoRange(x=False, y=False)
            except Exception:
                pass
        else:
            lock_plot_ranges(self.plot, x_range=(0.0, x1), y_range=(0.0, y1))
        try:
            self.plot.hideButtons()
        except Exception:
            pass
        set_heatmap_view_transparent(self.plot)

    def set_grayscale(self, enabled):
        """外部同步黑白／彩色模式。"""
        if self.chk_grayscale.isChecked() != bool(enabled):
            self.chk_grayscale.blockSignals(True)
            self.chk_grayscale.setChecked(bool(enabled))
            self.chk_grayscale.blockSignals(False)
        self._apply_colormap(bool(enabled))

    def _on_grayscale_toggled(self, checked):
        self._apply_colormap(checked)

    def _apply_colormap(self, grayscale):
        cmap = self.gray_map if grayscale else self.jet_map
        levels = self.hist.getLevels()
        self.hist.gradient.setColorMap(cmap)
        self.hist.setLevels(*levels)
        title = self._base_title + (" [Grayscale]" if grayscale else "")
        self.plot.setTitle(title)

    def on_viewer_mouse_clicked(self, evt):
        """雙擊：復原 XY 視野；M1 手動模式單擊：設定中心點。"""
        if self.matrix_data is None:
            return
        pos = evt.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return

        if evt.double():
            self.reset_view_to_data()
            # M1 手動：雙擊仍可清除中心點
            if self.is_m1:
                self._apply_m1_manual_click(None, clear=True)
            return

        if self.is_m1:
            mouse_point = self.plot.getViewBox().mapSceneToView(pos)
            cx = int(round(mouse_point.x()))
            cy = int(round(mouse_point.y()))
            h, w = self.matrix_data.shape
            if 0 <= cx < w and 0 <= cy < h:
                self._apply_m1_manual_click((cx, cy), clear=False)

    def _apply_m1_manual_click(self, point, clear=False):
        if self.app_parent is None:
            return
        radio_manual = getattr(self.app_parent, "radio_m1_manual", None)
        if not radio_manual:
            radio_manual = getattr(self.app_parent, "radio_batch_m1_manual", None)
        if not radio_manual or not radio_manual.isChecked():
            return

        is_batch = hasattr(self.app_parent, "batch_m1_center_point")
        if clear:
            if is_batch:
                self.app_parent.batch_m1_center_point = None
                self.app_parent.update_batch_calculations()
            else:
                self.app_parent.m1_center_point = None
                self.app_parent.update_all_m1_markers()
                self.app_parent.sync_dual_points_after_m1_change()
            return

        cx, cy = point
        if is_batch:
            self.app_parent.batch_m1_center_point = (cx, cy)
            self.app_parent.update_batch_calculations()
        else:
            self.app_parent.m1_center_point = (cx, cy)
            self.app_parent.update_all_m1_markers()
            self.app_parent.sync_dual_points_after_m1_change()

    def draw_marker(self, pt, pt2=None, pt3=None, r2=None, r3=None):
        self.clear_marker()
        if self.matrix_data is None:
            return
        h, w = self.matrix_data.shape
        if pt is not None:
            cx, cy = pt
            pen = pg.mkPen('#00C853', width=2.5, style=Qt.DashLine)
            v_item = pg.PlotCurveItem(x=[cx, cx], y=[0, h], pen=pen)
            h_item = pg.PlotCurveItem(x=[0, w], y=[cy, cy], pen=pen)
            self.plot.addItem(v_item)
            self.plot.addItem(h_item)
            self.marker_items.extend([v_item, h_item])
        if pt2 is not None:
            cx2, cy2 = pt2
            pen2 = pg.mkPen('#2962FF', width=2.5)
            v2 = pg.PlotCurveItem(x=[cx2, cx2], y=[0, h], pen=pen2)
            h2 = pg.PlotCurveItem(x=[0, w], y=[cy2, cy2], pen=pen2)
            self.plot.addItem(v2)
            self.plot.addItem(h2)
            self.marker_items.extend([v2, h2])
            if r2 is not None and r2 > 0:
                circle2 = self._make_circle_curve(cx2, cy2, r2, pg.mkPen('#2962FF', width=2))
                self.plot.addItem(circle2)
                self.marker_items.append(circle2)
        if pt3 is not None:
            cx3, cy3 = pt3
            pen3 = pg.mkPen('#D50000', width=2.5)
            v3 = pg.PlotCurveItem(x=[cx3, cx3], y=[0, h], pen=pen3)
            h3 = pg.PlotCurveItem(x=[0, w], y=[cy3, cy3], pen=pen3)
            self.plot.addItem(v3)
            self.plot.addItem(h3)
            self.marker_items.extend([v3, h3])
            if r3 is not None and r3 > 0:
                circle3 = self._make_circle_curve(
                    cx3, cy3, r3, pg.mkPen('#D50000', width=2))
                self.plot.addItem(circle3)
                self.marker_items.append(circle3)

    @staticmethod
    def _make_circle_curve(cx, cy, radius, pen, n=72):
        theta = np.linspace(0, 2 * np.pi, n)
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta)
        return pg.PlotCurveItem(x=xs, y=ys, pen=pen)

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
        layout.addWidget(self.win)

        self.plot_x = self.win.addPlot(row=0, col=0, title="X-Axis Cross Profile (Row Intensity)")
        self.plot_x.setLabel('bottom', 'X Position (px)')
        self.plot_x.setLabel('left', 'Intensity')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)

        self.plot_y = self.win.addPlot(row=1, col=0, title="Y-Axis Cross Profile (Column Intensity)")
        self.plot_y.setLabel('bottom', 'Y Position (px)')
        self.plot_y.setLabel('left', 'Intensity')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)

        apply_readable_plot_theme(self.win, [self.plot_x, self.plot_y])

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
# 互動式熱力圖面板（滾輪縮放／拖曳／雙擊復原／色條上下限／匯出）
# =========================================================================
_JET_POS = np.linspace(0.0, 1.0, 6)
_JET_COLORS = [
    (0, 0, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 0), (255, 0, 255),
]


class LevelAlignedHistogramLUTItem(pg.HistogramLUTItem):
    """修正 pyqtgraph 預設把上下限連到色條頂/底，造成拉條看起來不在設定值上。"""

    def paint(self, p, *args):
        if self.levelMode != "mono" or not self.region.isVisible():
            return

        pen = self.region.lines[0].pen
        mn, mx = self.getLevels()
        vbc = self.vb.viewRect().center()
        grad_rect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())

        if self.orientation == "vertical":
            p_mn = self.vb.mapFromViewToItem(self, pg.Point(vbc.x(), mn))
            p_mx = self.vb.mapFromViewToItem(self, pg.Point(vbc.x(), mx))
            x_grad = (
                grad_rect.left()
                if self.gradientPosition == "right"
                else grad_rect.right()
            )
            # 水平連到色條「同一 Y」，不再 ±5px、也不連到色條頂/底
            ends = (
                (p_mn, pg.Point(x_grad, p_mn.y())),
                (p_mx, pg.Point(x_grad, p_mx.y())),
            )
            tick_specs = (
                (pg.Point(grad_rect.left(), p_mn.y()), pg.Point(grad_rect.right(), p_mn.y())),
                (pg.Point(grad_rect.left(), p_mx.y()), pg.Point(grad_rect.right(), p_mx.y())),
            )
        else:
            p_mn = self.vb.mapFromViewToItem(self, pg.Point(mn, vbc.y()))
            p_mx = self.vb.mapFromViewToItem(self, pg.Point(mx, vbc.y()))
            y_grad = (
                grad_rect.top()
                if self.gradientPosition == "bottom"
                else grad_rect.bottom()
            )
            ends = (
                (p_mn, pg.Point(p_mn.x(), y_grad)),
                (p_mx, pg.Point(p_mx.x(), y_grad)),
            )
            tick_specs = (
                (pg.Point(p_mn.x(), grad_rect.top()), pg.Point(p_mn.x(), grad_rect.bottom())),
                (pg.Point(p_mx.x(), grad_rect.top()), pg.Point(p_mx.x(), grad_rect.bottom())),
            )

        from pyqtgraph.Qt import QtGui
        from pyqtgraph import functions as fn

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        for draw_pen in (fn.mkPen((0, 0, 0, 100), width=3), pen):
            p.setPen(draw_pen)
            for a, b in ends:
                p.drawLine(a, b)
            for a, b in tick_specs:
                p.drawLine(a, b)


class InteractiveHeatmapPanel(QWidget):
    """自訂座標熱力圖：滾輪縮放、拖曳平移、雙擊復原視野；
    色條可直接拖曳藍色拉條調整上下限（亦可點擊數值精調）；上方可匯出當前圖檔。
    with_profiles=True 時左側／下方顯示 X/Y 剖面（點擊更新，可隱藏十字）。"""

    levelsChanged = pyqtSignal(float, float)
    mouseMoved = pyqtSignal(object)  # QPointF in view coords, or None
    profilePointChanged = pyqtSignal(int, int)  # ix, iy
    _HIST_RANGE_PAD = 0.08  # 輸入上下限後色條可視範圍相對該區間的外擴

    def __init__(self, title="Heatmap", parent=None, x_label="X", y_label="Y",
                 aspect_locked=True, with_profiles=False):
        super().__init__(parent)
        self._title = title
        self._default_levels = None  # (min, max)
        self._data_hist_range = None  # 完整資料範圍，色條軸以此為準
        self._data_rect = None  # QRectF in data/view coordinates (e.g. mm)
        self._level_updating = False
        self._aspect_locked = bool(aspect_locked)
        self._with_profiles = bool(with_profiles)
        self._profile_matrix = None  # (ny, nx)
        self._profile_x = None
        self._profile_y = None
        self._profile_point = None  # (ix, iy)
        self._profile_cross_items = []
        self._show_profile_cross = True
        self._profile_view_ready = False
        self._x_label = x_label
        self._y_label = y_label

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self.btn_export = QPushButton("匯出當前圖檔")
        self.btn_export.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #0288D1; border: none; border-radius: 4px; "
            "padding: 5px 12px; }"
            "QPushButton:hover { background-color: #039BE5; }"
            "QPushButton:pressed { background-color: #01579B; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        self.btn_export.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_export.clicked.connect(self.export_current_image)
        toolbar.addWidget(self.btn_export)

        # 放在匯出旁，避免擠在上下限右側造成工具列跳動
        self.btn_toggle_cross = QPushButton("隱藏十字")
        self.btn_toggle_cross.setCheckable(True)
        self.btn_toggle_cross.setChecked(False)
        self.btn_toggle_cross.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_toggle_cross.setToolTip("隱藏圖上十字標示；X/Y 剖面波形仍保留")
        self.btn_toggle_cross.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: #37474F; "
            "background-color: #ECEFF1; border: 1px solid #B0BEC5; border-radius: 4px; "
            "padding: 5px 10px; }"
            "QPushButton:checked { color: white; background-color: #546E7A; }"
            "QPushButton:hover { background-color: #CFD8DC; }"
            "QPushButton:checked:hover { background-color: #607D8B; }"
        )
        self.btn_toggle_cross.toggled.connect(self._on_toggle_cross)
        self.btn_toggle_cross.setVisible(self._with_profiles)
        toolbar.addWidget(self.btn_toggle_cross)

        if self._with_profiles:
            tip_text = (
                "滾輪縮放主圖／剖面（剖面不跟主圖連動）· 點擊更新剖面 · "
                "雙擊主圖依比例復原 · 雙擊剖面復原"
            )
        else:
            tip_text = (
                "滾輪縮放 · 拖曳平移 · 雙擊圖面復原 · "
                "色條拖曳藍色拉條調上下限 · 雙擊色條復原"
            )
        tip = QLabel(tip_text)
        tip.setStyleSheet("color: #607D8B; font-size: 11px;")
        tip.setMinimumWidth(0)
        tip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(tip, 1)

        self.lbl_level_max = QLabel("上限: --")
        self.lbl_level_min = QLabel("下限: --")
        # 固定寬度，避免數值位數變化撐動整排工具列
        _level_lbl_w = 128
        for lbl in (self.lbl_level_max, self.lbl_level_min):
            lbl.setStyleSheet(
                "QLabel { color: #37474F; font-size: 12px; font-weight: bold; "
                "font-family: Consolas, 'Courier New', monospace; "
                "padding: 3px 6px; border: 1px solid #B0BEC5; border-radius: 3px; "
                "background-color: #ECEFF1; }"
                "QLabel:hover { background-color: #CFD8DC; }"
            )
            lbl.setFixedWidth(_level_lbl_w)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lbl.setCursor(QCursor(Qt.PointingHandCursor))
            lbl.setToolTip("主要請直接拖曳右側色條藍色拉條；點此可精準輸入數值")
            lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_level_max.mousePressEvent = self._on_max_label_clicked
        self.lbl_level_min.mousePressEvent = self._on_min_label_clicked
        toolbar.addWidget(self.lbl_level_max)
        toolbar.addWidget(self.lbl_level_min)
        root.addLayout(toolbar)

        self.win = pg.GraphicsLayoutWidget()
        self.win.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.win, 1)

        heat_row, heat_col = (0, 1) if self._with_profiles else (0, 0)
        self.plot = self.win.addPlot(row=heat_row, col=heat_col, title=title)
        self.plot.getViewBox().invertY(False)
        if self._aspect_locked:
            self.plot.getViewBox().setAspectLocked(lock=True, ratio=1.0)
        else:
            self.plot.getViewBox().setAspectLocked(False)
        self.plot.showGrid(x=True, y=True, alpha=0.4)
        configure_stable_plot_item(self.plot, mouse_enabled=True)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        self.hist = LevelAlignedHistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.hist.gradient.setColorMap(pg.ColorMap(_JET_POS, _JET_COLORS))
        self._enable_level_drag()

        self.plot_x_profile = None
        self.plot_y_profile = None
        self.plot_profile_corner = None

        if self._with_profiles:
            self.plot_y_profile = self.win.addPlot(row=0, col=0)
            self.plot_y_profile.setLabel("top", "Value")
            self.plot_y_profile.setLabel("left", y_label)
            self.plot_y_profile.showAxis("bottom", show=False)
            self.plot_y_profile.showAxis("top", show=True)
            self.plot_y_profile.showGrid(x=True, y=True, alpha=0.25)
            self.plot_y_profile.titleLabel.hide()
            configure_stable_plot_item(self.plot_y_profile, mouse_enabled=True)
            self.plot_y_profile.getViewBox().setMouseEnabled(x=True, y=True)

            self.win.addItem(self.hist, row=0, col=2)

            self.plot_x_profile = self.win.addPlot(row=1, col=1)
            self.plot_x_profile.setLabel("bottom", x_label)
            self.plot_x_profile.setLabel("right", "Value")
            self.plot_x_profile.showAxis("left", show=False)
            self.plot_x_profile.showAxis("right", show=True)
            self.plot_x_profile.showGrid(x=True, y=True, alpha=0.25)
            self.plot_x_profile.titleLabel.hide()
            configure_stable_plot_item(self.plot_x_profile, mouse_enabled=True)
            self.plot_x_profile.getViewBox().setMouseEnabled(x=True, y=True)

            corner = self.win.addPlot(row=1, col=0)
            corner.hideAxis("left")
            corner.hideAxis("bottom")
            corner.hideAxis("top")
            corner.hideAxis("right")
            corner.titleLabel.hide()
            configure_stable_plot_item(corner, mouse_enabled=False)
            self.plot_profile_corner = corner

            # 熱圖保留座標軸；剖面獨立（不 setXLink/setYLink）
            self.plot.showAxis("bottom", show=True)
            self.plot.showAxis("left", show=True)
            self.plot.setLabel("bottom", x_label)
            self.plot.setLabel("left", y_label)

            configure_equal_profile_strips(
                self.win,
                profile_col=0,
                profile_row=1,
                view_px=PROFILE_VIEW_PX_COMPACT,
                side_axis_px=PROFILE_SIDE_AXIS_PX_COMPACT,
                edge_axis_px=PROFILE_EDGE_AXIS_PX_COMPACT,
            )
            self.win.ci.layout.setColumnStretchFactor(1, 1)
            self.win.ci.layout.setColumnStretchFactor(2, 0)
            self.plot_y_profile.getAxis("left").setWidth(PROFILE_SIDE_AXIS_PX_COMPACT)
            self.plot_y_profile.getAxis("top").setHeight(PROFILE_EDGE_AXIS_PX_COMPACT)
            self.plot_x_profile.getAxis("right").setWidth(PROFILE_SIDE_AXIS_PX_COMPACT)
            self.plot_x_profile.getAxis("bottom").setHeight(PROFILE_EDGE_AXIS_PX_COMPACT)

            theme_plots = [self.plot, self.plot_x_profile, self.plot_y_profile]
        else:
            self.plot.setLabel("bottom", x_label)
            self.plot.setLabel("left", y_label)
            self.win.addItem(self.hist, row=0, col=1)
            theme_plots = [self.plot]

        apply_readable_plot_theme(self.win, theme_plots)

        self.hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._mouse_proxy = pg.SignalProxy(
            self.win.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

    def _level_region(self):
        """取得 HistogramLUT 的上下限 LinearRegionItem。"""
        if hasattr(self.hist, "region") and self.hist.region is not None:
            return self.hist.region
        regions = getattr(self.hist, "regions", None)
        if regions:
            return regions[0]
        return None

    def _enable_level_drag(self):
        """開啟並強化色條上下限拖曳（對齊原本 HistogramLUT 操作）。"""
        region = self._level_region()
        if region is None:
            return
        try:
            region.setMovable(True)
        except Exception:
            pass
        try:
            # 加粗上下限線，較容易抓取拉扯
            region.setPen(pg.mkPen((80, 80, 200), width=3))
            region.setHoverPen(pg.mkPen((30, 30, 180), width=4))
        except Exception:
            pass
        try:
            lines = region.lines if hasattr(region, "lines") else ()
            for line in lines:
                line.setMovable(True)
                if hasattr(line, "setPen"):
                    line.setPen(pg.mkPen((80, 80, 200), width=3))
                if hasattr(line, "setHoverPen"):
                    line.setHoverPen(pg.mkPen((30, 30, 180), width=4))
        except Exception:
            pass
        # 允許在色條上滾動／平移直方圖範圍，方便對準拉柄
        try:
            if self.hist.orientation == "vertical":
                self.hist.vb.setMouseEnabled(x=False, y=True)
            else:
                self.hist.vb.setMouseEnabled(x=True, y=False)
        except Exception:
            pass

    # ----- public API -------------------------------------------------
    def set_plot_title(self, title):
        self._title = title
        self.plot.setTitle(title)

    def set_tick_spacing(self, x_major, x_minor, y_major, y_minor, show_grid=True, grid_alpha=0.25):
        """設定熱圖與剖面軸刻度。"""
        self.plot.getAxis("bottom").setTickSpacing(major=x_major, minor=x_minor)
        self.plot.getAxis("left").setTickSpacing(major=y_major, minor=y_minor)
        if show_grid:
            self.plot.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.plot_x_profile is not None:
            self.plot_x_profile.getAxis("bottom").setTickSpacing(major=x_major, minor=x_minor)
            self.plot_x_profile.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.plot_y_profile is not None:
            self.plot_y_profile.getAxis("left").setTickSpacing(major=y_major, minor=y_minor)
            self.plot_y_profile.showGrid(x=True, y=True, alpha=grid_alpha)

    def set_axis_labels(self, x_label=None, y_label=None):
        if x_label is not None:
            self._x_label = x_label
            self.plot.setLabel("bottom", x_label)
            if self._with_profiles and self.plot_x_profile is not None:
                self.plot_x_profile.setLabel("bottom", x_label)
        if y_label is not None:
            self._y_label = y_label
            self.plot.setLabel("left", y_label)
            if self._with_profiles and self.plot_y_profile is not None:
                self.plot_y_profile.setLabel("left", y_label)

    def set_image(
        self,
        image,
        rect=None,
        levels=None,
        reset_view=True,
        source_matrix=None,
        x_coords=None,
        y_coords=None,
    ):
        """image: 已轉置後給 ImageItem 的陣列；levels=(min,max) 可選。
        source_matrix / x_coords / y_coords：剖面用（列=Y、欄=X）。"""
        self.image_item.setImage(image, autoLevels=False)
        self._apply_data_rect(rect)
        if levels is not None:
            self.set_default_levels(levels[0], levels[1], apply=True)
        elif image is not None:
            finite = np.asarray(image, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                vmin, vmax = float(np.min(finite)), float(np.max(finite))
                if vmin == vmax:
                    vmin -= 1.0
                    vmax += 1.0
                self.set_default_levels(vmin, vmax, apply=True)
        # levels 更新後再套一次 rect，避免 transform 被重設
        self._apply_data_rect(self._data_rect)
        if self._with_profiles:
            self._bind_profile_data(image, source_matrix, x_coords, y_coords)
        if reset_view:
            self.reset_view()
        if self._with_profiles and self._profile_matrix is not None:
            ny, nx = self._profile_matrix.shape
            if self._profile_point is None:
                self.set_profile_point(nx // 2, ny // 2, reset_view=True)
            else:
                ix, iy = self._profile_point
                self.set_profile_point(ix, iy, reset_view=reset_view)

    def _bind_profile_data(self, image, source_matrix, x_coords, y_coords):
        if source_matrix is not None:
            matrix = np.asarray(source_matrix, dtype=float)
        elif image is not None:
            # ImageItem 為 (nx, ny)，轉回 (ny, nx)
            matrix = np.asarray(image, dtype=float).T
        else:
            self._profile_matrix = None
            self._profile_x = None
            self._profile_y = None
            return
        ny, nx = matrix.shape
        if x_coords is not None:
            xs = np.asarray(x_coords, dtype=float)
        elif self._data_rect is not None and nx > 0:
            dx = self._data_rect.width() / nx
            xs = self._data_rect.left() + (np.arange(nx) + 0.5) * dx
        else:
            xs = np.arange(nx, dtype=float)
        if y_coords is not None:
            ys = np.asarray(y_coords, dtype=float)
        elif self._data_rect is not None and ny > 0:
            dy = self._data_rect.height() / ny
            ys = self._data_rect.top() + (np.arange(ny) + 0.5) * dy
        else:
            ys = np.arange(ny, dtype=float)
        if xs.size != nx:
            xs = np.arange(nx, dtype=float)
        if ys.size != ny:
            ys = np.arange(ny, dtype=float)
        self._profile_matrix = matrix
        self._profile_x = xs
        self._profile_y = ys
        if self._profile_point is not None:
            ix, iy = self._profile_point
            self._profile_point = (
                int(np.clip(ix, 0, nx - 1)),
                int(np.clip(iy, 0, ny - 1)),
            )

    def _apply_data_rect(self, rect):
        """將影像對應到真實座標（mm）；並記住供 reset_view 使用。"""
        if rect is None:
            return
        if not isinstance(rect, QRectF):
            rect = QRectF(rect)
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self._data_rect = QRectF(rect)
        try:
            # setOpts(rect=...) 比單獨 setRect 更穩定
            self.image_item.setOpts(rect=self._data_rect)
        except Exception:
            self.image_item.setRect(self._data_rect)

    def set_default_levels(self, vmin, vmax, apply=True):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        self._default_levels = (float(vmin), float(vmax))
        self._data_hist_range = (float(vmin), float(vmax))
        if apply:
            self.apply_levels(vmin, vmax, update_hist_range=True)

    def apply_levels(self, vmin, vmax, update_hist_range=False):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin -= 1e-9
            vmax += 1e-9
        vmin, vmax = float(vmin), float(vmax)
        self._level_updating = True
        try:
            self.hist.setLevels(vmin, vmax)
            if update_hist_range:
                # 輸入／復原上下限後，色條可視範圍跟著縮放到該區間（拉條自動放大）
                span = max(vmax - vmin, 1e-12)
                pad = self._HIST_RANGE_PAD * span
                self.hist.setHistogramRange(vmin - pad, vmax + pad, padding=0)
                self.hist.setLevels(vmin, vmax)
            self._enable_level_drag()
        finally:
            self._level_updating = False
        self._refresh_level_labels()
        self.levelsChanged.emit(vmin, vmax)

    def get_levels(self):
        return self.hist.getLevels()

    def _view_rect(self):
        """取得影像在 view 座標系中的矩形（優先用設定的 mm rect）。"""
        if self._data_rect is not None and self._data_rect.width() > 0:
            return QRectF(self._data_rect)
        vb = self.plot.getViewBox()
        if vb is None or self.image_item.image is None:
            return None
        try:
            # boundingRect 是局部像素座標，需映射到 view（含 setRect transform）
            local = self.image_item.boundingRect()
            poly = vb.mapFromItemToView(self.image_item, local)
            xs = [p.x() for p in poly]
            ys = [p.y() for p in poly]
            return QRectF(
                min(xs), min(ys),
                max(xs) - min(xs),
                max(ys) - min(ys),
            )
        except Exception:
            return None

    def reset_view(self):
        """雙擊圖面：依 mm 真實比例復原（不壓縮）；剖面獨立、不跟著動。"""
        vb = self.plot.getViewBox()
        if vb is None or self.image_item.image is None:
            return
        self._apply_data_rect(self._data_rect)
        rect = self._view_rect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return
        try:
            vb.enableAutoRange(x=False, y=False)
            # 必須先清掉 limits，否則無法為 1:1 比例在短軸方向留白（會被裁切壓縮）
            vb.setLimits(
                xMin=None, xMax=None, yMin=None, yMax=None,
                minXRange=None, maxXRange=None, minYRange=None, maxYRange=None,
            )
            x0, x1 = float(rect.left()), float(rect.right())
            y0, y1 = float(rect.top()), float(rect.bottom())
            span_x = max(x1 - x0, 1e-9)
            span_y = max(y1 - y0, 1e-9)
            if self._aspect_locked:
                self._apply_aspect_fit_range(vb, x0, x1, y0, y1, pad=0.06)
            else:
                vb.setAspectLocked(lock=False)
                vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.04)
            # limits 要比「比例復原後的視野」更寬，允許之後平移／縮放與再次留白
            try:
                vr = vb.viewRange()
                vx0, vx1 = float(vr[0][0]), float(vr[0][1])
                vy0, vy1 = float(vr[1][0]), float(vr[1][1])
                vw = max(vx1 - vx0, span_x)
                vh = max(vy1 - vy0, span_y)
            except Exception:
                vx0, vx1, vy0, vy1 = x0, x1, y0, y1
                vw, vh = span_x, span_y
            vb.setLimits(
                xMin=min(x0, vx0) - 2.0 * vw,
                xMax=max(x1, vx1) + 2.0 * vw,
                yMin=min(y0, vy0) - 2.0 * vh,
                yMax=max(y1, vy1) + 2.0 * vh,
            )
            vb.enableAutoRange(x=False, y=False)
        except Exception:
            pass
        try:
            self.plot.hideButtons()
        except Exception:
            pass
        # 刻意不改剖面視野：剖面與熱圖已斷開連動

    def _apply_aspect_fit_range(self, vb, x0, x1, y0, y1, pad=0.05):
        """鎖 1:1 mm 比例並完整包住資料（多餘方向留白，如圖二）。"""
        span_x = max(x1 - x0, 1e-9)
        span_y = max(y1 - y0, 1e-9)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        pw = max(float(vb.width()), 1.0)
        ph = max(float(vb.height()), 1.0)
        scale = max(span_x / pw, span_y / ph) * (1.0 + float(pad))
        view_w = scale * pw
        view_h = scale * ph
        xr = (cx - 0.5 * view_w, cx + 0.5 * view_w)
        yr = (cy - 0.5 * view_h, cy + 0.5 * view_h)
        # setAspectLocked(True) 會 updateViewRange 並可能裁切；先設好視野再寫入 state
        vb.setAspectLocked(lock=False)
        vb.setRange(xRange=xr, yRange=yr, padding=0, disableAutoRange=True)
        vb.state["aspectLocked"] = 1.0

    def reset_levels(self):
        """雙擊色條：復原預設色階上下限。"""
        if self._default_levels is None:
            return
        vmin, vmax = self._default_levels
        self.apply_levels(vmin, vmax, update_hist_range=True)

    def _on_toggle_cross(self, hidden):
        self._show_profile_cross = not bool(hidden)
        self.btn_toggle_cross.setText("顯示十字" if hidden else "隱藏十字")
        self._redraw_profile_cross()

    def set_profile_at_view(self, x, y, reset_view=False):
        """依視圖座標（mm）對應最近網格點並更新剖面。"""
        if self._profile_matrix is None or self._profile_x is None or self._profile_y is None:
            return
        ix = int(np.clip(np.abs(self._profile_x - float(x)).argmin(), 0, self._profile_x.size - 1))
        iy = int(np.clip(np.abs(self._profile_y - float(y)).argmin(), 0, self._profile_y.size - 1))
        self.set_profile_point(ix, iy, reset_view=reset_view)

    def set_profile_point(self, ix, iy, reset_view=False):
        if self._profile_matrix is None:
            return
        ny, nx = self._profile_matrix.shape
        ix = int(np.clip(ix, 0, nx - 1))
        iy = int(np.clip(iy, 0, ny - 1))
        self._profile_point = (ix, iy)
        self._redraw_profile_cross()
        self.update_inline_profiles(reset_view=reset_view)
        self.profilePointChanged.emit(ix, iy)

    def clear_profile_cross(self):
        for item in self._profile_cross_items:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self._profile_cross_items = []

    def _redraw_profile_cross(self):
        self.clear_profile_cross()
        if (
            not self._with_profiles
            or not self._show_profile_cross
            or self._profile_point is None
            or self._profile_x is None
            or self._profile_y is None
        ):
            return
        ix, iy = self._profile_point
        x = float(self._profile_x[ix])
        y = float(self._profile_y[iy])
        pen = pg.mkPen("#FF1744", width=1.5)
        v_item = pg.InfiniteLine(pos=x, angle=90, pen=pen)
        h_item = pg.InfiniteLine(pos=y, angle=0, pen=pen)
        self.plot.addItem(v_item, ignoreBounds=True)
        self.plot.addItem(h_item, ignoreBounds=True)
        self._profile_cross_items = [v_item, h_item]

    def _relink_profiles(self):
        """保留空實作：剖面刻意不與熱圖連動。"""
        return

    def update_inline_profiles(self, reset_view=False):
        """更新剖面波形。剖面與熱圖視野獨立。"""
        if (
            not self._with_profiles
            or self._profile_matrix is None
            or self._profile_point is None
            or self.plot_x_profile is None
            or self.plot_y_profile is None
        ):
            return
        ix, iy = self._profile_point
        x_profile = np.asarray(self._profile_matrix[iy, :], dtype=float)
        y_profile = np.asarray(self._profile_matrix[:, ix], dtype=float)
        x_axis = np.asarray(self._profile_x, dtype=float)
        y_axis = np.asarray(self._profile_y, dtype=float)
        x_pos = float(x_axis[ix])
        y_pos = float(y_axis[iy])

        keep_x_view = None
        keep_y_view = None
        if not reset_view and self._profile_view_ready:
            try:
                keep_x_view = self.plot_x_profile.viewRange()
                keep_y_view = self.plot_y_profile.viewRange()
            except Exception:
                pass

        self.plot_x_profile.clear()
        self.plot_y_profile.clear()
        self.plot_x_profile.addItem(
            pg.PlotCurveItem(x_axis, x_profile, pen=pg.mkPen("#00ACC1", width=1.5))
        )
        self.plot_x_profile.addItem(
            pg.InfiniteLine(
                pos=x_pos, angle=90, pen=pg.mkPen("#FF1744", width=1, style=Qt.DashLine)
            )
        )
        self.plot_y_profile.addItem(
            pg.PlotCurveItem(y_profile, y_axis, pen=pg.mkPen("#FF5722", width=1.5))
        )
        self.plot_y_profile.addItem(
            pg.InfiniteLine(
                pos=y_pos, angle=0, pen=pg.mkPen("#FF1744", width=1, style=Qt.DashLine)
            )
        )
        self.plot_x_profile.titleLabel.hide()
        self.plot_y_profile.titleLabel.hide()
        self.plot_x_profile.setToolTip(f"Horizontal profile at Y = {y_pos:.6g}")
        self.plot_y_profile.setToolTip(f"Vertical profile at X = {x_pos:.6g}")

        finite = np.concatenate(
            [x_profile[np.isfinite(x_profile)], y_profile[np.isfinite(y_profile)]]
        )
        if finite.size:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            pad = 0.05 * (hi - lo)
            default_i0, default_i1 = lo - pad, hi + pad
        else:
            default_i0, default_i1 = 0.0, 1.0

        if keep_x_view is not None and keep_y_view is not None:
            self.plot_x_profile.setRange(
                xRange=keep_x_view[0], yRange=keep_x_view[1], padding=0
            )
            self.plot_y_profile.setRange(
                xRange=keep_y_view[0], yRange=keep_y_view[1], padding=0
            )
        else:
            if x_axis.size:
                self.plot_x_profile.setXRange(
                    float(x_axis[0]), float(x_axis[-1]), padding=0.02
                )
            self.plot_x_profile.setYRange(default_i0, default_i1, padding=0)
            self.plot_y_profile.setXRange(default_i0, default_i1, padding=0)
            if y_axis.size:
                self.plot_y_profile.setYRange(
                    float(y_axis[0]), float(y_axis[-1]), padding=0.02
                )
            self._profile_view_ready = True

        configure_stable_plot_item(self.plot_x_profile, mouse_enabled=True)
        configure_stable_plot_item(self.plot_y_profile, mouse_enabled=True)
        self.plot_x_profile.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot_y_profile.getViewBox().setMouseEnabled(x=True, y=True)

    def _reset_profile_view(self, which="both"):
        """雙擊剖面：復原該剖面完整範圍（不影響熱圖）。"""
        self._profile_view_ready = False
        if which == "x":
            # 只重設 X 剖面：暫時清 keep 邏輯，強制 default
            if self.plot_x_profile is None or self._profile_matrix is None:
                return
            ix, iy = self._profile_point
            x_profile = np.asarray(self._profile_matrix[iy, :], dtype=float)
            x_axis = np.asarray(self._profile_x, dtype=float)
            finite = x_profile[np.isfinite(x_profile)]
            if finite.size == 0 or x_axis.size == 0:
                return
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            pad = 0.05 * (hi - lo)
            self.plot_x_profile.setXRange(float(x_axis[0]), float(x_axis[-1]), padding=0.02)
            self.plot_x_profile.setYRange(lo - pad, hi + pad, padding=0)
        elif which == "y":
            if self.plot_y_profile is None or self._profile_matrix is None:
                return
            ix, iy = self._profile_point
            y_profile = np.asarray(self._profile_matrix[:, ix], dtype=float)
            y_axis = np.asarray(self._profile_y, dtype=float)
            finite = y_profile[np.isfinite(y_profile)]
            if finite.size == 0 or y_axis.size == 0:
                return
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            pad = 0.05 * (hi - lo)
            self.plot_y_profile.setXRange(lo - pad, hi + pad, padding=0)
            self.plot_y_profile.setYRange(float(y_axis[0]), float(y_axis[-1]), padding=0.02)
        else:
            self.update_inline_profiles(reset_view=True)
        configure_stable_plot_item(self.plot_x_profile, mouse_enabled=True)
        configure_stable_plot_item(self.plot_y_profile, mouse_enabled=True)
        if self.plot_x_profile is not None:
            self.plot_x_profile.getViewBox().setMouseEnabled(x=True, y=True)
        if self.plot_y_profile is not None:
            self.plot_y_profile.getViewBox().setMouseEnabled(x=True, y=True)

    def export_current_image(self):
        safe_title = (
            self._title.replace(" ", "_").replace("/", "-").replace("\\", "-")
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出當前圖檔",
            f"{safe_title}.png",
            "PNG 圖片 (*.png);;JPEG 圖片 (*.jpg);;所有檔案 (*)",
        )
        if not path:
            return False
        try:
            # 優先匯出整組圖面＋色條
            target = getattr(self.win, "ci", None) or self.plot
            exporter = pg_export.ImageExporter(target)
            w = max(int(self.win.width()), 400)
            h = max(int(self.win.height()), 300)
            exporter.parameters()["width"] = w
            exporter.parameters()["height"] = h
            exporter.export(path)
            QMessageBox.information(self, "匯出完成", f"已匯出：\n{path}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出圖檔：\n{exc}")
            return False

    # ----- internal ---------------------------------------------------
    def _format_level_value(self, value):
        """固定顯示寬度用的色階數值格式（避免工具列被撐開）。"""
        try:
            v = float(value)
        except Exception:
            return "--"
        av = abs(v)
        if av == 0.0:
            return "0"
        if av >= 1e4 or (av < 1e-3 and av > 0):
            return f"{v:.3e}"
        return f"{v:.5g}"

    def _refresh_level_labels(self):
        try:
            vmin, vmax = self.hist.getLevels()
        except Exception:
            self.lbl_level_min.setText("下限: --")
            self.lbl_level_max.setText("上限: --")
            return
        self.lbl_level_min.setText(f"下限: {self._format_level_value(vmin)}")
        self.lbl_level_max.setText(f"上限: {self._format_level_value(vmax)}")

    def _on_hist_levels_changed(self):
        if self._level_updating:
            return
        self._refresh_level_labels()
        try:
            vmin, vmax = self.hist.getLevels()
            self.levelsChanged.emit(float(vmin), float(vmax))
        except Exception:
            pass

    def _on_scene_clicked(self, evt):
        pos = evt.scenePos()
        try:
            on_hist = self.hist.sceneBoundingRect().contains(pos)
        except Exception:
            on_hist = False

        if evt.double():
            if on_hist:
                self.reset_levels()
                return
            # 剖面雙擊：復原該剖面（不影響熱圖）
            if self._with_profiles:
                try:
                    if (
                        self.plot_x_profile is not None
                        and self.plot_x_profile.sceneBoundingRect().contains(pos)
                    ):
                        self._reset_profile_view(which="x")
                        return
                    if (
                        self.plot_y_profile is not None
                        and self.plot_y_profile.sceneBoundingRect().contains(pos)
                    ):
                        self._reset_profile_view(which="y")
                        return
                except Exception:
                    pass
            if self.plot.sceneBoundingRect().contains(pos):
                self.reset_view()
            return

        # 單擊熱圖：更新剖面十字與波形
        if self._with_profiles and not on_hist and self.plot.sceneBoundingRect().contains(pos):
            if evt.button() not in (Qt.LeftButton, Qt.NoButton):
                return
            point = self.plot.getViewBox().mapSceneToView(pos)
            self.set_profile_at_view(point.x(), point.y(), reset_view=False)

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mouse_point = self.plot.getViewBox().mapSceneToView(pos)
            self.mouseMoved.emit(mouse_point)
        else:
            self.mouseMoved.emit(None)
    def _prompt_level(self, which):
        """which: 'min' or 'max'（輔助精調；主要仍建議拖曳色條）。"""
        try:
            cur_min, cur_max = self.hist.getLevels()
        except Exception:
            return
        if which == "min":
            title, label, value = "設定色階下限", "請輸入最小值（色階下限）:", cur_min
        else:
            title, label, value = "設定色階上限", "請輸入最大值（色階上限）:", cur_max

        new_val, ok = QInputDialog.getDouble(
            self, title, label, float(value), -1e30, 1e30, 6
        )
        if not ok:
            return
        if which == "min":
            new_min, new_max = float(new_val), float(cur_max)
        else:
            new_min, new_max = float(cur_min), float(new_val)
        if new_min >= new_max:
            QMessageBox.warning(self, "提醒", "下限必須小於上限。")
            return
        self.apply_levels(new_min, new_max, update_hist_range=True)

    def _on_max_label_clicked(self, event):
        if event.button() == Qt.LeftButton:
            self._prompt_level("max")
        QLabel.mousePressEvent(self.lbl_level_max, event)

    def _on_min_label_clicked(self, event):
        if event.button() == Qt.LeftButton:
            self._prompt_level("min")
        QLabel.mousePressEvent(self.lbl_level_min, event)


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

        apply_readable_plot_theme(self.win, [self.plot])
        
        # 產生等高線
        min_v, max_v = float(np.min(matrix_data)), float(np.max(matrix_data))
        levels = np.linspace(min_v, max_v, 10)
        for level in levels:
            iso = pg.IsocurveItem(data=self.matrix_data.T, level=level, pen=pg.mkPen('#263238', width=0.8))
            self.plot.addItem(iso)