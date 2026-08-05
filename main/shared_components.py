import numpy as np
import pyqtgraph as pg
from scipy.ndimage import label as ndi_label
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QMainWindow, QWidget, QVBoxLayout, QCheckBox, QHBoxLayout
from PyQt5.QtCore import Qt


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
PROFILE_SIDE_AXIS_PX = 48   # V 左軸／H 右軸
PROFILE_EDGE_AXIS_PX = 28   # V 上軸／H 下軸


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


def configure_equal_profile_strips(layout_widget, profile_col=0, profile_row=1):
    """讓縱剖面欄寬與橫剖面列高一致，訊號 ViewBox 視覺尺寸相同。"""
    if layout_widget is None or not hasattr(layout_widget, "ci"):
        return
    col_w = PROFILE_VIEW_PX + PROFILE_SIDE_AXIS_PX
    row_h = PROFILE_VIEW_PX + PROFILE_EDGE_AXIS_PX
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
                             bg_subtract=True, largest_cc_only=True, subpixel=True):
    """計算光斑中心。centroid／thresh_geom 預設：背景扣除 + 最大連通區 + 亞像素。

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
        weights = work[mask]
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

        self.hist = pg.HistogramLUTItem()
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
            self.hist.setHistogramRange(min_v, max_v)
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