import math
import os
from datetime import datetime
import zipfile

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pg_export
from scipy.ndimage import (
    binary_fill_holes as ndi_fill_holes,
    distance_transform_edt as ndi_edt,
    gaussian_filter as ndi_gaussian_filter,
    label as ndi_label,
)
from PyQt5.QtWidgets import (
    QApplication, QSpinBox, QDoubleSpinBox, QMainWindow, QWidget, QVBoxLayout,
    QCheckBox, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
    QInputDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QTimer
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


def export_timestamp_tag():
    """匯出檔名用時間戳（本地時間）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# 匯出圖檔預設 PNG（體積較小、相容性好；已改為直接匯出資料夾無需避開 ZIP 壓縮）。
EXPORT_IMAGE_EXT = ".png"
EXPORT_IMAGE_FILTER = "PNG 圖片 (*.png);;所有檔案 (*)"
EXPORT_ZIP_COMPRESSION = zipfile.ZIP_STORED


def with_export_timestamp(path_or_base, stamp=None):
    """在檔名（副檔名前）加上 _{YYYYMMDD_HHMMSS}。

    - 可傳完整路徑或無副檔名的 base。
    - 若結尾已是同一個 stamp，不重複附加。
    - stamp 為 None 時產生當下時間戳。
    """
    stamp = stamp or export_timestamp_tag()
    directory = os.path.dirname(path_or_base)
    base = os.path.basename(path_or_base)
    stem, ext = os.path.splitext(base)
    suffix = f"_{stamp}"
    if stem.endswith(suffix):
        new_base = base
    else:
        new_base = f"{stem}{suffix}{ext}"
    if directory:
        return os.path.join(directory, new_base)
    return new_base


def export_stamped_filename(base_name, ext=EXPORT_IMAGE_EXT, stamp=None):
    """在檔名（不含副檔名）後加上時間戳，避免覆蓋舊檔。"""
    stamp = stamp or export_timestamp_tag()
    stem = os.path.splitext(str(base_name))[0]
    return with_export_timestamp(f"{stem}{ext}", stamp=stamp)


def export_stamped_path(folder, base_name, ext=EXPORT_IMAGE_EXT, stamp=None):
    return os.path.join(
        folder, export_stamped_filename(base_name, ext=ext, stamp=stamp)
    )



def normalize_export_image_path(path: str) -> str:
    """統一匯出為 PNG。"""
    base, ext = os.path.splitext(path)
    if ext.lower() in (".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ""):
        return f"{base}{EXPORT_IMAGE_EXT}"
    return path


def export_plot_image(target, path, width=None, height=None) -> None:
    """匯出 pyqtgraph 圖面為 PNG。"""
    out_path = normalize_export_image_path(path)
    exporter = pg_export.ImageExporter(target)
    if width is not None:
        exporter.parameters()["width"] = int(width)
    if height is not None:
        exporter.parameters()["height"] = int(height)
    exporter.export(out_path)


def save_qpixmap_export(pix, path) -> bool:
    """QPixmap 匯出為 PNG。"""
    out_path = normalize_export_image_path(path)
    return bool(pix.save(out_path, "PNG"))



# Mapping 合理量測值上限；超出或非有限值視為無效（如 DBL_MIN 哨兵值）
MAPPING_VALUE_ABS_MAX = 1e6


def sanitize_numeric_values(values, abs_max=MAPPING_VALUE_ABS_MAX):
    """將非有限值與極端值改為 NaN。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    out = arr.copy()
    bad = ~np.isfinite(out) | (np.abs(out) > float(abs_max))
    out[bad] = np.nan
    return out


def finite_value_minmax(values, default=(0.0, 1.0)):
    """回傳有效數值的 min/max，供色階與 histogram 使用。"""
    clean = sanitize_numeric_values(values)
    finite = clean[np.isfinite(clean)]
    if finite.size == 0:
        return default
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return default
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return vmin, vmax

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


def _reserve_axis_slot(plot, axis_name, size_px, visible_values=False):
    """保留軸佔位寬／高，使相鄰 PlotItem 的 ViewBox 邊緣對齊。"""
    if plot is None:
        return
    try:
        ax = plot.getAxis(axis_name)
    except Exception:
        return
    plot.showAxis(axis_name, show=True)
    if not visible_values:
        try:
            ax.setStyle(showValues=False, tickLength=0)
        except Exception:
            pass
        try:
            ax.setLabel("")
        except Exception:
            pass
        try:
            ax.setPen(pg.mkPen(None))
            ax.setTextPen(pg.mkPen(None))
        except Exception:
            pass
    size_px = int(max(0, size_px))
    if axis_name in ("left", "right"):
        ax.setWidth(size_px)
    else:
        ax.setHeight(size_px)


def align_profile_viewboxes(
    heat_plot,
    x_profile,
    y_profile,
    corner=None,
    side_axis_px=None,
    edge_axis_px=None,
):
    """
    統一熱圖／剖面軸佔位，讓：
    - 下方橫剖面 ViewBox 左右緣對齊熱圖 ViewBox（同欄：左／右軸寬一致）
    - 左側縱剖面 ViewBox 上下緣對齊熱圖 ViewBox（同列：上／下軸高＋標題列一致）
    """
    if heat_plot is None or x_profile is None or y_profile is None:
        return
    side = PROFILE_SIDE_AXIS_PX_COMPACT if side_axis_px is None else int(side_axis_px)
    edge = PROFILE_EDGE_AXIS_PX_COMPACT if edge_axis_px is None else int(edge_axis_px)

    # 同欄對齊：熱圖與橫剖面的左／右軸寬必須相同
    heat_plot.showAxis("left", show=True)
    heat_plot.showAxis("bottom", show=True)
    heat_plot.getAxis("left").setWidth(side)
    heat_plot.getAxis("bottom").setHeight(edge)
    _reserve_axis_slot(heat_plot, "right", side, visible_values=False)
    # 同列對齊：熱圖上緣佔位 = 縱剖面「Value」上軸高度
    _reserve_axis_slot(heat_plot, "top", edge, visible_values=False)

    x_profile.showAxis("bottom", show=True)
    x_profile.showAxis("right", show=True)
    x_profile.getAxis("bottom").setHeight(edge)
    x_profile.getAxis("right").setWidth(side)
    _reserve_axis_slot(x_profile, "left", side, visible_values=False)
    _reserve_axis_slot(x_profile, "top", 0, visible_values=False)

    y_profile.showAxis("left", show=True)
    y_profile.showAxis("top", show=True)
    y_profile.getAxis("left").setWidth(side)
    y_profile.getAxis("top").setHeight(edge)
    _reserve_axis_slot(y_profile, "bottom", edge, visible_values=False)
    _reserve_axis_slot(y_profile, "right", 0, visible_values=False)

    if corner is not None:
        _reserve_axis_slot(corner, "left", side, visible_values=False)
        _reserve_axis_slot(corner, "bottom", edge, visible_values=False)
        _reserve_axis_slot(corner, "top", 0, visible_values=False)
        _reserve_axis_slot(corner, "right", 0, visible_values=False)

    for plot in (heat_plot, x_profile, y_profile, corner):
        if plot is None:
            continue
        try:
            plot.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass

    # 標題留在熱圖畫布上方；縱剖面用同等高空白標題列佔位，ViewBox 才對齊
    _sync_profile_title_rows(heat_plot, y_profile, x_profile, corner)


def _sync_profile_title_rows(heat_plot, y_profile, x_profile=None, corner=None):
    """熱圖顯示標題；左側／下方剖面保留相同標題列高度（空白）。"""
    title_h = 22
    try:
        if hasattr(heat_plot, "titleLabel") and heat_plot.titleLabel is not None:
            heat_plot.titleLabel.setMaximumHeight(16777215)
            heat_plot.titleLabel.show()
            br = heat_plot.titleLabel.boundingRect()
            title_h = int(max(18.0, br.height()))
    except Exception:
        pass

    # 縱剖面：空白標題列，高度與熱圖標題一致
    try:
        y_profile.setTitle(" ")  # 保留列高，不顯示文字感
        if hasattr(y_profile, "titleLabel") and y_profile.titleLabel is not None:
            y_profile.titleLabel.setMaximumHeight(title_h)
            y_profile.titleLabel.setMinimumHeight(title_h)
            y_profile.titleLabel.show()
            # 透明／無字，避免旁邊多一行標題
            try:
                y_profile.titleLabel.setText("")
            except Exception:
                pass
    except Exception:
        pass

    # 橫剖面與角落：標題列高度歸零（它們在下一列，不影響同列對齊）
    for plot in (x_profile, corner):
        if plot is None:
            continue
        try:
            if hasattr(plot, "titleLabel") and plot.titleLabel is not None:
                plot.titleLabel.setMaximumHeight(0)
                plot.titleLabel.hide()
        except Exception:
            pass

    try:
        if hasattr(heat_plot, "titleLabel") and heat_plot.titleLabel is not None:
            heat_plot.titleLabel.setMinimumHeight(title_h)
            heat_plot.titleLabel.setMaximumHeight(title_h)
    except Exception:
        pass


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
    """背景扣除後依門檻建 mask，可選擇只保留最大連通區。

    最大連通區會填補內部空洞，讓門檻 overlay 與最大內切圓使用同一實體區域。
    """
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
        mask = ndi_fill_holes(mask)
    return mask


def fit_max_inscribed_circle(mask):
    """對二值 mask 求最大內切圓（歐氏距離變換）。

    先填補內部空洞，再取 EDT 最大值：門檻區內零星低於門檻的像素
    會形成封閉空洞，若不填洞則內切圓半徑會被卡住、圓心易貼近亮心。

    Returns:
        (cx, cy, radius_px) 或 None。座標為像素中心（與 mask 索引對齊）。
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return None
    filled = ndi_fill_holes(mask)
    dist = ndi_edt(filled)
    flat = int(np.argmax(dist))
    cy, cx = np.unravel_index(flat, dist.shape)
    radius = float(dist[cy, cx])
    if not np.isfinite(radius) or radius <= 0:
        return None
    return float(cx), float(cy), radius


def fit_max_inscribed_circle_from_matrix(
    matrix,
    thresh_percent=50.0,
    *,
    bg_subtract=True,
    largest_cc_only=True,
):
    """依門檻％建 mask 後求最大內切圓（與抓光斑門檻語意一致）。"""
    mask = build_robust_threshold_mask(
        matrix,
        use_threshold=True,
        thresh_percent=float(thresh_percent),
        bg_subtract=bg_subtract,
        largest_cc_only=largest_cc_only,
    )
    if mask is None:
        return None
    return fit_max_inscribed_circle(mask)


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


def _smooth_1d_profile(y, win=7):
    """1D 邊界延伸平滑（供波谷／雙峰偵測）。"""
    y = np.asarray(y, dtype=np.float64)
    k = max(1, int(win))
    if k % 2 == 0:
        k += 1
    if k <= 1 or y.size < k:
        return y.astype(np.float64, copy=True)
    pad = k // 2
    kernel = np.ones(k, dtype=np.float64) / float(k)
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_maxima_1d(y, *, min_prominence_frac=0.12):
    """找 1D 剖面上的局部極大（略過邊界 artifact）。"""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3:
        return [int(np.argmax(y))] if y.size else []
    peak = float(np.max(y))
    if peak <= 0:
        return []
    min_prom = peak * float(min_prominence_frac)
    margin = max(4, len(y) // 40)
    idxs = []
    for i in range(margin, len(y) - margin):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_prom:
            if y[i] > y[i - 1] or y[i] > y[i + 1]:
                idxs.append(i)
    if not idxs:
        band = y[margin: len(y) - margin]
        if band.size:
            idxs = [margin + int(np.argmax(band))]
        else:
            idxs = [int(np.argmax(y))]
    return idxs


def _min_lobe_separation(n, min_peak_distance):
    """峰間最小距離；尊重 UI min_peak_distance。"""
    if min_peak_distance is not None:
        return max(1, int(min_peak_distance))
    return max(3, min(8, max(3, n // 40)))


def _peak_fwhm_1d(y, p):
    """估算單峰 FWHM（px）；寬散射峰 → 較大值。"""
    return _peak_width_at_frac_1d(y, p, frac=0.5)


def _peak_width_at_frac_1d(y, p, frac=0.5):
    """峰寬：強度降至 amp×frac 的左右跨度（px）。

    frac>0.5（如 0.65）可避開近距伴峰肩膀把半高寬撐大，
    以免第二峰搜尋的排除帶吃掉真雙峰。
    """
    n = len(y)
    if n < 3 or p < 0 or p >= n:
        return float(n)
    amp = float(y[p])
    if amp <= 0:
        return float(n)
    thr = float(np.clip(frac, 0.05, 0.95)) * amp
    left = int(p)
    while left > 0 and float(y[left]) >= thr:
        left -= 1
    right = int(p)
    while right < n - 1 and float(y[right]) >= thr:
        right += 1
    return float(max(1, right - left))


def _estimate_max_pair_sep(n, y, p1, min_sep, *, max_sep_cap_px=None):
    """以 FWHM 推估伴峰搜尋窗寬；硬上限阻擋 dual+散射 (~700 µm) 誤配。"""
    fwhm = _peak_fwhm_1d(y, p1)
    est = max(min_sep * 3, int(4.0 * fwhm), 40)
    default_cap = max(48, min(80, n // 5))
    hard_cap = int(max_sep_cap_px) if max_sep_cap_px is not None else default_cap
    hard_cap = max(min_sep + 1, hard_cap)
    return int(np.clip(est, min_sep + 1, hard_cap))


def _dual_peak_pair_score(y, a, b):
    """局部雙峰配對分數：強度×平衡×波谷對比×銳度×緊緻度（不獎勵大間距）。"""
    lo, hi = (a, b) if a < b else (b, a)
    ya = float(y[a])
    yb = float(y[b])
    weaker = min(ya, yb)
    stronger = max(ya, yb)
    if stronger <= 0.0 or weaker <= 0.0:
        return -1.0
    global_peak = float(np.max(y)) if y.size else stronger
    abs_w = weaker / max(global_peak, 1e-12)
    if abs_w < 0.40:
        return -1.0
    balance = weaker / stronger
    sep = int(hi - lo)
    if sep < 2:
        valley = 0.5 * (ya + yb)
    else:
        valley = float(np.min(y[lo: hi + 1]))
    contrast = (weaker - valley) / max(weaker, 1e-12)
    if contrast <= 0.0:
        return -1.0
    wa = _peak_fwhm_1d(y, a)
    wb = _peak_fwhm_1d(y, b)
    sharp = 1.0 / (1.0 + 0.35 * (wa + wb))
    width_ref = max(wa, wb, 4.0)
    compact = 1.0 / (1.0 + max(0.0, float(sep) - 3.0 * width_ref) / 16.0)
    return weaker * abs_w * balance * contrast * sharp * compact


def _find_dual_peak_valley_detail(
    profile,
    *,
    smooth_win=7,
    min_peak_distance=5,
    max_sep_cap_px=None,
    sep_lo_px=None,
    sep_hi_px=None,
):
    """最佳局部雙峰對 → 波谷嚴格落在該對之間（見 docs/雙光斑波谷誤判與外圍散射修正.md）。"""
    y = _smooth_1d_profile(profile, int(smooth_win) if smooth_win else 7)
    n = len(y)
    if n < 5:
        if n == 0:
            return None
        mid = int(np.argmin(y))
        return {
            "valley": mid,
            "peak_lo": mid,
            "peak_hi": mid,
            "max_pair_sep": max(8, n),
        }

    margin = max(4, n // 40)
    min_sep = _min_lobe_separation(n, min_peak_distance)
    global_peak = (
        float(np.max(y[margin: n - margin])) if n > 2 * margin else float(np.max(y))
    )
    if global_peak <= 0:
        return None

    peaks = _local_maxima_1d(y, min_prominence_frac=0.12)
    merged = []
    for p in sorted(peaks, key=lambda i: float(y[i]), reverse=True):
        if p < margin or p >= n - margin:
            continue
        if all(abs(p - q) >= min_sep for q in merged):
            merged.append(p)
        if len(merged) >= 10:
            break
    if not merged:
        p1 = margin + int(np.argmax(y[margin: n - margin]))
        merged = [p1]

    candidates = []
    anchors = [p for p in merged if float(y[p]) >= 0.35 * global_peak]
    if not anchors:
        anchors = merged[:3]

    for p1 in anchors:
        max_sep = _estimate_max_pair_sep(
            n, y, p1, min_sep, max_sep_cap_px=max_sep_cap_px
        )
        if sep_hi_px is not None and math.isfinite(float(sep_hi_px)):
            max_sep = min(max_sep, max(min_sep + 1, int(round(float(sep_hi_px)))))
        amp1 = float(y[p1])
        cands = [
            p
            for p in merged
            if p != p1
            and min_sep <= abs(p - p1) <= max_sep
            and float(y[p]) >= 0.30 * amp1
        ]
        if not cands:
            lo = max(margin, p1 - max_sep)
            hi = min(n - margin, p1 + max_sep + 1)
            left = y[lo: max(lo + 1, p1 - min_sep + 1)]
            right = y[min(n - margin, p1 + min_sep): hi]
            side = []
            if left.size:
                side.append(lo + int(np.argmax(left)))
            if right.size:
                side.append(int(p1 + min_sep + np.argmax(right)))
            cands = [
                c
                for c in side
                if min_sep <= abs(c - p1) <= max_sep
                and float(y[c]) >= 0.25 * amp1
            ]
        for p2 in cands:
            sep = abs(p1 - p2)
            if sep_lo_px is not None and math.isfinite(float(sep_lo_px)):
                if sep < float(sep_lo_px):
                    continue
            if sep_hi_px is not None and math.isfinite(float(sep_hi_px)):
                if sep > float(sep_hi_px):
                    continue
            score = _dual_peak_pair_score(y, p1, p2)
            if score <= 0:
                continue
            lo, hi = (p1, p2) if p1 < p2 else (p2, p1)
            if hi - lo < 2:
                valley = int((lo + hi) // 2)
            else:
                valley = int(lo + int(np.argmin(y[lo: hi + 1])))
            if float(y[valley]) > 0.90 * min(float(y[lo]), float(y[hi])):
                continue
            valley = int(np.clip(valley, lo, hi))
            candidates.append(
                {
                    "valley": valley,
                    "peak_lo": int(lo),
                    "peak_hi": int(hi),
                    "max_pair_sep": int(max_sep),
                    "score": float(score),
                    "sep": int(sep),
                }
            )

    if not candidates:
        return None

    best_score = max(float(c["score"]) for c in candidates)
    near = [c for c in candidates if float(c["score"]) >= 0.82 * best_score]
    near.sort(key=lambda c: (int(c["sep"]), -float(c["score"])))
    best = near[0]
    return {
        "valley": int(best["valley"]),
        "peak_lo": int(best["peak_lo"]),
        "peak_hi": int(best["peak_hi"]),
        "max_pair_sep": int(best["max_pair_sep"]),
    }


def _best_cut_x_for_valley(img, seed_x=None, *, valley_kw=None):
    """在亮欄位掃描，選局部雙峰對比最強的縱切 X。"""
    h, w = img.shape
    col_max = img.max(axis=0)
    thr = float(np.percentile(col_max, 75.0))
    bright = np.where(col_max >= thr)[0]
    if bright.size == 0:
        bright = np.arange(w)
    x0 = int(bright.min())
    x1 = int(bright.max()) + 1

    best_x = int(np.clip(
        round(float(seed_x) if seed_x is not None else w / 2.0), 0, w - 1
    ))
    best_score = -1.0
    step = max(1, (x1 - x0) // 40)
    kw = dict(valley_kw or {})
    kw.setdefault("smooth_win", 7)
    kw.setdefault("min_peak_distance", 5)
    for x in range(x0, x1, step):
        half = 2
        xa = max(0, x - half)
        xb = min(w, x + half + 1)
        prof = img[:, xa:xb].mean(axis=1)
        detail = _find_dual_peak_valley_detail(prof, **kw)
        if detail is None:
            continue
        lo = int(detail["peak_lo"])
        hi = int(detail["peak_hi"])
        ys = _smooth_1d_profile(prof, int(kw.get("smooth_win") or 7))
        weaker = min(float(ys[lo]), float(ys[hi]))
        valley = float(ys[int(detail["valley"])])
        if weaker <= 0:
            continue
        contrast = (weaker - valley) / weaker
        if contrast <= 0:
            continue
        score = contrast * weaker * _dual_peak_pair_score(ys, lo, hi)
        if score > best_score:
            best_score = score
            best_x = int(x)
    return best_x


def _profile_from_matrix(work_mat, cut_x, col_half_width):
    """沿 cut_x ± col_half_width 取垂直 profile（已 clip 至 work_mat 範圍）。"""
    h, w = work_mat.shape
    cx = float(np.clip(cut_x, 0.0, w - 1.0))
    ci = int(round(cx))
    c0 = max(0, ci - int(col_half_width))
    c1 = min(w, ci + int(col_half_width) + 1)
    profile = np.mean(work_mat[:, c0:c1], axis=1)
    bg = float(np.median(profile)) if profile.size else 0.0
    return np.clip(profile - bg, 0.0, None), profile


def _pack_valley_result(
    *,
    full_h,
    full_w,
    valley,
    cut_x,
    detail,
    profile_raw,
    roi_bounds,
    y_offset=0.0,
):
    """組裝 find_dual_peak_valley_y 回傳 dict（含自動定位帶）。"""
    max_sep = int(detail["max_pair_sep"]) if detail else max(36, full_h // 10)
    peak_ys = None
    if detail is not None:
        peak_ys = (
            float(y_offset + int(detail["peak_lo"])),
            float(y_offset + int(detail["peak_hi"])),
        )

    locate_y0 = int(np.clip(valley - max_sep, 0, full_h - 1))
    locate_y1 = int(np.clip(valley + max_sep + 1, locate_y0 + 1, full_h))
    pad_x = max(8, max_sep // 4)
    locate_x0 = int(np.clip(cut_x - pad_x, 0, full_w - 1))
    locate_x1 = int(np.clip(cut_x + pad_x + 1, locate_x0 + 1, full_w))
    if peak_ys is not None:
        pad_y = max(8, max_sep // 4)
        locate_y0 = int(np.clip(min(peak_ys) - pad_y, 0, full_h - 1))
        locate_y1 = int(np.clip(max(peak_ys) + pad_y + 1, locate_y0 + 1, full_h))

    if roi_bounds is not None:
        rx0, ry0, rx1, ry1 = roi_bounds
        locate_x0 = max(locate_x0, rx0)
        locate_y0 = max(locate_y0, ry0)
        locate_x1 = min(locate_x1, rx1)
        locate_y1 = min(locate_y1, ry1)
        if locate_x1 <= locate_x0:
            locate_x0, locate_x1 = rx0, rx1
        if locate_y1 <= locate_y0:
            locate_y0, locate_y1 = ry0, ry1

    return {
        "valley_y": float(valley),
        "cx": float(cut_x),
        "peak_ys": peak_ys,
        "profile": profile_raw,
        "roi": roi_bounds,
        "max_pair_sep": float(max_sep),
        "locate_bounds": (locate_x0, locate_y0, locate_x1, locate_y1),
    }


def intersect_half_with_locate_band(matrix, split_y, half, locate_bounds=None, *, y_only=False):
    """切分半區 ∩ 自動定位帶 → (子矩陣, x_offset, y_offset) 或 (None, 0, 0)。

    y_only=True 時僅在 Y 方向夾定位帶、保留半區全寬 X。
    內切圓需完整 X 向 blob 才能算出正確半徑，不應過窄裁切 X。
    """
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return None, 0, 0
    h, w = matrix.shape
    y_i = split_y_index(split_y)
    if half == "below":
        y0, y1 = 0, y_i
    elif half == "above":
        y0, y1 = y_i + 1, h
    else:
        raise ValueError(f"unknown half: {half}")
    x0, x1 = 0, w
    if locate_bounds is not None:
        lx0, ly0, lx1, ly1 = locate_bounds
        if not y_only:
            x0 = max(x0, int(lx0))
            x1 = min(x1, int(lx1))
        y0 = max(y0, int(ly0))
        y1 = min(y1, int(ly1))
    if y1 <= y0 or x1 <= x0:
        return None, 0, 0
    return matrix[y0:y1, x0:x1], x0, y0


def find_dual_peak_valley_y(
    matrix,
    cx=None,
    col_half_width=2,
    smooth_win=7,
    min_peak_distance=5,
    roi=None,
    expected_distance_min_um=None,
    expected_distance_max_um=None,
    pixel_pitch_um=5.5,
    max_sep_cap_px=None,
):
    """沿質心 X 縱切取 1D profile，以「最佳局部雙峰對」找波谷 Y。
    無 ROI 時仍可抗遠距外圍散射；啟用 ROI 時質心 seed、縱切與定位帶皆與框相交。

    可選 expected_distance_min/max_um 以 µm 約束雙峰間距（DataRay pitch 預設 5.5）。

    Returns:
        dict: valley_y, cx, peak_ys, profile, roi,
              max_pair_sep, locate_bounds (x0,y0,x1,y1 半開區間，全圖座標)
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return {
            "valley_y": 0.0, "cx": 0.0, "peak_ys": None,
            "profile": np.array([]), "roi": None,
            "max_pair_sep": 0.0, "locate_bounds": None,
        }

    full_h, full_w = matrix.shape
    roi_bounds = clip_roi_to_matrix(matrix, roi)

    pitch = float(pixel_pitch_um) if pixel_pitch_um and pixel_pitch_um > 0 else 5.5
    sep_lo_px = None
    sep_hi_px = None
    if expected_distance_min_um is not None and math.isfinite(float(expected_distance_min_um)):
        sep_lo_px = float(expected_distance_min_um) / pitch
    if expected_distance_max_um is not None and math.isfinite(float(expected_distance_max_um)):
        sep_hi_px = float(expected_distance_max_um) / pitch

    valley_kw = {
        "smooth_win": int(smooth_win) if smooth_win else 7,
        "min_peak_distance": min_peak_distance,
        "max_sep_cap_px": max_sep_cap_px,
        "sep_lo_px": sep_lo_px,
        "sep_hi_px": sep_hi_px,
    }
    half_default = max(1, int(col_half_width) if col_half_width else 2)

    if roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        work_mat = matrix[y0:y1, x0:x1]
        y_offset = float(y0)
        x_offset = float(x0)
        if cx is not None:
            cut_x_local = float(cx) - x_offset
        else:
            try:
                cut_x_local, _cy = compute_auto_spot_center(
                    work_mat, "centroid", use_threshold=True, thresh_percent=50.0,
                    bg_subtract=True, largest_cc_only=True, subpixel=True,
                )
            except Exception:
                cut_x_local = (x1 - x0 - 1) / 2.0
        cut_x_local = float(np.clip(cut_x_local, 0.0, x1 - x0 - 1.0))
        half = max(1, min(half_default, max(1, (x1 - x0) // 4)))
        prof_work, prof_raw = _profile_from_matrix(work_mat, cut_x_local, half)
        detail = _find_dual_peak_valley_detail(prof_work, **valley_kw)
        if detail is None:
            valley = float((y0 + y1) // 2)
        else:
            valley = float(y0 + int(detail["valley"]))
        cut_x_full = float(np.clip(cut_x_local + x_offset, 0.0, full_w - 1.0))
        return _pack_valley_result(
            full_h=full_h,
            full_w=full_w,
            valley=valley,
            cut_x=cut_x_full,
            detail=detail,
            profile_raw=prof_raw,
            roi_bounds=roi_bounds,
            y_offset=y_offset,
        )

    work_mat = matrix
    seed_x = float(cx) if cx is not None else None
    if seed_x is None:
        try:
            seed_x, _ = compute_auto_spot_center(
                work_mat, "centroid", use_threshold=True, thresh_percent=50.0,
                bg_subtract=True, largest_cc_only=True, subpixel=True,
            )
        except Exception:
            seed_x = (full_w - 1) / 2.0

    cut_x = _best_cut_x_for_valley(work_mat, seed_x=seed_x, valley_kw=valley_kw)
    prof_work, prof_raw = _profile_from_matrix(work_mat, cut_x, half_default)
    detail = _find_dual_peak_valley_detail(prof_work, **valley_kw)
    if detail is None:
        sx = int(np.clip(round(float(seed_x)), 0, full_w - 1))
        for hx in (6, 10, 16):
            prof_work, prof_raw = _profile_from_matrix(work_mat, sx, hx)
            detail = _find_dual_peak_valley_detail(prof_work, **valley_kw)
            if detail is not None:
                cut_x = float(sx)
                break
    if detail is None:
        valley = float(full_h // 2)
        return _pack_valley_result(
            full_h=full_h,
            full_w=full_w,
            valley=valley,
            cut_x=float(cut_x),
            detail=None,
            profile_raw=prof_raw,
            roi_bounds=None,
        )

    valley = float(detail["valley"])
    return _pack_valley_result(
        full_h=full_h,
        full_w=full_w,
        valley=valley,
        cut_x=float(cut_x),
        detail=detail,
        profile_raw=prof_raw,
        roi_bounds=None,
    )


def _roi_centered_on_xy(matrix, center_xy, roi_w=160, roi_h=512):
    """以 (cx, cy) 為中心框出 W×H，並裁到影像範圍。回傳 (x, y, w, h)。"""
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return None
    h, w = matrix.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])
    rw = max(1, int(roi_w))
    rh = max(1, int(roi_h))
    x0 = int(round(cx - rw / 2.0))
    y0 = int(round(cy - rh / 2.0))
    x0 = int(np.clip(x0, 0, max(0, w - 1)))
    y0 = int(np.clip(y0, 0, max(0, h - 1)))
    rw = min(rw, w - x0)
    rh = min(rh, h - y0)
    if rw < 1 or rh < 1:
        return None
    return (x0, y0, rw, rh)


def _peaks_by_derivative_1d(profile, *, smooth_win=7, min_prominence_frac=0.12):
    """平滑後對 1D 剖面取微分，以 dy 正→負零交越找局部極大（亞像素）。

    Returns:
        list[(y_pos, amplitude)] 依強度由大到小。
    """
    raw = np.asarray(profile, dtype=np.float64)
    if raw.size < 3:
        if raw.size == 0:
            return []
        return [(float(np.argmax(raw)), float(np.max(raw)))]

    y = _smooth_1d_profile(raw, int(smooth_win) if smooth_win else 7)
    dy = np.gradient(y)
    peak_amp = float(np.max(y)) if y.size else 0.0
    if peak_amp <= 0:
        return []
    min_amp = peak_amp * float(min_prominence_frac)
    margin = max(2, len(y) // 50)
    peaks = []
    for i in range(margin, len(dy) - 1 - margin):
        if dy[i] >= 0.0 and dy[i + 1] < 0.0:
            denom = float(dy[i] - dy[i + 1])
            frac = float(dy[i] / denom) if abs(denom) > 1e-15 else 0.5
            pos = float(i) + frac
            amp = float(np.interp(pos, np.arange(len(y), dtype=np.float64), y))
            if amp >= min_amp:
                peaks.append((pos, amp))

    if not peaks:
        # 後備：局部極大（無乾淨零交越時）
        for i in _local_maxima_1d(y, min_prominence_frac=min_prominence_frac):
            peaks.append((float(i), float(y[i])))

    peaks.sort(key=lambda t: t[1], reverse=True)
    return peaks


def _pick_two_peaks_1d(peaks, *, min_peak_distance=8, n_profile=0):
    """從微分峰中選出最佳雙峰（強度高、間距足夠）。"""
    if not peaks:
        return None
    if len(peaks) == 1:
        return (peaks[0][0], peaks[0][0])

    min_sep = max(1, int(min_peak_distance))
    best = None
    best_score = -1.0
    for i, (ya, aa) in enumerate(peaks):
        for yb, ab in peaks[i + 1:]:
            sep = abs(float(ya) - float(yb))
            if sep < min_sep:
                continue
            weaker = min(float(aa), float(ab))
            stronger = max(float(aa), float(ab))
            if stronger <= 0:
                continue
            balance = weaker / stronger
            score = weaker * balance
            if score > best_score:
                best_score = score
                best = (float(ya), float(yb)) if ya <= yb else (float(yb), float(ya))

    if best is not None:
        return best

    # 間距不足時仍取最強兩峰
    y0, y1 = peaks[0][0], peaks[1][0]
    return (float(y0), float(y1)) if y0 <= y1 else (float(y1), float(y0))


def _primary_peak_from_hint(peaks, profile, primary_y_hint):
    """以短曝光對應 Y 為 hint，選最近／最強主峰。"""
    if not peaks:
        if profile.size == 0:
            return None
        return float(np.argmax(profile))
    hint = float(primary_y_hint) if primary_y_hint is not None else None
    if hint is None:
        return float(peaks[0][0])
    # 在 hint 附近優先；過遠則退回全域最強
    near = [(y, a) for y, a in peaks if abs(float(y) - hint) <= max(24.0, 0.12 * len(profile))]
    if near:
        near.sort(key=lambda t: (abs(float(t[0]) - hint), -float(t[1])))
        return float(near[0][0])
    peaks_sorted = sorted(peaks, key=lambda t: abs(float(t[0]) - hint))
    return float(peaks_sorted[0][0])


def _score_secondary_candidate(
    a,
    contrast,
    sep,
    core_w,
    primary_amp,
    *,
    k_near=2.2,
    k_far=3.8,
):
    """NEAR／FAR／MID 第二峰評分；不合格回傳 None。"""
    a = float(a)
    contrast = float(contrast)
    sep = float(sep)
    core_w = max(float(core_w), 1.0)
    primary_amp = max(float(primary_amp), 1e-12)
    ratio = sep / core_w
    amp_ratio = a / primary_amp
    dist_term = 1.0 / (1.0 + max(0.0, sep - core_w) / max(1.5 * core_w, 1.0))

    def _near():
        if contrast < 0.05:
            return None
        return a * (contrast ** 2) * (1.0 + 0.3 * dist_term)

    def _far():
        if contrast < 0.03 or a < 0.02 * primary_amp:
            return None
        return a * max(contrast, 0.08) * (0.15 + 0.85 * amp_ratio)

    if ratio <= k_near:
        return _near()
    if ratio >= k_far:
        return _far()
    # MID：兩式取高
    scores = [s for s in (_near(), _far()) if s is not None]
    return max(scores) if scores else None


def _stable_secondary_y_in_range(
    smooth,
    y0,
    y1,
    *,
    lobe_frac=0.40,
    extra_smooth=15,
):
    """在 [y0, y1) 取穩定第二峰 Y：加強平滑 → argmax → 瓣質心。

    弱下光斑／散斑常有多個相近局部峰，幀間雜訊會讓「最高局部峰」在
    相鄰尖峰間跳動；對下半區做較大平滑後取單峰＋強度質心可抑制跳動。
    """
    y0 = int(y0)
    y1 = int(y1)
    if y1 - y0 < 3:
        return None
    seg = np.asarray(smooth[y0:y1], dtype=np.float64)
    if seg.size < 3 or not np.isfinite(seg).any():
        return None
    peak0 = float(np.max(seg))
    if peak0 <= 0:
        return None
    win = int(extra_smooth) if extra_smooth else 15
    win = max(5, win if (win % 2) else win + 1)
    blur = _smooth_1d_profile(seg, win)
    j = int(np.argmax(blur))
    peak = float(blur[j])
    if peak <= 0:
        return float(y0 + j)
    thr = float(np.clip(lobe_frac, 0.15, 0.80)) * peak
    left = j
    while left > 0 and float(blur[left - 1]) >= thr:
        left -= 1
    right = j
    nseg = int(blur.size)
    while right < nseg - 1 and float(blur[right + 1]) >= thr:
        right += 1
    ww = np.asarray(blur[left : right + 1], dtype=np.float64)
    ww = np.where(ww >= thr, ww, 0.0)
    wsum = float(np.sum(ww))
    if wsum <= 1e-12:
        return float(y0 + j)
    xs = np.arange(left, right + 1, dtype=np.float64)
    return float(y0) + float(np.sum(xs * ww) / wsum)


def _secondary_y_after_valley(
    smooth,
    primary_y,
    exclude,
    *,
    primary_amp=None,
    min_amp_frac=0.004,
    extra_smooth=15,
):
    """主峰排除帶之後：先找波谷，再對波谷以下的下光斑瓣取穩定質心。

    不可直接對 exclude→末端做 argmax，否則會鎖在主峰拖尾（強度遠大於弱下光斑）。
    """
    smooth = np.asarray(smooth, dtype=np.float64)
    n = int(smooth.size)
    if n < 8:
        return None
    pi = int(np.clip(round(float(primary_y)), 0, n - 1))
    start = int(np.clip(pi + int(exclude) + 1, 0, n - 1))
    if n - start < 6:
        return None
    if primary_amp is None:
        primary_amp = float(smooth[pi])
    primary_amp = max(float(primary_amp), 1e-12)
    win = int(extra_smooth) if extra_smooth else 15
    win = max(5, win if (win % 2) else win + 1)
    blur = _smooth_1d_profile(smooth, win)

    # 自 exclude 起找「局部波谷 → 之後有足夠振幅的上升峰」
    valley = None
    for j in range(max(start, 1), n - 2):
        if not (float(blur[j]) <= float(blur[j - 1]) and float(blur[j]) <= float(blur[j + 1])):
            continue
        k = j + 1
        while k < n - 1 and float(blur[k + 1]) >= float(blur[k]):
            k += 1
        if float(blur[k]) < float(min_amp_frac) * primary_amp:
            continue
        if (k - j) < 3:
            continue
        valley = int(j)
        break
    if valley is None:
        # 後備：exclude 後前半段的最小點當波谷
        mid = min(n, start + max(40, (n - start) // 2))
        if mid - start < 3:
            return None
        valley = int(start + int(np.argmin(blur[start:mid])))

    return _stable_secondary_y_in_range(
        smooth,
        valley + 1,
        n,
        lobe_frac=0.40,
        extra_smooth=win,
    )


def _find_secondary_peak_away_from_primary(
    profile,
    primary_y,
    *,
    smooth_win=7,
    min_sep=12,
    min_prominence_frac=0.03,
    k_near=2.2,
    k_far=3.8,
    search_down_only=True,
):
    """在主峰排除帶外找第二峰。

    以高閾峰寬（0.65×峰頂）排除主峰核心後，依 sep/core_w 分 NEAR／FAR 評分：
    - NEAR：重 contrast（近距雙峰）
    - FAR：重絕對振幅比（遠距弱伴）

    search_down_only=True（預設）：只在「主峰下方」(Y 較大) 搜尋下光斑，
    避免大平滑時誤抓主峰上方肩峰。

    遠距弱下光斑（測試約 1/100 主峰）：建議以 SUM 波谷後瓣質心為準
    （`_secondary_y_after_valley`）；微分候選路徑保留不刪，供近距雙峰／對照。
    """
    profile = np.asarray(profile, dtype=np.float64)
    n = int(profile.size)
    if n < 5:
        return None
    py = float(primary_y)
    pi = int(np.clip(round(py), 0, n - 1))
    base_win = int(smooth_win) if smooth_win else 7
    smooth = _smooth_1d_profile(profile, base_win)
    core_w = max(6.0, _peak_width_at_frac_1d(smooth, pi, frac=0.65))
    exclude = int(np.clip(max(float(min_sep), 1.05 * core_w), 8, max(12, n // 5)))
    y_down_min = float(py + exclude) if search_down_only else None

    peaks = _peaks_by_derivative_1d(
        smooth, smooth_win=1, min_prominence_frac=min_prominence_frac
    )
    primary_amp = float(smooth[pi]) if smooth.size else 0.0
    candidates = []
    for y, a in peaks:
        y = float(y)
        a = float(a)
        if a <= 0:
            continue
        if search_down_only:
            if y < y_down_min:
                continue
        else:
            if abs(y - py) < exclude:
                continue
        sep = abs(y - py)
        lo = int(np.floor(min(y, py)))
        hi = int(np.ceil(max(y, py)))
        lo = int(np.clip(lo, 0, n - 1))
        hi = int(np.clip(hi, lo, n - 1))
        valley = (
            float(np.min(smooth[lo : hi + 1])) if hi > lo else 0.5 * (a + primary_amp)
        )
        contrast = (min(a, primary_amp) - valley) / max(min(a, primary_amp), 1e-12)
        score = _score_secondary_candidate(
            a,
            contrast,
            sep,
            core_w,
            primary_amp,
            k_near=k_near,
            k_far=k_far,
        )
        if score is None:
            continue
        candidates.append((float(score), y, a, sep, contrast))

    extra = max(15, base_win * 2 + 1)
    stable_down = None
    if search_down_only:
        stable_down = _secondary_y_after_valley(
            smooth,
            py,
            exclude,
            primary_amp=primary_amp,
            extra_smooth=extra,
        )

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        best_y = float(candidates[0][1])
        if stable_down is not None:
            if abs(best_y - float(stable_down)) <= max(18.0, 0.35 * core_w):
                return float(stable_down)
            if len(candidates) >= 2:
                s0, y0c = float(candidates[0][0]), float(candidates[0][1])
                s1, y1c = float(candidates[1][0]), float(candidates[1][1])
                if s0 > 0 and s1 / s0 >= 0.85 and abs(y0c - y1c) <= 24.0:
                    return float(stable_down)
        return best_y

    if stable_down is not None:
        return float(stable_down)

    # 後備：兩側／失敗時仍用波谷後瓣心或區間瓣心
    best_y = None
    best_score = -1.0
    if search_down_only:
        side_ranges = ((min(n, pi + exclude + 1), n),)
    else:
        side_ranges = (
            (0, max(0, pi - exclude)),
            (min(n, pi + exclude + 1), n),
        )
    for y0, y1 in side_ranges:
        if y1 - y0 < 3:
            continue
        if search_down_only:
            stable = _secondary_y_after_valley(
                smooth, py, exclude, primary_amp=primary_amp, extra_smooth=extra
            )
        else:
            stable = _stable_secondary_y_in_range(
                smooth, y0, y1, lobe_frac=0.40, extra_smooth=extra
            )
        if stable is None:
            continue
        yy = float(stable)
        if search_down_only and yy < y_down_min:
            continue
        sep = abs(yy - py)
        if sep < exclude:
            continue
        yi = int(np.clip(round(yy), 0, n - 1))
        la = float(smooth[yi])
        lo = int(np.clip(np.floor(min(yy, py)), 0, n - 1))
        hi = int(np.clip(np.ceil(max(yy, py)), lo, n - 1))
        valley = (
            float(np.min(smooth[lo : hi + 1]))
            if hi > lo
            else 0.5 * (la + primary_amp)
        )
        contrast = (min(la, primary_amp) - valley) / max(
            min(la, primary_amp), 1e-12
        )
        score = _score_secondary_candidate(
            la,
            contrast,
            sep,
            core_w,
            primary_amp,
            k_near=k_near,
            k_far=k_far,
        )
        if score is None:
            dist_term = 1.0 / (
                1.0 + max(0.0, sep - core_w) / max(1.5 * core_w, 1.0)
            )
            score = la * (0.30 + 0.70 * dist_term)
        if score > best_score:
            best_score = score
            best_y = yy
    return best_y


def _centroid_x_at_y_band(work, y_local, *, half_band=4):
    """在 ROI 子矩陣、峰 Y 附近列帶內做強度加權 X 質心（相對 work）。"""
    work = np.asarray(work, dtype=np.float64)
    if work.size == 0:
        return 0.0
    h, w = work.shape
    yi = int(round(float(y_local)))
    y0 = max(0, yi - int(half_band))
    y1 = min(h, yi + int(half_band) + 1)
    band = work[y0:y1, :]
    if band.size == 0 or not np.isfinite(band).any():
        return float((w - 1) * 0.5)
    bg = float(np.median(band))
    weights = np.clip(band - bg, 0.0, None)
    wsum = float(np.sum(weights))
    if wsum <= 1e-12:
        return float((w - 1) * 0.5)
    xs = np.arange(w, dtype=np.float64)
    return float(np.sum(weights * xs[None, :]) / wsum)


def _valley_between_peaks_1d(profile, y_a, y_b):
    """兩峰之間剖面最低點（含端點），作為上下半區切分。"""
    profile = np.asarray(profile, dtype=np.float64)
    lo = int(np.floor(min(float(y_a), float(y_b))))
    hi = int(np.ceil(max(float(y_a), float(y_b))))
    lo = int(np.clip(lo, 0, max(0, profile.size - 1)))
    hi = int(np.clip(hi, lo, max(0, profile.size - 1)))
    if hi <= lo:
        return float(lo)
    seg = profile[lo : hi + 1]
    return float(lo + int(np.argmin(seg)))


def _centroid_2d_in_y_range(
    work,
    y0,
    y1,
    *,
    thresh_frac=0.40,
    seed_y=None,
    seed_x=None,
    x0=None,
    x1=None,
    peak_sigma_y=None,
    peak_sigma_x=None,
):
    """在 work 的 [y0, y1) × [x0, x1) 半開區間做背景扣除＋門檻 2D 質心。

    若提供 seed_y，先以種子附近較亮連通區為主，避免弱光斑被外圍拖尾拉偏。
    peak_sigma_*：相對種子做高斯峰心加權，降低散射尾巴影響力。
    回傳相對 work 的 (cx, cy)；失敗則 None。
    """
    work = np.asarray(work, dtype=np.float64)
    if work.size == 0:
        return None
    h, w = work.shape
    y0 = int(np.clip(int(y0), 0, h))
    y1 = int(np.clip(int(y1), y0 + 1, h))
    xa = 0 if x0 is None else int(np.clip(int(x0), 0, w))
    xb = w if x1 is None else int(np.clip(int(x1), xa + 1, w))
    sub = work[y0:y1, xa:xb]
    if sub.size == 0 or not np.isfinite(sub).any():
        return None

    bg = float(np.median(sub))
    weights = np.clip(sub - bg, 0.0, None)
    peak = float(np.max(weights)) if weights.size else 0.0
    if peak <= 1e-12:
        return None

    frac = float(np.clip(thresh_frac, 0.05, 0.95))
    mask = weights >= (peak * frac)
    if not np.any(mask):
        mask = weights > 0
    if not np.any(mask):
        return None

    # 若有種子：只保留與種子列最近的連通區，減少散射／拖尾
    if seed_y is not None:
        seed_local = int(np.clip(round(float(seed_y) - y0), 0, sub.shape[0] - 1))
        labeled, num = ndi_label(mask)
        if num >= 1:
            best_lab = 0
            best_dist = 1e18
            for lab in range(1, num + 1):
                ys, _xs = np.where(labeled == lab)
                if ys.size == 0:
                    continue
                cy_lab = float(np.mean(ys))
                dist = abs(cy_lab - seed_local)
                covers = int(np.min(ys)) <= seed_local <= int(np.max(ys))
                score = dist - (50.0 if covers else 0.0)
                if score < best_dist:
                    best_dist = score
                    best_lab = lab
            if best_lab > 0:
                mask = labeled == best_lab

    # 峰心高斯加權（抑制拖尾）
    if seed_y is not None and (
        peak_sigma_y is not None or peak_sigma_x is not None
    ):
        sh, sw = weights.shape
        yy = np.arange(sh, dtype=np.float64)[:, None]
        xx = np.arange(sw, dtype=np.float64)[None, :]
        sy = float(seed_y) - float(y0)
        sx = (
            float(seed_x) - float(xa)
            if seed_x is not None
            else float((sw - 1) * 0.5)
        )
        gate = np.ones_like(weights)
        if peak_sigma_y is not None and float(peak_sigma_y) > 1e-6:
            sig_y = float(peak_sigma_y)
            gate = gate * np.exp(-((yy - sy) / sig_y) ** 2)
        if peak_sigma_x is not None and float(peak_sigma_x) > 1e-6:
            sig_x = float(peak_sigma_x)
            gate = gate * np.exp(-((xx - sx) / sig_x) ** 2)
        weights = weights * gate

    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    ww = weights[mask]
    wsum = float(np.sum(ww))
    if wsum <= 1e-12:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
    else:
        cx = float(np.sum(xs * ww) / wsum)
        cy = float(np.sum(ys * ww) / wsum)
    return cx + float(xa), cy + float(y0)


def _centroid_near_axis(
    work,
    seed_y,
    axis_x,
    *,
    y_half=32,
    x_half=36,
    thresh_frac=0.28,
    y0=None,
    y1=None,
    peak_sigma_y=None,
    peak_sigma_x=None,
):
    """在軸向窄窗（axis_x ± x_half, seed_y ± y_half）內做 2D 質心。

    可選 y0/y1 覆寫對稱 Y 窗（第二峰：靠主峰側裁到波谷、外側加長）。
    """
    work = np.asarray(work, dtype=np.float64)
    if work.size == 0:
        return None
    h, w = work.shape
    yi = int(round(float(seed_y)))
    xi = int(round(float(axis_x)))
    if y0 is None or y1 is None:
        ya = max(0, yi - int(y_half))
        yb = min(h, yi + int(y_half) + 1)
    else:
        ya = max(0, int(y0))
        yb = min(h, max(ya + 1, int(y1)))
    x0 = max(0, xi - int(x_half))
    x1 = min(w, xi + int(x_half) + 1)
    return _centroid_2d_in_y_range(
        work,
        ya,
        yb,
        thresh_frac=thresh_frac,
        seed_y=seed_y,
        seed_x=axis_x,
        x0=x0,
        x1=x1,
        peak_sigma_y=peak_sigma_y,
        peak_sigma_x=peak_sigma_x,
    )


def _soft_spring_x(cx, axis, x_half):
    """相對軸向軟彈簧：小偏移保留，大偏移漸進拉回（不再硬鎖 axis）。"""
    cx = float(cx)
    axis = float(axis)
    x_half = max(float(x_half), 1.0)
    dx = cx - axis
    adx = abs(dx)
    d_soft = 0.35 * x_half
    d_hard = 0.90 * x_half
    tau = 0.25 * x_half
    if adx <= d_soft:
        w = 1.0
    elif adx >= d_hard:
        w = 0.05
    else:
        w = float(np.exp(-(adx - d_soft) / max(tau, 1e-6)))
    return axis + dx * w


def _peak_x_in_y_band(work, y0, y1, *, smooth_col_win=5, core_frac=0.55):
    """在 [y0, y1) 列帶找亮核 X。

    扣背景後對「原始」≥ peak×core_frac 像素做強度質心。
    不做平滑門檻、不做最強連通區：下光斑右側尖峰／平滑後峰會把 X 拉右。
    """
    work = np.asarray(work, dtype=np.float64)
    if work.size == 0:
        return None
    h, w = work.shape
    y0 = int(np.clip(int(y0), 0, h))
    y1 = int(np.clip(int(y1), y0 + 1, h))
    band = work[y0:y1, :]
    if band.size == 0 or not np.isfinite(band).any():
        return None
    bg = float(np.median(band))
    weights = np.clip(band - bg, 0.0, None)
    peak = float(np.max(weights)) if weights.size else 0.0
    if peak <= 1e-12:
        return None
    frac = float(np.clip(core_frac, 0.20, 0.90))
    mask = weights >= (peak * frac)
    if not np.any(mask):
        return None
    xs = np.where(mask)[1].astype(np.float64)
    ww = weights[mask].astype(np.float64)
    wsum = float(np.sum(ww))
    if wsum <= 1e-12:
        return float(np.mean(xs))
    return float(np.sum(xs * ww) / wsum)


def _refine_primary_secondary_spots(
    work,
    profile,
    primary_y,
    secondary_y,
    *,
    axis_x=None,
    thresh_frac_primary=0.50,
    thresh_frac_secondary=0.55,
    outward_seed_bias=True,
):
    """主峰／第二峰精修。

    - 主峰：短曝光軸附近窄窗＋亮核質心（避免被散射拖到 ROI 邊緣）
    - 下光斑：只信自身 Y 帶亮核 X（高門檻，抑制右／左散射拖尾），不往藍線拉
    - outward_seed_bias：第一次精修時略把次峰種子往外推；迭代精修應關
    """
    h, w = np.asarray(work).shape
    try:
        py0 = float(primary_y)
        sy0 = float(secondary_y)
    except Exception:
        return (
            (float(axis_x) if axis_x is not None else 0.0, float(primary_y or 0.0)),
            (float(axis_x) if axis_x is not None else 0.0, float(secondary_y or 0.0)),
            0.0,
        )
    if not np.isfinite(py0) or not np.isfinite(sy0):
        ax = float(axis_x) if axis_x is not None and np.isfinite(float(axis_x)) else 0.0
        return (ax, py0 if np.isfinite(py0) else 0.0), (
            ax,
            sy0 if np.isfinite(sy0) else 0.0,
        ), float(py0 if np.isfinite(py0) else 0.0)

    smooth = _smooth_1d_profile(profile, 7)
    pi = int(np.clip(round(py0), 0, max(0, h - 1)))
    core_w = max(6.0, _peak_width_at_frac_1d(smooth, pi, frac=0.65))
    min_sep = max(8.0, 0.55 * core_w)

    if axis_x is not None and np.isfinite(float(axis_x)):
        axis = float(np.clip(float(axis_x), 0.0, max(0.0, w - 1.0)))
    else:
        axis = _centroid_x_at_y_band(
            work, py0, half_band=max(4, int(round(0.4 * core_w)))
        )
        axis = float(np.clip(axis, 0.0, max(0.0, w - 1.0)))

    # ---- 主峰：軸附近找亮核，再窄窗質心 ----
    p_half = max(8, int(round(0.70 * core_w)))
    p_y0 = max(0, pi - p_half)
    p_y1 = min(h, pi + p_half + 1)
    p_peak_x = _peak_x_in_y_band(work, p_y0, p_y1, smooth_col_win=5, core_frac=0.45)
    if p_peak_x is None:
        p_seed_x = float(axis)
    else:
        # 亮核與軸折衷，避免完全鎖藍線，也避免飛到遠側散射
        p_seed_x = 0.55 * float(p_peak_x) + 0.45 * float(axis)
    p_x_half = max(10, min(22, int(round(0.55 * core_w))))
    p_x0 = max(0, int(round(p_seed_x)) - p_x_half)
    p_x1 = min(w, int(round(p_seed_x)) + p_x_half + 1)
    c_primary = _centroid_2d_in_y_range(
        work,
        p_y0,
        p_y1,
        thresh_frac=thresh_frac_primary,
        seed_y=py0,
        seed_x=p_seed_x,
        x0=p_x0,
        x1=p_x1,
        peak_sigma_y=max(2.5, 0.35 * float(p_half)),
        peak_sigma_x=max(3.0, 0.40 * float(p_x_half)),
    )
    if c_primary is None:
        c_primary = (float(p_seed_x), float(py0))
    else:
        # 大偏移時柔拉回種子，防止右／左拖尾；Y 也限縮，避免上光斑內多核幀間跳動
        cx_p = _soft_spring_x(
            float(c_primary[0]), float(p_seed_x), max(8.0, 0.7 * p_x_half)
        )
        cy_p = _soft_spring_x(
            float(c_primary[1]), float(py0), max(4.0, 0.20 * float(p_half))
        )
        c_primary = (float(cx_p), float(cy_p))

    # ---- 下光斑：波谷外自身亮核，不混軸向 ----
    sep = abs(float(sy0) - float(py0))
    y_half = int(np.clip(0.40 * sep if sep > 1 else 12, 10, 32))
    sign = 1.0 if float(sy0) >= float(py0) else -1.0
    if outward_seed_bias:
        sy = float(sy0) + sign * float(min(8.0, max(3.0, 0.10 * sep)))
    else:
        sy = float(sy0)
    if not np.isfinite(sy):
        sy = float(sy0)
    sy = float(np.clip(sy, 0.0, max(0.0, h - 1.0)))
    if not np.isfinite(sy):
        sy = float(np.clip(sy0 if np.isfinite(sy0) else 0.0, 0.0, max(0.0, h - 1.0)))

    valley = _valley_between_peaks_1d(smooth, py0, sy0)
    if not np.isfinite(float(valley)):
        valley = 0.5 * (float(py0) + float(sy0))
    y_toward = int(np.clip(0.45 * y_half, 6, y_half))
    y_away = int(np.clip(1.35 * y_half, y_half, 44))
    sy_i = int(round(float(sy)))
    if sign >= 0.0:
        y0 = max(int(np.floor(valley)) + 1, sy_i - y_toward)
        y1 = min(h, sy_i + y_away + 1)
    else:
        y0 = max(0, sy_i - y_away)
        y1 = min(int(np.ceil(valley)), sy_i + y_toward + 1)
    if y1 <= y0 + 2:
        y0 = max(0, sy_i - y_half)
        y1 = min(h, sy_i + y_half + 1)

    # 下光斑 X 種子：原始亮核質心（core_frac≈0.55 在散斑上最穩，約對綠心）
    x_seed = _peak_x_in_y_band(work, y0, y1, smooth_col_win=5, core_frac=0.55)
    if x_seed is None:
        x_seed = _centroid_x_at_y_band(
            work, sy, half_band=max(4, int(round(0.35 * (y1 - y0))))
        )
    x_seed = float(np.clip(float(x_seed), 0.0, max(0.0, w - 1.0)))

    sep_ratio = sep / max(core_w, 1.0)
    f_sec = float(thresh_frac_secondary)
    # 遠距弱斑提高門檻，只吃亮心；近距略降
    if sep_ratio >= 3.5:
        f_sec = max(f_sec, 0.58)
    elif sep_ratio <= 2.2:
        f_sec = min(f_sec, 0.45)
    else:
        f_sec = max(f_sec, 0.52)

    # 窄窗繞亮核；避免右側藍暈進質心
    sec_x_half = max(8, min(14, int(round(0.07 * w + 4))))
    sig_y = max(3.0, 0.35 * float(y_half))
    sig_x = max(2.5, 0.35 * float(sec_x_half))
    c_secondary = _centroid_near_axis(
        work,
        sy,
        x_seed,
        y_half=y_half,
        x_half=sec_x_half,
        thresh_frac=f_sec,
        y0=y0,
        y1=y1,
        peak_sigma_y=sig_y,
        peak_sigma_x=sig_x,
    )
    if c_secondary is None:
        c_secondary = (float(x_seed), float(sy))

    # 下光斑 X：以亮核種子為準（散斑質心易被右側拖尾拉偏）
    cx_sec = float(x_seed)
    cy_2d = float(c_secondary[1])
    alpha = float(
        np.clip(0.35 + 0.40 * np.tanh((sep_ratio - 1.5) / 2.0), 0.30, 0.80)
    )
    cy_sec = alpha * float(sy) + (1.0 - alpha) * cy_2d
    c_secondary = (float(cx_sec), float(cy_sec))

    if abs(float(c_secondary[1]) - float(c_primary[1])) < min_sep:
        c_secondary = (float(x_seed), float(sy))

    return c_primary, c_secondary, float(valley)


def refine_hsum_dual_peaks(
    matrix,
    center_xy,
    primary_y_full,
    secondary_y_full,
    *,
    roi_w=160,
    roi_h=512,
    n_refine=1,
    profile=None,
):
    """對水平加總主／次峰做 N 次 2D 亮核精修（迭代回饋）。

    Args:
        matrix: 長曝光影像
        center_xy: 條帶中心（通常為短曝光錨點）
        primary_y_full / secondary_y_full: 全圖 Y（1D 峰種子）
        n_refine: 精修次數；<=0 時不做精修，原樣回傳種子
        profile: 可選，ROI 內扣背景水平加總 1D；None 則現場重算

    Returns:
        dict:
            primary_xy, secondary_xy — 全圖 (x, y)
            primary_y, secondary_y — 全圖 Y
            n_done — 實際執行次數
            valley_y — 最後一次波谷（全圖 Y）或 None
        失敗回傳 None
    """
    if primary_y_full is None or secondary_y_full is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0 or center_xy is None:
        return None
    n = int(n_refine)
    p_y = float(primary_y_full)
    s_y = float(secondary_y_full)
    if n <= 0:
        return {
            "primary_xy": (float(center_xy[0]), p_y),
            "secondary_xy": (float(center_xy[0]), s_y),
            "primary_y": p_y,
            "secondary_y": s_y,
            "n_done": 0,
            "valley_y": None,
        }

    xywh = _roi_centered_on_xy(matrix, center_xy, roi_w=roi_w, roi_h=roi_h)
    if xywh is None:
        return None
    bounds = clip_roi_to_matrix(matrix, xywh)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    work = matrix[y0:y1, x0:x1]
    if work.size == 0:
        return None

    if profile is None:
        raw_sum = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
        bg = float(np.median(raw_sum)) if raw_sum.size else 0.0
        prof = np.clip(raw_sum - bg, 0.0, None)
    else:
        prof = np.asarray(profile, dtype=np.float64)
        if prof.size != work.shape[0]:
            raw_sum = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
            bg = float(np.median(raw_sum)) if raw_sum.size else 0.0
            prof = np.clip(raw_sum - bg, 0.0, None)

    py = float(np.clip(p_y - float(y0), 0.0, max(0.0, work.shape[0] - 1.0)))
    sy = float(np.clip(s_y - float(y0), 0.0, max(0.0, work.shape[0] - 1.0)))
    axis_x = float(np.clip(float(center_xy[0]) - float(x0), 0.0, max(0.0, work.shape[1] - 1.0)))
    valley_local = None
    done = 0
    c_pri = (axis_x, py)
    c_sec = (axis_x, sy)
    for i in range(n):
        c_pri, c_sec, valley_local = _refine_primary_secondary_spots(
            work,
            prof,
            py,
            sy,
            axis_x=axis_x,
            # 僅第一次外推次峰種子；後續用收斂座標避免越跑越遠
            outward_seed_bias=(i == 0),
        )
        py = float(c_pri[1])
        sy = float(c_sec[1])
        axis_x = float(c_pri[0])
        done += 1

    pri_xy = (float(x0) + float(c_pri[0]), float(y0) + float(c_pri[1]))
    sec_xy = (float(x0) + float(c_sec[0]), float(y0) + float(c_sec[1]))
    valley_full = (
        None if valley_local is None else float(y0) + float(valley_local)
    )
    return {
        "primary_xy": pri_xy,
        "secondary_xy": sec_xy,
        "primary_y": float(pri_xy[1]),
        "secondary_y": float(sec_xy[1]),
        "n_done": int(done),
        "valley_y": valley_full,
    }


def _subpixel_argmax_1d(ys, vals):
    """回傳 vals 最大值的亚像素 Y（三点抛物线；鄰居非有限則回整點）。"""
    ys = np.asarray(ys, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if ys.size == 0 or vals.size != ys.size:
        return None
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None
    work = np.where(finite, vals, -np.inf)
    j = int(np.argmax(work))
    if not np.isfinite(vals[j]):
        return None
    y_j = float(ys[j])
    if 0 < j < len(vals) - 1:
        y0v, y1v, y2v = float(vals[j - 1]), float(vals[j]), float(vals[j + 1])
        if (
            np.isfinite(y0v)
            and np.isfinite(y1v)
            and np.isfinite(y2v)
            and y1v >= y0v
            and y1v >= y2v
        ):
            denom = y0v - 2.0 * y1v + y2v
            if abs(denom) > 1e-12 and np.isfinite(denom):
                delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
                if np.isfinite(delta):
                    return y_j + delta
    return y_j


def peak_y_max_in_window(
    ys,
    vals,
    lo=None,
    hi=None,
    *,
    exclude_y=None,
    min_sep=1.0,
    subpixel=True,
):
    """在 [lo, hi] 內找 vals 強度最大的 Y。

    subpixel=True：三點拋物線亚像素；False：只回整點樣本 Y。
    """
    ys = np.asarray(ys, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if ys.size == 0 or vals.size != ys.size:
        return None
    mask = np.isfinite(vals) & np.isfinite(ys)
    if lo is not None:
        mask &= ys >= float(lo)
    if hi is not None:
        mask &= ys <= float(hi)
    if exclude_y is not None:
        mask &= np.abs(ys - float(exclude_y)) >= float(max(1.0, min_sep))
    if not np.any(mask):
        return None
    idx = np.flatnonzero(mask)
    j_local = int(np.argmax(vals[idx]))
    j = int(idx[j_local])
    y_j = float(ys[j])
    if subpixel and 0 < j < len(vals) - 1 and bool(mask[j - 1]) and bool(mask[j + 1]):
        y0v, y1v, y2v = float(vals[j - 1]), float(vals[j]), float(vals[j + 1])
        denom = y0v - 2.0 * y1v + y2v
        if (
            np.isfinite(y0v)
            and np.isfinite(y1v)
            and np.isfinite(y2v)
            and abs(denom) > 1e-12
            and y1v >= y0v
            and y1v >= y2v
        ):
            # 網格可能非等距：用 ys 實際間距做拋物線頂點
            x0, x1, x2 = float(ys[j - 1]), float(ys[j]), float(ys[j + 1])
            # 等距時等同經典公式；非等距用二次頂點
            try:
                # fit y = a x^2 + b x + c through three points; peak at -b/2a
                A = np.array(
                    [
                        [x0 * x0, x0, 1.0],
                        [x1 * x1, x1, 1.0],
                        [x2 * x2, x2, 1.0],
                    ],
                    dtype=float,
                )
                coeff = np.linalg.solve(A, np.array([y0v, y1v, y2v], dtype=float))
                a, b = float(coeff[0]), float(coeff[1])
                if abs(a) > 1e-18 and a < 0:
                    y_peak = -b / (2.0 * a)
                    if min(x0, x2) <= y_peak <= max(x0, x2):
                        y_j = float(y_peak)
                    else:
                        delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
                        if np.isfinite(delta):
                            y_j = float(ys[j]) + delta * (x2 - x0) * 0.5
                else:
                    delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
                    if np.isfinite(delta):
                        y_j = float(ys[j]) + delta
            except Exception:
                delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
                if np.isfinite(delta):
                    y_j = float(ys[j]) + delta
    if lo is not None and hi is not None:
        y_j = float(np.clip(y_j, float(min(lo, hi)), float(max(lo, hi))))
    return float(y_j)

def _lagrange_quadratic(x, x0, x1, x2, y0, y1, y2):
    """三點 Lagrange 二次内插。"""
    x = float(x)
    x0, x1, x2 = float(x0), float(x1), float(x2)
    y0, y1, y2 = float(y0), float(y1), float(y2)
    d0 = (x0 - x1) * (x0 - x2)
    d1 = (x1 - x0) * (x1 - x2)
    d2 = (x2 - x0) * (x2 - x1)
    if abs(d0) < 1e-18 or abs(d1) < 1e-18 or abs(d2) < 1e-18:
        # 退化成線性
        if abs(x2 - x0) < 1e-18:
            return y0
        t = (x - x0) / (x2 - x0)
        return y0 * (1.0 - t) + y2 * t
    l0 = (x - x1) * (x - x2) / d0
    l1 = (x - x0) * (x - x2) / d1
    l2 = (x - x0) * (x - x1) / d2
    return y0 * l0 + y1 * l1 + y2 * l2


def refine_profile_together(ys, vals, n_refine, *, base_win=7):
    """在 after smooth 上以分段二次内插加密剖面（供顯示 after refine）。

    峰位精修另以峰頂三點二次處理；此函數負責畫出精修後波形。
    N＝相鄰樣本之間的等分段數。N<=0：回傳原始副本。
    base_win 保留相容、不使用。

    Returns:
        dict: y_px, sum_refine（長度可能大於輸入）
    """
    ys = np.asarray(ys, dtype=float)
    vals = np.asarray(vals, dtype=float)
    empty = {
        "y_px": ys.copy() if ys.size else np.asarray([], dtype=float),
        "sum_refine": vals.copy() if vals.size else np.asarray([], dtype=float),
    }
    if ys.size == 0 or vals.size != ys.size:
        return empty
    n = max(0, int(n_refine))
    if n <= 0:
        return {
            "y_px": ys.copy(),
            "sum_refine": vals.copy(),
        }
    finite = np.isfinite(ys) & np.isfinite(vals)
    if int(np.count_nonzero(finite)) < 2:
        return {
            "y_px": ys.copy(),
            "sum_refine": vals.copy(),
        }
    ys_u = ys[finite]
    vals_u = vals[finite]
    order = np.argsort(ys_u)
    ys_u = ys_u[order]
    vals_u = vals_u[order]
    if ys_u.size >= 2:
        uniq = np.concatenate([[True], np.diff(ys_u) > 1e-12])
        ys_u = ys_u[uniq]
        vals_u = vals_u[uniq]
    m = int(ys_u.size)
    if m < 2:
        return {
            "y_px": ys.copy(),
            "sum_refine": vals.copy(),
        }
    factor = max(1, n)
    n_seg = m - 1
    ys_f = np.empty(n_seg * factor + 1, dtype=float)
    vals_f = np.empty_like(ys_f)
    for i in range(n_seg):
        if m >= 3:
            if i + 2 < m:
                i0, i1, i2 = i, i + 1, i + 2
            else:
                i0, i1, i2 = i - 1, i, i + 1
        else:
            i0, i1, i2 = 0, 0, 1
        x0, x1, x2 = float(ys_u[i0]), float(ys_u[i1]), float(ys_u[i2])
        y0, y1, y2 = float(vals_u[i0]), float(vals_u[i1]), float(vals_u[i2])
        xa, xb = float(ys_u[i]), float(ys_u[i + 1])
        va, vb = float(vals_u[i]), float(vals_u[i + 1])
        for k in range(factor):
            t = float(k) / float(factor)
            x = xa * (1.0 - t) + xb * t
            idx = i * factor + k
            ys_f[idx] = x
            if m >= 3 and abs(x2 - x0) > 1e-12:
                vals_f[idx] = _lagrange_quadratic(x, x0, x1, x2, y0, y1, y2)
            else:
                vals_f[idx] = va * (1.0 - t) + vb * t
    ys_f[-1] = float(ys_u[-1])
    vals_f[-1] = float(vals_u[-1])
    return {
        "y_px": ys_f,
        "sum_refine": np.asarray(vals_f, dtype=float),
    }


def refine_dual_peaks_quadratic(
    ys,
    vals,
    *,
    secondary_lo=None,
    secondary_hi=None,
    min_sep=12.0,
    subpixel=True,
):
    """兩峰峰頂精修：在 vals 上抓最大；預設三點二次亚像素。

    Returns:
        (primary_y, secondary_y)；失敗的峰為 None
    """
    p_y = peak_y_max_in_window(
        ys, vals, min_sep=min_sep, subpixel=subpixel
    )
    s_y = None
    if p_y is not None:
        s_y = peak_y_max_in_window(
            ys,
            vals,
            secondary_lo,
            secondary_hi,
            exclude_y=p_y,
            min_sep=min_sep,
            subpixel=subpixel,
        )
    return p_y, s_y

def refine_peak_y_on_profile(
    ys,
    vals,
    y_seed,
    *,
    half_win=6.0,
    n_iter=1,
    clamp=None,
):
    """在 1D 波形上對峰位做加權質心精修（迭代 N 次）。

    每次在 [y±half_win] 內用強度加權質心更新 Y；可選 clamp 到搜尋窗。
    用於水平加總：調整精修次數時 ΔY 小數會跟著變，且仍貼近原峰。
    """
    ys = np.asarray(ys, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if ys.size == 0 or vals.size != ys.size or y_seed is None:
        return y_seed
    try:
        y = float(y_seed)
    except Exception:
        return y_seed
    if not np.isfinite(y):
        return y_seed
    n = max(0, int(n_iter))
    if n <= 0:
        return y
    hw0 = float(max(2.0, half_win))
    # 視窗隨總次數放大，使 N=0／1／2／… 的質心與 ΔY 小數可區分
    for i in range(n):
        hw = hw0 + 1.25 * float(max(0, n - 1)) + 0.5 * float(i)
        lo = y - hw
        hi = y + hw
        if clamp is not None and len(clamp) >= 2:
            c0, c1 = float(min(clamp[0], clamp[1])), float(max(clamp[0], clamp[1]))
            lo = max(lo, c0)
            hi = min(hi, c1)
        mask = (ys >= lo) & (ys <= hi) & np.isfinite(vals)
        if not np.any(mask):
            break
        w = np.clip(vals[mask], 0.0, None)
        wsum = float(np.sum(w))
        if wsum <= 1e-12:
            # 允許非正值時改用絕對值權重
            w = np.abs(vals[mask])
            wsum = float(np.sum(w))
            if wsum <= 1e-12:
                break
        y_new = float(np.sum(ys[mask] * w) / wsum)
        if not np.isfinite(y_new):
            break
        y = y_new
    if clamp is not None and len(clamp) >= 2:
        c0, c1 = float(min(clamp[0], clamp[1])), float(max(clamp[0], clamp[1]))
        y = float(np.clip(y, c0, c1))
    return float(y)


def find_dual_spots_horizontal_sum(
    matrix,
    center_xy,
    *,
    roi_w=160,
    roi_h=512,
    smooth_win=7,
    min_peak_distance=12,
    n_refine=1,
    secondary_window_y=None,
):
    """水平加總雙峰演算法（波形找峰＋整段精修曲線＋雙峰共用質心）。

    流程：
      1. 以 center 為中心框 ROI → 水平加總剖面
      2. 主峰 = 平滑前剖面最高（亚像素）
      3. 次峰 = 搜尋窗內平滑後最高（未給窗則取主峰外側半段）
      4. n_refine=0：兩峰用整點最高（間距較「固定」）
         n_refine>0：分段二次加密曲線後，峰位＝該曲線強度最大點（對齊圖上峰頂）

    Heatmap／原圖不被平滑參數改寫；平滑／精修僅作用於剖面曲線。

    Returns:
        dict: spots, peak_ys, primary_index, roi_xywh, profile 摘要, secondary_window_y
        失敗時 spots=None
    """
    empty = {
        "roi_xywh": None,
        "roi_bounds": None,
        "profile": None,
        "peak_ys": None,
        "spots": None,
        "primary_index": 0,
        "secondary_window_y": None,
        "refine_done": 0,
    }
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0 or center_xy is None:
        return empty

    prof = compute_horizontal_sum_profile(
        matrix,
        center_xy,
        roi_w=roi_w,
        roi_h=roi_h,
        smooth_win=smooth_win,
        min_peak_distance=min_peak_distance,
        find_secondary=False,
    )
    if not prof:
        return empty

    ys = np.asarray(prof.get("y_full_px"), dtype=float)
    before = np.asarray(prof.get("sum_minus_median"), dtype=float)
    after = np.asarray(prof.get("sum_smooth"), dtype=float)
    if ys.size == 0 or before.size != ys.size:
        return {
            **empty,
            "roi_xywh": prof.get("roi_xywh"),
            "profile": before,
        }

    primary_y = _subpixel_argmax_1d(ys, before)
    if primary_y is None:
        return {
            **empty,
            "roi_xywh": prof.get("roi_xywh"),
            "profile": before,
        }

    y0 = float(np.nanmin(ys))
    y1 = float(np.nanmax(ys))
    min_sep = max(1.0, float(min_peak_distance))
    if secondary_window_y is not None and len(secondary_window_y) >= 2:
        lo = float(min(secondary_window_y[0], secondary_window_y[1]))
        hi = float(max(secondary_window_y[0], secondary_window_y[1]))
    else:
        # 預設：主峰下方（或上方）外側搜尋
        if float(primary_y) < 0.5 * (y0 + y1):
            lo, hi = float(primary_y) + min_sep, y1
        else:
            lo, hi = y0, float(primary_y) - min_sep
        if hi <= lo:
            lo, hi = y0, y1

    vals_sec = after if after.size == ys.size else before
    mask = (ys >= lo) & (ys <= hi) & np.isfinite(vals_sec)
    mask &= np.abs(ys - float(primary_y)) >= min_sep
    secondary_y = None
    if np.any(mask):
        # 只在窗內有效點上找最高，避免 -inf 鄰居造成亚像素 NaN
        idx = np.flatnonzero(mask)
        j = int(idx[int(np.argmax(vals_sec[idx]))])
        secondary_y = float(ys[j])
        if 0 < j < len(vals_sec) - 1 and bool(mask[j - 1]) and bool(mask[j + 1]):
            y0v = float(vals_sec[j - 1])
            y1v = float(vals_sec[j])
            y2v = float(vals_sec[j + 1])
            denom = y0v - 2.0 * y1v + y2v
            if (
                np.isfinite(y0v)
                and np.isfinite(y1v)
                and np.isfinite(y2v)
                and abs(denom) > 1e-12
                and y1v >= y0v
                and y1v >= y2v
            ):
                delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
                if np.isfinite(delta):
                    secondary_y = float(ys[j]) + delta
        secondary_y = float(np.clip(secondary_y, lo, hi))

    if secondary_y is None or not np.isfinite(float(secondary_y)):
        return {
            **empty,
            "roi_xywh": prof.get("roi_xywh"),
            "roi_bounds": None,
            "profile": before,
            "secondary_window_y": [lo, hi],
        }
    if primary_y is None or not np.isfinite(float(primary_y)):
        return {
            **empty,
            "roi_xywh": prof.get("roi_xywh"),
            "profile": before,
            "secondary_window_y": [lo, hi],
        }

    axis_x = float(center_xy[0])
    p_y = float(primary_y)
    s_y = float(secondary_y)
    refine_done = 0
    n_ref = max(0, int(n_refine))
    # 精修：N=0 整點峰；N>0 在二次加密曲線上抓峰（N 影響 ΔY）
    refine_src = after if after.size == ys.size else before
    refine_pack = refine_profile_together(
        ys, refine_src, n_ref, base_win=int(smooth_win) if smooth_win else 7
    )
    ys_r = np.asarray(refine_pack.get("y_px"), dtype=float)
    refined = np.asarray(refine_pack.get("sum_refine"), dtype=float)
    if n_ref <= 0:
        p_new = peak_y_max_in_window(
            ys, before if before.size == ys.size else refine_src,
            min_sep=min_sep, subpixel=False,
        )
        if p_new is not None and np.isfinite(p_new):
            p_y = float(p_new)
        s_new = peak_y_max_in_window(
            ys,
            refine_src,
            lo,
            hi,
            exclude_y=p_y,
            min_sep=min_sep,
            subpixel=False,
        )
        if s_new is not None and np.isfinite(s_new):
            s_y = float(s_new)
        refine_done = 0
    else:
        # 在加密曲線上直接取強度最大（不再二次偏移，避免虛線偏離峰頂）
        if ys_r.size and refined.size == ys_r.size:
            p_new, s_new = refine_dual_peaks_quadratic(
                ys_r,
                refined,
                secondary_lo=lo,
                secondary_hi=hi,
                min_sep=min_sep,
                subpixel=False,
            )
        else:
            p_new, s_new = refine_dual_peaks_quadratic(
                ys,
                refine_src,
                secondary_lo=lo,
                secondary_hi=hi,
                min_sep=min_sep,
                subpixel=True,
            )
        if p_new is not None and np.isfinite(p_new):
            p_y = float(p_new)
        if s_new is not None and np.isfinite(s_new):
            s_y = float(s_new)
        refine_done = n_ref
        if not np.isfinite(p_y):
            p_y = float(primary_y)
        if not np.isfinite(s_y):
            s_y = float(secondary_y)

    p_xy = (axis_x, float(p_y))
    s_xy = (axis_x, float(s_y))

    if p_xy[1] <= s_xy[1]:
        spots = [p_xy, s_xy]
        primary_index = 0
        peak_ys = (p_xy[1], s_xy[1])
    else:
        spots = [s_xy, p_xy]
        primary_index = 1
        peak_ys = (s_xy[1], p_xy[1])

    return {
        "roi_xywh": prof.get("roi_xywh"),
        "roi_bounds": None,
        "profile": before,
        "profile_smooth": after,
        "profile_refine": refined,
        "y_refine_px": ys_r,
        "y_full_px": ys,
        "peak_ys": peak_ys,
        "spots": spots,
        "primary_index": primary_index,
        "primary_y": float(p_xy[1]),
        "secondary_y": float(s_xy[1]),
        "secondary_window_y": [float(lo), float(hi)],
        "refine_done": refine_done,
        "smooth_win": int(smooth_win),
    }


def find_dual_spots_axial_1d_projection(
    matrix,
    center_xy,
    roi_w=160,
    roi_h=512,
    smooth_win=7,
    min_peak_distance=12,
):
    """軸向一維投影剖面與微分法：框選長條 ROI → 水平加總 → 微分找雙峰。

    主峰錨定短曝光對應位置；第二峰為排除主峰本體後、對比最佳的近距／遠距伴峰，
    再做軸向窄窗 2D 質心。兼顧「遠距弱第二峰」與「近距雙峰」兩種樣態。

    Returns:
        dict:
            roi_xywh: 實際使用的 (x, y, w, h)
            roi_bounds: 半開 (x0, y0, x1, y1)
            profile: 水平加總後的 1D 剖面（全列，ROI 內）
            peak_ys: (y_lo, y_hi) 全圖座標（精修後）
            spots: [(cx, cy), (cx, cy)] 依 Y 由小到大
            primary_index: 較接近 center_xy 的峰索引（0 或 1）
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    empty = {
        "roi_xywh": None,
        "roi_bounds": None,
        "profile": np.array([]),
        "peak_ys": None,
        "spots": None,
        "primary_index": 0,
    }
    if matrix.size == 0 or center_xy is None:
        return empty

    xywh = _roi_centered_on_xy(matrix, center_xy, roi_w=roi_w, roi_h=roi_h)
    if xywh is None:
        return empty
    x, y, rw, rh = xywh
    bounds = clip_roi_to_matrix(matrix, xywh)
    if bounds is None:
        return empty
    x0, y0, x1, y1 = bounds
    work = matrix[y0:y1, x0:x1]
    if work.size == 0:
        return empty

    # 水平加總 → 沿垂直軸的 1D 強度分佈（扣列中值背景）
    raw_sum = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
    bg_row = float(np.median(raw_sum)) if raw_sum.size else 0.0
    profile = np.clip(raw_sum - bg_row, 0.0, None)

    peaks = _peaks_by_derivative_1d(
        profile, smooth_win=smooth_win, min_prominence_frac=0.04
    )
    primary_hint = float(center_xy[1]) - float(y0)
    primary_y = _primary_peak_from_hint(peaks, profile, primary_hint)
    if primary_y is None:
        return {
            **empty,
            "roi_xywh": (x, y, rw, rh),
            "roi_bounds": bounds,
            "profile": profile,
        }

    min_sep = max(10, int(min_peak_distance))
    secondary_y = _find_secondary_peak_away_from_primary(
        profile,
        primary_y,
        smooth_win=smooth_win,
        min_sep=min_sep,
        min_prominence_frac=0.02,
    )
    if secondary_y is None or abs(float(secondary_y) - float(primary_y)) < 2.0:
        return {
            **empty,
            "roi_xywh": (x, y, rw, rh),
            "roi_bounds": bounds,
            "profile": profile,
        }

    axis_x_local = float(center_xy[0]) - float(x0)
    c_pri, c_sec, _valley = _refine_primary_secondary_spots(
        work,
        profile,
        primary_y,
        secondary_y,
        axis_x=axis_x_local,
    )

    # 組成依 Y 排序的兩點，並標 primary_index
    p_xy = (float(x0) + float(c_pri[0]), float(y0) + float(c_pri[1]))
    s_xy = (float(x0) + float(c_sec[0]), float(y0) + float(c_sec[1]))
    if p_xy[1] <= s_xy[1]:
        spots = [p_xy, s_xy]
        primary_index = 0
        peak_ys = (p_xy[1], s_xy[1])
    else:
        spots = [s_xy, p_xy]
        primary_index = 1
        peak_ys = (s_xy[1], p_xy[1])

    return {
        "roi_xywh": (x, y, int(x1 - x0), int(y1 - y0)),
        "roi_bounds": bounds,
        "profile": profile,
        "peak_ys": peak_ys,
        "spots": spots,
        "primary_index": primary_index,
    }


def compute_horizontal_sum_profile(
    matrix,
    center_xy,
    *,
    roi_w=160,
    roi_h=512,
    smooth_win=7,
    min_peak_distance=12,
    find_secondary=True,
):
    """計算一維投影用的水平加總剖面（供 UI 檢視）。

    Returns:
        dict:
            y_full_px, sum_raw, sum_minus_median, sum_smooth,
            primary_y_full, secondary_y_full, roi_xywh, center_xy
            失敗時回傳 None
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0 or center_xy is None:
        return None
    xywh = _roi_centered_on_xy(matrix, center_xy, roi_w=roi_w, roi_h=roi_h)
    if xywh is None:
        return None
    bounds = clip_roi_to_matrix(matrix, xywh)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    work = matrix[y0:y1, x0:x1]
    if work.size == 0:
        return None

    raw_sum = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
    bg_row = float(np.median(raw_sum)) if raw_sum.size else 0.0
    profile = np.clip(raw_sum - bg_row, 0.0, None)
    smooth = _smooth_1d_profile(profile, smooth_win)
    y_full = np.arange(len(profile), dtype=np.float64) + float(y0)

    peaks = _peaks_by_derivative_1d(
        profile, smooth_win=smooth_win, min_prominence_frac=0.04
    )
    primary_hint = float(center_xy[1]) - float(y0)
    primary_y = _primary_peak_from_hint(peaks, profile, primary_hint)
    secondary_y = None
    if find_secondary and primary_y is not None:
        min_sep = max(10, int(min_peak_distance))
        secondary_y = _find_secondary_peak_away_from_primary(
            profile,
            primary_y,
            smooth_win=smooth_win,
            min_sep=min_sep,
            min_prominence_frac=0.02,
        )

    return {
        "y_full_px": y_full,
        "sum_raw": np.asarray(raw_sum, dtype=np.float64),
        "sum_minus_median": np.asarray(profile, dtype=np.float64),
        "sum_smooth": np.asarray(smooth, dtype=np.float64),
        "primary_y_full": (
            None if primary_y is None else float(primary_y) + float(y0)
        ),
        "secondary_y_full": (
            None if secondary_y is None else float(secondary_y) + float(y0)
        ),
        "roi_xywh": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        "center_xy": (float(center_xy[0]), float(center_xy[1])),
        "smooth_win": int(smooth_win),
    }


def _centroid_largest_cc_weighted(weights, mask):
    """最大連通區的加權質心；失敗回傳 None。"""
    labeled, num = ndi_label(mask)
    if num <= 0:
        return None
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    lid = int(np.argmax(counts))
    sel = labeled == lid
    ww = weights[sel]
    ys, xs = np.where(sel)
    wsum = float(np.sum(ww))
    if wsum <= 1e-12 or ys.size == 0:
        return None
    cx = float(np.sum(xs.astype(np.float64) * ww) / wsum)
    cy = float(np.sum(ys.astype(np.float64) * ww) / wsum)
    return cx, cy, int(ys.size)


def find_dual_spots_gaussian(
    matrix,
    center_xy,
    *,
    sigma=4.0,
    thresh_frac=0.35,
    search_half_x=80,
    min_sep_y=120,
    primary_blank_r=90.0,
):
    """高斯平滑後以 2D 連通區抓下光斑（不接一維投影）。

    流程:
      1. ``gaussian_filter(σ)``
      2. 上光斑 = 短曝光錨點 ``center_xy``（不再於長曝光重抓）
      3. 抹掉上光斑鄰域圓（``primary_blank_r``），自 ``min_sep_y`` 以下
         軸向帶內相對峰門檻＋最大連通區質心 → 下光斑

    Returns:
        dict:
            primary / secondary: (cx, cy) 或 None
            spots: 依 Y 排序的兩點，或找不到時 None
            primary_index: spots 內上光斑索引
            search_roi_xywh: 下光斑搜尋帶 (x, y, w, h) 供 UI 框線
            amp_ratio: 上／下峰強度比（參考）
            reason: 狀態字串
    """
    empty = {
        "primary": None,
        "secondary": None,
        "spots": None,
        "primary_index": 0,
        "search_roi_xywh": None,
        "amp_ratio": None,
        "reason": "empty",
    }
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0 or center_xy is None:
        return empty

    filtered = ndi_gaussian_filter(
        matrix, sigma=float(sigma), mode="nearest"
    )
    h, w = filtered.shape
    sx, sy = float(center_xy[0]), float(center_xy[1])
    primary = (sx, sy)

    # 參考振幅：短曝光點附近小窗峰值（僅供 amp_ratio）
    ix = int(np.clip(round(sx), 0, w - 1))
    iy = int(np.clip(round(sy), 0, h - 1))
    r_amp = 8
    xa0 = max(0, ix - r_amp)
    xa1 = min(w, ix + r_amp + 1)
    ya0 = max(0, iy - r_amp)
    ya1 = min(h, iy + r_amp + 1)
    peak_pri = float(np.max(filtered[ya0:ya1, xa0:xa1])) if ya1 > ya0 and xa1 > xa0 else 0.0

    # --- 下光斑 ---
    px, py = primary
    y_lo = int(np.clip(round(py) + int(min_sep_y), 0, h - 1))
    xa = int(np.clip(round(px) - int(search_half_x), 0, w - 1))
    xb = int(np.clip(round(px) + int(search_half_x) + 1, xa + 1, w))
    search_roi = (xa, y_lo, int(xb - xa), int(h - y_lo))

    if y_lo >= h - 3:
        return {
            **empty,
            "primary": primary,
            "search_roi_xywh": search_roi,
            "reason": "no_room_below",
        }

    work = filtered.copy()
    br = float(primary_blank_r)
    if br > 0:
        yy, xx = np.ogrid[0:h, 0:w]
        work[(xx - px) ** 2 + (yy - py) ** 2 <= br * br] = 0.0

    region = work[y_lo:h, xa:xb]
    pos = region > 0
    bg2 = float(np.median(region[pos])) if np.any(pos) else 0.0
    weights2 = np.clip(region - bg2, 0.0, None)
    peak2 = float(np.max(weights2)) if weights2.size else 0.0
    if peak2 <= 1e-12:
        return {
            **empty,
            "primary": primary,
            "search_roi_xywh": search_roi,
            "reason": "no_secondary_signal",
        }

    mask2 = weights2 >= (peak2 * float(thresh_frac))
    local2 = _centroid_largest_cc_weighted(weights2, mask2)
    if local2 is None:
        return {
            **empty,
            "primary": primary,
            "search_roi_xywh": search_roi,
            "reason": "no_cc",
        }

    secondary = (local2[0] + xa, local2[1] + y_lo)
    if primary[1] <= secondary[1]:
        spots = [primary, secondary]
        primary_index = 0
    else:
        spots = [secondary, primary]
        primary_index = 1

    return {
        "primary": primary,
        "secondary": secondary,
        "spots": spots,
        "primary_index": primary_index,
        "search_roi_xywh": search_roi,
        "amp_ratio": (peak_pri / peak2) if peak2 > 1e-12 else None,
        "reason": "ok",
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


def hit_test_plot_axis_outside_view(plot, scene_pos, *, edge_margin_px=10):
    """點在 PlotItem／PlotWidget 座標軸外側（非波形區）→ ('x'|'y', 'min'|'max')。

    波形 ViewBox 內部回傳 None；靠近 ViewBox 邊緣（edge_margin_px）亦視為調軸。
    """
    try:
        vb = plot.getViewBox()
        vb_rect = vb.sceneBoundingRect()
    except Exception:
        return None

    def _frac_y(rect):
        return (scene_pos.y() - rect.top()) / max(float(rect.height()), 1.0)

    def _frac_x(rect):
        return (scene_pos.x() - rect.left()) / max(float(rect.width()), 1.0)

    # ViewBox 內：僅靠邊緣才當調軸，其餘留給縮放／雙擊復原
    if vb_rect.contains(scene_pos):
        m = float(max(0.0, edge_margin_px))
        if m <= 0:
            return None
        left_d = scene_pos.x() - vb_rect.left()
        right_d = vb_rect.right() - scene_pos.x()
        top_d = scene_pos.y() - vb_rect.top()
        bottom_d = vb_rect.bottom() - scene_pos.y()
        # 取最近邊
        dists = [
            (left_d, "y", "left"),
            (right_d, "y", "right"),
            (top_d, "x", "top"),
            (bottom_d, "x", "bottom"),
        ]
        dist, axis, side = min(dists, key=lambda t: t[0])
        if dist > m:
            return None
        if axis == "y":
            edge = "max" if _frac_y(vb_rect) < 0.5 else "min"
        else:
            edge = "min" if _frac_x(vb_rect) < 0.5 else "max"
        return axis, edge

    # (軸名, 對應 view 軸)
    specs = (
        ("left", "y"),
        ("right", "y"),
        ("bottom", "x"),
        ("top", "x"),
    )
    for name, axis in specs:
        try:
            ax = plot.getAxis(name)
            rect = ax.sceneBoundingRect()
        except Exception:
            continue
        if rect.width() <= 1 or rect.height() <= 1:
            continue
        if not rect.contains(scene_pos):
            continue
        if axis == "y":
            edge = "max" if _frac_y(rect) < 0.5 else "min"
        else:
            edge = "min" if _frac_x(rect) < 0.5 else "max"
        return axis, edge
    return None


def prompt_plot_axis_limit(parent, plot, axis, edge, *, x_label="X", y_label="Y"):
    """彈出輸入框設定 plot 的 X／Y 上下限。"""
    try:
        (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
    except Exception:
        return False
    if axis == "x":
        cur = float(x0 if edge == "min" else x1)
        title = f"設定 {x_label} 下限" if edge == "min" else f"設定 {x_label} 上限"
        label = (
            f"請輸入 {x_label} 下限:"
            if edge == "min"
            else f"請輸入 {x_label} 上限:"
        )
    else:
        cur = float(y0 if edge == "min" else y1)
        title = f"設定 {y_label} 下限" if edge == "min" else f"設定 {y_label} 上限"
        label = (
            f"請輸入 {y_label} 下限:"
            if edge == "min"
            else f"請輸入 {y_label} 上限:"
        )
    new_val, ok = QInputDialog.getDouble(
        parent, title, label, cur, -1e30, 1e30, 3
    )
    if not ok:
        return False
    new_val = float(new_val)
    if axis == "x":
        lo, hi = (new_val, float(x1)) if edge == "min" else (float(x0), new_val)
        if lo >= hi:
            QMessageBox.warning(parent, "提醒", "下限必須小於上限。")
            return False
        plot.setXRange(lo, hi, padding=0)
    else:
        lo, hi = (new_val, float(y1)) if edge == "min" else (float(y0), new_val)
        if lo >= hi:
            QMessageBox.warning(parent, "提醒", "下限必須小於上限。")
            return False
        plot.setYRange(lo, hi, padding=0)
    return True


class WaveformAxisScaleController:
    """波型圖：雙擊復原；僅點座標軸外側才開上下限（延遲避免誤觸雙擊）。"""

    def __init__(
        self,
        plot,
        parent,
        *,
        x_label="X",
        y_label="Y",
        reset_callback=None,
        connect_scene=True,
    ):
        self.plot = plot
        self.parent = parent
        self.x_label = x_label
        self.y_label = y_label
        self.reset_callback = reset_callback
        self._default = None  # (x0, x1, y0, y1)
        self._user_locked = False  # 使用者設過軸限後，參數更新勿覆寫視窗
        self._pending = None
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fire_pending)
        try:
            tip = plot.toolTip() or ""
            extra = "點座標軸外側可輸入上下限；雙擊圖面復原預設範圍"
            if extra not in tip:
                plot.setToolTip((tip + "\n" + extra).strip() if tip else extra)
        except Exception:
            pass
        if connect_scene:
            try:
                plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
            except Exception:
                pass

    def set_default_range(self, x0, x1, y0, y1):
        self._default = (float(x0), float(x1), float(y0), float(y1))

    def is_user_locked(self):
        return bool(self._user_locked)

    def mark_user_range(self):
        self._user_locked = True

    def clear_pending(self):
        self._timer.stop()
        self._pending = None

    def reset_view(self):
        self.clear_pending()
        self._user_locked = False
        if callable(self.reset_callback):
            self.reset_callback()
            return
        if self._default is not None:
            x0, x1, y0, y1 = self._default
            self.plot.setXRange(x0, x1, padding=0.02)
            self.plot.setYRange(y0, y1, padding=0)
            return
        try:
            self.plot.enableAutoRange()
        except Exception:
            pass

    def schedule_axis_prompt(self, axis, edge):
        """延遲開啟軸限輸入（供外部 ViewBox handler 使用）。"""
        self._pending = (axis, edge)
        self._timer.start(280)

    def on_mouse_clicked(self, evt):
        """可供外部 scene handler 轉呼叫；回傳 True 表示已處理。"""
        try:
            pos = evt.scenePos()
        except Exception:
            return False

        # 必須落在此 plot 的外框內（共用 scene 時過濾）
        try:
            if not self.plot.sceneBoundingRect().contains(pos):
                return False
        except Exception:
            return False

        if evt.double():
            self.clear_pending()
            # 雙擊：在波形區或整塊 plot 上都復原
            self.reset_view()
            return True

        if evt.button() not in (Qt.LeftButton, Qt.NoButton):
            return False

        hit = hit_test_plot_axis_outside_view(self.plot, pos, edge_margin_px=18)
        if hit is None:
            return False
        axis, edge = hit
        self.schedule_axis_prompt(axis, edge)
        return True

    def _fire_pending(self):
        pending = self._pending
        self._pending = None
        if not pending:
            return
        axis, edge = pending
        ok = prompt_plot_axis_limit(
            self.parent,
            self.plot,
            axis,
            edge,
            x_label=self.x_label,
            y_label=self.y_label,
        )
        if ok:
            self.mark_user_range()


def attach_waveform_axis_scale(
    plot,
    parent,
    *,
    x_label="X",
    y_label="Y",
    reset_callback=None,
    connect_scene=True,
):
    """幫波型 PlotWidget／PlotItem 掛上雙擊復原＋軸外側輸入上下限。"""
    return WaveformAxisScaleController(
        plot,
        parent,
        x_label=x_label,
        y_label=y_label,
        reset_callback=reset_callback,
        connect_scene=connect_scene,
    )


def _curve_display_name(item):
    name = None
    if hasattr(item, "name"):
        try:
            name = item.name()
        except Exception:
            name = None
    if not name:
        opts = getattr(item, "opts", None) or {}
        name = opts.get("name")
    return str(name) if name else "曲線"


class CurveHoverReadout:
    """滑鼠靠近曲線時，標出該點像素與畫面上的數值。"""

    def __init__(self, plot, *, pixel_name="像素", pixel_axis="x", max_px=14.0):
        self.plot = plot
        self.pixel_name = str(pixel_name)
        self.pixel_axis = "y" if str(pixel_axis).lower() == "y" else "x"
        self.max_px = float(max_px)
        self.label = pg.TextItem(
            color="#212121",
            fill=pg.mkBrush(255, 255, 255, 235),
            border=pg.mkPen("#212121", width=1),
            anchor=(0, 0),
        )
        self.label.setZValue(40)
        self.dot = pg.ScatterPlotItem(
            size=9,
            pen=pg.mkPen("#212121", width=1.2),
            brush=pg.mkBrush(255, 235, 59, 230),
        )
        self.dot.setZValue(39)
        try:
            self.label.setAcceptedMouseButtons(Qt.NoButton)
            self.dot.setAcceptedMouseButtons(Qt.NoButton)
        except Exception:
            pass
        self._add_overlay()
        scene = plot.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self._on_moved)

    def _plot_item(self):
        return self.plot.plotItem if hasattr(self.plot, "plotItem") else self.plot

    def _add_overlay(self):
        plot = self._plot_item()
        for item in (self.dot, self.label):
            try:
                if item.scene() is None:
                    plot.addItem(item, ignoreBounds=True)
                    item.hide()
            except Exception:
                pass

    def _curve_items(self):
        plot = self._plot_item()
        found = []
        seen = set()
        for item in list(getattr(plot, "items", []) or []):
            if item is self.dot or item is self.label or id(item) in seen:
                continue
            getter = getattr(item, "getData", None)
            if getter is None:
                continue
            try:
                data = getter()
            except Exception:
                continue
            if not data or len(data) < 2 or data[0] is None or data[1] is None:
                continue
            xd = np.asarray(data[0], dtype=float)
            yd = np.asarray(data[1], dtype=float)
            if xd.size == 0 or yd.size != xd.size:
                continue
            seen.add(id(item))
            found.append((item, xd, yd))
        return found

    def _hide(self):
        self.label.hide()
        self.dot.hide()

    def _on_moved(self, pos):
        plot = self._plot_item()
        self._add_overlay()
        try:
            if not plot.sceneBoundingRect().contains(pos):
                self._hide()
                return
        except Exception:
            self._hide()
            return
        vb = plot.getViewBox()
        if vb is None:
            self._hide()
            return
        mouse = vb.mapSceneToView(pos)
        mx, my = float(mouse.x()), float(mouse.y())
        try:
            (x0, x1), (y0, y1) = vb.viewRange()
        except Exception:
            self._hide()
            return
        rect = plot.sceneBoundingRect()
        sx = rect.width() / (float(x1) - float(x0)) if float(x1) != float(x0) else 0.0
        sy = rect.height() / (float(y1) - float(y0)) if float(y1) != float(y0) else 0.0
        if sx <= 0.0 or sy <= 0.0:
            self._hide()
            return

        hits = []
        for item, xd, yd in self._curve_items():
            finite = np.isfinite(xd) & np.isfinite(yd)
            if not np.any(finite):
                continue
            dx = (xd - mx) * sx
            dy = (yd - my) * sy
            dist = np.hypot(dx, dy)
            dist = np.where(finite, dist, np.inf)
            i = int(np.argmin(dist))
            d = float(dist[i])
            if d > self.max_px:
                continue
            if self.pixel_axis == "y":
                pixel, value = float(yd[i]), float(xd[i])
            else:
                pixel, value = float(xd[i]), float(yd[i])
            hits.append(
                {
                    "dist": d,
                    "pixel": pixel,
                    "value": value,
                    "name": _curve_display_name(item),
                    "vx": float(xd[i]),
                    "vy": float(yd[i]),
                }
            )
        if not hits:
            self._hide()
            return
        hits.sort(key=lambda h: h["dist"])
        best = hits[0]
        axis = self.pixel_name.lower()
        px_txt = (
            str(int(round(best["pixel"])))
            if abs(best["pixel"] - round(best["pixel"])) < 0.05
            else f"{best['pixel']:.1f}"
        )
        lines = [f"{axis} = {px_txt} px"]
        if len(hits) == 1:
            lines.append(f"{float(best['value']):.3e}")
        else:
            for hit in hits:
                lines.append(f"{hit['name']}  {float(hit['value']):.3e}")
        self.label.setText("\n".join(lines))
        self.dot.setData([best["vx"]], [best["vy"]])
        rect = vb.sceneBoundingRect()
        origin = rect.topLeft()
        scene_pt = origin.__class__(origin.x() + 8, origin.y() + 6)
        view_pt = vb.mapSceneToView(scene_pt)
        self.label.setPos(float(view_pt.x()), float(view_pt.y()))
        self.label.show()
        self.dot.show()


def attach_curve_hover_readout(plot, *, pixel_name="像素", pixel_axis="x"):
    """在剖面圖上顯示游標所在曲線點的像素與畫面數值。"""
    existing = getattr(plot, "_curve_hover_readout", None)
    if existing is not None:
        return existing
    readout = CurveHoverReadout(
        plot, pixel_name=pixel_name, pixel_axis=pixel_axis
    )
    try:
        plot._curve_hover_readout = readout
    except Exception:
        pass
    return readout


class InteractiveHeatmapPanel(QWidget):
    """自訂座標熱力圖：滾輪縮放、拖曳平移、雙擊復原視野；
    色條可直接拖曳藍色拉條調整上下限（亦可點擊數值精調）；上方可匯出當前圖檔。
    with_profiles=True 時左側／下方顯示 X/Y 剖面（點擊更新，可隱藏十字）。"""

    levelsChanged = pyqtSignal(float, float)
    mouseMoved = pyqtSignal(object)  # QPointF in view coords, or None
    profilePointChanged = pyqtSignal(int, int)  # ix, iy
    crossVisibilityChanged = pyqtSignal(bool)  # True=顯示十字／極值, False=隱藏
    _HIST_RANGE_PAD = 0.08  # 輸入上下限後色條可視範圍相對該區間的外擴

    def __init__(self, title="Heatmap", parent=None, x_label="X", y_label="Y",
                 aspect_locked=True, with_profiles=False,
                 show_colorbar=True, show_tip=True):
        super().__init__(parent)
        self._title = title
        self._default_levels = None  # (min, max)
        self._data_hist_range = None  # 完整資料範圍，色條軸以此為準
        self._data_rect = None  # QRectF in data/view coordinates (e.g. mm)
        self._level_updating = False
        self._aspect_locked = bool(aspect_locked)
        self._with_profiles = bool(with_profiles)
        self._show_colorbar = bool(show_colorbar)
        self._show_tip = bool(show_tip)
        self._profile_matrix = None  # (ny, nx)
        self._profile_x = None
        self._profile_y = None
        self._profile_point = None  # (ix, iy)
        self._profile_cross_items = []
        self._show_profile_cross = True
        self._profile_view_ready = False
        self._x_label = x_label
        self._y_label = y_label
        self._profile_scale_x = None
        self._profile_scale_y = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        if self._with_profiles:
            tip_text = (
                "滾輪縮放主圖／剖面（剖面不跟主圖連動）· 點擊更新剖面 · "
                "雙擊主圖／剖面復原 · 點剖面座標軸外側可輸入上下限"
            )
        else:
            tip_text = (
                "滾輪縮放 · 拖曳平移 · 雙擊圖面復原 · "
                "色條拖曳藍色拉條調上下限 · 雙擊色條復原"
            )
        self.lbl_tip = QLabel(tip_text)
        self.lbl_tip.setStyleSheet("color: #607D8B; font-size: 11px;")
        self.lbl_tip.setWordWrap(True)
        self.lbl_tip.setVisible(self._show_tip)
        # 提示獨立一行，放在「匯出當前圖檔」上方，避免與按鈕擠在同一列
        root.addWidget(self.lbl_tip)

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

        # 剖面十字顏色（非紅）：熱圖十字與剖面位置線共用
        self._profile_cross_pen_color = "#00ACC1"
        self._profile_cross_pen_width = 1.6

        # 放在匯出旁，避免擠在上下限右側造成工具列跳動
        self.btn_toggle_cross = QPushButton("隱藏剖面十字")
        self.btn_toggle_cross.setCheckable(True)
        self.btn_toggle_cross.setChecked(False)
        self.btn_toggle_cross.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_toggle_cross.setToolTip(
            "隱藏熱圖上的剖面十字與剖面圖位置線；剖面波形本身仍保留"
        )
        self.btn_toggle_cross.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: #006064; "
            "background-color: #E0F7FA; border: 1px solid #4DD0E1; border-radius: 4px; "
            "padding: 5px 10px; }"
            "QPushButton:checked { color: white; background-color: #00838F; "
            "border-color: #006064; }"
            "QPushButton:hover { background-color: #B2EBF2; }"
            "QPushButton:checked:hover { background-color: #0097A7; }"
        )
        self.btn_toggle_cross.toggled.connect(self._on_toggle_cross)
        self.btn_toggle_cross.setVisible(self._with_profiles)
        toolbar.addWidget(self.btn_toggle_cross)
        toolbar.addStretch(1)

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

        self.thresh_overlay_item = pg.ImageItem()
        self.thresh_overlay_item.setZValue(5)
        self.thresh_overlay_item.setOpacity(1.0)
        self.plot.addItem(self.thresh_overlay_item)

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
            self.plot_y_profile.showGrid(x=True, y=True, alpha=0.25)
            configure_stable_plot_item(self.plot_y_profile, mouse_enabled=True)
            self.plot_y_profile.getViewBox().setMouseEnabled(x=True, y=True)

            if self._show_colorbar:
                self.win.addItem(self.hist, row=0, col=2)

            self.plot_x_profile = self.win.addPlot(row=1, col=1)
            self.plot_x_profile.setLabel("bottom", x_label)
            self.plot_x_profile.setLabel("right", "Value")
            self.plot_x_profile.showGrid(x=True, y=True, alpha=0.25)
            configure_stable_plot_item(self.plot_x_profile, mouse_enabled=True)
            self.plot_x_profile.getViewBox().setMouseEnabled(x=True, y=True)

            corner = self.win.addPlot(row=1, col=0)
            configure_stable_plot_item(corner, mouse_enabled=False)
            self.plot_profile_corner = corner

            self.plot.setLabel("bottom", x_label)
            self.plot.setLabel("left", y_label)

            # 剖面欄／列固定寬高；軸佔位對齊 ViewBox 邊緣
            configure_equal_profile_strips(
                self.win,
                profile_col=0,
                profile_row=1,
                view_px=PROFILE_VIEW_PX_COMPACT,
                side_axis_px=PROFILE_SIDE_AXIS_PX_COMPACT,
                edge_axis_px=PROFILE_EDGE_AXIS_PX_COMPACT,
            )
            self.win.ci.layout.setColumnStretchFactor(1, 1)
            if self._show_colorbar:
                self.win.ci.layout.setColumnStretchFactor(2, 0)
            align_profile_viewboxes(
                self.plot,
                self.plot_x_profile,
                self.plot_y_profile,
                corner=self.plot_profile_corner,
                side_axis_px=PROFILE_SIDE_AXIS_PX_COMPACT,
                edge_axis_px=PROFILE_EDGE_AXIS_PX_COMPACT,
            )
            # 可見軸標籤在對齊後再設一次，避免被 reserve 清掉
            self.plot.setLabel("bottom", x_label)
            self.plot.setLabel("left", y_label)
            self.plot_y_profile.setLabel("top", "Value")
            self.plot_y_profile.setLabel("left", y_label)
            self.plot_x_profile.setLabel("bottom", x_label)
            self.plot_x_profile.setLabel("right", "Value")

            theme_plots = [self.plot, self.plot_x_profile, self.plot_y_profile]
        else:
            self.plot.setLabel("bottom", x_label)
            self.plot.setLabel("left", y_label)
            if self._show_colorbar:
                self.win.addItem(self.hist, row=0, col=1)
            theme_plots = [self.plot]

        if not self._show_colorbar:
            self.hist.hide()

        apply_readable_plot_theme(self.win, theme_plots)
        if self._with_profiles:
            self._realign_profile_geometry()

        self.hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._mouse_proxy = pg.SignalProxy(
            self.win.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        if self._with_profiles:
            self._profile_scale_x = attach_waveform_axis_scale(
                self.plot_x_profile,
                self,
                x_label=self._x_label,
                y_label="Value",
                reset_callback=lambda: self._reset_profile_view("x"),
                connect_scene=False,
            )
            self._profile_scale_y = attach_waveform_axis_scale(
                self.plot_y_profile,
                self,
                x_label="Value",
                y_label=self._y_label,
                reset_callback=lambda: self._reset_profile_view("y"),
                connect_scene=False,
            )
            attach_curve_hover_readout(self.plot_x_profile, pixel_name="X")
            attach_curve_hover_readout(
                self.plot_y_profile, pixel_name="Y", pixel_axis="y"
            )
            # ViewBox 訊號較可靠（精簡剖面軸區窄，scene 點擊易漏）
            self._install_profile_vb_click_handlers()

    def _install_profile_vb_click_handlers(self):
        """剖面 ViewBox：雙擊復原；靠邊緣／軸外側輸入上下限。

        pyqtgraph.ViewBox 沒有 sigMouseClicked；改包裝 mouseClickEvent。
        AxisItem 會把軸上點擊轉發給 linked ViewBox，因此軸區也可調上下限。
        """
        pairs = (
            ("x", self.plot_x_profile, self._profile_scale_x),
            ("y", self.plot_y_profile, self._profile_scale_y),
        )
        for which, plot, ctrl in pairs:
            if plot is None:
                continue
            vb = plot.getViewBox()
            if vb is None:
                continue
            try:
                plot.setToolTip(
                    "滾輪縮放 · 拖曳平移 · 雙擊復原 · "
                    "點座標軸或波形邊緣可輸入上下限"
                )
            except Exception:
                pass

            # 避免重複包裝
            if getattr(vb, "_beam_profile_click_wrapped", False):
                continue
            orig = vb.mouseClickEvent

            def _mouse_click(evt, w=which, p=plot, c=ctrl, _orig=orig):
                handled = self._on_profile_vb_clicked(w, p, c, evt)
                if handled:
                    return
                try:
                    _orig(evt)
                except Exception:
                    pass

            vb.mouseClickEvent = _mouse_click
            vb._beam_profile_click_wrapped = True

    def _on_profile_vb_clicked(self, which, plot, ctrl, evt):
        """剖面 ViewBox／軸點擊。回傳 True 表示已處理。"""
        try:
            double = bool(evt.double())
        except Exception:
            double = False
        if double:
            if ctrl is not None:
                ctrl.clear_pending()
            self._reset_profile_view(which=which)
            try:
                evt.accept()
            except Exception:
                pass
            return True

        try:
            if evt.button() not in (Qt.LeftButton, Qt.NoButton):
                return False
        except Exception:
            pass

        try:
            pos = evt.scenePos()
        except Exception:
            return False

        # 靠邊緣／軸外側 → 輸入該向上下限（剖面條偏矮，邊緣寬一些）
        hit = hit_test_plot_axis_outside_view(plot, pos, edge_margin_px=18)
        if hit is None:
            return False
        axis, edge = hit
        if ctrl is not None:
            ctrl.schedule_axis_prompt(axis, edge)
        else:
            prompt_plot_axis_limit(
                self,
                plot,
                axis,
                edge,
                x_label=self._x_label if which == "x" else "Value",
                y_label="Value" if which == "x" else self._y_label,
            )
        try:
            evt.accept()
        except Exception:
            pass
        return True

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
        if self._with_profiles:
            self._realign_profile_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._with_profiles:
            self._realign_profile_geometry()

    def _realign_profile_geometry(self):
        if not self._with_profiles:
            return
        # 確保熱圖標題仍在畫布上方
        try:
            self.plot.setTitle(self._title or "")
        except Exception:
            pass
        align_profile_viewboxes(
            self.plot,
            self.plot_x_profile,
            self.plot_y_profile,
            corner=self.plot_profile_corner,
            side_axis_px=PROFILE_SIDE_AXIS_PX_COMPACT,
            edge_axis_px=PROFILE_EDGE_AXIS_PX_COMPACT,
        )
        # 對齊後還原可見軸標籤，並再鎖一次寬高（避免 label 改變 preferred size）
        try:
            self.plot.setLabel("bottom", self._x_label)
            self.plot.setLabel("left", self._y_label)
            self.plot.getAxis("left").setWidth(PROFILE_SIDE_AXIS_PX_COMPACT)
            self.plot.getAxis("bottom").setHeight(PROFILE_EDGE_AXIS_PX_COMPACT)
            if self.plot_x_profile is not None:
                self.plot_x_profile.setLabel("bottom", self._x_label)
                self.plot_x_profile.setLabel("right", "Value")
                self.plot_x_profile.getAxis("right").setWidth(PROFILE_SIDE_AXIS_PX_COMPACT)
                self.plot_x_profile.getAxis("bottom").setHeight(PROFILE_EDGE_AXIS_PX_COMPACT)
                self.plot_x_profile.getAxis("left").setWidth(PROFILE_SIDE_AXIS_PX_COMPACT)
            if self.plot_y_profile is not None:
                self.plot_y_profile.setLabel("left", self._y_label)
                self.plot_y_profile.setLabel("top", "Value")
                self.plot_y_profile.getAxis("left").setWidth(PROFILE_SIDE_AXIS_PX_COMPACT)
                self.plot_y_profile.getAxis("top").setHeight(PROFILE_EDGE_AXIS_PX_COMPACT)
                self.plot_y_profile.getAxis("bottom").setHeight(PROFILE_EDGE_AXIS_PX_COMPACT)
            _reserve_axis_slot(self.plot, "top", PROFILE_EDGE_AXIS_PX_COMPACT, visible_values=False)
            _reserve_axis_slot(self.plot, "right", PROFILE_SIDE_AXIS_PX_COMPACT, visible_values=False)
            _sync_profile_title_rows(
                self.plot,
                self.plot_y_profile,
                self.plot_x_profile,
                self.plot_profile_corner,
            )
        except Exception:
            pass

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

    def set_threshold_overlay(self, mask, visible=True, rgba_color=(255, 64, 255, 90)):
        """半透明門檻區域（mask 為 ny×nx，與 source_matrix／顯示矩陣同向）。"""
        if getattr(self, "thresh_overlay_item", None) is None:
            return
        if (not visible) or mask is None:
            self.clear_threshold_overlay()
            return
        mask = np.asarray(mask, dtype=bool)
        ref = self._profile_matrix
        if ref is not None and mask.shape != ref.shape:
            self.clear_threshold_overlay()
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        if np.any(mask):
            rgba[mask, 0] = int(rgba_color[0])
            rgba[mask, 1] = int(rgba_color[1])
            rgba[mask, 2] = int(rgba_color[2])
            rgba[mask, 3] = int(rgba_color[3])
        # 與 set_image(display.T) 相同轉置
        self.thresh_overlay_item.setImage(
            np.transpose(rgba, (1, 0, 2)), levels=(0, 255)
        )
        if self._data_rect is not None:
            try:
                self.thresh_overlay_item.setRect(self._data_rect)
            except Exception:
                pass

    def clear_threshold_overlay(self):
        item = getattr(self, "thresh_overlay_item", None)
        if item is not None:
            item.clear()

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
            vmin, vmax = finite_value_minmax(image)
            self.set_default_levels(vmin, vmax, apply=True)
        # levels 更新後再套一次 rect，避免 transform 被重設
        self._apply_data_rect(self._data_rect)
        if self._with_profiles:
            self._bind_profile_data(image, source_matrix, x_coords, y_coords)
            self._realign_profile_geometry()
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
        vmin, vmax = finite_value_minmax([vmin, vmax], default=(0.0, 1.0))
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
        vmin, vmax = finite_value_minmax([vmin, vmax], default=(0.0, 1.0))
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
        self.btn_toggle_cross.setText("顯示剖面十字" if hidden else "隱藏剖面十字")
        self._redraw_profile_cross()
        self.update_inline_profiles(reset_view=False)
        self.crossVisibilityChanged.emit(self._show_profile_cross)

    def set_profile_cross_visible(self, visible):
        """顯示／隱藏剖面十字（含剖面位置線）。"""
        hidden = not bool(visible)
        btn = getattr(self, "btn_toggle_cross", None)
        if btn is not None and btn.isChecked() != hidden:
            btn.blockSignals(True)
            btn.setChecked(hidden)
            btn.blockSignals(False)
        self._on_toggle_cross(hidden)

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
        color = getattr(self, "_profile_cross_pen_color", "#00ACC1")
        width = float(getattr(self, "_profile_cross_pen_width", 1.6))
        pen = pg.mkPen(color, width=width)
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
            pg.PlotCurveItem(x_axis, x_profile, pen=pg.mkPen("#00838F", width=1.5))
        )
        self.plot_y_profile.addItem(
            pg.PlotCurveItem(y_profile, y_axis, pen=pg.mkPen("#5C6BC0", width=1.5))
        )
        if self._show_profile_cross:
            cross_color = getattr(self, "_profile_cross_pen_color", "#00ACC1")
            cross_pen = pg.mkPen(cross_color, width=1.2, style=Qt.DashLine)
            self.plot_x_profile.addItem(
                pg.InfiniteLine(pos=x_pos, angle=90, pen=cross_pen)
            )
            self.plot_y_profile.addItem(
                pg.InfiniteLine(pos=y_pos, angle=0, pen=cross_pen)
            )
        self.plot_x_profile.setToolTip(f"Horizontal profile at Y = {y_pos:.6g}")
        self.plot_y_profile.setToolTip(f"Vertical profile at X = {x_pos:.6g}")
        # 橫剖面可藏標題；縱剖面需保留與熱圖同高的標題列佔位
        try:
            self.plot_x_profile.titleLabel.hide()
        except Exception:
            pass
        _sync_profile_title_rows(
            self.plot,
            self.plot_y_profile,
            self.plot_x_profile,
            self.plot_profile_corner,
        )

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
            if self._profile_scale_x is not None and x_axis.size:
                self._profile_scale_x.set_default_range(
                    float(x_axis[0]),
                    float(x_axis[-1]),
                    float(default_i0),
                    float(default_i1),
                )
            if self._profile_scale_y is not None and y_axis.size:
                self._profile_scale_y.set_default_range(
                    float(default_i0),
                    float(default_i1),
                    float(y_axis[0]),
                    float(y_axis[-1]),
                )

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
            if self._profile_point is None:
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
            x0, x1 = float(x_axis[0]), float(x_axis[-1])
            y0, y1 = lo - pad, hi + pad
            self.plot_x_profile.setXRange(x0, x1, padding=0.02)
            self.plot_x_profile.setYRange(y0, y1, padding=0)
            if self._profile_scale_x is not None:
                self._profile_scale_x.set_default_range(x0, x1, y0, y1)
            self._profile_view_ready = True
        elif which == "y":
            if self.plot_y_profile is None or self._profile_matrix is None:
                return
            if self._profile_point is None:
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
            x0, x1 = lo - pad, hi + pad
            y0, y1 = float(y_axis[0]), float(y_axis[-1])
            self.plot_y_profile.setXRange(x0, x1, padding=0)
            self.plot_y_profile.setYRange(y0, y1, padding=0.02)
            if self._profile_scale_y is not None:
                self._profile_scale_y.set_default_range(x0, x1, y0, y1)
            self._profile_view_ready = True
        else:
            self.update_inline_profiles(reset_view=True)
        configure_stable_plot_item(self.plot_x_profile, mouse_enabled=True)
        configure_stable_plot_item(self.plot_y_profile, mouse_enabled=True)
        if self.plot_x_profile is not None:
            self.plot_x_profile.getViewBox().setMouseEnabled(x=True, y=True)
        if self.plot_y_profile is not None:
            self.plot_y_profile.getViewBox().setMouseEnabled(x=True, y=True)

    def _profile_marker_items(self):
        """剖面圖上的位置指示線。"""
        items = []
        for plot in (self.plot_x_profile, self.plot_y_profile):
            if plot is None:
                continue
            for item in plot.items:
                if isinstance(item, pg.InfiniteLine):
                    items.append(item)
        return items

    def _prepare_clean_bundle_export(self):
        """匯出 heatmap+剖面 bundle 時暫藏 colorbar 與十字／位置線。回傳 restore。"""
        restore_fns = []

        if self.hist is not None:
            try:
                was_visible = self.hist.isVisible()
                self.hist.hide()
                restore_fns.append(
                    lambda h=self.hist, vis=was_visible: h.setVisible(vis)
                )
            except Exception:
                pass

        for item in list(self._profile_cross_items or []):
            try:
                was_visible = item.isVisible()
                item.hide()
                restore_fns.append(
                    lambda it=item, vis=was_visible: it.setVisible(vis)
                )
            except Exception:
                pass

        for item in self._profile_marker_items():
            try:
                was_visible = item.isVisible()
                item.hide()
                restore_fns.append(
                    lambda it=item, vis=was_visible: it.setVisible(vis)
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

    def export_heatmap_only(self, base_path, width=None, height=None, stamp=None):
        """匯出主熱圖（不含 colorbar、十字、剖面）。檔名必含時間戳。"""
        stamp = stamp or export_timestamp_tag()
        out_path = os.path.splitext(normalize_export_image_path(base_path))[0]
        out_path = with_export_timestamp(f"{out_path}{EXPORT_IMAGE_EXT}", stamp=stamp)
        restore = self._prepare_clean_bundle_export()
        try:
            QApplication.processEvents()
            export_plot_image(
                self.plot,
                out_path,
                width=width or max(int(self.plot.width()), 400),
                height=height or max(int(self.plot.height()), 300),
            )
            return [out_path]
        finally:
            restore()

    def export_plot_bundle(self, base_path, width=None, height=None, stamp=None):
        """匯出熱圖＋剖面合成圖（不含 colorbar 與十字標示；供範圍圖等使用）。

        base_path 不含副檔名，例如 .../mapping_roi。
        檔名必含時間戳；同一批匯出請傳入相同 stamp。
        「匯出當前圖檔」仍使用 export_current_image()，會保留 colorbar 與十字。
        """
        stamp = stamp or export_timestamp_tag()
        out_path = os.path.splitext(normalize_export_image_path(base_path))[0]
        out_path = with_export_timestamp(f"{out_path}{EXPORT_IMAGE_EXT}", stamp=stamp)
        if not self._with_profiles:
            return self.export_heatmap_only(
                base_path, width=width, height=height, stamp=stamp
            )

        restore = self._prepare_clean_bundle_export()
        try:
            QApplication.processEvents()
            if self.win is not None:
                pix = self.win.grab()
                if not pix.isNull():
                    save_qpixmap_export(pix, out_path)
                    return [out_path]
            export_plot_image(
                self.plot,
                out_path,
                width=width or max(int(self.plot.width()), 400),
                height=height or max(int(self.plot.height()), 300),
            )
            return [out_path]
        finally:
            restore()

    def export_current_image(self):
        safe_title = (
            self._title.replace(" ", "_").replace("/", "-").replace("\\", "-")
        )
        stamp = export_timestamp_tag()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出當前圖檔",
            export_stamped_filename(safe_title, stamp=stamp),
            EXPORT_IMAGE_FILTER,
        )
        if not path:
            return False
        try:
            # 使用者改檔名時仍強制加上本次時間戳
            path = with_export_timestamp(
                normalize_export_image_path(path), stamp=stamp
            )
            target = getattr(self.win, "ci", None) or self.plot
            w = max(int(self.win.width()), 400)
            h = max(int(self.win.height()), 300)
            export_plot_image(target, path, width=w, height=h)
            out_path = normalize_export_image_path(path)
            QMessageBox.information(self, "匯出完成", f"已匯出：\n{out_path}")
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
            if self._profile_scale_x is not None:
                self._profile_scale_x.clear_pending()
            if self._profile_scale_y is not None:
                self._profile_scale_y.clear_pending()
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
                        try:
                            evt.accept()
                        except Exception:
                            pass
                        return
                    if (
                        self.plot_y_profile is not None
                        and self.plot_y_profile.sceneBoundingRect().contains(pos)
                    ):
                        self._reset_profile_view(which="y")
                        try:
                            evt.accept()
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
            if self.plot.sceneBoundingRect().contains(pos):
                self.reset_view()
            return

        # 剖面：座標軸／波形邊緣可輸入上下限（延遲，避免雙擊誤觸）
        if self._with_profiles and not on_hist:
            # 優先走已包裝的 ViewBox 邏輯（與軸轉發一致）
            try:
                if (
                    self.plot_x_profile is not None
                    and self.plot_x_profile.sceneBoundingRect().contains(pos)
                    and self._on_profile_vb_clicked(
                        "x", self.plot_x_profile, self._profile_scale_x, evt
                    )
                ):
                    return
                if (
                    self.plot_y_profile is not None
                    and self.plot_y_profile.sceneBoundingRect().contains(pos)
                    and self._on_profile_vb_clicked(
                        "y", self.plot_y_profile, self._profile_scale_y, evt
                    )
                ):
                    return
            except Exception:
                pass
            for ctrl in (self._profile_scale_x, self._profile_scale_y):
                if ctrl is not None and ctrl.on_mouse_clicked(evt):
                    return
            # 點在剖面波形區：不要當成熱圖點擊
            try:
                if (
                    self.plot_x_profile is not None
                    and self.plot_x_profile.sceneBoundingRect().contains(pos)
                ):
                    return
                if (
                    self.plot_y_profile is not None
                    and self.plot_y_profile.sceneBoundingRect().contains(pos)
                ):
                    return
            except Exception:
                pass

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