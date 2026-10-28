"""Basler 批量分析分頁：沿用 M2 Batch 流程，pixel pitch 使用 3.45 um/px。"""

import math
import os

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import uniform_filter

from shared_components import (
    EXPORT_IMAGE_EXT,
    EXPORT_IMAGE_FILTER,
    InteractiveHeatmapPanel,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    clip_roi_to_matrix,
    export_plot_image,
    export_stamped_filename,
    normalize_export_image_path,
)

from tab_batch_m2 import DataRayBatchM2Tab

# 黑／白門檻預設（原始強度單位；log 顯示時會自動換算）
_DEFAULT_BLACK_BELOW = 128.0
_DEFAULT_WHITE_ABOVE = 65535.0


def _build_bw_clip_colormap(base_cmap, vmin, vmax, black_floor=None, white_ceil=None):
    """在目前 levels 內：≤black 黑、≥white 白，中間走 base colormap。"""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return base_cmap
    span = vmax - vmin
    t_lo = 0.0
    t_hi = 1.0
    if black_floor is not None and np.isfinite(black_floor):
        t_lo = float(np.clip((float(black_floor) - vmin) / span, 0.0, 1.0))
    if white_ceil is not None and np.isfinite(white_ceil):
        t_hi = float(np.clip((float(white_ceil) - vmin) / span, 0.0, 1.0))
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    if t_lo <= 0.0 and t_hi >= 1.0:
        return base_cmap
    if t_lo >= 1.0:
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (0, 0, 0)])
    if t_hi <= 0.0:
        return pg.ColorMap([0.0, 1.0], [(255, 255, 255), (255, 255, 255)])

    pos = []
    colors = []
    if t_lo > 0.0:
        pos.extend([0.0, max(t_lo - 1e-9, 0.0)])
        colors.extend([(0, 0, 0), (0, 0, 0)])
    mid_span = max(t_hi - t_lo, 1e-9)
    samples = np.linspace(0.0, 1.0, 48)
    mapped = base_cmap.map(samples, mode="byte")
    for i, s in enumerate(samples):
        pos.append(t_lo + mid_span * float(s))
        colors.append(
            (int(mapped[i, 0]), int(mapped[i, 1]), int(mapped[i, 2]))
        )
    if t_hi < 1.0:
        pos.extend([min(t_hi + 1e-9, 1.0), 1.0])
        colors.extend([(255, 255, 255), (255, 255, 255)])
    return pg.ColorMap(pos, colors)


class BaslerRangeWindow(QMainWindow):
    """Basler 範圍檢視視窗：heatmap + 剖面 + colorbar，可匯出圖檔。"""

    def __init__(
        self,
        matrix,
        x0,
        y0,
        x1,
        y1,
        levels=None,
        grayscale=False,
        bw_enabled=False,
        black_floor=None,
        white_ceil=None,
        parent=None,
    ):
        super().__init__(parent)
        self.matrix = np.asarray(matrix, dtype=float)
        # 原圖裁切範圍（僅供標題／檔名）；視窗內座標獨立從 0 開始
        self.origin_x0 = int(x0)
        self.origin_y0 = int(y0)
        self.origin_x1 = int(x1)
        self.origin_y1 = int(y1)
        self.x0 = self.origin_x0
        self.y0 = self.origin_y0
        self.x1 = self.origin_x1
        self.y1 = self.origin_y1
        ny, nx = self.matrix.shape
        self.x_coords = np.arange(nx, dtype=float)
        self.y_coords = np.arange(ny, dtype=float)
        self._bw_enabled = bool(bw_enabled)
        self._black_floor = black_floor
        self._white_ceil = white_ceil
        self._grayscale = bool(grayscale)
        self._cmap_guard = False
        self._range_sync_guard = False

        self.setWindowTitle(
            f"Basler 範圍檢視｜原圖 X={self.origin_x0}~{self.origin_x1 - 1} "
            f"Y={self.origin_y0}~{self.origin_y1 - 1}｜視窗內座標從 0 起算"
        )
        self.resize(980, 820)

        host = QWidget(self)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.panel = InteractiveHeatmapPanel(
            title=(
                f"Basler Range  {nx}×{ny} px  "
                f"(origin X={self.origin_x0}, Y={self.origin_y0})"
            ),
            x_label="X Pixels",
            y_label="Y Pixels",
            aspect_locked=True,
            with_profiles=True,
            show_colorbar=True,
        )
        # 與主 tab 一致：Y=0 在上
        self.panel.plot.getViewBox().invertY(True)
        if self.panel.plot_y_profile is not None:
            self.panel.plot_y_profile.getViewBox().invertY(True)
        # 匯出當前圖檔：保留 colorbar，但不含十字
        try:
            self.panel.btn_export.clicked.disconnect()
        except TypeError:
            pass
        self.panel.btn_export.clicked.connect(self.export_current_without_cross)
        layout.addWidget(self.panel, 1)

        export_row = QHBoxLayout()
        self.btn_export_bundle = QPushButton("匯出 Heatmap＋剖面（無 colorbar）")
        self.btn_export_bundle.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #0288D1; border: none; border-radius: 4px; "
            "padding: 6px 12px; }"
            "QPushButton:hover { background-color: #039BE5; }"
        )
        self.btn_export_bundle.clicked.connect(self.export_heatmap_with_profiles)
        export_row.addWidget(self.btn_export_bundle)

        self.btn_export_parts = QPushButton("分開匯出 Heatmap／剖面／colorbar")
        self.btn_export_parts.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #455A64; border: none; border-radius: 4px; "
            "padding: 6px 12px; }"
            "QPushButton:hover { background-color: #546E7A; }"
        )
        self.btn_export_parts.clicked.connect(self.export_parts_separately)
        export_row.addWidget(self.btn_export_parts)
        export_row.addStretch(1)
        layout.addLayout(export_row)

        self.lbl_info = QLabel(
            "範圍視窗為獨立座標（X/Y 從 0 開始）；"
            "在 heatmap 上單擊可更新剖面；匯出不含十字／位置線。"
        )
        self.lbl_info.setStyleSheet("color: #546E7A; font-size: 11px;")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        self.lbl_profile_point = QLabel("剖面位置: （點擊 heatmap 選擇）")
        self.lbl_profile_point.setStyleSheet("color: #37474F; font-size: 12px;")
        layout.addWidget(self.lbl_profile_point)
        self.setCentralWidget(host)

        finite = self.matrix[np.isfinite(self.matrix)]
        if levels is not None:
            vmin, vmax = float(levels[0]), float(levels[1])
        elif finite.size:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        self.panel.set_image(
            self.matrix.T,
            levels=(vmin, vmax),
            reset_view=True,
            source_matrix=self.matrix,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
        )
        self.panel.set_profile_point(nx // 2, ny // 2, reset_view=True)
        self._on_profile_point_changed(nx // 2, ny // 2)
        self.panel.profilePointChanged.connect(self._on_profile_point_changed)
        self._apply_basler_style_colormap(vmin, vmax)
        self.panel.hist.sigLevelsChanged.connect(self._on_levels_changed)
        self._align_profile_pixel_axes()
        # heatmap 縮放時剖面 Pixels 軸跟著對齊
        try:
            self.panel.plot.getViewBox().sigRangeChanged.connect(
                self._on_heatmap_range_changed
            )
        except Exception:
            pass
        self._range_sync_guard = False

    def _nice_tick_step(self, span, target_labels=8):
        span = max(float(span), 1.0)
        raw = span / max(float(target_labels), 1.0)
        magnitude = 10 ** math.floor(math.log10(raw))
        for nice in (1, 2, 5, 10):
            step = nice * magnitude
            if step >= raw:
                return float(step)
        return float(10 * magnitude)

    def _align_profile_pixel_axes(self):
        """剖面 Pixels 軸與 heatmap 使用相同範圍與刻度。"""
        heat = self.panel.plot
        x_prof = self.panel.plot_x_profile
        y_prof = self.panel.plot_y_profile
        if heat is None or x_prof is None or y_prof is None:
            return
        ny, nx = self.matrix.shape
        try:
            x_range, y_range = heat.getViewBox().viewRange()
            x_range = (float(x_range[0]), float(x_range[1]))
            y_range = (float(y_range[0]), float(y_range[1]))
        except Exception:
            x_range = (0.0, float(max(nx - 1, 0)))
            y_range = (0.0, float(max(ny - 1, 0)))

        x_span = max(x_range[1] - x_range[0], 1.0)
        y_span = max(y_range[1] - y_range[0], 1.0)
        x_major = self._nice_tick_step(x_span)
        y_major = self._nice_tick_step(y_span)
        self.panel.set_tick_spacing(
            x_major, x_major / 5.0, y_major, y_major / 5.0, grid_alpha=0.25
        )

        self._range_sync_guard = True
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
            self._range_sync_guard = False

    def _on_heatmap_range_changed(self, *_args):
        if getattr(self, "_range_sync_guard", False):
            return
        self._align_profile_pixel_axes()

    def _on_profile_point_changed(self, ix, iy):
        ix, iy = int(ix), int(iy)
        if (
            iy < 0
            or ix < 0
            or iy >= self.matrix.shape[0]
            or ix >= self.matrix.shape[1]
        ):
            return
        value = float(self.matrix[iy, ix])
        # 視窗內獨立座標（從 0 起算）
        self.lbl_profile_point.setText(
            f"剖面位置: X={ix}, Y={iy}, Value={value:.6g}"
        )
        self._align_profile_pixel_axes()

    def _hide_cross_markers(self):
        """匯出前暫藏十字與剖面位置線，回傳 restore。"""
        restore_fns = []
        for item in list(getattr(self.panel, "_profile_cross_items", None) or []):
            try:
                was_visible = item.isVisible()
                item.hide()
                restore_fns.append(
                    lambda it=item, vis=was_visible: it.setVisible(vis)
                )
            except Exception:
                pass
        try:
            markers = self.panel._profile_marker_items()
        except Exception:
            markers = []
        for item in markers:
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

    def _base_colormap(self):
        if self._grayscale:
            return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
        colors = [
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ]
        pos = np.linspace(0.0, 1.0, len(colors))
        return pg.ColorMap(pos, colors)

    @staticmethod
    def _colormap_black_below(base_cmap, vmin, vmax, floor):
        return _build_bw_clip_colormap(
            base_cmap, vmin, vmax, black_floor=floor, white_ceil=None
        )

    def _apply_basler_style_colormap(self, vmin=None, vmax=None):
        if self._cmap_guard or self.panel.hist is None:
            return
        try:
            if vmin is None or vmax is None:
                vmin, vmax = self.panel.hist.getLevels()
            vmin, vmax = float(vmin), float(vmax)
        except Exception:
            return
        base = self._base_colormap()
        if self._bw_enabled:
            cmap = _build_bw_clip_colormap(
                base,
                vmin,
                vmax,
                black_floor=self._black_floor,
                white_ceil=self._white_ceil,
            )
        else:
            cmap = base
        self._cmap_guard = True
        try:
            self.panel.hist.gradient.setColorMap(cmap)
            self.panel.hist.setLevels(vmin, vmax)
        finally:
            self._cmap_guard = False

    def _on_levels_changed(self):
        self._apply_basler_style_colormap()

    def _default_export_base(self):
        return export_stamped_filename(
            f"basler_range_X{self.x0}-{self.x1 - 1}_Y{self.y0}-{self.y1 - 1}"
        )

    def export_current_without_cross(self):
        """匯出當前圖檔（含 colorbar／剖面，不含十字）。"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出當前圖檔",
            self._default_export_base(),
            EXPORT_IMAGE_FILTER,
        )
        if not path:
            return
        restore = self._hide_cross_markers()
        try:
            path = normalize_export_image_path(path)
            QApplication.processEvents()
            target = getattr(self.panel.win, "ci", None) or self.panel.plot
            w = max(int(self.panel.win.width()), 400)
            h = max(int(self.panel.win.height()), 300)
            export_plot_image(target, path, width=w, height=h)
            QMessageBox.information(
                self, "匯出完成", f"已匯出（不含十字）：\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出圖檔：\n{exc}")
        finally:
            restore()

    def export_heatmap_with_profiles(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出 Heatmap＋剖面（無 colorbar）",
            self._default_export_base(),
            EXPORT_IMAGE_FILTER,
        )
        if not path:
            return
        try:
            path = normalize_export_image_path(path)
            base = os.path.splitext(path)[0]
            if not base.endswith("_Heatmap_With_Profiles"):
                base = f"{base}_Heatmap_With_Profiles"
            # export_plot_bundle 內已隱藏 colorbar 與十字
            exported = self.panel.export_plot_bundle(base)
            if not exported:
                raise RuntimeError("無法產生圖檔")
            QMessageBox.information(
                self,
                "匯出完成",
                "已匯出 Heatmap＋剖面（不含 colorbar／十字）：\n"
                + "\n".join(exported),
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出：\n{exc}")

    def export_parts_separately(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "分開匯出（選檔名作為前綴）",
            self._default_export_base(),
            EXPORT_IMAGE_FILTER,
        )
        if not path:
            return
        restore = self._hide_cross_markers()
        try:
            path = normalize_export_image_path(path)
            base = os.path.splitext(path)[0]
            outputs = []
            QApplication.processEvents()

            heat_path = f"{base}_Heatmap{EXPORT_IMAGE_EXT}"
            export_plot_image(self.panel.plot, heat_path)
            outputs.append(heat_path)

            if self.panel.plot_y_profile is not None:
                v_path = f"{base}_V_Profile{EXPORT_IMAGE_EXT}"
                export_plot_image(self.panel.plot_y_profile, v_path)
                outputs.append(v_path)
            if self.panel.plot_x_profile is not None:
                h_path = f"{base}_H_Profile{EXPORT_IMAGE_EXT}"
                export_plot_image(self.panel.plot_x_profile, h_path)
                outputs.append(h_path)

            if self.panel.hist is not None:
                cb_path = f"{base}_Colorbar{EXPORT_IMAGE_EXT}"
                export_plot_image(self.panel.hist, cb_path)
                outputs.append(cb_path)

            npy_path = f"{base}.npy"
            np.save(npy_path, self.matrix)
            outputs.append(npy_path)

            QMessageBox.information(
                self,
                "匯出完成",
                "已分開匯出（圖檔不含十字）：\n" + "\n".join(outputs),
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出：\n{exc}")
        finally:
            restore()


class BaslerBatchTab(DataRayBatchM2Tab):
    """Basler Batch：單一 Basler 資料夾、位置配置、暫停/續跑與 ZIP 匯出。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_pixel_pitch_um = 3.45
        self._basler_cmap_guard = False
        self.basler_range_win = None
        # 手動標記圓：位置固定直到清除
        self.basler_circle1_center = None  # (cx, cy) 像素
        self.basler_circle2_center = None
        self.basler_circle_items = []
        self._adapt_ui_for_basler()

        self.lbl_batch_ab_real_distance_xy.setText(
            "M2(above)→M2(below) 實際 (*3.45): ΔX: -- μm, ΔY: -- μm"
        )
        self.lbl_batch_ab_real_distance_total.setText(
            "M2(above)→M2(below) 實際總距離: -- μm"
        )

    def _adapt_ui_for_basler(self):
        self.btn_batch_m2_dir.setText("I. 選擇 Basler 主資料夾（含位置子資料夾）")
        self.lbl_batch_m2_info.setText(
            "未選擇 Basler 主資料夾\n"
            "格式: Basler/<位置>/<檔名>.npy / .csv / .xlsx"
        )
        self.lbl_batch_pair_info.setText("掃描結果: 尚未選擇 Basler 主資料夾")
        self.btn_batch_run.setText("載入已配置位置並運算（Basler）")
        self.btn_batch_view_m2.setText("查看 Basler Heatmap（彈窗）")
        self.plot_batch_heat.setTitle("Basler Heatmap (Batch)")
        # Basler：Y=0 從上方開始（僅此 tab）
        self.plot_batch_heat.getViewBox().invertY(True)
        self.plot_batch_y_profile.getViewBox().invertY(True)
        self.lbl_batch_status.setText("狀態: 等待匯入 Basler 檔案")
        self.setWindowTitle("Basler Batch")

        for widget in self.findChildren(QLabel):
            widget.setText(widget.text().replace("*5.5", "*3.45"))

        left = getattr(self, "batch_left_settings_layout", None)
        insert_after = None
        if left is not None and hasattr(self, "chk_batch_heatmap_gray"):
            insert_after = left.indexOf(self.chk_batch_heatmap_gray)

        # —— 顯示選項（放在 heatmap 上方）——
        self.chk_batch_heatmap_log = QCheckBox("取 log10")
        self.chk_batch_heatmap_log.setChecked(False)
        self.chk_batch_heatmap_log.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_batch_heatmap_log.setToolTip(
            "僅影響 Heatmap／colorbar／剖面顯示；定位與距離運算仍使用原始數值。"
        )
        self.chk_batch_heatmap_log.toggled.connect(self._on_basler_display_option_changed)

        self.chk_basler_bw_scale = QCheckBox("黑／白門檻")
        self.chk_basler_bw_scale.setChecked(False)
        self.chk_basler_bw_scale.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_basler_bw_scale.setToolTip(
            "勾選後：數值 ≤「以下黑色」顯示黑，≥「以上白色」顯示白，中間為正常色階。\n"
            "不勾選則完全正常顯示。門檻以原始強度輸入（有 log 時會自動換算）。"
        )
        self.chk_basler_bw_scale.toggled.connect(self._on_basler_bw_scale_toggled)

        self.lbl_basler_black_below = QLabel("≤黑")
        self.lbl_basler_black_below.setStyleSheet("color: #455A64;")
        self.spin_basler_black_below = NoWheelDoubleSpinBox()
        self.spin_basler_black_below.setRange(0.0, 1e9)
        self.spin_basler_black_below.setDecimals(1)
        self.spin_basler_black_below.setValue(_DEFAULT_BLACK_BELOW)
        self.spin_basler_black_below.setFixedWidth(72)
        self.spin_basler_black_below.setEnabled(False)
        self.spin_basler_black_below.valueChanged.connect(self._on_basler_bw_threshold_changed)
        self.lbl_basler_white_above = QLabel("≥白")
        self.lbl_basler_white_above.setStyleSheet("color: #455A64;")
        self.spin_basler_white_above = NoWheelDoubleSpinBox()
        self.spin_basler_white_above.setRange(0.0, 1e9)
        self.spin_basler_white_above.setDecimals(1)
        self.spin_basler_white_above.setValue(_DEFAULT_WHITE_ABOVE)
        self.spin_basler_white_above.setFixedWidth(72)
        self.spin_basler_white_above.setEnabled(False)
        self.spin_basler_white_above.valueChanged.connect(self._on_basler_bw_threshold_changed)

        self.chk_batch_heatmap_smooth = QCheckBox("平滑")
        self.chk_batch_heatmap_smooth.setChecked(False)
        self.chk_batch_heatmap_smooth.setStyleSheet("font-weight: bold; color: #37474F;")
        self.chk_batch_heatmap_smooth.setToolTip(
            "對目前顯示資料做移動平均平滑。\n"
            "有勾選 log：先 log10 再平滑；未勾選 log：直接平滑原始數值。"
        )
        self.chk_batch_heatmap_smooth.toggled.connect(self._on_basler_smooth_toggled)

        self.lbl_batch_heatmap_smooth = QLabel("核")
        self.lbl_batch_heatmap_smooth.setStyleSheet("color: #455A64;")
        self.spin_batch_heatmap_smooth = NoWheelSpinBox()
        self.spin_batch_heatmap_smooth.setRange(1, 201)
        self.spin_batch_heatmap_smooth.setSingleStep(2)
        self.spin_batch_heatmap_smooth.setValue(31)
        self.spin_batch_heatmap_smooth.setFixedWidth(60)
        self.spin_batch_heatmap_smooth.setToolTip(
            "移動平均核大小（建議奇數）。1 表示不平滑。"
        )
        self.spin_batch_heatmap_smooth.setEnabled(False)
        self.spin_batch_heatmap_smooth.valueChanged.connect(
            self._on_basler_display_option_changed
        )

        self._place_display_controls_above_heatmap()

        # —— 左側：範圍檢視／標記圓 ——
        range_title = QLabel("範圍檢視／匯出")
        range_title.setStyleSheet("font-weight: bold; color: #1565C0;")
        range_grid = QGridLayout()
        range_grid.setContentsMargins(0, 0, 0, 0)
        range_grid.setHorizontalSpacing(6)
        range_grid.setVerticalSpacing(4)
        self.spin_basler_range_x = NoWheelSpinBox()
        self.spin_basler_range_y = NoWheelSpinBox()
        self.spin_basler_range_w = NoWheelSpinBox()
        self.spin_basler_range_h = NoWheelSpinBox()
        for spin, default in (
            (self.spin_basler_range_x, 0),
            (self.spin_basler_range_y, 0),
            (self.spin_basler_range_w, 200),
            (self.spin_basler_range_h, 200),
        ):
            spin.setRange(0 if spin in (self.spin_basler_range_x, self.spin_basler_range_y) else 1, 100000)
            spin.setValue(default)
            spin.setFixedWidth(72)
        range_grid.addWidget(QLabel("X"), 0, 0)
        range_grid.addWidget(self.spin_basler_range_x, 0, 1)
        range_grid.addWidget(QLabel("Y"), 0, 2)
        range_grid.addWidget(self.spin_basler_range_y, 0, 3)
        range_grid.addWidget(QLabel("W"), 1, 0)
        range_grid.addWidget(self.spin_basler_range_w, 1, 1)
        range_grid.addWidget(QLabel("H"), 1, 2)
        range_grid.addWidget(self.spin_basler_range_h, 1, 3)

        self.btn_basler_view_range = QPushButton("查看範圍視窗")
        self.btn_basler_view_range.setStyleSheet(
            "QPushButton { font-size: 12px; font-weight: bold; color: white; "
            "background-color: #6A1B9A; border: none; border-radius: 4px; "
            "padding: 6px 10px; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        self.btn_basler_view_range.setToolTip(
            "依 X/Y/W/H 裁切目前顯示資料（含 log／平滑），"
            "開啟可匯出 heatmap／剖面／colorbar 的視窗。"
        )
        self.btn_basler_view_range.clicked.connect(self.show_basler_range_window)

        # 手動標記圓：兩個獨立勾選＋半徑，點擊 heatmap 定位後固定
        circle_title = QLabel("標記圓（點擊 heatmap 定位）")
        circle_title.setStyleSheet("font-weight: bold; color: #212121;")
        circle_title.setToolTip(
            "勾選後在 heatmap 單擊設定圓心；設定後會固定，需按清除才可重設。"
            "圓為紅色半透明填滿，邊框為細虛線。"
        )

        self.chk_basler_circle1 = QCheckBox("圓1")
        self.chk_basler_circle1.setChecked(False)
        self.chk_basler_circle1.setStyleSheet("font-weight: bold; color: #212121;")
        self.chk_basler_circle1.toggled.connect(self._on_basler_circle_option_changed)
        self.spin_basler_circle1_r = NoWheelDoubleSpinBox()
        self.spin_basler_circle1_r.setRange(0.5, 10000.0)
        self.spin_basler_circle1_r.setDecimals(1)
        self.spin_basler_circle1_r.setValue(20.0)
        self.spin_basler_circle1_r.setFixedWidth(72)
        self.spin_basler_circle1_r.setToolTip("圓1 半徑（px）")
        self.spin_basler_circle1_r.valueChanged.connect(self._redraw_basler_manual_circles)
        self.lbl_basler_circle1_pos = QLabel("位置: --")
        self.lbl_basler_circle1_pos.setStyleSheet("color: #546E7A;")
        self.btn_basler_clear_circle1 = QPushButton("清除1")
        self.btn_basler_clear_circle1.setFixedWidth(56)
        self.btn_basler_clear_circle1.clicked.connect(
            lambda: self._clear_basler_circle(1)
        )

        self.chk_basler_circle2 = QCheckBox("圓2")
        self.chk_basler_circle2.setChecked(False)
        self.chk_basler_circle2.setStyleSheet("font-weight: bold; color: #212121;")
        self.chk_basler_circle2.toggled.connect(self._on_basler_circle_option_changed)
        self.spin_basler_circle2_r = NoWheelDoubleSpinBox()
        self.spin_basler_circle2_r.setRange(0.5, 10000.0)
        self.spin_basler_circle2_r.setDecimals(1)
        self.spin_basler_circle2_r.setValue(20.0)
        self.spin_basler_circle2_r.setFixedWidth(72)
        self.spin_basler_circle2_r.setToolTip("圓2 半徑（px）")
        self.spin_basler_circle2_r.valueChanged.connect(self._redraw_basler_manual_circles)
        self.lbl_basler_circle2_pos = QLabel("位置: --")
        self.lbl_basler_circle2_pos.setStyleSheet("color: #546E7A;")
        self.btn_basler_clear_circle2 = QPushButton("清除2")
        self.btn_basler_clear_circle2.setFixedWidth(56)
        self.btn_basler_clear_circle2.clicked.connect(
            lambda: self._clear_basler_circle(2)
        )

        circle_grid = QGridLayout()
        circle_grid.setContentsMargins(0, 0, 0, 0)
        circle_grid.setHorizontalSpacing(6)
        circle_grid.setVerticalSpacing(4)
        circle_grid.addWidget(self.chk_basler_circle1, 0, 0)
        circle_grid.addWidget(QLabel("半徑"), 0, 1)
        circle_grid.addWidget(self.spin_basler_circle1_r, 0, 2)
        circle_grid.addWidget(self.lbl_basler_circle1_pos, 0, 3)
        circle_grid.addWidget(self.btn_basler_clear_circle1, 0, 4)
        circle_grid.addWidget(self.chk_basler_circle2, 1, 0)
        circle_grid.addWidget(QLabel("半徑"), 1, 1)
        circle_grid.addWidget(self.spin_basler_circle2_r, 1, 2)
        circle_grid.addWidget(self.lbl_basler_circle2_pos, 1, 3)
        circle_grid.addWidget(self.btn_basler_clear_circle2, 1, 4)

        self.btn_basler_clear_circles = QPushButton("清除全部圓")
        self.btn_basler_clear_circles.clicked.connect(self._clear_basler_circles)

        if left is not None:
            widgets = [
                range_title,
                ("layout", range_grid),
                self.btn_basler_view_range,
                circle_title,
                ("layout", circle_grid),
                self.btn_basler_clear_circles,
            ]
            if insert_after is not None and insert_after >= 0:
                pos = insert_after + 1
                for item in widgets:
                    if isinstance(item, tuple) and item[0] == "layout":
                        left.insertLayout(pos, item[1])
                    else:
                        left.insertWidget(pos, item)
                    pos += 1
            else:
                for item in widgets:
                    if isinstance(item, tuple) and item[0] == "layout":
                        left.addLayout(item[1])
                    else:
                        left.addWidget(item)

        self._apply_basler_colorbar_floor()
        self._update_basler_circle_labels()

    def _place_display_controls_above_heatmap(self):
        """把 log／黑白門檻／平滑放到 heatmap 上方工具列。"""
        heat = getattr(self, "win_batch_top", None)
        if heat is None:
            return
        parent = heat.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is None:
            return
        idx = layout.indexOf(heat)
        if idx < 0:
            return
        layout.removeWidget(heat)

        heat_col = QWidget()
        heat_col_layout = QVBoxLayout(heat_col)
        heat_col_layout.setContentsMargins(0, 0, 0, 0)
        heat_col_layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 2, 2, 2)
        toolbar.setSpacing(8)
        toolbar.addWidget(self.chk_batch_heatmap_log)
        toolbar.addWidget(self.chk_basler_bw_scale)
        toolbar.addWidget(self.lbl_basler_black_below)
        toolbar.addWidget(self.spin_basler_black_below)
        toolbar.addWidget(self.lbl_basler_white_above)
        toolbar.addWidget(self.spin_basler_white_above)
        toolbar.addWidget(self.chk_batch_heatmap_smooth)
        toolbar.addWidget(self.lbl_batch_heatmap_smooth)
        toolbar.addWidget(self.spin_batch_heatmap_smooth)
        toolbar.addStretch(1)

        bar_host = QWidget()
        bar_host.setObjectName("baslerDisplayBar")
        bar_host.setLayout(toolbar)
        bar_host.setStyleSheet(
            "#baslerDisplayBar { background: #ECEFF1; border: 1px solid #CFD8DC; "
            "border-radius: 4px; }"
        )
        heat_col_layout.addWidget(bar_host)
        heat_col_layout.addWidget(heat, 1)
        layout.insertWidget(idx, heat_col, 1)
        self._basler_heat_column = heat_col

    def _basler_log_enabled(self):
        chk = getattr(self, "chk_batch_heatmap_log", None)
        return chk is not None and chk.isChecked()

    def _basler_smooth_enabled(self):
        chk = getattr(self, "chk_batch_heatmap_smooth", None)
        return chk is not None and chk.isChecked()

    def _basler_smooth_size(self):
        spin = getattr(self, "spin_batch_heatmap_smooth", None)
        if spin is None:
            return 1
        return max(1, int(spin.value()))

    def _basler_display_matrix(self):
        """回傳目前要畫在 heatmap／剖面的矩陣：先可選 log10，再可選平滑。"""
        matrix = getattr(self, "batch_result_matrix", None)
        if matrix is None:
            return None
        data = np.asarray(matrix, dtype=float)
        # 1) 有勾選 log → 先取 log10；沒勾選 → 維持原始數值
        if self._basler_log_enabled():
            data = np.log10(np.maximum(data, 1.0))
        # 2) 有啟用平滑 → 對「目前這份顯示資料」做移動平均
        if self._basler_smooth_enabled():
            k = self._basler_smooth_size()
            if k > 1:
                data = uniform_filter(data, size=k, mode="nearest")
        return data

    def _basler_bw_scale_enabled(self):
        chk = getattr(self, "chk_basler_bw_scale", None)
        return chk is not None and chk.isChecked()

    def _to_display_threshold(self, value):
        """原始強度門檻 → 目前顯示座標（log 時轉 log10）。"""
        value = float(value)
        if self._basler_log_enabled():
            return float(np.log10(max(value, 1.0)))
        return value

    def _basler_bw_thresholds_display(self):
        """回傳顯示座標下的 (black_floor, white_ceil)；未啟用則 (None, None)。"""
        if not self._basler_bw_scale_enabled():
            return None, None
        black = self._to_display_threshold(self.spin_basler_black_below.value())
        white = self._to_display_threshold(self.spin_basler_white_above.value())
        return black, white

    def _update_basler_heatmap_title(self):
        title = "Basler Heatmap (Batch)"
        if self._basler_log_enabled():
            title += " [log10]"
        if self._basler_smooth_enabled() and self._basler_smooth_size() > 1:
            title += f" [smooth {self._basler_smooth_size()}]"
        if self._basler_bw_scale_enabled():
            title += (
                f" [BW ≤{self.spin_basler_black_below.value():.0f}"
                f"/≥{self.spin_basler_white_above.value():.0f}]"
            )
        if self.chk_batch_heatmap_gray.isChecked():
            title += " [Grayscale]"
        self.plot_batch_heat.setTitle(title)

    def _refresh_basler_heatmap_display(self, reset_levels=True):
        display = self._basler_display_matrix()
        if display is None or self.batch_image_item is None:
            self._update_basler_heatmap_title()
            return
        self.batch_image_item.setImage(display.T, autoLevels=False)
        if reset_levels and self.batch_hist is not None:
            finite = display[np.isfinite(display)]
            if finite.size:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmin == vmax:
                    vmax = vmin + 1.0
                self.batch_hist.setLevels(vmin, vmax)
        self._update_basler_heatmap_title()
        self._apply_basler_colorbar_floor()
        self._update_basler_profile_axis_labels()
        if (
            self.chk_batch_pixel_profile.isChecked()
            and self.batch_profile_point is not None
        ):
            self.update_batch_inline_profiles(reset_view=True)

    def _update_basler_profile_axis_labels(self):
        if self._basler_log_enabled():
            intensity_label = "log10"
        else:
            intensity_label = "Intensity"
        if self._basler_smooth_enabled() and self._basler_smooth_size() > 1:
            intensity_label += " (smooth)"
        try:
            self.plot_batch_x_profile.setLabel("right", intensity_label)
            self.plot_batch_y_profile.setLabel("top", intensity_label)
        except Exception:
            pass

    def update_batch_inline_profiles(self, reset_view=False):
        """剖面與 heatmap 同步：同一套 log／平滑顯示資料。"""
        display = self._basler_display_matrix()
        if display is None:
            return
        original = self.batch_result_matrix
        self._update_basler_profile_axis_labels()
        try:
            self.batch_result_matrix = display
            super().update_batch_inline_profiles(reset_view=reset_view)
        finally:
            self.batch_result_matrix = original

    def _on_basler_smooth_toggled(self, checked=False):
        self.spin_batch_heatmap_smooth.setEnabled(bool(checked))
        self._on_basler_display_option_changed()

    def _on_basler_bw_scale_toggled(self, checked=False):
        enabled = bool(checked)
        self.spin_basler_black_below.setEnabled(enabled)
        self.spin_basler_white_above.setEnabled(enabled)
        self._update_basler_heatmap_title()
        self._apply_basler_colorbar_floor()

    def _on_basler_bw_threshold_changed(self, *_args):
        if not self._basler_bw_scale_enabled():
            return
        self._update_basler_heatmap_title()
        self._apply_basler_colorbar_floor()

    def _on_basler_display_option_changed(self, *_args):
        self._refresh_basler_heatmap_display(reset_levels=True)

    def _basler_range_roi(self):
        return (
            self.spin_basler_range_x.value(),
            self.spin_basler_range_y.value(),
            self.spin_basler_range_w.value(),
            self.spin_basler_range_h.value(),
        )

    def show_basler_range_window(self):
        display = self._basler_display_matrix()
        if display is None:
            QMessageBox.warning(self, "警告", "請先載入 Basler 資料。")
            return
        clipped = clip_roi_to_matrix(display, self._basler_range_roi())
        if clipped is None:
            QMessageBox.warning(self, "範圍無效", "請確認 X/Y/W/H 為有效像素範圍。")
            return
        x0, y0, x1, y1 = clipped
        crop = np.asarray(display[y0:y1, x0:x1], dtype=float)
        if crop.size == 0:
            QMessageBox.warning(self, "範圍無效", "裁切後沒有資料。")
            return

        # 同步 spin 為實際裁切後數值
        self.spin_basler_range_x.setValue(x0)
        self.spin_basler_range_y.setValue(y0)
        self.spin_basler_range_w.setValue(x1 - x0)
        self.spin_basler_range_h.setValue(y1 - y0)

        levels = None
        try:
            levels = self.batch_hist.getLevels()
        except Exception:
            levels = None

        if self.basler_range_win is not None:
            try:
                self.basler_range_win.close()
            except Exception:
                pass
            self.basler_range_win = None

        self.basler_range_win = BaslerRangeWindow(
            crop,
            x0,
            y0,
            x1,
            y1,
            levels=levels,
            grayscale=self.chk_batch_heatmap_gray.isChecked(),
            bw_enabled=self._basler_bw_scale_enabled(),
            black_floor=self._basler_bw_thresholds_display()[0],
            white_ceil=self._basler_bw_thresholds_display()[1],
            parent=self,
        )
        self.basler_range_win.show()
        self.basler_range_win.raise_()
        self.basler_range_win.activateWindow()
        QApplication.processEvents()

    @staticmethod
    def _colormap_black_below(base_cmap, vmin, vmax, floor):
        return _build_bw_clip_colormap(
            base_cmap, vmin, vmax, black_floor=floor, white_ceil=None
        )

    def _apply_basler_colorbar_floor(self):
        """依黑／白門檻更新 colorbar／heatmap 色階；未啟用則正常顯示。"""
        if self._basler_cmap_guard:
            return
        hist = getattr(self, "batch_hist", None)
        if hist is None:
            return
        try:
            vmin, vmax = hist.getLevels()
        except Exception:
            return
        vmin, vmax = float(vmin), float(vmax)
        base = (
            self.batch_gray_map
            if self.chk_batch_heatmap_gray.isChecked()
            else self.batch_jet_map
        )
        if self._basler_bw_scale_enabled():
            black, white = self._basler_bw_thresholds_display()
            cmap = _build_bw_clip_colormap(
                base, vmin, vmax, black_floor=black, white_ceil=white
            )
        else:
            cmap = base
        self._basler_cmap_guard = True
        try:
            hist.gradient.setColorMap(cmap)
            hist.setLevels(vmin, vmax)
        finally:
            self._basler_cmap_guard = False

    def on_colorbar_levels_changed(self):
        self._apply_basler_colorbar_floor()

    def on_batch_heatmap_colormap_toggled(self, checked=False):
        super().on_batch_heatmap_colormap_toggled(checked)
        self._update_basler_heatmap_title()
        self._apply_basler_colorbar_floor()

    def _update_basler_circle_labels(self):
        def _fmt(center):
            if center is None:
                return "位置: --"
            return f"位置: ({center[0]:.0f}, {center[1]:.0f})"

        if hasattr(self, "lbl_basler_circle1_pos"):
            self.lbl_basler_circle1_pos.setText(_fmt(self.basler_circle1_center))
        if hasattr(self, "lbl_basler_circle2_pos"):
            self.lbl_basler_circle2_pos.setText(_fmt(self.basler_circle2_center))

    def _clear_basler_circle_items(self):
        for item in list(getattr(self, "basler_circle_items", []) or []):
            try:
                self.plot_batch_heat.removeItem(item)
            except Exception:
                pass
        self.basler_circle_items = []

    def _make_basler_filled_circle(self, cx, cy, radius):
        """紅色半透明填滿圓 + 較細虛線邊框。"""
        from PyQt5.QtGui import QBrush, QColor, QPen
        from PyQt5.QtWidgets import QGraphicsEllipseItem
        from PyQt5.QtCore import QRectF

        item = QGraphicsEllipseItem(
            QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius)
        )
        item.setPen(QPen(QColor(255, 0, 0, 160), 1, Qt.DashLine))
        item.setBrush(QBrush(QColor(255, 0, 0, 70)))
        item.setZValue(20)
        return item

    def _redraw_basler_manual_circles(self, *_args):
        self._clear_basler_circle_items()
        if getattr(self, "plot_batch_heat", None) is None:
            return
        specs = (
            (
                getattr(self, "chk_basler_circle1", None),
                self.basler_circle1_center,
                getattr(self, "spin_basler_circle1_r", None),
            ),
            (
                getattr(self, "chk_basler_circle2", None),
                self.basler_circle2_center,
                getattr(self, "spin_basler_circle2_r", None),
            ),
        )
        for chk, center, spin in specs:
            if chk is None or not chk.isChecked() or center is None or spin is None:
                continue
            radius = float(spin.value())
            if radius <= 0:
                continue
            item = self._make_basler_filled_circle(center[0], center[1], radius)
            self.plot_batch_heat.addItem(item)
            self.basler_circle_items.append(item)
        self._update_basler_circle_labels()

    def _on_basler_circle_option_changed(self, *_args):
        self._redraw_basler_manual_circles()

    def _clear_basler_circle(self, which):
        if which == 1:
            self.basler_circle1_center = None
        elif which == 2:
            self.basler_circle2_center = None
        self._redraw_basler_manual_circles()

    def _clear_basler_circles(self):
        self.basler_circle1_center = None
        self.basler_circle2_center = None
        self._redraw_basler_manual_circles()

    def _next_basler_circle_to_place(self):
        """回傳下一個待定位的圓編號（1/2），皆已固定或未勾選則 None。"""
        if (
            getattr(self, "chk_basler_circle1", None) is not None
            and self.chk_basler_circle1.isChecked()
            and self.basler_circle1_center is None
        ):
            return 1
        if (
            getattr(self, "chk_basler_circle2", None) is not None
            and self.chk_basler_circle2.isChecked()
            and self.basler_circle2_center is None
        ):
            return 2
        return None

    def _try_place_basler_circle(self, cx, cy):
        which = self._next_basler_circle_to_place()
        if which is None:
            return False
        if which == 1:
            self.basler_circle1_center = (float(cx), float(cy))
        else:
            self.basler_circle2_center = (float(cx), float(cy))
        self._redraw_basler_manual_circles()
        return True

    def on_batch_process_mouse_clicked(self, evt):
        # 先處理標記圓定位（單擊 heatmap，未固定前有效）
        if (
            self.batch_result_matrix is not None
            and not evt.double()
            and self._next_basler_circle_to_place() is not None
        ):
            pos = evt.scenePos()
            on_x = self.plot_batch_x_profile.sceneBoundingRect().contains(pos)
            on_y = self.plot_batch_y_profile.sceneBoundingRect().contains(pos)
            if (
                not on_x
                and not on_y
                and self.plot_batch_heat.sceneBoundingRect().contains(pos)
            ):
                mouse_point = self.plot_batch_heat.getViewBox().mapSceneToView(pos)
                cx = float(mouse_point.x())
                cy = float(mouse_point.y())
                h, w = self.batch_result_matrix.shape
                if 0 <= cx < w and 0 <= cy < h:
                    self._try_place_basler_circle(cx, cy)
        super().on_batch_process_mouse_clicked(evt)

    def redraw_batch_crosses(self):
        super().redraw_batch_crosses()
        self._redraw_basler_manual_circles()

    def _import_folder_dialog_title(self):
        return "選擇 Basler 主資料夾（內含位置子資料夾）"

    def load_batch_m2_folder(self):
        super().load_batch_m2_folder()
        self.btn_batch_m2_dir.setText("I. 選擇 Basler 主資料夾（含位置子資料夾）")

    def load_batch_group(self, idx):
        super().load_batch_group(idx)
        if hasattr(self, "lbl_batch_current_files"):
            self.lbl_batch_current_files.setText(
                self.lbl_batch_current_files.text()
                .replace("M2:", "Basler:")
                .replace("M2", "Basler")
            )
        self.lbl_batch_status.setText(
            f"狀態: 已載入第 {idx + 1}/{self.batch_total_count} 組（Basler）"
        )
        # 依目前影像尺寸給範圍預設（首次或超出時）
        matrix = getattr(self, "batch_result_matrix", None)
        if matrix is not None:
            h, w = matrix.shape
            self.spin_basler_range_x.setMaximum(max(0, w - 1))
            self.spin_basler_range_y.setMaximum(max(0, h - 1))
            self.spin_basler_range_w.setMaximum(max(1, w))
            self.spin_basler_range_h.setMaximum(max(1, h))
            if self.spin_basler_range_w.value() > w:
                self.spin_basler_range_w.setValue(w)
            if self.spin_basler_range_h.value() > h:
                self.spin_basler_range_h.setValue(h)
        self._refresh_basler_heatmap_display(reset_levels=True)
        self._redraw_basler_manual_circles()
