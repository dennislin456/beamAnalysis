"""一維投影高優先改良 — 互動模擬（不改正式演算法）。

執行（在 basler 目錄）:
  python simulate_axial_improvements.py

可調：NEAR/FAR 評分、Y 混合 α、color scale；對照現行 score。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared_components import (
    _peak_width_at_frac_1d,
    _peaks_by_derivative_1d,
    _primary_peak_from_hint,
    _roi_centered_on_xy,
    _smooth_1d_profile,
    clip_roi_to_matrix,
    compute_auto_spot_center,
)

_TEST_DIR = Path(__file__).resolve().parent / "test"


def _clip_cs(m, lo, hi):
    out = np.asarray(m, dtype=float).copy()
    out = np.where(out < lo, 0.0, out)
    out = np.where(out > hi, hi, out)
    return out


def _load_profile(cs_lo=256.0, cs_hi=3038.0, roi_w=160, roi_h=512, smooth_win=7):
    short = np.load(next(_TEST_DIR.glob("*0001*.npy"))).astype(float)
    long = np.load(next(_TEST_DIR.glob("*0002*.npy"))).astype(float)
    short_a = _clip_cs(short, cs_lo, cs_hi)
    long_a = _clip_cs(long, cs_lo, cs_hi)
    cx, cy = compute_auto_spot_center(
        short_a,
        "centroid",
        use_threshold=True,
        thresh_percent=50.0,
        bg_subtract=True,
        largest_cc_only=True,
        subpixel=True,
        power=1.0,
    )
    xywh = _roi_centered_on_xy(long_a, (cx, cy), roi_w=roi_w, roi_h=roi_h)
    bounds = clip_roi_to_matrix(long_a, xywh)
    x0, y0, x1, y1 = bounds
    work = long_a[y0:y1, x0:x1]
    raw = np.sum(np.nan_to_num(work, nan=0.0), axis=1)
    profile = np.clip(raw - float(np.median(raw)), 0.0, None)
    smooth = _smooth_1d_profile(profile, smooth_win)
    dy = np.gradient(smooth)
    peaks = _peaks_by_derivative_1d(
        profile, smooth_win=smooth_win, min_prominence_frac=0.02
    )
    hint = float(cy) - float(y0)
    primary_y = _primary_peak_from_hint(peaks, profile, hint)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "y0": int(y0),
        "xywh": xywh,
        "profile": profile,
        "smooth": smooth,
        "dy": dy,
        "peaks": peaks,
        "primary_y": float(primary_y),
        "hint": hint,
        "smooth_win": smooth_win,
    }


def _score_current(a, contrast, sep, core_w):
    dist = 1.0 / (1.0 + max(0.0, sep - core_w) / max(1.5 * core_w, 1.0))
    return a * max(contrast, 0.05) * (0.30 + 0.70 * dist), "current", dist


def _score_improved(a, contrast, sep, core_w, A_p, k_near, k_far):
    """NEAR / FAR / MID 雙模式評分（模擬用）。"""
    ratio = sep / max(core_w, 1.0)
    amp_ratio = a / max(A_p, 1e-12)
    dist = 1.0 / (1.0 + max(0.0, sep - core_w) / max(1.5 * core_w, 1.0))

    if ratio <= k_near:
        if contrast < 0.05:
            return None
        score = a * (contrast ** 2) * (1.0 + 0.3 * dist)
        return score, "NEAR", dist
    if ratio >= k_far:
        if contrast < 0.03 or a < 0.02 * A_p:
            return None
        score = a * max(contrast, 0.08) * (0.15 + 0.85 * amp_ratio)
        return score, "FAR", dist
    # MID：兩式都算取高
    cands = []
    if contrast >= 0.05:
        cands.append(
            (a * (contrast ** 2) * (1.0 + 0.3 * dist), "MID~NEAR", dist)
        )
    if contrast >= 0.03 and a >= 0.02 * A_p:
        cands.append(
            (
                a * max(contrast, 0.08) * (0.15 + 0.85 * amp_ratio),
                "MID~FAR",
                dist,
            )
        )
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0]


def _build_candidates(state, min_sep, k_near, k_far, use_improved):
    profile = state["profile"]
    smooth = state["smooth"]
    py = state["primary_y"]
    n = len(profile)
    pi = int(np.clip(round(py), 0, n - 1))
    core_w = max(6.0, _peak_width_at_frac_1d(smooth, pi, frac=0.65))
    exclude = int(
        np.clip(max(float(min_sep), 1.05 * core_w), 8, max(12, n // 5))
    )
    A_p = float(smooth[pi])
    peaks = _peaks_by_derivative_1d(
        smooth, smooth_win=1, min_prominence_frac=0.02
    )
    rows = []
    for y, a in peaks:
        y = float(y)
        a = float(a)
        sep = abs(y - py)
        if sep < exclude or a <= 0:
            continue
        lo = int(np.clip(np.floor(min(y, py)), 0, n - 1))
        hi = int(np.clip(np.ceil(max(y, py)), lo, n - 1))
        valley = float(np.min(smooth[lo : hi + 1])) if hi > lo else 0.5 * (a + A_p)
        contrast = (min(a, A_p) - valley) / max(min(a, A_p), 1e-12)
        if use_improved:
            got = _score_improved(a, contrast, sep, core_w, A_p, k_near, k_far)
            if got is None:
                continue
            score, mode, dist = got
        else:
            if contrast <= 0.02:
                continue
            score, mode, dist = _score_current(a, contrast, sep, core_w)
        rows.append(
            {
                "y": y,
                "a": a,
                "sep": sep,
                "contrast": contrast,
                "score": score,
                "mode": mode,
                "dist": dist,
                "ratio": sep / max(core_w, 1.0),
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, core_w, exclude, A_p


class SimWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Axial 1D 高優先改良模擬")
        self.resize(1100, 780)
        self._state = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # controls
        side = QVBoxLayout()
        form = QFormLayout()
        self.spin_cs_lo = QDoubleSpinBox()
        self.spin_cs_lo.setRange(0, 1e6)
        self.spin_cs_lo.setValue(256)
        self.spin_cs_hi = QDoubleSpinBox()
        self.spin_cs_hi.setRange(1, 1e6)
        self.spin_cs_hi.setValue(3038)
        self.spin_min_sep = QSpinBox()
        self.spin_min_sep.setRange(4, 200)
        self.spin_min_sep.setValue(12)
        self.spin_k_near = QDoubleSpinBox()
        self.spin_k_near.setRange(0.5, 10)
        self.spin_k_near.setDecimals(2)
        self.spin_k_near.setSingleStep(0.1)
        self.spin_k_near.setValue(2.2)
        self.spin_k_far = QDoubleSpinBox()
        self.spin_k_far.setRange(1.0, 20)
        self.spin_k_far.setDecimals(2)
        self.spin_k_far.setSingleStep(0.1)
        self.spin_k_far.setValue(3.8)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0, 1)
        self.spin_alpha.setDecimals(2)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.60)
        self.spin_alpha.setToolTip(
            "Y 混合：cy = α·y_1d + (1-α)·y_2d_proxy（此模擬用 1D 峰當 y_1d，"
            "y_2d_proxy 用現行二次精修後 Y；僅示意）"
        )
        form.addRow("CS lo", self.spin_cs_lo)
        form.addRow("CS hi", self.spin_cs_hi)
        form.addRow("min_sep", self.spin_min_sep)
        form.addRow("k_near", self.spin_k_near)
        form.addRow("k_far", self.spin_k_far)
        form.addRow("Y α (1D)", self.spin_alpha)
        side.addLayout(form)

        self.chk_improved = QCheckBox("使用改良 NEAR/FAR 評分")
        self.chk_improved.setChecked(True)
        self.chk_show_all = QCheckBox("圖上標全部候選")
        self.chk_show_all.setChecked(True)
        side.addWidget(self.chk_improved)
        side.addWidget(self.chk_show_all)

        btn = QPushButton("重新計算")
        btn.clicked.connect(self.recompute)
        side.addWidget(btn)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        side.addWidget(self.lbl_info)

        box = QGroupBox("候選峰（依 score）")
        box_l = QVBoxLayout(box)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Y", "amp", "sep", "contrast", "ratio", "mode", "score"]
        )
        self.table.setMaximumHeight(280)
        box_l.addWidget(self.table)
        side.addWidget(box)
        side.addStretch(1)
        layout.addLayout(side, 0)

        self.fig = Figure(figsize=(8, 6), dpi=110)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        for w in (
            self.spin_cs_lo,
            self.spin_cs_hi,
            self.spin_min_sep,
            self.spin_k_near,
            self.spin_k_far,
            self.spin_alpha,
        ):
            w.valueChanged.connect(self.recompute)
        self.chk_improved.toggled.connect(self.recompute)
        self.chk_show_all.toggled.connect(self.recompute)

        self.recompute()

    def recompute(self):
        try:
            self._state = _load_profile(
                cs_lo=float(self.spin_cs_lo.value()),
                cs_hi=float(self.spin_cs_hi.value()),
            )
        except Exception as e:
            self.lbl_info.setText(f"載入失敗: {e}")
            return

        st = self._state
        rows, core_w, exclude, A_p = _build_candidates(
            st,
            min_sep=int(self.spin_min_sep.value()),
            k_near=float(self.spin_k_near.value()),
            k_far=float(self.spin_k_far.value()),
            use_improved=self.chk_improved.isChecked(),
        )
        # 對照現行最佳
        rows_cur, _, _, _ = _build_candidates(
            st,
            min_sep=int(self.spin_min_sep.value()),
            k_near=float(self.spin_k_near.value()),
            k_far=float(self.spin_k_far.value()),
            use_improved=False,
        )

        sec = rows[0] if rows else None
        sec_cur = rows_cur[0] if rows_cur else None
        alpha = float(self.spin_alpha.value())
        # 示意：假設 2D 精修 Y 會被拉向主峰 8px（弱下斑常見偏差）
        y1d = sec["y"] if sec else None
        y2d_proxy = (y1d - 8.0) if (y1d is not None and y1d > st["primary_y"]) else (
            (y1d + 8.0) if y1d is not None else None
        )
        y_blend = (
            alpha * y1d + (1.0 - alpha) * y2d_proxy
            if y1d is not None and y2d_proxy is not None
            else None
        )

        self.table.setRowCount(len(rows[:12]))
        for i, r in enumerate(rows[:12]):
            vals = [
                f"{r['y']:.1f}",
                f"{r['a']:.0f}",
                f"{r['sep']:.1f}",
                f"{r['contrast']:.3f}",
                f"{r['ratio']:.2f}",
                r["mode"],
                f"{r['score']:.3g}",
            ]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))

        mode = "改良 NEAR/FAR" if self.chk_improved.isChecked() else "現行"
        lines = [
            f"短曝光中心 ({st['cx']:.1f}, {st['cy']:.1f})  ROI={st['xywh']}",
            f"primary Y={st['primary_y']:.1f}  core_w={core_w:.1f}  exclude={exclude}  A_p={A_p:.0f}",
            f"[{mode}] 第二峰 Y={sec['y']:.1f} score={sec['score']:.3g} ({sec['mode']})"
            if sec
            else f"[{mode}] 無第二峰",
        ]
        if sec_cur:
            lines.append(
                f"[現行] 第二峰 Y={sec_cur['y']:.1f} score={sec_cur['score']:.3g}"
            )
        if y_blend is not None:
            lines.append(
                f"Y 混合示意 α={alpha:.2f}: 1D={y1d:.1f}, 2D≈{y2d_proxy:.1f} → blend={y_blend:.1f}"
            )
        self.lbl_info.setText("\n".join(lines))

        # plot
        self.fig.clear()
        ax0 = self.fig.add_subplot(211)
        ax1 = self.fig.add_subplot(212, sharex=ax0)
        y = np.arange(len(st["profile"]))
        ax0.plot(y, st["profile"], color="#90A4AE", lw=0.8, label="raw")
        ax0.plot(y, st["smooth"], color="#1565C0", lw=1.6, label="smooth")
        ax0.axvline(st["primary_y"], color="#C62828", ls="--", lw=1.2, label="primary")
        if sec:
            ax0.axvline(sec["y"], color="#EF6C00", ls="--", lw=1.2, label="secondary")
        if y_blend is not None:
            ax0.axvline(y_blend, color="#6A1B9A", ls=":", lw=1.4, label="Y blend")
        if self.chk_show_all.isChecked():
            for r in rows[:8]:
                ax0.plot(r["y"], np.interp(r["y"], y, st["smooth"]), "v", color="#6A1B9A", ms=5)
        ax0.set_ylabel("Sum")
        ax0.set_title(
            f"Sim | {'improved' if self.chk_improved.isChecked() else 'current'} | "
            f"k_near={self.spin_k_near.value():.1f} k_far={self.spin_k_far.value():.1f}"
        )
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        ax1.plot(y, st["dy"], color="#2E7D32", lw=1.3)
        ax1.axhline(0, color="#BDBDBD", lw=0.8)
        ax1.axvline(st["primary_y"], color="#C62828", ls="--", lw=1.2)
        if sec:
            ax1.axvline(sec["y"], color="#EF6C00", ls="--", lw=1.2)
        ax1.set_xlabel("Y (ROI local px)")
        ax1.set_ylabel("d/dy")
        ax1.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = SimWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
