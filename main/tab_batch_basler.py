"""Basler 批量分析分頁：沿用 M2 Batch 流程，pixel pitch 使用 3.45 um/px。"""

import numpy as np

from PyQt5.QtWidgets import (
    QLabel, QWidget, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)

from tab_batch_m2 import DataRayBatchM2Tab


class BaslerBatchTab(DataRayBatchM2Tab):
    """Basler Batch：單一 Basler 資料夾、位置配置、暫停/續跑與 ZIP 匯出。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_pixel_pitch_um = 3.45
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
        self.lbl_batch_status.setText("狀態: 等待匯入 Basler 檔案")
        self.setWindowTitle("Basler Batch")

        for widget in self.findChildren(QLabel):
            widget.setText(widget.text().replace("*5.5", "*3.45"))

    def _add_distance_table(self):
        """在 Basler Heatmap 右側加入批量距離清單。"""
        right_layout = self.win_batch_top.parentWidget().layout()
        if right_layout is None:
            return

        heatmap_index = right_layout.indexOf(self.win_batch_top)
        if heatmap_index < 0:
            return
        right_layout.removeWidget(self.win_batch_top)

        host = QWidget()
        host_layout = QHBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(6)
        host_layout.addWidget(self.win_batch_top, 1)

        self.batch_distance_table = QTableWidget(0, 4)
        self.batch_distance_table.setHorizontalHeaderLabels(
            ["組別", "位置", "檔名", "距離(μm)"]
        )
        self.batch_distance_table.setMinimumWidth(330)
        self.batch_distance_table.setMaximumWidth(430)
        self.batch_distance_table.setAlternatingRowColors(True)
        self.batch_distance_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_distance_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_distance_table.cellClicked.connect(self._on_distance_row_clicked)
        self.batch_distance_table.verticalHeader().setVisible(False)
        header = self.batch_distance_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        host_layout.addWidget(self.batch_distance_table)

        right_layout.insertWidget(heatmap_index, host, 3)

    def _update_distance_table(self, idx):
        if not hasattr(self, "batch_distance_table"):
            return
        if idx < 0 or idx >= len(self.batch_pairs):
            return

        below = getattr(self, "batch_m2_center_point", None)
        above = getattr(self, "batch_m2_above_point", None)
        if below is None or above is None:
            distance_um = "--"
        else:
            distance_px_value = float(np.hypot(
                below[0] - above[0], below[1] - above[1]
            ))
            distance_um = f"{distance_px_value * self.batch_pixel_pitch_um:.2f}"

        pair = self.batch_pairs[idx]
        values = [
            str(idx + 1),
            str(pair.get("location", "")),
            str(pair.get("filename", "")),
            distance_um,
        ]
        self.batch_distance_table.setRowCount(max(self.batch_distance_table.rowCount(), idx + 1))
        for col, value in enumerate(values):
            self.batch_distance_table.setItem(idx, col, QTableWidgetItem(value))
        self.batch_distance_table.resizeRowsToContents()

    def _on_distance_row_clicked(self, row, _column):
        """點擊距離清單列時切換到該組 Basler Heatmap。"""
        if 0 <= row < self.batch_total_count:
            self.load_batch_group(row)

    def load_batch_m2_folder(self):
        super().load_batch_m2_folder()
        self.btn_batch_m2_dir.setText("I. 選擇 Basler 主資料夾（含位置子資料夾）")
        self.lbl_batch_m2_info.setText(
            self.lbl_batch_m2_info.text().replace("M2", "Basler")
        )

    def load_batch_group(self, idx):
        super().load_batch_group(idx)
        if hasattr(self, "lbl_batch_current_files"):
            self.lbl_batch_current_files.setText(
                self.lbl_batch_current_files.text().replace("M2:", "Basler:").replace("M2", "Basler")
            )
        self.lbl_batch_status.setText(
            f"狀態: 已載入第 {idx + 1}/{self.batch_total_count} 組（Basler）"
        )
        self._update_distance_table(idx)
