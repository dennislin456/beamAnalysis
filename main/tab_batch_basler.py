"""Basler 批量分析分頁：沿用 M2 Batch 流程，pixel pitch 使用 3.45 um/px。"""

from PyQt5.QtWidgets import QLabel

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
