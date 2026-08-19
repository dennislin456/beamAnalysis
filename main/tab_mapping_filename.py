"""依檔名座標與 JSON dist_ifc 建立 Mapping 熱圖。"""

import json
import os
import re

import numpy as np

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from tab_mapping import MappingTab


_COORD_RE = re.compile(
    r"X(?P<x_sign>-?)(?P<x_int>\d+(?:p\d+)?)_Y(?P<y_sign>-?)(?P<y_int>\d+(?:p\d+)?)",
    re.IGNORECASE,
)


class MappingFilenameTab(MappingTab):
    """從檔名 X/Y 與 JSON value.dist_ifc 建立平均 Mapping。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_mapping_mode.hide()
        self.btn_load_mapping.setText("I. 匯入 JSON 檔案（可多選）")
        self.btn_load_mapping_f2.hide()
        self.lbl_mapping_f2_path.hide()
        self.lbl_mapping_path.setText("未選擇 JSON 檔案")
        self._filename_groups = {}
        self._filename_units = set()

    @staticmethod
    def _parse_coordinate(file_path):
        name = os.path.basename(file_path)
        match = _COORD_RE.search(name)
        if match is None:
            raise ValueError(
                f"檔名找不到 X/Y 座標：{name}\n"
                "預期格式例如 X20p5_Y24p5"
            )

        def parse_part(sign, value):
            number = float(value.replace("p", ".").replace("P", "."))
            return -number if sign == "-" else number

        return (
            parse_part(match.group("x_sign"), match.group("x_int")),
            parse_part(match.group("y_sign"), match.group("y_int")),
        )

    @staticmethod
    def _read_dist_ifc(file_path):
        with open(file_path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        value = data.get("value", {})
        if not isinstance(value, dict):
            value = {}
        dist = value.get("dist_ifc", data.get("dist_ifc"))
        unit = value.get("dist_ifc_unit", data.get("dist_ifc_unit", ""))
        try:
            dist_value = float(dist)
        except (TypeError, ValueError):
            dist_value = float("nan")
        if not np.isfinite(dist_value):
            raise ValueError(f"JSON 沒有有效的 value.dist_ifc：{os.path.basename(file_path)}")
        return dist_value, str(unit)

    def load_mapping_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇檔案 Mapping JSON（可多選）",
            "",
            "JSON 檔 (*.json);;所有檔案 (*)",
        )
        if not paths:
            return

        try:
            groups = {}
            units = set()
            skipped = []
            for path in paths:
                try:
                    coordinate = self._parse_coordinate(path)
                    distance, unit = self._read_dist_ifc(path)
                    groups.setdefault(coordinate, []).append(distance)
                    if unit:
                        units.add(unit)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    skipped.append(f"{os.path.basename(path)}: {exc}")

            if not groups:
                detail = "\n".join(skipped[:8])
                raise ValueError(f"沒有可繪製的 X/Y 資料。\n{detail}")

            x_coords = np.array(sorted({point[0] for point in groups}), dtype=float)
            y_coords = np.array(sorted({point[1] for point in groups}), dtype=float)
            matrix = np.full((y_coords.size, x_coords.size), np.nan, dtype=float)
            for (x, y), values in groups.items():
                ix = int(np.searchsorted(x_coords, x))
                iy = int(np.searchsorted(y_coords, y))
                matrix[iy, ix] = float(np.mean(values))

            self._filename_groups = groups
            self._filename_units = units
            self.mapping_matrix_f1 = matrix
            self.mapping_matrix_f2 = None
            self.mapping_matrix = None
            self.x_coords = x_coords
            self.y_coords = y_coords
            self.mapping_f1_path = paths[0]
            self.mapping_path = paths[0]
            self.lbl_mapping_path.setText(
                f"已匯入 {len(paths)} 筆 JSON｜平均後 {len(groups)} 個 X/Y 點"
            )
            unit_text = ", ".join(sorted(units)) if units else "未提供單位"
            self.lbl_status.setText(
                f"狀態: 已載入檔名座標並平均 dist_ifc（單位: {unit_text}），請點「畫圖」"
            )
            if skipped:
                QMessageBox.warning(
                    self,
                    "部分檔案略過",
                    f"成功匯入 {len(paths) - len(skipped)} 筆，略過 {len(skipped)} 筆。\n\n"
                    + "\n".join(skipped[:12])
                    + ("\n..." if len(skipped) > 12 else ""),
                )
            self._update_mapping_import_state()
            self.btn_export_mapping.setEnabled(False)
            self.chk_roi_enabled.setEnabled(True)
            self._set_roi_spin_ranges()
            self.plot_drawn = False
        except Exception as exc:
            QMessageBox.critical(self, "匯入失敗", f"無法讀取檔案：\n{exc}")
            self.mapping_matrix = None
            self.mapping_matrix_f1 = None
            self.x_coords = None
            self.y_coords = None
            self.btn_plot_mapping.setEnabled(False)
            self.btn_export_mapping.setEnabled(False)

    def plot_mapping(self):
        if self.mapping_matrix_f1 is None:
            QMessageBox.warning(self, "提醒", "請先匯入 JSON 檔案。")
            return
        self.mapping_matrix = self.mapping_matrix_f1.copy()
        super().plot_mapping()
        self.heatmap_panel.set_plot_title("Filename Mapping (average dist_ifc)")
        self.contour_panel.set_plot_title("Filename Mapping Contour")
        unit_text = ", ".join(sorted(self._filename_units)) if self._filename_units else ""
        self.lbl_status.setText(
            f"狀態: 已繪製平均 Mapping｜{len(self._filename_groups)} 個點"
            + (f"｜單位: {unit_text}" if unit_text else "")
        )
