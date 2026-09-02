"""依檔名座標與 JSON dist_ifc 建立 Mapping 熱圖。"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QGraphicsRectItem
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtCore import Qt, QRectF

from tab_mapping import MappingTab, MappingRoiWindow
from shared_components import (
    export_stamped_path,
    sanitize_numeric_values,
    export_plot_image,
    EXPORT_IMAGE_EXT,
    EXPORT_IMAGE_FILTER,
    normalize_export_image_path,
)

try:
    import orjson as _fast_json
except ImportError:  # pragma: no cover
    _fast_json = None


_COORD_RE = re.compile(
    r"X(?P<x_sign>-?)(?P<x_int>\d+(?:p\d+)?)_Y(?P<y_sign>-?)(?P<y_int>\d+(?:p\d+)?)",
    re.IGNORECASE,
)

# 大量 JSON 時用執行緒吃磁碟／解析；上限避免開太多 thread
_IMPORT_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 2))


def _load_json_obj(file_path):
    """優先 orjson（若已安裝），否則標準 json。"""
    if _fast_json is not None:
        with open(file_path, "rb") as fh:
            return _fast_json.loads(fh.read())
    with open(file_path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _read_one_mapping_json(file_path):
    """單一檔讀取（給 thread pool 用）。"""
    try:
        data = _load_json_obj(file_path)
        value = data.get("value", {})
        if not isinstance(value, dict):
            value = {}
        dist = value.get("dist_ifc", data.get("dist_ifc"))
        unit = value.get("dist_ifc_unit", data.get("dist_ifc_unit", "")) or ""
        dist_value = float(sanitize_numeric_values([dist])[0])
        if not np.isfinite(dist_value):
            raise ValueError(f"JSON 沒有有效的 value.dist_ifc：{os.path.basename(file_path)}")
        x_value = data.get("x_mm")
        y_value = data.get("y_mm")
        if x_value is None or y_value is None:
            raise ValueError(f"JSON 沒有有效的 x_mm/y_mm：{os.path.basename(file_path)}")
        return file_path, dist_value, str(unit), float(x_value), float(y_value), None
    except Exception as exc:
        return file_path, None, None, None, None, exc


class MappingFilenameTab(MappingTab):
    """從檔名 X/Y 與 JSON value.dist_ifc 建立平均 Mapping。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_mapping_mode.hide()
        self.btn_load_mapping.setText("I. 匯入 JSON 資料夾／檔案")
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
        _path, dist_value, unit, x_value, y_value, err = _read_one_mapping_json(file_path)
        if err is not None:
            raise err
        return dist_value, unit, x_value, y_value

    def _collect_json_paths(self):
        """優先選資料夾（適合上萬筆）；取消後可改多選檔案。"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇 JSON 資料夾（取消則改為多選檔案）",
            "",
        )
        if folder:
            paths = [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.lower().endswith(".json")
            ]
            paths.sort()
            return paths

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇檔案 Mapping JSON（可多選）",
            "",
            "JSON 檔 (*.json);;所有檔案 (*)",
        )
        return paths

    def _ingest_paths_parallel(self, paths):
        """平行讀取 JSON，回傳 (groups, units, skipped, source_points)。"""
        groups = {}
        units = set()
        skipped = []
        path_coords = {}
        total = len(paths)
        done = 0
        workers = min(_IMPORT_WORKERS, max(1, total))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_read_one_mapping_json, path) for path in paths]
            for future in as_completed(futures):
                path, distance, unit, x_mm, y_mm, err = future.result()
                done += 1
                if done == 1 or done == total or done % 500 == 0:
                    self.lbl_status.setText(f"狀態: 匯入中 {done}/{total} …")
                    QApplication.processEvents()
                if err is not None:
                    skipped.append(f"{os.path.basename(path)}: {err}")
                    continue
                path_coords[path] = (x_mm, y_mm)
                groups.setdefault((x_mm, y_mm), []).append(distance)
                if unit:
                    units.add(unit)

        source_points = []
        seen = set()
        for path in paths:
            key = path_coords.get(path)
            if key is None or key in seen:
                continue
            seen.add(key)
            source_points.append([key[0], key[1]])
        source_arr = np.asarray(source_points, dtype=float) if source_points else None
        return groups, units, skipped, source_arr

    def _std_for_point(self, ix, iy):
        """取得目前 X/Y 點所有原始 dist_ifc 的母體標準差。"""
        if self.x_coords is None or self.y_coords is None:
            return float("nan")
        x = float(self.x_coords[ix])
        y = float(self.y_coords[iy])
        values = self._filename_groups.get((x, y), [])
        if not values:
            return float("nan")
        return float(np.std(values, ddof=0))

    def _update_point_info_with_std(self, ix, iy, prefix):
        value = self.mapping_matrix[iy, ix]
        value_text = "NaN" if np.isnan(value) else f"{value:.6f}"
        values = self._filename_groups.get(
            (float(self.x_coords[ix]), float(self.y_coords[iy])), []
        )
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        min_text = f"{np.min(finite_values):.6f}" if finite_values.size else "NaN"
        max_text = f"{np.max(finite_values):.6f}" if finite_values.size else "NaN"
        std = self._std_for_point(ix, iy)
        std_text = "NaN" if not np.isfinite(std) else f"{std:.6f}"
        self.lbl_mouse_info.setText(
            f"{prefix}: X={self.x_coords[ix]:.3f} mm, "
            f"Y={self.y_coords[iy]:.3f} mm, Value={value_text}, "
            f"Min={min_text}, Max={max_text}, STD={std_text}"
        )

    def _build_image_rect(self, matrix):
        """單列／單欄時父類 dy/dx 會變成 0，影像無法對到 mm，點擊與紅框都會失效。"""
        if self.x_coords is None or self.y_coords is None or matrix is None:
            return None
        dx = float(self._grid_spacing(self.x_coords))
        dy = float(self._grid_spacing(self.y_coords))
        if not np.isfinite(dx) or dx <= 0:
            dx = 1.0
        if not np.isfinite(dy) or dy <= 0:
            dy = 1.0
        x0 = float(np.min(self.x_coords)) - dx / 2.0
        y0 = float(np.min(self.y_coords)) - dy / 2.0
        return QRectF(x0, y0, dx * matrix.shape[1], dy * matrix.shape[0])

    def _cell_rect_bounds(self, ix, iy):
        """回傳與 heatmap 色塊對齊的格子邊界（畫面座標）。"""
        rect = self._build_image_rect(self.mapping_matrix)
        ny, nx = self.mapping_matrix.shape
        if rect is None or nx <= 0 or ny <= 0:
            x0, x1 = MappingRoiWindow._cell_bounds(self.x_coords, ix)
            y0, y1 = MappingRoiWindow._cell_bounds(self.y_coords, iy)
            return x0, x1, y0, y1
        dx = rect.width() / nx
        dy = rect.height() / ny
        x0 = rect.left() + ix * dx
        y0 = rect.top() + iy * dy
        return float(x0), float(x0 + dx), float(y0), float(y0 + dy)

    def _cell_index_from_view(self, x, y):
        """依點擊落在哪個色塊來選格，而不是用 mm 最近點（1 列資料時會永遠選到第一點）。"""
        ny, nx = self.mapping_matrix.shape
        rect = self._build_image_rect(self.mapping_matrix)
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            ix = int(np.abs(self.x_coords - x).argmin())
            iy = int(np.abs(self.y_coords - y).argmin())
            return ix, iy
        ix = int(np.floor((float(x) - rect.left()) / (rect.width() / nx)))
        iy = int(np.floor((float(y) - rect.top()) / (rect.height() / ny)))
        return int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))

    @staticmethod
    def _disable_overlay_mouse(item):
        """選取框不得攔截滑鼠，否則無法再點其他格子。"""
        targets = [item]
        for name in ("curve", "scatter"):
            sub = getattr(item, name, None)
            if sub is not None:
                targets.append(sub)
        for target in targets:
            try:
                target.setAcceptedMouseButtons(Qt.NoButton)
            except Exception:
                pass
            if hasattr(target, "setClickable"):
                try:
                    target.setClickable(False)
                except Exception:
                    pass

    def _draw_selected_point_box(self, ix, iy):
        self._clear_selected_point()
        x0, x1, y0, y1 = self._cell_rect_bounds(ix, iy)
        item = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        pen = QPen(QColor("#FF0000"))
        pen.setWidth(3)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setZValue(1000)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self.plot.addItem(item, ignoreBounds=True)
        self.selected_point_item = item

    def _apply_selected_cell(self, ix, iy):
        self.selected_point = (ix, iy)
        self._update_point_info_with_std(ix, iy, "已固定")
        self._draw_selected_point_box(ix, iy)

    def _on_heatmap_mouse_moved(self, mouse_point):
        if self.mapping_matrix is None or self.x_coords is None or self.y_coords is None:
            super()._on_heatmap_mouse_moved(mouse_point)
            return
        if self.selected_point is not None:
            ix, iy = self.selected_point
            self._update_point_info_with_std(ix, iy, "已固定")
            return
        if mouse_point is None:
            self.lbl_mouse_info.setText("滑鼠位置: X=--, Y=--, Value=--")
            return
        ix, iy = self._cell_index_from_view(mouse_point.x(), mouse_point.y())
        self._update_point_info_with_std(ix, iy, "滑鼠位置")

    def _on_heatmap_clicked(self, event):
        if self.mapping_matrix is None or not self.plot_drawn:
            return
        # 雙擊交給 InteractiveHeatmapPanel 復原視野
        if event.double():
            return
        button = event.button()
        if button not in (Qt.LeftButton, Qt.NoButton):
            return
        pos = event.scenePos()
        try:
            if self.heatmap_panel.hist.sceneBoundingRect().contains(pos):
                return
        except Exception:
            pass
        if not self.plot.sceneBoundingRect().contains(pos):
            return

        point = self.plot.getViewBox().mapSceneToView(pos)
        ix, iy = self._cell_index_from_view(point.x(), point.y())
        self._apply_selected_cell(ix, iy)

    def load_mapping_file(self):
        paths = self._collect_json_paths()
        if not paths:
            return

        self.btn_load_mapping.setEnabled(False)
        self.lbl_status.setText(f"狀態: 開始平行匯入 {len(paths)} 筆 JSON …")
        QApplication.processEvents()
        try:
            groups, units, skipped, source_points = self._ingest_paths_parallel(paths)

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
            self._source_points = source_points
            self.mapping_f1_path = paths[0]
            self.mapping_path = paths[0]
            self.lbl_mapping_path.setText(
                f"已匯入 {len(paths)} 筆 JSON｜平均後 {len(groups)} 個 X/Y 點"
            )
            unit_text = ", ".join(sorted(units)) if units else "未提供單位"
            self.lbl_status.setText(
                f"狀態: 已載入並平均 dist_ifc（單位: {unit_text}），請點「畫圖」"
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
            self._update_point_count_label(raw_files=len(paths))
        except Exception as exc:
            QMessageBox.critical(self, "匯入失敗", f"無法讀取檔案：\n{exc}")
            self.mapping_matrix = None
            self.mapping_matrix_f1 = None
            self.x_coords = None
            self.y_coords = None
            self._source_points = None
            self.btn_plot_mapping.setEnabled(False)
            self.btn_export_mapping.setEnabled(False)
            self._update_point_count_label()
        finally:
            self.btn_load_mapping.setEnabled(True)

    def _update_point_count_label(self, raw_files=None):
        """顯示平均後 X/Y 點數；若有提供則附上匯入 JSON 筆數。"""
        n_points = len(self._filename_groups) if self._filename_groups else 0
        if n_points <= 0:
            super()._update_point_count_label()
            return
        if raw_files is not None:
            self.lbl_point_count.setText(
                f"總點數: {n_points} 個 X/Y（匯入 {raw_files} 筆 JSON）"
            )
        else:
            sample_n = sum(len(v) for v in self._filename_groups.values())
            self.lbl_point_count.setText(
                f"總點數: {n_points} 個 X/Y（樣本 {sample_n}）"
            )

    def _update_roi_overlay(self):
        super()._update_roi_overlay()
        if self.roi_rect_item is not None:
            try:
                self.roi_rect_item.setZValue(500)
            except Exception:
                pass
            self._disable_overlay_mouse(self.roi_rect_item)

    def plot_mapping(self):
        if self.mapping_matrix_f1 is None:
            QMessageBox.warning(self, "提醒", "請先匯入 JSON 檔案。")
            return
        self.selected_point = None
        self._clear_selected_point()
        self.mapping_matrix = self.mapping_matrix_f1.copy()
        super().plot_mapping()
        try:
            self.image_item.setZValue(0)
        except Exception:
            pass
        self.heatmap_panel.set_plot_title("Filename Mapping (average dist_ifc)")
        self.contour_panel.set_plot_title("Filename Mapping Contour")
        unit_text = ", ".join(sorted(self._filename_units)) if self._filename_units else ""
        self.lbl_status.setText(
            f"狀態: 已繪製平均 Mapping｜{len(self._filename_groups)} 個點"
            + (f"｜單位: {unit_text}" if unit_text else "")
        )
        self._update_point_count_label()

    def _point_stats(self, x, y, mean_value):
        """回傳單一 X/Y 點的 mean / min / max / std（母體）。"""
        values = np.asarray(self._filename_groups.get((float(x), float(y)), []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            mean_text = "nan" if not np.isfinite(mean_value) else f"{mean_value:.6f}"
            return mean_text, "nan", "nan", "nan", 0
        mean_text = f"{float(np.mean(values)):.6f}"
        min_text = f"{float(np.min(values)):.6f}"
        max_text = f"{float(np.max(values)):.6f}"
        std_text = f"{float(np.std(values, ddof=0)):.6f}"
        return mean_text, min_text, max_text, std_text, int(values.size)

    def export_mapping(self):
        """匯出圖檔，並在 CSV 附上各點 mean / min / max / std。"""
        if self.mapping_matrix is None:
            QMessageBox.warning(self, "提醒", "請先匯入並繪製 Mapping。")
            return
        if not self.export_dir:
            QMessageBox.warning(self, "提醒", "請先選擇儲存資料夾。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出 Heatmap 檔名",
            export_stamped_path(self.export_dir, "mapping_heatmap"),
            EXPORT_IMAGE_FILTER,
        )
        if not save_path:
            return

        avg_matrix = self._get_processed_matrix()
        selected_item = self.selected_point_item
        try:
            heatmap_path = normalize_export_image_path(save_path)
            base_name = os.path.splitext(os.path.basename(heatmap_path))[0]
            heatmap_base = os.path.join(self.export_dir, base_name)
            contour_path = os.path.join(
                self.export_dir, f"{base_name}_contour{EXPORT_IMAGE_EXT}"
            )

            if selected_item is not None:
                selected_item.hide()
            try:
                exported_images = list(
                    self.heatmap_panel.export_heatmap_only(heatmap_base)
                )
                restore_contour = self.contour_panel._prepare_clean_bundle_export()
                try:
                    QApplication.processEvents()
                    export_plot_image(
                        self.contour_panel.plot,
                        contour_path,
                        width=max(int(self.contour_panel.plot.width()), 400),
                        height=max(int(self.contour_panel.plot.height()), 300),
                    )
                finally:
                    restore_contour()
                exported_images.append(contour_path)
            finally:
                if selected_item is not None:
                    selected_item.show()

            csv_path = os.path.join(self.export_dir, f"{base_name}.csv")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("x_rel_mm,y_rel_mm,value,min,max,std,count\n")
                for x, y, value in self._iter_export_mapping_points(avg_matrix):
                    mean_text, min_text, max_text, std_text, count = self._point_stats(
                        x, y, value
                    )
                    fh.write(
                        f"{x:.6f},{y:.6f},{mean_text},{min_text},"
                        f"{max_text},{std_text},{count}\n"
                    )

            self.lbl_status.setText(
                "狀態: 已匯出 heatmap、contour 與含 min/max/std 的 CSV"
            )
            image_list = "\n".join(exported_images)
            QMessageBox.information(
                self,
                "匯出完成",
                f"已匯出：\n{image_list}\n\nCSV：\n{csv_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", f"無法匯出檔案：\n{exc}")
        finally:
            if selected_item is not None:
                selected_item.show()
