# Basler 雙光斑演算法文件索引

`tab_batch_basler_dual` 分析工具目前提供三種演算法，**各一份說明**（含公式／參數／匯出 id）：

| UI 名稱 | 文件 | 匯出 id | 入口函式 |
| --- | --- | --- | --- |
| 質心／門檻 | [`centroid_threshold_dual_spot.md`](centroid_threshold_dual_spot.md) | `centroid_threshold_geom` | `compute_auto_spot_center` |
| 一維投影＋微分 | [`axial_1d_projection_dual_spot.md`](axial_1d_projection_dual_spot.md)（技術／公式）<br>[`axial_1d_projection_customer_overview.md`](axial_1d_projection_customer_overview.md)（客戶報告版） | `axial_1d_projection_diff` | `find_dual_spots_axial_1d_projection` |
| 高斯濾波 | [`gaussian_2d_cc_dual_spot.md`](gaussian_2d_cc_dual_spot.md) | `gaussian_2d_cc` | `find_dual_spots_gaussian` |

一維投影流程圖（作圖用）：[`axial_1d_algo_diagram.md`](axial_1d_algo_diagram.md)

距離換算（三種共通）：`|d|_μm = |d|_px × 3.45`。
