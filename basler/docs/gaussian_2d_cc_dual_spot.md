# 高斯濾波 2D 連通區（雙光斑）

實作入口：`shared_components.find_dual_spots_gaussian`  
UI：`tab_batch_basler_dual` → 演算法選「高斯濾波」→「Apply」  
匯出 id：`gaussian_2d_cc`

設計目標：在長曝光上先做 **2D 高斯平滑**；**上光斑直接沿用短曝光質心／門檻結果**，再**抹掉主斑鄰域**後，於下方軸向帶以相對峰門檻＋最大連通區加權質心抓下光斑。不接一維投影／微分管線。

> 文件對齊程式現況：**2026-09-22**（上光斑＝短曝光，不再於長曝光重抓）  
> 同模組其他演算法：[`centroid_threshold_dual_spot.md`](centroid_threshold_dual_spot.md)、[`axial_1d_projection_dual_spot.md`](axial_1d_projection_dual_spot.md)

## UI 操作摘要

1. **選擇資料夾**：檔名編號奇數＝短、偶數＝長
2. 點選清單一列載入該組短／長檔
3. 演算法分段鈕選「高斯濾波」
4. 調整短曝光門檻％、σ、下斑相對峰、搜尋半寬 X、最小分離 Y、主斑抹除半徑
5. 按「Apply」

分析矩陣同樣可先套 color scale（`< black_below → 0` 等）。長曝光圖上橘虛線框為下光斑搜尋帶。

---

## 演算法公式

輸入長曝光分析矩陣 `I`，短曝光對應點 `c = (s_x, s_y)`（已對齊到長曝光座標）。

### ① 2D 高斯平滑

```text
G = I * K_σ

K_σ(x,y) ∝ exp( -(x² + y²) / (2σ²) )
```

實作：`scipy.ndimage.gaussian_filter(I, sigma=σ, mode="nearest")`。  
`σ` 愈大，弱下斑散點愈易聚成一塊，但質心解析度下降、距離易系統性偏短。

### ② 上光斑＝短曝光結果

```text
primary = (s_x, s_y)
```

短曝光本身以質心／門檻（`compute_auto_spot_center`）求得；本演算法**不再**於長曝光高斯圖上重算上光斑。

### ③ 主斑鄰域抹除（防黏下緣散射）

以 `primary = (p_x, p_y)`、半徑 `R = primary_blank_r`：

```text
work(x,y) = 0    if  (x - p_x)² + (y - p_y)² ≤ R²
work(x,y) = G(x,y)  otherwise
```

### ④ 下光斑搜尋帶

```text
y_lo = round(p_y) + min_sep_y
x_a  = round(p_x) - search_half_x
x_b  = round(p_x) + search_half_x + 1

region = work[ y_lo : H , x_a : x_b ]
```

UI 顯示的搜尋 ROI：

```text
search_roi_xywh = (x_a, y_lo, x_b - x_a, H - y_lo)
```

### ⑤ 下光斑門檻＋最大連通區質心

```text
bg_s  = median( { region > 0 } )     # 無正值時為 0
w_s   = max(region - bg_s, 0)
P_s   = max(w_s)

mask_s = ( w_s ≥ τ · P_s )           # τ = thresh_frac（UI「下斑相對峰」）
```

最大連通區加權質心 → 加回 `(x_a, y_lo)` 得 `secondary`。

### ⑥ 距離與振幅比

```text
|d|_μm = hypot(x_sec - s_x, y_sec - s_y) · 3.45

amp_ratio ≈ max(G near primary) / P_s   # 參考用
```

失敗時 `secondary = None`，`reason` ∈  
`{ empty, no_room_below, no_secondary_signal, no_cc }`。

---

## 流程總覽

```
短曝光質心／門檻 → primary = (s_x, s_y)     ← 即上光斑
        │
        ▼
長曝光 I
        │
        ▼
① gaussian_filter(σ) → G
        │
        ▼
② 圓盤 R=blank_r（以 primary 為心）抹成 0
        │
        ▼
③ 自 primary 下方 min_sep_y 起、半寬 search_half_x 的帶
        │
        ▼
④ τ·峰門檻 → 最大 CC 加權質心 → secondary
        │
        ▼
⑤ 輸出短→下距離（μm = px × 3.45）
```

---

## 參數速查（與 UI 預設）

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| 短曝光門檻 % | 50 | 決定上光斑（短曝光質心） |
| `sigma`（高斯 σ） | 4.0 px | 平滑尺度；試驗上 σ=4 比 σ=2 穩 |
| `thresh_frac`（下斑相對峰） | 0.35 | 下帶相對於帶內峰的門檻 |
| `search_half_x` | 80 px | 下斑搜尋 X 半寬 |
| `min_sep_y` | 120 px | 上斑下方起搜 Y 偏移 |
| `primary_blank_r` | 90 px | 主斑抹除半徑 |
| 距離換算 | `px × 3.45` | |

### 試驗備註（700 μm 資料、前 100 pair）

| 設定 | dist mean | std | 最大幀間跳 |
| --- | ---: | ---: | ---: |
| σ=4 + blank | ~543 μm | ~1 μm | ~3 μm |
| 同條件一維 SUM | ~591 μm | ~3 μm | ~11 μm |
| σ=2 + blank | ~523 μm | ~26 μm | ~59 μm |
| 無 blank（易黏下緣） | ~190 μm | — | — |

結論：本路徑**幀間很穩**，但 2D 弱斑質心常略偏上，距離相對 SUM 可系統性偏短；`blank_r`／`min_sep_y` 過小會抓到主峰下緣散射。

---

## 適用樣態與注意

- **適合**：需要比一維投影更「黏」的幀間穩定度、弱下斑經平滑後仍成一塊。
- **注意**：σ 與 blank／min_sep 強耦合；調 σ 時應一併檢查是否又黏回主峰暈。
- **與一維投影差異**：無波谷／NEAR·FAR 評分；純 2D 連通區，語意更單純、對軸向剖面多峰較不敏感。
- 雜訊門檻建議與 color scale black（常用 256）一致，避免背景碎斑成假 CC。

---

## 相關函式一覽

| 函式 | 角色 |
| --- | --- |
| `find_dual_spots_gaussian` | 總入口（下光斑） |
| `_centroid_largest_cc_weighted` | 最大連通區加權質心 |
| `compute_auto_spot_center` | 短曝光＝上光斑 |
| `tab_batch_basler_dual.grab_dual_spots_gaussian` | Apply／狀態列 |
| `tab_batch_basler_dual._gaussian_algo_kwargs` | UI → 參數 |

---

## 回傳／匯出

### `find_dual_spots_gaussian` 回傳

| 欄位 | 意義 |
| --- | --- |
| `primary` / `secondary` | `(cx, cy)` 或 `None`；`primary`＝短曝光點 |
| `spots` | 依 Y 排序的兩點 |
| `primary_index` | 上光斑在 `spots` 的索引 |
| `search_roi_xywh` | 下斑搜尋帶 |
| `amp_ratio` | 上／下強度比（參考） |
| `reason` | `ok` 或失敗原因 |

### 匯出 `algorithm_params`

| 鍵 | 意義 |
| --- | --- |
| `short_thresh_percent` | 短曝光門檻％ |
| `gaussian_sigma_px` | σ |
| `secondary_thresh_frac` | τ |
| `search_half_x_px` | X 半寬 |
| `min_sep_y_px` | Y 分離 |
| `primary_blank_r_px` | 抹除半徑 |
| `search_roi_xywh` | 實際搜尋帶（若有） |
