# 質心／門檻幾何（雙光斑）

實作入口：`shared_components.compute_auto_spot_center`  
UI 包裝：`tab_batch_basler_dual._find_spot` → `grab_short_spot` / `grab_long_spot2`  
匯出 id：`centroid_threshold_geom`

設計目標：在短曝光（全圖）與長曝光（可選搜尋 ROI）上，各自以**相對峰門檻＋最大連通區＋強度加權質心**抓一個光斑中心；兩點差距即 UI 顯示的雙光斑距離。

> 文件對齊程式現況：**2026-09-22**  
> 同模組其他演算法：[`axial_1d_projection_dual_spot.md`](axial_1d_projection_dual_spot.md)、[`gaussian_2d_cc_dual_spot.md`](gaussian_2d_cc_dual_spot.md)

## UI 操作摘要

1. **選擇資料夾**：檔名編號奇數＝短、偶數＝長
2. 點選清單一列載入該組短／長檔
3. 演算法分段鈕選「質心／門檻」
4. 調整短／長曝光門檻％；可選啟用長曝光搜尋 ROI（X/Y/W/H）
5. 按「Apply」（或勾批次分析）

分析矩陣會先套用各側 color scale（若啟用）：`< black_below → 0`，可選 `> white_above → white`。

---

## 演算法公式

以下以搜尋區矩陣 `I`（短曝光為全圖；長曝光可為 ROI 子圖）說明。ROI 時質心再加回偏移 `(ox, oy)`。

### 邊界背景估計

```text
border = concat( I[0,:], I[-1,:], I[1:-1,0], I[1:-1,-1] )
bg     = median(border)
```

（影像高或寬 `< 2` 時改為 `bg = median(I)`。）

### 背景扣除

```text
W(x,y) = max( I(x,y) - bg, 0 )
```

### 相對峰門檻 mask

門檻％為 `T`（UI：短／長曝光門檻 %）：

```text
P_max = max(W)
frac  = T / 100
mask  = ( W ≥ frac · P_max )
```

若 `mask` 全空，退回 `argmax(W)` 的整數峰位。

### 最大連通區

對 `mask` 做 4／8 鄰接標籤（`scipy.ndimage.label`）：

```text
取像素數最多的標籤 ℓ*
mask ← (label == ℓ*)
```

（質心路徑不強制填洞；門檻 overlay／Beam size 用的 `build_robust_threshold_mask` 會再 `fill_holes`。）

### 強度加權質心（mode = centroid, power = 1）

```text
(cx, cy) = ( Σ_{mask} x · W(x,y)  ,  Σ_{mask} y · W(x,y) )
           / Σ_{mask} W(x,y)
```

`power = p ≠ 1` 時權重改為 `W^p`（本 UI 固定 `p = 1`）。

座標裁切到影像範圍後以亞像素 `float` 輸出。

### 長曝光 ROI

若啟用搜尋範圍 `(x, y, w, h)`：

```text
work = I[y0:y1, x0:x1]     # 半開區間，已 clip 到影像
(cx', cy') = centroid(work)
(cx, cy)   = (cx' + x0, cy' + y0)
```

### 雙光斑距離

```text
Δx = x_long - x_short
Δy = y_long - y_short
|d|_px = hypot(Δx, Δy)
|d|_μm = |d|_px · 3.45
```

---

## 流程總覽

```
短曝光分析矩陣
        │
        ▼
① 邊界背景 → 扣除 → 門檻％ → 最大連通區 → 加權質心  → 短曝光光斑
        │
        ▼
長曝光分析矩陣（可選 ROI）
        │
        ▼
② 同上流程 → 長曝光「第二」光斑（此路徑不區分主／伴峰語意，只抓 ROI／全圖內最亮核）
        │
        ▼
③ 輸出 Δ 與 μm
```

---

## 參數速查（與 UI 預設）

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| 短曝光門檻 % | 50 | 相對扣除背景後峰頂 |
| 長曝光門檻 % | 50 | 同上 |
| 長曝光搜尋 ROI | 開（預設約 200×200，載入後可拉滿） | 限縮第二點搜尋 |
| `bg_subtract` | True | 邊界中位背景 |
| `largest_cc_only` | True | 只留最大連通區 |
| `power` | 1 | 線性強度權重 |
| 距離換算 | `px × 3.45` | Basler 像素間距 μm |

---

## 適用樣態與注意

- **適合**：單亮核／短曝光清晰、長曝光第二斑也夠亮且與主斑可被 ROI 分開。
- **不適合**：長曝光上主斑遠亮於伴斑且兩者落入同一 ROI——質心會被主斑主導，易抓不到弱下斑。
- 此路徑**沒有**「主峰排除帶／波谷」語意；弱伴峰請改用一維投影或高斯濾波。
- Beam size 最大內切圓另見 repo 根目錄 `docs/inscribed_circle.md`；門檻 mask 與本演算法共用 `build_robust_threshold_mask`。

---

## 相關函式一覽

| 函式 | 角色 |
| --- | --- |
| `compute_auto_spot_center` | 質心／門檻幾何核心 |
| `estimate_border_background` | 邊界中位背景 |
| `build_robust_threshold_mask` | 門檻 overlay／Beam size mask |
| `clip_roi_to_matrix` | ROI 裁切 |
| `tab_batch_basler_dual._find_spot` | UI：ROI 偏移＋呼叫質心 |

---

## 匯出欄位（algorithm_params）

| 鍵 | 意義 |
| --- | --- |
| `short_thresh_percent` | 短曝光門檻％ |
| `long_thresh_percent` | 長曝光門檻％ |
| `long_search_roi_enabled` | 是否啟用長曝光 ROI |
| `long_search_roi_xywh` | `(x, y, w, h)` |
