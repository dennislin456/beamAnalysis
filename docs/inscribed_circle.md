# M2 Above／Below：門檻 Contour 內切圓演算法

本文說明 DataRay **Batch**／**M2 Batch** 中，以「門檻 contour 最大內切圓」自動抓取 **M2 above**、**M2 below** 的演算法（模式代碼：`m2_inscribed`）。

實作位置：`main/tab_batch.py`（`_fit_inscribed_circle` 等）；`main/tab_batch_m2.py` 繼承同一套內切圓邏輯，僅切分 Y 來源不同。

---

## 1. 用途與輸出

對 **M2 強度矩陣** 依切分線分成上下兩區，各區以強度門檻形成二值區域，取**最大連通區**後以歐氏距離變換（EDT）求**最大內切圓**：

| 輸出 | 意義 | 儲存欄位 |
|------|------|----------|
| M2 below 圓心 `(cx, cy)` | 切分 Y **以下** 區域的內切圓中心 | `batch_m2_center_point` |
| M2 below 半徑 `r` | 同區內切圓半徑（px） | `batch_m2_below_circle_r` |
| M2 above 圓心 `(cx, cy)` | 切分 Y **以上** 區域的內切圓中心 | `batch_m2_above_point` |
| M2 above 半徑 `r` | 同區內切圓半徑（px） | `batch_m2_above_circle_r` |

UI 選項文字：

> 自動抓取 (M2 門檻 contour 內切圓中心，Y 上／下)

設計意圖：中心落在「整塊高強度區域」的幾何意義上（外輪廓可內切的最大圓心），而非單一峰值像素或強度加權質心。

---

## 2. 整體流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 決定切分 Y（split_y）                                      │
│    · Batch：M1 光斑中心的 Y                                   │
│    · M2 Batch：M2 質心 X 縱切 → 雙峰波谷 Y                    │
│      （可選框選 ROI，僅在框內搜尋波谷）                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 依 split_y 切出兩個 ROI（不含切分列本身）                   │
│    · below：matrix[0 : y_i, :]                                │
│    · above：matrix[y_i+1 : H, :]                              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 各 ROI 獨立執行 _fit_inscribed_circle                      │
│    背景扣除 → 門檻 mask → 最大連通區 → EDT → 最大內切圓        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. above 的局部 Y 加回 y_i+1，轉回全圖座標                     │
│    寫入圓心／半徑；可再算 M1→below、above→below 距離            │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 切分 Y 來源差異

| 分頁 | 切分依據 | 說明 |
|------|----------|------|
| **DataRay Batch** | M1 中心 Y（`m1_y`） | 先定位 M1，再以該 Y 切開 M2 |
| **DataRay M2 Batch** | 波谷 Y（`batch_split_y`） | 僅載入 M2；`find_dual_peak_valley_y` 沿質心 X 縱切找雙峰之間的波谷；可啟用框選 ROI 排除散射 |

切分索引：

```text
y_i = split_y_index(split_y) = int(round(split_y))
```

座標約定（影像列由上而下遞增）：

- **below**：`rows < y_i` → `matrix[:y_i, :]`
- **above**：`rows > y_i` → `matrix[y_i+1:, :]`
- 切分列 `y_i` **不納入**任一侧，避免上下光斑共用邊界列互相污染

對應函式：

- `_find_inscribed_circle_below_y(matrix, y1, ...)`
- `_find_inscribed_circle_above_y(matrix, y1, ...)`（回傳前將局部 `cy` 加上 `y1 + 1`）

---

## 3. 核心：最大內切圓（`_fit_inscribed_circle`）

對單一 ROI（above 或 below 裁切後的子矩陣）執行。

### 3.1 步驟明細

| 步驟 | 運算 | 說明 |
|------|------|------|
| ① 背景估計 | `bg = median(邊界像素)` | `estimate_border_background`：取上下左右邊界中位數 |
| ② 背景扣除 | `work = clip(matrix − bg, 0)` | 抑制均勻偏置 |
| ③ 峰值與門檻 | `peak = max(work)`；`T = peak × (thresh%/100)` | 若未啟用門檻，固定用 `0.5 × peak` |
| ④ 二值化 | `mask = (work ≥ T)` | 高於門檻視為光斑候選 |
| ⑤ 最大連通區 | `ndi.label` → 取像素數最多的標籤 | 忽略雜訊小島，只保留主光斑 |
| ⑥ 距離變換 | `dist = distance_transform_edt(blob)` | 每個前景像素到最近背景的歐氏距離 |
| ⑦ 取最大 | `argmax(dist)` → `(cx, cy)`，`r = dist[cy, cx]` | EDT 最大值即最大內切圓半徑；該點即圓心 |

### 3.2 數學意義

對二值集合 \(B\)（最大連通前景）：

\[
r(x) = \mathrm{dist}(x,\, B^{c})
\]

\[
(c^{*},\, r^{*}) = \arg\max_{x \in B}\, r(x)
\]

\(c^{*}\) 為最大內切圓圓心，\(r^{*}\) 為半徑。圓完全落在 \(B\) 內，且在所有此類圓中半徑最大。

### 3.3 座標與失敗條件

- 回傳座標為距離變換峰值所在**像素格點**的整數索引，以 `float` 輸出（註解稱為格點中心語意）。
- 下列情況回傳 `None`：空矩陣、峰值無效／≤0、mask 全空、無有效連通區、below 時 `y_i ≤ 0`、above 時 `y_i ≥ H−1`、裁切後 ROI 為空。

### 3.4 偽代碼

```text
function fit_inscribed_circle(matrix, use_threshold, thresh_percent):
    bg ← median(border pixels of matrix)
    work ← max(matrix − bg, 0)
    peak ← max(work)
    if peak ≤ 0: return None
    T ← peak * (thresh_percent/100)   if use_threshold
        peak * 0.5                    otherwise
    mask ← (work ≥ T)
    blob ← largest_connected_component(mask)
    dist ← EDT(blob)
    (cy, cx) ← argmax(dist)
    r ← dist[cy, cx]
    return (cx, cy, r)
```

---

## 4. M2 Batch：波谷切分（補充）

僅 **M2 Batch** 使用；Batch（M1+M2）以 M1 Y 切分，不走此路徑。

`find_dual_peak_valley_y`（`shared_components.py`）摘要：

1. **可選框選 ROI** `roi=(x, y, width, height)`：僅在矩形內估質心與縱切（用來排除上方散射雜點）。回傳的 `valley_y`／`cx`／`peak_ys` 皆為**全圖座標**。
2. 若未指定 `cx`，以（ROI 或全圖）質心（門檻 50%、背景扣除、最大連通）估 `cx`。
3. 取 `cx ± col_half_width`（預設 ±2）欄位平均成垂直 profile。
4. 扣除 profile 中位數後做 1D 平滑（預設窗長 7）。
5. `find_peaks` 找雙峰；取 prominence（或高度）最高的兩峰。
6. 兩峰之間取最小值為波谷；平坦底以容差內候選的中位數當波谷 Y。
7. 失敗時回退搜尋區中線。

得到的 `valley_y` 即作為上述 `split_y`，再對**全圖** above／below 各跑一次內切圓（框選只影響切分 Y 的判斷，不裁切後續定位）。

### 4.1 UI：波谷搜尋框選

| 控制項 | 說明 |
|--------|------|
| `chk_batch_valley_roi` | 啟用後才用框選；關閉則等同整張圖搜尋（舊行為） |
| `spin_batch_valley_roi_x / y` | 框左上角（px） |
| `spin_batch_valley_roi_w / h` | 框寬／高（px） |

熱圖在啟用時以**橘色虛線**畫出實際裁切後的 ROI；紫色切分線／縱切線會限縮畫在框內。

建議：框住上下兩顆主光斑，避免把上方散射光點納入縱切 profile。

---

## 5. UI 參數

| 參數 | UI 控制 | 作用 |
|------|---------|------|
| 啟用門檻 | `chk_batch_p2_use_threshold` | `True`：用下方百分比；`False`：固定 50% of peak |
| 門檻百分比 | `spin_batch_p2_thresh_percent` | \(T = peak \times \%/100\)（相對扣除背景後的峰值） |
| 波谷框選 | 見 §4.1 | 僅影響 `split_y`／縱切 `cx` 的搜尋範圍 |

門檻愈高 → mask 愈小 → 內切圓通常愈小、圓心愈靠近高強度核心。  
門檻愈低 → 外輪廓愈大，但較易併入背景雜訊或另一側殘影（故仍靠切分 Y 與最大連通區隔離）。

---

## 6. 與其他定位模式的差異

同一套 above／below 切分也可搭配其他模式；內切圓特有處如下：

| 模式 | 中心定義 | 是否輸出半徑 |
|------|----------|--------------|
| `m2_inscribed` | 最大連通 mask 的最大內切圓圓心 | 是 |
| `m2_thresh_geom` | 門檻 mask（含背景扣除／最大 CC）像素幾何平均 | 否 |
| `m2_centroid` | 同 mask 上強度加權質心 | 否 |
| `auto_min` | 區域內最小值位置（均值若多點） | 否 |
| `manual` | 使用者點擊（above 仍常自動算） | 否 |

內切圓適合「希望圓心落在可被外輪廓包圍的最大實心圓」的場合；質心則較受強度分布偏斜影響。

---

## 7. 視覺化與匯出

- 熱圖可畫 M2-below（藍）、M2-above（紅）十字，以及對應內切圓（有半徑時）。
- CSV／Excel 相關欄位示例：
  - `m2_below_x_px` / `m2_below_y_px`
  - `m2_above_x_px` / `m2_above_y_px`
  - `m2_below_inscribed_r_px` / `m2_above_inscribed_r_px`
  - above→below 的 ΔX／ΔY／總距離（px 與 μm，DataRay pitch = 5.5 μm／px）

---

## 8. 已知限制與注意事項

1. **內部空洞（壞點）**  
   若光斑內部少數像素低於門檻，會在 mask 形成封閉空洞。EDT 會把空洞當背景，內切圓可能被「卡住」而半徑偏小、圓心偏移。目前實作**未**對 blob 做填洞（`binary_fill_holes`）；若遇此現象，可暫調高／低門檻觀察，或後續再加填洞步驟。

2. **非圓形光斑**  
   內切圓圓心不一定等於質心或外接圓中心；細長或 L 形區域時，圓心會偏向「最寬」處。

3. **切分錯誤**  
   波谷或 M1 Y 切錯時，上下 ROI 可能只含半顆光斑或兩顆擠在同一側，導致內切圓失敗或落點異常。M2 Batch 可啟用**波谷搜尋框選**，把縱切限制在主光斑附近，減少散射雜點干擾雙峰判斷。

4. **尺寸一致性（Batch）**  
   M1 與 M2 矩陣 shape 需一致，否則自動抓取會警告並中止該次定位。

5. **最大連通區假設**  
   同一 ROI 內若有兩個相近強度的大斑，只會保留面積最大者。

---

## 9. 程式對照

| 函式／位置 | 檔案 | 職責 |
|------------|------|------|
| `_fit_inscribed_circle` | `tab_batch.py` | 單 ROI 內切圓核心 |
| `_find_inscribed_circle_below_y` | `tab_batch.py` | 切 below 並擬合 |
| `_find_inscribed_circle_above_y` | `tab_batch.py` | 切 above、擬合、Y 座標還原 |
| `_compute_m2_above_point` | `tab_batch.py` / `tab_batch_m2.py` | 依模式算 above；inscribed 時寫半徑 |
| `update_batch_calculations` | 同上 | 串起切分 → below → above |
| `estimate_border_background` | `shared_components.py` | 邊界中位數背景 |
| `split_y_index` | `shared_components.py` | 亞像素 Y → 整數列 |
| `find_dual_peak_valley_y` | `shared_components.py` | M2 Batch 波谷切分（可選 ROI） |
| `clip_roi_to_matrix` | `shared_components.py` | 框選裁切到矩陣範圍 |

---

## 10. 建議驗證步驟

1. 選 `m2_inscribed`，門檻約 50%，確認上下各出現一圓且圓心大致在光斑核心。
2. 調整門檻 ±10～20%，觀察半徑與圓心是否平滑變化。
3. 對有內部淺色壞點的樣張，確認目前是否出現圓心被拉偏；若有，記錄為已知限制或後續填洞改善項目。
4. 匯出檢查 `m2_*_inscribed_r_px` 與畫面圓弧是否一致。
