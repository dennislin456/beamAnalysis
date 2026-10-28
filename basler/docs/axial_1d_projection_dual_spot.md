# 軸向一維投影＋微分法（雙光斑）

實作入口：`shared_components.find_dual_spots_axial_1d_projection`  
UI：`tab_batch_basler_dual` → 演算法選「一維投影＋微分」→「Apply」  
匯出 id：`axial_1d_projection_diff`

設計目標：在長曝光影像上，以短曝光光斑位置為軸心，框出長條 ROI，用垂直方向的 1D 剖面同時抓出**主峰**與**第二峰**（近距雙峰或遠距弱伴峰）。

> 文件對齊程式現況：**2026-09-22**（⑤ 精修含變更紀錄與歷史表；UI 已另增高斯濾波）。  
> 作圖用流程摘要：[`axial_1d_algo_diagram.md`](axial_1d_algo_diagram.md)  
> 同模組其他演算法：[`centroid_threshold_dual_spot.md`](centroid_threshold_dual_spot.md)、[`gaussian_2d_cc_dual_spot.md`](gaussian_2d_cc_dual_spot.md)

## UI 操作摘要

1. **選擇資料夾**：檔名編號奇數＝短、偶數＝長，清單顯示配對（如 `0001（短）＋ 0002（長）`）
2. 點選清單一列載入該組短／長檔
3. 演算法分段鈕選「一維投影＋微分」，調整條帶 W/H、平滑窗、最小峰距等
4. 按「Apply」

同頁另有「質心／門檻」「高斯濾波」；參數頁隨分段鈕切換。

---

## 演算法公式

以下座標皆在 ROI 子圖 `work` 內；最終輸出再加回 `(x0, y0)` 成全圖座標。

### ROI 中心裁切

以短曝光中心 `(cx, cy)` 為中心，條帶寬高 `W = roi_w`、`H = roi_h`：

```text
x0 = clip(round(cx) - floor(W/2), 0, W_img - 1)
y0 = clip(round(cy) - floor(H/2), 0, H_img - 1)
work = I[y0:y1, x0:x1]
```



### 一維投影（水平加總＋背景）

```text
S(y)       = sum_x work(y, x)
profile(y) = max(S(y) - median(S), 0)
```



### 剖面平滑（移動平均）

窗寬 `k = smooth_win`（須為奇數；若為偶數則 `k ← k + 1`），edge pad 後：

```text
p_smooth(y) = (1/k) * sum_{j=-(k-1)/2 .. (k-1)/2} profile(y + j)
```



### 微分找峰（正→負零交越）

```text
dp[i] ≈ p_smooth[i+1] - p_smooth[i]     （或同等一階差分）

零交越：dp[i] ≥ 0 且 dp[i+1] < 0

亞像素峰位：
  y_peak = i + dp[i] / (dp[i] - dp[i+1])

振幅門檻：
  a(y_peak) ≥ α * max(p_smooth)
  主峰搜尋 α = 0.04；第二峰候選 α = 0.02
```



### 主峰鎖定（短曝光 hint）

```text
y_hint = cy - y0
N      = len(p_smooth)

在 |y - y_hint| ≤ max(24, 0.12·N) 的峰中，
取 |y - y_hint| 最小者（同距取振幅大）；
否則取全域距 hint 最近峰。

記為 y_p（主峰 Y）、A_p = p_smooth(y_p)
```



### 核心寬與排除帶

峰寬 = 強度降至 `f · A_p` 時的左右跨度：

```text
core_w  = max(6, width_at(f = 0.65))
min_sep = max(10, min_peak_distance)
exclude = clip(max(min_sep, 1.05·core_w), 8, floor(N/5))
```



### 第二峰評分（NEAR／FAR）

對每個候選峰 `(y, a)`，若 `|y - y_p| ≥ exclude`：

```text
V        = min(p_smooth) between y_p and y
contrast = (min(a, A_p) - V) / min(a, A_p)
ratio    = |y - y_p| / core_w
d        = 1 / (1 + max(0, |y - y_p| - core_w) / (1.5·core_w))

NEAR (ratio ≤ 2.2):  score = a · contrast² · (1 + 0.3·d)   （contrast < 0.05 捨棄）
FAR  (ratio ≥ 3.8):  score = a · max(contrast, 0.08) · (0.15 + 0.85·a/A_p)
                     （contrast < 0.03 或 a < 0.02·A_p 捨棄）
MID：兩式取高
```

取 `score` 最大者為 `y_s`。

### 第二峰種子外推與不對稱精修窗

```text
sign = sign(y_s - y_p)          （下斑為 +1）

y_s'     = y_s + sign · min(8, max(3, 0.10·|y_s - y_p|))
y_half   = clip(0.40·|y_s - y_p|, 10, 32)
y_toward = clip(0.45·y_half, 6, y_half)     （靠主峰側）
y_away   = clip(1.35·y_half, y_half, 44)    （外側加長）
```

靠主峰側裁到波谷 `y_v`。下斑（`sign ≥ 0`）精修列範圍：

```text
y0_sec = max(floor(y_v) + 1, round(y_s') - y_toward)
y1_sec = min(H, round(y_s') + y_away + 1)
精修列 ∈ [y0_sec, y1_sec)
```

### 亮核 X（`_peak_x_in_y_band`，現行）

在精修 Y 帶 `[y0, y1)` 內：

```text
w(x,y) = max(work - median(band), 0)
mask   = (w ≥ core_frac · max(w))     # 下光斑 core_frac=0.55；主峰 0.45
x_seed = Σ x·w / Σ w                   # 只對 mask 像素；不用縱切 argmax
```

避免右側單點尖峰／藍暈把 X 拉偏。

### 2D 質心（背景扣除＋門檻＋峰心加權；用於主峰 XY、下光斑 Y）

```text
w(x,y) = max(work - median(sub), 0) · exp(-(y-sy)²/σy²) · exp(-(x-xs)²/σx²)
mask   = (w ≥ f · max(w))
         主峰 f_pri = 0.50
         第二峰：近距 ≤0.45、中距 ≥0.52、遠距（sep/core_w≥3.5）≥0.58
         （預設 thresh_frac_secondary=0.55）

(cx, cy)_2d = Σ (x,y)·w / Σ w
```

共軸參考：`x_axis = cx_short - x0`（主峰種子會與亮核折衷；**下光斑 X 不混藍線**）。

```text
# 主峰
p_seed_x = 0.55·亮核X + 0.45·x_axis
cx_pri   = soft_spring(cx_2d, p_seed_x, …)

# 下光斑（2026-09-21 起）
cx_sec = x_seed                         # 直接用亮核質心，不再用 2D 質心 X
α      = clip(0.35 + 0.40·tanh((sep/core_w - 1.5)/2), 0.30, 0.80)
cy_sec = α·y_s' + (1-α)·cy_2d
```

若 `|cy_sec - cy_pri| < max(8, 0.55·core_w)`，則退回 `(x_seed, y_s')`。

> 較早的「窗中心混軸／軟彈簧拉回 axis」寫法見下方 **⑤ 2D 精修 → 變更紀錄**。

### 全圖輸出

```text
spot_pri = (x0 + cx_pri, y0 + cy_pri)
spot_sec = (x0 + cx_sec, y0 + cy_sec)
```

依 Y 排序；`primary_index` 為較接近短曝光中心者。

---



## 流程總覽

```
短曝光中心 (cx, cy)
        │
        ▼
① 以 (cx,cy) 為中心裁長條 ROI（預設 W=160 × H=512）
        │
        ▼
② 水平加總 → 1D 垂直剖面 profile[y]，扣列中值背景
        │
        ▼
③ 平滑 + 微分零交越找峰 → 以短曝光 Y 為 hint 鎖定主峰 primary_y
        │
        ▼
④ 排除主峰「核心寬」後，**只往下**（Y 較大）依對比×近距權重選第二峰 secondary_y
        │
        ▼
⑤ 2D 精修（主峰：窄窗質心；下光斑：亮核 X＋1D/2D 混合 Y）
        │
        ▼
⑥ 輸出兩點 (cx,cy)，依 Y 排序，並標 primary_index
```

---



## ① ROI

- 函式：`_roi_centered_on_xy` → `clip_roi_to_matrix`
- 預設：`roi_w=160`，`roi_h=512`
- 中心為短曝光抓到的 `(cx, cy)`（全圖座標）
- 裁切後子圖 `work = matrix[y0:y1, x0:x1]`

假設兩光斑大致共軸（X 相近），長條框把側向干涉／第三光斑擋在 ROI 外。

---



## ② 一維投影

對 ROI 每一列做水平加總，再扣背景：

```text
S(y)       = sum_x work(y, x)
profile(y) = max(S(y) - median(S), 0)
```

得到沿 **Y（垂直）** 的強度剖面。

---



## ③ 微分找峰與主峰鎖定



### 平滑

`_smooth_1d_profile`：奇數窗移動平均（預設 `smooth_win=7`），邊界用 edge pad。

### 微分峰

`_peaks_by_derivative_1d`：

1. 對平滑剖面取 `dy = ∇y`
2. 找 **正→負** 零交越：`dy[i] ≥ 0` 且 `dy[i+1] < 0`
3. 亞像素位置：線性插值 `pos = i + dy[i] / (dy[i] - dy[i+1])`
4. 振幅須 ≥ `max(profile) × min_prominence_frac`（主峰搜尋用 `0.04`）
5. 無乾淨零交越時，退回局部極大 `_local_maxima_1d`



### 主峰

`_primary_peak_from_hint`：

- hint = 短曝光 `cy` 轉成 ROI 內座標：`center_xy[1] - y0`
- 優先選距離 hint ≤ `max(24, 0.12 × N)` 的峰（近者優先，同距取強）
- 否則取距離 hint 最近的峰

---



## ④ 第二峰搜尋

函式：`_find_secondary_peak_away_from_primary`

### 選型結論（2026-09-22，700 μm 等測試）

遠距弱下光斑（SUM 剖面約為上光斑的 **~1/100**）時：

| 作法 | 結論 |
| --- | --- |
| **SUM**（波谷後加強平滑＋瓣質心） | **建議採用**：跨幀最穩 |
| Log 剖面定 Y | 峰較明顯，但易跳 |
| 微分零交越定 Y | 峰較銳利，但易跳 |

**正式流程以 SUM 瓣心為準**。微分相關程式（`_peaks_by_derivative_1d`、候選 NEAR／FAR 評分、測試腳本 `try_sum_log_deriv_secondary.py`）**保留不刪**，供主峰鎖定、近距雙峰候選與對照實驗使用。

### 搜尋方向（預設只往下）

影像座標 **Y 向下增大**。預設 `search_down_only=True`：先鎖定上光斑（主峰）後，
第二峰**只在** `Y ≥ primary_y + exclude` 搜尋，避免大平滑（如 win=50）時
誤抓主峰上方肩峰。若需舊行為（上下都搜），傳 `search_down_only=False`。

### 為何不用半高寬（FWHM）當排除帶

近距雙峰時，第二峰肩膀會把主峰的 **0.5×峰頂寬** 撐得很寬（例如 80px），硬排除會把真伴峰（約 40–50px）略過，改抓到遠端弱噪聲。

因此改用 **高閾核心寬**：

- `core_w = width_at(0.65 × 峰頂振幅)`（至少 6px）
- `exclude = clip(max(min_sep, 1.05 × core_w), 8 … N/5)`
- UI／預設 `min_peak_distance = 12`（內部 `min_sep ≥ 10`）



### 候選評分（NEAR／FAR）（微分路徑，保留）

對排除帶外的每個微分峰算 contrast（V＝兩峰間波谷）：

```text
contrast = (min(a, A_p) - V) / min(a, A_p)
ratio    = sep / core_w
```

依 `ratio` 分模式（預設 `k_near=2.2`、`k_far=3.8`）：

```text
NEAR (ratio ≤ k_near):
  contrast < 0.05 → 捨棄
  score = a · contrast² · (1 + 0.3·d)

FAR (ratio ≥ k_far):
  contrast < 0.03 或 a < 0.02·A_p → 捨棄
  score = a · max(contrast, 0.08) · (0.15 + 0.85·a/A_p)

MID：兩式都算，取較高分
d = 1 / (1 + max(0, sep - core_w) / (1.5·core_w))
```

取分數最高者為候選；若與 **SUM 波谷後瓣心**接近、或前兩名分數接近（散斑競爭），改採瓣心以穩定遠距弱峰。

### 後備（SUM 穩定路徑）

無合格微分候選、或判定為散斑多峰時：主峰排除帶後先找波谷，再對波谷以下做加強平滑＋瓣質心（`_secondary_y_after_valley`）。此為遠距弱下光斑的**建議輸出**。

---



## ⑤ 2D 精修

函式：`_refine_primary_secondary_spots`、`_peak_x_in_y_band`

共軸參考 X：`axis_x = center_xy[0] - x0`（短曝光軸；否則用主峰列帶質心 X）。  
**下光斑最終 X 不以藍線／短曝光軸為準**（見現行表與變更紀錄）。

### 現行（2026-09-21）

| 項目 | 主峰 | 第二峰（下光斑） |
| --- | --- | --- |
| Y 窗 | `±0.70×core_w`（≥8） | 靠主峰側裁到波谷，外側加長；種子再略往外側推（約 `0.1×sep`，3–8 px） |
| X 種子 | 亮核 `_peak_x_in_y_band(core_frac=0.45)` 與 `axis` 折衷：`0.55·亮核 + 0.45·axis` | **亮核** `_peak_x_in_y_band(core_frac=0.55)`（原始 ≥55% peak 像素強度質心；**不混 axis**） |
| X 窗 | 種子 ±`max(10, min(22, 0.55·core_w))` | 種子 ±`max(8, min(14, 0.07·W+4))`（較窄，壓藍暈） |
| 質心門檻 | `0.50` | 預設 `0.55`；近距 ≤`0.45`；中距 ≥`0.52`；遠距（`sep/core_w≥3.5`）≥`0.58` |
| 峰心加權 | 有（相對種子） | 有：`σ_y≈0.35·y_half`、`σ_x≈0.35·x_half`（2D 窗主要服務 **Y**） |
| 最終 X | 2D 質心後 `_soft_spring_x` 往主峰種子 | **`cx_sec = x_seed`**（直接採用亮核；2D 質心 X 不用） |
| 最終 Y | 2D 質心 Y | `cy = α·sy + (1−α)·cy_2d`，`α = clip(0.35+0.40·tanh((ratio−1.5)/2), 0.30, 0.80)` |

若精修後 `|Y_sec − Y_pri| < max(8, 0.55×core_w)` → 退回 `(x_seed, sy)`。

### 變更紀錄（舊寫法保留對照）

| 時間 | 變更摘要 |
| --- | --- |
| **2026-09-21** | 下光斑 X 偏右：改 `_peak_x_in_y_band` 為「原始亮核強度質心」（不用縱切 argmax／最強連通區）；最終 **`cx_sec = x_seed`**；提高第二峰門檻、縮窄 X 窗。 |
| **2026-09-20** | 導入 NEAR／FAR 評分；精修加峰心高斯加權與 Y 的 1D／2D 混合；下光斑一度「不強制對齊藍線」、改信自身亮心／質心；主峰改軸向窄窗。中間版本曾用「列帶水平質心／縱切峰值」當 `x_seed`，並以軟彈簧約束 X。 |

#### 歷史版本表（精修曾用過，供對照；非現行）

以下為文件原記載、變更前的精修行為（約 2026-09-20 中後期至 09-21 修正前）：

| 項目 | 主峰 | 第二峰 |
| --- | --- | --- |
| 窗 | Y：`±0.85×core_w`（≥10），全寬 X | 靠主峰側裁到波谷，外側加長；**X 窗中心**＝`0.7·x_seed + 0.3·axis` |
| 種子 | 1D 主峰 | 1D 第二峰再略往外側推（約 `0.1×sep`，3–8 px） |
| `x_seed` | — | 精修 Y 窗列帶水平質心（或縱切峰值鄰域質心） |
| 質心門檻 | `0.45` | 近距約 `0.22`；遠距（`sep/core_w≥3.5`）提高到 ≥`0.32` |
| 峰心加權 | — | 高斯 `σ_y≈0.35·y_half`、`σ_x≈0.45·x_half` |
| Y 混合 | — | `cy = α·sy + (1−α)·cy_2d`，`α = clip(0.30+0.40·tanh((ratio−1.5)/2), 0.25, 0.75)` |
| X 約束 | — | **軟彈簧** `_soft_spring_x`（小偏移保留，大偏移漸進拉回；不再硬鎖 axis） |

若精修後 `|Y_sec − Y_pri| < max(8, 0.55×core_w)` → 退回 `(x_seed, sy)`（不退回 axis）。

> 為何改掉歷史版下光斑 X：弱下斑右側散射／單點尖峰會把「縱切 argmax」或「低門檻 2D 質心」拉右；亮核質心（≥55% peak）較貼視覺綠心，且最終不再把 X 混回短曝光軸。

---



## ⑥ 輸出


| 欄位                        | 意義                            |
| ------------------------- | ----------------------------- |
| `roi_xywh` / `roi_bounds` | 實際 ROI                        |
| `profile`                 | ROI 內 1D 剖面                   |
| `spots`                   | `[(cx,cy), (cx,cy)]`，依 Y 由小到大 |
| `primary_index`           | 較接近短曝光中心的那一點（0 或 1）           |
| `peak_ys`                 | 兩峰 Y（全圖座標）                    |


失敗時 `spots` / `peak_ys` 為 `None`（仍可能帶回 ROI 與 profile）。

---



## 參數速查（與 UI 預設）


| 參數                  | 預設        | 說明                   |
| ------------------- | --------- | -------------------- |
| `roi_w` × `roi_h`   | 160 × 512 | 長條框                  |
| `smooth_win`        | 7         | 剖面平滑窗                |
| `min_peak_distance` | 12        | 第二峰最小間距下限（與核心寬取 max） |
| 主峰 prominence       | 0.04      | 相對全域峰頂               |
| 第二峰 prominence      | 0.02      | 較寬鬆，利於弱伴峰            |
| 核心寬分數               | 0.65      | 排除帶／精修窗依據            |
| `search_down_only`  | True      | 第二峰只搜主峰下方            |
| 主峰質心門檻              | 0.50      | 精修 2D                |
| 第二峰質心門檻             | 0.55      | 精修；近／遠距再調            |
| 下光斑亮核 `core_frac`  | 0.55      | `_peak_x_in_y_band`  |


---



## 適用樣態與注意

- **遠距弱第二峰**：排除核心後仍可憑振幅×對比入選；FAR 評分偏振幅比。
- **近距雙峰**：核心寬避免 FWHM 膨脹；NEAR 評分偏 contrast。
- **側向干涉／第三斑**：靠窄 ROI；下光斑 X 另以高門檻亮核質心抑制右／左拖尾。
- **下光斑與藍線不必共軸**：現行不把下光斑 X 拉回短曝光軸；兩斑 X 差大時屬預期（以各斑亮心為準）。
- 本路徑**不**做內切圓 fit；圓心語意見 `docs/inscribed_circle.md`（DataRay M2 內切圓模式）。

---



## 相關函式一覽


| 函式                                                | 角色            |
| ------------------------------------------------- | ------------- |
| `find_dual_spots_axial_1d_projection`             | 總入口           |
| `_peaks_by_derivative_1d`                         | 平滑＋微分找峰       |
| `_primary_peak_from_hint`                         | 短曝光錨定主峰       |
| `_find_secondary_peak_away_from_primary`          | 第二峰（含 search_down_only） |
| `_score_secondary_candidate`                      | NEAR／FAR 評分    |
| `_peak_width_at_frac_1d`                          | 峰寬（含核心寬／FWHM） |
| `_peak_x_in_y_band`                               | 列帶亮核 X 質心     |
| `_refine_primary_secondary_spots`                 | 2D 精修         |
| `_centroid_near_axis` / `_centroid_2d_in_y_range` | 門檻 2D 質心      |
| `_soft_spring_x`                                  | 主峰 X 軟約束     |


