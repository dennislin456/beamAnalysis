# 多項式 Refine（下峰亞像素）

對齊程式：`analysis/quadratic_peaks.py`（UI 選「多項式擬合」、代碼 `quadratic`）

只用在**條帶波峰**路徑：raw 條帶加總先抓到整數下峰後，再對下峰附近的 window 做多項式擬合，得到 float Y。不改寫 2D 熱圖；Y 加總圖可畫出擬合曲線與峰點。

與 S-G／Gaussian refine 的差別：本方法**不先平滑**，直接對 window 內的 raw 加總做 `polyfit`。

---

## 適用條件

1. 分析模式為條帶波峰（`peak`）。
2. Refine 啟用，方法為「多項式擬合」。
3. 下峰已由 raw 峰距／下峰比／谷深條件找到。

關閉 Refine 或選其他方法時，不走此模組。

---

## 流程

1. 以 raw 下峰 Y 為中心，取奇數長度 `window`（左右各 \(k\) 點 → `window = 2k+1`）。
2. 對 window 內每一點的條帶加總做多項式擬合；次數依半寬：

   | 左右點數 \(k\) | window | 次數 degree |
   | --- | --- | --- |
   | 1 | 3 | 2（二次） |
   | 2 | 5 | 4 |
   | \(k\) | \(2k+1\) | \(2k\) |

   實作：`poly_degree_for_window(n) = 2 * ((n-1)//2)`，且不超過 `n - 1`。

3. 擬合前把局部座標縮放到約 \([-1, 1]\)（減中點、除半寬），減輕大 window 時 Vandermonde 病態／`RankWarning`。
4. 在 window 內找擬合曲線的最大值：對導數求根，並與兩端點比較，取最高者。
5. 還原成剖面座標上的 float Y；一併回傳擬合係數、峰強度，供畫面藍線密採樣。

擬合失敗時退回 window 內 raw／擬合取樣上的一般峰值處理（`finish_refine`）。

---

## 主要介面

| 名稱 | 說明 |
| --- | --- |
| `refine_lower_peak(profile, crest_y, window=7)` | 入口；回傳 `RefinedLowerPeak` 或 `None` |
| `poly_degree_for_window(n)` | 由點數決定次數 |
| `eval_poly_local(coeff, local_x, x_mid=…, x_scale=…)` | 用歸一化基底係數在局部 index 上求值 |

共用型別與抽窗：`analysis/peak_refine_common.py`（`RefinedLowerPeak`、`extract_window` 等）。

呼叫鏈：`analysis/spots/peak.py` → `quadratic_peaks.refine_lower_peak`；結果放在 spots 的 `savgol` meta（歷史鍵名，三種 refine 共用）。

---

## 參數（UI）

| 參數 | 作用 |
| --- | --- |
| window | 下峰附近奇數窗寬；決定 \(k\) 與次數 \(2k\) |
| order | 多項式擬合時不用（S-G 才用） |
| sigma | 多項式擬合時不用（Gaussian 才用） |

Y 加總圖會標 window 範圍；啟用時以藍色密曲線畫擬合結果，峰點對齊亞像素 Y。

---

## 注意

- 次數隨 window 變大而升高，點數剛好可定出 degree \(2k\)；大窗較易貼近雜訊，必要時縮小 window。
- 峰位是擬合曲線在窗內的**最大值**，不是強制落在整數格點。
- 座標歸一化只影響擬合數值穩定；輸出 Y 仍是剖面列座標。

---

## 相關檔案

| 檔案 | 角色 |
| --- | --- |
| `analysis/quadratic_peaks.py` | 本方法 |
| `analysis/savgol_peaks.py` | S-G refine |
| `analysis/gaussian_peaks.py` | Gaussian 平滑後找峰 |
| `analysis/peak_refine_common.py` | 共用抽窗／結果型別 |
| `analysis/spots/peak.py` | 條帶波峰＋呼叫 refine |
| `docs/dual_spot.md` | 分析區整體說明 |
| `docs/spot_finding.md` | 四種抓法摘要 |

| 日期 | 變更 |
| --- | --- |
| 2026-11-02 | 初版：整理多項式 refine 次數規則、歸一化與找峰流程。 |
