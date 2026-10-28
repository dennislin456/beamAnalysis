# 上下光斑抓法

此檔由 `script/update_spot_finding_doc.py` **自動產生**。改四種抓法的模組 docstring（含 `DOC_TITLE` / `DOC_SUMMARY`）後執行：

```text
python script/update_spot_finding_doc.py
```

對齊程式：`analysis/spots/`（2026-10-28）

共用流程在 `analysis/spots/common.py`：整張圖上門檻質心決定條帶 X → 條帶加總找下峰與波谷（距離、下峰比、谷深 65%）→ 各模式決定上下座標（下光斑 blob 用下門檻）。

---

## 共用流程

1. 在整張圖用門檻百分比抓最強連通核心的質心，得到上光斑種子 xy，並以此 X 擺直條帶。
2. 條帶內每一列沿 X 加總（不扣中位數）。
3. 由種子 Y（或條帶波峰模式的加總最高峰）往下找通過「距離／下峰比／谷深」的下峰；兩峰間最低點當波谷。
4. 再依所選演算法決定上、下光斑座標（見下列四節）。

---

## 形狀中心

- 模組：`analysis/spots/shape.py`
- 代碼：`shape`

上光斑用門檻核心質心擺條帶；下光斑是波谷以下最亮連通區的幾何中心（每像素權重相同）。下光斑門檻可與上光斑分開設定。

波谷之後：
1. Upper spot = threshold-core center above the valley (same seed used for strip X).
2. Lower spot = unweighted mean of pixels in the brightest connected blob below the valley
   (uses lower_thresh_percent vs local peak below the valley).

---

## 強度質心

- 模組：`analysis/spots/intensity.py`
- 代碼：`intensity`

上光斑是門檻核心的強度加權質心；下光斑是波谷以下最亮連通區的強度質心。下光斑門檻可與上光斑分開設定。

波谷之後：
1. Upper spot = intensity-weighted centroid of the threshold core above the valley.
2. Lower spot = intensity-weighted centroid of the brightest connected blob below the valley
   (uses lower_thresh_percent vs local peak below the valley).

---

## 條帶波峰

- 模組：`analysis/spots/peak.py`
- 代碼：`peak`

只看條帶 Y 加總的兩個峰；X 固定在條帶中心（上光斑種子 X），距離用兩個 Y 的差。

波谷之後：
1. 上峰 Y：波谷以上（沒有波谷則整段條帶）加總曲線最高峰。
2. 下峰 Y：通過距離／下峰比／谷深的那個下峰。
3. 兩邊 X 都固定為條帶中心（門檻種子 X），不再取各列最亮點。

---

## 顯示用斜剖面

- 模組：`analysis/spots/line.py`
- 代碼：`line`

抓點方式與強度質心相同；Y 剖面／加總改沿上下光斑連線做 X 線性內插，只影響顯示。

波谷之後：
1. Upper / lower coordinates match intensity-centroid mode（下光斑用獨立門檻）。
2. 回傳的剖面是穿過兩光斑的斜線取樣（每列 X 線性內插），供顯示用 Y 剖面／加總。

---

## 斜向迭代（測試）

- 模組：`analysis/spots/slant_iter.py`
- 代碼：`slant_iter`

先用直條帶強度質心抓兩點，再沿連線做斜條帶、重找波谷與質心。直條帶已包住兩團、ΔX 小時結果幾乎不變；下光斑被直條帶切邊時才會明顯移動。若第一次沒抓到下光斑，會先用加寬直條帶找種子再斜向重抓。

波谷之後：
1. 第一次：與強度質心相同（垂直條帶）。
2. 若只有上光斑：用加寬直條帶暫找下光斑種子（仍可能失敗）。
3. 若上下都有：沿兩點連線做斜條帶加總，重找下峰／波谷；在斜條帶遮罩內重算強度質心（寬度用使用者設定的「寬」）。
