# 批次距離

對齊程式：`script/hdr_batch_distance.py`、`script/hdr_group_match.py`（2026-10-29）

不開 `hdr_test.py` 的視窗。第一支依膝關檔名分點位，做和測試視窗相同的 HDR 融合與雙光斑距離，寫成 CSV。第二支讀這份 CSV，在 index 不重複的條件下，找出最接近預期平均與標準差的分組。

依賴與測試視窗相同，見 `HDR/requirements.txt`。兩支都可以在任何目錄執行，路徑用引號包住含空白的資料夾。

```text
python HDR/script/hdr_batch_distance.py "D:/data/for dennis"
python HDR/script/hdr_group_match.py HDR/data/for dennis/hdr_out/distance_5000us_100000us.csv --group-size 2 --targets targets.csv
```

---

## 資料夾

選一個裡面有輪次資料夾的目錄。輪次資料夾名稱是 `L01`、`L02`……。影像直接放在輪次資料夾裡，檔名要能认出點位、輪次、cycle、曝光和 count，例如：

```text
Basler_20261029_005206_L01_C01_X17p00_Y99p60_Zoff_R1_359p180_R2_329p260_S3_100000us_010_20261029_010950.npy
```

| 檔名片段 | 意義 |
| --- | --- |
| `L01` | 輪次。以資料夾名稱為準 |
| `C01` | cycle |
| `X17p00` `Y99p60` | 點位 X 17.00、Y 99.60。`p` 是小數點 |
| `R1_359p180` `R2_329p260` | 膝關角度 |
| `100000us` | 曝光時間 |
| `010` | count，長曝光的序號 |

同一點位要有要融合的兩種曝光。短曝光通常只有 `001` 一張，長曝光有 `001`、`002`……。檔名含獨立一段 `merged` 的融合成品會略過。認不出上述格式的檔案也略過。

只選到 `L01` 這一個資料夾時，那一層就當成一個輪次。輪次若再放進一層 cycle 資料夾（`C01/L01`），也會被找到。

---

## 1. 融合並輸出距離

```text
python HDR/script/hdr_batch_distance.py "D:/data/for dennis"
```

未加 `--pair` 時各寫一份：

- `5000us + 100000us`
- `500us + 100000us`

輸出在輸入資料夾下的 `hdr_out/`：

| 檔案 | 內容 |
| --- | --- |
| `distance_5000us_100000us.csv` | 這一組曝光的距離表 |
| `distance_5000us_100000us.params.json` | 這次用的演算法、模糊、條帶和點位欄名 |
| `distance_5000us_100000us.warnings.txt` | 只有失敗或找不到光斑時才有 |

另一組曝光的檔名是 `distance_500us_100000us.csv`。`--output` 可改輸出資料夾。

### CSV

`count` 在最左欄，接著輪次與 cycle，每個點位一欄。點位欄名固定含 X、Y、R1、R2。距離是像素，不乘相機像素尺寸（不乘 3.45 μm）。找不到下光斑的格子留白。

```text
count,loop,cycle,X16.00_Y123.80_R1_359.310_R2_341.320,X17.00_Y99.60_R1_359.180_R2_329.260
1,L01,C01,137,156
2,L01,C01,137,157
1,L02,C01,...
```

列的順序是 cycle、輪次、count。count 用張數較多那一側的檔名序號，由上往下排。同一輪次裡，張數較少的曝光用到最後一張後就固定重用，和測試視窗的批次融合相同。所以 500us 只有一張時，count 1 到 10 都跟這張 500us 融合，長曝光則各用自己的那一張。

同一個 X/Y/R1/R2 若還有不同的 Z，欄名後面會再加 Z，避免兩欄同名。同一 count 來自不同拍攝時段時，會多一欄 `session`。

### 預設

和這次要跑的測試視窗設定相同。

| 項目 | 預設 |
| --- | --- |
| 演算法 | Mertens |
| 對比 / 飽和 / 曝光 | 1 / 0 / 1 |
| 高斯模糊 | 開。水平 80、垂直 80、sigma 2。模糊在融合前，每一張曝光各做一次 |
| 滿量 | 4095 |
| 條帶寬 | 250 |
| 上下峰最小距離 | 110 px |
| 下峰比 | 10% |
| 上門檻 | 50% |
| 演算法（光斑） | 條帶波峰（`peak`） |

條帶波峰的距離是兩個峰的像素差。下門檻在這個模式不使用。

### 可改的參數

```text
python HDR/script/hdr_batch_distance.py "D:/data/for dennis" --pair 5000 100000 --jobs 4
```

| 參數 | 作用 |
| --- | --- |
| `folder` | 含 `L01`、`L02` 的資料夾，或單一輪次資料夾 |
| `--output` | 輸出資料夾。未指定時是輸入資料夾下的 `hdr_out` |
| `--pair US_A US_B` | 一組曝光，單位 us，可重複。未指定時跑上面兩組 |
| `--algorithm` | `mertens`、`mertens_local`、`debevec`、`robertson` |
| `--contrast` `--saturation` `--exposure-weight` | Mertens 三個權重 |
| `--local-sigma` | 只給 Mertens 自寫。預設 0.2 |
| `--full-scale` | 非 8-bit 的滿量。預設 4095 |
| `--strip-width` | 條帶寬。預設 250 |
| `--peak-distance` | 上下峰最小距離，像素。預設 110 |
| `--peak-ratio` | 下峰比，百分比。預設 10 |
| `--thresh` | 上門檻，百分比。預設 50 |
| `--lower-thresh` | 下門檻，百分比。預設 50。條帶波峰不使用 |
| `--center-mode` | `peak`、`shape`、`intensity`、`line`、`slant_iter` |
| `--blur-x` `--blur-y` `--sigma` | 高斯核寬與 sigma |
| `--no-blur` | 融合前不做高斯模糊 |
| `--jobs` | 同時算幾個點位。預設 1 |

64 個點位、每個點位上百筆時，可把 `--jobs` 調到 4 或 8。終端機會印出目前點位，以及成功筆數。

---

## 2. 分組，對預期平均與標準差

對哪一份融合結果做分組，就把那一份距離 CSV 交給第二支。兩組曝光各自跑一次。

目標寫在 CSV。一列一組，十組就寫十列。`group` 決定組的順序；沒有這欄時，檔案由上往下就是第 1 組到第 10 組。平均和標準差的單位跟距離 CSV 相同，是像素。

```text
group,mean,std
1,156,0.5
2,157,0.4
```

欄名也可以用 `平均`、`預期平均`、`標準差`、`預期標準差`。每個點位的目標不一樣時，加上 `point`（或 `點位`），內容要和距離 CSV 的欄名完全相同：

```text
point,group,mean,std
X17.00_Y99.60_R1_359.180_R2_329.260,1,156,0.5
X17.00_Y99.60_R1_359.180_R2_329.260,2,157,0.4
```

沒有 `point` 欄時，同一份目標套到每一個點位。

```text
python HDR/script/hdr_group_match.py distance_5000us_100000us.csv --group-size 10 --targets targets.csv
```

`--group-size` 是同一個點位取幾筆當成一組。組數等於目標列數。每個點位需要的筆數是「組數 × 每組筆數」。空白格子不算。資料不夠的點位會在終端機說明並略過，其他點位繼續。

少量組也可以不建檔，直接重複 `--target`：

```text
python HDR/script/hdr_group_match.py distance_500us_100000us.csv --group-size 2 --target 138,0.5 --target 137,1.2
```

`--point` 可重複，只處理列出的點位欄名。

### index

index 是距離 CSV 的資料列號，從 1 起，不含標題列。它不是 `count`。`L01` 和 `L02` 的 count 都會從 1 再數，列號才不會撞。同一個點位裡，一個 index 只進一個組。某個點位該列是空白時，這個 index 不會被那個點位用到。

### 結果

寫在距離 CSV 旁邊。`--output` 可改摘要檔的路徑，明細檔名會在主檔名後加 `_long`。

`distance_5000us_100000us_groups.csv` 每個點位、每一組一列：

| 欄 | 內容 |
| --- | --- |
| `point` `group` | 點位與第幾組 |
| `indices` | 用到的列號，以 `;` 分開 |
| `n` `mean` `std` | 筆數、平均、樣本標準差 |
| `target_mean` `target_std` | 這一組的預期值 |
| `mean_error` `std_error` | 實際值減預期值 |
| `cost` | 這一組的誤差。`權重 × (誤差)²`，平均和標準差各自一項，預設權重都是 1 |
| `point_cost` | 這個點位所有組的 cost 加總 |
| `method` | `hungarian`、`exact` 或 `local` |

樣本標準差和 HDR 分析視窗相同，用 n−1。只有 1 筆時標準差是 0。

`distance_5000us_100000us_groups_long.csv` 每個被選中的 index 一列，並帶出該列的 `count`、`loop`、`cycle` 和距離。

### 搜尋

| 情況 | 做法 |
| --- | --- |
| 每組 1 筆 | 匈牙利演算法，是這一組權重下的最佳解。`method` 為 `hungarian` |
| 只有一組，而且組合數不超過 20 萬 | 逐一列舉。`method` 為 `exact` |
| 其餘 | 多起點，先依平均挑接近的，再交換 index。`method` 為 `local` |

`local` 在組合很大時不保證是唯一的全域最佳。`--restarts` 預設 12，`--seed` 預設 1，同一份 CSV 和目標會得到同一組結果。`--mean-weight` 與 `--std-weight` 分別乘在平均誤差的平方和標準差誤差的平方上。

```text
python HDR/script/hdr_group_match.py --self-test
```

這行只檢查分組程式本身，不讀距離 CSV。印出「自我測試通過」即可。

---

## 修改紀錄

| 日期 | 修改 |
| --- | --- |
| 2026-10-29 | 新增批次融合距離與分組的使用說明。距離為像素，點位欄名含 X/Y/R1/R2。 |
