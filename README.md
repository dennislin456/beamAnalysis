# BeamAnalysis

DataRay & Basler 光束／光斑模組化分析工具（v4.0）。

以 PyQt5 桌面介面整合 **DataRay 光束輪廓儀** 與 **Basler 相機** 的數據讀取、熱圖視覺化、光斑定位、距離量測與結果匯出，適用於光學實驗與產線量測後處理。

---

## 功能概覽

應用程式採分頁架構，進入後可見四個工作區：

| 分頁 | 說明 |
|------|------|
| **DataRay（單檔／雙檔）** | 讀取 DataRay Excel（`.xlsx` / `.xls`），支援單檔熱圖、雙檔運算（峰值校正相減、純相減、純相除）、參數檔重繪、光斑抓取、十字波形、等高線與結果匯出 |
| **DataRay（Batch 批量）** | 依資料夾批次配對 M1／M2 Excel，逐組運算與檢視，可暫存各組參數，並匯出 ZIP |
| **DataRay（M2 Batch）** | 以 M2 為主的批量流程；支援切分 Y 上下光斑自動定位（含門檻 contour 內切圓） |
| **Basler** | 讀取 BMP／PNG／JPG／CSV，光斑自動／手動定位、正圓／橢圓外框、門檻寬度、兩點距離量測（像素 pitch = 3.45 μm），匯出 JSON／Excel／PNG |

### 共通能力

- 熱圖（Jet colormap）與 Histogram LUT 色階調整
- 光斑中心：最高值幾何中心、質心、門檻區域幾何中心、手動點擊；Batch／M2 Batch 另支援 **門檻 contour 內切圓中心**
- 十字標記、剖面波形（線性／Log）
- 匯出：JSON 參數、Excel（含圖表）、PNG 截圖；Batch 另支援 ZIP

### 像素換算（內建常數）

| 來源 | Pixel pitch |
|------|-------------|
| DataRay | 5.5 μm／px |
| Basler | 3.45 μm／px |

---

## 專案結構

```
BeamAnalysis/
├── README.md
├── requirements.txt
├── docs/
│   └── inscribed_circle.md   # M2 above／below 內切圓演算法說明
└── main/
    ├── main.py                 # 程式入口（主視窗＋分頁）
    ├── shared_components.py    # 共用 UI 與光斑定位演算法
    ├── tab_dataray.py          # DataRay 單檔／雙檔分頁
    ├── tab_batch.py            # DataRay 批量分頁（含內切圓實作）
    ├── tab_batch_m2.py         # DataRay M2 Batch 分頁
    └── tab_basler.py           # Basler 分頁
```

---

## 文件

| 文件 | 內容 |
|------|------|
| [README.md](README.md) | 安裝、啟動、使用概要（本檔） |
| [docs/inscribed_circle.md](docs/inscribed_circle.md) | M2 above／below 門檻 contour 內切圓演算法、切分流程、參數與已知限制 |

---

## 環境需求

- **作業系統**：Windows 10／11（開發與測試環境）
- **Python**：建議 **3.12**（以 conda 環境為準）
- **套件管理**：Conda（Anaconda／Miniconda）
- **顯示**：需可運行 Qt GUI（一般桌面環境即可）

### 需安裝的 Python 套件

| 套件 | 用途 |
|------|------|
| `PyQt5` | 桌面 GUI |
| `numpy` | 矩陣運算 |
| `pandas` | 讀取 Excel／CSV |
| `pyqtgraph` | 熱圖、波形、等高線繪圖與匯出 |
| `scipy` | 平滑濾波、連通標註、距離變換、峰值偵測 |
| `openpyxl` | Excel 寫入與圖表 |
| `Pillow` | Basler 影像（BMP／PNG／JPG）讀取 |

標準函式庫（`os`、`json`、`glob`、`zipfile` 等）無需額外安裝。

---

## 安裝步驟

於專案根目錄開啟終端機（Anaconda Prompt 或已初始化 conda 的 PowerShell）：

```powershell
# 建立並啟用 conda 環境（Python 3.12）
conda create -n beam python=3.12
conda activate beam

# 安裝依賴
pip install -r requirements.txt
```

之後每次使用前，請先執行 `conda activate beam`。

---

## 啟動方式

```powershell
conda activate beam
cd main
python main.py
```

視窗標題：**DataRay & Basler Modular Analyzer v4.0**。

---

## 使用說明（簡要）

### DataRay（單檔／雙檔）

1. 選擇工作模式（單檔／雙檔運算／參數重繪）。
2. 匯入 Excel（DataRay 匯出格式，程式會略過前 4 列標頭後讀取數值矩陣）。
3. 執行運算／畫圖，調整光斑與量測選項。
4. 可開啟 M1／M2 Heatmap、十字波形視窗；完成後匯出 JSON／Excel／PNG。

### DataRay（Batch）

1. 分別選擇 M1、M2 資料夾（內含依檔名自然排序的 `.xlsx`／`.xls`）。
2. 選擇運算模式後「開始批量載入與運算」。
3. 用 ◀／▶ 切換資料組，必要時暫存各組參數。
4. M2 第二點可選：門檻幾何中心、質心、或 **門檻 contour 內切圓中心（Y 上／下）**。
5. 一鍵匯出 ZIP 至選定儲存資料夾。

### DataRay（M2 Batch）

1. 選擇 M2 相關資料夾並執行批量載入。
2. 以切分 Y 區分上下光斑；第二點模式同 Batch（含內切圓）。
3. 檢視、暫存參數後匯出結果。

內切圓演算法細節見 [docs/inscribed_circle.md](docs/inscribed_circle.md)。

### Basler

1. 選擇「單獨匯入畫圖」或「圖像＋JSON 參數重繪」。
2. 匯入影像或 CSV（CSV 會略過前 25 列後讀取）。
3. 設定中心模式、正圓／橢圓、門檻比例後分析。
4. 匯出 JSON 參數、Excel 與光斑分析圖。

---

## 支援的輸入格式

| 分頁 | 輸入 |
|------|------|
| DataRay | `.xlsx`、`.xls`（略過前 4 列） |
| DataRay Batch／M2 Batch | 資料夾內多個 `.xlsx`／`.xls` |
| Basler | `.bmp`、`.png`、`.jpg`／`.jpeg`、`.csv`（略過前 25 列）；可選 `.json` 參數檔 |

---

## 授權與注意事項

- 本工具為內部／實驗用途之分析介面；硬體驅動與即時取像不在本程式範圍內（僅處理已匯出之檔案）。
- 像素 pitch 為程式內建常數，若感測器規格不同，需自行修改對應程式碼中的 `pixel_pitch_um`。
