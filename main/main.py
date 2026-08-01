import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

# 匯入獨立分頁
from tab_dataray import DataRayTab
from tab_batch import DataRayBatchTab
from tab_batch_m2 import DataRayBatchM2Tab
from tab_basler import BaslerTab

class ModularMatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataRay & Basler Modular Analyzer v4.0")
        self.setGeometry(100, 100, 1400, 920)
        
        self.initUI()

    def initUI(self):
        # 建立 Tab 容器
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # 實例化分頁
        self.tab_dataray = DataRayTab(self)
        self.tab_batch = DataRayBatchTab(self)
        self.tab_batch_m2 = DataRayBatchM2Tab(self)
        self.tab_basler = BaslerTab(self)
        
        # 將分頁加入 Tab 容器
        self.tabs.addTab(self.tab_dataray, "DataRay (單檔/雙檔)")
        self.tabs.addTab(self.tab_batch, "DataRay (Batch 批量)")
        self.tabs.addTab(self.tab_batch_m2, "DataRay (M2 Batch)")
        self.tabs.addTab(self.tab_basler, "Basler")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # 讓外觀看起來更現代
    window = ModularMatrixApp()
    window.show()
    sys.exit(app.exec_())