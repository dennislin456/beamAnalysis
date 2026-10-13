"""BeamProfiler standalone entry — local modules only (not beamAnalysis/main).

Run from this directory so imports resolve locally:

    python main.py
"""
import os
import sys

# Ensure this package directory is first on sys.path (avoid clashing with ../main).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)
# Drop sibling main/ if present so local modules win.
_MAIN_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "main"))
sys.path[:] = [p for p in sys.path if os.path.normpath(p) != _MAIN_DIR]

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from tab_batch_m2 import DataRayBatchM2Tab
from tab_batch_json import BaslerJsonBatchTab


class BeamProfilerApp(QMainWindow):
    """獨立視窗：DataRay M2 Batch ＋ Json Batch（本資料夾模組）。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeamProfiler")
        self.setGeometry(100, 100, 1400, 920)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_batch_m2 = DataRayBatchM2Tab(self)
        self.tab_batch_json = BaslerJsonBatchTab(self)
        self.tabs.addTab(self.tab_batch_m2, "DataRay (M2 Batch)")
        self.tabs.addTab(self.tab_batch_json, "Json Batch")


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BeamProfilerApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
