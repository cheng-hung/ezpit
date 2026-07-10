import sys
import os

# [중요] pyqtgraph가 PySide6를 사용하도록 강제 설정
os.environ["QT_API"] = "pyside6"

from PySide6.QtWidgets import QApplication
from ui.main_window import PDFApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PDFApp()
    window.show()
    sys.exit(app.exec())
