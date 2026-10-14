"""从仓库根目录运行：python -m visual_plan"""

import sys

from PyQt6.QtWidgets import QApplication

from visual_plan.window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("视觉学习")
    window = MainWindow()
    window.resize(960, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
