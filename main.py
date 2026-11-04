import sys

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("English Companion")
        self.resize(900, 650)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("English Companion · Sprint 1"))
        layout.addWidget(QLabel("Готов к работе"))

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())