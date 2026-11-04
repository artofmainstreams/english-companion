import sys

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("English Companion")
        self.resize(900, 650)

        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setPlaceholderText("Здесь появится разговор")

        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Напиши сообщение на английском")

        self.send_button = QPushButton("Отправить")
        self.status_label = QLabel("Готов")

        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.textChanged.connect(self.update_send_button)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addWidget(self.history)
        layout.addLayout(input_layout)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_send_button()

    def update_send_button(self) -> None:
        has_text = bool(self.message_input.text().strip())
        self.send_button.setEnabled(has_text)

    def send_message(self) -> None:
        message = self.message_input.text().strip()
        if not message:
            return

        self.history.appendPlainText(f"You: {message}")
        self.message_input.clear()
        self.message_input.setFocus()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())