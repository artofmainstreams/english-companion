from PySide6.QtCore import QSignalBlocker

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from .session_controller import SessionController


class MainWindow(QMainWindow):
    def __init__(self, controller: SessionController) -> None:
        super().__init__()

        self.setWindowTitle("English Companion")
        self.resize(900, 650)

        self.controller = controller

        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setPlaceholderText("Здесь появится разговор")

        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText(
            "Напиши сообщение на английском"
        )

        self.send_button = QPushButton("Отправить")
        self.status_label = QLabel("Готов")

        self.model_combo = QComboBox()
        self.model_combo.setPlaceholderText("Нет доступных моделей")
        self.model_combo.setMinimumWidth(250)
        self.model_combo.setEnabled(False)

        self.refresh_button = QPushButton("Обновить")

        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.textChanged.connect(self.update_send_button)
        self.model_combo.currentIndexChanged.connect(
            self.select_model
        )
        self.refresh_button.clicked.connect(controller.load_models)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.addWidget(self.history, 1)
        layout.addLayout(input_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(model_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        controller.state_changed.connect(self.render_state)
        controller.message_added.connect(
            lambda message: self.history.appendPlainText(f"You: {message}")
        )
        self.render_state()

    def render_state(self) -> None:
        controller = self.controller
        with QSignalBlocker(self.model_combo):
            self.model_combo.clear()
            self.model_combo.addItems(controller.models)
            self.model_combo.setCurrentIndex(
                self.model_combo.findText(controller.selected_model)
            )
        self.model_combo.setEnabled(bool(controller.models) and not controller.loading)
        self.refresh_button.setEnabled(not controller.loading)
        self.status_label.setText(controller.status)
        self.update_send_button()

    def select_model(self) -> None:
        self.controller.select_model(self.model_combo.currentText())

    def update_send_button(self) -> None:
        self.send_button.setEnabled(
            bool(self.message_input.text().strip())
            and self.controller.can_send
        )

    def send_message(self) -> None:
        if self.controller.send_message(self.message_input.text()):
            self.message_input.clear()
            self.message_input.setFocus()
