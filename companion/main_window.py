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
from .audio_controller import AudioController
from .message import Message


class MainWindow(QMainWindow):
    def __init__(
        self,
        controller: SessionController,
        audio: AudioController,
    ) -> None:
        super().__init__()

        self.setWindowTitle("English Companion")
        self.resize(900, 650)

        self.controller = controller
        self.audio = audio

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

        self.record_button = QPushButton("Записать")
        self.play_button = QPushButton("Прослушать")
        self.audio_status_label = QLabel("Голос: готов")

        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.textChanged.connect(self.update_send_button)
        self.model_combo.currentIndexChanged.connect(
            self.select_model
        )
        self.refresh_button.clicked.connect(controller.load_models)
        self.record_button.clicked.connect(self.toggle_record)
        self.play_button.clicked.connect(self.toggle_play)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(self.record_button)
        audio_layout.addWidget(self.play_button)
        audio_layout.addWidget(self.audio_status_label, 1)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.addWidget(self.history, 1)
        layout.addLayout(input_layout)
        layout.addLayout(audio_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(model_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        controller.state_changed.connect(self.render_state)
        controller.message_added.connect(self.append_message)
        audio.state_changed.connect(self.render_audio)
        self.render_state()
        self.render_audio()

    def render_state(self) -> None:
        controller = self.controller
        with QSignalBlocker(self.model_combo):
            self.model_combo.clear()
            self.model_combo.addItems(controller.models)
            self.model_combo.setCurrentIndex(
                self.model_combo.findText(controller.selected_model)
            )
        self.model_combo.setEnabled(bool(controller.models) and not controller.busy)
        self.refresh_button.setEnabled(not controller.busy)
        self.status_label.setText(controller.status)
        self.update_send_button()

    def append_message(self, message: Message) -> None:
        speaker = "You" if message.role == "user" else f"Teacher · {message.model}"
        self.history.appendPlainText(f"{speaker}: {message.content}\n")

    def select_model(self) -> None:
        self.controller.select_model(self.model_combo.currentText())

    def update_send_button(self) -> None:
        self.send_button.setEnabled(
            bool(self.message_input.text().strip())
            and self.controller.can_send
        )

    def render_audio(self) -> None:
        audio = self.audio
        recording = audio.state == audio.RECORDING
        playing = audio.state == audio.PLAYING

        self.record_button.setEnabled(
            audio.can_record or recording
        )
        self.record_button.setText("Остановить" if recording else "Записать")

        self.play_button.setEnabled(audio.can_play or playing)
        self.play_button.setText("Стоп" if playing else "Прослушать")

        self.audio_status_label.setText(audio.status)

    def toggle_record(self) -> None:
        if self.audio.state == self.audio.RECORDING:
            self.audio.stop_recording()
        else:
            self.audio.record()

    def toggle_play(self) -> None:
        if self.audio.state == self.audio.PLAYING:
            self.audio.stop_playback()
        else:
            self.audio.play()

    def send_message(self) -> None:
        if self.controller.send_message(self.message_input.text()):
            self.message_input.clear()
            self.message_input.setFocus()
