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
    QMessageBox,
)


from .session_controller import SessionController
from .audio_controller import AudioController
from .stt_engine import SttEngine
from .message import Message


class MainWindow(QMainWindow):
    def __init__(
        self,
        controller: SessionController,
        audio: AudioController,
        stt: SttEngine,
    ) -> None:
        super().__init__()

        self.setWindowTitle("English Companion")
        self.resize(900, 650)

        self.controller = controller
        self.audio = audio
        self.stt = stt
        # Отложенное закрытие окна: если при попытке закрыть Speech
        # ещё работает, завершаем работу и закрываем её позже.
        self._pending_close = False

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
        self.recognize_button = QPushButton("Распознать")
        self.save_button = QPushButton("Сохранить")
        self.audio_status_label = QLabel("Голос: готов")
        self.stt_status_label = QLabel("Распознавание: готов")

        # Выбор устройства ввода: позиция 0 — «Системный (по умолчанию)»,
        # дальше — доступные микрофоны. Во время записи переключение
        # запрещено (комбо заблокировано, контроллер его тоже отклонит).
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(180)

        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.textChanged.connect(self.update_send_button)
        self.model_combo.currentIndexChanged.connect(
            self.select_model
        )
        self.refresh_button.clicked.connect(controller.load_models)
        self.record_button.clicked.connect(self.toggle_record)
        self.play_button.clicked.connect(self.toggle_play)
        self.recognize_button.clicked.connect(self.start_recognition)
        self.save_button.clicked.connect(self.save_recording)
        self.mic_combo.currentIndexChanged.connect(self._on_mic_changed)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(self.record_button)
        audio_layout.addWidget(self.play_button)
        audio_layout.addWidget(self.recognize_button)
        audio_layout.addWidget(self.save_button)
        audio_layout.addWidget(QLabel("Микрофон:"))
        audio_layout.addWidget(self.mic_combo)
        audio_layout.addWidget(self.audio_status_label, 1)
        audio_layout.addWidget(self.stt_status_label)

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
        audio.devices_changed.connect(self._refresh_mic_combo)
        stt.state_changed.connect(self._on_stt_state_changed)
        stt.recognized.connect(self._on_stt_recognized)
        stt.failed.connect(self._on_stt_failed)
        self.render_state()
        self.render_audio()
        self._refresh_mic_combo()
        self._on_stt_state_changed()

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
        stt_busy = self.stt.is_busy
        recording = audio.state == audio.RECORDING
        playing = audio.state == audio.PLAYING

        # Во время распознавания новую запись запрещаем: файл ещё нужен.
        self.record_button.setEnabled(
            (audio.can_record and not stt_busy) or recording
        )
        self.record_button.setText("Остановить" if recording else "Записать")

        self.play_button.setEnabled(audio.can_play or playing)
        self.play_button.setText("Стоп" if playing else "Прослушать")

        # Сохранить доступна для завершённой записи; после shutdown
        # has_recording становится False (путь сбрасывается).
        self.save_button.setEnabled(audio.has_recording)

        # Переключение микрофона — только вне записи (требование);
        # после shutdown контроллер не работает, комбо блокируем тоже.
        self.mic_combo.setEnabled(
            not audio.is_shutdown and audio.state != audio.RECORDING
        )

        self.audio_status_label.setText(audio.status)
        self._render_recognize_button()

    def _render_recognize_button(self) -> None:
        # Кнопка доступна для завершённой записи и заблокирована, пока
        # идёт распознавание или идёт запись.
        audio = self.audio
        self.recognize_button.setEnabled(
            self.stt.state == self.stt.IDLE
            and audio.has_recording
            and audio.state != audio.RECORDING
        )

    def _refresh_mic_combo(self) -> None:
        """Перечитывает список микрофонов, сохраняя текущий выбор.

        Вызывается при старте окна и при каждом изменении списка
        (подключение/отключение устройства). Сигналы комбо на время
        перестроения блокированы, чтобы «сброс» индекса не сбивал выбор.
        """
        audio = self.audio
        selected = audio.selected_input_name()
        items = ["Системный (по умолчанию)"] + audio.input_device_names()
        with QSignalBlocker(self.mic_combo):
            self.mic_combo.clear()
            self.mic_combo.addItems(items)
            self.mic_combo.setCurrentIndex(
                max(self.mic_combo.findText(selected), 0)
                if selected is not None
                else 0
            )
        self.render_audio()  # статус контроллера + состояние кнопок

    def _on_mic_changed(self) -> None:
        """Позиция 0 — системное устройство; дальше — конкретный микрофон."""
        if self.mic_combo.currentIndex() <= 0:
            self.audio.set_input_device(None)
        else:
            self.audio.set_input_device(self.mic_combo.currentText())

    def _on_stt_state_changed(self) -> None:
        self.stt_status_label.setText(self.stt.status)
        self._render_recognize_button()
        # Кнопка записи тоже зависит от занятости STT.
        self.render_audio()
        # Если закрытие было отложено из-за активной работы — завершаем.
        if self._pending_close and not self.stt.is_busy:
            self._pending_close = False
            self.close()

    def _on_stt_recognized(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            # Пустой результат не должен стирать введённый текст.
            self.stt_status_label.setText(
                "Распознавание не дало текста; введённый текст сохранён."
            )
            self.status_label.setText("Речь не распознана (пустой результат).")
            return

        current = self.message_input.text().strip()
        if current:
            answer = QMessageBox.question(
                self,
                "Заменить текст?",
                "В поле уже есть текст. Заменить его результатом "
                "распознавания?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.stt_status_label.setText(
                    "Текст не изменён (подтверждение отменено)."
                )
                self.status_label.setText(
                    "Распознанный текст проигнорирован; поле не изменено."
                )
                return

        self.message_input.setText(text)
        self.message_input.setFocus()
        self.stt_status_label.setText("Распознавание: готов")
        self.status_label.setText(
            "Текст распознан. Проверь и отправь вручную."
        )

    def _on_stt_failed(self, message: str) -> None:
        self.stt_status_label.setText(message)
        self.status_label.setText("Ошибка распознавания.")

    def start_recognition(self) -> None:
        # Путь берём из публичного API контроллера, не трогая приватные поля.
        audio_path = self.audio.last_recording_path
        if audio_path is None:
            self.status_label.setText(
                "Сначала запиши речь, потом нажми «Распознать»."
            )
            return
        if self.stt.is_busy:
            return
        self.stt.recognize(audio_path)

    def toggle_record(self) -> None:
        if self.audio.state == self.audio.RECORDING:
            self.audio.stop_recording()
        else:
            # Во время STT новую запись запрещаем: файл ещё читается.
            if self.stt.is_busy:
                self.status_label.setText(
                    "Дождись окончания распознавания, чтобы начать запись."
                )
                return
            self.audio.record()

    def toggle_play(self) -> None:
        if self.audio.state == self.audio.PLAYING:
            self.audio.stop_playback()
        else:
            self.audio.play()

    def save_recording(self) -> None:
        """Сохраняет копию последней записи для диагностики (путь — в статусе)."""
        dest = self.audio.save_last_recording()
        if dest is not None:
            self.status_label.setText(
                "Запись сохранена для диагностики; при закрытии она не удалится."
            )
        # Если записи нет — статус контроллера уже объясняет причину.

    def closeEvent(self, event) -> None:  # noqa: N802 — имя из Qt
        # Если STT ещё работает — нельзя закрываться: файл читается, а
        # прерывать вычисление безопасно нельзя. Откладываем закрытие.
        if self.stt.is_busy:
            self._pending_close = True
            self.stt_status_label.setText(
                "Завершаю распознавание… закрытие отложено."
            )
            self.status_label.setText(
                "Идёт распознавание. Окно закроется после завершения."
            )
            event.ignore()
            return

        self._shutdown_for_close()
        event.accept()

    def _shutdown_for_close(self) -> None:
        # Порядок важен: STT завершается и освобождает файл, потом
        # AudioController удаляет временные файлы.
        self.stt.shutdown()
        self.audio.shutdown()

    def send_message(self) -> None:
        if self.controller.send_message(self.message_input.text()):
            self.message_input.clear()
            self.message_input.setFocus()
