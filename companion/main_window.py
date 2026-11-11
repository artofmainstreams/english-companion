from PySide6.QtCore import QSignalBlocker

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from .lesson import Lesson, LessonError, load_lessons
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

        # Выбор урока: показываем name и level («Урок 12 — Elementary»).
        # Пока доступен один урок (hw12); список растёт добавлением JSON.
        self.lesson_combo = QComboBox()
        self.lesson_combo.setMinimumWidth(250)
        # Выбор сцены: «Свободная беседа» (None) + сцены текущего урока
        # (данные — SceneSpec). Источник состояния сцены — этот выбор.
        self.scene_combo = QComboBox()
        self.scene_combo.setMinimumWidth(200)
        self.scene_combo.setEnabled(False)
        self.new_session_button = QPushButton("Новое занятие")

        self.record_button = QPushButton("Записать")
        self.play_button = QPushButton("Прослушать")
        # «Повторить ответ» — повторный проигрыш последней готовой
        # голосовой реплики преподавателя, без нового запроса к Ollama.
        self.repeat_button = QPushButton("Повторить ответ")
        # «Перезаписать» — новая запись поверх последней реплики
        # пользователя (пока после неё нет следующего сообщения
        # пользователя): замена реплики и связанного ответа.
        self.rerecord_button = QPushButton("Перезаписать")
        # «Отмена перезаписи» — отбросить текущую запись; прежний обмен
        # остаётся на месте, незавершённый ответ не перезапускается.
        self.rerecord_cancel_button = QPushButton("Отмена перезаписи")
        # «Повторить запрос» — повторить сбойный запрос к преподавателю
        # без дублирования реплики пользователя.
        self.retry_button = QPushButton("Повторить запрос")
        self.audio_status_label = QLabel("Голос: готов")
        self.stt_status_label = QLabel("Распознавание: готов")
        # Перезапись последней реплики: активна от клика «Перезаписать»
        # до успешной замены / отмены / пустого результата / ошибки STT.
        # До успешного распознавания прежний обмен НЕ трогается.
        self._rerecord_active = False
        self._rerecord_restore: Message | None = None
        self._rerecord_had_pending = False

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
        self.repeat_button.clicked.connect(self.repeat_reply)
        self.rerecord_button.clicked.connect(self.on_rerecord)
        self.rerecord_cancel_button.clicked.connect(self.on_rerecord_cancel)
        self.retry_button.clicked.connect(self.on_retry_request)
        self.mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        self.new_session_button.clicked.connect(self.on_new_session)
        self.lesson_combo.currentIndexChanged.connect(self._on_lesson_changed)
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(self.record_button)
        audio_layout.addWidget(self.play_button)
        audio_layout.addWidget(self.repeat_button)
        audio_layout.addWidget(self.rerecord_button)
        audio_layout.addWidget(self.rerecord_cancel_button)
        audio_layout.addWidget(self.retry_button)
        audio_layout.addWidget(QLabel("Микрофон:"))
        audio_layout.addWidget(self.mic_combo)
        audio_layout.addWidget(self.audio_status_label, 1)
        audio_layout.addWidget(self.stt_status_label)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.refresh_button)

        lesson_layout = QHBoxLayout()
        lesson_layout.addWidget(QLabel("Урок:"))
        lesson_layout.addWidget(self.lesson_combo, 1)
        lesson_layout.addWidget(QLabel("Сцена:"))
        lesson_layout.addWidget(self.scene_combo, 1)
        lesson_layout.addWidget(self.new_session_button)

        layout = QVBoxLayout()
        layout.addWidget(self.history, 1)
        layout.addLayout(input_layout)
        layout.addLayout(audio_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(model_layout)
        layout.addLayout(lesson_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        controller.state_changed.connect(self.render_state)
        controller.message_added.connect(self.append_message)
        # Перезапись/отмена: окно перерисовывает чат целиком из
        # контроллера (история + реплика в работе).
        controller.history_changed.connect(self._render_history)
        audio.state_changed.connect(self.render_audio)
        audio.devices_changed.connect(self._refresh_mic_combo)
        # Успешное завершение записи → сам запускаем распознавание по готовому
        # файлу (одна запись — один запуск STT; сбой/пустая запись не приходят).
        audio.recording_completed.connect(self._on_recording_completed)
        stt.state_changed.connect(self._on_stt_state_changed)
        stt.recognized.connect(self._on_stt_recognized)
        stt.failed.connect(self._on_stt_failed)
        self.render_state()
        self.render_audio()
        self._refresh_mic_combo()
        self._on_stt_state_changed()
        self._load_lessons()
        self._update_session_gate()

    def _load_lessons(self) -> None:
        """Загружает ВСЕ уроки из папки ``lessons/`` в комбо («name — level»).

        Натуральный порядок по имени файла (Урок 9 — до Урока 12);
        выбирается первый. Некорректный файл не отменяет остальные —
        его проблема показывается в строке статуса. Если урок не
        загрузился ни один — placeholder в комбо и отправка недоступна
        (``controller.can_send`` False — урока нет).
        """
        self.lessons: list[Lesson] = []
        errors: list[str] = []
        try:
            self.lessons, errors = load_lessons()
        except LessonError as exc:
            errors = [str(exc)]
        with QSignalBlocker(self.lesson_combo):
            for lesson in self.lessons:
                self.lesson_combo.addItem(
                    f"{lesson.name} — {lesson.level}", lesson
                )
        if self.lessons:
            self.controller.set_lesson(self.lessons[0])
        else:
            self.lesson_combo.setPlaceholderText("Не удалось загрузить уроки")
        # Список сцен выбранного урока (свободная беседа по умолчанию);
        # программное обновление — без отправки сообщений.
        self._refresh_scene_combo()
        if errors:
            detail = "; ".join(errors)
            if self.lessons:
                self.status_label.setText(f"Не все уроки загрузились: {detail}")
            else:
                self.status_label.setText(f"Не удалось загрузить уроки: {detail}")

    def _operations_idle(self) -> bool:
        """Текущие операции завершены: отправка/распознавание/синтез/аудио свободны."""
        if self.controller.busy:
            return False
        if self.stt.is_busy:
            return False
        if self.audio.state != self.audio.IDLE:
            return False
        if self._tts_synth_busy():
            return False
        return True

    def _update_session_gate(self) -> None:
        """«Новое занятие», смена урока и смена сцены — только после
        завершения операций."""
        has_lessons = self.lesson_combo.count() > 0
        idle = self._operations_idle()
        self.new_session_button.setEnabled(has_lessons and idle)
        self.lesson_combo.setEnabled(has_lessons and idle)
        self.scene_combo.setEnabled(has_lessons and idle)

    def on_new_session(self) -> None:
        """«Новое занятие»: чистая история + выбранный урок.

        Непустая история — сначала подтверждение; отмена — ничего не
        меняется. Доступно только после завершения текущих операций.
        """
        lesson = self.lesson_combo.currentData()
        if lesson is None or not self._operations_idle():
            return
        if self._confirm_new_session("Начать новое занятие с выбранным уроком?"):
            self._start_new_session(lesson)

    def _on_lesson_changed(self) -> None:
        """Смена урока = начало нового занятия (с подтверждением при
        непустой истории); история чата очищается.

        Отмена — комбо возвращается к уроку, который хранит контроллер
        (история и сцена сохраняются); на время установки пункта сигналы
        комбо заблокированы, чтобы обработчик не сработал повторно.
        """
        lesson = self.lesson_combo.currentData()
        if lesson is None or lesson is self.controller.lesson:
            return
        if not self._operations_idle():
            return
        if self._confirm_new_session(
            f"Сменить урок на «{lesson.name} — {lesson.level}» "
            "и начать новое занятие?"
        ):
            self._start_new_session(lesson)
        else:
            self._restore_lesson_combo()

    def _restore_lesson_combo(self) -> None:
        """Возвращает комбо урока к ``controller.lesson`` (история/сцена
        не трогаются); установка пункта не вызывает ``_on_lesson_changed``
        повторно (сигналы на время установки заблокированы)."""
        current = self.controller.lesson
        index = 0
        for i in range(self.lesson_combo.count()):
            if self.lesson_combo.itemData(i) is current:
                index = i
                break
        with QSignalBlocker(self.lesson_combo):
            self.lesson_combo.setCurrentIndex(index)

    # --- Выбор сцены (комбо «Сцена» — источник состояния) ---------------

    def _refresh_scene_combo(self) -> None:
        """Перестраивает список сцен текущего урока: «Свободная беседа»
        (первый пункт, данные None) + сцены урока (данные — ``SceneSpec``).

        Программное обновление: сигналы комбо на время перестроения
        блокированы, выбор возвращается на «Свободную беседу» — сообщений
        НЕ отправляется. Вызывается при загрузке уроков, новом занятии и
        подтверждённой смене урока (сцена завершается — комбо сбрасывается).
        """
        lesson = self.controller.lesson
        scenes = lesson.scenes if lesson is not None else ()
        with QSignalBlocker(self.scene_combo):
            self.scene_combo.clear()
            self.scene_combo.addItem("Свободная беседа", None)
            for spec in scenes:
                self.scene_combo.addItem(spec.name, spec)
            self.scene_combo.setCurrentIndex(0)

    def _restore_scene_combo(self) -> None:
        """Возвращает комбо сцен к сцене контроллера (None — «Свободная
        беседа»); установка пункта не вызывает ``_on_scene_changed``
        повторно (сигналы на время установки заблокированы)."""
        current = self.controller.scene
        index = 0
        for i in range(self.scene_combo.count()):
            spec = self.scene_combo.itemData(i)
            if current is None:
                match = spec is None
            else:
                match = spec is not None and spec.id == current.id
            if match:
                index = i
                break
        with QSignalBlocker(self.scene_combo):
            self.scene_combo.setCurrentIndex(index)

    def _on_scene_changed(self) -> None:
        """Выбор сцены в комбо (источник состояния — пользователь).

        «Свободная беседа» завершает сцену и отправляет прощальную
        реплику одним запросом (как авто-старт сцены); если сцены уже
        нет — запрос не уходит. Сцена — контроллер устанавливает её и
        отправляет стартовую реплику (ровно один запрос); историю чата
        сохраняет. Переключение заблокировано, пока идут операции (тот
        же затвор, что смена урока); сбой отправки (например, файл
        промпта не читается) — комбо и сцена возвращаются к прежним.
        """
        spec = self.scene_combo.currentData()
        if spec is None:
            if self.controller.scene is None:
                return  # уже свободная беседа — запрос не уходит
            if not self._operations_idle():
                self._restore_scene_combo()
                return
            if self.controller.exit_scene():
                self.status_label.setText("Свободная беседа")
            else:
                # Отправку начать нельзя: прежний пункт и прежняя сцена.
                self._restore_scene_combo()
            self._update_session_gate()
            return
        current = self.controller.scene
        if current is not None and current.id == spec.id:
            return  # та же сцена активна — повторной отправки нет
        if not self._operations_idle():
            self._restore_scene_combo()
            return
        if self.controller.select_scene(spec):
            self.status_label.setText(f"Сцена: {spec.name}")
        else:
            # Отправку начать нельзя: прежний пункт и прежняя сцена.
            self._restore_scene_combo()
        self._update_session_gate()

    def _confirm_new_session(self, question_text: str) -> bool:
        """Пустая история — подтверждения не требуется; непустая — диалог."""
        if not self.controller.history:
            return True
        answer = QMessageBox.question(
            self,
            "Новое занятие",
            f"{question_text}\nИстория разговора будет очищена.",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _start_new_session(self, lesson: Lesson) -> None:
        self.controller.set_lesson(lesson)
        self.controller.reset_history()
        self.history.clear()
        # Сцена завершена (свободная беседа): комбо — на первый пункт,
        # список — сцены НОВОГО урока (без отправки сообщений).
        self._refresh_scene_combo()
        self.status_label.setText(f"Новое занятие: {lesson.name} · {lesson.level}")
        self._update_session_gate()

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
        # Ожидание ответа (controller.busy) тоже блокирует запись.
        self._update_record_gate()
        self._update_session_gate()
        self._update_rerecord_gates()

    @staticmethod
    def _format_message(message: Message) -> str:
        speaker = "You" if message.role == "user" else f"Teacher · {message.model}"
        return f"{speaker}: {message.content}\n"

    def append_message(self, message: Message) -> None:
        # Текст сразу виден в чате; затем — озвучивание реплики преподавателя.
        self.history.appendPlainText(self._format_message(message))
        if message.role == "assistant":
            self.audio.speak_reply(message.content)

    def _render_history(self) -> None:
        """Перерисовывает весь чат по контроллеру (перезапись/отмена).

        Источник правды — ``SessionController``: история (только завершённые
        обмены) + реплика в работе (её ответ ещё не пришёл). Старая
        реплика и связанный с ней ответ после замены исчезают из окна —
        как из истории, отправляемой в Ollama.
        """
        parts = [self._format_message(m) for m in self.controller.history]
        if self.controller.pending_message is not None:
            parts.append(self._format_message(self.controller.pending_message))
        self.history.setPlainText("".join(parts))

    def select_model(self) -> None:
        self.controller.select_model(self.model_combo.currentText())

    def update_send_button(self) -> None:
        self.send_button.setEnabled(
            bool(self.message_input.text().strip())
            and self.controller.can_send
        )

    def render_audio(self) -> None:
        audio = self.audio
        playing = audio.state == audio.PLAYING

        # Кнопка «Записать/Остановить» — единый затвор записей (см.
        # _update_record_gate / _can_start_recording): пока голосовая
        # реплика обрабатывается, новая запись запрещена.
        self._update_record_gate()
        self._update_session_gate()
        self._update_rerecord_gates()

        self.play_button.setEnabled(audio.can_play or playing)
        self.play_button.setText("Стоп" if playing else "Прослушать")

        # «Повторить ответ»: только если есть готовая реплика и аудио
        # свободно (запись/воспроизведение не идут). После shutdown — нет.
        self.repeat_button.setEnabled(audio.can_replay)

        # Переключение микрофона — только вне записи (требование);
        # после shutdown контроллер не работает, комбо блокируем тоже.
        self.mic_combo.setEnabled(
            not audio.is_shutdown and audio.state != audio.RECORDING
        )

        self.audio_status_label.setText(audio.status)

    def _tts_synth_busy(self) -> bool:
        """Ещё идёт синтез ответа (WAV пишется, файл не готов)."""
        return bool(getattr(self.audio.tts, "is_busy", False))

    def _can_start_recording(self) -> bool:
        """Можно ли начать новую запись прямо сейчас.

        Пока голосовая реплика обрабатывается — запись блокируем: идёт
        запись/воспроизведение, распознавание, ожидание ответа от Ollama
        или ещё идёт синтез. Доступность возвращается после ответа,
        остановки воспроизведения или ошибки (см. render_audio / render_state
        / сигналы STT — они и вызывают _update_record_gate).
        """
        if not self.audio.can_record:
            return False
        if self.stt.is_busy:
            return False
        if self.controller.busy:
            return False
        if self._tts_synth_busy():
            return False
        return True

    def _update_record_gate(self) -> None:
        """Возвращает/забирает доступность «Записать» (единый затвор).

        Пока идёт сама запись — кнопка остаётся активной («Остановить»).
        В остальное время запись доступна только когда свободны запись/
        воспроизведение, распознавание, ожидание ответа и синтез.
        """
        if self.audio.state == self.audio.RECORDING:
            self.record_button.setEnabled(True)
            self.record_button.setText("Остановить")
        else:
            self.record_button.setEnabled(self._can_start_recording())
            self.record_button.setText("Записать")

    def _refresh_mic_combo(self) -> None:
        """Перечитывает список микрофонов, сохраняя текущий выбор.

        В ``itemData`` храним стабильный ``QAudioDevice.id()``: названия
        (description) могут совпадать у разных устройств, по id — нет.
        Позиция 0 — системное устройство, его ``itemData`` равно ``None``.
        Сигналы комбо на время перестроения блокированы, чтобы «сброс»
        индекса не сбивал текущий выбор.
        """
        audio = self.audio
        selected_id = audio.selected_input_id()
        with QSignalBlocker(self.mic_combo):
            self.mic_combo.clear()
            self.mic_combo.addItem("Системный (по умолчанию)", None)
            for device_id, name in audio.input_devices():
                self.mic_combo.addItem(name, device_id)
            if selected_id is not None:
                self.mic_combo.setCurrentIndex(
                    max(self.mic_combo.findData(selected_id), 0)
                )
            else:
                self.mic_combo.setCurrentIndex(0)
        self.render_audio()  # статус контроллера + состояние кнопок

    def _on_mic_changed(self) -> None:
        """Передаём контроллеру ``itemData``: ``None`` — системное устройство."""
        self.audio.set_input_device(self.mic_combo.currentData())

    def _on_stt_state_changed(self) -> None:
        self.stt_status_label.setText(self.stt.status)
        # Кнопка записи зависит от занятости STT (и всего остального).
        self.render_audio()
        # Если закрытие было отложено из-за активной работы — завершаем.
        if self._pending_close and not self.stt.is_busy:
            self._pending_close = False
            self.close()

    def _on_recording_completed(self, path: str) -> None:
        """Успешное завершение записи → автозапуск STT по готовому файлу.

        Сигнал приходит ровно раз на успешную запись (см. рекордер), поэтому
        одна запись даёт один запуск распознавания. Сбой/пустая запись сигнал
        не подают — здесь всегда есть реальный готовый файл, а не «пишущийся».
        """
        self.stt_status_label.setText("Распознаю речь…")
        self.status_label.setText("Запись завершена. Распознаю речь…")
        # SttEngine сам отклонит повторный/запоздалый вызов (busy/shutdown)
        # и проверит файл, поэтому окно просто передаёт путь готовой записи.
        self.stt.recognize(path)

    def _on_stt_recognized(self, text: str) -> None:
        text = (text or "").strip()
        if self._rerecord_active:
            # Перезапись: до успешного распознавания прежний обмен не тронут.
            if not text:
                self._finish_rerecord_incomplete("пустой результат")
                return
            target = self._rerecord_restore
            if self.controller.replace_last_user_message(text, target):
                # Замена подтверждена: чат перерисован (history_changed),
                # новый ответ придёт и будет озвучен (message_added).
                self._rerecord_active = False
                self._rerecord_restore = None
                self._rerecord_had_pending = False
                self.status_label.setText("Перезаписано; отвечаю…")
            else:
                self._finish_rerecord_incomplete("замена не прошла")
            self._update_record_gate()
            self._update_rerecord_gates()
            return
        if not text:
            # Пустой результат: ничего не отправляем.
            self.status_label.setText("Речь не распознана (пустой результат).")
            self._update_record_gate()
            return

        # Голосовой ввод НЕ трогает поле ввода (оно — для ручного текста):
        # расшифровка уходит либо в чат (отправка), либо в строку статуса.
        if not self.controller.can_send:
            # Отправка недоступна: причина и распознанный текст — в статусе.
            self.status_label.setText(
                "Отправка недоступна (нет выбранной модели или идёт запрос). "
                f"Распознанный текст: «{text}»."
            )
            self._update_record_gate()
            return

        # Непустая расшифровка → автоотправка (реплика в чате как обычная).
        # «recognized» срабатывает разово → одна запись даёт одну отправку.
        if self.controller.send_message(text):
            self.status_label.setText("Отправлено; отвечаю…")
        else:
            # can_send было True, но отправка не прошла — причина в
            # статусе контроллера (например, файл промпта: путь и
            # повтор после исправления); не затираем.
            reason = self.controller.status
            prefix = (
                f"{reason} " if reason and reason != "Готов"
                else "Отправка не прошла, повторите. "
            )
            self.status_label.setText(f"{prefix}Распознанный текст: «{text}».")
        self._update_record_gate()

    def _on_stt_failed(self, message: str) -> None:
        if self._rerecord_active:
            # Ошибка STT во время перезаписи: прежний обмен остаётся.
            self.stt_status_label.setText(message)
            self._finish_rerecord_incomplete("ошибка распознавания")
            return
        self.stt_status_label.setText(message)
        self.status_label.setText("Ошибка распознавания.")
        # Ошибка распознавания возвращает доступность записи.
        self._update_record_gate()

    def toggle_record(self) -> None:
        if self.audio.state == self.audio.RECORDING:
            self.audio.stop_recording()
        else:
            # Кнопка активна только когда запись разрешена (см. _update_record_gate);
            # фасад дополнительно сам отклоняет запись при занятости аудио.
            self.audio.record()

    def toggle_play(self) -> None:
        if self.audio.state == self.audio.PLAYING:
            self.audio.stop_playback()
        else:
            self.audio.play()

    def repeat_reply(self) -> None:
        """«Повторить ответ»: повторный проигрыш последней готовой реплики.

        Новый запрос к Ollama не выполняется — играет тот же готовый файл.
        Если готового ответа нет или аудио занято, фасад публикует причину
        в строку статуса.
        """
        self.audio.replay_reply()

    # --- Перезапись последней реплики пользователя -------------------------

    def _can_rerecord(self) -> bool:
        """Можно ли начать перезапись последней реплики пользователя.

        Доступна, пока после этой реплики нет следующего сообщения
        пользователя (ответ преподавателя после неё не препятствует — он
        будет убран вместе с ней). Во время активной перезаписи,
        распознавания или без модели/урока — недоступна.
        """
        if self._rerecord_active:
            return False
        if self.stt.is_busy:
            return False
        if not (self.controller.selected_model and self.controller.lesson):
            return False
        return self.controller.last_user_message() is not None

    def on_rerecord(self) -> None:
        """«Перезаписать»: остановить текущий ответ и озвучивание, начать
        новую запись поверх последней реплики пользователя.

        До успешного распознавания прежний обмен остаётся на месте.
        Запоздалые результаты прежних операций отбрасываются. Одна
        перезапись — максимум одна замена и один новый запрос.
        """
        if self._rerecord_active or not self._can_rerecord():
            return
        target = self.controller.last_user_message()
        if target is None:
            return
        self._rerecord_active = True
        self._rerecord_restore = target
        self._rerecord_had_pending = self.controller.pending_message is not None
        # Останавливаем текущий ответ преподавателя и его озвучивание.
        self.audio.stop_playback()
        # Старый голосовой ответ — недоступен для повтора.
        self.audio.suppress_reply()
        # Запрос в работе отменяем: поздний результат отбрасывается тихо.
        self.controller.discard_pending()
        # Новая запись (или продолжаем уже идущую — она станет перезаписью).
        if self.audio.state != self.audio.RECORDING:
            self.audio.record()
        self.status_label.setText("Перезаписываю последнюю реплику…")
        self._update_rerecord_gates()

    def on_rerecord_cancel(self) -> None:
        """«Отмена перезаписи»: отбросить запись; прежний обмен остаётся.

        Незавершённый старый ответ НЕ запускается автоматически; прежний
        повтор (если готовый файл сохранился) возвращается.
        """
        if not self._rerecord_active:
            return
        self.audio.cancel_recording()
        self._finish_rerecord_incomplete("отмена")

    def _finish_rerecord_incomplete(self, reason: str) -> None:
        """Отмена / пустой результат / ошибка STT: прежний обмен на месте.

        Если запрос был в работе, его реплика возвращается в «ждёт ответ»
        БЕЗ нового запроса; прежний повтор возвращается (если файл есть).
        """
        self._rerecord_active = False
        had_pending = self._rerecord_had_pending
        target = self._rerecord_restore
        # Причина сбоя замены (например, файл промпта: путь) — в статусе
        # контроллера; сохраняем ДО restore_pending (он меняет статус).
        detail = self.controller.status
        self._rerecord_restore = None
        self._rerecord_had_pending = False
        if had_pending and target is not None:
            self.controller.restore_pending(target)
        self.audio.unsuppress_reply()
        if reason == "замена не прошла" and detail and detail != "Готов":
            self.status_label.setText(f"{detail} Прежний обмен сохранён.")
        else:
            self.status_label.setText(
                f"Перезапись не завершена ({reason}): прежний обмен сохранён."
            )
        self._update_record_gate()
        self._update_rerecord_gates()

    def on_retry_request(self) -> None:
        """«Повторить запрос»: повторить сбойный запрос к преподавателю
        без дублирования реплики пользователя (она уже в чате)."""
        if self.controller.retry_last_user_message():
            self.status_label.setText("Повторяю запрос…")
        self._update_rerecord_gates()

    def _update_rerecord_gates(self) -> None:
        """Доступность «Перезаписать» / «Отмена перезаписи» / «Повторить запрос»."""
        self.rerecord_button.setEnabled(self._can_rerecord())
        self.rerecord_cancel_button.setEnabled(self._rerecord_active)
        self.retry_button.setEnabled(
            self.controller.retry_message is not None
            and not self.controller.busy
            and not self._rerecord_active
        )

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
