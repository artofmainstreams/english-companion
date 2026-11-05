import logging
import os
import shutil
import tempfile
import time

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import (
    QAudioDevice,
    QAudioInput,
    QAudioOutput,
    QMediaCaptureSession,
    QMediaDevices,
    QMediaFormat,
    QMediaPlayer,
    QMediaRecorder,
)

logger = logging.getLogger(__name__)


class AudioController(QObject):
    """Управляет записью и воспроизведением речи без блокировки интерфейса.

    Работает как отдельный слой (как SessionController/OllamaClient):
    MainWindow только показывает состояние и вызывает методы, сам Qt
    Multimedia живёт здесь и не мешает текстовому чату.

    Жизненный цикл:
    * ``shutdown()`` идемпотентна — повторные вызовы безопасны;
    * после ``shutdown()`` ``record()``/``play()`` отклоняются, а поздние
      сигналы Qt игнорируются (состояние больше не «оживает»);
    * разбор графи по порядку: остановка плеера → отключение источника →
      выход → остановка записи → удаление временных файлов;
    * выход QAudioOutput отсоединяется от плеера явно
      (``setAudioOutput(None)``) до любого удаления, поэтому разбор не
      зависит от порядка отложенных ``deleteLater()``.
    """

    state_changed = Signal()
    # Список устройств ввода изменился (подключение/отключение микрофона):
    # интерфейс перечитывает список; если выбранный микрофон пропал,
    # контроллер сам возвращается к системному устройству.
    devices_changed = Signal()

    IDLE = "idle"
    RECORDING = "recording"
    PLAYING = "playing"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = self.IDLE
        self.status = "Голос: готов"

        # Базовая точка времени для диагностических событий (элипсис в мс).
        self._t0 = time.monotonic()

        self._format = self._pick_format()
        self._temp_dir: str | None = None
        self._record_file: str | None = None
        # Каждая запись получает новый файл. QMediaPlayer кэширует медиа-
        # источник по URL: если записывать поверх того же пути, setSource()
        # вернёт уже декодированную (старую) запись, а не перечитает файл.
        self._record_seq = 0
        # Готовая (успешно завершённая) запись — то, что можно слушать,
        # сохранять и передавать STT. Отличается от ``_record_file``: то —
        # текущий файл, который ещё может писаться (готовым НЕ считается);
        # это — последняя запись, подтверждённая завершением рекордером.
        self._ready_recording: str | None = None

        # Сбой текущей сессии записи (errorOccurred): частичный файл
        # после такого не считается готовой записью — независимо от того,
        # пришёл сигнал до или после перехода в StoppedState.
        self._session_failed = False

        # Выбранное устройство ввода: ``None`` — системное устройство
        # по умолчанию. Идентификатор берём из ``QAudioDevice.id()`` —
        # он стабилен для сессии, в отличие от порядкового номера в списке.
        self._selected_input_id: str | None = None

        # Подключение/отключение устройств (статический сигнал класса,
        # без аргументов). В этой сборке PySide6 его нельзя подключить
        # через класс (нет ``connect``) — держим экземпляр QMediaDevices
        # и подключаем через него; список перечитываем по факту.
        self._devices = QMediaDevices()
        self._devices.audioInputsChanged.connect(
            self._on_input_devices_changed
        )

        # Граф записи создаётся на каждый сеанс, чтобы корректно
        # освобождать микрофон и пересоздавать устройство ввода.
        self._audio_input: QAudioInput | None = None
        self._recorder: QMediaRecorder | None = None
        self._session: QMediaCaptureSession | None = None

        # На этой сборке Qt у QMediaPlayer нет «по умолчанию» подключённого
        # выхода (audioOutput() == None); явный QAudioOutput создаём при play().
        self._audio_output: QAudioOutput | None = None
        self._player: QMediaPlayer | None = QMediaPlayer(self)
        self._player.playingChanged.connect(self._on_player_playing_changed)
        self._player.mediaStatusChanged.connect(self._on_player_media_status)
        self._player.errorOccurred.connect(self._on_player_error)

        # Флаг shutdown: после него контроллер не восстанавливается.
        self._shutdown = False

    # --- Свойства для интерфейса ------------------------------------------

    @staticmethod
    def _valid_recording_file(path: str | None) -> bool:
        """Файл существует и непустой — пригоден как «готовая запись»."""
        return (
            path is not None
            and os.path.exists(path)
            and os.path.getsize(path) > 0
        )

    @property
    def has_recording(self) -> bool:
        """Есть ли успешно завершённая запись (не «пишущийся» файл)."""
        return self._valid_recording_file(self._ready_recording)

    @property
    def can_record(self) -> bool:
        return (
            not self._shutdown
            and self.state == self.IDLE
            and self._format is not None
        )

    @property
    def can_play(self) -> bool:
        return (
            not self._shutdown
            and self.state == self.IDLE
            and self.has_recording
        )

    @property
    def is_shutdown(self) -> bool:
        """``True`` после ``shutdown()``: контроллер больше не работает."""
        return self._shutdown

    @property
    def last_recording_path(self) -> str | None:
        """Публичный доступ к пути последней УСПЕШНО ЗАВЕРШЁННОЙ записи.

        Возвращает путь к файлу, если запись готова (существует и непуста),
        иначе ``None``. «Пишущийся» файл текущей сессии здесь не отдаётся —
        его нельзя сохранять, слушать или передавать STT как готовый. Из окна
        и STT берётся путь именно через это свойство, а не через приватные поля.
        """
        if self.has_recording and self._ready_recording is not None:
            return self._ready_recording
        return None

    def input_devices(self) -> list[tuple[str, str]]:
        """Доступные устройства ввода: пары ``(id, имя)``.

        ``id`` — стабильный ``QAudioDevice.id()``: по нему устройство
        выбираем и храним в интерфейсе (``itemData``). Имя (description)
        только для отображения — у разных устройств оно может совпадать.
        """
        return [
            (self._device_id(dev), dev.description())
            for dev in QMediaDevices.audioInputs()
            if not dev.isNull()
        ]

    def input_device_names(self) -> list[str]:
        """Названия доступных устройств ввода (только для отображения)."""
        return [name for _id, name in self.input_devices()]

    def selected_input_id(self) -> str | None:
        """Идентификатор выбранного микрофона; ``None`` — системное устройство.

        Если выбранное устройство отключили, возвращает ``None``:
        фактическое сбрасывание выбора делает ``_reselect_if_needed()``.
        """
        if self._selected_input_id is None:
            return None
        for dev in QMediaDevices.audioInputs():
            if not dev.isNull() and self._device_id(dev) == self._selected_input_id:
                return self._selected_input_id
        return None

    def selected_input_name(self) -> str | None:
        """Название выбранного микрофона; ``None`` — системное устройство.

        Ищет устройство по идентификатору: если микрофон отключили,
        имя вернёт ``None`` (как бы выбор сбросился — факт отключения
        отражает ``_reselect_if_needed()``).
        """
        if self._selected_input_id is None:
            return None
        for dev in QMediaDevices.audioInputs():
            if not dev.isNull() and self._device_id(dev) == self._selected_input_id:
                return dev.description()
        return None

    def set_input_device(self, name_or_id: str | None) -> bool:
        """Выбирает микрофон по названию или идентификатору.

        ``None`` — вернуться к системному устройству по умолчанию.
        Возвращает ``False`` (и объясняет почему в ``status``), если:
        контроллер остановлен, идёт запись (переключение запрещено)
        или устройство не найдено.
        """
        if self._shutdown:
            return False
        if self.state == self.RECORDING:
            self.status = (
                "Сначала останови запись: во время записи "
                "переключать микрофон нельзя."
            )
            self.state_changed.emit()
            return False

        if name_or_id is None:
            self._selected_input_id = None
            self.status = "Микрофон: системный (по умолчанию)."
            self.state_changed.emit()
            return True

        for dev in QMediaDevices.audioInputs():
            if dev.isNull():
                continue
            if dev.description() == name_or_id or self._device_id(dev) == name_or_id:
                self._selected_input_id = self._device_id(dev)
                self.status = f"Микрофон: {dev.description()}"
                self.state_changed.emit()
                return True

        self.status = f"Микрофон «{name_or_id}» не найден — выбор не изменён."
        self.state_changed.emit()
        return False

    # --- Публичные действия ------------------------------------------------

    def record(self) -> None:
        if self._shutdown or self.state != self.IDLE:
            return
        if self._format is None:
            self.status = "Нет поддерживаемого формата записи аудио."
            self.state_changed.emit()
            return

        # Гарантированно чистим предыдущую сессию записи.
        self._teardown_recording()

        # Выбранный микрофон; при его отсутствии разрешается
        # в системное устройство (см. _selected_input_device).
        device = self._selected_input_device()
        if device.isNull():
            self.status = (
                "Нет микрофона. Подключи устройство ввода звука."
            )
            self.state_changed.emit()
            return

        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="english-companion-audio-")
        self._record_seq += 1
        self._record_file = os.path.join(
            self._temp_dir, f"recording_{self._record_seq}{self._extension()}"
        )
        url = QUrl.fromLocalFile(self._record_file)
        self._session_failed = False
        # Новая запись сменяет предыдущую «готовую»: текущий файл ещё не
        # готов (станет им только после подтверждения завершения рекордером),
        # поэтому до финализации ничего не публикуется как готовое.
        self._ready_recording = None

        self._audio_input = QAudioInput(device)
        self._recorder = QMediaRecorder()
        self._recorder.setMediaFormat(self._format)
        self._recorder.setOutputLocation(url)
        self._recorder.recorderStateChanged.connect(self._on_recorder_state)
        self._recorder.errorOccurred.connect(self._on_recorder_error)
        self._session = QMediaCaptureSession(
            audioInput=self._audio_input,
            recorder=self._recorder,
        )

        self.state = self.RECORDING
        self.status = "Записываю… (нажми «Остановить»)"
        self.state_changed.emit()
        self._recorder.record()

    def stop_recording(self) -> None:
        if self._shutdown or self.state != self.RECORDING or self._recorder is None:
            return
        self._log_event("record:stop-requested")
        self._recorder.stop()

    def play(self) -> None:
        if self._shutdown or self.state != self.IDLE:
            return
        if self._player is None:
            return
        if not self.has_recording:
            self.status = "Сначала запиши речь."
            self.state_changed.emit()
            return
        src = self._ready_recording

        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.status = "Нет устройства вывода звука."
            self.state_changed.emit()
            return

        # На этой сборке Qt без явного QAudioOutput звук никуда не идёт
        # (EndOfMedia не приходит, плеер «зависает» в BufferingMedia).
        # Предыдущий выход отсоединяем от плеера явно, а не только
        # deleteLater(): корректность не зависит от порядка очереди Qt.
        self._release_audio_output()
        self._audio_output = QAudioOutput(device)
        self._audio_output.setVolume(1.0)
        self._player.setAudioOutput(self._audio_output)
        self._log_event(
            "play:start",
            output=device.description(),
            source=os.path.basename(src),
        )

        self._player.setSource(QUrl.fromLocalFile(src))
        self.state = self.PLAYING
        self.status = "Воспроизведение…"
        self.state_changed.emit()
        self._player.play()

    def stop_playback(self) -> None:
        if self._shutdown or self._player is None:
            return
        self._log_event("play:stop")
        self._player.stop()
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Голос: готов"
            self.state_changed.emit()

    def _log_event(self, name: str, **kv) -> None:
        """Диагностическое событие с элипсисом (мс от старта контроллера).

        Помогает восстановить таймлайн «завершение записи → освобождение
        входа → выбор выхода → смена source → mediaStatus → playbackState →
        ошибки». Включается уровнем logging (INFO).
        """
        try:
            elapsed_ms = int((time.monotonic() - self._t0) * 1000)
        except (AttributeError, ValueError):
            elapsed_ms = -1
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info("[audio] %sms %s%s", elapsed_ms, name, (" " + extra) if extra else "")

    def save_last_recording(self, dest_dir: str | None = None) -> str | None:
        """Сохраняет копию последней УСПЕШНО ЗАВЕРШЁННОЙ записи наружу.

        Копия уходит во внешнюю папку (по умолчанию — Рабочий стол) и переживает
        закрытие окна; временный оригинал (и сам временный каталог) удаляются.
        Экспорт ВНУТРЬ управляемого временного каталога запрещён — такая копия
        исчезла бы вместе с ним. Существующий файл не перезаписывается:
        выбирается свободное имя. Возвращает путь к копии или ``None`` (нет
        готовой записи, контроллер остановлен, либо отказ — причина в ``status``).
        """
        if self._shutdown:
            self.status = "Сохранить нечего: контроллер остановлен."
            self.state_changed.emit()
            return None
        if not self.has_recording:
            self.status = "Сохранить нечего: нет завершённой записи."
            self.state_changed.emit()
            return None
        src = self._ready_recording
        dest_dir = os.path.abspath(os.path.expanduser(dest_dir or "~/Desktop"))

        # Запрет экспорта внутрь управляемого временного каталога: иначе копия
        # уйдёт при закрытии вместе с временными файлами. Принадлежность
        # проверяем по realpath (разрешая симв. ссылки) и commonpath —
        # корректно, в отличие от строкового startswith.
        if self._temp_dir is not None and self._is_within(dest_dir, self._temp_dir):
            self.status = (
                "Нельзя сохранять внутрь временного каталога — файл удалится "
                "при закрытии. Укажи папку вне него (например, Рабочий стол)."
            )
            self.state_changed.emit()
            return None

        try:
            os.makedirs(dest_dir, exist_ok=True)
            ext = os.path.splitext(src)[1] or ".m4a"
            base = "english-companion"
            # Не перезаписываем существующий файл (в т.ч. симссылку):
            # ищем свободное имя.
            dest = os.path.join(dest_dir, f"{base}{ext}")
            n = 1
            while os.path.lexists(dest):
                dest = os.path.join(dest_dir, f"{base}-{n}{ext}")
                n += 1
            shutil.copy2(src, dest)
        except OSError as exc:
            self.status = f"Не удалось сохранить запись: {exc}"
            self.state_changed.emit()
            return None

        self.status = f"Запись сохранена: {dest}"
        self.state_changed.emit()
        return dest

    @staticmethod
    def _is_within(path: str, base: str) -> bool:
        """``True``, если ``path`` совпадает с ``base`` или находится внутри него.

        Сравниваем по ``os.path.realpath`` (разрешая символические ссылки) и
        через ``os.path.commonpath`` — в отличие от строкового ``startswith``,
        это корректно: ``/tmp/foo-evil`` не считается «внутри» ``/tmp/foo``.
        """
        real_path = os.path.realpath(path)
        real_base = os.path.realpath(base)
        try:
            common = os.path.commonpath([real_path, real_base])
        except ValueError:
            return False  # разные диски — «внутри» быть не может
        return common == real_base

    def _release_audio_output(self) -> None:
        """Явно отсоединяет QAudioOutput от QMediaPlayer и освобождает его.

        ``setAudioOutput(None)`` разбивает ссылку «плеер → выход» прямо,
        а не в момент отложенного удаления, плюс снимает связи сигналов
        выход ↔ плеер. Тогда висячих ссылок нет независимо от того,
        в каком порядке обработаются отложенные ``deleteLater()``.
        """
        output = self._audio_output
        self._audio_output = None
        if output is None:
            return
        if self._player is not None:
            try:
                self._player.setAudioOutput(None)
            except Exception:
                logger.warning(
                    "Не удалось отключить QAudioOutput от QMediaPlayer",
                    exc_info=True,
                )
            try:
                output.disconnect(self._player)
                self._player.disconnect(output)
            except Exception:
                logger.warning(
                    "Не удалось снять связи выход ↔ плеер", exc_info=True
                )
        output.deleteLater()

    def shutdown(self) -> None:
        """Останавливает запись/воспроизведение и удаляет временные файлы.

        Идемпотентна: повторные вызовы (например, ``aboutToQuit`` и явный
        вызов из кода) безопасны. После вызова контроллер не работает:
        ``record()``/``play()`` отклоняются, поздние сигналы Qt игнорируются.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # 0) Снимаем обработчик hotplug-сигнала и отпускаем экземпляр
        #    QMediaDevices, чтобы поздние события не доходили до мёртвого
        #    контроллера.
        if self._devices is not None:
            try:
                self._devices.audioInputsChanged.disconnect(
                    self._on_input_devices_changed
                )
            except (TypeError, RuntimeError):
                pass  # сигнал уже снят (повторный shutdown) — нормально

        # 1) Выход: явное отсоединение от плеера (setAudioOutput(None)) и
        #    снятие связей, затем отложенное удаление. Раньше здесь был
        #    только deleteLater(), и корректность зависела от порядка
        #    очереди Qt (проверено: сбойный сценарий воспроизводился,
        #    когда старый выход умирал после переключения плеера); явное
        #    отсоединение сделало разбор независимым от порядка.
        self._release_audio_output()

        # 2) Плеер: сначала остановка, затем отключение источника, чтобы
        #    плеер отпустил файл до удаления. Уничтожаем после обеих команд.
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
            self._player.deleteLater()
            self._player = None

        # 3) Запись: стоп до разрушения объекта, затем разбор графа.
        self._teardown_recording()

        # 4) Временные файлы (после того, как плеер отпустил источник).
        self._cleanup_temp_dir()

        self._record_file = None
        self._ready_recording = None
        self._record_seq = 0
        self.state = self.IDLE
        self.status = "Голос: остановлен"

    # --- Обработчики сигналов ---------------------------------------------

    def _on_recorder_state(self, state: QMediaRecorder.RecorderState) -> None:
        # Поздний сигнал после shutdown не должен «оживлять» состояние.
        if self._shutdown:
            return
        self._log_event("recorderState", state=str(state))
        if state == QMediaRecorder.RecorderState.RecordingState:
            self.state = self.RECORDING
            self.status = "Записываю… (нажми «Остановить»)"
            self.state_changed.emit()
        elif state == QMediaRecorder.RecorderState.StoppedState:
            self._finalize_recording()

    def _on_input_devices_changed(self) -> None:
        """Список входов изменился (подключили/отключили устройство)."""
        if self._shutdown:
            return
        # Если выбранный микрофон пропал — возвращаемся к системному,
        # чтобы запись не упала на «пустом» устройстве.
        self._reselect_if_needed()
        self.devices_changed.emit()

    def _reselect_if_needed(self) -> None:
        """Выбранный микрофон отсутствует — сбрасываем выбор на системный.

        Вызывается при hotplug; безопасна к повторным вызовам и к
        случаю, когда выбран системный (``None``) — тогда ничего не делает.
        """
        if self._selected_input_id is None:
            return
        for dev in QMediaDevices.audioInputs():
            if (
                not dev.isNull()
                and self._device_id(dev) == self._selected_input_id
            ):
                return  # выбранный микрофон ещё на месте
        self._selected_input_id = None
        self.status = (
            "Выбранный микрофон отключён — запись пойдёт через "
            "системное устройство по умолчанию."
        )
        self.state_changed.emit()

    def _selected_input_device(self) -> QAudioDevice:
        """Разрешает выбранный микрофон; при отсутствии — системный."""
        if self._selected_input_id is not None:
            for dev in QMediaDevices.audioInputs():
                if (
                    not dev.isNull()
                    and self._device_id(dev) == self._selected_input_id
                ):
                    return dev
        return QMediaDevices.defaultAudioInput()

    @staticmethod
    def _device_id(device: QAudioDevice) -> str:
        """Строковый идентификатор устройства (``QAudioDevice.id()``)."""
        return device.id().data().decode("utf-8", "replace")

    def _on_recorder_error(
        self, error: QMediaRecorder.Error, message: str
    ) -> None:
        if error == QMediaRecorder.Error.NoError or self._shutdown:
            return
        # Ошибка сессии отменяет «готовность» записи даже если состояние
        # уже Stopped: файл мог быть записан частично (например, устройство
        # отключили во время записи) — как «запись готова» он не считается.
        # Слот отключается до deleteLater в _teardown_recording, поэтому
        # сигналы уже разобранной сессии к нам не доходят.
        self._session_failed = True
        self.state = self.IDLE
        self.status = "Ошибка записи: " + (message or error.name)
        self._record_file = None
        self._ready_recording = None
        self._teardown_recording()
        self.state_changed.emit()

    def _on_player_playing_changed(self, playing: bool) -> None:
        if self._shutdown:
            return
        self._log_event(
            "playbackState",
            playing=playing,
            position_ms=(self._player.position() if self._player is not None else -1),
        )
        if playing:
            self.state = self.PLAYING
            self.status = "Воспроизведение…"
        elif self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Голос: готов"
        self.state_changed.emit()

    def _on_player_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if self._shutdown:
            return
        self._log_event("mediaStatus", status=str(status))
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._player_fail("не удалось открыть файл записи")

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError or self._shutdown:
            return
        self._log_event("player_error", error=str(error), message=message)
        self._player_fail(message or error.name)

    # --- Вспомогательные ---------------------------------------------------

    def _finalize_recording(self) -> None:
        """Завершает сессию: запись публикуется «готовой» только при успехе.

        Успех = рекордер подтвердил завершение (StoppedState) и файл непустой.
        Сбой сессии или пустой файл => готовой записи нет (``_ready_recording``
        не ставится, а ранее выставленная «готовность» снимается).
        """
        if self.state == self.RECORDING:
            self.state = self.IDLE
            if self._session_failed:
                # Ошибка сессии: файл (даже не пустой) частичный — готовым
                # не публикуем ни в контроллере, ни в UI.
                self.status = "Ошибка записи — файл не считается готовым."
                self._ready_recording = None
                self._record_file = None
            elif self._valid_recording_file(self._record_file):
                # Готовность публикуем только после подтверждения завершения —
                # «пишущийся» файл (состояние RECORDING) сюда не попадает.
                self._ready_recording = self._record_file
                self.status = "Запись готова. Можно «Прослушать»."
            else:
                self.status = (
                    "Запись пуста — вероятно, нет доступа к микрофону. "
                    "Проверь разрешения macOS."
                )
                self._ready_recording = None
                self._record_file = None
        # Флаг сессии завершён (успешно или со сбоем): следующая запись
        # стартует с чистого состояния.
        self._session_failed = False
        self._log_event("record:finalized", ready=self.has_recording)
        self._teardown_recording()
        self.state_changed.emit()

    def _player_fail(self, message: str) -> None:
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Ошибка воспроизведения: " + message
            self.state_changed.emit()

    def _teardown_recording(self) -> None:
        """Останавливает запись и разбирает граф ввод/кодер/сессия."""
        recorder = self._recorder
        session = self._session
        audio_input = self._audio_input
        if audio_input is not None:
            self._log_event("record:teardown")
        self._recorder = None
        self._session = None
        self._audio_input = None

        if recorder is not None:
            # Стоп вызываем до разрушения объекта: так Qt завершает написание
            # файла и освобождает микрофон.
            if recorder.recorderState() != QMediaRecorder.RecorderState.StoppedState:
                recorder.stop()
            # Отключаем слоты: после разбора сигналы нам уже не нужны.
            try:
                recorder.recorderStateChanged.disconnect(self._on_recorder_state)
                recorder.errorOccurred.disconnect(self._on_recorder_error)
            except TypeError:
                # Сигнал уже не подключён (повторный разбор) — это нормально.
                pass
            recorder.deleteLater()

        for obj in (session, audio_input):
            if obj is not None:
                obj.deleteLater()

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is None:
            return
        # Экспорт ВНУТРЬ временного каталога запрещён (см. save_last_recording),
        # поэтому здесь только временные файлы — их безопасно удаляем целиком.
        # Копии, сохранённые НАРУЖУ, лежат вне каталога и не затрагиваются.
        try:
            shutil.rmtree(self._temp_dir)
        except OSError as exc:
            # Ошибку не гасим молча: так папка могла быть занята системой.
            logger.warning(
                "Не удалось удалить временную папку %s: %s",
                self._temp_dir,
                exc,
            )
        finally:
            self._temp_dir = None

    def _pick_format(self) -> QMediaFormat | None:
        """Выбирает контейнер/кодек, которые система умеет и записывать, и читать.

        Нужен именно контейнер (MPEG-4/M4A или WAV). «Голый» ADTS AAC
        (FileFormat.AAC) — это поток без контейнера: у такого файла нет
        длительности и корректного конца потока, и плеер «зависает» при
        воспроизведении. Поэтому выбираем реальный контейнер.
        """
        candidates = [
            (QMediaFormat.FileFormat.MPEG4, QMediaFormat.AudioCodec.AAC),
            (QMediaFormat.FileFormat.QuickTime, QMediaFormat.AudioCodec.AAC),
            (QMediaFormat.FileFormat.Wave, QMediaFormat.AudioCodec.Wave),
        ]
        mode = QMediaFormat.ConversionMode
        for file_format, audio_codec in candidates:
            fmt = QMediaFormat(file_format)
            fmt.setAudioCodec(audio_codec)
            if fmt.isSupported(mode.Encode) and fmt.isSupported(mode.Decode):
                return fmt
        return None

    def _extension(self) -> str:
        mapping = {
            QMediaFormat.FileFormat.MPEG4: ".m4a",
            QMediaFormat.FileFormat.QuickTime: ".mov",
            QMediaFormat.FileFormat.Wave: ".wav",
        }
        if self._format is None:
            return ".m4a"
        return mapping.get(self._format.fileFormat(), ".m4a")
