"""Запись речи: устройства ввода, сессия записи, результат финализации.

``AudioRecorder`` владеет устройствами ввода (список/выбор/hotplug),
графом записи (``QAudioInput``/``QMediaRecorder``/``QMediaCaptureSession``)
и результатом успешно завершённой записи: ``has_recording``,
``last_recording_path`` и сигнал ``recording_completed(path)``
(успешное завершение записи — интерфейс по нему автоматически запускает
распознавание). Записанная временная запись живёт, пока нужна для
распознавания/прослушивания; экспорт записи наружу удалён — временные
файлы удаляются штатно в ``shutdown()``.

Компонент является ЯВНЫМ ВЛАДЕЛЬЦЕМ своих временных файлов: создаёт
временный каталог при первой записи и удаляет его в ``shutdown()``.
Воспроизведением записи он не занимается — это ``AudioPlayer``.
"""

import logging
import os
import shutil
import tempfile
import time

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import (
    QAudioDevice,
    QAudioInput,
    QMediaCaptureSession,
    QMediaDevices,
    QMediaFormat,
    QMediaRecorder,
)

logger = logging.getLogger(__name__)


class AudioRecorder(QObject):
    """Записывает речь без блокировки интерфейса."""

    state_changed = Signal()
    # Список устройств ввода изменился (подключение/отключение микрофона):
    # интерфейс перечитывает список; если выбранный микрофон пропал,
    # компонент сам возвращается к системному устройству.
    devices_changed = Signal()
    # Успешно завершённая запись готова (файл существует и непуст): интерфейс
    # по этому сигналу автоматически запускает распознавание по этому файлу
    # (одна запись — один запуск STT). Сбой/пустая запись сигнал НЕ подаёт.
    recording_completed = Signal(str)

    IDLE = "idle"
    RECORDING = "recording"

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

        # Флаг shutdown: после него компонент не восстанавливается.
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
    def is_shutdown(self) -> bool:
        """``True`` после ``shutdown()``: компонент больше не работает."""
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
        компонент остановлен, идёт запись (переключение запрещено)
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

    def shutdown(self) -> None:
        """Останавливает запись и удаляет СВОИ временные файлы. Идемпотентна.

        После вызова компонент не работает: ``record()`` отклоняется,
        поздние сигналы Qt игнорируются. Записанная временная запись, пока она
        нужна для распознавания/прослушивания, живёт; после shutdown её файл
        и временный каталог удаляются штатно.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # Снимаем обработчик hotplug-сигнала и отпускаем экземпляр
        # QMediaDevices, чтобы поздние события не доходили до мёртвого
        # компонента.
        if self._devices is not None:
            try:
                self._devices.audioInputsChanged.disconnect(
                    self._on_input_devices_changed
                )
            except (TypeError, RuntimeError):
                pass  # сигнал уже снят (повторный shutdown) — нормально

        # Запись: стоп до разрушения объекта, затем разбор графа.
        self._teardown_recording()

        # Временные файлы: компонент — их владелец.
        self._cleanup_temp_dir()

        self._record_file = None
        self._ready_recording = None
        self._record_seq = 0
        self.state = self.IDLE
        self.status = "Голос: остановлен"
        self.state_changed.emit()

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

    # --- Вспомогательные ---------------------------------------------------

    def _log_event(self, name: str, **kv) -> None:
        """Диагностическое событие с элипсисом (мс от старта компонента).

        Помогает восстановить таймлайн «завершение записи → освобождение
        входа → разбор сессии». Включается уровнем logging (INFO).
        """
        try:
            elapsed_ms = int((time.monotonic() - self._t0) * 1000)
        except (AttributeError, ValueError):
            elapsed_ms = -1
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info("[audio] %sms %s%s", elapsed_ms, name, (" " + extra) if extra else "")

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
                # не публикуем ни в компоненте, ни в UI.
                self.status = "Ошибка записи — файл не считается готовым."
                self._ready_recording = None
                self._record_file = None
            elif self._valid_recording_file(self._record_file):
                # Готовность публикуем только после подтверждения завершения —
                # «пишущийся» файл (состояние RECORDING) сюда не попадает.
                self._ready_recording = self._record_file
                self.status = "Запись готова. Распознаю речь…"
                # Успешное завершение: интерфейс по сигналу сам запускает STT
                # по ЭТОМУ файлу (одна запись — один запуск).
                self.recording_completed.emit(self._ready_recording)
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
        # Здесь только временные файлы записи — их безопасно удаляем целиком
        # (экспорт наружу удалён, поэтому копий вне каталога уже нет).
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
