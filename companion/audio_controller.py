import logging
import os
import shutil
import tempfile

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import (
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

    IDLE = "idle"
    RECORDING = "recording"
    PLAYING = "playing"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = self.IDLE
        self.status = "Голос: готов"

        self._format = self._pick_format()
        self._temp_dir: str | None = None
        self._record_file: str | None = None
        # Каждая запись получает новый файл. QMediaPlayer кэширует медиа-
        # источник по URL: если записывать поверх того же пути, setSource()
        # вернёт уже декодированную (старую) запись, а не перечитает файл.
        self._record_seq = 0

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

    @property
    def has_recording(self) -> bool:
        return (
            self._record_file is not None
            and os.path.exists(self._record_file)
            and os.path.getsize(self._record_file) > 0
        )

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

        device = QMediaDevices.defaultAudioInput()
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

        self._player.setSource(QUrl.fromLocalFile(self._record_file))
        self.state = self.PLAYING
        self.status = "Воспроизведение…"
        self.state_changed.emit()
        self._player.play()

    def stop_playback(self) -> None:
        if self._shutdown or self._player is None:
            return
        self._player.stop()
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Голос: готов"
            self.state_changed.emit()

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
        self._record_seq = 0
        self.state = self.IDLE
        self.status = "Голос: остановлен"

    # --- Обработчики сигналов ---------------------------------------------

    def _on_recorder_state(self, state: QMediaRecorder.RecorderState) -> None:
        # Поздний сигнал после shutdown не должен «оживлять» состояние.
        if self._shutdown:
            return
        if state == QMediaRecorder.RecorderState.RecordingState:
            self.state = self.RECORDING
            self.status = "Записываю… (нажми «Остановить»)"
            self.state_changed.emit()
        elif state == QMediaRecorder.RecorderState.StoppedState:
            self._finalize_recording()

    def _on_recorder_error(
        self, error: QMediaRecorder.Error, message: str
    ) -> None:
        if error == QMediaRecorder.Error.NoError or self._shutdown:
            return
        if self.state != self.RECORDING:
            return
        self.state = self.IDLE
        self.status = "Ошибка записи: " + (message or error.name)
        self._record_file = None
        self._teardown_recording()
        self.state_changed.emit()

    def _on_player_playing_changed(self, playing: bool) -> None:
        if self._shutdown:
            return
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
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._player_fail("не удалось открыть файл записи")

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError or self._shutdown:
            return
        self._player_fail(message or error.name)

    # --- Вспомогательные ---------------------------------------------------

    def _finalize_recording(self) -> None:
        if self.state == self.RECORDING:
            self.state = self.IDLE
            if self.has_recording:
                self.status = "Запись готова. Можно «Прослушать»."
            else:
                self.status = (
                    "Запись пуста — вероятно, нет доступа к микрофону. "
                    "Проверь разрешения macOS."
                )
                self._record_file = None
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
