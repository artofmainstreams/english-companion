"""Воспроизведение готового аудиофайла и потокового синтеза.

``AudioPlayer`` владеет ``QMediaPlayer`` и ``QAudioOutput``: ``play(path)``
воспроизводит любой готовый файл, ``stop()`` — останавливает, ``shutdown()``
безопасно и идемпотентно разбирает плеер и выход. Компонент не зависит от
рекордера и его приватных полей — это готовит аудиослой к озвучиванию
ответов: TTS-файл будет проигрываться так же, как запись.

Плеер НЕ удаляет и не перемещает переданный файл: судьбу файла решает его
владелец (например, ``AudioRecorder`` удаляет СВОИ временные файлы при
``shutdown``; пользовательский файл ему просто не принадлежит).

Потоковый вывод (речь частями) — отдельный путь поверх ``sounddevice``
(PortAudio): ``QAudioSink`` в этой сборке PySide6 не даёт писать куски
(нет ``write``/``bufferedAmount``), поэтому речь идёт из очереди
numpy-кусков в аудио-колбэк PortAudio (тот же механизм, что в проверенном
прототипе ``real_tts_stream_trial.py``). API: ``start_stream(sr)`` →
``write_chunk(chunk)`` (раз в приход) → ``end_stream()``; ``stop_stream()``
— остановка вместе с очередью. Пусто в очереди ДО конца речи — молчание
(подсчёт недочётов ``stream_underrun_samples``). После ``end_stream()`` и
опустошения очереди плеер сам переходит в IDLE. Полный WAV и поток взаимно
исключаются через общее состояние ``IDLE/PLAYING``.
"""

import logging
import os
import queue
import threading
import time

import numpy as np

from PySide6.QtCore import QObject, QUrl, QTimer, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer

logger = logging.getLogger(__name__)


class _StreamPlayer:
    """Потоковое воспроизведение кусков через ``sounddevice`` (PortAudio).

    Отдельный от ``QMediaPlayer`` путь: ``QAudioSink`` в этой сборке PySide6
    не даёт писать куски, а речь идёт из очереди numpy-кусков в аудио-колбэк
    PortAudio. Очередь — мост между потоком Qt (куда фасад кладёт куски
    ``write``) и аудио-потоком PortAudio (``callback`` — его «потребитель»).
    Пусто в очереди до конца речи — молчание (недочёт, считается); конец
    речи (``end``) + опустошение очереди — ``drained``, и владелец переводит
    плеер в IDLE (тумблер ``QTimer``, главный поток).
    """

    def __init__(self, owner, sample_rate: int, max_duration_s: float = 300.0) -> None:
        self._owner = owner
        self.sample_rate = int(sample_rate)
        self._queue: "queue.Queue" = queue.Queue()
        self._ended = False
        self._drained = False
        self._underrun = 0
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self.max_duration_s = max_duration_s
        # Реальный поток создаёт владелец (в тестах — фэйк): у него .stop()
        # и .close(), и он уже запущен (sd.InputStream стартует при создании).
        self._stream = owner._make_stream(self, self.sample_rate)
        self._timer = QTimer(owner)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # --- Покормить из потока Qt (фада) --------------------------------------

    def write(self, chunk) -> None:
        """Кладёт кусок в очередь воспроизведения (безопасно из любого потока)."""
        self._queue.put(np.asarray(chunk, dtype=np.float32).ravel())

    def end(self) -> None:
        """Куски больше не придут: после опустошения очереди — конец речи."""
        self._ended = True

    # --- Аудио-поток PortAudio ---------------------------------------------

    def callback(self, outdata, frames, time_info, status) -> None:
        """Запрашивает у PortAudio ``frames`` сэмплов (из очереди или молчание)."""
        n = 0
        while n < frames:
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                break
            take = min(chunk.size, frames - n)
            outdata[n:n + take, 0] = chunk[:take]
            if take < chunk.size:  # остаток куска — назад в очередь
                self._queue.put(chunk[take:])
            n += take
        if n < frames:  # недочёт или хвост конца речи — дописываем тишиной
            outdata[n:, 0] = 0.0
            if not self._ended:  # конец речи — хвост тишины, НЕ недочёт
                with self._lock:
                    self._underrun += (frames - n)
        if self._ended and self._queue.empty():
            with self._lock:
                self._drained = True

    @property
    def drained(self) -> bool:
        with self._lock:
            return self._drained

    @property
    def underrun(self) -> int:
        with self._lock:
            return self._underrun

    # --- Контроль (главный поток) ------------------------------------------

    def _poll(self) -> None:
        if self._drained:
            self._timer.stop()
            self._owner._on_stream_ended(timeout=False)
            return
        if time.monotonic() - self._started > self.max_duration_s:
            self._timer.stop()
            self._owner._on_stream_ended(timeout=True)

    def stop(self) -> None:
        """Останавливает поток и таймер (безопасно повторять)."""
        self._timer.stop()
        for op in ("stop", "close"):
            try:
                getattr(self._stream, op, lambda: None)()
            except Exception:  # разбор не должен падать из-за PortAudio
                pass


class AudioPlayer(QObject):
    """Воспроизводит готовый аудиофайл, не блокируя интерфейс."""

    state_changed = Signal()

    IDLE = "idle"
    PLAYING = "playing"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = self.IDLE
        self.status = "Голос: готов"

        # Базовая точка времени для диагностических событий (элипсис в мс).
        self._t0 = time.monotonic()

        # На этой сборке Qt у QMediaPlayer нет «по умолчанию» подключённого
        # выхода (audioOutput() == None); явный QAudioOutput создаём при play().
        self._audio_output: QAudioOutput | None = None
        self._player: QMediaPlayer | None = QMediaPlayer(self)
        self._player.playingChanged.connect(self._on_player_playing_changed)
        self._player.mediaStatusChanged.connect(self._on_player_media_status)
        self._player.errorOccurred.connect(self._on_player_error)

        # Потоковый вывод (речь частями): активный _StreamPlayer или None.
        # Полный WAV (QMediaPlayer) и поток взаимно исключаются по state.
        self._stream_player: "_StreamPlayer | None" = None

        # Флаг shutdown: после него плеер не восстанавливается.
        self._shutdown = False

    # --- Публичный API ------------------------------------------------------

    @property
    def can_play(self) -> bool:
        """Готов принять ``play(path)``: не остановлен и не играет."""
        return not self._shutdown and self.state == self.IDLE

    @property
    def is_shutdown(self) -> bool:
        """``True`` после ``shutdown()``: плеер больше не работает."""
        return self._shutdown

    def play(self, path: str) -> None:
        """Воспроизводит готовый аудиофайл ``path``.

        Файл не удаляется и не перемещается — владелец файла решает его
        судьбу. Повторный ``play`` во время воспроизведения игнорируется.
        """
        if self._shutdown or self.state != self.IDLE or self._player is None:
            return
        if not path or not os.path.exists(path):
            self.status = "Нет аудиофайла для воспроизведения."
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
        self._log_event(
            "play:start",
            output=device.description(),
            source=os.path.basename(path),
        )

        self._player.setSource(QUrl.fromLocalFile(path))
        self.state = self.PLAYING
        self.status = "Воспроизведение…"
        self.state_changed.emit()
        self._player.play()

    def stop(self) -> None:
        """Останавливает воспроизведение. Файл на диске не трогает."""
        if self._shutdown or self._player is None:
            return
        self._log_event("play:stop")
        self._player.stop()
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Голос: готов"
            self.state_changed.emit()

    # --- Потоковый вывод (речь частями) -------------------------------------

    @property
    def stream_active(self) -> bool:
        """Идёт ли потоковый вывод (речь частями)."""
        return self._stream_player is not None

    def stream_underrun_samples(self) -> int:
        """Сколько сэмплов тишины вставлено (недочёты) с начала потока."""
        sp = self._stream_player
        return sp.underrun if sp is not None else 0

    def start_stream(self, sample_rate: int) -> None:
        """Начинает потоковый вывод при заданной частоте дискретизации.

        Применим только из IDLE (взаимное исключение с полным WAV и
        записью); иначе игнорируется. Куски кладутся ``write_chunk``,
        конец — ``end_stream``.
        """
        if self._shutdown or self.state != self.IDLE:
            return
        try:
            self._stream_player = _StreamPlayer(self, sample_rate)
        except Exception as exc:  # нет PortAudio/устройства → не запускаем
            logger.warning("stream: не удалось открыть поток", exc_info=True)
            self.status = "Не удалось начать воспроизведение."
            self.state_changed.emit()
            return
        self.state = self.PLAYING
        self.status = "Воспроизведение…"
        self._log_event("stream:start", sr=sample_rate)
        self.state_changed.emit()

    def write_chunk(self, chunk) -> None:
        """Кладёт очередной кусок речи в очередь потока (если поток активен)."""
        sp = self._stream_player
        if sp is not None:
            sp.write(chunk)

    def end_stream(self) -> None:
        """Куски больше не придут: после опустошения очереди — конец речи."""
        sp = self._stream_player
        if sp is not None:
            sp.end()

    def stop_stream(self) -> None:
        """Останавливает поток и сбрасывает очередь (без ожидания конца)."""
        if self._stream_player is None:
            return
        self._log_event(
            "stream:stop", underrun=self.stream_underrun_samples()
        )
        self._teardown_stream()
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Голос: готов"
            self.state_changed.emit()

    def shutdown(self) -> None:
        """Останавливает плеер и разбирает выход. Идемпотентна.

        Переданный ранее файл НЕ удаляется: владелец файла (например,
        ``AudioRecorder``) управляет им отдельно.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # Поток (если шёл): остановка PortAudio + таймера — до всего остального.
        self._teardown_stream()

        # Выход: явное отсоединение от плеера (setAudioOutput(None)) и
        # снятие связей, затем отложенное удаление. Раньше здесь был
        # только deleteLater(), и корректность зависела от порядка
        # очереди Qt; явное отсоединение сделало разбор независимым от него.
        self._release_audio_output()

        # Плеер: сначала остановка, затем отключение источника, чтобы
        # плеер отпустил файл до удаления. Уничтожаем после обеих команд.
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
            self._player.deleteLater()
            self._player = None

        self.state = self.IDLE
        self.status = "Голос: остановлен"
        self.state_changed.emit()

    # --- Внутренние ---------------------------------------------------------

    def _log_event(self, name: str, **kv) -> None:
        """Диагностическое событие с элипсисом (мс от старта компонента)."""
        try:
            elapsed_ms = int((time.monotonic() - self._t0) * 1000)
        except (AttributeError, ValueError):
            elapsed_ms = -1
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info("[audio] %sms %s%s", elapsed_ms, name, (" " + extra) if extra else "")

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

    def _teardown_stream(self) -> None:
        """Разбирает активный поток (остановка + таймер). Идемпотентно."""
        sp = self._stream_player
        self._stream_player = None
        if sp is not None:
            sp.stop()

    def _on_stream_ended(self, timeout: bool = False) -> None:
        """Поток опустел (или тайм-аут): плеер переходит в IDLE (главный поток)."""
        if self._shutdown:
            return
        underrun = self.stream_underrun_samples()
        self._teardown_stream()
        self._log_event("stream:ended", timeout=timeout, underrun=underrun)
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = (
                "Голос: готов"
                if not timeout
                else "Воспроизведение: тайм-аут"
            )
        self.state_changed.emit()

    def _make_stream(self, sp: "_StreamPlayer", sample_rate: int):
        """Создаёт и ЗАПУСКАЕТ аудио-поток (тестовый шов): ``sounddevice``.

        ``sounddevice`` импортируется ЛЕНИВО (внутри), чтобы компонент и
        тесты работали без PortAudio; в тестах метод подменяется фэйком.
        ВАЖНО: ``sd.InputStream`` при создании НЕ авто-стартует
        (``active`` False), поэтому callback не вызывается, пока не
        вызван ``start()`` — без него ни звука, ни ``drained`` (и плеер
        зависает в PLAYING). Отсюда явный ``start()``.
        """
        import sounddevice as sd  # лениво: ядро не требует PortAudio

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=sp.callback,
        )
        stream.start()  # без этого callback не сработает (active==False)
        return stream

    # --- Обработчики сигналов -----------------------------------------------

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
            self._player_fail("не удалось открыть файл")

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError or self._shutdown:
            return
        self._log_event("player_error", error=str(error), message=message)
        self._player_fail(message or error.name)

    def _player_fail(self, message: str) -> None:
        if self.state == self.PLAYING:
            self.state = self.IDLE
            self.status = "Ошибка воспроизведения: " + message
            self.state_changed.emit()
