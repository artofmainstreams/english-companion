"""Воспроизведение готового аудиофайла.

``AudioPlayer`` владеет ``QMediaPlayer`` и ``QAudioOutput``: ``play(path)``
воспроизводит любой готовый файл, ``stop()`` — останавливает, ``shutdown()``
безопасно и идемпотентно разбирает плеер и выход. Компонент не зависит от
рекордера и его приватных полей — это готовит аудиослой к озвучиванию
ответов: TTS-файл будет проигрываться так же, как запись.

Плеер НЕ удаляет и не перемещает переданный файл: судьбу файла решает его
владелец (например, ``AudioRecorder`` удаляет СВОИ временные файлы при
``shutdown``; пользовательский файл ему просто не принадлежит).
"""

import logging
import os
import time

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer

logger = logging.getLogger(__name__)


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

    def shutdown(self) -> None:
        """Останавливает плеер и разбирает выход. Идемпотентна.

        Переданный ранее файл НЕ удаляется: владелец файла (например,
        ``AudioRecorder``) управляет им отдельно.
        """
        if self._shutdown:
            return
        self._shutdown = True

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
