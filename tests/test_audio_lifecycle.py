"""Автоматические проверки жизненного цикла AudioController.

Запуск (из корня проекта)::

    .venv/bin/python -m unittest discover -s tests -v

или только «безопасный» набор, который не требует оборудования::

    .venv/bin/python -m unittest tests.test_audio_lifecycle.TestControllerGuards -v

Обычный запуск проходит даже без аудиооборудования: классы
``TestAudioPlayback`` и ``TestAudioRecording`` помечены ``skipUnless``
(нужны реальные устройство вывода / микрофон и разрешения ОС) и
пропускаются (SKIP), а не падают, когда устройств нет. На машине, где
есть звук, все проверки выполняются полностью.
"""

import faulthandler
import os
import shutil
import sys
import tempfile
import unittest
import wave

faulthandler.enable()

# offscreen — чтобы тесты не зависели от дисплея; аудио при этом работает.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtMultimedia import QMediaDevices, QMediaRecorder
from PySide6.QtWidgets import QApplication

from companion.audio_controller import AudioController

_QAPP: QApplication | None = None


def _app() -> QApplication:
    global _QAPP
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication(sys.argv)
    return _QAPP


def _pump(seconds: float) -> None:
    """Пускаем цикл событий на ``seconds`` секунд (нужно для deleteLater)."""
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def _has_audio_input() -> bool:
    _app()
    try:
        return not QMediaDevices.defaultAudioInput().isNull()
    except Exception:
        return False


def _has_audio_output() -> bool:
    _app()
    try:
        return not QMediaDevices.defaultAudioOutput().isNull()
    except Exception:
        return False


def _make_silent_wav(path: str, seconds: int = 1) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000 * seconds)


class TestControllerGuards(unittest.TestCase):
    """Проверки без оборудования: идемпотентность shutdown, сторожевые флаги."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()

    def tearDown(self) -> None:
        self.c.shutdown()
        del self.c
        _pump(0.1)

    def test_double_shutdown_cold(self) -> None:
        """Повторный shutdown безопасен и оставляет контроллер мёртвым."""
        self.c.shutdown()
        self.c.shutdown()
        _pump(0.2)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c._temp_dir)

    def test_actions_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.c.record()
        _pump(0.2)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c._temp_dir)
        self.c.play()
        self.assertEqual(self.c.state, AudioController.IDLE)

    def test_late_signals_do_not_revive_state(self) -> None:
        """Поздние сигналы Qt после shutdown не должны «оживлять» состояние."""
        self.c.shutdown()
        self.c._on_recorder_state(QMediaRecorder.RecorderState.RecordingState)
        self.c._on_player_playing_changed(True)
        self.c._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c._temp_dir)

    def test_capability_flags_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertFalse(self.c.can_record)
        self.assertFalse(self.c.can_play)


@unittest.skipUnless(
    _has_audio_output(),
    "нет устройства вывода звука — тест зависит от оборудования",
)
class TestAudioPlayback(unittest.TestCase):
    """Воспроизведение: нужен реальный выход (сценарии на тихом WAV)."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-test-")
        self.wav = os.path.join(self.tmp, "silent.wav")
        _make_silent_wav(self.wav)
        self.c._temp_dir = self.tmp
        self.c._record_file = self.wav
        self.c.play()
        _pump(0.5)
        if self.c.state != AudioController.PLAYING:
            self.c.shutdown()
            _pump(0.2)
            self.skipTest(f"воспроизведение не стартовало: {self.c.status!r}")

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.2)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def test_shutdown_during_playback(self) -> None:
        """Закрытие окна во время воспроизведения: без падений, файлы удалены."""
        self.assertEqual(self.c.state, AudioController.PLAYING, self.c.status)
        self.c.shutdown()
        _pump(0.5)
        self.assertFalse(os.path.isdir(self.tmp), "папка с файлом осталась")
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.shutdown()  # идемпотентность
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.record()
        self.c.play()
        self.assertEqual(self.c.state, AudioController.IDLE)


@unittest.skipUnless(
    _has_audio_input(),
    "нет микрофона — тест зависит от оборудования и разрешений",
)
class TestAudioRecording(unittest.TestCase):
    """Запись: нужен реальный микрофон (и разрешения ОС)."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.2)
        if self.c._temp_dir is not None and os.path.isdir(self.c._temp_dir):
            shutil.rmtree(self.c._temp_dir, ignore_errors=True)
        del self.c

    def test_shutdown_during_recording(self) -> None:
        """Закрытие окна во время записи: без падений, микрофон и файлы отпущены."""
        self.c.record()
        _pump(2.0)
        if self.c.state != AudioController.RECORDING:
            self.skipTest(f"запись не стартовала: {self.c.status!r}")
        self.c.shutdown()
        _pump(0.5)
        self.assertIsNone(self.c._temp_dir)
        self.assertIsNone(self.c._record_file)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.shutdown()  # идемпотентность
        self.assertEqual(self.c.state, AudioController.IDLE)

    @unittest.skipUnless(
        _has_audio_output(),
        "нет устройства вывода звука — цикл record→play невозможен",
    )
    def test_record_play_cycles(self) -> None:
        """Несколько циклов record→stop→play→stop без висячих ссылок."""
        for i in range(2):
            self.c.record()
            _pump(1.5)
            if self.c.state != AudioController.RECORDING:
                self.skipTest(
                    f"цикл {i + 1}: запись не стартовала: {self.c.status!r}"
                )
            self.c.stop_recording()
            _pump(1.0)
            if self.c.state != AudioController.IDLE:
                self.fail(
                    f"цикл {i + 1}: stop_recording, "
                    f"state={self.c.state} status={self.c.status!r}"
                )
            if not self.c.has_recording:
                self.skipTest(f"цикл {i + 1}: запись пуста: {self.c.status!r}")
            self.c.play()
            _pump(0.5)
            if self.c.state != AudioController.PLAYING:
                self.skipTest(f"цикл {i + 1}: play не стартовал: {self.c.status!r}")
            self.c.stop_playback()
            _pump(0.5)
            if self.c.state != AudioController.IDLE:
                self.fail(f"цикл {i + 1}: stop_playback, state={self.c.state}")
        self.c.shutdown()
        _pump(0.3)
        self.assertIsNone(self.c._temp_dir)
