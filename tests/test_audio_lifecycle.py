"""Автоматические проверки жизненного цикла AudioController.

Запуск (из корня проекта)::

    .venv/bin/python -m unittest discover -s tests -v

— безопасные тесты контроллера (идемпотентность shutdown, сторожевые
флаги); к устройствам звука они не обращаются и проходят без них.

Полный прогон с реальными записью и воспроизведением (нужны микрофон,
выход звука и разрешения macOS)::

    RUN_AUDIO_TESTS=1 .venv/bin/python -m unittest discover -s tests -v

Семантика результатов:
* RUN_AUDIO_TESTS не задан — реальные аудиотесты SKIP без запроса
  устройств;
* устройство отсутствует — допустимая причина SKIP;
* отказ записи/воспроизведения уже после начала сценария, пустой файл
  записи и неверное состояние — FAIL: это дефект контроллера, а не
  особенность окружения.
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


def _audio_tests_enabled() -> bool:
    """Реальные аудиотесты включаются только явным флагом."""
    return os.environ.get("RUN_AUDIO_TESTS") == "1"


def _skip_if_audio_tests_disabled() -> None:
    raise unittest.SkipTest(
        "RUN_AUDIO_TESTS не равен 1 — реальные аудиотесты отключены"
    )


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


class TestSaveRecording(unittest.TestCase):
    """Сохранение последней записи для диагностики (без оборудования)."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-savetest-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"fake-audio-bytes")
        # «Завершённая запись» (в тесте допустимо трогать приватные поля;
        # окно и контроллер работают через публичный API).
        self.c._temp_dir = self.tmp
        self.c._record_file = self.rec
        self.dest_dir = tempfile.mkdtemp(prefix="english-companion-saved-")

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        for d in (self.tmp, self.dest_dir):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        del self.c

    def test_save_copies_recording_and_reports_path(self) -> None:
        dest = self.c.save_last_recording(self.dest_dir)
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"fake-audio-bytes")
        # Оригинал во временной папке до закрытия не тронут.
        self.assertTrue(os.path.exists(self.rec))
        self.assertIn("Запись сохранена", self.c.status)

    def test_saved_file_survives_shutdown(self) -> None:
        """После сохранения копия переживает закрытие; временные файлы уходят."""
        dest = self.c.save_last_recording(self.dest_dir)
        self.assertIsNotNone(dest)
        self.c.shutdown()
        _pump(0.2)
        self.assertTrue(
            os.path.exists(dest),
            f"сохранённая копия удалена при закрытии: {dest!r}",
        )
        self.assertFalse(os.path.isdir(self.tmp), "временная папка должна уйти")

    def test_saved_inside_temp_dir_survives_shutdown(self) -> None:
        """Даже копия внутри временной папки не удаляется при закрытии."""
        dest = self.c.save_last_recording(self.tmp)
        self.assertIsNotNone(dest)
        self.c.shutdown()
        _pump(0.2)
        self.assertTrue(
            os.path.exists(dest),
            f"копия во временной папке удалена при закрытии: {dest!r}",
        )

    def test_save_without_recording_returns_none(self) -> None:
        self.c._record_file = None
        self.assertIsNone(self.c.save_last_recording(self.dest_dir))

    def test_save_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertIsNone(self.c.save_last_recording(self.dest_dir))


class TestAudioPlayback(unittest.TestCase):
    """Воспроизведение: реальный выход (сценарий на тихом WAV)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _audio_tests_enabled():
            _skip_if_audio_tests_disabled()
        if not _has_audio_output():
            raise unittest.SkipTest("нет устройства вывода звука")
        super().setUpClass()

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-test-")
        self.wav = os.path.join(self.tmp, "silent.wav")
        _make_silent_wav(self.wav)
        self.c._temp_dir = self.tmp
        self.c._record_file = self.wav
        # Старт сценария. Если он не запускается, тест сам сообщит FAIL
        # (устройство же проверено в setUpClass) — а не пропустится.
        self.c.play()
        _pump(0.5)

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


class TestAudioRecording(unittest.TestCase):
    """Запись: реальный микрофон (и разрешения ОС)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _audio_tests_enabled():
            _skip_if_audio_tests_disabled()
        if not _has_audio_input():
            raise unittest.SkipTest(
                "нет микрофона — аудиотесты требуют оборудования"
            )
        super().setUpClass()

    def setUp(self) -> None:
        _app()
        self.c = AudioController()

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.2)
        temp_dir = self.c._temp_dir
        if temp_dir is not None and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        del self.c

    def test_shutdown_during_recording(self) -> None:
        """Закрытие окна во время записи: без падений, микрофон и файлы отпущены."""
        self.c.record()
        _pump(2.0)
        # Устройство проверено в setUpClass, значит «не стартовала» —
        # это отказ уже начавшегося сценария, а не SKIP.
        self.assertEqual(
            self.c.state,
            AudioController.RECORDING,
            f"запись не стартовала: {self.c.status!r}",
        )
        # Путь сохраняем до shutdown(): проверяем фактическое удаление
        # каталога, а не только None в поле контроллера.
        temp_dir = self.c._temp_dir
        record_file = self.c._record_file
        self.c.shutdown()
        _pump(0.5)
        self.assertFalse(
            os.path.isdir(temp_dir or ""),
            f"каталог записи не удалён: {temp_dir!r}",
        )
        if record_file is not None:
            self.assertFalse(
                os.path.exists(record_file),
                f"файл записи остался: {record_file!r}",
            )
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.shutdown()  # идемпотентность
        self.assertEqual(self.c.state, AudioController.IDLE)

    def test_record_play_cycles(self) -> None:
        """Несколько циклов record→stop→play→stop без висячих ссылок."""
        if not _has_audio_output():
            self.skipTest("нет устройства вывода — цикл record→play невозможен")
        for i in range(2):
            self.c.record()
            _pump(1.5)
            self.assertEqual(
                self.c.state,
                AudioController.RECORDING,
                f"цикл {i + 1}: запись не стартовала: {self.c.status!r}",
            )
            self.c.stop_recording()
            _pump(1.0)
            self.assertEqual(
                self.c.state,
                AudioController.IDLE,
                f"цикл {i + 1}: после stop_recording, status={self.c.status!r}",
            )
            # Пустой файл после успешной записи — это дефект, не SKIP.
            self.assertTrue(
                self.c.has_recording,
                f"цикл {i + 1}: файл записи пуст: {self.c.status!r}",
            )
            self.c.play()
            _pump(0.5)
            self.assertEqual(
                self.c.state,
                AudioController.PLAYING,
                f"цикл {i + 1}: play не стартовал: {self.c.status!r}",
            )
            self.c.stop_playback()
            _pump(0.5)
            self.assertEqual(
                self.c.state,
                AudioController.IDLE,
                f"цикл {i + 1}: после stop_playback, status={self.c.status!r}",
            )
        # Путь до shutdown(): проверяем, что каталог реально удалён.
        temp_dir = self.c._temp_dir
        self.c.shutdown()
        _pump(0.3)
        self.assertFalse(
            os.path.isdir(temp_dir or ""),
            f"каталог записи не удалён: {temp_dir!r}",
        )
