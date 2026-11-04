"""Автоматические проверки оркестрации SttEngine (без оборудования и веса).

Запуск::

    .venv/bin/python -m unittest discover -s tests -v

Используются подмены из ``tests/stt_fakes.py``: ни микрофон, ни интернет, ни
скачивание модели не требуются. Проверяется реальная логика ``SttEngine`` —
ленивая загрузка модели и её переиспользование, блокировка повторного запуска,
безопасный ``shutdown``, пустой и ошибочный результат.
"""

import faulthandler
import os
import sys
import tempfile
import unittest
import wave

faulthandler.enable()

# offscreen — чтобы Qt не требовал дисплей; аудио/распознавание не нужны.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.config import AppConfig  # noqa: E402
from tests.stt_fakes import FakeModel, FakeSttEngine  # noqa: E402

_QAPP: "QApplication | None" = None


def _app() -> "QApplication":
    global _QAPP
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication(sys.argv)
    return _QAPP


def _pump(seconds: float) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def _make_audio(path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)


class TestSttEngineLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        _app()
        self.config = AppConfig()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-stt-")
        self.audio_path = os.path.join(self.tmp, "clip.wav")
        _make_audio(self.audio_path)

    def tearDown(self) -> None:
        import shutil

        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_model_loaded_lazily_once_and_reused(self) -> None:
        """Модель грузится лениво один раз и переиспользуется."""
        stt = FakeSttEngine(self.config, result_text="hi")
        self.assertFalse(stt.model_loaded)

        stt.recognize_sync(self.audio_path)
        self.assertTrue(stt.model_loaded)
        first = stt._ensure_model()

        stt.recognize_sync(self.audio_path)
        second = stt._ensure_model()

        self.assertIs(first, second, "модель должна переиспользоваться")
        self.assertEqual(stt._fake_model.transcribe_calls, 2)

    def test_first_call_loads_then_transcribes(self) -> None:
        """Первый вызов: загрузка модели, затем распознавание; результат текст."""
        stt = FakeSttEngine(self.config, result_text="hello world")
        text = stt.recognize_sync(self.audio_path)
        self.assertEqual(text, "hello world")

    def test_empty_result_returns_empty_string(self) -> None:
        """Пустое распознавание — пустая строка (поле UI не очистим в UI-тесте)."""
        stt = FakeSttEngine(self.config, result_text="")
        self.assertEqual(stt.recognize_sync(self.audio_path), "")

    def test_transcribe_error_propagates_in_sync_mode(self) -> None:
        """Ошибка распознавания пробрасывается (в UI-потоке это сигнал failed)."""
        boom = RuntimeError("whisper exploded")
        stt = FakeSttEngine(self.config, raise_on_transcribe=boom)
        with self.assertRaises(RuntimeError):
            stt.recognize_sync(self.audio_path)

    def test_restart_blocked_while_busy(self) -> None:
        """Второй recognize во время работы игнорируется."""
        stt = FakeSttEngine(self.config, result_text="hi", delay=0.4)
        stt.recognize(self.audio_path)
        self.assertTrue(stt.is_busy, "после recognize движок должен быть занят")
        # Повторный запуск должен быть проигнорирован (не стартует новый поток).
        stt.recognize(self.audio_path)
        stt.shutdown()  # join рабочего потока
        _pump(0.1)
        self.assertFalse(stt.is_busy)
        self.assertEqual(
            stt._fake_model.transcribe_calls,
            1,
            "повторный запуск во время работы должен быть проигнорирован",
        )

    def test_recognize_refuses_missing_file(self) -> None:
        """Отсутствующий файл: сигнал failed, движок остаётся в IDLE."""
        stt = FakeSttEngine(self.config, result_text="hi")
        failures: list[str] = []
        stt.failed.connect(failures.append)
        stt.recognize("/nonexistent/path/does-not-exist.wav")
        _pump(0.2)
        self.assertEqual(len(failures), 1)
        self.assertIn("недоступен", failures[0])
        self.assertFalse(stt.is_busy)

    def test_shutdown_idempotent_and_joins_worker(self) -> None:
        """shutdown() ждёт завершения потока и идемпотентен."""
        stt = FakeSttEngine(self.config, result_text="hi", delay=0.2)
        stt.recognize(self.audio_path)
        stt.shutdown()  # должен дождаться рабочего потока
        self.assertFalse(stt.is_busy)
        stt.shutdown()  # повторный вызов безопасен
        self.assertFalse(stt.is_busy)

    def test_no_new_work_after_shutdown(self) -> None:
        """После shutdown() новые recognize отклоняются."""
        stt = FakeSttEngine(self.config, result_text="hi")
        stt.shutdown()
        stt.recognize(self.audio_path)
        _pump(0.1)
        self.assertFalse(stt.is_busy)
        self.assertFalse(stt.model_loaded, "после shutdown модель не грузится")
