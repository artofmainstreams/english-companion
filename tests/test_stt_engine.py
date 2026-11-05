"""Автоматические проверки оркестрации SttEngine (без оборудования и веса).

Запуск::

    .venv/bin/python -m unittest discover -s tests -v

Оркестрация проверяется с подменным backend'ом (``tests/stt_fakes.py``):
ни микрофона, ни интернета, ни скачивания модели не требуются, и без
наследования ради подмены приватных методов. Проверяется реальная логика
``SttEngine`` — ленивая загрузка модели и её переиспользование, сигналы,
блокировка повторного запуска, безопасный ``shutdown``, пустой и ошибочный
результат (распознавания и загрузки), общий sync/async затвор.
"""

import faulthandler
import os
import sys
import tempfile
import time
import unittest
import wave

faulthandler.enable()

# offscreen — чтобы Qt не требовал дисплей; аудио/распознавание не нужны.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.config import AppConfig  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402
from tests.stt_fakes import FakeBackend  # noqa: E402

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


def _wait_stt_idle(stt, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while stt.is_busy and time.monotonic() < deadline:
        _pump(0.02)
    _pump(0.1)


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
        backend = FakeBackend(result_text="hi")
        stt = SttEngine(backend)
        self.assertFalse(stt.model_loaded)

        stt.recognize_sync(self.audio_path)
        self.assertTrue(stt.model_loaded)
        first = stt._ensure_model()

        stt.recognize_sync(self.audio_path)
        second = stt._ensure_model()

        self.assertIs(first, second, "модель должна переиспользоваться")
        self.assertEqual(backend.transcribe_calls, 2)
        stt.shutdown()

    def test_first_call_loads_then_transcribes(self) -> None:
        """Первый вызов: загрузка модели, затем распознавание; результат текст."""
        backend = FakeBackend(result_text="hello world")
        stt = SttEngine(backend)
        text = stt.recognize_sync(self.audio_path)
        self.assertEqual(text, "hello world")
        stt.shutdown()

    def test_empty_result_returns_empty_string(self) -> None:
        """Пустое распознавание — пустая строка (поле UI не очистим в UI-тесте)."""
        backend = FakeBackend(result_text="")
        stt = SttEngine(backend)
        self.assertEqual(stt.recognize_sync(self.audio_path), "")
        stt.shutdown()

    def test_success_emits_recognized_signal(self) -> None:
        """Успех в UI-режиме: сигнал recognized с текстом, состояние IDLE."""
        backend = FakeBackend(result_text="hi")
        stt = SttEngine(backend)
        seen: list[str] = []
        stt.recognized.connect(seen.append)
        stt.recognize(self.audio_path)
        _wait_stt_idle(stt)
        self.assertEqual(seen, ["hi"])
        self.assertEqual(stt.state, SttEngine.IDLE)
        stt.shutdown()

    def test_transcribe_error_propagates_in_sync_mode(self) -> None:
        """Ошибка распознавания пробрасывается (в UI-потоке это сигнал failed)."""
        boom = RuntimeError("whisper exploded")
        backend = FakeBackend(raise_on_transcribe=boom)
        stt = SttEngine(backend)
        with self.assertRaises(RuntimeError):
            stt.recognize_sync(self.audio_path)
        stt.shutdown()

    def test_load_error_sync_propagates_and_releases(self) -> None:
        """Ошибка загрузки модели пробрасывается в sync-режиме и не оставляет
        движок «занятым» (без fallback на другой бэкенд)."""
        boom = RuntimeError(
            "Бэкенд MLX недоступен: … стt_backend='faster-whisper'"
        )
        backend = FakeBackend(raise_on_load=boom)
        stt = SttEngine(backend)
        with self.assertRaises(RuntimeError):
            stt.recognize_sync(self.audio_path)
        self.assertFalse(stt.is_busy, "сбой загрузки оставил движок занятым")
        self.assertFalse(stt.model_loaded, "fallback на другой бэкенд запрещён")
        self.assertEqual(backend.load_calls, 1, "повторной попытки загрузки нет")

    def test_load_error_emits_failed_and_stays_idle(self) -> None:
        """В UI-режиме ошибка загрузки модели приходит сигналом failed."""
        boom = RuntimeError(
            "Бэкенд MLX недоступен: … стt_backend='faster-whisper'"
        )
        backend = FakeBackend(raise_on_load=boom)
        stt = SttEngine(backend)
        failures: list[str] = []
        stt.failed.connect(failures.append)
        stt.recognize(self.audio_path)
        _wait_stt_idle(stt)
        self.assertEqual(len(failures), 1)
        self.assertIn("Бэкенд MLX недоступен", failures[0])
        self.assertEqual(stt.state, SttEngine.IDLE)
        stt.shutdown()

    def test_restart_blocked_while_busy(self) -> None:
        """Второй recognize во время работы игнорируется."""
        backend = FakeBackend(result_text="hi", delay=0.4)
        stt = SttEngine(backend)
        stt.recognize(self.audio_path)
        self.assertTrue(stt.is_busy, "после recognize движок должен быть занят")
        # Повторный запуск должен быть проигнорирован (не стартует новый поток).
        stt.recognize(self.audio_path)
        stt.shutdown()  # дождётся рабочего потока
        _pump(0.1)
        self.assertFalse(stt.is_busy)
        self.assertEqual(
            backend.transcribe_calls,
            1,
            "повторный запуск во время работы должен быть проигнорирован",
        )

    def test_recognize_refuses_missing_file(self) -> None:
        """Отсутствующий файл: сигнал failed, движок остаётся в IDLE."""
        backend = FakeBackend(result_text="hi")
        stt = SttEngine(backend)
        failures: list[str] = []
        stt.failed.connect(failures.append)
        stt.recognize("/nonexistent/path/does-not-exist.wav")
        _pump(0.2)
        self.assertEqual(len(failures), 1)
        self.assertIn("недоступен", failures[0])
        self.assertFalse(stt.is_busy)
        stt.shutdown()

    def test_shutdown_idempotent_and_joins_worker(self) -> None:
        """shutdown() ждёт завершения потока и идемпотентен."""
        backend = FakeBackend(result_text="hi", delay=0.2)
        stt = SttEngine(backend)
        stt.recognize(self.audio_path)
        stt.shutdown()  # должен дождаться рабочего потока
        self.assertFalse(stt.is_busy)
        stt.shutdown()  # повторный вызов безопасен
        self.assertFalse(stt.is_busy)

    def test_close_during_processing(self) -> None:
        """Закрытие во время обработки: shutdown дожидается потока,
        новая работа после этого отклоняется, модель не грузится повторно."""
        backend = FakeBackend(result_text="hi", delay=0.3)
        stt = SttEngine(backend)
        stt.recognize(self.audio_path)
        stt.shutdown()  # должен дождаться рабочего потока
        self.assertFalse(stt.is_busy)
        self.assertEqual(stt.state, SttEngine.IDLE)

        # После закрытия новые recognize отклоняются.
        stt.recognize(self.audio_path)
        _pump(0.1)
        self.assertFalse(stt.is_busy)
        self.assertEqual(backend.load_calls, 1, "после shutdown модель не грузится")
        self.assertEqual(backend.transcribe_calls, 1)
        # Файл-вход не тронут самим движком (его освобождает AudioController
        # после завершения — порядок проверен в тесте окна).
        self.assertTrue(os.path.exists(self.audio_path))

    def test_no_new_work_after_shutdown(self) -> None:
        """После shutdown() новые recognize отклоняются."""
        backend = FakeBackend(result_text="hi")
        stt = SttEngine(backend)
        stt.shutdown()
        stt.recognize(self.audio_path)
        _pump(0.1)
        self.assertFalse(stt.is_busy)
        self.assertFalse(stt.model_loaded, "после shutdown модель не грузится")


class TestSyncGuard(unittest.TestCase):
    """Общая защита sync/async (регрессия дефекта 3): после ``shutdown`` и
    во время работы новые работы отклоняются; исключение не оставляет движок
    «занятым»; блокировка состояния свободна на время тяжёлых вызовов.
    """

    def setUp(self) -> None:
        _app()
        self.config = AppConfig()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-sync-")
        self.audio_path = os.path.join(self.tmp, "clip.wav")
        _make_audio(self.audio_path)

    def tearDown(self) -> None:
        import shutil

        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sync_rejected_after_shutdown(self) -> None:
        """recognize_sync после shutdown отклоняется (было: работало)."""
        backend = FakeBackend(result_text="hi")
        stt = SttEngine(backend)
        stt.shutdown()
        with self.assertRaises(RuntimeError):
            stt.recognize_sync(self.audio_path)
        self.assertFalse(stt.is_busy)
        self.assertFalse(stt.model_loaded, "после shutdown модель не грузится")

    def test_sync_rejected_while_async_busy(self) -> None:
        """recognize_sync во время асинхронной работы отклоняется
        (было: шла параллельно, обходя ограничение «одно в работе»)."""
        backend = FakeBackend(result_text="hi", delay=0.6)
        stt = SttEngine(backend)
        stt.recognize(self.audio_path)  # асинхронная работа идёт
        self.assertTrue(stt.is_busy, "после recognize движок должен быть занят")
        with self.assertRaises(RuntimeError):
            stt.recognize_sync(self.audio_path)
        stt.shutdown()
        self.assertFalse(stt.is_busy)
        self.assertEqual(
            backend.transcribe_calls, 1, "стартовала вторая работа"
        )

    def test_sync_error_does_not_leave_busy(self) -> None:
        """Сбой sync не «заклинивает» движок: следующая работа проходит claim."""
        backend = FakeBackend(
            raise_on_transcribe=RuntimeError("whisper-boom")
        )
        stt = SttEngine(backend)
        with self.assertRaises(RuntimeError):
            stt.recognize_sync(self.audio_path)
        self.assertFalse(stt.is_busy, "исключение оставило движок занятым")
        # Следующий вызов не отклоняется по «занятости»: ошибается уже
        # распознавание, а не guard.
        with self.assertRaises(RuntimeError) as ctx:
            stt.recognize_sync(self.audio_path)
        self.assertNotIn("занят", str(ctx.exception), "движок застрял «занятым»")
        self.assertIn("whisper-boom", str(ctx.exception))

    def test_state_lock_free_during_heavy_work(self) -> None:
        """Блокировка состояния свободна, пока идёт работа (не удерживается
        на время тяжёлых вызовов — загрузка модели и распознавание)."""
        backend = FakeBackend(result_text="hi", delay=0.8)
        stt = SttEngine(backend)
        stt.recognize(self.audio_path)  # асинхронная работа (задержка 0.8s)
        _pump(0.05)  # даём потоку дойти до тяжёлого вызова
        self.assertTrue(stt.is_busy)
        acquired = stt._state_lock.acquire(timeout=0.1)
        self.assertTrue(acquired, "блокировка удерживается на время работы")
        stt._state_lock.release()
        stt.shutdown()
        self.assertFalse(stt.is_busy)


if __name__ == "__main__":
    unittest.main()
