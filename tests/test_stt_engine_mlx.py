"""Проверки MLX-бэкенда (по умолчанию) — с подменой, без весов и интернета.

Запуск::

    .venv/bin/python -m unittest discover -s tests -v

Ни микрофона, ни скачивания весов, ни реального MLX не требуется:
``FakeSttEngine`` подменяет создание модели и распознавание, а путь
отсутствующих зависимостей проверяется точечной подменой
``SttEngine._require_mlx_imports``.

Проверяются:
* дефолты AppConfig (mlx + whisper-large-v3-mlx, language=en, task=transcribe);
* обычный запуск и сигналы успеха;
* повторное использование модели (загрузка один раз);
* блокировка повторного запуска;
* ошибки: сигнал failed, состояние IDLE, отсутствие автоматического fallback;
* неизвестный бэкенд — явная ошибка (ValueError), а не смена движка;
* закрытие во время обработки: shutdown дожидается потока, новая работа
  отклоняется, модель не грузится повторно.
"""

import faulthandler
import os
import shutil
import sys
import tempfile
import time
import unittest
import wave
from unittest import mock

faulthandler.enable()

# offscreen — чтобы Qt не требовал дисплей; аудио/распознавание не нужны.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.config import AppConfig  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402
from tests.stt_fakes import FakeSttEngine  # noqa: E402

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


class MlxTestBase(unittest.TestCase):
    def setUp(self) -> None:
        _app()
        self.config = AppConfig()  # дефолт — mlx-бэкенд
        self.tmp = tempfile.mkdtemp(prefix="english-companion-stt-mlx-")
        self.audio_path = os.path.join(self.tmp, "clip.wav")
        _make_audio(self.audio_path)

    def tearDown(self) -> None:
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)


class TestMlxDefaults(MlxTestBase):
    def test_default_backend_is_mlx_large_v3(self) -> None:
        """По умолчанию: mlx + whisper-large-v3-mlx, en/transcribe."""
        config = AppConfig()
        self.assertEqual(config.stt_backend, "mlx")
        self.assertEqual(
            config.stt_mlx_model, "mlx-community/whisper-large-v3-mlx"
        )
        self.assertEqual(config.stt_language, "en")
        self.assertEqual(config.stt_task, "transcribe")

    def test_faster_whisper_is_explicit_option(self) -> None:
        """faster-whisper выбирается явно; свои параметры сохранены."""
        config = AppConfig(stt_backend="faster-whisper")
        self.assertEqual(config.stt_backend, "faster-whisper")
        self.assertEqual(config.stt_whisper_model, "base.en")


class TestMlxPathOrchestration(MlxTestBase):
    """Реальная оркестрация SttEngine при mlx-конфигурации (модель — фейк)."""

    def test_normal_run_returns_text(self) -> None:
        stt = FakeSttEngine(self.config, result_text="hello mlx")
        self.assertEqual(stt.recognize_sync(self.audio_path), "hello mlx")
        stt.shutdown()

    def test_success_emits_recognized_signal(self) -> None:
        stt = FakeSttEngine(self.config, result_text="hi mlx")
        seen: list[str] = []
        stt.recognized.connect(seen.append)
        stt.recognize(self.audio_path)
        _wait_stt_idle(stt)
        self.assertEqual(seen, ["hi mlx"])
        self.assertEqual(stt.state, SttEngine.IDLE)
        stt.shutdown()

    def test_model_loaded_once_and_reused(self) -> None:
        """Модель загружается один раз и переиспользуется между запусками."""
        stt = FakeSttEngine(self.config, result_text="hi")
        stt.recognize_sync(self.audio_path)
        stt.recognize_sync(self.audio_path)
        self.assertTrue(stt.model_loaded)
        self.assertEqual(stt.model_load_calls, 1, "модель не должна грузиться повторно")
        self.assertEqual(stt._fake_model.transcribe_calls, 2)
        stt.shutdown()

    def test_restart_blocked_while_busy(self) -> None:
        """Второй recognize во время работы игнорируется."""
        stt = FakeSttEngine(self.config, result_text="hi", delay=0.4)
        stt.recognize(self.audio_path)
        self.assertTrue(stt.is_busy)
        stt.recognize(self.audio_path)
        stt.shutdown()
        self.assertEqual(
            stt._fake_model.transcribe_calls,
            1,
            "повторный запуск во время работы должен быть проигнорирован",
        )

    def test_error_emits_failed_and_stays_idle(self) -> None:
        """Ошибка распознавания: failed, IDLE, без повторных попыток."""
        boom = RuntimeError("mlx transcribe exploded")
        stt = FakeSttEngine(self.config, raise_on_transcribe=boom)
        failures: list[str] = []
        stt.failed.connect(failures.append)
        stt.recognize(self.audio_path)
        _wait_stt_idle(stt)
        self.assertEqual(len(failures), 1)
        self.assertIn("mlx transcribe exploded", failures[0])
        self.assertEqual(stt.state, SttEngine.IDLE)
        stt.shutdown()

    def test_close_during_processing(self) -> None:
        """Закрытие во время обработки: shutdown дожидается потока,

        новая работа после этого отклоняется, модель не грузится повторно.
        """
        stt = FakeSttEngine(self.config, result_text="hi", delay=0.3)
        stt.recognize(self.audio_path)
        stt.shutdown()  # должен дождаться рабочего потока
        self.assertFalse(stt.is_busy)
        self.assertEqual(stt.state, SttEngine.IDLE)

        # После закрытия новые recognize отклоняются.
        stt.recognize(self.audio_path)
        _pump(0.1)
        self.assertFalse(stt.is_busy)
        self.assertEqual(stt.model_load_calls, 1, "после shutdown модель не грузится")
        self.assertEqual(stt._fake_model.transcribe_calls, 1)
        # Файл-вход не тронут самим движком (его освобождает AudioController
        # после завершения — порядок проверен в тесте окна).
        self.assertTrue(os.path.exists(self.audio_path))


class TestNoFallback(MlxTestBase):
    """Автоматического fallback нет: ошибка — это ошибка, а не смена бэкенда."""

    def test_unknown_backend_raises_value_error(self) -> None:
        stt = FakeSttEngine(AppConfig(stt_backend="openai"), result_text="hi")
        with self.assertRaises(ValueError) as ctx:
            stt.recognize_sync(self.audio_path)
        self.assertIn("Неизвестный STT-бэкенд", str(ctx.exception))
        self.assertFalse(stt.model_loaded)

    def test_missing_mlx_dependencies_sync_error(self) -> None:
        """Отсутствие mlx/mlx-whisper — понятная ошибка, без перехода на faster-whisper."""
        boom = RuntimeError(
            "Бэкенд MLX недоступен: не установлены зависимости mlx / mlx-whisper. "
            "python -m pip install '.[mlx]' (плюс системный ffmpeg). "
            "Или выберите stt_backend='faster-whisper' в AppConfig."
        )
        stt = FakeSttEngine(self.config, raise_on_load=boom)
        with self.assertRaises(RuntimeError) as ctx:
            stt.recognize_sync(self.audio_path)
        message = str(ctx.exception)
        self.assertIn("mlx", message)
        self.assertIn("faster-whisper", message)
        self.assertFalse(stt.model_loaded, "fallback на другой бэкенд запрещён")
        self.assertEqual(stt.model_load_calls, 1, "повторной попытки загрузки нет")

    def test_missing_mlx_dependencies_failed_signal_async(self) -> None:
        """В UI-режиме ошибка недоступности MLX приходит сигналом failed."""
        boom = RuntimeError("Бэкенд MLX недоступен: … stt_backend='faster-whisper'")
        stt = FakeSttEngine(self.config, raise_on_load=boom)
        failures: list[str] = []
        stt.failed.connect(failures.append)
        stt.recognize(self.audio_path)
        _wait_stt_idle(stt)
        self.assertEqual(len(failures), 1)
        self.assertIn("faster-whisper", failures[0])
        self.assertEqual(stt.state, SttEngine.IDLE)
        stt.shutdown()


class TestMlxImportGuard(MlxTestBase):
    """Реальный ``_require_mlx_imports``: сообщение об ошибке и счастливый путь."""

    def test_import_error_becomes_clear_runtime_error(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name.split(".")[0] in ("mlx", "mlx_whisper"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", _blocked_import):
            with self.assertRaises(RuntimeError) as ctx:
                SttEngine._require_mlx_imports()
        message = str(ctx.exception)
        self.assertIn("mlx-whisper", message)
        self.assertIn("pip install", message)
        self.assertIn("ffmpeg", message)
        self.assertIn("faster-whisper", message)

    def test_imports_ok_when_mlx_installed(self) -> None:
        """Если mlx стоит в окружении — защита проходит (без загрузки весов)."""
        try:
            import mlx.core  # noqa: F401
            import mlx_whisper  # noqa: F401
        except ImportError:
            self.skipTest("mlx/mlx-whisper не установлены в этом окружении")
        SttEngine._require_mlx_imports()  # не должно бросать исключение


if __name__ == "__main__":
    unittest.main()
