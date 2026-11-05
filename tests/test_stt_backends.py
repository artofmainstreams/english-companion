"""Проверки STT-backend'ов (mlx-whisper, faster-whisper) — с подменой
внешних библиотек, без весов, интернета и оборудования.

Запуск::

    .venv/bin/python -m unittest discover -s tests -v

Внешние библиотеки подменяются (модули-заглушки в ``sys.modules`` и
``unittest.mock``): проверяются правильные параметры вызовов, извлечение
текста, понятные ошибки при отсутствующих зависимостях и обход ленивого
генератора faster-whisper в рабочем потоке. Реальные модели не загружаются
и веса не скачиваются.

Проверяются:
* дефолты AppConfig (mlx + whisper-large-v3-mlx, en/transcribe);
* выбор реализации по AppConfig (``build_backend``): mlx (по умолчанию),
  faster-whisper (явно), неизвестный бэкенд — явная ошибка без fallback;
* MLX: параметры загрузки (репозиторий, fp16), параметры распознавания
  (path_or_hf_repo/language/task), извлечение текста (dict и не-dict),
  отсутствие зависимостей — понятная ошибка, а не смена движка;
* faster-whisper: параметры модели (device/compute_type), параметры
  распознавания (language/task/vad_filter), склейка сегментов, обход
  ленивого генератора в рабочем потоке (не в UI-потоке).
"""

import builtins
import faulthandler
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from unittest import mock

faulthandler.enable()

# offscreen — чтобы Qt не требовал дисплей; аудио/распознавание не нужны.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.config import AppConfig  # noqa: E402
from companion.stt_backends import (  # noqa: E402
    FasterWhisperBackend,
    MlxBackend,
    build_backend,
)
from companion.stt_engine import SttEngine  # noqa: E402

_QAPP: "QApplication | None" = None

_MISSING = object()


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


class _SysModulesPatch:
    """Заменяет модули в ``sys.modules`` на время теста и восстанавливает.

    Без этого реальные (установленные в venv) mlx/faster-whisper
    подтянулись бы в код backend'ов, а ``import`` нашёл бы их в кэше.
    """

    def __init__(self, modules: dict) -> None:
        self.modules = modules
        self.saved: dict[str, object] = {}

    def __enter__(self) -> "_SysModulesPatch":
        for name in self.modules:
            self.saved[name] = sys.modules.get(name, _MISSING)
        sys.modules.update(self.modules)
        return self

    def __exit__(self, *exc) -> None:
        for name, value in self.saved.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _fake_mlx_modules() -> tuple[dict, "mock.Mock"]:
    """Подменные mlx.* / mlx_whisper* модули.

    Возвращает ``(карта модулей, заглушка mlx_whisper.transcribe)``.
    """
    mx = types.ModuleType("mlx.core")
    mx.float16 = "float16"

    holder = mock.Mock(name="ModelHolder")
    holder.get_model.return_value = "mlx-model"
    sub = types.ModuleType("mlx_whisper.transcribe")
    sub.ModelHolder = holder

    transcribe_fn = mock.Mock(name="mlx_whisper.transcribe")
    transcribe_fn.return_value = {"text": "  hello mlx  "}

    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx  # чтобы `import mlx.core as mx` не упирался в getattr
    mod = types.ModuleType("mlx_whisper")
    mod.transcribe = transcribe_fn
    mod.transcribe_module = sub  # имя подмодуля не важно; импортируют по пути

    return (
        {
            "mlx": mlx_pkg,
            "mlx.core": mx,
            "mlx_whisper": mod,
            "mlx_whisper.transcribe": sub,
        },
        transcribe_fn,
    )


class TestAppConfigDefaults(unittest.TestCase):
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


class TestBackendSelection(unittest.TestCase):
    """Выбор реализации по AppConfig — при сборке приложения, без fallback."""

    def test_default_config_selects_mlx(self) -> None:
        self.assertIsInstance(build_backend(AppConfig()), MlxBackend)

    def test_explicit_faster_whisper_selected(self) -> None:
        backend = build_backend(AppConfig(stt_backend="faster-whisper"))
        self.assertIsInstance(backend, FasterWhisperBackend)

    def test_unknown_backend_raises_value_error(self) -> None:
        """Неизвестный бэкенд — явная ошибка, а не смена движка (no fallback)."""
        with self.assertRaises(ValueError) as ctx:
            build_backend(AppConfig(stt_backend="openai"))
        message = str(ctx.exception)
        self.assertIn("Неизвестный STT-бэкенд", message)
        self.assertIn("mlx", message)
        self.assertIn("faster-whisper", message)


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


class TestMlxBackend(MlxTestBase):
    """Реальный ``MlxBackend`` с подменёнными mlx.*: параметры и текст."""

    def test_load_model_uses_config_repo_and_fp16(self) -> None:
        """Загрузка: репозиторий из конфига, fp16; модель возвращается как есть."""
        modules, _fn = _fake_mlx_modules()
        with _SysModulesPatch(modules):
            backend = MlxBackend(self.config)
            model = backend.load_model()
        mx = modules["mlx.core"]
        holder = modules["mlx_whisper.transcribe"].ModelHolder
        holder.get_model.assert_called_once_with(
            self.config.stt_mlx_model, mx.float16
        )
        self.assertEqual(model, "mlx-model")

    def test_transcribe_uses_config_params_and_extracts_text(self) -> None:
        modules, transcribe_fn = _fake_mlx_modules()
        with _SysModulesPatch(modules):
            backend = MlxBackend(self.config)
            text = backend.transcribe("mlx-model", self.audio_path)
        self.assertEqual(text, "hello mlx", "текст не извлечён из dict")
        transcribe_fn.assert_called_once_with(
            self.audio_path,
            path_or_hf_repo=self.config.stt_mlx_model,
            language=self.config.stt_language,
            task=self.config.stt_task,
        )

    def test_transcribe_accepts_non_dict_result(self) -> None:
        modules, transcribe_fn = _fake_mlx_modules()
        transcribe_fn.return_value = "  plain string  "
        with _SysModulesPatch(modules):
            backend = MlxBackend(self.config)
            text = backend.transcribe("mlx-model", self.audio_path)
        self.assertEqual(text, "plain string")

    def test_missing_dependencies_clear_error(self) -> None:
        """Отсутствие mlx/mlx-whisper — понятная ошибка, без перехода на
        faster-whisper. Определинно независимо от порядка тестов: сначала
        выгружаем уже импортированные модули, потом блокируем импорт."""
        purged = {}
        for name in ("mlx", "mlx.core", "mlx_whisper", "mlx_whisper.transcribe"):
            purged[name] = sys.modules.pop(name, _MISSING)

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name.split(".")[0] in ("mlx", "mlx_whisper"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", _blocked_import):
                backend = MlxBackend(self.config)
                with self.assertRaises(RuntimeError) as ctx:
                    backend.load_model()
        finally:
            for name, value in purged.items():
                if value is not _MISSING:
                    sys.modules[name] = value

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
        MlxBackend._require_imports()  # не должно бросать исключение


class TestFasterWhisperBackend(unittest.TestCase):
    """Реальный ``FasterWhisperBackend`` с подменённым ``faster_whisper``."""

    def setUp(self) -> None:
        _app()
        self.config = AppConfig(stt_backend="faster-whisper")
        self.tmp = tempfile.mkdtemp(prefix="english-companion-fw-")
        self.audio_path = os.path.join(self.tmp, "clip.wav")
        _make_audio(self.audio_path)

    def tearDown(self) -> None:
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _fw_module(self, model) -> types.ModuleType:
        mod = types.ModuleType("faster_whisper")
        mod.WhisperModel = mock.Mock(name="WhisperModel", return_value=model)
        return mod

    def test_load_model_uses_config_params(self) -> None:
        """Загрузка: модель из конфига + device/compute_type."""
        model = "fw-model"
        mod = self._fw_module(model)
        with _SysModulesPatch({"faster_whisper": mod}):
            backend = FasterWhisperBackend(self.config)
            self.assertIs(backend.load_model(), model)
        mod.WhisperModel.assert_called_once_with(
            self.config.stt_whisper_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
        )

    def test_transcribe_params_and_text_joining(self) -> None:
        """Распознавание: language/task/vad_filter из конфига; пустые
        сегменты пропускаются, текст склеивается."""

        class _Seg:
            def __init__(self, text: str) -> None:
                self.text = text

        captured: dict = {}

        def _transcribe(audio_path, **kwargs):
            captured.update(kwargs)
            captured["audio_path"] = audio_path

            def gen():
                yield _Seg("  hello  ")
                yield _Seg("   ")
                yield _Seg("world")

            return gen(), None

        model = mock.Mock(name="model")
        model.transcribe.side_effect = _transcribe
        backend = FasterWhisperBackend(self.config)
        text = backend.transcribe(model, self.audio_path)
        self.assertEqual(text, "hello world", "сегменты не склеены корректно")
        self.assertEqual(captured["audio_path"], self.audio_path)
        self.assertEqual(captured["language"], self.config.stt_language)
        self.assertEqual(captured["task"], self.config.stt_task)
        self.assertIs(captured["vad_filter"], self.config.stt_vad_filter)

    def test_lazy_generator_consumed_in_worker_thread(self) -> None:
        """Обход ленивого генератора faster-whisper идёт в рабочем потоке
        (stt-worker), а не в UI-потоке: движок запускает распознавание в
        фоне, текст возвращается сигналом recognized."""
        thread_names: list[str] = []

        class _Seg:
            def __init__(self, text: str) -> None:
                self.text = text

        def _transcribe(audio_path, **kwargs):
            def gen():
                # Лазый: тело выполняется, когда потребитель обходит генератор.
                thread_names.append(threading.current_thread().name)
                yield _Seg("hello from worker")
                thread_names.append(threading.current_thread().name)

            return gen(), None

        model = mock.Mock(name="model")
        model.transcribe.side_effect = _transcribe
        mod = self._fw_module(model)

        with _SysModulesPatch({"faster_whisper": mod}):
            backend = FasterWhisperBackend(self.config)
            stt = SttEngine(backend)
            seen: list[str] = []
            stt.recognized.connect(seen.append)
            stt.recognize(self.audio_path)
            _wait_stt_idle(stt)
            self.assertEqual(seen, ["hello from worker"])
            stt.shutdown()

        mod.WhisperModel.assert_called_once_with(
            self.config.stt_whisper_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
        )
        self.assertEqual(
            len(thread_names), 2,
            "генератор не был обходован лениво (ожидаются 2 отметки)",
        )
        main_name = threading.main_thread().name
        self.assertNotIn(main_name, thread_names, "генератор обходован в UI-потоке")
        self.assertTrue(
            all(name == "stt-worker" for name in thread_names),
            f"обход не в рабочем потоке: {thread_names}",
        )


if __name__ == "__main__":
    unittest.main()
