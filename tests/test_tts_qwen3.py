"""Озвучивание ответов Qwen3-TTS (MLX) с подменой синтеза — без mlx-audio.

Запуск (из корня проекта)::

    .venv/bin/python -m unittest discover -s tests -v

Синтезатор подменяется фэйком (вызываемый ``text -> (массив, 24000)``):
mlx-audio не нужен, модель не грузится, веса не скачиваются. Проверки —
те же сценарии, что для движка ``say`` (``test_tts_reply.py``):

* успешный ответ: WAV-файл создан, ``reply_ready``, автозапуск;
* новый ответ СМЕНЯЕТ незавершённый: старое задание отброшено;
* «Повторить ответ»: тот же готовый файл — без нового синтеза;
* ошибка: ``failed``, понятный статус, частичного файла нет,
  компонент жив и следующий синтез успешен;
* занятое аудио: автозапуска нет, ответ ждёт ручного повтора;
* закрытие: поток остановлен, временных файлов нет, компонент мёртв;
* ``build_tts``: по умолчанию ``TtsSynth``, ``qwen3`` — ``Qwen3TtsSynth``
  (режимы voice_design/reference), неизвестный движок — ошибка.

Слышимость и качество голоса — ручная проверка (см. README и
``tools/qwen3_voice_consistency_check.py``).
"""

import faulthandler
import os
import sys
import threading
import time
import unittest

faulthandler.enable()

# offscreen — чтобы тесты не зависели от дисплея; аудио при этом работает.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest import mock  # noqa: E402

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import numpy as np  # noqa: E402

from companion.audio_controller import (  # noqa: E402
    AudioController,
    build_tts,
)
from companion.audio_player import AudioPlayer  # noqa: E402
from companion.audio_recorder import AudioRecorder  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.tts_qwen3 import Qwen3TtsSynth  # noqa: E402
from companion.tts_synth import TtsSynth  # noqa: E402

_QAPP: QApplication | None = None


def _app() -> QApplication:
    global _QAPP
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication(sys.argv)
    return _QAPP


def _pump(seconds: float) -> None:
    """Пускаем цикл событий: доставка межпотоковых сигналов + deleteLater."""
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


class FakeSynth:
    """Фэйк синтеза: текст -> (тихий float-массив, 24000 Hz). Без модели."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.fail = False
        self.gate: threading.Event | None = None

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("synth boom")
        return np.full(2400, 0.1, dtype=np.float32), 24000


class _FakePlayer:
    """Заглушка QMediaPlayer: фиксирует play/stop, без аудиооборудования."""

    def __init__(self) -> None:
        self.play_calls = 0
        self.stop_calls = 0

    def play(self) -> None:
        self.play_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def position(self) -> int:
        return 0

    def setSource(self, *args) -> None:
        pass

    def setAudioOutput(self, *args) -> None:
        pass

    def disconnect(self, *args) -> None:
        pass

    def deleteLater(self) -> None:
        pass


class Qwen3TtsTestBase(unittest.TestCase):
    """Фасад ``AudioController`` + ``Qwen3TtsSynth`` с фэйковым синтезом."""

    def setUp(self) -> None:
        _app()
        self.fake = FakeSynth()
        self.c = AudioController(tts=Qwen3TtsSynth(synthesizer=self.fake))
        self.ready_paths: list[str] = []
        self.failed_reasons: list[str] = []
        self.c.tts.reply_ready.connect(self.ready_paths.append)
        self.c.tts.failed.connect(self.failed_reasons.append)
        # Граница AudioPlayer: без реальных устройств и QAudioOutput.
        self.output_cls = mock.Mock(name="QAudioOutput-class")
        self.devices = mock.Mock(name="QMediaDevices")
        self.device = mock.Mock(name="output-device")
        self.device.isNull.return_value = False
        self.device.description.return_value = "fake output"
        self.devices.defaultAudioOutput.return_value = self.device
        self._p_output = mock.patch(
            "companion.audio_player.QAudioOutput", self.output_cls
        )
        self._p_devices = mock.patch(
            "companion.audio_player.QMediaDevices", self.devices
        )
        self._p_output.start()
        self._p_devices.start()
        self.fake_player = _FakePlayer()
        self.c.player._player = self.fake_player

    def tearDown(self) -> None:
        try:
            self.c.shutdown()
            _pump(0.1)
        finally:
            self._p_output.stop()
            self._p_devices.stop()
            del self.c


class TestSpeakReplySuccess(Qwen3TtsTestBase):
    """Успешный ответ: WAV-файл, reply_ready, автозапуск."""

    def test_reply_ready_autoplays_when_audio_free(self) -> None:
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1, "лишних готовых ответов")
        out_path = self.ready_paths[0]
        self.assertTrue(out_path.endswith(".wav"))
        self.assertTrue(os.path.exists(out_path) and os.path.getsize(out_path) > 0)
        self.assertEqual(self.c.state, AudioController.PLAYING)
        self.assertEqual(self.fake_player.play_calls, 1)
        self.assertTrue(self.c.has_reply)
        self.assertEqual(self.c.tts.last_reply_path, out_path)

    def test_empty_text_rejected(self) -> None:
        self.c.speak_reply("   ")
        _pump(0.1)
        self.assertEqual(self.fake.calls, 0, "синтез пустого текста")
        self.assertEqual(self.ready_paths, [])
        self.assertIn("нечего", self.c.status)

    def test_new_reply_replaces_inflight_synthesis(self) -> None:
        # Первое задание держим «в полёте» (замок в фэйке).
        self.fake.gate = threading.Event()
        self.c.speak_reply("first reply")
        time.sleep(0.2)  # рабочий поток вошёл в фэйк и ждёт
        self.c.speak_reply("second reply")
        self.fake.gate.set()  # отпустим оба задания
        _pump(0.5)
        # Готовым стало ТОЛЬКО второе задание; автозапуск один раз.
        self.assertEqual(len(self.ready_paths), 1, "старое задание опубликовано")
        self.assertEqual(self.fake_player.play_calls, 1)
        self.assertEqual(self.c.tts.last_reply_path, self.ready_paths[0])


class TestReplayReply(Qwen3TtsTestBase):
    """«Повторить ответ»: тот же готовый файл, без нового синтеза."""

    def test_replay_plays_same_file_without_new_synth(self) -> None:
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        out_path = self.ready_paths[0]
        self.c.stop_playback()
        _pump(0.1)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertTrue(self.c.can_replay)
        before = self.fake.calls
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake.calls, before, "повтор запустил синтез")
        self.assertEqual(self.fake_player.play_calls, 2)
        self.assertEqual(self.c.tts.last_reply_path, out_path)

    def test_replay_without_reply_rejected(self) -> None:
        self.assertFalse(self.c.can_replay)
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 0)
        self.assertIn("Повторять нечего", self.c.status)


class TestSynthErrors(Qwen3TtsTestBase):
    """Ошибка синтеза: нет автозапуска, понятный статус, компонент жив."""

    def test_failure_clear_status_no_partial_file(self) -> None:
        self.fake.fail = True
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        self.assertEqual(self.ready_paths, [], "готовый при ошибке синтеза")
        self.assertEqual(self.failed_reasons, ["synth boom"])
        self.assertEqual(self.fake_player.play_calls, 0)
        self.assertFalse(self.c.has_reply)
        self.assertIn("Не удалось озвучить ответ", self.c.status)
        self.assertIn("synth boom", self.c.status)
        # Частичного файла нет (временная папка не создавалась).
        self.assertFalse(self.c.tts._temp_dir, "осталась временная папка")
        # Компонент жив: следующий синтез завершается успешно.
        self.fake.fail = False
        self.c.speak_reply("again")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1)
        self.assertTrue(self.c.has_reply)

    def test_failed_reply_does_not_lose_previous_ready(self) -> None:
        """Сбой нового ответа не стирает предыдущий готовый (повтор жив)."""
        self.c.speak_reply("first reply")
        _pump(0.3)
        first = self.ready_paths[0]
        self.c.stop_playback()
        _pump(0.1)
        self.fake.fail = True
        self.c.speak_reply("second reply")
        _pump(0.3)
        self.assertEqual(self.c.tts.last_reply_path, first)
        self.assertTrue(self.c.can_replay)
        self.assertEqual(self.failed_reasons, ["synth boom"])


class TestBusyAudio(Qwen3TtsTestBase):
    """Занятое аудио: автозапуска нет, ответ ждёт ручного повтора."""

    def test_busy_recording_keeps_reply_for_manual_replay(self) -> None:
        self.c.recorder.state = AudioRecorder.RECORDING
        try:
            self.c.speak_reply("Hello teacher")
            _pump(0.3)
            self.assertEqual(self.c.state, AudioController.RECORDING)
            self.assertEqual(
                self.fake_player.play_calls, 0, "автозапуск во время записи"
            )
            self.assertIn("занято", self.c.status)
            self.assertTrue(self.c.has_reply, "готовый ответ потерян")
        finally:
            self.c.recorder.state = AudioRecorder.IDLE
        self.assertTrue(self.c.can_replay)
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 1)

    def test_busy_playback_keeps_reply_for_manual_replay(self) -> None:
        self.c.player.state = AudioPlayer.PLAYING
        try:
            self.c.speak_reply("Hello teacher")
            _pump(0.3)
            self.assertEqual(self.fake_player.play_calls, 0)
            self.assertIn("занято", self.c.status)
            self.assertTrue(self.c.has_reply)
        finally:
            self.c.player.state = AudioPlayer.IDLE
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 1)


class TestCloseDuringSynthesis(Qwen3TtsTestBase):
    """Закрытие во время синтеза: поток остановлен, файлов нет, мёртв."""

    def test_shutdown_stops_worker_and_cleans(self) -> None:
        self.fake.delay = 1.0  # синтез идёт дольше, чем вызов shutdown()
        self.c.speak_reply("Hello teacher")
        time.sleep(0.2)  # рабочий поток в синтезе
        self.c.shutdown()
        _pump(0.2)
        self.assertFalse(self.c.tts._temp_dir, "осталась временная папка")
        self.assertEqual(self.ready_paths, [], "готовый после shutdown")
        # Компонент мёртв: синтез и повтор отклоняются.
        self.assertFalse(self.c.can_replay)
        self.assertFalse(self.c.speak_reply("again"))
        self.assertEqual(self.fake.calls, 1, "синтез после shutdown")
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 0)

    def test_shutdown_idempotent(self) -> None:
        self.c.shutdown()
        self.c.shutdown()
        _pump(0.1)
        self.assertTrue(self.c.tts.is_shutdown)
        self.assertEqual(self.c.state, AudioController.IDLE)


class TestBuildTts(unittest.TestCase):
    """build_tts: выбор движка по конфигу — без загрузки моделей."""

    def test_default_engine_is_say(self) -> None:
        tts = build_tts(AppConfig())
        try:
            self.assertIsInstance(tts, TtsSynth)
        finally:
            tts.shutdown()

    def test_qwen3_voice_design(self) -> None:
        config = AppConfig(tts_engine="qwen3")
        tts = build_tts(config)
        try:
            self.assertIsInstance(tts, Qwen3TtsSynth)
            self.assertEqual(tts.mode, "voice_design")
            self.assertEqual(
                tts.model_id, "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
            )
            self.assertIn("deep, naturally low alto", tts.instruct)
        finally:
            tts.shutdown()

    def test_qwen3_reference_mode(self) -> None:
        config = AppConfig(
            tts_engine="qwen3",
            tts_qwen3_method="reference",
            tts_qwen3_ref_audio="/tmp/ref_phrase.wav",
        )
        tts = build_tts(config)
        try:
            self.assertEqual(tts.mode, "reference")
            self.assertEqual(
                tts.model_id, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
            )
            self.assertEqual(tts.ref_audio, "/tmp/ref_phrase.wav")
            self.assertTrue(tts.ref_text)
        finally:
            tts.shutdown()

    def test_unknown_engine_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_tts(AppConfig(tts_engine="kokoro"))

    def test_default_config_is_say(self) -> None:
        # Переключение на Qwen3 — только явным решением (после прослушивания).
        self.assertEqual(AppConfig().tts_engine, "say")


if __name__ == "__main__":
    unittest.main()
