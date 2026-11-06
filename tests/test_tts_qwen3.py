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
  сбой задания, пришедший после закрытия, — отброшен (без сигналов);
* ``build_tts``: по умолчанию ``Qwen3TtsSynth`` (reference-режим,
  референс p4 в папке данных приложения), ``say`` — ``TtsSynth``,
  неизвестный движок — ошибка, ``reference`` без пары «файл + текст» —
  ошибка;
* отсутствие файла референса: ``failed`` с понятной причиной
  (проверка ДО загрузки модели — тест работает без mlx-audio).

Слышимость и качество голоса — ручная проверка (см. README и
``tools/qwen3_voice_consistency_check.py``).
"""

import faulthandler
import os
import sys
import tempfile
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
from companion.app_data import tts_reference_path  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.tts_qwen3 import (  # noqa: E402
    DEFAULT_MODEL_REFERENCE,
    Qwen3TtsSynth,
)
from companion.tts_synth import TtsSynth  # noqa: E402

try:
    import sounddevice  # noqa: F401
    _HAS_SOUNDDEVICE = True
except Exception:  # pragma: no cover - окружение без PortAudio
    _HAS_SOUNDDEVICE = False

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


class _Chunk:
    """Кусок как в ``GenerationResult``: поле ``.audio`` (1-D float-массив)."""

    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


class StreamFakeSynth:
    """Фэйк синтеза СО стримингом: ``.stream(text)`` → итератор кусков,
    плюс полный путь ``__call__`` (для fallback). Без модели/PortAudio.

    * ``chunks`` — сколько кусков выдаёт ``stream``;
    * ``chunk_len`` — длина куска (сэмпл);
    * ``fail_at`` — индекс куска, на котором ``stream`` падает (None — нет);
    * ``gate`` — ``threading.Event``: пока не set, поток ждёт (удержание).
    """

    def __init__(
        self, chunks: int = 3, chunk_len: int = 480,
        delay: float = 0.0, fail_at: "int | None" = None,
        gate: "threading.Event | None" = None,
    ) -> None:
        self.chunks = chunks
        self.chunk_len = chunk_len
        self.delay = delay
        self.fail_at = fail_at
        self.gate = gate
        self.sample_rate = 24000
        self.stream_calls = 0
        self.full_calls = 0

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        self.full_calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        if self.delay:
            time.sleep(self.delay)
        n = self.chunks * self.chunk_len
        return np.full(n, 0.1, dtype=np.float32), self.sample_rate

    def stream(self, text: str):
        self.stream_calls += 1
        emitted = 0
        for _ in range(self.chunks):
            if self.gate is not None:
                self.gate.wait(timeout=2.0)
            if self.delay:
                time.sleep(self.delay)
            if self.fail_at is not None and emitted >= self.fail_at:
                raise RuntimeError("stream boom")
            emitted += 1
            yield _Chunk(np.full(self.chunk_len, 0.1, dtype=np.float32))


class _FakeStream:
    """Фэйк ``sd.InputStream``: только ``stop()``/``close()``, без PortAudio."""

    def __init__(self) -> None:
        self.stopped = False
        self.closed = False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class Qwen3TtsTestBase(unittest.TestCase):
    """Фасад ``AudioController`` + ``Qwen3TtsSynth`` с фэйковым синтезом."""

    def _make_fake(self) -> FakeSynth:
        """Точка подмены синтеза (потомки могут вернуть стриминг-фэйк)."""
        return FakeSynth()

    def setUp(self) -> None:
        _app()
        self.fake = self._make_fake()
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


class StreamingTestBase(Qwen3TtsTestBase):
    """Фасад + стриминг-фэйк; PortAudio заменён фэйком (без звука/железа)."""

    def _make_fake(self) -> StreamFakeSynth:
        return StreamFakeSynth(chunks=3, chunk_len=480)

    def setUp(self) -> None:
        super().setUp()
        # ПортAudio-поток подменяем фэйком (реального звука не будет).
        self.stream_fakes: list[_FakeStream] = []
        self.c.player._make_stream = self._new_stream
        # Малый стартовый буфер (40 мс): старт после ~2 кусков по 20 мс.
        self.c.stream_start_buffer_ms = 40

    def _new_stream(self, sp, sample_rate):
        fake = _FakeStream()
        self.stream_fakes.append(fake)
        return fake

    def _drain_stream(self) -> None:
        """Имитируем PortAudio: тянем очередь до конца (drained), затем IDLE."""
        sp = self.c.player._stream_player
        self.assertIsNotNone(sp, "поток не активен")
        frames = 480
        out = np.zeros((frames, 1), dtype=np.float32)
        guard = 0
        while not sp.drained and guard < 200000:
            sp.callback(out, frames, None, None)
            out.fill(0.0)
            guard += 1
        self.assertTrue(sp.drained, "поток не опустел")
        _pump(0.15)  # QTimer(50ms) заметит drained → IDLE

    @staticmethod
    def _wav_seconds(path: str) -> float:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())


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

    def test_late_failure_after_shutdown_is_dropped(self) -> None:
        """Сбой задания, пришедший ПОСЛЕ shutdown: отброшен, без сигналов.

        Фэйк-синтезатор держит ~2 c (таймаут gate), затем выбрасывает
        исключение — то есть уже после того, как ``shutdown()`` выполнен.
        Поздний сбой не публикуется: ``failed``/``state_changed`` не
        уходят, статус «остановлено» не затирается, файлов нет — так же,
        как поздний успех (``_finish_ready`` гасит ``_shutdown``).
        """
        self.fake.gate = threading.Event()  # НЕ set: фэйк сам выйдет через 2 c
        self.fake.fail = True
        self.c.speak_reply("Hello teacher")
        time.sleep(0.2)  # рабочий поток вошёл в фэйк
        self.c.shutdown()
        _pump(0.2)
        time.sleep(2.5)  # поздний сбой уже «пришёл»
        _pump(0.3)
        self.assertEqual(self.failed_reasons, [], "поздний failed после shutdown")
        self.assertEqual(self.ready_paths, [], "готовый после shutdown")
        self.assertEqual(self.c.tts.status, "Озвучивание: остановлено")
        self.assertEqual(self.c.tts.state, self.c.tts.IDLE)
        self.assertFalse(self.c.tts._temp_dir, "осталась временная папка")
        workers = [t for t in threading.enumerate() if t.name == "tts-qwen3"]
        self.assertTrue(
            all(not t.is_alive() for t in workers), "рабочий поток жив"
        )


class TestMissingReferenceFile(unittest.TestCase):
    """Отсутствие файла референса: понятная ошибка, модель не грузится.

    Проверка выполняется ДО импорта mlx-audio и загрузки модели, поэтому
    тест работает офлайн: mlx не нужен, веса не скачиваются.
    """

    def test_missing_ref_file_fails_with_clear_reason(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "no_such_tts_ref.wav")
        self.assertFalse(os.path.exists(missing))
        synth = Qwen3TtsSynth(
            model_id=DEFAULT_MODEL_REFERENCE,
            ref_audio=missing,
            ref_text="That was a really good answer.",
        )
        try:
            self.assertEqual(synth.mode, "reference")
            failures: list[str] = []
            synth.failed.connect(failures.append)
            self.assertTrue(synth.synthesize("Hello teacher"))
            # Дожидаемся сигнала failed (рабочий поток); межпотоковая
            # доставка требует обработки событий Qt в главном потоке.
            app = QApplication.instance() or QApplication(sys.argv)
            deadline = time.time() + 5.0
            while not failures and time.time() < deadline:
                app.processEvents()
                time.sleep(0.05)
            self.assertEqual(len(failures), 1, "не пришёл failed")
            reason = failures[0]
            self.assertIn("референс", reason)
            self.assertIn("не найден", reason)
            self.assertIn(missing, reason)
            self.assertFalse(synth.has_reply, "готовый ответ при сбое")
            self.assertEqual(synth.state, synth.IDLE)
            # Модель НЕ загружалась (файла нет — проверка до модели):
            self.assertIsNone(synth._model, "модель грузилась без необходимости")
        finally:
            synth.shutdown()


class TestBuildTts(unittest.TestCase):
    """build_tts: выбор движка по конфигу — без загрузки моделей."""

    def test_default_engine_is_qwen3_reference(self) -> None:
        # По умолчанию — Qwen3 reference-режим, референс p4 в папке данных.
        tts = build_tts(AppConfig())
        try:
            self.assertIsInstance(tts, Qwen3TtsSynth)
            self.assertEqual(tts.mode, "reference")
            self.assertEqual(
                tts.model_id, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
            )
            self.assertEqual(tts.ref_audio, str(tts_reference_path()))
            self.assertEqual(
                tts.ref_text,
                "That was a really good answer. You explained your idea, "
                "gave an example, and kept your rhythm natural.",
            )
        finally:
            tts.shutdown()

    def test_say_engine(self) -> None:
        config = AppConfig(tts_engine="say")
        tts = build_tts(config)
        try:
            self.assertIsInstance(tts, TtsSynth)
        finally:
            tts.shutdown()

    def test_qwen3_voice_design(self) -> None:
        config = AppConfig(tts_engine="qwen3", tts_qwen3_method="voice_design")
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

    def test_default_config_is_qwen3_reference(self) -> None:
        # Голос p4 одобрен: по умолчанию Qwen3 reference-режим.
        config = AppConfig()
        self.assertEqual(config.tts_engine, "qwen3")
        self.assertEqual(config.tts_qwen3_method, "reference")
        self.assertEqual(config.tts_qwen3_ref_audio, str(tts_reference_path()))
        self.assertTrue(config.tts_qwen3_ref_text)

    def test_reference_mode_without_pair_rejected(self) -> None:
        # reference без пары «файл + текст» — явная ошибка, не смена режима.
        for kwargs in (
            {"tts_qwen3_ref_audio": "", "tts_qwen3_ref_text": "text"},
            {"tts_qwen3_ref_audio": "/tmp/ref.wav", "tts_qwen3_ref_text": ""},
            {"tts_qwen3_ref_audio": "", "tts_qwen3_ref_text": ""},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                build_tts(AppConfig(tts_engine="qwen3", **kwargs))


class TestStreamSuccess(StreamingTestBase):
    """Стриминг успешен: речь идёт, WAV собирается из тех же кусков,
    повтор — тот же файл (без повторного синтеза)."""

    def test_stream_plays_and_assembles_wav(self) -> None:
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1, "готовый ответ")
        path = self.ready_paths[0]
        self.assertTrue(path.endswith(".wav"))
        self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)
        import wave
        with wave.open(path, "rb") as w:
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getnframes(), 3 * 480)
        # Стриминг, а не полный WAV, и не файл через QMediaPlayer.
        self.assertEqual(self.fake.stream_calls, 1, "не стриминг")
        self.assertEqual(self.fake.full_calls, 0, "полный путь при стриминге")
        self.assertEqual(self.fake_player.play_calls, 0)
        self.assertEqual(len(self.stream_fakes), 1, "поток не стартовал")
        self._drain_stream()
        self.assertEqual(self.c.state, AudioController.IDLE)

    def test_replay_plays_assembled_file_without_re_synth(self) -> None:
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        path = self.ready_paths[0]
        self._drain_stream()
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertTrue(self.c.can_replay)
        before = self.fake.stream_calls + self.fake.full_calls
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 1, "повтор не проиграл")
        self.assertEqual(
            self.fake.stream_calls + self.fake.full_calls, before,
            "повтор запустил синтез",
        )
        self.assertEqual(self.c.tts.last_reply_path, path)


class _MultiChunkBase(StreamingTestBase):
    """4 куска: стартовый буфер (40 мс = 2 куска) закрывается, ПОТОМ ещё 2 —
    их фасад обязан дописать в очередь вывода, а не отбросить как «занято»."""

    def _make_fake(self) -> StreamFakeSynth:
        return StreamFakeSynth(chunks=4, chunk_len=480)


class TestStreamNoTruncation(_MultiChunkBase):
    """РЕГРЕССИЯ «обрыв ответа»: активный потоковый вывод — это НАШ собственный
    PLAYING, и фасад не может считать его «занятостью» (state != IDLE) и не
    отбрасывать куски ПОСЛЕ старта. Вся речь должна дойти до очереди вывода.

    Раньше ``_on_chunk`` бросал каждый кусок после того, как вывод стал
    активным (state == PLAYING): игрался только стартовый буфер, ответ
    обрывался посреди фразы, а «Повторить ответ» мог быть недоступен.
    """

    def test_all_chunks_written_into_stream_queue(self) -> None:
        self.c.speak_reply("Hello teacher, nice work today")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1, "готовый ответ")
        sp = self.c.player._stream_player
        self.assertIsNotNone(sp, "поток не стартовал")
        # Фэйк-поток не «играет»: очередь копит ВСЁ, что дошло до вывода
        # (callback ещё не вызывался). Старый баг оставлял здесь 2*480
        # (стартовый буфер), теряя два последующих куска.
        queued = sum(int(np.asarray(c).size) for c in list(sp._queue.queue))
        self.assertEqual(queued, 4 * 480, "куски после старта отброшены")
        self._drain_stream()
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertTrue(self.c.can_replay)

    def test_full_stream_then_replay(self) -> None:
        self.c.speak_reply("Hello teacher, nice work today")
        _pump(0.3)
        self._drain_stream()
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertTrue(self.c.can_replay, "после вывода доступен повтор")
        before = self.fake.stream_calls + self.fake.full_calls
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 1, "повтор не проиграл")
        self.assertEqual(
            self.fake.stream_calls + self.fake.full_calls, before,
            "повтор запустил синтез",
        )


@unittest.skipUnless(_HAS_SOUNDDEVICE, "нет sounddevice/PortAudio")
class TestStreamOutputPath(unittest.TestCase):
    """РЕГРЕССИЯ «нет звука авто-озвучивания».

    Два реальных бага, которые тест обязан поймать (проверялось на слух):

    * воспроизведение должно идти ПОТОКОМ ВЫХОДА — ``sd.OutputStream``.
      ``sd.InputStream`` — это ВХОД (микрофон): писать в его callback звук
      некуда, и пользователь слышит ТИШИНУ (а «Повторить ответ» — через
      QMediaPlayer — звучит). Раньше как раз был InputStream.
    * ``OutputStream`` при создании НЕ авто-стартует (``active == False``):
      без явного ``start()`` callback не сработает → ни звука, ни ``drained``
      (плеер зависает в PLAYING, «Записать» блокируется).

    Тест без PortAudio: подменяем ``sounddevice.OutputStream`` и убеждаемся,
    что ``_make_stream`` именно его вызывает (а не InputStream), запускает и
    пробрасывает callback.
    """

    def test_make_stream_uses_output_stream_and_starts_it(self) -> None:
        _app()
        fake_out = mock.Mock(name="sd.OutputStream-instance")
        fake_in = mock.Mock(name="sd.InputStream-instance")
        with mock.patch("sounddevice.OutputStream", return_value=fake_out) as m_out, \
             mock.patch("sounddevice.InputStream", return_value=fake_in) as m_in:
            player = AudioPlayer()
            sp = mock.Mock(name="_StreamPlayer")
            try:
                player._make_stream(sp, 24000)
            finally:
                player.shutdown()
        # Ключевое: это ПОТОК ВЫХОДА, а не вход (микрофон).
        m_out.assert_called_once()
        m_in.assert_not_called()
        # Поток запущен (а не только создан) и callback проброшен.
        fake_out.start.assert_called_once_with()
        _, kwargs = m_out.call_args
        self.assertIs(kwargs.get("callback"), sp.callback, "callback не проброшен")


class TestStreamStop(StreamingTestBase):
    """Stop во время синтеза: куски отброшены, готовый (первый) жив."""

    def test_stop_mid_stream_keeps_previous_reply(self) -> None:
        # Первый ответ — успех.
        self.c.speak_reply("first reply")
        _pump(0.3)
        first = self.ready_paths[0]
        self._drain_stream()
        self.assertEqual(self.c.tts.last_reply_path, first)
        # Второй ответ держим в гейте (рабочий поток в синтезе).
        gate = threading.Event()
        self.fake.gate = gate
        self.c.speak_reply("second reply")
        deadline = time.time() + 2.0
        while self.fake.stream_calls < 2 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.fake.stream_calls, 2, "второй не в потоке")
        self.c.stop_playback()
        gate.set()  # отпустить — куски уже устарели
        _pump(0.4)
        self.assertEqual(self.ready_paths, [first], "второй опубликован")
        self.assertEqual(self.c.tts.last_reply_path, first)
        self.assertEqual(self.fake_player.play_calls, 0, "второй проигрался")


class TestStreamReplace(StreamingTestBase):
    """Новый ответ сменяет стриминг: поздние куски отброшены по номеру."""

    def test_new_reply_replaces_streaming_job(self) -> None:
        gate = threading.Event()
        self.fake.gate = gate
        self.c.speak_reply("first reply")
        deadline = time.time() + 2.0
        while self.fake.stream_calls < 1 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.fake.stream_calls, 1, "первый не в потоке")
        # Второй ответ сменяет первый (первый всё ещё в гейте).
        self.fake.gate = None
        self.c.speak_reply("second reply")
        _pump(0.5)
        gate.set()  # поздние куски первого — устарели
        _pump(0.5)
        self.assertEqual(len(self.ready_paths), 1, "старое задание опубликовано")
        self.assertEqual(self.fake.stream_calls, 2, "поток второго не открыт")
        self.assertEqual(self.fake.full_calls, 0)
        self.assertGreaterEqual(len(self.stream_fakes), 1, "второй не проигрался")
        self.assertTrue(os.path.exists(self.ready_paths[0]))


class TestStreamFallback(StreamingTestBase):
    """Стриминг недоступен/выключен до звука → полный WAV (как раньше)."""

    def test_streaming_off_uses_full_wav(self) -> None:
        self.c.tts.streaming = False
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1)
        self.assertEqual(self.fake.stream_calls, 0, "стриминг при выключенном")
        self.assertEqual(self.fake.full_calls, 1, "не полный WAV")
        self.assertEqual(self.fake_player.play_calls, 1, "автозапуск")

    def test_fake_without_stream_falls_back(self) -> None:
        plain = FakeSynth()
        self.c.tts._synthesizer = plain  # у него нет .stream
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1)
        self.assertEqual(plain.calls, 1, "не полный WAV")
        self.assertEqual(self.fake.stream_calls, 0, "стриминг без .stream")
        self.assertEqual(self.fake_player.play_calls, 1)

    def test_error_before_any_chunk_falls_back(self) -> None:
        self.fake.fail_at = 0  # падает до первого куска → полный WAV
        self.c.speak_reply("Hello teacher")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1)
        self.assertEqual(self.fake.stream_calls, 1, "поток не открыт")
        self.assertEqual(self.fake.full_calls, 1, "fallback не сработал")
        self.assertEqual(self.fake_player.play_calls, 1)


class TestStreamErrorAfterSound(StreamingTestBase):
    """Ошибка после начала звука: вывод остановлен, статус, без перезапуска."""

    def test_error_after_sound_stops_no_restart(self) -> None:
        self.fake.chunks = 4
        self.fake.fail_at = 3  # куски 0,1,2 выданы (звук шёл), 3-й падает
        self.c.speak_reply("Hello teacher")
        _pump(0.5)
        self.assertEqual(self.failed_reasons, ["stream boom"])
        self.assertEqual(self.ready_paths, [], "частичный WAV")
        self.assertFalse(self.c.has_reply, "не должно быть готового")
        self.assertFalse(self.c.player.stream_active, "поток не остановлен")
        self.assertEqual(self.fake_player.play_calls, 0)
        self.assertIn("Не удалось озвучить ответ", self.c.status)
        self.assertIn("stream boom", self.c.status)
        # Компонент жив: следующий синтез успешен.
        self.fake.fail_at = None
        self.c.speak_reply("again")
        _pump(0.3)
        self.assertEqual(len(self.ready_paths), 1)

    def test_error_after_sound_keeps_previous_ready(self) -> None:
        self.c.speak_reply("first reply")
        _pump(0.3)
        first = self.ready_paths[0]
        self._drain_stream()
        self.assertEqual(self.c.tts.last_reply_path, first)
        self.fake.chunks = 4
        self.fake.fail_at = 3
        self.c.speak_reply("second reply")
        _pump(0.5)
        self.assertEqual(self.failed_reasons, ["stream boom"])
        self.assertEqual(self.c.tts.last_reply_path, first, "готовый стёрт")
        self.assertTrue(self.c.can_replay)


class TestStreamBusy(StreamingTestBase):
    """Занятое аудио: стриминг не стартует, куски отброшены, ответ жив."""

    def test_busy_recording_discards_chunks_keeps_reply(self) -> None:
        self.c.recorder.state = AudioRecorder.RECORDING
        try:
            self.c.speak_reply("Hello teacher")
            _pump(0.3)
            self.assertEqual(
                self.fake_player.play_calls, 0, "автозапуск во время записи"
            )
            self.assertFalse(self.c.player.stream_active, "поток стартовал")
            self.assertEqual(len(self.stream_fakes), 0, "поток создан")
            self.assertIn("занято", self.c.status)
            self.assertTrue(self.c.has_reply, "готовый ответ потерян")
        finally:
            self.c.recorder.state = AudioRecorder.IDLE
        self.assertTrue(self.c.can_replay)
        self.c.replay_reply()
        _pump(0.1)
        self.assertEqual(self.fake_player.play_calls, 1)


class TestStreamClose(StreamingTestBase):
    """Закрытие во время синтеза: поток мёртв, поздних сигналов нет,
    файлов нет (отдельно от возврата join(timeout) — факт смерти потока)."""

    def test_shutdown_during_stream_drops_and_thread_dies(self) -> None:
        self.fake.gate = threading.Event()  # НЕ set: уйдёт через таймаут
        self.fake.chunks = 5
        self.c.speak_reply("Hello teacher")
        time.sleep(0.2)
        self.c.shutdown()
        _pump(0.2)
        time.sleep(2.4)  # поздний кусок уже «пришёл» (gate timeout в фэйке)
        _pump(0.3)
        self.assertEqual(self.failed_reasons, [], "поздний failed")
        self.assertEqual(self.ready_paths, [], "поздний готовый")
        self.assertFalse(self.c.tts._temp_dir, "осталась временная папка")
        workers = [t for t in threading.enumerate() if t.name == "tts-qwen3"]
        self.assertTrue(all(not t.is_alive() for t in workers), "поток жив")


class TestStreamPlayerFake(StreamingTestBase):
    """Состояния плеера (фэйк-бэкенд): start/write/end/stop, недочёты."""

    def test_player_stream_state_transitions(self) -> None:
        p = self.c.player
        self.assertEqual(p.state, AudioPlayer.IDLE)
        self.assertFalse(p.stream_active)
        p.start_stream(24000)
        self.assertEqual(p.state, AudioPlayer.PLAYING)
        self.assertTrue(p.stream_active)
        p.write_chunk(np.zeros(480, dtype=np.float32))
        sp = p._stream_player
        self.assertIsNotNone(sp)
        out = np.zeros((1000, 1), dtype=np.float32)
        sp.callback(out, 1000, None, None)  # 480 куска + 520 тишины (недочёт)
        self.assertEqual(p.stream_underrun_samples(), 520)
        self.assertFalse(sp.drained)
        p.end_stream()
        sp.callback(np.zeros((1000, 1), dtype=np.float32), 1000, None, None)
        self.assertTrue(sp.drained, "поток не опустел")
        _pump(0.15)
        self.assertEqual(p.state, AudioPlayer.IDLE)
        self.assertFalse(p.stream_active)

    def test_player_stop_stream_drops_queue(self) -> None:
        p = self.c.player
        p.start_stream(24000)
        p.write_chunk(np.zeros(480, dtype=np.float32))
        p.stop_stream()
        self.assertEqual(p.state, AudioPlayer.IDLE)
        self.assertFalse(p.stream_active)
        self.assertTrue(p.can_play, "после stop полный WAV недоступен")

    def test_write_before_start_is_noop(self) -> None:
        p = self.c.player
        p.write_chunk(np.zeros(100, dtype=np.float32))
        self.assertFalse(p.stream_active)
        self.assertEqual(p.state, AudioPlayer.IDLE)

    def test_start_stream_rejected_when_playing(self) -> None:
        p = self.c.player
        p.start_stream(24000)
        self.assertEqual(p.state, AudioPlayer.PLAYING)
        p.start_stream(24000)  # повторный — игнорится (уже PLAYING)
        self.assertEqual(len(self.stream_fakes), 1, "второй поток создан")
        p.stop_stream()


if __name__ == "__main__":
    unittest.main()
