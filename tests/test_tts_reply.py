"""Проверки озвучивания ответов (TTS) с подменой синтеза — без реального say.

Запуск (из корня проекта)::

    .venv/bin/python -m unittest discover -s tests -v

Синтез подменяется ``FakeQProcess`` (заменяет ``companion.tts_synth.QProcess``):
системный ``say`` не запускается, звука нет; выходной файл тест создаёт сам —
как его создаёт ``say`` при успехе. Плеер тоже подменяется (``_FakePlayer``),
поэтому проверки не зависят от устройств звука.

Сценарии:
* успешный ответ: команда ``say -v <голос> -f -`` (текст — данные через
  stdin), готовый AIFF автозапускается, когда аудио свободно;
* «Повторить ответ»: проигрывается тот же готовый файл — без нового
  запроса к Ollama и без нового синтеза;
* ошибка синтеза: автозапуска нет, понятный статус, частичный файл
  удалён, текст ответа остаётся доступным (история чата);
* занятое аудио (запись/воспроизведение): ответ сохраняется для
  ручного повтора;
* закрытие во время синтеза: процесс остановлен, временная папка
  удалена, компонент перестал работать.

Слышимость и качество голоса — ручная проверка (см. README).
"""

import faulthandler
import os
import sys
import unittest

faulthandler.enable()

# offscreen — чтобы тесты не зависели от дисплея; аудио при этом работает.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest import mock  # noqa: E402

from PySide6.QtCore import QEventLoop, QProcess, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.audio_player import AudioPlayer  # noqa: E402
from companion.audio_recorder import AudioRecorder  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.main_window import MainWindow  # noqa: E402
from companion.message import Message  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.session_controller import SessionController  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402
from tests.stt_fakes import FakeBackend  # noqa: E402

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


class _FakePlayer:
    """Заглушка QMediaPlayer: фиксирует play/stop, без аудиооборудования."""

    def __init__(self) -> None:
        self.play_calls = 0
        self.stop_calls = 0
        self._state = "Stopped"

    def play(self) -> None:
        self.play_calls += 1
        self._state = "Playing"

    def stop(self) -> None:
        self.stop_calls += 1
        self._state = "Stopped"

    def playbackState(self) -> str:
        return self._state

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


class FakeQProcess(QObject):
    """Подмена QProcess: фиксирует вызовы, реальный ``say`` не запускает.

    Компонент использует синхронные ``waitForStarted``/``waitForBytesWritten``
    и асинхронный ``finished`` — подмена покрывает оба пути. Выходной файл
    создаёт тест (``_complete_success``) — как это делает ``say`` при успехе.
    """

    finished = Signal(int, object)
    errorOccurred = Signal(object)

    # Классовый флаг: тест может «сломать» запуск процесса.
    start_ok = True
    instances: list["FakeQProcess"] = []

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.program: str | None = None
        self.args: list[str] = []
        self.stdin_data = bytearray()
        self.write_closed = False
        self.kill_calls = 0
        self.wait_finished_calls = 0
        self.stderr_data = b""
        self.state_value = QProcess.ProcessState.NotRunning
        FakeQProcess.instances.append(self)

    def start(self, program: str, args: list[str]) -> None:
        self.program = program
        self.args = list(args)
        if FakeQProcess.start_ok:
            self.state_value = QProcess.ProcessState.Running

    def waitForStarted(self, timeout_ms: int) -> bool:
        return FakeQProcess.start_ok

    def write(self, data: bytes) -> int:
        self.stdin_data.extend(data)
        return len(data)

    def closeWriteChannel(self) -> None:
        self.write_closed = True

    def waitForBytesWritten(self, timeout_ms: int) -> bool:
        return True

    def state(self) -> QProcess.ProcessState:
        return self.state_value

    def kill(self) -> None:
        self.kill_calls += 1
        self.state_value = QProcess.ProcessState.NotRunning

    def waitForFinished(self, timeout_ms: int) -> bool:
        self.wait_finished_calls += 1
        return True

    def readAllStandardError(self) -> bytes:
        return self.stderr_data

    def deleteLater(self) -> None:
        pass


class TtsTestBase(unittest.TestCase):
    """Фасад ``AudioController`` с подменёнными синтезом и плеером."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
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
        # Синтез: реальный say не запускаем.
        self._p_process = mock.patch("companion.tts_synth.QProcess", FakeQProcess)
        self._p_process.start()
        FakeQProcess.instances.clear()

    def tearDown(self) -> None:
        try:
            self.c.shutdown()
            _pump(0.1)
        finally:
            self._p_output.stop()
            self._p_devices.stop()
            self._p_process.stop()
            FakeQProcess.instances.clear()
            del self.c

    # --- Вспомогательные ---------------------------------------------------

    def _last(self) -> FakeQProcess:
        self.assertGreaterEqual(
            len(FakeQProcess.instances), 1, "процесс синтеза не запущен"
        )
        return FakeQProcess.instances[-1]

    def _complete_success(self, text: bytes = b"fake-aiff-bytes") -> str:
        """Имитирует завершение ``say``: файл создан, finished(success)."""
        proc = self._last()
        out_path = proc.args[-1]
        with open(out_path, "wb") as f:
            f.write(text)
        proc.finished.emit(0, QProcess.ExitStatus.NormalExit)
        return out_path


class TestSpeakReplySuccess(TtsTestBase):
    """Успешный ответ: команда, stdin, готовый AIFF, автозапуск."""

    def test_command_voice_and_stdin(self) -> None:
        self.c.speak_reply("Hello teacher")
        proc = self._last()
        self.assertEqual(proc.program, "say")
        # Английский голос из конфига (по умолчанию Samantha).
        self.assertIn("-v", proc.args)
        self.assertEqual(proc.args[proc.args.index("-v") + 1], "Samantha")
        # Текст — данные через stdin; выходной файл задан явно (AIFF).
        self.assertIn("-f", proc.args)
        self.assertEqual(proc.args[proc.args.index("-f") + 1], "-")
        self.assertIn("-o", proc.args)
        self.assertTrue(proc.args[-1].endswith(".aiff"))
        self.assertEqual(bytes(proc.stdin_data), b"Hello teacher")
        self.assertTrue(proc.write_closed)

    def test_reply_ready_autoplays_when_audio_free(self) -> None:
        self.c.speak_reply("Hello teacher")
        # Синтез не занимает аудио: запись/воспроизведение не запущены.
        self.assertEqual(self.c.state, AudioController.IDLE)
        out_path = self._complete_success()
        self.assertEqual(self.c.state, AudioController.PLAYING)
        self.assertEqual(self.fake_player.play_calls, 1)
        self.assertTrue(self.c.has_reply)
        self.assertEqual(self.c.tts.last_reply_path, out_path)

    def test_empty_text_rejected(self) -> None:
        self.assertFalse(self.c.speak_reply("   "))
        self.assertEqual(len(FakeQProcess.instances), 0)
        self.assertIn("нечего", self.c.status)

    def test_new_reply_replaces_inflight_synthesis(self) -> None:
        self.c.speak_reply("first reply")
        first = FakeQProcess.instances[-1]
        self.c.speak_reply("second reply")
        second = FakeQProcess.instances[-1]
        self.assertGreaterEqual(
            first.kill_calls, 1, "предыдущий синтез не остановлен"
        )
        # Готовым становится ответ второго файла.
        out_path = self._complete_success()
        self.assertEqual(out_path, second.args[-1])
        self.assertEqual(self.c.tts.last_reply_path, out_path)
        self.assertEqual(self.fake_player.play_calls, 1)


class TestReplayReply(TtsTestBase):
    """«Повторить ответ»: тот же готовый файл, без нового синтеза."""

    def test_replay_plays_same_file_without_new_synth(self) -> None:
        self.c.speak_reply("Hello teacher")
        out_path = self._complete_success()
        self.c.stop_playback()
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertTrue(self.c.can_replay)
        before = len(FakeQProcess.instances)
        self.c.replay_reply()
        self.assertEqual(
            len(FakeQProcess.instances), before, "повтор запустил синтез"
        )
        self.assertEqual(self.fake_player.play_calls, 2)
        self.assertEqual(self.c.tts.last_reply_path, out_path)

    def test_replay_without_reply_rejected(self) -> None:
        self.assertFalse(self.c.can_replay)
        self.c.replay_reply()
        self.assertEqual(self.fake_player.play_calls, 0)
        self.assertIn("Повторять нечего", self.c.status)

    def test_replay_while_busy_rejected(self) -> None:
        self.c.speak_reply("Hello teacher")
        self._complete_success()
        self.assertEqual(self.c.state, AudioController.PLAYING)
        self.c.replay_reply()
        self.assertEqual(
            self.fake_player.play_calls, 1, "повтор во время воспроизведения"
        )


class TestSynthErrors(TtsTestBase):
    """Ошибка синтеза: нет автозапуска, понятный статус, нет частичного файла."""

    def test_failure_no_autoplay_and_clear_status(self) -> None:
        self.c.speak_reply("Hello teacher")
        proc = self._last()
        proc.stderr_data = b"Opening output file failed: wht?"
        out_path = proc.args[-1]
        proc.finished.emit(1, QProcess.ExitStatus.NormalExit)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertEqual(
            self.fake_player.play_calls, 0, "автозапуск при ошибке синтеза"
        )
        self.assertFalse(self.c.has_reply)
        self.assertIn("Не удалось озвучить ответ", self.c.status)
        self.assertIn("Opening output file failed", self.c.status)
        self.assertFalse(os.path.exists(out_path), "остался частичный файл")

    def test_start_failure_reports_and_recovers(self) -> None:
        FakeQProcess.start_ok = False
        try:
            self.c.speak_reply("Hello teacher")
            self.assertEqual(self.c.state, AudioController.IDLE)
            self.assertIn("Не удалось озвучить ответ", self.c.status)
            self.assertEqual(self.fake_player.play_calls, 0)
            self.assertFalse(self.c.tts.is_busy, "синтезатор остался занятым")
        finally:
            FakeQProcess.start_ok = True
        # Синтезатор жив: следующий синтез завершается успешно.
        self.c.speak_reply("Hello teacher")
        self._complete_success()
        self.assertEqual(self.c.state, AudioController.PLAYING)

    def test_failed_reply_does_not_lose_previous_ready(self) -> None:
        """Сбой нового ответа не стирает предыдущий готовый (повтор жив)."""
        self.c.speak_reply("first reply")
        first = self._complete_success()
        self.c.stop_playback()
        self.c.speak_reply("second reply")
        self._last().finished.emit(1, QProcess.ExitStatus.NormalExit)
        self.assertEqual(self.c.tts.last_reply_path, first)
        self.assertTrue(self.c.can_replay)


class TestBusyAudio(TtsTestBase):
    """Занятое аудио: автозапуска нет, ответ ждёт ручного повтора."""

    def test_busy_recording_keeps_reply_for_manual_replay(self) -> None:
        self.c.recorder.state = AudioRecorder.RECORDING
        try:
            self.c.speak_reply("Hello teacher")
            self._complete_success()
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
        self.assertEqual(self.fake_player.play_calls, 1)

    def test_busy_playback_keeps_reply_for_manual_replay(self) -> None:
        self.c.player.state = AudioPlayer.PLAYING
        try:
            self.c.speak_reply("Hello teacher")
            self._complete_success()
            self.assertEqual(
                self.fake_player.play_calls, 0, "автозапуск во время playback"
            )
            self.assertIn("занято", self.c.status)
            self.assertTrue(self.c.has_reply)
        finally:
            self.c.player.state = AudioPlayer.IDLE
        self.c.replay_reply()
        self.assertEqual(self.fake_player.play_calls, 1)


class TestCloseDuringSynthesis(TtsTestBase):
    """Закрытие во время синтеза: процесс остановлен, файлы удалены."""

    def test_shutdown_kills_process_and_cleans_files(self) -> None:
        self.c.speak_reply("Hello teacher")
        proc = self._last()
        temp_dir = self.c.tts._temp_dir
        self.assertIsNotNone(temp_dir)
        self.assertTrue(os.path.isdir(temp_dir))

        self.c.shutdown()
        _pump(0.1)

        self.assertGreaterEqual(
            proc.kill_calls, 1, "процесс синтеза не остановлен"
        )
        self.assertFalse(os.path.isdir(temp_dir), "осталась временная папка")
        # Компонент мёртв: новый синтез и повтор отклоняются.
        self.assertFalse(self.c.can_replay)
        self.c.speak_reply("again")
        self.assertEqual(
            len(FakeQProcess.instances), 1, "синтез после shutdown"
        )
        self.c.replay_reply()
        self.assertEqual(self.fake_player.play_calls, 0)


class TtsWindowTestBase(unittest.TestCase):
    """Окно + фасад + подменённые синтез и плеер."""

    def setUp(self) -> None:
        _app()
        config = AppConfig()
        client = OllamaClient(config)
        self.controller = SessionController(client)
        self.audio = AudioController(config.tts_voice)
        self.stt = SttEngine(FakeBackend(result_text="x"))
        self.window = MainWindow(self.controller, self.audio, self.stt)
        # Граница AudioPlayer: без реальных устройств.
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
        self.audio.player._player = self.fake_player
        # Синтез: реальный say не запускаем.
        self._p_process = mock.patch("companion.tts_synth.QProcess", FakeQProcess)
        self._p_process.start()
        FakeQProcess.instances.clear()
        self.window.show()
        _pump(0.1)

    def tearDown(self) -> None:
        try:
            self.window.close()
            self.stt.shutdown()
            self.audio.shutdown()
            _pump(0.1)
        finally:
            self._p_output.stop()
            self._p_devices.stop()
            self._p_process.stop()
            FakeQProcess.instances.clear()
            del self.window

    def _complete_success(self) -> str:
        proc = FakeQProcess.instances[-1]
        out_path = proc.args[-1]
        with open(out_path, "wb") as f:
            f.write(b"fake-aiff-bytes")
        proc.finished.emit(0, QProcess.ExitStatus.NormalExit)
        return out_path


class TestWindowSpeakReply(TtsWindowTestBase):
    """Окно: ответ преподавателя — текст в чате, затем синтез; повтор."""

    def test_assistant_reply_visible_then_spoken(self) -> None:
        self.controller.message_added.emit(
            Message("assistant", "Great job! Try again.", "llama3")
        )
        _pump(0.1)
        # Текст сразу виден в чате...
        self.assertIn(
            "Great job! Try again.", self.window.history.toPlainText()
        )
        # ...и синтез запущен текстом из ответа.
        self.assertGreaterEqual(len(FakeQProcess.instances), 1)
        proc = FakeQProcess.instances[-1]
        self.assertEqual(bytes(proc.stdin_data), b"Great job! Try again.")
        # Пока синтез не завершён — «Повторить ответ» недоступен.
        self.assertFalse(self.window.repeat_button.isEnabled())

    def test_user_message_not_spoken(self) -> None:
        self.controller.message_added.emit(Message("user", "hello"))
        _pump(0.1)
        self.assertEqual(
            len(FakeQProcess.instances), 0, "реплика пользователя озвучена"
        )
        self.assertIn("hello", self.window.history.toPlainText())

    def test_repeat_button_works_after_ready(self) -> None:
        self.controller.message_added.emit(
            Message("assistant", "Hello there!", "llama3")
        )
        _pump(0.1)
        self._complete_success()
        _pump(0.1)
        # Ответ автозапустился: во время воспроизведения кнопка
        # недоступна (аудио занято) — как и «Прослушать» во время playback.
        self.assertEqual(self.audio.state, AudioController.PLAYING)
        self.assertFalse(self.window.repeat_button.isEnabled())
        # После «Стоп» — доступна.
        self.window.play_button.click()  # «Стоп»
        _pump(0.1)
        self.assertEqual(self.audio.state, AudioController.IDLE)
        self.assertTrue(self.window.repeat_button.isEnabled())
        # Кнопка проигрывает готовый ответ без нового синтеза.
        before = len(FakeQProcess.instances)
        self.window.repeat_button.click()
        _pump(0.1)
        self.assertEqual(len(FakeQProcess.instances), before)
        self.assertEqual(self.fake_player.play_calls, 2)


class TestWindowCloseAndErrors(TtsWindowTestBase):
    """Окно: закрытие во время синтеза; ошибка оставляет текст доступным."""

    def test_close_during_synthesis_stops_and_cleans(self) -> None:
        self.controller.message_added.emit(
            Message("assistant", "Hello there!", "llama3")
        )
        _pump(0.1)
        proc = FakeQProcess.instances[-1]
        temp_dir = self.audio.tts._temp_dir
        self.assertTrue(os.path.isdir(temp_dir))

        self.window.close()
        _pump(0.2)

        self.assertFalse(self.window.isVisible())
        self.assertGreaterEqual(
            proc.kill_calls, 1, "синтез не остановлен при закрытии"
        )
        self.assertFalse(os.path.isdir(temp_dir), "осталась временная папка")

    def test_error_keeps_text_and_disables_repeat(self) -> None:
        self.controller.message_added.emit(
            Message("assistant", "Hello teacher!", "llama3")
        )
        _pump(0.1)
        proc = FakeQProcess.instances[-1]
        proc.finished.emit(1, QProcess.ExitStatus.NormalExit)
        _pump(0.1)
        # Текст ответа остаётся доступным в чате.
        self.assertIn("Hello teacher!", self.window.history.toPlainText())
        # Понятный статус; «Повторить ответ» недоступен.
        self.assertIn(
            "Не удалось озвучить ответ", self.window.audio_status_label.text()
        )
        self.assertFalse(self.window.repeat_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
