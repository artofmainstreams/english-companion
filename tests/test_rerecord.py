"""«Перезаписать» последнюю реплику пользователя: окно + контроллер + аудио.

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Подмены (без микрофона, сети, TTS и скачивания весов):
* запись — эмит ``recording_completed`` (как в приложении запускает STT),
  отмена — реальный ``AudioController.cancel_recording``;
* STT — ``SttEngine`` + ``FakeBackend`` (tests/stt_fakes.py);
* Ollama — mock ``send_chat`` либо подмена сетевого слоя;
* TTS — подмена ``speak_reply`` (ответы не озвучиваются), повтор —
  логика фасада (suppress/unsuppress) с фейковым синтезатором.

Проверяются:
* успешная замена: реплика и связанный ответ убраны из окна и истории,
  новый запрос — с исправленной историей (payload проверен), новый ответ
  показан и озвучен; окно и история к Ollama согласованы;
* замена, пока ответ в работе: запрос отменён, поздний результат отбрасывается;
* отмена / пустой результат / ошибка STT — прежний обмен на месте,
  незавершённый старый ответ НЕ запускается;
* одна перезапись — максимум одна замена и один новый запрос (без дублей);
* «Повторить ответ»: недоступен во время перезаписи и после замены,
  после отмены возвращается (если готовый файл сохранился), после нового
  ответа — новый;
* сбой нового запроса: исправленная реплика сохранена, повтор — без
  дублирования сообщения пользователя.
"""

import faulthandler
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from unittest import mock

faulthandler.enable()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtNetwork import QNetworkReply  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.audio_recorder import AudioRecorder  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.main_window import MainWindow  # noqa: E402
from companion.message import Message  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.session_controller import SessionController  # noqa: E402
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
    """Ждём, пока движок не станет свободным, и доставляем сигналы."""
    import time

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


class RerecordTestBase(unittest.TestCase):
    """Окно + реальный AudioController + SttEngine/FakeBackend; Ollama — mock."""

    def setUp(self) -> None:
        _app()
        self.config = AppConfig()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-rerecord-")
        self.audio_path = os.path.join(self.tmp, "recording_1.m4a")
        _make_audio(self.audio_path)

        client = OllamaClient(self.config)
        self.controller = SessionController(client)
        self.audio = AudioController()
        # Озвучивание ответов не нужно — подменяем (тесты без TTS/аудио).
        self.audio.speak_reply = mock.MagicMock(name="speak_reply")
        # Реальный микрофон в тестах не используем: старт записи — фикс.
        self.audio.record = mock.MagicMock(name="record")
        # «Завершённая запись» (тестовый путь; в приложении так же —
        # сигнал поднимает рекордер, окно берёт путь из сигнала).
        self.audio.recorder._temp_dir = self.tmp
        self.audio.recorder._record_file = self.audio_path
        self.audio.recorder._ready_recording = self.audio_path

        self.backend = FakeBackend(result_text="a quiet room, please", delay=0.2)
        self.stt = SttEngine(self.backend)
        self.window = MainWindow(self.controller, self.audio, self.stt)
        self.window.show()
        _pump(0.1)

    def tearDown(self) -> None:
        try:
            self.stt.shutdown()
            self.audio.shutdown()
            _pump(0.1)
        finally:
            if os.path.isdir(self.tmp):
                shutil.rmtree(self.tmp, ignore_errors=True)
            del self.window

    def _allow_send(self):
        """Делаем отправку доступной: модель выбрана, сетевой вызов подменён."""
        send_mock = mock.MagicMock(name="send_chat")
        self.controller.client.send_chat = send_mock
        self.controller.selected_model = "m"
        return send_mock

    def _seed_exchange(
        self, user_text: str = "I want a room", reply_text: str = "The room is 50 dollars."
    ):
        """Один завершённый обмен (история непуста). Возвращает mock."""
        send_mock = self._allow_send()
        self.controller.send_message(user_text)
        self.controller.client.chat_completed.emit(reply_text)
        return send_mock

    def _set_recording(self, recording: bool) -> None:
        """Имитируем состояние рекордера (запись идёт / остановлена)."""
        self.audio.recorder.state = (
            AudioRecorder.RECORDING if recording else AudioRecorder.IDLE
        )
        self.audio.state_changed.emit()
        _pump(0.05)

    def _finish_recording(self) -> None:
        """Имитируем успешное завершение записи (сигнал → авто-STT)."""
        self.audio.recording_completed.emit(self.audio_path)

    def _rerecord_and_finish(self) -> None:
        """Клик «Перезаписать» → запись → остановка → STT → результат."""
        self.window.on_rerecord()
        self._set_recording(True)
        self._set_recording(False)
        self._finish_recording()
        _wait_stt_idle(self.stt)


class TestRerecordAvailability(RerecordTestBase):
    """Доступность действия: последняя реплика пользователя, без дублей."""

    def test_disabled_without_user_message(self) -> None:
        self.assertFalse(self.window.rerecord_button.isEnabled())

    def test_enabled_for_last_user_message(self) -> None:
        self._seed_exchange()
        self.assertTrue(self.window.rerecord_button.isEnabled())

    def test_enabled_while_reply_in_flight(self) -> None:
        """Ответ в работе: реплика в «ожидании» — перезапись разрешена."""
        self._allow_send()
        self.controller.send_message("I want a room")
        self.assertTrue(self.controller.busy)
        self.assertTrue(self.window.rerecord_button.isEnabled())

    def test_disabled_during_active_rerecord(self) -> None:
        self._seed_exchange()
        self.window.on_rerecord()
        self.assertFalse(self.window.rerecord_button.isEnabled())
        self.assertTrue(self.window.rerecord_cancel_button.isEnabled())
        # Очистка состояния: отмена.
        self.window.on_rerecord_cancel()
        self.assertFalse(self.window._rerecord_active)

    def test_disabled_while_stt_busy(self) -> None:
        self._seed_exchange()
        # STT занят (имитация: запись завершилась, распознавание идёт).
        self._finish_recording()
        self.assertTrue(self.stt.is_busy)
        self.assertFalse(self.window.rerecord_button.isEnabled())
        _wait_stt_idle(self.stt)


class TestSuccessfulRerecord(RerecordTestBase):
    """Успешная замена: окно и история к Ollama согласованы."""

    def test_replaces_message_and_reply(self) -> None:
        send_mock = self._seed_exchange()
        self.assertIn("You: I want a room", self.window.history.toPlainText())
        self.assertIn("Teacher", self.window.history.toPlainText())

        self.backend.result_text = "I want a quiet room, please"
        self._rerecord_and_finish()

        # Клик «Перезаписать» запустил новую запись (подменённый старт).
        self.audio.record.assert_called_once()

        # Прежний обмен убран из окна; новая реплика — одна.
        text = self.window.history.toPlainText()
        self.assertNotIn("You: I want a room", text)
        self.assertNotIn("Teacher", text)
        self.assertEqual(text.count("You: I want a quiet room, please"), 1)
        # История: старый обмен удалён, новая реплика ещё в работе.
        self.assertEqual(self.controller.history, [])
        self.assertEqual(
            self.controller.pending_message,
            Message("user", "I want a quiet room, please"),
        )
        # Один новый запрос (после seed) — с исправленной историей.
        self.assertEqual(send_mock.call_count, 2)
        args, kwargs = send_mock.call_args
        self.assertEqual(
            [m.content for m in args[1]], ["I want a quiet room, please"]
        )
        self.assertIn("the student's practice material", kwargs.get("system", ""),
                      "системный контекст урока уходит с запросом")
        # Повтор старого ответа недоступен.
        self.assertFalse(self.window.repeat_button.isEnabled())

        # Новый ответ: показан и озвучен, история согласована.
        self.controller.client.chat_completed.emit("We have a quiet room.")
        self.assertEqual(
            self.controller.history,
            [
                Message("user", "I want a quiet room, please"),
                Message("assistant", "We have a quiet room.", "m"),
            ],
        )
        text = self.window.history.toPlainText()
        self.assertEqual(text.count("You: I want a quiet room, please"), 1)
        self.assertIn("Teacher · m: We have a quiet room.", text)
        self.assertEqual(
            self.audio.speak_reply.call_count, 2,
            "озвучены seed-ответ и новый ответ",
        )
        self.assertEqual(self.audio.speak_reply.call_args[0][0], "We have a quiet room.")
        # Retry сброшен после успеха.
        self.assertIsNone(self.controller.retry_message)

    def test_payload_with_earlier_exchange(self) -> None:
        """История перед заменяемой репликой сохраняется в payload."""
        send_mock = self._seed_exchange()  # [u1, a1]
        # Второй ход: [u1, a1, u2] — ответ a2.
        self.controller.send_message("I want a quiet room")
        self.controller.client.chat_completed.emit("Sure, 60 dollars.")
        self.assertEqual(len(self.controller.history), 4)

        self.backend.result_text = "How much is a quiet room?"
        self._rerecord_and_finish()

        self.assertEqual(send_mock.call_count, 3)
        args, _kwargs = send_mock.call_args
        # u1, a1 — сохранены; u2 заменён; a2 убран.
        self.assertEqual(
            [m.content for m in args[1]],
            ["I want a room", "The room is 50 dollars.", "How much is a quiet room?"],
        )
        text = self.window.history.toPlainText()
        self.assertIn("You: I want a room", text)
        self.assertNotIn("You: I want a quiet room", text)
        self.assertNotIn("Sure, 60 dollars.", text)


class _FakeReply(QObject):
    """Сетевой ответ с телом Ollama; фиксирует ``abort``."""

    finished = Signal()

    def __init__(self, body: bytes) -> None:
        super().__init__()
        self._body = body
        self.abort_calls = 0

    def error(self):
        return QNetworkReply.NetworkError.NoError

    def errorString(self) -> str:
        return ""

    def readAll(self) -> bytes:
        return self._body

    def abort(self) -> None:
        self.abort_calls += 1

    def deleteLater(self) -> None:
        pass


class _FakeNetwork:
    """Подмена QNetworkAccessManager: ловит payload, отвечает очередью
    готовых тел Ollama (по одному на запрос, в порядке запросов)."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.replies: list[_FakeReply] = []
        self.last_payload: dict | None = None

    def post(self, request, data: bytes):
        self.last_payload = json.loads(bytes(data).decode("utf-8"))
        answer = self.answers.pop(0) if self.answers else "ok"
        body = json.dumps(
            {"message": {"role": "assistant", "content": answer}, "done": True}
        ).encode("utf-8")
        reply = _FakeReply(body)
        self.replies.append(reply)
        QTimer.singleShot(0, reply.finished.emit)
        return reply


class TestRerecordInFlight(RerecordTestBase):
    """Ответ в работе: запрос отменён, поздний результат отброшен."""

    def test_in_flight_replaced_stale_dropped(self) -> None:
        """Поздний ``finished`` отменённого запроса не порождает сигналов
        и не меняет историю (реальный клиент + подмена сетевого слоя)."""
        net = _FakeNetwork(answers=["OLD ANSWER", "NEW ANSWER"])
        self.controller.client.network = net
        self.controller.selected_model = "m"

        self.controller.send_message("I want a room")
        self.assertTrue(self.controller.busy)
        self.assertIn("You: I want a room", self.window.history.toPlainText())

        # «Перезаписать»: текущий ответ отменён, новая запись запущена.
        self.window.on_rerecord()
        self.assertEqual(len(net.replies), 1)
        self.assertEqual(net.replies[0].abort_calls, 1, "запрос отменён")
        self.assertIsNone(self.controller.client._chat_reply)

        # Запись → STT → замена: новый запрос получил «NEW ANSWER».
        self.backend.result_text = "a quiet room, please"
        self._set_recording(True)
        self._set_recording(False)
        self._finish_recording()
        _wait_stt_idle(self.stt)
        _pump(0.1)  # finished обоих «ответов» — в прогоне цикла

        # Новый ответ пришёл к НОВОЙ реплике; окно согласовано с историей.
        self.assertEqual(
            self.controller.history,
            [
                Message("user", "a quiet room, please"),
                Message("assistant", "NEW ANSWER", "m"),
            ],
        )
        text = self.window.history.toPlainText()
        self.assertNotIn("You: I want a room", text)
        self.assertNotIn("OLD ANSWER", text)
        self.assertIn("NEW ANSWER", text)
        self.assertEqual(text.count("You: "), 1)
        self.assertEqual(text.count("Teacher"), 1)

        # Запоздалый finished отменённого запроса — слот отключён, тихо.
        net.replies[0].finished.emit()
        _pump(0.1)
        self.assertEqual(
            [m.content for m in self.controller.history],
            ["a quiet room, please", "NEW ANSWER"],
        )
        self.assertEqual(text.count("You: "), 1)

    def test_in_flight_cancel_keeps_message_without_replay(self) -> None:
        """Отмена: прежняя реплика возвращена, запрос НЕ запускается."""
        send_mock = self._allow_send()
        self.controller.send_message("I want a room")

        self.window.on_rerecord()
        self.assertIsNone(self.controller.pending_message)
        self.assertFalse(self.controller.busy)
        self.window.on_rerecord_cancel()

        # Прежний обмен на месте: реплика в «ждёт ответ», без нового запроса.
        self.assertEqual(
            self.controller.pending_message, Message("user", "I want a room")
        )
        self.assertEqual(
            self.controller.retry_message, Message("user", "I want a room")
        )
        self.assertFalse(self.controller.busy)
        self.assertEqual(
            send_mock.call_count, 1,
            "незавершённый старый ответ не запускается автоматически",
        )
        self.assertIn("You: I want a room", self.window.history.toPlainText())
        # Повтор (файла нет — ответа не было) всё равно недоступен:
        # ответ отменён.
        self.assertFalse(self.window.repeat_button.isEnabled())
        # Реплика retry-абельна — и повтор не дублирует её в чате.
        self.assertTrue(self.window.retry_button.isEnabled())
        self.window.on_retry_request()
        self.assertEqual(send_mock.call_count, 2)
        args, _kwargs = send_mock.call_args
        self.assertEqual([m.content for m in args[1]], ["I want a room"])
        self.assertEqual(
            self.window.history.toPlainText().count("You: I want a room"), 1,
            "повтор запроса не дублирует реплику",
        )


class TestRerecordAbort(RerecordTestBase):
    """Отмена / пустой результат / ошибка STT: прежний обмен на месте."""

    def test_cancel_keeps_exchange(self) -> None:
        send_mock = self._seed_exchange()
        before = self.window.history.toPlainText()

        self.window.on_rerecord()
        self._set_recording(True)
        self.window.on_rerecord_cancel()

        self.assertEqual(self.window.history.toPlainText(), before)
        self.assertEqual(
            [m.content for m in self.controller.history],
            ["I want a room", "The room is 50 dollars."],
        )
        self.assertEqual(send_mock.call_count, 1, "запросов не было")
        self.assertFalse(self.window._rerecord_active)
        self.assertTrue(
            self.window.rerecord_button.isEnabled(),
            "после отмены перезапись снова доступна",
        )

    def test_empty_result_keeps_exchange(self) -> None:
        send_mock = self._seed_exchange()
        before = self.window.history.toPlainText()
        self.backend.result_text = ""

        self._rerecord_and_finish()

        self.assertEqual(self.window.history.toPlainText(), before)
        self.assertEqual(send_mock.call_count, 1, "запросов не было")
        self.assertIn("не завершена", self.window.status_label.text())
        self.assertFalse(self.window._rerecord_active)

    def test_stt_error_keeps_exchange(self) -> None:
        send_mock = self._seed_exchange()
        before = self.window.history.toPlainText()
        self.backend.raise_on_transcribe = RuntimeError("boom")

        self._rerecord_and_finish()

        self.assertEqual(self.window.history.toPlainText(), before)
        self.assertEqual(send_mock.call_count, 1, "запросов не было")
        self.assertIn("Не удалось распознать", self.window.stt_status_label.text())
        self.assertFalse(self.window._rerecord_active)


class TestOneRerecordOneReplacement(RerecordTestBase):
    """Одна перезапись — максимум одна замена и один новый запрос."""

    def test_double_click_single_replacement(self) -> None:
        send_mock = self._seed_exchange()
        self.window.on_rerecord()
        self.window.on_rerecord()  # повторный клик — игнор
        self._set_recording(True)
        self._set_recording(False)
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(send_mock.call_count, 2)
        self.assertEqual(
            self.window.history.toPlainText().count("You: a quiet room, please"), 1
        )

    def test_second_recording_not_doubled(self) -> None:
        """Повторный эмит записи во время ожидания ответа — без дубля."""
        send_mock = self._seed_exchange()
        self._rerecord_and_finish()
        self.assertTrue(self.controller.busy)
        # «Вторая запись» пришло: расшифрована, но замена невозможна
        # (реплика в работе) — дублей нет.
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(send_mock.call_count, 2)
        self.assertEqual(
            self.window.history.toPlainText().count("You: a quiet room, please"), 1
        )


class TestRetryAfterFailedRequest(RerecordTestBase):
    """Сбой нового запроса: реплика сохранена, повтор без дублирования."""

    def test_failed_request_retry_without_duplicate(self) -> None:
        send_mock = self._seed_exchange()
        self.backend.result_text = "a quiet room, please"
        self.window.on_rerecord()
        self._set_recording(True)
        self._set_recording(False)
        self._finish_recording()
        _wait_stt_idle(self.stt)
        # Замена ушла: реплика в чате, в работе (seed + замена = 2 запроса).
        self.assertIn("You: a quiet room, please", self.window.history.toPlainText())
        self.assertEqual(send_mock.call_count, 2)

        # Запрос завершился ошибкой: исправленная реплика СОХРАНИЛАСЯ.
        self.controller.client.chat_failed.emit("timeout")
        text = self.window.history.toPlainText()
        self.assertEqual(text.count("You: a quiet room, please"), 1)
        self.assertEqual(self.controller.history, [])
        self.assertEqual(
            self.controller.retry_message, Message("user", "a quiet room, please")
        )
        self.assertTrue(self.window.retry_button.isEnabled())
        self.assertIn("повторите запрос", self.controller.status)

        # Повтор — без дублирования реплики.
        self.window.on_retry_request()
        self.assertEqual(send_mock.call_count, 3)
        args, _kwargs = send_mock.call_args
        self.assertEqual([m.content for m in args[1]], ["a quiet room, please"])
        self.assertEqual(
            self.window.history.toPlainText().count("You: a quiet room, please"), 1,
            "повтор не дублирует реплику",
        )

        # Успех: история согласована с окном.
        self.controller.client.chat_completed.emit("A quiet room is free.")
        self.assertEqual(
            self.controller.history,
            [
                Message("user", "a quiet room, please"),
                Message("assistant", "A quiet room is free.", "m"),
            ],
        )
        self.assertEqual(
            self.window.history.toPlainText().count("You: a quiet room, please"), 1
        )
        self.assertIsNone(self.controller.retry_message)

    def test_stale_failure_after_discard_ignored(self) -> None:
        """Поздний сбой отменённого запроса не затирает состояние."""
        self._allow_send()
        self.controller.send_message("I want a room")
        self.window.on_rerecord()
        # Поздний сбой отменённого запроса (pending уже сброшен).
        self.controller.client.chat_failed.emit("late failure")
        self.assertIsNone(self.controller.retry_message)
        self.assertEqual(self.controller.status, "Готов")
        # Отмена — прежний обмен на месте, без побочных статусов.
        self.window.on_rerecord_cancel()
        self.assertIn("You: I want a room", self.window.history.toPlainText())


class _FakeTts(QObject):
    """Фейковый синтезатор для проверки «Повторить ответ» (без say)."""

    state_changed = Signal()
    reply_ready = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.status = "tts"
        self.is_busy = False
        self._path: str | None = None

    @property
    def has_reply(self) -> bool:
        return self._path is not None

    @property
    def last_reply_path(self) -> str:
        return self._path

    def synthesize(self, text: str) -> bool:
        return False

    def shutdown(self) -> None:
        pass


class TestReplyReplayGate(unittest.TestCase):
    """«Повторить ответ»: suppress/unsuppress + новый ответ снимает запрет."""

    def setUp(self) -> None:
        _app()
        self.tts = _FakeTts()
        self.audio = AudioController(tts=self.tts)
        # Готовый ответ существует (файл — тестовый, плеер не запускаем).
        self.reply_file = os.path.join(
            tempfile.gettempdir(), "english-companion-fake-reply.wav"
        )
        with open(self.reply_file, "wb") as fh:
            fh.write(b"RIFF")
        self.tts._path = self.reply_file
        _pump(0.05)

    def tearDown(self) -> None:
        self.audio.shutdown()
        _pump(0.05)
        try:
            os.remove(self.reply_file)
        except OSError:
            pass

    def test_suppress_blocks_replay_until_cancel(self) -> None:
        self.assertTrue(self.audio.can_replay)
        self.audio.suppress_reply()
        self.assertFalse(self.audio.can_replay, "во время перезаписи — нет повтора")
        with mock.patch.object(self.audio.player, "play") as play:
            self.audio.replay_reply()
            play.assert_not_called()
        self.assertIn("отменён", self.audio.status)
        # Отмена: готовый файл сохранился — повтор возвращается.
        self.audio.unsuppress_reply()
        self.assertTrue(self.audio.can_replay)

    def test_new_reply_clears_suppression(self) -> None:
        self.audio.suppress_reply()
        self.assertFalse(self.audio.can_replay)
        # Новый готовый ответ (сигнал синтезатора) — повтор НОВОГО.
        with mock.patch.object(self.audio.player, "play") as play:
            self.audio._on_reply_ready(self.reply_file)
            play.assert_called_once()
        self.assertTrue(self.audio.can_replay)


if __name__ == "__main__":
    unittest.main()
