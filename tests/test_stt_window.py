"""Интеграционные проверки окна: авто-STT после записи и автоотправка.

Запуск::

    .venv/bin/python -m unittest discover -s tests -v

Голосовой ввод теперь автоматический: успешное завершение записи (сигнал
``recording_completed``) само запускает распознавание, а непустая расшифровка
сама отправляется через ``SessionController.send_message`` — в чат она
попадает как обычная реплика пользователя, а в поле ввода остаётся черновик.
Пустой результат, ошибка или недоступная отправка НЕ отправляют. Пока
реплика обрабатывается, новая запись блокируется.

Подмены (``tests/stt_fakes.py``): без микрофона, интернета и скачивания весов.
Завершение записи имитируется эмитом ``recording_completed`` (так же, как в
приложении запускается STT); сетевой вызов Ollama подменяется, чтобы проверить
отправку офлайн.

Проверяются:
* успешная запись → авто-STT → непустая расшифровка → автоотправка, черновик
  остался в поле и реплика появилась в чате;
* пустой результат / ошибка — ничего не отправлено, введённый текст сохранён;
* одна запись — максимум одна отправка (без дублей);
* отправка недоступна — расшифровка для ручной отправки, ручной черновик не
  затирается;
* пока идёт распознавание/ожидание ответа — запись заблокирована, после —
  снова доступна;
* закрытие окна откладывается, пока идёт распознавание, и файл не удаляется
  до завершения работы; после — окно закрывается и файл очищается.
"""

import faulthandler
import os
import shutil
import sys
import tempfile
import unittest
import wave
from unittest import mock

faulthandler.enable()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.audio_recorder import AudioRecorder  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.main_window import MainWindow  # noqa: E402
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
    """Ждём, пока движок не станет свободным, и сбрасываем очередь сигналов."""
    import time

    deadline = time.monotonic() + timeout
    while stt.is_busy and time.monotonic() < deadline:
        _pump(0.02)
    # Финальный прогон: доставим сигналы, которые рабочий поток поставил в
    # очередь до перехода в IDLE (recognized / failed / state_changed).
    _pump(0.1)


def _make_audio(path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)


class SttWindowTestBase(unittest.TestCase):
    """Общее окружение: окно + реальный AudioController + SttEngine/FakeBackend."""

    def setUp(self) -> None:
        _app()
        self.config = AppConfig()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-sttwin-")
        self.audio_path = os.path.join(self.tmp, "recording_1.m4a")
        _make_audio(self.audio_path)

        client = OllamaClient(self.config)
        controller = SessionController(client)
        self.audio = AudioController()
        # Задаём «завершённую запись» (в тесте допустимо; окно берёт путь
        # через публичный last_recording_path). Готовой является именно
        # ``_ready_recording``; ``_record_file`` — тот же файл текущей сессии.
        self.audio.recorder._temp_dir = self.tmp
        self.audio.recorder._record_file = self.audio_path
        self.audio.recorder._ready_recording = self.audio_path

        self.backend = FakeBackend(result_text="hello there", delay=0.2)
        self.stt = SttEngine(self.backend)
        self.window = MainWindow(controller, self.audio, self.stt)
        self.window.show()
        _pump(0.1)

    def _finish_recording(self) -> None:
        """Имитируем успешное завершение записи (сигнал → авто-STT).

        В приложении этот сигнал поднимает рекордер по факту завершения; здесь
        эмитим его, чтобы пройти ту же цепочку ``recording_completed`` → окно →
        ``stt.recognize`` (одна запись — один запуск).
        """
        self.audio.recording_completed.emit(self.audio_path)

    def _allow_send(self):
        """Делаем отправку доступной: модель выбрана, сетевой вызов подменён."""
        send_mock = mock.MagicMock(name="send_chat")
        self.window.controller.client.send_chat = send_mock
        self.window.controller.selected_model = "m"
        return send_mock

    def tearDown(self) -> None:
        try:
            self.stt.shutdown()
            self.audio.shutdown()
            _pump(0.1)
        finally:
            if os.path.isdir(self.tmp):
                shutil.rmtree(self.tmp, ignore_errors=True)
            del self.window


class TestAutoTriggerAndSend(SttWindowTestBase):
    """Успешная запись → авто-STT → непустая расшифровка → автоотправка."""

    def test_recording_completed_triggers_stt(self) -> None:
        """Сигнал завершения записи сам запускает распознавание."""
        self._allow_send()
        self.assertFalse(self.stt.is_busy)
        self._finish_recording()
        # После эмита STT уже работает (фоновый поток запущен).
        self.assertTrue(self.stt.is_busy)
        _wait_stt_idle(self.stt)
        self.assertFalse(self.stt.is_busy)

    def test_success_auto_sends_and_keeps_draft(self) -> None:
        """Непустая расшифровка: отправлена в чат, черновик остался в поле."""
        send_mock = self._allow_send()
        self.window.message_input.clear()
        self._finish_recording()
        _wait_stt_idle(self.stt)
        # Черновик остался в поле ввода.
        self.assertEqual(self.window.message_input.text(), "hello there")
        # И появилась в чате как обычная реплика пользователя.
        self.assertIn("You: hello there", self.window.history.toPlainText())
        # Ровно одна реплика (одна запись — одна отправка).
        self.assertEqual(
            self.window.history.toPlainText().count("You: hello there"), 1
        )
        send_mock.assert_called_once()


class TestEmptyAndErrorNoSend(SttWindowTestBase):
    """Пустой результат и ошибка НЕ отправляют сообщение."""

    def test_empty_result_no_send_keeps_draft(self) -> None:
        """Пустая расшифровка: ничего не отправлено, введённый текст сохранён."""
        self._allow_send()
        self.backend.result_text = ""
        self.window.message_input.setText("my typed sentence")
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "my typed sentence")
        self.assertEqual(len(self.window.controller.history), 0)
        self.assertNotIn(
            "You: my typed sentence", self.window.history.toPlainText()
        )

    def test_empty_result_no_send_empty_field(self) -> None:
        """Пустая расшифровка: пустое поле остаётся пустым, не отправлено."""
        self._allow_send()
        self.backend.result_text = ""
        self.window.message_input.clear()
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "")
        self.assertEqual(len(self.window.controller.history), 0)

    def test_error_no_send_keeps_draft(self) -> None:
        """Ошибка распознавания: ничего не отправлено, текст сохранён."""
        send_mock = self._allow_send()
        self.backend.raise_on_transcribe = RuntimeError("boom")
        self.window.message_input.setText("my typed sentence")
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "my typed sentence")
        self.assertEqual(len(self.window.controller.history), 0)
        send_mock.assert_not_called()
        self.assertIn("Не удалось распознать", self.window.stt_status_label.text())


class TestNoDuplicates(SttWindowTestBase):
    """Одна запись → максимум одна отправка; дублей нет."""

    def test_one_recording_one_send(self) -> None:
        """Один эмит сигнала — ровно одна отправка."""
        send_mock = self._allow_send()
        self._finish_recording()
        _wait_stt_idle(self.stt)
        send_mock.assert_called_once()
        self.assertEqual(
            self.window.history.toPlainText().count("You: hello there"), 1
        )

    def test_second_recording_while_busy_not_resent(self) -> None:
        """Пока «думает» первая, вторая запись НЕ создаёт дубль."""
        send_mock = self._allow_send()
        self._finish_recording()
        _wait_stt_idle(self.stt)
        # Первая отправка ушла; ответа нет (send_chat подменён) → busy.
        self.assertTrue(self.window.controller.busy)
        # Вторая запись: расшифрована, но отправка недоступна (busy).
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(send_mock.call_count, 1)
        self.assertEqual(
            self.window.history.toPlainText().count("You: hello there"), 1
        )


class TestManualDraftPreserved(SttWindowTestBase):
    """Отправка недоступна: расшифровка для вручную, черновик не затирается."""

    def test_send_unavailable_keeps_manual_draft(self) -> None:
        """Ручной черновик не затирается; расшифровка видна в статусе."""
        # can_send False: модель не выбрана (send_chat не трогаем).
        self.assertFalse(self.window.controller.can_send)
        self.window.message_input.setText("my manual draft")
        self._finish_recording()
        _wait_stt_idle(self.stt)
        # Черновик не затёрт, ничего не отправлено.
        self.assertEqual(self.window.message_input.text(), "my manual draft")
        self.assertEqual(len(self.window.controller.history), 0)
        # Расшифровка показана (строка статуса: движок вернул свою строку).
        self.assertIn("hello there", self.window.status_label.text())

    def test_send_unavailable_puts_transcript_in_empty_field(self) -> None:
        """Пустое поле: расшифровка кладётся в поле для ручной отправки."""
        self.assertFalse(self.window.controller.can_send)
        self.window.message_input.clear()
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "hello there")
        self.assertEqual(len(self.window.controller.history), 0)


class TestRecordGate(SttWindowTestBase):
    """Пока реплика обрабатывается, новая запись заблокирована."""

    def test_record_blocked_while_stt_busy(self) -> None:
        """Во время распознавания запись недоступна."""
        self._finish_recording()
        self.assertTrue(self.stt.is_busy)
        self.assertFalse(self.window.record_button.isEnabled())
        _wait_stt_idle(self.stt)

    def test_record_blocked_while_controller_busy(self) -> None:
        """Во время ожидания ответа запись недоступна."""
        self._allow_send()
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertTrue(self.window.controller.busy)
        self.assertFalse(self.window.record_button.isEnabled())

    def test_record_enabled_when_idle(self) -> None:
        """После завершения (без ожидания ответа) запись снова доступна."""
        # can_send False → отправки нет → controller не busy.
        self._finish_recording()
        _wait_stt_idle(self.stt)
        self.assertFalse(self.window.controller.busy)
        self.assertFalse(self.stt.is_busy)
        self.assertTrue(self.window.record_button.isEnabled())


class TestCloseHandling(SttWindowTestBase):
    """Закрытие во время STT отложено; после — окно закрывается, файл очищен."""

    def test_close_deferred_during_stt_then_closes(self) -> None:
        """Закрытие во время STT отложено; после завершения окно закрывается."""
        self._finish_recording()
        self.assertTrue(self.stt.is_busy)
        # Файл ещё нужен — должен остаться.
        self.assertTrue(os.path.exists(self.audio_path))

        self.window.close()  # closeEvent: откладываем (event.ignore())
        # Пока идёт распознавание, окно должно оставаться «живым».
        _pump(0.1)
        self.assertTrue(self.window.isVisible() or not self.window.isHidden())

        _wait_stt_idle(self.stt)
        # После завершения отложенное закрытие должно сработать.
        _pump(0.2)
        self.assertFalse(self.window.isVisible())
        # Файл удалён только после завершения STT и корректного разбора.
        self.assertFalse(os.path.exists(self.audio_path))

    def test_close_when_idle_is_immediate(self) -> None:
        """Закрытие в IDLE: окно закрывается сразу, файл очищен."""
        self.assertFalse(self.stt.is_busy)
        self.window.close()
        _pump(0.2)
        self.assertFalse(self.window.isVisible())
        self.assertFalse(os.path.exists(self.audio_path))


class TestMicCombo(unittest.TestCase):
    """Комбо микрофонов: itemData — стабильный id устройства,
    отображение — название; выбор согласован с AudioController."""

    def setUp(self) -> None:
        _app()
        self.config = AppConfig()
        self.audio = AudioController()
        controller = SessionController(OllamaClient(self.config))
        self.stt = SttEngine(FakeBackend(result_text="x"))
        self.w = MainWindow(controller, self.audio, self.stt)
        self.w.show()
        _pump(0.1)

    def tearDown(self) -> None:
        self.stt.shutdown()
        self.audio.shutdown()
        _pump(0.1)
        del self.w

    def test_item0_system_and_items_carry_device_ids(self) -> None:
        self.assertEqual(
            self.w.mic_combo.itemText(0), "Системный (по умолчанию)"
        )
        self.assertIsNone(self.w.mic_combo.itemData(0))
        pairs = self.audio.input_devices()
        self.assertEqual(self.w.mic_combo.count(), 1 + len(pairs))
        for i, (dev_id, name) in enumerate(pairs, start=1):
            self.assertEqual(
                self.w.mic_combo.itemData(i), dev_id, "itemData не id"
            )
            self.assertEqual(self.w.mic_combo.itemText(i), name)

    def test_selection_roundtrip_via_item_data(self) -> None:
        pairs = self.audio.input_devices()
        if not pairs:
            self.skipTest("нет устройств ввода — выбрать нечего")
        dev_id, name = pairs[0]
        self.w.mic_combo.setCurrentIndex(1)  # сигнал → контроллер
        _pump(0.05)
        self.assertEqual(self.audio.selected_input_id(), dev_id)
        self.assertEqual(self.audio.selected_input_name(), name)
        # Позиция 0 → None (системное устройство).
        self.w.mic_combo.setCurrentIndex(0)
        _pump(0.05)
        self.assertIsNone(self.audio.selected_input_id())

    def test_refresh_preserves_selection_by_id(self) -> None:
        """Перестроение списка (hotplug) не теряет выбранный микрофон."""
        pairs = self.audio.input_devices()
        if not pairs:
            self.skipTest("нет устройств ввода — выбрать нечего")
        dev_id, _name = pairs[0]
        self.assertTrue(self.audio.set_input_device(dev_id), self.audio.status)
        self.w._refresh_mic_combo()
        self.assertEqual(
            self.w.mic_combo.findData(dev_id),
            self.w.mic_combo.currentIndex(),
            "выбор по id потерян при перестроении",
        )

    def test_hotplug_resets_selection_and_ui(self) -> None:
        """Микрофон отключили: контроллер сбросил выбор, UI — на «Системный»."""
        pairs = self.audio.input_devices()
        if not pairs:
            self.skipTest("нет устройств ввода — выбрать нечего")
        self.assertTrue(self.audio.set_input_device(pairs[0][0]), self.audio.status)
        # «Отключение»: id уже не существует, приходит hotplug.
        self.audio.recorder._selected_input_id = "b'gone-mic:input'"
        self.audio.recorder._on_input_devices_changed()
        _pump(0.05)
        self.assertIsNone(self.audio.selected_input_id())
        self.assertEqual(
            self.w.mic_combo.currentIndex(),
            0,
            "UI не согласован с контроллером после отключения микрофона",
        )

    def test_combo_disabled_while_recording(self) -> None:
        self.assertTrue(self.w.mic_combo.isEnabled())
        self.audio.recorder.state = AudioRecorder.RECORDING
        self.w.render_audio()
        self.assertFalse(
            self.w.mic_combo.isEnabled(), "переключение запрещено в записи"
        )
        self.audio.recorder.state = AudioRecorder.IDLE
        self.w.render_audio()
        self.assertTrue(self.w.mic_combo.isEnabled())


if __name__ == "__main__":
    unittest.main()
