"""Интеграционные проверки окна: «Распознать» → поле ввода → закрытие.

Запуск::

    .venv/bin/python -m unittest discover -s tests -v

Используются подмены (``tests/stt_fakes.py``): без микрофона, интернета и
скачивания веса. Реальный AudioController задаётся «завершённой записью»
(в тестах допустимо трогать приватное поле; окно использует публичный API).

Проверяются:
* успешный результат — текст попадает в поле;
* пустой результат — введённый текст не стирается;
* ошибка — введённый текст не стирается, показана ошибка;
* повторный запуск заблокирован;
* непустое поле требует подтверждения замены;
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
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.main_window import MainWindow  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.session_controller import SessionController  # noqa: E402
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
    """Общее окружение: окно + реальный AudioController + FakeSttEngine."""

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
        self.audio._temp_dir = self.tmp
        self.audio._record_file = self.audio_path
        self.audio._ready_recording = self.audio_path

        self.stt = FakeSttEngine(
            self.config, result_text="hello there", delay=0.2
        )
        self.window = MainWindow(controller, self.audio, self.stt)
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


class TestRecognizeFillsInput(SttWindowTestBase):
    def test_success_fills_empty_input(self) -> None:
        """Успех: распознанный текст попадает в пустое поле ввода."""
        self.window.message_input.clear()
        self.window.start_recognition()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "hello there")
        # Ничего не отправлялось в чат автоматически.
        self.assertEqual(len(self.window.controller.history), 0)

    def test_success_when_input_empty_via_button(self) -> None:
        """Кнопка «Распознать» вызывает распознавание (путь из публичного API)."""
        self.window.message_input.clear()
        self.assertTrue(self.window.recognize_button.isEnabled())
        self.window.recognize_button.click()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "hello there")


class TestEmptyAndErrorKeepInput(SttWindowTestBase):
    def test_empty_result_keeps_existing_text(self) -> None:
        """Пустой результат НЕ стирает введённый текст."""
        self.stt.result_text = ""
        self.window.message_input.setText("my typed sentence")
        self.window.start_recognition()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "my typed sentence")

    def test_error_keeps_existing_text(self) -> None:
        """Ошибка НЕ стирает введённый текст и показывает сообщение."""
        self.stt.raise_on_transcribe = RuntimeError("boom")
        self.window.message_input.setText("my typed sentence")
        self.window.start_recognition()
        _wait_stt_idle(self.stt)
        self.assertEqual(self.window.message_input.text(), "my typed sentence")
        self.assertIn("Не удалось распознать", self.window.stt_status_label.text())


class TestBusyBlocksRestart(SttWindowTestBase):
    def test_recognize_button_disabled_while_busy(self) -> None:
        """Во время распознавания кнопка «Распознать» заблокирована."""
        self.window.start_recognition()
        self.assertTrue(self.stt.is_busy)
        self.assertFalse(self.window.recognize_button.isEnabled())

    def test_new_recording_blocked_while_busy(self) -> None:
        """Во время STT новую запись отклоняем."""
        self.window.start_recognition()
        self.assertTrue(self.stt.is_busy)
        self.assertFalse(self.window.record_button.isEnabled())
        # И защита в обработчике: toggle_record не должен начать запись.
        self.window.toggle_record()
        _pump(0.1)
        self.assertEqual(self.audio.state, AudioController.IDLE)
        _wait_stt_idle(self.stt)

    def test_recognize_button_reenabled_after_idle(self) -> None:
        """После завершения STT кнопка снова доступна (запись есть)."""
        self.window.start_recognition()
        _wait_stt_idle(self.stt)
        self.assertTrue(self.stt.state == self.stt.IDLE)
        self.assertTrue(self.window.recognize_button.isEnabled())


class TestReplacementConfirmation(SttWindowTestBase):
    def test_nonempty_input_replaced_on_yes(self) -> None:
        """Непустое поле: подтверждение «Да» → текст заменён."""
        self.window.message_input.setText("existing text")
        with mock.patch.object(
            QMessageBox, "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as m:
            self.window.start_recognition()
            _wait_stt_idle(self.stt)
            m.assert_called_once()
        self.assertEqual(self.window.message_input.text(), "hello there")

    def test_nonempty_input_kept_on_no(self) -> None:
        """Непустое поле: подтверждение «Нет» → текст сохранён."""
        self.window.message_input.setText("existing text")
        with mock.patch.object(
            QMessageBox, "question",
            return_value=QMessageBox.StandardButton.No,
        ) as m:
            self.window.start_recognition()
            _wait_stt_idle(self.stt)
            m.assert_called_once()
        self.assertEqual(self.window.message_input.text(), "existing text")

    def test_empty_input_no_confirmation(self) -> None:
        """Пустое поле: подтверждение не запрашивается, текст просто вставлен."""
        self.window.message_input.clear()
        with mock.patch.object(QMessageBox, "question") as m:
            self.window.start_recognition()
            _wait_stt_idle(self.stt)
            m.assert_not_called()
        self.assertEqual(self.window.message_input.text(), "hello there")


class TestCloseHandling(SttWindowTestBase):
    def test_close_deferred_during_stt_then_closes(self) -> None:
        """Закрытие во время STT отложено; после завершения окно закрывается."""
        self.window.start_recognition()
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
        self.stt = FakeSttEngine(self.config, result_text="x")
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
        self.audio._selected_input_id = "b'gone-mic:input'"
        self.audio._on_input_devices_changed()
        _pump(0.05)
        self.assertIsNone(self.audio.selected_input_id())
        self.assertEqual(
            self.w.mic_combo.currentIndex(),
            0,
            "UI не согласован с контроллером после отключения микрофона",
        )

    def test_combo_disabled_while_recording(self) -> None:
        self.assertTrue(self.w.mic_combo.isEnabled())
        self.audio.state = self.audio.RECORDING
        self.w.render_audio()
        self.assertFalse(
            self.w.mic_combo.isEnabled(), "переключение запрещено в записи"
        )
        self.audio.state = self.audio.IDLE
        self.w.render_audio()
        self.assertTrue(self.w.mic_combo.isEnabled())


if __name__ == "__main__":
    unittest.main()
