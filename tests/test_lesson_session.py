"""Подключение урока к разговору: окно + контроллер + системный контекст.

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Клиент Ollama подменён (mock ``send_chat``): без сети проверяются
сборка системного контекста из урока (level/topics/phrases/words +
правила преподавателя), его передача в Ollama (поле ``system``) и то,
что в чате выводятся ТОЛЬКО реплики пользователя и преподавателя.
«Новое занятие»: сброс непустой истории — только с подтверждением
(отмена — ничего не меняется); смена урока = новое занятие с
подтверждением; действия заблокированы, пока идут операции.
Сбой загрузки урока — понятная ошибка, отправка недоступна.
"""

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.lesson import (  # noqa: E402
    Lesson,
    LessonError,
    default_lesson_path,
    load_lesson,
)
from companion.main_window import MainWindow  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.session_controller import SessionController  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402
from companion.teacher_context import build_teacher_context  # noqa: E402
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


class LessonSessionBase(unittest.TestCase):
    """Окно + контроллер; Ollama подменяется в каждом тесте (без сети)."""

    def setUp(self) -> None:
        _app()
        self.config = AppConfig()
        self.controller = SessionController(OllamaClient(self.config))
        self.audio = AudioController()
        # Озвучивание реплик не нужно — подменяем (тесты без TTS/аудио).
        self.audio.speak_reply = mock.MagicMock(name="speak_reply")
        self.stt = SttEngine(FakeBackend(result_text="hello there"))
        self.window = MainWindow(self.controller, self.audio, self.stt)
        self.window.show()
        _pump(0.1)

    def tearDown(self) -> None:
        try:
            self.stt.shutdown()
            self.audio.shutdown()
            _pump(0.1)
        finally:
            del self.window

    def _allow_send(self):
        """Делаем отправку доступной: модель выбрана, клиент подменён."""
        send_mock = mock.MagicMock(name="send_chat")
        self.controller.client.send_chat = send_mock
        self.controller.selected_model = "m"
        return send_mock

    def _exchange(self, user_text: str = "Hello!", reply_text: str = "Hi! How are you?") -> None:
        """Один завершённый обмен — история чата непуста."""
        self._allow_send()
        self.window.message_input.setText(user_text)
        self.window.send_message()
        self.controller.client.chat_completed.emit(reply_text)

    def _second_lesson(self) -> Lesson:
        lesson = Lesson(
            id="hw13",
            level="Advanced",
            name="Урок 13",
            topics=["Topics"],
            phrases=(),
            words=(),
        )
        self.window.lesson_combo.addItem(f"{lesson.name} — {lesson.level}", lesson)
        return lesson


class TestTeacherContext(unittest.TestCase):
    """Сборка системного контекста: level/topics/phrases/words + правила."""

    def test_context_contains_lesson_parts(self) -> None:
        lesson = load_lesson(default_lesson_path())
        context = build_teacher_context(lesson)
        self.assertIn("Elementary", context)
        self.assertIn("Урок 12", context)
        self.assertIn(lesson.topics[0], context)
        self.assertIn("Can you pass me the salt, please?", context)
        self.assertIn("reservation", context)

    def test_context_contains_teacher_rules(self) -> None:
        lesson = load_lesson(default_lesson_path())
        context = build_teacher_context(lesson)
        self.assertIn("1-3 sentences", context)
        self.assertIn("one question", context)
        self.assertIn("ONLY in English", context)
        self.assertIn("Russian", context)
        self.assertIn("clarifying", context)
        self.assertIn("Do not correct small mistakes", context)

    def test_empty_phrases_and_words_ok(self) -> None:
        lesson = Lesson(
            id="hw-x",
            level="Advanced",
            name="Урок 1",
            topics=["Тема"],
            phrases=(),
            words=(),
        )
        context = build_teacher_context(lesson)
        self.assertIn("Advanced", context)
        self.assertIn("Тема", context)
        self.assertNotIn("Phrases from the lesson", context)
        self.assertNotIn("Vocabulary from the lesson", context)


class TestLessonUi(LessonSessionBase):
    """Выбор урока в окне: name + level, урок выбран в контроллере."""

    def test_combo_shows_name_and_level(self) -> None:
        combo = self.window.lesson_combo
        self.assertEqual(combo.count(), 1)
        self.assertEqual(combo.itemText(0), "Урок 12 — Elementary")
        lesson = combo.currentData()
        self.assertIsInstance(lesson, Lesson)
        self.assertIs(self.controller.lesson, lesson)
        self.assertTrue(self.window.new_session_button.isEnabled())

    def test_lesson_load_error_shown(self) -> None:
        """Урок не загрузился: понятная ошибка, отправка недоступна."""
        controller = SessionController(OllamaClient(self.config))
        controller.selected_model = "m"
        error = LessonError("Урок некорректен: id: обязательное непустое строковое поле")
        with mock.patch("companion.main_window.load_lesson", side_effect=error):
            window = MainWindow(controller, self.audio, self.stt)
            _pump(0.1)
        try:
            self.assertEqual(window.lesson_combo.count(), 0)
            self.assertIn(
                "Не удалось загрузить урок",
                window.lesson_combo.placeholderText(),
            )
            self.assertIn("Урок не загрузился", window.status_label.text())
            self.assertIn(str(error), window.status_label.text())
            self.assertFalse(window.new_session_button.isEnabled())
            self.assertFalse(controller.can_send, "без урока отправка запрещена")
        finally:
            del window


class TestSystemContextPassing(LessonSessionBase):
    """Контекст уходит в Ollama вместе с историей; чат — только реплики."""

    def test_send_passes_system_context(self) -> None:
        send_mock = self._allow_send()
        self.window.message_input.setText("What are you having?")
        self.window.send_message()
        send_mock.assert_called_once()
        system = send_mock.call_args.kwargs["system"]
        self.assertIn("Elementary", system)
        self.assertIn("reservation", system)
        self.assertIn("ONLY in English", system)
        # В чате — только реплика пользователя; системный контекст не виден.
        chat = self.window.history.toPlainText()
        self.assertIn("You: What are you having?", chat)
        self.assertNotIn("Student level", chat)
        self.assertNotIn("ONLY in English", chat)

    def test_reply_shown_and_history_grows(self) -> None:
        self._exchange()
        chat = self.window.history.toPlainText()
        self.assertIn("You: Hello!", chat)
        self.assertIn("Teacher · m: Hi! How are you?", chat)
        self.assertEqual(len(self.controller.history), 2)


class TestNewSession(LessonSessionBase):
    """«Новое занятие»: сброс с подтверждением; отмена — ничего не меняется."""

    def test_reset_confirmed_clears_history(self) -> None:
        self._exchange()
        self.assertTrue(self.controller.history)
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.Yes)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.on_new_session()
        question.assert_called_once()
        self.assertEqual(self.controller.history, [])
        self.assertEqual(self.window.history.toPlainText(), "")
        self.assertIn("Новое занятие", self.window.status_label.text())
        # После сброса отправка снова работает.
        send_mock = self._allow_send()
        self.window.message_input.setText("Hi again")
        self.window.send_message()
        send_mock.assert_called_once()

    def test_reset_cancelled_keeps_history(self) -> None:
        self._exchange()
        chat_before = self.window.history.toPlainText()
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.No)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.on_new_session()
        question.assert_called_once()
        self.assertEqual(len(self.controller.history), 2)
        self.assertEqual(self.window.history.toPlainText(), chat_before)

    def test_reset_empty_history_no_confirmation(self) -> None:
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.Yes)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.on_new_session()
        question.assert_not_called()
        self.assertEqual(self.controller.history, [])
        self.assertIn("Новое занятие", self.window.status_label.text())

    def test_lesson_change_starts_new_session(self) -> None:
        self._exchange()
        lesson = self._second_lesson()
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.Yes)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.lesson_combo.setCurrentIndex(1)
        question.assert_called_once()
        self.assertIs(self.controller.lesson, lesson)
        self.assertEqual(self.controller.history, [])
        self.assertEqual(self.window.history.toPlainText(), "")

    def test_lesson_change_cancelled_keeps_everything(self) -> None:
        self._exchange()
        lesson = self._second_lesson()
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.No)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.lesson_combo.setCurrentIndex(1)
        self.assertIsNot(self.controller.lesson, lesson)
        self.assertEqual(len(self.controller.history), 2)

    def test_actions_blocked_while_operation_in_flight(self) -> None:
        """Пока ожидается ответ — «Новое занятие» и смена урока заблокированы."""
        self._allow_send()
        self.window.message_input.setText("Hello!")
        self.window.send_message()
        self.assertTrue(self.controller.busy)
        self.assertFalse(self.window.new_session_button.isEnabled())
        self.assertFalse(self.window.lesson_combo.isEnabled())
        # После завершения ответа действия снова доступны.
        self.controller.client.chat_completed.emit("Hi!")
        _pump(0.1)
        self.assertFalse(self.controller.busy)
        self.assertTrue(self.window.new_session_button.isEnabled())
        self.assertTrue(self.window.lesson_combo.isEnabled())


if __name__ == "__main__":
    unittest.main()
