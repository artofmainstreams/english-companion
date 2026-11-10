"""Подключение урока к разговору: окно + контроллер + системный контекст.

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Клиент Ollama подменяется (mock ``send_chat`` / подмена сетевого
слоя): без сети проверяются сборка системного контекста из урока
(level/topics/phrases/words + правила преподавателя, «только
английский», без эмодзи/шуток) и учебное поведение: фразы/слова —
материал ученика, роли и место до ситуации, факты ситуации
задаёт преподаватель, лестница помощи при «I don't know», разные
способы практики — плюс ФАКТИЧЕСКИЙ payload запроса Ollama
(системное сообщение первым, урок/уровень/English-only в нём, история
после него) и то, что в чате выводятся ТОЛЬКО реплики пользователя и
преподавателя. Все уроки из папки ``lessons/`` попадают в комбо
(«name — level», натуральный порядок), выбирается первый;
некорректный файл — понятная ошибка в статусе, остальные уроки
работают; урок не загрузился ни один — отправка недоступна.
«Новое занятие»: сброс непустой истории — только с
подтверждением (отмена — ничего не меняется), контекст урока
сохраняется; смена урока = новое занятие с подтверждением; действия
заблокированы, пока идут операции.
"""

import json
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtNetwork import QNetworkReply  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.lesson import (  # noqa: E402
    Lesson,
    LessonError,
    default_lesson_path,
    load_lesson,
    load_lessons,
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

    def _combo_index_of(self, lesson: Lesson) -> int:
        """Индекс урока в комбо по содержимому (позиция заранее не известна)."""
        for i in range(self.window.lesson_combo.count()):
            if self.window.lesson_combo.itemData(i) == lesson:
                return i
        raise AssertionError("урок не найден в комбо")

    def _select_lesson(self, lesson: Lesson) -> None:
        """Выбирает урок в комбо (реальный путь окна: ``_on_lesson_changed``).

        История в момент выбора пуста — подтверждения нет.
        """
        self.window.lesson_combo.setCurrentIndex(self._combo_index_of(lesson))


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
        self.assertIn("calm clarifying question", context)
        self.assertIn("Do not correct small mistakes", context)
        self.assertIn("no emojis", context)
        self.assertIn("no jokes", context)
        self.assertIn("plain text", context)

    def test_context_marks_material_as_students(self) -> None:
        """Фразы/слова — материал УЧЕНИКА; не сценарий преподавателя."""
        lesson = load_lesson(default_lesson_path())
        context = build_teacher_context(lesson)
        self.assertIn("the student's practice material", context)
        self.assertIn("Do not read the list as your own script", context)
        self.assertIn(
            "the current situation decides which expressions are natural",
            context,
        )
        self.assertIn("natural for your role", context)

    def test_context_roles_situation_and_questions(self) -> None:
        """Роли и место; преподаватель задаёт факты; вопросы — только
        на сообщённые сведения, предпочтения или действие ученика."""
        lesson = load_lesson(default_lesson_path())
        context = build_teacher_context(lesson)
        self.assertIn("briefly set the roles and the place", context)
        self.assertIn("Keep the roles until the situation clearly ends", context)
        self.assertIn("You provide the invented facts of the situation", context)
        self.assertIn("prices, breakfast time, room availability", context)
        self.assertIn("their preferences, or an action they can take", context)

    def test_context_i_dont_know_help_ladder(self) -> None:
        """«I don't know»: информация → подсказка → начало фразы → пример
        → ученик пробует сам."""
        lesson = load_lesson(default_lesson_path())
        context = build_teacher_context(lesson)
        self.assertIn("I don't know", context)
        self.assertIn("give any missing facts", context)
        self.assertIn("short English hint", context)
        self.assertIn("phrase starter, then a full example", context)
        self.assertIn("let the student try again on their own", context)

    def test_context_varies_practice(self) -> None:
        """Практика разными способами: вопросы, пожелания, просьбы, ответы."""
        lesson = load_lesson(default_lesson_path())
        context = build_teacher_context(lesson)
        self.assertIn("Vary the practice", context)
        self.assertIn("expresses wishes, makes requests", context)
        self.assertIn("asks for clarification", context)

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
    """Выбор урока в окне: все уроки из lessons/, выбран первый."""

    def test_combo_shows_all_lessons_first_selected(self) -> None:
        """Все уроки из lessons/ в комбо («name — level»); выбран первый."""
        lessons, errors = load_lessons()
        self.assertEqual(errors, [], "уроки в lessons/ должны быть корректны")
        combo = self.window.lesson_combo
        self.assertEqual(combo.count(), len(lessons))
        for i, lesson in enumerate(lessons):
            self.assertEqual(combo.itemText(i), f"{lesson.name} — {lesson.level}")
            self.assertEqual(combo.itemData(i), lesson)
        self.assertEqual(combo.currentData(), lessons[0])
        self.assertEqual(self.controller.lesson, lessons[0])
        self.assertTrue(self.window.new_session_button.isEnabled())

    def test_lesson_load_error_shown(self) -> None:
        """Урок не загрузился ни один: понятная ошибка, отправка недоступна."""
        controller = SessionController(OllamaClient(self.config))
        controller.selected_model = "m"
        error = LessonError("Папка уроков не найдена: /tmp/lessons")
        with mock.patch("companion.main_window.load_lessons", side_effect=error):
            window = MainWindow(controller, self.audio, self.stt)
            _pump(0.1)
        try:
            self.assertEqual(window.lesson_combo.count(), 0)
            self.assertIn(
                "Не удалось загрузить уроки",
                window.lesson_combo.placeholderText(),
            )
            self.assertIn("Не удалось загрузить уроки", window.status_label.text())
            self.assertIn(str(error), window.status_label.text())
            self.assertFalse(window.new_session_button.isEnabled())
            self.assertFalse(controller.can_send, "без урока отправка запрещена")
        finally:
            del window

    def test_partial_errors_keep_good_lessons(self) -> None:
        """Некорректный файл — в статусе; остальные уроки в комбо и доступны."""
        lessons, _ = load_lessons()
        self.assertTrue(lessons, "в lessons/ должен быть хотя бы один урок")
        controller = SessionController(OllamaClient(self.config))
        controller.selected_model = "m"
        with mock.patch(
            "companion.main_window.load_lessons",
            return_value=(lessons[:1], ["broken.json: level: одно из [...]"]),
        ):
            window = MainWindow(controller, self.audio, self.stt)
            _pump(0.1)
        try:
            self.assertEqual(window.lesson_combo.count(), 1)
            self.assertEqual(window.lesson_combo.currentData(), lessons[0])
            self.assertIn("Не все уроки загрузились", window.status_label.text())
            self.assertIn("broken.json", window.status_label.text())
            self.assertTrue(controller.can_send, "работоспособные уроки доступны")
        finally:
            del window


class TestSystemContextPassing(LessonSessionBase):
    """Контекст уходит в Ollama вместе с историей; чат — только реплики."""

    def test_send_passes_system_context(self) -> None:
        # Урок 12 выбран в комбо явно: его содержимое проверяется ниже.
        self._select_lesson(load_lesson(default_lesson_path()))
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


class _FakeReply(QObject):
    """Завершённый сетевой ответ с телом ответа Ollama."""

    finished = Signal()

    def __init__(self, body: bytes) -> None:
        super().__init__()
        self._body = body

    def error(self):
        return QNetworkReply.NetworkError.NoError

    def errorString(self) -> str:
        return ""

    def readAll(self) -> bytes:
        return self._body

    def deleteLater(self) -> None:
        pass


class _FakeNetwork:
    """Подмена QNetworkAccessManager: ловит payload и отвечает готовым
    телом Ollama (через сигнал ``finished`` в следующем прогоне цикла)."""

    def __init__(self, answer: str) -> None:
        self.last_payload: dict | None = None
        self._body = json.dumps(
            {"message": {"role": "assistant", "content": answer}, "done": True}
        ).encode("utf-8")

    def post(self, request, data: bytes):
        self.last_payload = json.loads(bytes(data).decode("utf-8"))
        reply = _FakeReply(self._body)
        QTimer.singleShot(0, reply.finished.emit)
        return reply


class TestOllamaPayload(LessonSessionBase):
    """Фактический payload запроса Ollama (подмена на сетевом слое)."""

    def _wire_network(self, answer: str = "Sure! What would you like?") -> _FakeNetwork:
        net = _FakeNetwork(answer)
        self.controller.client.network = net
        self.controller.selected_model = "m"
        return net

    def test_system_message_first_with_lesson_and_english_only(self) -> None:
        net = self._wire_network()
        # Урок 12 выбран в комбо явно: его содержимое проверяется ниже.
        self._select_lesson(load_lesson(default_lesson_path()))
        self.window.message_input.setText("I'd like a table for two.")
        self.window.send_message()
        _pump(0.1)
        self.assertIsNotNone(net.last_payload)
        messages = net.last_payload["messages"]
        # Системное сообщение присутствует ПЕРВЫМ.
        self.assertEqual(messages[0]["role"], "system")
        system_text = messages[0]["content"]
        self.assertIn("Урок 12", system_text)
        self.assertIn("Elementary", system_text)
        self.assertIn("ONLY in English", system_text)
        self.assertIn("no emojis", system_text)
        self.assertIn("no jokes", system_text)
        self.assertIn("1-3 sentences", system_text)
        # Правила учебного поведения дошли до Ollama (подмена сети).
        self.assertIn("the student's practice material", system_text)
        self.assertIn("set the roles and the place", system_text)
        self.assertIn("You provide the invented facts", system_text)
        # После системного сообщения — текущая история (на момент запроса
        # это реплика пользователя; ответ придёт позже — см. второй ход).
        self.assertEqual([m["role"] for m in messages[1:]], ["user"])
        self.assertEqual(messages[1]["content"], "I'd like a table for two.")
        # Ответ получен и история выросла (для следующего хода).
        self.assertEqual(len(self.controller.history), 2)
        # Лишнего верхнеуровневого "system" нет — всё в messages.
        self.assertNotIn("system", net.last_payload)

    def test_second_turn_full_history_after_system(self) -> None:
        net = self._wire_network("Hi! How are you?")
        self.window.message_input.setText("Hello!")
        self.window.send_message()
        _pump(0.1)
        self.window.message_input.setText("I'm hungry.")
        self.window.send_message()
        _pump(0.1)
        messages = net.last_payload["messages"]
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(messages[2]["content"], "Hi! How are you?")
        self.assertEqual(messages[3]["content"], "I'm hungry.")

    def test_new_session_resets_history_keeps_lesson_context(self) -> None:
        net = self._wire_network("Hi! How are you?")
        # Урок 12 выбран в комбо явно: его содержимое проверяется ниже.
        self._select_lesson(load_lesson(default_lesson_path()))
        self.window.message_input.setText("Hello!")
        self.window.send_message()
        _pump(0.1)
        self.assertEqual(len(self.controller.history), 2)
        # Непустая история → подтверждение сброса.
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.Yes)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.on_new_session()
        question.assert_called_once()
        self.assertEqual(self.controller.history, [])
        # Новая отправка: контекст урока сохранён, история чистая.
        self.window.message_input.setText("I'd like a table for two.")
        self.window.send_message()
        _pump(0.1)
        messages = net.last_payload["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Урок 12", messages[0]["content"])
        self.assertIn("Elementary", messages[0]["content"])
        self.assertIn("ONLY in English", messages[0]["content"])
        self.assertEqual([m["role"] for m in messages[1:]], ["user"])
        self.assertEqual(messages[1]["content"], "I'd like a table for two.")


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
            self.window.lesson_combo.setCurrentIndex(self._combo_index_of(lesson))
        question.assert_called_once()
        self.assertIs(self.controller.lesson, lesson)
        self.assertEqual(self.controller.history, [])
        self.assertEqual(self.window.history.toPlainText(), "")

    def test_lesson_change_cancelled_keeps_everything(self) -> None:
        self._exchange()
        lesson = self._second_lesson()
        question = mock.MagicMock(return_value=QMessageBox.StandardButton.No)
        with mock.patch("companion.main_window.QMessageBox.question", question):
            self.window.lesson_combo.setCurrentIndex(self._combo_index_of(lesson))
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
