"""Промпт преподавателя из prompts/teacher.txt + режимы разговора (сцена).

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Подмены (без сети и TTS):
* Ollama — mock ``send_chat`` (payload — в ``call_args``);
* TTS/запись — mock ``speak_reply``/``record`` (оконные тесты);
* файл промпта — временный (путь передаётся в ``SessionController``),
  при этом отдельные проверки идут по РЕАЛЬНОМУ ``prompts/teacher.txt``.

Проверяются:
* файл: путь относительно приложения (запуск из другой рабочей папки),
  ошибки «отсутствует/пуст/не читается» — с путём;
* правка файла подхватывается со СЛЕДУЮЩЕГО запроса (тот же
  контроллер); сбой файла блокирует отправку с понятной ошибкой и
  повтор после исправления работает (реплика не дублируется);
* режим разговора: начальная свободная беседа; маркер сцены ставит
  состояние в контроллер; следующие запросы несут «сцена активна» +
  роли; выход из сцены — свободная беседа, история и учебный материал
  сохранены; повторное начало сцены; отсутствие маркера — состояние
  не меняется; «Новое занятие» сбрасывает сцену;
* маркер убирается из ответа: в чат и озвучивание не попадает;
* окно: ошибка файла — понятный статус с путём, текст ввода сохранён.
"""

import faulthandler
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

faulthandler.enable()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.audio_controller import AudioController  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.lesson import Lesson  # noqa: E402
from companion.main_window import MainWindow  # noqa: E402
from companion.message import Message  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.session_controller import SessionController  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402
from companion.teacher_context import (  # noqa: E402
    Scene,
    TeacherPromptError,
    build_teacher_context,
    load_teacher_prompt,
    parse_scene_marker,
    scene_from_marker,
    teacher_prompt_path,
)
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


def _lesson() -> Lesson:
    return Lesson(
        id="hw-t",
        level="Elementary",
        name="Урок 9",
        topics=["Booking a hotel room"],
        phrases=("Can I have a quiet room, please?",),
        words=("reservation",),
    )


def _last_system(send_mock: mock.MagicMock) -> str:
    return send_mock.call_args.kwargs["system"]


class TestTeacherPromptFile(unittest.TestCase):
    """Файл промпта: путь, загрузка, ошибки, {level}."""

    def test_default_file_contains_permanent_instructions(self) -> None:
        text = load_teacher_prompt()
        # Текущие правила — без переформулировок (включая English-only).
        self.assertIn("ONLY in English", text)
        self.assertIn("no emojis", text)
        self.assertIn("no jokes", text)
        self.assertIn("1-3 sentences", text)
        self.assertIn("I don't know", text)
        # Режимы разговора и маркер сцены.
        self.assertIn("[SCENE:free]", text)
        self.assertIn("propose it first", text)
        self.assertIn("genuinely struggles", text)
        self.assertIn("Thanks, that's all", text)

    def test_default_path_relative_to_app(self) -> None:
        """Путь — относительно приложения; чужая рабочая папка не ломает
        загрузку."""
        expected = (
            Path(__file__).resolve().parent.parent / "prompts" / "teacher.txt"
        )
        self.assertEqual(teacher_prompt_path(), expected)
        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="english-companion-cwd-")
        try:
            os.chdir(tmp)
            self.assertEqual(teacher_prompt_path(), expected)
            text = load_teacher_prompt()
            self.assertIn("ONLY in English", text)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_file_error_with_path(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "no-such-teacher.txt")
        with self.assertRaises(TeacherPromptError) as ctx:
            load_teacher_prompt(missing)
        self.assertIn(missing, str(ctx.exception))
        self.assertIn("не найден", str(ctx.exception))

    def test_empty_and_whitespace_file_error(self) -> None:
        tmp = tempfile.mkdtemp(prefix="english-companion-teacher-empty-")
        try:
            path = os.path.join(tmp, "teacher.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("")
            with self.assertRaises(TeacherPromptError) as ctx:
                load_teacher_prompt(path)
            self.assertIn(path, str(ctx.exception))
            self.assertIn("пуст", str(ctx.exception))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(" \n\t \n")
            with self.assertRaises(TeacherPromptError):
                load_teacher_prompt(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(
        os.geteuid == 0, "root читает файлы без прав — тест неприменим"
    )
    def test_unreadable_file_error_with_path(self) -> None:
        tmp = tempfile.mkdtemp(prefix="english-companion-teacher-unreadable-")
        try:
            path = os.path.join(tmp, "teacher.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("secret")
            os.chmod(path, 0)
            try:
                with self.assertRaises(TeacherPromptError) as ctx:
                    load_teacher_prompt(path)
                self.assertIn(path, str(ctx.exception))
                self.assertIn("не читается", str(ctx.exception))
            finally:
                os.chmod(path, 0o644)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_level_placeholder_substituted(self) -> None:
        tmp = tempfile.mkdtemp(prefix="english-companion-teacher-level-")
        try:
            path = os.path.join(tmp, "teacher.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Speak at the {level} level.\n")
            lesson = Lesson(
                id="hw-t",
                level="Advanced",
                name="Урок 9",
                topics=["Booking a hotel room"],
                phrases=("Can I have a quiet room, please?",),
                words=("reservation",),
            )
            context = build_teacher_context(lesson, prompt_path=path)
            self.assertIn("Speak at the Advanced level", context)
            self.assertNotIn("{level}", context)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ControllerBase(unittest.TestCase):
    """Контроллер с mock-клиентом и временным файлом промпта."""

    def setUp(self) -> None:
        _app()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-teacher-c-")
        self.prompt_file = os.path.join(self.tmp, "teacher.txt")
        self._write_prompt("ALPHA RULE — teach well.")
        self.controller = SessionController(
            OllamaClient(AppConfig()), prompt_path=self.prompt_file
        )
        self.send_mock = mock.MagicMock(name="send_chat")
        self.controller.client.send_chat = self.send_mock
        self.controller.selected_model = "m"
        self.controller.set_lesson(_lesson())
        _pump(0.02)

    def tearDown(self) -> None:
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_prompt(self, text: str) -> None:
        with open(self.prompt_file, "w", encoding="utf-8") as fh:
            fh.write(text)


class TestEditPickedUp(ControllerBase):
    """Правка teacher.txt подхватывается со следующего запроса."""

    def test_edit_applies_from_next_request(self) -> None:
        self.assertTrue(self.controller.send_message("hi"))
        self.assertIn("ALPHA RULE", _last_system(self.send_mock))
        # Первый запрос завершён (иначе следующий заблокирован — busy).
        self.controller.client.chat_completed.emit("Hello!")
        _pump(0.02)

        # Правка файла — тот же контроллер, без перезапуска.
        self._write_prompt("BETA RULE — teach better.")
        self.assertTrue(self.controller.send_message("hello"))
        system = _last_system(self.send_mock)
        self.assertIn("BETA RULE", system)
        self.assertNotIn("ALPHA RULE", system)
        # Урок и режим — на месте.
        self.assertIn("Урок 9", system)
        self.assertIn("free talk", system)

    def test_deleted_file_blocks_send_with_path_and_retry_works(self) -> None:
        os.remove(self.prompt_file)
        self.assertFalse(self.controller.send_message("hi"))
        self.send_mock.assert_not_called()
        self.assertIn(self.prompt_file, self.controller.status)
        self.assertIn("Повторите отправку", self.controller.status)
        # Реплика в чат/историю НЕ попала — повтор не задублирует.
        self.assertEqual(self.controller.history, [])
        self.assertIsNone(self.controller.pending_message)

        # Исправление — повтор работает (один запрос).
        self._write_prompt("GAMMA RULE — teach well.")
        self.assertTrue(self.controller.send_message("hi"))
        self.send_mock.assert_called_once()
        self.assertIn("GAMMA RULE", _last_system(self.send_mock))

    def test_empty_file_blocks_send(self) -> None:
        self._write_prompt("   ")
        self.assertFalse(self.controller.send_message("hi"))
        self.send_mock.assert_not_called()
        self.assertIn("пуст", self.controller.status)
        self.assertIn(self.prompt_file, self.controller.status)


class TestSceneMode(ControllerBase):
    """Режим разговора: сцена → свободная беседа → сцена снова."""

    def test_initial_request_is_free_talk(self) -> None:
        self.assertTrue(self.controller.send_message("hi"))
        system = _last_system(self.send_mock)
        self.assertIn("free talk (no role-play scene is active)", system)
        self.assertNotIn("role-play scene is ACTIVE", system)

    def test_scene_marker_sets_scene_and_cleans_reply(self) -> None:
        added: list[Message] = []
        self.controller.message_added.connect(added.append)
        reply = (
            "You are a guest, I am the hotel receptionist.\n"
            "[SCENE:hotel check-in|you are the hotel receptionist, "
            "the student is the guest]"
        )
        self.assertTrue(self.controller.send_message("I'm looking for a room"))
        self.controller.client.chat_completed.emit(reply)
        _pump(0.02)

        self.assertEqual(
            self.controller.scene,
            Scene(
                name="hotel check-in",
                roles="you are the hotel receptionist, the student is the guest",
            ),
        )
        # Маркер убран: и в истории, и в сигнале окна.
        answer = self.controller.history[-1]
        self.assertEqual(answer.role, "assistant")
        self.assertNotIn("[SCENE:", answer.content)
        self.assertEqual(answer.content, "You are a guest, I am the hotel receptionist.")
        self.assertEqual(
            added[-1].content, "You are a guest, I am the hotel receptionist."
        )

    def test_next_request_carries_scene_and_roles(self) -> None:
        reply = (
            "Welcome!\n"
            "[SCENE:hotel check-in|you are the hotel receptionist, "
            "the student is the guest]"
        )
        self.assertTrue(self.controller.send_message("I'm looking for a room"))
        self.controller.client.chat_completed.emit(reply)
        _pump(0.02)
        self.assertTrue(self.controller.send_message("How much is the room?"))
        system = _last_system(self.send_mock)
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn("Scene: hotel check-in", system)
        self.assertIn("you are the hotel receptionist, the student is the guest",
                      system)

    def test_exit_to_free_talk_keeps_history_and_material(self) -> None:
        # Сцена активна.
        reply = (
            "You are a guest, I am the receptionist.\n"
            "[SCENE:hotel check-in|you are the receptionist, the student is the guest]"
        )
        self.assertTrue(self.controller.send_message("I'm looking for a room"))
        self.controller.client.chat_completed.emit(reply)
        _pump(0.02)
        self.assertIsNotNone(self.controller.scene)

        # Ученик завершает сцену.
        exit_reply = "Sure — tell me about your trip.\n[SCENE:free]"
        self.assertTrue(
            self.controller.send_message("Thanks, that's all. Let's talk about my trip.")
        )
        self.controller.client.chat_completed.emit(exit_reply)
        _pump(0.02)

        self.assertIsNone(self.controller.scene)
        # История сохранена целиком (4 реплики), маркер нигде нет.
        self.assertEqual(len(self.controller.history), 4)
        for message in self.controller.history:
            self.assertNotIn("[SCENE:", message.content)
        self.assertNotIn(
            "[SCENE:",
            self.controller.history[-1].content,
        )
        # Следующий запрос: свободная беседа, без сцены/ролей;
        # учебный материал — на месте.
        self.assertTrue(self.controller.send_message("I went to the mountains."))
        system = _last_system(self.send_mock)
        self.assertIn("free talk (no role-play scene is active)", system)
        self.assertNotIn("role-play scene is ACTIVE", system)
        self.assertNotIn("hotel check-in", system)
        self.assertIn("Урок 9", system)
        self.assertIn("reservation", system)

    def test_restart_scene_after_free_talk(self) -> None:
        # Свободная беседа: ученик снова просит сцену.
        self.assertTrue(self.controller.send_message("Can we play the hotel scene again?"))
        restart_reply = (
            "Of course — you are a guest, I am the receptionist.\n"
            "[SCENE:hotel check-in|you are the receptionist, the student is the guest]"
        )
        self.controller.client.chat_completed.emit(restart_reply)
        _pump(0.02)
        self.assertIsNotNone(self.controller.scene)
        self.assertTrue(self.controller.send_message("Hello!"))
        system = _last_system(self.send_mock)
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn("Scene: hotel check-in", system)

    def test_missing_marker_keeps_state(self) -> None:
        # Свободная беседа + ответ без маркера — состояние не меняется.
        self.assertTrue(self.controller.send_message("hi"))
        self.controller.client.chat_completed.emit("Hello! How can I help?")
        _pump(0.02)
        self.assertIsNone(self.controller.scene)

        # Сцена активна + ответ без маркера — сцена не сбрасывается.
        self.assertTrue(self.controller.send_message("I'm looking for a room"))
        self.controller.client.chat_completed.emit(
            "Welcome!\n[SCENE:hotel check-in|you are the receptionist, the student is the guest]"
        )
        _pump(0.02)
        scene = self.controller.scene
        self.assertTrue(self.controller.send_message("Breakfast?"))
        self.controller.client.chat_completed.emit("Breakfast is from 7 to 10.")
        _pump(0.02)
        self.assertEqual(self.controller.scene, scene)

    def test_reset_history_clears_scene(self) -> None:
        self.assertTrue(self.controller.send_message("I'm looking for a room"))
        self.controller.client.chat_completed.emit(
            "Welcome!\n[SCENE:hotel check-in|you are the receptionist, the student is the guest]"
        )
        _pump(0.02)
        self.assertIsNotNone(self.controller.scene)
        self.controller.reset_history()
        self.assertIsNone(self.controller.scene)
        self.assertTrue(self.controller.send_message("hi"))
        self.assertIn("free talk", _last_system(self.send_mock))


class TestSceneRewrite(ControllerBase):
    """Перезапись реплики: сцена восстанавливается до исходного обмена."""

    def _complete(self, user_text: str, reply_text: str) -> None:
        self.assertTrue(self.controller.send_message(user_text))
        self.controller.client.chat_completed.emit(reply_text)
        _pump(0.02)

    def test_rewrite_uses_free_after_scene_reply(self) -> None:
        """Свободная беседа → сцена: замена несёт СВОБОДНУЮ беседу."""
        self._complete(
            "I'm looking for a room",
            "You are a guest, I am the receptionist.\n"
            "[SCENE:hotel check-in|you are the receptionist, the student is the guest]",
        )
        self.assertIsNotNone(self.controller.scene, "ответ установил сцену")

        # Удаляемый ответ мог сменить сцену — используем сцену ДО обмена.
        self.assertTrue(self.controller.replace_last_user_message("I need a room"))
        system = _last_system(self.send_mock)
        self.assertIn("free talk (no role-play scene is active)", system)
        self.assertNotIn("role-play scene is ACTIVE", system)
        # Сцена контроллера восстановлена до обмена (удалённый ответ не в силе).
        self.assertIsNone(self.controller.scene)
        # Ответ без маркера сцену не меняет.
        self.controller.client.chat_completed.emit("Sure. How many nights?")
        _pump(0.02)
        self.assertIsNone(self.controller.scene)

    def test_rewrite_uses_scene_after_free_reply(self) -> None:
        """Сцена → свободная беседа: замена несёт АКТИВНУЮ сцену."""
        self._complete(
            "I'm looking for a room",
            "You are a guest, I am the receptionist.\n"
            "[SCENE:hotel check-in|you are the receptionist, the student is the guest]",
        )
        self.assertIsNotNone(self.controller.scene)
        # Второй ход в сцене; ответ завершает её (свободная беседа).
        self._complete(
            "What is the price?",
            "It is fifty dollars. Thanks, that's all.\n[SCENE:free]",
        )
        self.assertIsNone(self.controller.scene, "ответ завершил сцену")

        # Перезапись второго хода: сцена ДО обмена была АКТИВНА.
        self.assertTrue(self.controller.replace_last_user_message("How much is it?"))
        system = _last_system(self.send_mock)
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn("Scene: hotel check-in", system)
        self.assertNotIn("free talk (no role-play scene is active)", system)
        # Сцена контроллера восстановлена до обмена.
        self.assertIsNotNone(self.controller.scene)
        self.assertEqual(self.controller.scene.name, "hotel check-in")

    def test_rewrite_prompt_error_preserves_history_and_scene(self) -> None:
        """Сбой чтения промпта: история и сцена сохраняются, запрос не ушёл."""
        self._complete(
            "I'm looking for a room",
            "You are a guest.\n[SCENE:hotel check-in|you are the receptionist]",
        )
        self.assertIsNotNone(self.controller.scene)
        history_before = list(self.controller.history)
        scene_before = self.controller.scene

        os.remove(self.prompt_file)
        self.assertFalse(self.controller.replace_last_user_message("I need a room"))
        self.assertEqual(self.send_mock.call_count, 1, "запросов не добавилось")
        # История и сцена не тронуты.
        self.assertEqual(self.controller.history, history_before)
        self.assertEqual(self.controller.scene, scene_before)

    def test_retry_after_failed_rewrite_uses_restored_scene(self) -> None:
        """Замена прошла, запрос упал — повтор несёт восстановленную сцену."""
        self._complete(
            "I'm looking for a room",
            "You are a guest.\n[SCENE:hotel check-in|you are the receptionist]",
        )
        self.assertIsNotNone(self.controller.scene)
        # Перезапись: сцена ДО обмена = свободная беседа.
        self.assertTrue(self.controller.replace_last_user_message("I need a room"))
        self.assertIsNone(self.controller.scene)
        # Новый запрос упал: реплика сохранена для повтора.
        self.controller.client.chat_failed.emit("timeout")
        self.assertIsNotNone(self.controller.retry_message)
        # Повтор — восстановленная (свободная) сцена.
        self.assertTrue(self.controller.retry_last_user_message())
        system = _last_system(self.send_mock)
        self.assertIn("free talk (no role-play scene is active)", system)
        self.assertNotIn("role-play scene is ACTIVE", system)


class TestModelSigning(ControllerBase):
    """Ответ подписывается моделью, которой реально ушёл запрос."""

    def test_retry_signed_with_current_model(self) -> None:
        self.controller.selected_model = "m1"
        self.assertTrue(self.controller.send_message("Hello"))
        self.controller.client.chat_failed.emit("timeout")
        self.assertIsNotNone(self.controller.retry_message)
        # Модель сменили — повтор уйдёт m2, ответ подпишется m2.
        self.controller.selected_model = "m2"
        self.assertTrue(self.controller.retry_last_user_message())
        self.assertEqual(self.send_mock.call_args.args[0], "m2")
        self.controller.client.chat_completed.emit("Hi there.")
        _pump(0.02)
        answer = self.controller.history[-1]
        self.assertEqual(answer.role, "assistant")
        self.assertEqual(answer.model, "m2", "ответ подписан моделью запроса")

    def test_replace_signed_with_current_model(self) -> None:
        self.controller.selected_model = "m1"
        self.assertTrue(self.controller.send_message("Hello"))
        self.controller.client.chat_completed.emit("Hi there.")
        _pump(0.02)
        self.assertEqual(self.controller.history[-1].model, "m1")
        # Модель сменили — замена уйдёт m2, ответ подпишется m2.
        self.controller.selected_model = "m2"
        self.assertTrue(self.controller.replace_last_user_message("Hello again"))
        self.assertEqual(self.send_mock.call_args.args[0], "m2")
        self.controller.client.chat_completed.emit("Hello again!")
        _pump(0.02)
        self.assertEqual(self.controller.history[-1].model, "m2")


class TestNewSessionClearsRequestState(ControllerBase):
    """«Новое занятие» (reset_history) сбрасывает состояние запроса."""

    def test_reset_clears_pending_and_retry(self) -> None:
        self.assertTrue(self.controller.send_message("Hello"))
        self.controller.client.chat_failed.emit("timeout")
        self.assertIsNotNone(self.controller.retry_message)

        self.controller.reset_history()

        self.assertIsNone(self.controller.retry_message)
        self.assertIsNone(self.controller.pending_message)
        self.assertEqual(self.controller.history, [])
        self.assertFalse(self.controller.busy)
        # Повтор после сброса не отправляет сообщение из прошлого занятия.
        self.assertFalse(self.controller.retry_last_user_message())
        self.assertEqual(self.send_mock.call_count, 1, "запросов не добавилось")


class TestMarkerHelpers(unittest.TestCase):
    """parse_scene_marker / scene_from_marker — юнит-проверки."""

    def test_marker_last_line_removed(self) -> None:
        clean, marker = parse_scene_marker(
            "Goodbye!\n[SCENE:free]"
        )
        self.assertEqual(clean, "Goodbye!")
        self.assertEqual(marker, "free")

    def test_marker_with_roles(self) -> None:
        clean, marker = parse_scene_marker(
            "You are a guest.\n"
            "[SCENE:hotel check-in|you are the receptionist, the student is the guest]"
        )
        self.assertEqual(clean, "You are a guest.")
        self.assertEqual(
            scene_from_marker(marker),
            Scene(
                name="hotel check-in",
                roles="you are the receptionist, the student is the guest",
            ),
        )

    def test_marker_not_last_line_untouched(self) -> None:
        text = "Look: [SCENE:free] and more"
        clean, marker = parse_scene_marker(text)
        self.assertEqual(clean, text)
        self.assertIsNone(marker)

    def test_no_marker(self) -> None:
        clean, marker = parse_scene_marker("just text")
        self.assertEqual(clean, "just text")
        self.assertIsNone(marker)

    def test_scene_from_marker_variants(self) -> None:
        self.assertIsNone(scene_from_marker(None))
        self.assertIsNone(scene_from_marker(""))
        self.assertIsNone(scene_from_marker("free"))
        self.assertIsNone(scene_from_marker("  FREE  "))
        self.assertEqual(
            scene_from_marker("hotel check-in"), Scene(name="hotel check-in", roles="")
        )
        self.assertEqual(
            scene_from_marker("hotel check-in | roles here"),
            Scene(name="hotel check-in", roles="roles here"),
        )


class TestWindowIntegration(ControllerBase):
    """Окно: маркер не виден и не озвучен; ошибка файла — с путём."""

    def setUp(self) -> None:
        super().setUp()
        _app()
        self.audio = AudioController()
        self.audio.speak_reply = mock.MagicMock(name="speak_reply")
        self.audio.record = mock.MagicMock(name="record")
        self.stt = SttEngine(FakeBackend(result_text=""))
        self.window = MainWindow(self.controller, self.audio, self.stt)
        self.window.show()
        _pump(0.05)

    def tearDown(self) -> None:
        try:
            self.stt.shutdown()
            self.audio.shutdown()
            _pump(0.05)
        finally:
            del self.window
            super().tearDown()

    def test_reply_marker_hidden_from_chat_and_speech(self) -> None:
        self.window.message_input.setText("I'm looking for a room")
        self.window.send_message()
        _pump(0.05)
        self.controller.client.chat_completed.emit(
            "You are a guest, I am the hotel receptionist.\n"
            "[SCENE:hotel check-in|you are the hotel receptionist, "
            "the student is the guest]"
        )
        _pump(0.05)

        text = self.window.history.toPlainText()
        self.assertIn("You are a guest, I am the hotel receptionist.", text)
        self.assertNotIn("[SCENE:", text)
        self.assertEqual(self.audio.speak_reply.call_count, 1)
        self.assertEqual(
            self.audio.speak_reply.call_args[0][0],
            "You are a guest, I am the hotel receptionist.",
        )
        self.assertIsNotNone(self.controller.scene)

    def test_prompt_error_shown_with_path_and_input_kept(self) -> None:
        os.remove(self.prompt_file)
        self.window.message_input.setText("hello")
        self.window.send_message()
        _pump(0.05)

        self.send_mock.assert_not_called()
        self.assertIn(self.prompt_file, self.window.status_label.text())
        # Текст ввода сохранён — повтор после исправления без дубля.
        self.assertEqual(self.window.message_input.text(), "hello")
        self.assertEqual(self.controller.history, [])

        self._write_prompt("DELTA RULE — teach well.")
        self.window.send_message()
        _pump(0.05)
        self.send_mock.assert_called_once()
        self.assertIn("DELTA RULE", _last_system(self.send_mock))
        self.assertEqual(self.window.message_input.text(), "")


if __name__ == "__main__":
    unittest.main()
