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
from companion.lesson import Lesson, SceneSpec  # noqa: E402
from companion.main_window import MainWindow  # noqa: E402
from companion.message import Message  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.session_controller import (  # noqa: E402
    FREE_TALK_EXIT_MESSAGE,
    SessionController,
)
from companion.stt_engine import SttEngine  # noqa: E402
from companion.teacher_context import (  # noqa: E402
    TeacherPromptError,
    build_teacher_context,
    load_teacher_prompt,
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


def _scene_spec() -> SceneSpec:
    """Сцена урока (то, что хранится в ``Lesson.scenes``)."""
    return SceneSpec(
        id="hotel_check_in",
        name="Заселение в отель",
        teacher_role="hotel receptionist",
        student_role="hotel guest",
        start_message="Hello! I'd like to check in, please.",
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
        self.assertIn("one question", text)
        self.assertIn("I don't know", text)
        # Режимы разговора задаёт приложение (маркера сцены в промпте нет).
        self.assertIn("Free talk: follow the topic the student chooses", text)
        self.assertIn("Play that role from the first reply", text)
        self.assertIn("Thanks, that's all", text)
        self.assertNotIn("[SCENE:", text)

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
    """Режим разговора: сцену выбирает пользователь (комбо «Сцена»)."""

    def test_initial_request_is_free_talk(self) -> None:
        self.assertTrue(self.controller.send_message("hi"))
        system = _last_system(self.send_mock)
        self.assertIn("free talk (no role-play scene is active)", system)
        self.assertNotIn("role-play scene is ACTIVE", system)

    def test_select_scene_sends_start_message_once_with_scene_context(self) -> None:
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        # На одно переключение — ровно один запрос.
        self.send_mock.assert_called_once()
        args, kwargs = self.send_mock.call_args
        # Стартовая реплика — как обычная реплика пользователя (в чате — «You»).
        self.assertEqual(
            [m.content for m in args[1]], [spec.start_message]
        )
        # Учитель получает активную сцену и ОБЕ роли уже с первого запроса.
        system = kwargs["system"]
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn("Scene: Заселение в отель", system)
        self.assertIn("hotel receptionist", system)
        self.assertIn("hotel guest", system)
        # Сцена установлена; стартовая реплика в истории/в работе.
        self.assertEqual(self.controller.scene, spec)
        self.assertEqual(
            self.controller.pending_message, Message("user", spec.start_message)
        )
        self.assertTrue(self.controller.busy)

    def test_select_scene_keeps_existing_history(self) -> None:
        spec = _scene_spec()
        # Непустая история — сохраняется при переключении сцены.
        self.assertTrue(self.controller.send_message("I'm looking for a room"))
        self.controller.client.chat_completed.emit("Welcome to the hotel.")
        _pump(0.02)
        self.assertEqual(len(self.controller.history), 2)

        self.assertTrue(self.controller.select_scene(spec))
        args, kwargs = self.send_mock.call_args
        # История + стартовая реплика сцены — в одном запросе.
        self.assertEqual(
            [m.content for m in args[1]],
            ["I'm looking for a room", "Welcome to the hotel.", spec.start_message],
        )
        self.assertIn("role-play scene is ACTIVE", kwargs["system"])

    def test_exit_scene_sends_farewell_line_once(self) -> None:
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        self.controller.client.chat_completed.emit("Welcome to the hotel.")
        _pump(0.02)
        self.assertIsNotNone(self.controller.scene)
        history_before = list(self.controller.history)

        # «Свободная беседа»: сцена снимается ДО отправки, прощальная
        # реплика — как обычная реплика пользователя; ровно один запрос.
        self.assertTrue(self.controller.exit_scene())
        self.assertIsNone(self.controller.scene)
        self.assertEqual(self.send_mock.call_count, 2, "ровно один запрос")
        args, kwargs = self.send_mock.call_args
        # История сохранена + прощальная реплика — в одном запросе.
        self.assertEqual(
            [m.content for m in args[1]],
            [
                spec.start_message,
                "Welcome to the hotel.",
                FREE_TALK_EXIT_MESSAGE,
            ],
        )
        # Прощальная реплика ушла уже в режиме свободной беседы.
        self.assertIn("free talk (no role-play scene is active)", kwargs["system"])
        self.assertNotIn("role-play scene is ACTIVE", kwargs["system"])
        # Реплика — в работе, в чате будет как обычная реплика «You».
        self.assertEqual(
            self.controller.pending_message, Message("user", FREE_TALK_EXIT_MESSAGE)
        )

        # Ответ пришёл: история = прежняя + прощальный обмен.
        self.controller.client.chat_completed.emit("Sure — tell me about your trip.")
        _pump(0.02)
        self.assertEqual(len(self.controller.history), len(history_before) + 2)
        self.assertEqual(
            self.controller.history[-2].content, FREE_TALK_EXIT_MESSAGE
        )
        self.assertEqual(self.controller.history[-2].role, "user")

        # Следующий запрос — свободная беседа; материал сохранён.
        self.assertTrue(self.controller.send_message("Tell me about your trip."))
        system = _last_system(self.send_mock)
        self.assertIn("free talk (no role-play scene is active)", system)
        self.assertNotIn("role-play scene is ACTIVE", system)
        self.assertNotIn("Заселение в отель", system)
        self.assertIn("Урок 9", system)
        self.assertIn("reservation", system)

    def test_exit_scene_without_scene_rejected(self) -> None:
        # Сцены нет — «Свободная беседа» запрос не отправляет.
        self.assertFalse(self.controller.exit_scene())
        self.assertIsNone(self.controller.scene)
        self.send_mock.assert_not_called()

    def test_model_reply_does_not_change_scene(self) -> None:
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        # Ответ модели «завершает сцену» (маркер из старой модели) — сцену
        # это НЕ меняет: режим — явный выбор пользователя.
        self.controller.client.chat_completed.emit(
            "Thanks, that's all.\n[SCENE:free]"
        )
        _pump(0.02)
        self.assertEqual(self.controller.scene, spec)
        # Следующий запрос — сцена по-прежнему активна.
        self.assertTrue(self.controller.send_message("Goodbye!"))
        system = _last_system(self.send_mock)
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn(spec.teacher_role, system)
        self.assertIn(spec.student_role, system)

    def test_select_scene_prompt_error_keeps_previous_scene(self) -> None:
        spec_a = _scene_spec()
        spec_b = SceneSpec(
            id="airport",
            name="В аэропорту",
            teacher_role="check-in clerk",
            student_role="passenger",
            start_message="Hello, I have a reservation.",
        )
        self.assertTrue(self.controller.select_scene(spec_a))
        self.controller.client.chat_completed.emit("Sure, what name?")
        _pump(0.02)

        # Файл промпта не читается: переключение НЕ начинается.
        os.remove(self.prompt_file)
        self.assertFalse(self.controller.select_scene(spec_b))
        self.send_mock.assert_called_once()
        # Прежняя сцена восстановлена; причина — в статусе (с путём).
        self.assertEqual(self.controller.scene, spec_a)
        self.assertIn(self.prompt_file, self.controller.status)

    def test_exit_scene_prompt_error_keeps_scene(self) -> None:
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        self.controller.client.chat_completed.emit("Sure, what name?")
        _pump(0.02)

        # Файл промпта не читается: выход из сцены НЕ начинается —
        # прежняя сцена восстановлена, запрос не ушёл.
        os.remove(self.prompt_file)
        self.assertFalse(self.controller.exit_scene())
        self.send_mock.assert_called_once()
        self.assertEqual(self.controller.scene, spec)
        self.assertIn(self.prompt_file, self.controller.status)

    def test_select_scene_while_busy_rejected(self) -> None:
        spec = _scene_spec()
        self.assertTrue(self.controller.send_message("hi"))
        self.assertTrue(self.controller.busy)
        # Во время операции переключение блокируется (как смена урока).
        self.assertFalse(self.controller.select_scene(spec))
        self.assertIsNone(self.controller.scene)

    def test_reset_history_clears_scene(self) -> None:
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        self.controller.client.chat_completed.emit("Welcome to the hotel.")
        _pump(0.02)
        self.assertIsNotNone(self.controller.scene)
        self.controller.reset_history()
        self.assertIsNone(self.controller.scene)
        self.assertTrue(self.controller.send_message("hi"))
        self.assertIn("free talk", _last_system(self.send_mock))


class TestSceneRewrite(ControllerBase):
    """Перезапись реплики: активная сцена (явный выбор) сохраняется."""

    def test_rewrite_start_message_keeps_scene_and_roles(self) -> None:
        """Перезапись СТАРТОВОЙ реплики сцены: роли сохраняются."""
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        self.controller.client.chat_completed.emit("Welcome to the hotel!")
        _pump(0.02)
        self.assertEqual(self.controller.scene, spec)

        # Перезапись не отменяет явный выбор пользователя в комбо.
        self.assertTrue(
            self.controller.replace_last_user_message("Hello, I need to check in.")
        )
        system = _last_system(self.send_mock)
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn(spec.teacher_role, system)
        self.assertIn(spec.student_role, system)
        self.assertEqual(self.controller.scene, spec)
        # Стартовая реплика заменена (без дубля).
        self.assertEqual(
            self.controller.pending_message,
            Message("user", "Hello, I need to check in."),
        )

    def test_rewrite_prompt_error_preserves_history_and_scene(self) -> None:
        """Сбой чтения промпта: история и сцена сохраняются, запрос не ушёл."""
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        self.controller.client.chat_completed.emit("Welcome to the hotel!")
        _pump(0.02)
        self.assertEqual(self.controller.scene, spec)
        history_before = list(self.controller.history)

        os.remove(self.prompt_file)
        self.assertFalse(self.controller.replace_last_user_message("I need a room"))
        self.assertEqual(self.send_mock.call_count, 1, "запросов не добавилось")
        # История и сцена не тронуты.
        self.assertEqual(self.controller.history, history_before)
        self.assertEqual(self.controller.scene, spec)

    def test_retry_after_failed_request_keeps_scene(self) -> None:
        """Запрос сцены упал — повтор несёт ту же (активную) сцену."""
        spec = _scene_spec()
        self.assertTrue(self.controller.select_scene(spec))
        # Новый запрос упал: реплика сохранена для повтора.
        self.controller.client.chat_failed.emit("timeout")
        self.assertIsNotNone(self.controller.retry_message)
        # Повтор — с активной сценой и её ролями.
        self.assertTrue(self.controller.retry_last_user_message())
        system = _last_system(self.send_mock)
        self.assertIn("role-play scene is ACTIVE", system)
        self.assertIn(spec.teacher_role, system)
        self.assertIn(spec.student_role, system)
        self.assertEqual(self.controller.scene, spec)


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


class TestWindowIntegration(ControllerBase):
    """Окно: выбор сцены + авто-старт; ошибка файла промпта — с путём."""

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

    def test_scene_combo_lists_lesson_scenes(self) -> None:
        """Комбо «Сцена»: «Свободная беседа» + сцены выбранного урока (Урок 9)."""
        combo = self.window.scene_combo
        lesson = self.controller.lesson
        self.assertIsNotNone(lesson)
        self.assertEqual(combo.itemText(0), "Свободная беседа")
        self.assertIsNone(combo.itemData(0))
        for i, spec in enumerate(lesson.scenes, start=1):
            self.assertEqual(combo.itemText(i), spec.name)
            self.assertIs(combo.itemData(i), spec)
        self.assertEqual(combo.currentIndex(), 0)

    def test_select_scene_starts_scene_and_shows_start_message(self) -> None:
        combo = self.window.scene_combo
        spec = combo.itemData(1)
        self.assertIsNotNone(spec)
        self.controller.selected_model = "m"
        combo.setCurrentIndex(1)
        _pump(0.05)
        # Ровно один запрос; стартовая реплика — как обычная реплика «You».
        self.send_mock.assert_called_once()
        chat = self.window.history.toPlainText()
        self.assertIn(f"You: {spec.start_message}", chat)
        self.assertIn("role-play scene is ACTIVE", _last_system(self.send_mock))
        self.assertIsNotNone(self.controller.scene)
        # Ответ показан и озвучен; сцена сохранена.
        self.controller.client.chat_completed.emit("Welcome to the hotel!")
        _pump(0.05)
        self.assertIn("Teacher", self.window.history.toPlainText())
        self.assertEqual(self.audio.speak_reply.call_count, 1)
        self.assertIsNotNone(self.controller.scene)

    def test_free_talk_selection_sends_farewell_line(self) -> None:
        combo = self.window.scene_combo
        spec = combo.itemData(1)
        self.assertIsNotNone(spec)
        self.controller.selected_model = "m"
        combo.setCurrentIndex(1)
        self.controller.client.chat_completed.emit("Welcome to the hotel!")
        _pump(0.05)
        self.assertIsNotNone(self.controller.scene)
        history_before = list(self.controller.history)
        # «Свободная беседа» — сцена завершается, прощальная реплика
        # отправляется как обычная реплика «You» (ровно один запрос).
        combo.setCurrentIndex(0)
        _pump(0.05)
        self.assertIsNone(self.controller.scene)
        self.assertEqual(self.send_mock.call_count, 2, "ровно один запрос")
        self.assertIn(
            f"You: {FREE_TALK_EXIT_MESSAGE}", self.window.history.toPlainText()
        )
        self.assertIn("free talk (no role-play scene is active)",
                      _last_system(self.send_mock))
        # Прежняя история сохранена; новая реплика — в работе/чате.
        self.assertEqual(len(self.controller.history), len(history_before))

    def test_prompt_error_restores_scene_combo_and_scene(self) -> None:
        combo = self.window.scene_combo
        self.controller.selected_model = "m"
        os.remove(self.prompt_file)
        combo.setCurrentIndex(1)
        _pump(0.05)
        # Отправку начать нельзя: прежний пункт и прежняя сцена.
        self.assertIsNone(self.controller.scene)
        self.assertEqual(combo.currentIndex(), 0)
        self.assertEqual(self.controller.history, [])
        self.assertIn(self.prompt_file, self.window.status_label.text())
        self.send_mock.assert_not_called()
        # После исправления файла выбор сцены работает (один запрос).
        self._write_prompt("DELTA RULE — teach well.")
        combo.setCurrentIndex(1)
        _pump(0.05)
        self.send_mock.assert_called_once()
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
