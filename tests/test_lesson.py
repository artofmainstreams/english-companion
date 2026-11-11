"""Тесты загрузки урока (companion/lesson.py).

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Без Qt и сети: только чтение JSON и проверка полей. Согласованный
формат: ``id`` (непустая строка), ``level`` (Elementary | Pre-
Intermediate | Intermediate | Advanced), ``name`` (непустая строка),
``topics`` (хотя бы одна), ``phrases``/``words`` — списки строк
(могут быть пустыми). Проверяются: реальный урок
``lessons/homework12.json``, корректные данные (в т.ч. пустые
phrases/words), понятные ``LessonError`` для отсутствующих/неверных
данных (все проблемы сразу, с именами полей) и ``load_lessons()`` —
все уроки папки в натуральном порядке, сломанный файл не отменяет
остальные (ошибка — по файлу).
"""

import json
import tempfile
import unittest
from pathlib import Path

from companion.lesson import (
    LEVELS,
    Lesson,
    LessonError,
    SceneSpec,
    default_lesson_path,
    load_lesson,
    load_lessons,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_LESSON = ROOT / "lessons" / "homework12.json"

#: Поля, удалённые из формата (не должны встречаться в JSON).
REMOVED_KEYS = (
    "source_document",
    "objectives",
    "grammar",
    "exercises",
    "resources",
    "flags",
)


def _valid() -> dict:
    """Корректный урок (все поля)."""
    return {
        "id": "hw-test",
        "level": "Advanced",
        "name": "Урок 1",
        "topics": ["Фразы «В ресторане»", "Present Simple"],
        "phrases": ["Can you pass me the salt, please?", "We'd like to pay by card."],
        "words": ["salt", "bill", "pay by card"],
    }


def _scene() -> dict:
    """Корректная ролевая сцена (все поля)."""
    return {
        "id": "hotel_check_in",
        "name": "Заселение в отель",
        "teacher_role": "hotel receptionist",
        "student_role": "hotel guest",
        "start_message": "Hello! I'd like to check in, please.",
    }


def _write(data) -> Path:
    """Записать dict (или строку) во временный файл и вернуть путь."""
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False, dir=tempfile.gettempdir()
    )
    with handle:
        handle.write(text)
    return Path(handle.name)


class RealLessonTest(unittest.TestCase):
    def test_default_path_is_real_file(self):
        self.assertEqual(default_lesson_path(), REAL_LESSON)
        self.assertTrue(REAL_LESSON.is_file(), f"нет файла {REAL_LESSON}")

    def test_project_lessons_load_clean(self):
        """Все уроки папки lessons/ проекта загружаются без ошибок."""
        lessons, errors = load_lessons()
        self.assertEqual(errors, [], f"некорректные уроки в lessons/: {errors}")
        self.assertTrue(
            any(l.id == "hw12" for l in lessons),
            "Урок 12 должен быть в lessons/",
        )

    def test_identity(self):
        lesson = load_lesson(REAL_LESSON)
        self.assertIsInstance(lesson, Lesson)
        self.assertEqual(lesson.id, "hw12")
        self.assertEqual(lesson.level, "Elementary")
        self.assertEqual(lesson.name, "Урок 12")

    def test_removed_keys_are_gone(self):
        raw = json.loads(REAL_LESSON.read_text(encoding="utf-8"))
        for key in REMOVED_KEYS:
            self.assertNotIn(key, raw, f"поле {key!r} осталось в формате")

    def test_topics_cover_structures(self):
        lesson = load_lesson(REAL_LESSON)
        joined = " ".join(lesson.topics)
        self.assertIn("Present Simple", joined)
        self.assertIn("Present Continuous", joined)
        self.assertIn("Can/Could", joined)

    def test_phrases_kept_and_typos_fixed(self):
        lesson = load_lesson(REAL_LESSON)
        self.assertGreaterEqual(len(lesson.phrases), 30)
        self.assertIn("Smoking or non-smoking?", lesson.phrases)
        for phrase in lesson.phrases:
            self.assertNotIn("Smocking", phrase, "опечатка «Smocking» не исправлена")
            self.assertNotIn("Сan", phrase, "кириллическая «С» в «Can»")

    def test_words_extracted(self):
        lesson = load_lesson(REAL_LESSON)
        self.assertTrue(lesson.words)
        for expected in ("tea", "bill", "reservation", "menu", "spicy food"):
            self.assertIn(expected, lesson.words)


class ValidLessonTest(unittest.TestCase):
    def test_all_levels_accepted(self):
        for level in LEVELS:
            data = _valid()
            data["level"] = level
            self.assertEqual(load_lesson(_write(data)).level, level)

    def test_empty_phrases_and_words_allowed(self):
        data = _valid()
        data["phrases"] = []
        data["words"] = []
        lesson = load_lesson(_write(data))
        self.assertEqual(lesson.phrases, ())
        self.assertEqual(lesson.words, ())

    def test_single_topic_allowed(self):
        data = _valid()
        data["topics"] = ["одна тема"]
        lesson = load_lesson(_write(data))
        self.assertEqual(lesson.topics, ("одна тема",))


class ScenesTest(unittest.TestCase):
    """Необязательное поле ``scenes``: валидация + реальные уроки."""

    def _error(self, data, *substrings: str) -> str:
        path = _write(data)
        with self.assertRaises(LessonError) as ctx:
            load_lesson(path)
        message = str(ctx.exception)
        for substring in substrings:
            self.assertIn(substring, message)
        return message

    def test_scenes_absent_defaults_empty(self):
        lesson = load_lesson(_write(_valid()))
        self.assertEqual(lesson.scenes, ())

    def test_scene_fields_loaded(self):
        data = {**_valid(), "scenes": [_scene()]}
        lesson = load_lesson(_write(data))
        self.assertEqual(len(lesson.scenes), 1)
        spec = lesson.scenes[0]
        self.assertIsInstance(spec, SceneSpec)
        self.assertEqual(spec.id, "hotel_check_in")
        self.assertEqual(spec.name, "Заселение в отель")
        self.assertEqual(spec.teacher_role, "hotel receptionist")
        self.assertEqual(spec.student_role, "hotel guest")
        self.assertEqual(spec.start_message, "Hello! I'd like to check in, please.")

    def test_real_homework9_has_hotel_scene(self):
        lesson = load_lesson(ROOT / "lessons" / "homework9.json")
        self.assertEqual([s.id for s in lesson.scenes], ["hotel_check_in"])
        spec = lesson.scenes[0]
        self.assertEqual(spec.name, "Заселение в отель")
        self.assertEqual(spec.teacher_role, "hotel receptionist")
        self.assertEqual(spec.student_role, "hotel guest")
        self.assertEqual(spec.start_message, "Hello! I'd like to check in, please.")

    def test_real_homework12_has_no_scenes(self):
        lesson = load_lesson(REAL_LESSON)
        self.assertEqual(lesson.scenes, ())

    def test_scenes_must_be_list(self):
        data = {**_valid(), "scenes": _scene()}
        self._error(data, "scenes")

    def test_scene_item_must_be_object(self):
        data = {**_valid(), "scenes": ["hotel_check_in"]}
        self._error(data, "scenes[0]")

    def test_scene_missing_field_reported_with_number(self):
        broken = {k: v for k, v in _scene().items() if k != "name"}
        data = {**_valid(), "scenes": [broken]}
        self._error(data, "scenes[0].name")

    def test_scene_empty_field_rejected(self):
        scene = _scene()
        scene["teacher_role"] = "   "
        data = {**_valid(), "scenes": [scene]}
        self._error(data, "scenes[0].teacher_role")

    def test_scene_id_unique_per_lesson(self):
        second = {**_scene(), "name": "Вторая сцена"}
        data = {**_valid(), "scenes": [_scene(), second]}
        message = self._error(data, "scenes[1].id")
        # Ошибка указывает обе сцены (поле и номера).
        self.assertIn("scenes[0]", message)

    def test_scene_unknown_key_rejected(self):
        scene = {**_scene(), "extra": "field"}
        data = {**_valid(), "scenes": [scene]}
        self._error(data, "scenes[0]", "extra")


class ErrorTest(unittest.TestCase):
    def _error(self, data, *substrings: str) -> str:
        path = _write(data)
        with self.assertRaises(LessonError) as ctx:
            load_lesson(path)
        message = str(ctx.exception)
        for substring in substrings:
            self.assertIn(substring, message)
        return message

    def test_missing_file(self):
        with self.assertRaises(LessonError) as ctx:
            load_lesson(Path(tempfile.gettempdir()) / "no_such_lesson.json")
        self.assertIn("не найден", str(ctx.exception))

    def test_invalid_json(self):
        self._error("{not a json", "JSON")

    def test_top_level_must_be_object(self):
        self._error([1, 2, 3], "объект")

    def test_missing_id_name_level_reported_together(self):
        data = _valid()
        del data["id"]
        del data["name"]
        del data["level"]
        message = self._error(data, "id", "name", "level")
        self.assertIn("Урок некорректен", message)

    def test_level_must_be_known(self):
        data = _valid()
        data["level"] = "elementary"
        self._error(data, "level")
        data = _valid()
        data["level"] = "B1"
        self._error(data, "level")

    def test_topics_cannot_be_empty(self):
        data = _valid()
        data["topics"] = []
        self._error(data, "topics")

    def test_topic_item_must_be_nonempty_string(self):
        data = _valid()
        data["topics"] = ["норма", ""]
        self._error(data, "topics[1]")

    def test_phrases_must_be_string_list(self):
        data = _valid()
        data["phrases"] = "не список"
        self._error(data, "phrases")
        data = _valid()
        data["phrases"] = ["норма", 42]
        self._error(data, "phrases[1]")

    def test_words_must_be_string_list(self):
        data = _valid()
        data["words"] = ["tea", ""]
        self._error(data, "words[1]")

    def test_unknown_key_rejected(self):
        data = _valid()
        data["objectives"] = "старый формат"
        self._error(data, "неизвестные поля", "objectives")


class LoadLessonsTest(unittest.TestCase):
    """``load_lessons()``: все уроки папки, натуральный порядок, ошибки по файлам."""

    def _make_dir(self, files: dict[str, object]) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="english-companion-lessons-"))
        for name, data in files.items():
            text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            (directory / name).write_text(text, encoding="utf-8")
        return directory

    def test_all_lessons_in_natural_order(self) -> None:
        """homework9.json — до homework12.json (а не лексикографически)."""
        directory = self._make_dir({
            "homework12.json": {**_valid(), "id": "hw12", "name": "Урок 12"},
            "homework9.json": {**_valid(), "id": "hw9", "name": "Урок 9"},
            "homework2.json": {**_valid(), "id": "hw2", "name": "Урок 2"},
        })
        lessons, errors = load_lessons(directory)
        self.assertEqual(errors, [])
        self.assertEqual([l.id for l in lessons], ["hw2", "hw9", "hw12"])

    def test_bad_file_does_not_break_others(self) -> None:
        """Сломанный файл — в ошибках (с именем файла); остальные загружены."""
        directory = self._make_dir({
            "bad.json": {**_valid(), "level": "B1"},
            "good.json": _valid(),
        })
        lessons, errors = load_lessons(directory)
        self.assertEqual([l.id for l in lessons], ["hw-test"])
        self.assertEqual(len(errors), 1)
        self.assertIn("bad.json", errors[0])
        self.assertIn("level", errors[0])

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(LessonError) as ctx:
            load_lessons(Path(tempfile.gettempdir()) / "no_such_lessons")
        self.assertIn("Папка уроков не найдена", str(ctx.exception))

    def test_empty_directory(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="english-companion-lessons-empty-"))
        self.assertEqual(load_lessons(directory), ([], []))


if __name__ == "__main__":
    unittest.main()
