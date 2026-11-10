"""Тесты загрузки урока (companion/lesson.py).

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Без Qt и сети: только чтение JSON и проверка полей. Проверяются:
* реальный урок ``lessons/homework12.json`` — идентичность (курс, уровень,
  номер), наполненность, ссылки примеров на строки источника,
  разделение целей «из документа» / «интерпретация»;
* корректный минимальный и полный учебные данные;
* понятные ``LessonError`` для отсутствующих/неверных данных
  (все проблемы сразу, с именами полей).
"""

import json
import tempfile
import unittest
from pathlib import Path

from companion.lesson import (
    Lesson,
    LessonError,
    default_lesson_path,
    load_lesson,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_LESSON = ROOT / "lessons" / "homework12.json"


def _valid() -> dict:
    """Полный корректный урок (все опциональные секции)."""
    return {
        "id": "hw-test",
        "course": "Тестовый курс",
        "level": "elementary",
        "number": 1,
        "title": "Тестовый урок",
        "source_document": "raw/test.doc",
        "topics": [{"text": "Тема", "origin": "document"}],
        "objectives": [
            {"text": "Цель из источника", "origin": "document", "source_ref": "строка 1"},
            {"text": "Цель как интерпретация", "origin": "derived"},
        ],
        "phrases": [{"en": "Hello!", "ru": "Привет!", "source_ref": "строка 2"}],
        "grammar": [
            {
                "name": "Приветствие",
                "examples": ["Hello!"],
                "interpretation": "целевая форма",
                "origin": "derived",
                "source_ref": "строка 2",
            }
        ],
        "exercises": [{"task": "Поздороваться", "items": ["Hello!"], "source_ref": "строка 2"}],
        "resources": [
            {"kind": "video", "title": "Видео", "note": "под документом", "source_ref": "строка 5"}
        ],
        "flags": [{"text": "Спорная формулировка", "location": "строка 2", "reason": "опечатка"}],
    }


def _minimal() -> dict:
    """Минимальный корректный урок (только обязательные поля)."""
    return {
        "id": "hw-test",
        "course": "Тестовый курс",
        "level": "a1",
        "number": 3,
        "title": "Тестовый урок",
        "topics": [{"text": "Тема", "origin": "document"}],
        "objectives": [{"text": "Цель", "origin": "document"}],
        "phrases": [{"en": "Hello!", "ru": "Привет!"}],
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

    def test_loads_and_identity(self):
        lesson = load_lesson(REAL_LESSON)
        self.assertIsInstance(lesson, Lesson)
        self.assertEqual(lesson.id, "hw12")
        self.assertEqual(lesson.course, "Выживание за границей")
        self.assertEqual(lesson.level, "elementary")
        self.assertEqual(lesson.number, 12)
        self.assertIn("ресторан", lesson.title.lower())
        self.assertTrue(lesson.source_document.endswith("homework12_online.doc"))

    def test_content_is_populated(self):
        lesson = load_lesson(REAL_LESSON)
        self.assertTrue(lesson.topics)
        self.assertTrue(lesson.objectives)
        self.assertGreaterEqual(len(lesson.phrases), 20)
        self.assertTrue(lesson.grammar)
        self.assertTrue(lesson.exercises)
        self.assertTrue(lesson.resources)
        self.assertTrue(lesson.flags)

    def test_phrases_have_text_and_source_refs(self):
        lesson = load_lesson(REAL_LESSON)
        for phrase in lesson.phrases:
            self.assertTrue(phrase.en.strip(), "пустой en")
            self.assertTrue(phrase.ru.strip(), "пустой ru")
            self.assertIn("строка", phrase.source_ref, "нет ссылки на строку источника")

    def test_goals_are_split_by_origin(self):
        lesson = load_lesson(REAL_LESSON)
        origins = {objective.origin for objective in lesson.objectives}
        self.assertIn("document", origins)
        self.assertIn("derived", origins)

    def test_translation_exercise_keeps_all_items(self):
        lesson = load_lesson(REAL_LESSON)
        translation = [ex for ex in lesson.exercises if "перевести" in ex.task.lower()]
        self.assertTrue(translation, "нет упражнения на перевод")
        self.assertGreaterEqual(len(translation[0].items), 20)


class ValidLessonTest(unittest.TestCase):
    def test_minimal_loads(self):
        lesson = load_lesson(_write(_minimal()))
        self.assertEqual(lesson.id, "hw-test")
        self.assertEqual(lesson.number, 3)
        self.assertEqual(lesson.grammar, ())
        self.assertEqual(lesson.exercises, ())
        self.assertEqual(lesson.flags, ())

    def test_full_loads(self):
        lesson = load_lesson(_write(_valid()))
        self.assertEqual(lesson.topics[0].origin, "document")
        self.assertEqual(lesson.objectives[1].origin, "derived")
        self.assertEqual(lesson.objectives[0].source_ref, "строка 1")
        self.assertEqual(lesson.phrases[0].en, "Hello!")
        self.assertEqual(lesson.grammar[0].examples, ("Hello!",))
        self.assertEqual(lesson.grammar[0].interpretation, "целевая форма")
        self.assertEqual(lesson.exercises[0].items, ("Hello!",))
        self.assertEqual(lesson.resources[0].title, "Видео")
        self.assertEqual(lesson.flags[0].location, "строка 2")


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

    def test_missing_id(self):
        data = _minimal()
        del data["id"]
        self._error(data, "id")

    def test_missing_course_and_level_reported_together(self):
        data = _minimal()
        del data["course"]
        del data["level"]
        message = self._error(data, "course", "level")
        self.assertIn("Урок некорректен", message)

    def test_number_must_be_positive_int(self):
        data = _minimal()
        data["number"] = "12"
        self._error(data, "number")
        data = _minimal()
        data["number"] = 0
        self._error(data, "number")
        data = _minimal()
        data["number"] = True
        self._error(data, "number")

    def test_topics_cannot_be_empty(self):
        data = _minimal()
        data["topics"] = []
        self._error(data, "topics")

    def test_topic_item_must_be_object(self):
        data = _minimal()
        data["topics"] = ["просто строка"]
        self._error(data, "topics[0]")

    def test_objective_origin_must_be_known(self):
        data = _minimal()
        data["objectives"] = [{"text": "Цель", "origin": "magic"}]
        self._error(data, "origin")

    def test_phrase_without_russian(self):
        data = _minimal()
        data["phrases"] = [{"en": "Hello!"}]
        self._error(data, "ru")

    def test_unknown_key_rejected(self):
        data = _minimal()
        data["objectivs"] = "опечатка"
        self._error(data, "неизвестные поля", "objectivs")


if __name__ == "__main__":
    unittest.main()
