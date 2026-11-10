"""Тесты загрузки урока (companion/lesson.py).

Запуск::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests

Без Qt и сети: только чтение JSON и проверка полей. Согласованный
формат: ``id`` (непустая строка), ``level`` (Elementary | Pre-
Intermediate | Intermediate | Advanced), ``name`` (непустая строка),
``topics`` (хотя бы одна), ``phrases``/``words`` — списки строк
(могут быть пустыми). Проверяются: реальный урок
``lessons/homework12.json``, корректные данные (в т.ч. пустые
phrases/words) и понятные ``LessonError`` для отсутствующих/неверных
данных (все проблемы сразу, с именами полей).
"""

import json
import tempfile
import unittest
from pathlib import Path

from companion.lesson import (
    LEVELS,
    Lesson,
    LessonError,
    default_lesson_path,
    load_lesson,
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


if __name__ == "__main__":
    unittest.main()
