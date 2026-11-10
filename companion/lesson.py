"""Загрузка и валидация уроков (JSON).

Согласованный формат, обязательные поля:
* ``id`` — непустая строка;
* ``level`` — одно из ``Elementary``, ``Pre-Intermediate``,
  ``Intermediate``, ``Advanced``;
* ``name`` — непустая строка (например, «Урок 12»);
* ``topics`` — список тем и конструкций для изучения, хотя бы одна;
* ``phrases`` — список английских фраз (строки), может быть пустым;
* ``words`` — список английских слов и устойчивых сочетаний (строки),
  может быть пустым.

Урок по умолчанию — ``lessons/homework12.json`` (содержание опирается
на исходный материал ``raw/homework12_online.doc``; папка ``raw/`` —
только локальная, в Git не попадает).

``load_lesson(path)`` проверяет поля и типы и возвращает ``Lesson``.
При проблемах бросает ``LessonError`` со ВСЕМИ найденными проблемами
сразу (с именами полей).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Уровни урока (согласованный формат).
LEVELS = ("Elementary", "Pre-Intermediate", "Intermediate", "Advanced")

_KNOWN = ("id", "level", "name", "topics", "phrases", "words")


class LessonError(ValueError):
    """Урок не найден, не читается или имеет отсутствующие/неверные поля."""


@dataclass(frozen=True)
class Lesson:
    """Прошедший валидацию урок."""

    id: str
    level: str
    name: str
    topics: tuple[str, ...]
    phrases: tuple[str, ...]
    words: tuple[str, ...]


# ---------------------------------------------------------------------------
# Внутренние помощники валидации (собирают ВСЕ ошибки, а не первую)
# ---------------------------------------------------------------------------


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _required_text(data: dict, key: str, errors: list[str]) -> str:
    value = data.get(key)
    if _is_text(value):
        return value
    errors.append(f"{key}: обязательное непустое строковое поле")
    return ""


def _string_list(data: dict, key: str, errors: list[str], min_items: int) -> tuple[str, ...]:
    """Список непустых строк; ``min_items=1`` для topics, 0 для phrases/words."""
    value = data.get(key)
    if not isinstance(value, list):
        hint = "список строк (может быть пустым)" if min_items == 0 else "список строк"
        errors.append(f"{key}: {hint}")
        return ()
    if len(value) < min_items:
        errors.append(f"{key}: нужна хотя бы одна тема")
    items: list[str] = []
    for i, item in enumerate(value):
        if _is_text(item):
            items.append(item)
        else:
            errors.append(f"{key}[{i}]: обязательное непустое строковое значение")
    return tuple(items)


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def default_lesson_path() -> Path:
    """Путь к первому уроку: корень проекта → ``lessons/homework12.json``."""
    return Path(__file__).resolve().parent.parent / "lessons" / "homework12.json"


def load_lesson(path: str | Path) -> Lesson:
    """Считать урок из JSON, проверив поля и типы.

    Возвращает:
        ``Lesson`` (списки — кортежи).

    Бросает:
        LessonError: файл отсутствует, JSON невалиден или поля
            отсутствующие/неверные — ВСЕ найденные проблемы сразу,
            с именами полей.
    """
    path = Path(path)
    if not path.is_file():
        raise LessonError(f"Файл урока не найден: {path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LessonError(f"Файл урока не читается ({path}): {exc}") from exc
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LessonError(
            f"JSON урока некорректен ({path}, строка {exc.lineno}, "
            f"столбец {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise LessonError("JSON урока должен быть объектом верхнего уровня (ключ: значение)")

    errors: list[str] = []
    unknown = sorted(set(data) - set(_KNOWN))
    if unknown:
        errors.append("неизвестные поля: " + ", ".join(unknown))

    lesson_id = _required_text(data, "id", errors)

    level = data.get("level")
    if level not in LEVELS:
        errors.append(f"level: одно из {list(LEVELS)}")
        level = LEVELS[0]

    name = _required_text(data, "name", errors)
    topics = _string_list(data, "topics", errors, min_items=1)
    phrases = _string_list(data, "phrases", errors, min_items=0)
    words = _string_list(data, "words", errors, min_items=0)

    if errors:
        raise LessonError("Урок некорректен: " + "; ".join(errors))

    return Lesson(
        id=lesson_id,
        level=level,
        name=name,
        topics=topics,
        phrases=phrases,
        words=words,
    )
