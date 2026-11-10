"""Загрузка и валидация уроков (JSON).

Урок — структурированный файл, опирающийся на исходный урок
(``raw/homework12_online.doc``; папка ``raw/`` — только локальная, в Git
не попадает) — по умолчанию ``lessons/homework12.json``.

Правила содержания:
* фразы, упражнения и примеры в грамматике цитируются дословно из
  источника, со ссылкой на строку извлечённого текста (``source_ref``);
* цели и темы помечены ``origin``: ``"document"`` — формулировка из
  источника, ``"derived"`` — интерпретация, добавленная автором JSON;
* целевой английский в грамматике — в поле ``interpretation``
  (интерпретация, не из документа);
* ``flags`` — спорные формулировки источника (опечатки, переводы,
  can/could), отмеченные для проверки.

``load_lesson(path)`` проверяет обязательные поля и типы и возвращает
``Lesson``. При проблемах бросает ``LessonError`` со ВСЕМИ найденными
проблемами сразу (с именами полей).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Допустимые значения ``origin`` у целей и тем.
ORIGINS = ("document", "derived")

_REQUIRED = (
    "id",
    "course",
    "level",
    "number",
    "title",
    "topics",
    "objectives",
    "phrases",
)
_OPTIONAL = ("source_document", "grammar", "exercises", "resources", "flags")
_KNOWN = _REQUIRED + _OPTIONAL


class LessonError(ValueError):
    """Урок не найден, не читается или имеет отсутствующие/неверные поля."""


@dataclass(frozen=True)
class Topic:
    """Тема урока. ``origin``: 'document' — из источника, 'derived' — интерпретация."""

    text: str
    origin: str


@dataclass(frozen=True)
class Objective:
    """Учебная цель. ``origin``: 'document' — из источника, 'derived' — интерпретация."""

    text: str
    origin: str
    source_ref: str = ""


@dataclass(frozen=True)
class Phrase:
    """Фраза из документа (английский + перевод) со ссылкой на строку источника."""

    en: str
    ru: str
    source_ref: str = ""


@dataclass(frozen=True)
class GrammarPoint:
    """Грамматика: название + примеры из документа; ``interpretation`` — целевая форма (интерпретация)."""

    name: str
    examples: tuple[str, ...]
    interpretation: str = ""
    origin: str = "derived"
    source_ref: str = ""


@dataclass(frozen=True)
class Exercise:
    """Упражнение из документа (задание + пункты, со ссылкой на строки)."""

    task: str
    items: tuple[str, ...]
    source_ref: str = ""


@dataclass(frozen=True)
class Resource:
    """Ресурс урока (например, видео) с примечанием, где находится ссылка."""

    kind: str
    title: str
    note: str = ""
    source_ref: str = ""


@dataclass(frozen=True)
class Flag:
    """Спорная формулировка источника (опечатка, перевод) — отмечена для проверки."""

    text: str
    location: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Lesson:
    """Прошедший валидацию урок."""

    id: str
    course: str
    level: str
    number: int
    title: str
    topics: tuple[Topic, ...]
    objectives: tuple[Objective, ...]
    phrases: tuple[Phrase, ...]
    grammar: tuple[GrammarPoint, ...]
    exercises: tuple[Exercise, ...]
    resources: tuple[Resource, ...]
    flags: tuple[Flag, ...]
    source_document: str = ""


# ---------------------------------------------------------------------------
# Внутренние помощники валидации (собирают ВСЕ ошибки, а не первую)
# ---------------------------------------------------------------------------


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _required_text(data: dict, key: str, errors: list[str], where: str = "") -> str:
    value = data.get(key)
    if _is_text(value):
        return value
    label = f"{where}.{key}" if where else key
    errors.append(f"{label}: обязательное непустое строковое поле")
    return ""


def _optional_text(data: dict, key: str, errors: list[str], where: str = "") -> str:
    value = data.get(key)
    if value is None:
        return ""
    if _is_text(value):
        return value
    label = f"{where}.{key}" if where else key
    errors.append(f"{label}: строка или отсутствие поля")
    return ""


def _required_list(data: dict, key: str, errors: list[str], min_items: int) -> list[Any]:
    value = data.get(key)
    if isinstance(value, list) and len(value) >= min_items:
        return value
    errors.append(f"{key}: обязательный список (не короче {min_items} элем.)")
    return []


def _items_text_list(
    data: dict, key: str, errors: list[str], where: str
) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"{where}.{key}: обязательный непустой список строк")
        return ()
    items: list[str] = []
    for i, item in enumerate(value):
        if _is_text(item):
            items.append(item)
        else:
            errors.append(f"{where}.{key}[{i}]: обязательное непустое строковое значение")
    return tuple(items)


def _origin(data: dict, errors: list[str], where: str) -> str:
    value = data.get("origin")
    if value in ORIGINS:
        return value
    errors.append(f"{where}.origin: одно из {list(ORIGINS)}")
    return "derived"


def _optional_origin(data: dict, errors: list[str], where: str) -> str:
    value = data.get("origin", "derived")
    if value in ORIGINS:
        return value
    errors.append(f"{where}.origin: одно из {list(ORIGINS)}")
    return "derived"


def _dict_items(data: Any, key: str, errors: list[str]) -> list[Any]:
    """Секция-список объектов (grammar/exercises/resources/flags)."""
    if key not in data:
        return []
    raw = data[key]
    if not isinstance(raw, list):
        errors.append(f"{key}: список (или отсутствие поля)")
        return []
    items = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            items.append((i, item))
        else:
            errors.append(f"{key}[{i}]: обязательный объект")
    return items


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def default_lesson_path() -> Path:
    """Путь к первому уроку: корень проекта → ``lessons/homework12.json``."""
    return Path(__file__).resolve().parent.parent / "lessons" / "homework12.json"


def load_lesson(path: str | Path) -> Lesson:
    """Считать урок из JSON, проверив обязательные поля и типы.

    Возвращает:
        ``Lesson`` с типизированными полями (кортежи).

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
    course = _required_text(data, "course", errors)
    level = _required_text(data, "level", errors)
    title = _required_text(data, "title", errors)

    number_raw = data.get("number")
    if isinstance(number_raw, bool) or not isinstance(number_raw, int) or number_raw <= 0:
        errors.append("number: положительное целое (int)")
        number = 1
    else:
        number = number_raw

    topics: list[Topic] = []
    for i, item in enumerate(_required_list(data, "topics", errors, 1)):
        if not isinstance(item, dict):
            errors.append(f"topics[{i}]: обязательный объект {{text, origin}}")
            continue
        text = _required_text(item, "text", errors, where=f"topics[{i}]")
        origin = _origin(item, errors, where=f"topics[{i}]")
        if _is_text(text):
            topics.append(Topic(text=text, origin=origin))

    objectives: list[Objective] = []
    for i, item in enumerate(_required_list(data, "objectives", errors, 1)):
        if not isinstance(item, dict):
            errors.append(f"objectives[{i}]: обязательный объект {{text, origin, source_ref?}}")
            continue
        text = _required_text(item, "text", errors, where=f"objectives[{i}]")
        origin = _origin(item, errors, where=f"objectives[{i}]")
        ref = _optional_text(item, "source_ref", errors, where=f"objectives[{i}]")
        if _is_text(text):
            objectives.append(Objective(text=text, origin=origin, source_ref=ref))

    phrases: list[Phrase] = []
    for i, item in enumerate(_required_list(data, "phrases", errors, 1)):
        if not isinstance(item, dict):
            errors.append(f"phrases[{i}]: обязательный объект {{en, ru, source_ref?}}")
            continue
        en = _required_text(item, "en", errors, where=f"phrases[{i}]")
        ru = _required_text(item, "ru", errors, where=f"phrases[{i}]")
        ref = _optional_text(item, "source_ref", errors, where=f"phrases[{i}]")
        if _is_text(en) and _is_text(ru):
            phrases.append(Phrase(en=en, ru=ru, source_ref=ref))

    grammar: list[GrammarPoint] = []
    for i, item in _dict_items(data, "grammar", errors):
        name = _required_text(item, "name", errors, where=f"grammar[{i}]")
        examples = _items_text_list(item, "examples", errors, f"grammar[{i}]")
        interpretation = _optional_text(item, "interpretation", errors, where=f"grammar[{i}]")
        origin = _optional_origin(item, errors, where=f"grammar[{i}]")
        ref = _optional_text(item, "source_ref", errors, where=f"grammar[{i}]")
        if _is_text(name) and examples:
            grammar.append(
                GrammarPoint(
                    name=name,
                    examples=examples,
                    interpretation=interpretation,
                    origin=origin,
                    source_ref=ref,
                )
            )

    exercises: list[Exercise] = []
    for i, item in _dict_items(data, "exercises", errors):
        task = _required_text(item, "task", errors, where=f"exercises[{i}]")
        items = _items_text_list(item, "items", errors, f"exercises[{i}]")
        ref = _optional_text(item, "source_ref", errors, where=f"exercises[{i}]")
        if _is_text(task) and items:
            exercises.append(Exercise(task=task, items=items, source_ref=ref))

    resources: list[Resource] = []
    for i, item in _dict_items(data, "resources", errors):
        title_r = _required_text(item, "title", errors, where=f"resources[{i}]")
        kind = _optional_text(item, "kind", errors, where=f"resources[{i}]")
        note = _optional_text(item, "note", errors, where=f"resources[{i}]")
        ref = _optional_text(item, "source_ref", errors, where=f"resources[{i}]")
        if _is_text(title_r):
            resources.append(
                Resource(kind=kind, title=title_r, note=note, source_ref=ref)
            )

    flags: list[Flag] = []
    for i, item in _dict_items(data, "flags", errors):
        text = _required_text(item, "text", errors, where=f"flags[{i}]")
        location = _optional_text(item, "location", errors, where=f"flags[{i}]")
        reason = _optional_text(item, "reason", errors, where=f"flags[{i}]")
        if _is_text(text):
            flags.append(Flag(text=text, location=location, reason=reason))

    source_document = _optional_text(data, "source_document", errors)

    if errors:
        raise LessonError("Урок некорректен: " + "; ".join(errors))

    return Lesson(
        id=lesson_id,
        course=course,
        level=level,
        number=number,
        title=title,
        topics=tuple(topics),
        objectives=tuple(objectives),
        phrases=tuple(phrases),
        grammar=tuple(grammar),
        exercises=tuple(exercises),
        resources=tuple(resources),
        flags=tuple(flags),
        source_document=source_document,
    )
