"""Загрузка и валидация уроков (JSON).

Согласованный формат, обязательные поля:
* ``id`` — непустая строка;
* ``level`` — одно из ``Elementary``, ``Pre-Intermediate``,
  ``Intermediate``, ``Advanced``;
* ``name`` — непустая строка (например, «Урок 12»);
* ``topics`` — список тем и конструкций для изучения, хотя бы одна;
* ``phrases`` — список английских фраз (строки), может быть пустым;
* ``words`` — список английских слов и устойчивых сочетаний (строки),
  может быть пустым;
* ``scenes`` — необязательный список ролевых сцен: каждый объект —
  непустые строки ``id``, ``name``, ``teacher_role``, ``student_role``,
  ``start_message``; ``id`` уникален в пределах урока. Поле отсутствует —
  урок поддерживает только свободную беседу.

Уроки лежат в папке ``lessons/`` (JSON-файлы). ``load_lessons()``
загружает ВСЕ уроки из папки (натуральный порядок по имени файла:
Урок 9 — до Урока 12); ``load_lesson(path)`` — один файл. Сломанный
файл не отменяет остальные — его проблема попадает в список ошибок.

``load_lesson(path)`` проверяет поля и типы и возвращает ``Lesson``.
При проблемах бросает ``LessonError`` со ВСЕМИ найденными проблемами
сразу (с именами полей).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Уровни урока (согласованный формат).
LEVELS = ("Elementary", "Pre-Intermediate", "Intermediate", "Advanced")

_KNOWN = ("id", "level", "name", "topics", "phrases", "words", "scenes")

#: Поля ролевой сцены (все — непустые строки).
_SCENE_FIELDS = ("id", "name", "teacher_role", "student_role", "start_message")


class LessonError(ValueError):
    """Урок не найден, не читается или имеет отсутствующие/неверные поля."""


@dataclass(frozen=True)
class SceneSpec:
    """Ролевая сцена урока (список ``Lesson.scenes``).

    Все поля — непустые строки; ``id`` уникален в пределах урока.
    """

    id: str
    name: str
    teacher_role: str
    student_role: str
    start_message: str


@dataclass(frozen=True)
class Lesson:
    """Прошедший валидацию урок."""

    id: str
    level: str
    name: str
    topics: tuple[str, ...]
    phrases: tuple[str, ...]
    words: tuple[str, ...]
    scenes: tuple[SceneSpec, ...] = ()


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


def _scenes(data: dict, errors: list[str]) -> tuple[SceneSpec, ...]:
    """Необязательное поле ``scenes``: ролевые сцены урока.

    Каждая сцена — объект с непустыми строковыми полями
    ``id/name/teacher_role/student_role/start_message``; ``id`` уникален
    в пределах урока. Поле отсутствует — урок только со свободной
    беседой. Ошибки указывают поле и номер сцены (``scenes[N].field``).
    """
    value = data.get("scenes")
    if value is None:
        return ()
    if not isinstance(value, list):
        errors.append(
            "scenes: список ролевых сцен (объекты с полями id, name, "
            "teacher_role, student_role, start_message)"
        )
        return ()
    scenes: list[SceneSpec] = []
    seen_ids: dict[str, int] = {}
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(
                f"scenes[{i}]: объект с полями id, name, teacher_role, "
                "student_role, start_message"
            )
            continue
        unknown = sorted(set(item) - set(_SCENE_FIELDS))
        if unknown:
            errors.append(f"scenes[{i}]: неизвестные поля: " + ", ".join(unknown))
        fields: dict[str, str] = {}
        ok = True
        for field in _SCENE_FIELDS:
            text = item.get(field)
            if _is_text(text):
                fields[field] = text
            else:
                errors.append(
                    f"scenes[{i}].{field}: обязательное непустое строковое поле"
                )
                ok = False
        if not ok:
            continue
        scene_id = fields["id"]
        if scene_id in seen_ids:
            errors.append(
                f"scenes[{i}].id: дублируется с scenes[{seen_ids[scene_id]}] — "
                "id уникален в пределах урока"
            )
            continue
        seen_ids[scene_id] = i
        scenes.append(SceneSpec(**fields))
    return tuple(scenes)


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def lessons_dir() -> Path:
    """Папка с уроками: корень проекта → ``lessons``."""
    return Path(__file__).resolve().parent.parent / "lessons"


def default_lesson_path() -> Path:
    """Путь к уроку по умолчанию: ``lessons/homework12.json``."""
    return lessons_dir() / "homework12.json"


def _natural_key(filename: str) -> tuple:
    """Натуральный порядок имён файлов: homework9.json — до homework12.json."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", filename.lower())
    )


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
    scenes = _scenes(data, errors)

    if errors:
        raise LessonError("Урок некорректен: " + "; ".join(errors))

    return Lesson(
        id=lesson_id,
        level=level,
        name=name,
        topics=topics,
        phrases=phrases,
        words=words,
        scenes=scenes,
    )


def load_lessons(directory: str | Path | None = None) -> tuple[list[Lesson], list[str]]:
    """Загружает ВСЕ уроки (``*.json``) из папки ``lessons/``.

    Возвращает:
        ``(уроки, ошибки)`` — отсортированные уроки (натуральный порядок
        по имени файла: Урок 9 до Урока 12) и понятные ошибки
        (``имя файла: проблема``) для некорректных файлов. Сломанный
        файл не отменяет остальные.

    Бросает:
        LessonError: папка уроков не найдена или это не папка.
    """
    directory = Path(directory) if directory is not None else lessons_dir()
    if not directory.is_dir():
        raise LessonError(f"Папка уроков не найдена: {directory}")
    lessons: list[Lesson] = []
    errors: list[str] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: _natural_key(p.name)):
        try:
            lessons.append(load_lesson(path))
        except LessonError as exc:
            errors.append(f"{path.name}: {exc}")
    return lessons, errors
