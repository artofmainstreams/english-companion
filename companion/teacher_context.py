"""Системный контекст преподавателя: файл промпта + урок + режим разговора.

Собирает системный контекст, который ``SessionController`` передаёт в
Ollama вместе с историей разговора (поле ``system``). Контекст — НЕ
часть чата: в окно выводятся только реплики пользователя и
преподавателя.

Постоянные инструкции преподавателя — в ``prompts/teacher.txt``
(UTF-8, редактируется пользователем) и читаются ПЕРЕД КАЖДЫМ запросом
(не кэшируются): правка файла подхватывается со следующего запроса.
Путь разрешается относительно приложения (корень проекта), запуск из
другой рабочей папки загрузку не ломает. Файл отсутствует, пуст или
не читается — ``TeacherPromptError`` с путём; отправка блокируется,
повтор после исправления работает. ``{level}`` в файле подставляется
уровнем выбранного урока.

Состав системного сообщения:
* текст файла (идентичность + учебные и языковые правила + правила
  режимов разговора и маркера сцены);
* блок урока: ``Lesson: name``, ``Student level``, темы, фразы, слова
  (материал ученика);
* актуальный режим разговора: сцена активна — её имя и роли, иначе —
  свободная беседа. Состояние сцены хранит и определяет
  ``SessionController`` (по маркеру ``[SCENE:...]`` в ответе
  преподавателя; маркер из ответа убирается — в чат, озвучивание и
  историю не попадает).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .lesson import Lesson


class TeacherPromptError(ValueError):
    """Файл промпта преподавателя отсутствует, пуст или не читается."""


@dataclass(frozen=True)
class Scene:
    """Активная ролевая сцена; ``roles`` — свободный текст ролей."""

    name: str
    roles: str = ""


def teacher_prompt_path() -> Path:
    """Путь к ``prompts/teacher.txt`` относительно приложения (корень
    проекта — запуск из другой рабочей папки загрузку не ломает)."""
    return Path(__file__).resolve().parent.parent / "prompts" / "teacher.txt"


def load_teacher_prompt(path: str | Path | None = None) -> str:
    """Читает постоянный промпт преподавателя — без кэша, заново.

    Бросает ``TeacherPromptError`` (со путём): файл отсутствует,
    пуст (только пробельные) или не читается.
    """
    path = Path(path) if path is not None else teacher_prompt_path()
    if not path.is_file():
        raise TeacherPromptError(
            f"Файл промпта преподавателя не найден: {path}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TeacherPromptError(
            f"Файл промпта преподавателя не читается ({path}): {exc}"
        ) from exc
    if not text.strip():
        raise TeacherPromptError(
            f"Файл промпта преподавателя пуст: {path}"
        )
    return text


def build_teacher_context(
    lesson: Lesson,
    scene: Scene | None = None,
    prompt_path: str | Path | None = None,
) -> str:
    """Системный контекст для Ollama: промпт файла + урок + режим.

    Промпт читается из файла ПЕРЕД каждым запросом (``{level}``
    подставляется уровнем урока). Чистая функция: без Qt и сети —
    легко проверять в тестах.
    """
    text = load_teacher_prompt(prompt_path).replace("{level}", lesson.level)
    lines: list[str] = [text.rstrip(), ""]
    lines.append(f"Lesson: {lesson.name}")
    lines.append(f"Student level: {lesson.level}")
    lines.append("")
    lines.append("Topics and structures to practice:")
    for topic in lesson.topics:
        lines.append(f"- {topic}")
    if lesson.phrases:
        lines.append("")
        lines.append("Phrases from the lesson (the student's practice material):")
        lines.append(", ".join(lesson.phrases))
    if lesson.words:
        lines.append("")
        lines.append("Vocabulary from the lesson (the student's practice material):")
        lines.append(", ".join(lesson.words))
    lines.append("")
    if scene is not None:
        lines.append("Conversation mode: role-play scene is ACTIVE.")
        lines.append(f"Scene: {scene.name}")
        if scene.roles:
            lines.append(f"Roles: {scene.roles}")
        lines.append(
            "Keep the roles and the situation while the scene is active; "
            "the scene ends only when the student clearly finishes it."
        )
    else:
        lines.append(
            "Conversation mode: free talk (no role-play scene is active). "
            "Continue as the teacher-conversation partner; the student's "
            "topic may be discussed."
        )
    return "\n".join(lines)


#: Маркер сцены — последняя непустая строка ответа преподавателя.
_SCENE_MARKER = re.compile(r"^\[SCENE:(.+?)\]$")


def parse_scene_marker(text: str) -> tuple[str, str | None]:
    """Возвращает ``(текст_без_маркера, маркер_или_None)``.

    Маркер — последняя непустая строка вида ``[SCENE:...]``:
    ``free`` (свободная беседа) или ``<имя>|<роли>`` (сцена активна).
    Из ответа маркер убирается: в чат, озвучивание и историю он не
    попадает. Маркер не на последней строке (или отсутствует) —
    текст возвращается как есть.
    """
    lines = text.splitlines()
    marker: str | None = None
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue
        match = _SCENE_MARKER.match(lines[i].strip())
        if match:
            marker = match.group(1).strip()
            del lines[i]
        break
    return "\n".join(lines), marker


def scene_from_marker(marker: str | None) -> Scene | None:
    """``free``/пусто/отсутствует → ``None`` (свободная беседа);
    иначе ``Scene(name, roles)`` (``name|roles`` разбивается по
    первому ``|``)."""
    if not marker:
        return None
    name, _, roles = marker.partition("|")
    name = name.strip()
    if not name or name.lower() == "free":
        return None
    return Scene(name=name, roles=roles.strip())
