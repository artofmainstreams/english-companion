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
  режимов разговора);
* блок урока: ``Lesson: name``, ``Student level``, темы, фразы, слова
  (материал ученика);
* актуальный режим разговора: сцена активна — её имя (``SceneSpec.name``)
  и ОБЕ РОЛИ (``teacher_role``/``student_role``), иначе — свободная
  беседа. Режим — явный выбор пользователя (комбо «Сцена»), состояние
  хранит ``SessionController`` (``select_scene``/``clear_scene``); ответ
  модели режим НЕ меняет.
"""

from __future__ import annotations

from pathlib import Path

from .lesson import Lesson, SceneSpec


class TeacherPromptError(ValueError):
    """Файл промпта преподавателя отсутствует, пуст или не читается."""


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
    scene: SceneSpec | None = None,
    prompt_path: str | Path | None = None,
) -> str:
    """Системный контекст для Ollama: промпт файла + урок + режим.

    Промпт читается из файла ПЕРЕД каждым запросом (``{level}``
    подставляется уровнем урока). Чистая функция: без Qt и сети —
    легко проверять в тестах.

    ``scene`` — активная ролевая сцена урока (``SceneSpec``); ``None`` —
    свободная беседа. Сцена передаёт имя и ОБЕ роли (учитель/ученик).
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
        lines.append(f"Your role: {scene.teacher_role}")
        lines.append(f"Student role: {scene.student_role}")
        lines.append(
            "Play your role from the first reply. Let the student formulate "
            "their own lines: respond to what they say and keep the situation "
            "moving with new facts and short questions."
        )
        lines.append(
            "Keep the roles while the scene is active; the scene ends only "
            "when the student clearly finishes it."
        )
    else:
        lines.append(
            "Conversation mode: free talk (no role-play scene is active). "
            "Follow the topic the student chooses; use lesson vocabulary "
            "when it fits their topic."
        )
    return "\n".join(lines)
