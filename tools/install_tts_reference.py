#!/usr/bin/env python
"""Установка референса голоса TTS в постоянную папку данных приложения.

Копирует одобренный референс (по умолчанию — реплика p4 голоса D,
``~/Desktop/qwen-voice-low-register/qwen3_D_vd_p4.wav``) в
``app_data.tts_reference_path()`` — на macOS это
``~/Library/Application Support/english-companion/tts/qwen3_p4_ref.wav``.

Нужно после переустановки окружения или переноса проекта: сам файл референса
живёт вне проекта и вне .venv, поэтому его нужно восстановить
вручную один раз. Проверка наличия: приложение при отсутствии файла
показывает понятную ошибку (см. ``companion/tts_qwen3.py``).

Запуск (из корня проекта)::

    .venv/bin/python tools/install_tts_reference.py
    # свой исходный файл:
    .venv/bin/python tools/install_tts_reference.py /путь/к/ref.wav
"""

import os
import shutil
import sys
from pathlib import Path

# Скрипт в tools/; гарантируем импорт пакета ``companion``.
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from companion.app_data import tts_reference_path  # noqa: E402

# Одобренный референс: реплика p4 варианта D (4-я реплика VoiceDesign).
DEFAULT_SOURCE = (
    Path.home() / "Desktop" / "qwen-voice-low-register" / "qwen3_D_vd_p4.wav"
)


def install(source: Path, dest: Path | None = None) -> Path:
    """Копирует референс в папку данных; возвращает путь назначения."""
    if not source.is_file():
        raise FileNotFoundError(
            f"исходный референс не найден: {source}. "
            "Передай путь к файлу аргументом скрипта."
        )
    dest = dest or tts_reference_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def main() -> int:
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    source = Path(args[0]).expanduser() if args else DEFAULT_SOURCE
    try:
        dest = install(source)
    except FileNotFoundError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    size = dest.stat().st_size
    print(f"Референс установлен: {dest} ({size} байт)")
    print("Проверка: .venv/bin/python real_tts_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
