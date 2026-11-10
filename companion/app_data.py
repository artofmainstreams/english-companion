"""Постоянная папка данных приложения (вне репозитория и вне .venv).

Для файлов, которые переживают переустановку/обновление кода — например,
референс голоса TTS (``tts_reference_path``). Папка создаётся по
потребности; содержимое НЕ попадает в Git (как и веса моделей).
"""

import os
import sys
from pathlib import Path

APP_NAME = "english-companion"


def app_data_dir() -> Path:
    """Корень постоянных данных приложения для текущей платформы.

    macOS: ``~/Library/Application Support/english-companion``;
    Windows: ``%APPDATA%/english-companion``;
    Linux: ``$XDG_DATA_HOME/english-companion`` (по умолчанию
    ``~/.local/share/english-companion``).
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / APP_NAME


def tts_reference_path() -> Path:
    """Путь к референсу голоса TTS (Qwen3 Base, реплика p4).

    Файл кладётся сюда установщиком ``tools/install_tts_reference.py`` —
    он переживает переустановку приложения, в отличие от Desktop/веса.
    """
    return app_data_dir() / "tts" / "qwen3_p4_ref.wav"
