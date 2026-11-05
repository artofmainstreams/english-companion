#!/usr/bin/env python
"""Явно запускаемая проверка реального распознавания.

Отличается от обычных тестов: использует НАСТОЯЩУЮ модель, а не подмену.
По умолчанию проверяется бэкенд из AppConfig — ``mlx``
(mlx-community/whisper-large-v3-mlx; macOS arm64 + системный ffmpeg).
Альтернативу можно выбрать явно:

    .venv/bin/python real_stt_check.py --backend faster-whisper

При первом запуске нужны интернет-веса (скачиваются и кэшируются), далее —
полностью локально. Замеряет загрузку модели ОТДЕЛЬНО от повторного
распознавания.

Запуск::

    # Пайплайн-дым-тест (генерирует тон; речь не распознается — пусто):
    .venv/bin/python real_stt_check.py

    # С твоим файлом речи (лучший результат):
    .venv/bin/python real_stt_check.py путь/к/записи.m4a

Никаких GUI, TTS или чата: только модель + аудиофайл.
"""

import os
import sys
import time
import wave


def _ensure_project_on_path() -> None:
    """Скрипт лежит в корне; гарантируем импорт пакета ``companion``."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def _make_tone_wav(path: str, seconds: int = 2, hz: float = 440.0) -> None:
    """Генерирует тон 440 Гц как аудиофайл для дым-теста пайплайна."""
    import math

    rate = 8000
    frames = bytearray()
    for i in range(rate * seconds):
        value = int(0.3 * 32767 * math.sin(2 * math.pi * hz * i / rate))
        frames += value.to_bytes(2, "little", signed=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def main(argv: list[str]) -> int:
    _ensure_project_on_path()
    from companion.config import AppConfig
    from companion.stt_backends import build_backend

    # Опционально: --backend mlx|faster-whisper (иначе — дефолт AppConfig).
    backend = None
    if "--backend" in argv:
        idx = argv.index("--backend")
        if idx + 1 >= len(argv):
            print("ОШИБКА: --backend требует значение (mlx | faster-whisper)")
            return 2
        backend = argv[idx + 1]
        positional = [a for i, a in enumerate(argv) if i not in (idx, idx + 1)]
    else:
        positional = argv

    config = AppConfig(stt_backend=backend) if backend else AppConfig()
    print(
        "STT настройки:",
        f"backend={config.stt_backend} "
        f"mlx_model={config.stt_mlx_model} "
        f"whisper_model={config.stt_whisper_model} "
        f"device={config.stt_device} compute_type={config.stt_compute_type} "
        f"language={config.stt_language} task={config.stt_task} "
        f"vad_filter={config.stt_vad_filter}",
    )

    # Аудиофайл: аргумент или сгенерированный тон (argv[0] — имя скрипта).
    if len(positional) > 1:
        audio_path = os.path.abspath(positional[1])
        if not os.path.exists(audio_path):
            print(f"ОШИБКА: файл не найден: {audio_path}")
            return 2
        source = "аргумент"
    else:
        import tempfile

        tmp = tempfile.mkdtemp(prefix="english-companion-realstt-")
        audio_path = os.path.join(tmp, "tone.wav")
        _make_tone_wav(audio_path)
        source = "сгенерированный тон (дым-тест, без речи)"

    print(f"Аудио ({source}): {audio_path}")

    backend = build_backend(config)

    # 1) Загрузка модели — отдельный замер (при первом запуске скачивает веса).
    print("\n[1] Загрузка модели…")
    t0 = time.perf_counter()
    model = backend.load_model()
    load_s = time.perf_counter() - t0
    print(f"    модель: {type(model).__name__} · загрузка: {load_s:.2f} с")

    # 2) Распознавание (модель уже загружена) — первый замер.
    print("\n[2] Распознавание #1…")
    t0 = time.perf_counter()
    text1 = backend.transcribe(model, audio_path).strip()
    t1 = time.perf_counter() - t0
    print(f"    результат: {text1!r}")
    print(f"    распознавание #1: {t1:.2f} с")

    # 3) Повторное распознавание (модель переиспользуется) — отдельный замер.
    print("\n[3] Повторное распознавание #2…")
    t0 = time.perf_counter()
    text2 = backend.transcribe(model, audio_path).strip()
    t2 = time.perf_counter() - t0
    print(f"    результат: {text2!r}")
    print(f"    распознавание #2 (повторное): {t2:.2f} с")

    print("\nИтог:")
    print(f"    загрузка модели:            {load_s:.2f} с")
    print(f"    распознавание (1-й раз):    {t1:.2f} с")
    print(f"    распознавание (повторное): {t2:.2f} с")
    if not text1:
        print("    примечание: текст пуст (ожидается для тона/тишины — это дым-тест)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
