#!/usr/bin/env python
"""Диагностика распознавания одного аудиофайла.

Без изменения настроек приложения сравнивает:
  A) рабочий backend (путь приложения: AppConfig() → build_backend() →
     load_model() → transcribe() — фактический бэкенд и модель из конфига);
  B) прямой вызов mlx_whisper.transcribe с ИДЕНТИЧНЫМИ параметрами
     (path_or_hf_repo / language / task), включая метрики сегментов, которые
     библиотека отдаёт (avg_logprob, compression_ratio, no_speech_prob);
  C) референс: faster-whisper base.en (проверка, не «похож» ли результат на
     поведение более слабой модели).

Никакого initial_prompt, никаких LLM-исправлений, никаких переопределений
параметров: только то, что использует приложение.

Запуск (из корня проекта)::

    .venv/bin/python tools/stt_diagnose.py /путь/recording_1.m4a
"""

import json
import os
import subprocess
import sys
import time


def _ensure_project_on_path() -> None:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tools_dir)
    if root not in sys.path:
        sys.path.insert(0, root)


def _file_info(path: str) -> None:
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    fmt = data.get("format", {})
    stream = (data.get("streams") or [{}])[0]
    print(
        f"Файл: {path}\n"
        f"  codec={stream.get('codec_name')} sr={stream.get('sample_rate')} "
        f"ch={stream.get('channels')} duration={fmt.get('duration')} с"
    )


def _print_mlx_segments(result: dict) -> None:
    segments = result.get("segments") or []
    print(f"  сегментов: {len(segments)}")
    for i, seg in enumerate(segments):
        print(
            f"  [{i}] {seg.get('start')}..{seg.get('end')} с | "
            f"avg_logprob={seg.get('avg_logprob'):.3f} "
            f"compression_ratio={seg.get('compression_ratio'):.3f} "
            f"no_speech_prob={seg.get('no_speech_prob'):.3f} | "
            f"text={seg.get('text')!r}"
        )


def _run_engine(config, label: str) -> None:
    from companion.stt_backends import build_backend

    print(f"\n=== {label} ===")
    print(
        f"  backend={config.stt_backend} "
        f"model={config.stt_mlx_model if config.stt_backend == 'mlx' else config.stt_whisper_model} "
        f"language={config.stt_language} task={config.stt_task} "
        f"vad_filter={config.stt_vad_filter} "
        f"(device={config.stt_device}, compute_type={config.stt_compute_type})"
    )
    backend = build_backend(config)

    t0 = time.perf_counter()
    model = backend.load_model()
    load_s = time.perf_counter() - t0
    print(f"  загрузка модели: {load_s:.2f} с ({type(model).__name__})")

    t0 = time.perf_counter()
    text = backend.transcribe(model, AUDIO_PATH).strip()
    dt = time.perf_counter() - t0
    print(f"  распознавание: {dt:.2f} с")
    print(f"  ТЕКСТ: {text!r}")

    # Метрики сегментов (faster-whisper отдаёт их в каждом сегменте).
    if config.stt_backend == "faster-whisper":
        segs, _info = model.transcribe(
            AUDIO_PATH,
            language=config.stt_language,
            task=config.stt_task,
            vad_filter=config.stt_vad_filter,
        )
        n = 0
        for seg in segs:
            n += 1
            print(
                f"  [{n - 1}] {seg.start:.2f}..{seg.end:.2f} с | "
                f"avg_logprob={seg.avg_logprob:.3f} "
                f"compression_ratio={seg.compression_ratio:.3f} "
                f"no_speech_prob={seg.no_speech_prob:.3f} | "
                f"text={seg.text!r}"
            )


def _run_mlx_direct(config, label: str) -> None:
    import mlx_whisper

    print(f"\n=== {label} ===")
    print(
        f"  path_or_hf_repo={config.stt_mlx_model} "
        f"language={config.stt_language} task={config.stt_task} "
        f"(fp16/verbose — дефолты библиотеки)"
    )
    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        AUDIO_PATH,
        path_or_hf_repo=config.stt_mlx_model,
        language=config.stt_language,
        task=config.stt_task,
    )
    dt = time.perf_counter() - t0
    print(f"  распознавание: {dt:.2f} с")
    print(f"  language={result.get('language')}")
    print(f"  ТЕКСТ: {result.get('text')!r}")
    _print_mlx_segments(result)


def main() -> int:
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print("Использование: stt_diagnose.py /путь/к/записи.m4a", file=sys.stderr)
        return 2
    global AUDIO_PATH
    AUDIO_PATH = os.path.abspath(sys.argv[1])

    _ensure_project_on_path()
    from companion.config import AppConfig

    _file_info(AUDIO_PATH)
    default = AppConfig()

    # A) Путь приложения (фактический бэкенд и модель из AppConfig по умолчанию).
    _run_engine(default, f"A) backend (путь приложения, backend={default.stt_backend})")

    # B) Прямой вызов той же библиотеки с идентичными параметрами.
    _run_mlx_direct(default, "B) mlx_whisper.transcribe напрямую (те же параметры)")

    # C) Референс: faster-whisper base.en (явно выбранный вариант).
    _run_engine(
        AppConfig(stt_backend="faster-whisper"),
        "C) backend (референс: faster-whisper base.en)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
