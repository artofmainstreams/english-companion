#!/usr/bin/env python
"""Сравнение качества/скорости распознавания на ОДНОЙ и той же записи пользователя.

Три реализации на идентичном входе:
  * base.en          — faster-whisper (текущая реализация, Systran/faster-whisper-base.en)
  * large-v3         — mlx-whisper, mlx-community/whisper-large-v3-mlx   (полная точность)
  * large-v3-turbo   — mlx-whisper, mlx-community/whisper-large-v3-turbo (полная точность)

Правила:
  * language=en, task=transcribe;
  * без LLM-правки и без подсказки ожидаемой фразы (initial_prompt не задаём);
  * без агрессивной квантизации (полные версии весов);
  * каждая модель — в ОТДЕЛЬНОМ процессе → чистый замер пиковой RAM;
  * отдельно: загрузка модели, первое распознавание, повторное (модель уже в памяти);
  * полные расшифровки, времена, пиковая RAM, идентификаторы и размер весов.

Запуск (из корня проекта):
  # сравнение всех трёх на одной (или нескольких) записях:
  .venv/bin/python tools/mlx_stt_compare.py /путь/recording_1.m4a
  # одна модель (используется и compare-режимом как отдельный процесс):
  .venv/bin/python tools/mlx_stt_compare.py --single large-v3 /путь/recording_1.m4a
"""

import contextlib
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time

_ALL_MODELS = ["base", "large-v3", "large-v3-turbo"]


def _ensure_project_on_path() -> None:
    # Скрипт лежит в tools/; корень проекта — родительская директория.
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tools_dir)
    if root not in sys.path:
        sys.path.insert(0, root)


def _peak_rss_gb() -> float:
    """Пиковая RSS процесса (модель + рантайм). macOS: байты; Linux: КБ."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 ** 3)
    return (rss * 1024) / (1024 ** 3)


def _total_ram_gb() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
    ).stdout.strip()
    try:
        return int(out) / (1024 ** 3)
    except ValueError:
        return -1.0


def _find_weights(repo_dir_name: str):
    """Суммарный размер весов и список файлов в кэше HuggingFace."""
    import huggingface_hub

    base = os.path.join(
        huggingface_hub.constants.HF_HUB_CACHE,
        "models--" + repo_dir_name.replace("/", "--"),
    )
    total = 0
    files = []
    if os.path.isdir(base):
        for dirpath, _dirs, names in os.walk(base):
            for name in names:
                if name.endswith((".npz", ".safetensors", ".bin")):
                    p = os.path.join(dirpath, name)
                    size = os.path.getsize(p)
                    if size:
                        total += size
                        files.append(f"{os.path.relpath(p, base)} ({size/1024**3:.2f} GB)")
    return total, sorted(files)


def _repo_for(model_key: str) -> str:
    return {
        "base": "Systran/faster-whisper-base.en",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }[model_key]


def _make_transcriber(model_key: str):
    """Возвращает (load_s, transcribe_fn). transcribe_fn(path) -> str."""
    quiet = contextlib.redirect_stdout(sys.stderr)

    if model_key == "base":
        from companion.config import AppConfig
        from companion.stt_engine import SttEngine

        # Явно faster-whisper: дефолт AppConfig — уже mlx-бэкенд.
        eng = SttEngine(AppConfig(stt_backend="faster-whisper"))
        t0 = time.perf_counter()
        eng._ensure_model()
        load_s = time.perf_counter() - t0
        model = eng._ensure_model()

        def transcribe(path: str) -> str:
            with quiet:
                return eng._run_transcribe(model, path).strip()
        return load_s, transcribe

    # MLX: ModelHolder кэширует модель по пути → отдельный замер загрузки и
    # повторное распознавание переиспользуют уже загруженную модель.
    repo = _repo_for(model_key)
    import mlx_whisper
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    t0 = time.perf_counter()
    # fp16 (стандартная точность whisper, НЕ квантизация) — совпадает с
    # дефолтом transcribe(fp16=True), поэтому кэш совпадёт.
    ModelHolder.get_model(repo, mx.float16)
    load_s = time.perf_counter() - t0

    def transcribe(path: str) -> str:
        with quiet:
            res = mlx_whisper.transcribe(
                path, path_or_hf_repo=repo, language="en", task="transcribe"
            )
            return str(res["text"]).strip()
    return load_s, transcribe


def run_single(model_key: str, recordings: list[str]) -> dict:
    """Замеряет одну модель в этом процессе; возвращает словарь-результат."""
    load_s, transcribe = _make_transcriber(model_key)

    texts = {}
    first_s = 0.0
    for i, rec in enumerate(recordings):
        t0 = time.perf_counter()
        texts[rec] = transcribe(rec)
        if i == 0:
            first_s = time.perf_counter() - t0

    # Повторное распознавание (модель уже загружена) — steady-state задержка.
    t0 = time.perf_counter()
    transcribe(recordings[0])
    repeat_s = time.perf_counter() - t0

    repo = _repo_for(model_key)
    weight_bytes, weight_files = _find_weights(repo)
    return {
        "model": model_key,
        "repo": repo,
        "load_s": round(load_s, 3),
        "first_transcribe_s": round(first_s, 3),
        "repeat_transcribe_s": round(repeat_s, 3),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "weight_gb": round(weight_bytes / 1024 ** 3, 3),
        "weight_files": weight_files,
        "texts": {os.path.basename(k): v for k, v in texts.items()},
    }


def _fmt(seconds: float) -> str:
    return f"{seconds:.2f} с"


def _print_table(results: list[dict], recordings: list[str]) -> None:
    names = [os.path.basename(r) for r in recordings]
    label = {r: names[i] for i, r in enumerate(recordings)}

    print("=" * 78)
    print("СРАВНЕНИЕ  (вход: одна и та же запись; language=en, task=transcribe)")
    print("Входные файлы: " + ", ".join(names))
    print("=" * 78)

    # Расшифровки построчно (качество)
    for rec in names:
        print(f"\n— {rec} —")
        for res in results:
            text = res["texts"].get(rec, "")
            print(f"   {res['model']:14s}: {text!r}")

    # Время + память (производительность)
    print("\n" + "-" * 78)
    print(f"{'модель':16s} {'загрузка':>10s} {'1-й раз':>10s} {'повтор':>10s} "
          f"{'пик RAM':>10s} {'веса':>9s}")
    print("-" * 78)
    for res in results:
        print(
            f"{res['model']:16s} "
            f"{_fmt(res['load_s']):>10s} "
            f"{_fmt(res['first_transcribe_s']):>10s} "
            f"{_fmt(res['repeat_transcribe_s']):>10s} "
            f"{res['peak_rss_gb']:>8.2f}ГБ "
            f"{res['weight_gb']:>7.2f}ГБ"
        )
    print("-" * 78)

    # Идентификаторы весов + версии
    print("\nТочные идентификаторы весов:")
    for res in results:
        print(f"  {res['model']:14s}: {res['repo']}  ({res['weight_gb']:.2f} GB)")
        for f in res["weight_files"]:
            print(f"                  ↳ {f}")
    print()


def _single_main(argv: list[str]) -> int:
    _ensure_project_on_path()
    model_key = argv[0]
    recordings = argv[1:]
    if not recordings:
        print("нет файла записи", file=sys.stderr)
        return 2
    logging.disable(logging.CRITICAL)
    result = run_single(model_key, recordings)
    # На stdout — только JSON (лог/прогресс ушли в stderr через redirect).
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _compare_main(argv: list[str]) -> int:
    _ensure_project_on_path()
    recordings = [os.path.abspath(a) for a in argv if os.path.exists(a)]
    if not recordings:
        print("Не указано валидное аудиофайл.", file=sys.stderr)
        return 2

    print(f"Система: {platform.platform()} · RAM всего: {_total_ram_gb():.0f} ГБ")
    print(f"ffmpeg CLI: {_which_ffmpeg()}")

    py = sys.executable
    script = os.path.abspath(__file__)
    results = []
    for model_key in _ALL_MODELS:
        print(f"\n>>> запускаю {model_key} (отдельный процесс)…", file=sys.stderr)
        proc = subprocess.run(
            [py, script, "--single", model_key, *recordings],
            capture_output=True, text=True,
        )
        # stdout должен заканчиваться JSON-строкой; берём её.
        out = proc.stdout.strip()
        line = None
        for ln in out.splitlines():
            ln = ln.strip()
            if ln.startswith("{") and ln.endswith("}"):
                line = ln
        if line is None:
            print(f"!! {model_key}: не удалось получить результат", file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            continue
        results.append(json.loads(line))

    if not results:
        print("Ни одна модель не запустилась.", file=sys.stderr)
        return 1
    _print_table(results, recordings)
    return 0


def _which_ffmpeg() -> str:
    path = subprocess.run(
        ["sh", "-c", "command -v ffmpeg"], capture_output=True, text=True
    ).stdout.strip()
    if not path:
        return "НЕ НАЙДЕН (mlx-whisper не сможет декодировать аудио)"
    ver = subprocess.run([path, "-version"], capture_output=True, text=True).stdout.splitlines()
    return f"{path} — {ver[0] if ver else ''}"


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--single":
        return _single_main(argv[1:])
    return _compare_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
