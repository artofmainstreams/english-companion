#!/usr/bin/env python3
"""Qwen3-TTS-12Hz-1.7B-VoiceDesign (MLX) — подбор мягкого британского женского голоса.

Сравнение с Kokoro bf_emma (bf_emma.wav лежит в той же папке). Для двух
описаний голоса (варианты A и B) генерируются ТЕ ЖЕ четыре реплики, что
в tools/kokoro_voice_samples.py — отдельно, по одной на вызов, чтобы
проверить, сохраняются ли тембр и акцент между ответами. Для каждого
варианта — один WAV с короткими паузами между репликами.

Запуск — в том же экспериментальном окружении, что и Kokoro (НЕ в .venv
проекта; дополнительные пакеты не нужны — модуль qwen3_tts входит в
mlx-audio):

    <env>/bin/python tools/qwen3_voice_design_samples.py \
        --out ~/Desktop/kokoro-voices

Отчёт на stdout: загрузка модели (отдельно), синтез каждой реплики,
итог по варианту. Описания голосов можно переопределить флагами
--instruct-a / --instruct-b; фразы — позиционными аргументами;
имена выходных файлов — --prefix (по умолчанию qwen3_vd →
qwen3_vd_A.wav / qwen3_vd_B.wav).

Модель: mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 (Hugging Face;
веса в кэше ~/.cache/huggingface, вне репозитория). Приложение при этом
не меняется — TTS остаётся на macOS say (Samantha) до выбора голоса.
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np

DEFAULT_PHRASES = [
    "Hi! How was your day? Tell me one thing you enjoyed.",
    "Of course. Would you like a table by the window?",
    "Try that again a little more slowly. Take your time.",
    "After class, take a glass of water and walk past the park.",
]
DEFAULT_INSTRUCT_A = (
    "A young adult woman with a natural southern English accent. "
    "A soft, warm timbre, clear diction and a relaxed conversational pace. "
    "She sounds like a patient English tutor speaking to one student."
)
DEFAULT_INSTRUCT_B = (
    "A young adult woman with a natural southern English accent. "
    "A light, gentle voice with a subtle smile and lively but restrained "
    "intonation. She speaks clearly and warmly at an unhurried pace."
)
DEFAULT_REPO_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
PAUSE_SECONDS = 0.6


def log(msg: str) -> None:
    print(msg, flush=True)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def join_segments(segments: list[np.ndarray], sample_rate: int, pause_s: float) -> np.ndarray:
    silence = np.zeros(max(1, int(sample_rate * pause_s)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i:
            parts.append(silence)
        parts.append(np.asarray(seg, dtype=np.float32))
    return np.concatenate(parts)


def synth_variant(
    model,
    sample_rate: int,
    label: str,
    instruct: str,
    phrases: list[str],
    out_path: Path,
) -> None:
    log(f"[{label}] instruct: {instruct[:60]}...")
    segments: list[np.ndarray] = []
    total = 0.0
    for idx, phrase in enumerate(phrases):
        t0 = time.perf_counter()
        results = list(
            model.generate_voice_design(
                text=phrase,
                instruct=instruct,
                language="English",
            )
        )
        dt = time.perf_counter() - t0
        total += dt
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32).ravel() for r in results])
        if not audio.size:
            raise RuntimeError(f"{label}: пустое аудио для «{phrase}»")
        segments.append(audio)
        log(f"[{label}] реплика {idx + 1}/{len(phrases)}: {dt:.2f} c | {phrase[:48]}")
    write_wav(out_path, join_segments(segments, sample_rate, PAUSE_SECONDS), sample_rate)
    log(f"[{label}] итог синтеза: {total:.2f} c; сохранено: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phrases", nargs="*", default=None,
                    help="фразы (по умолчанию — те же 4, что в сравнении Kokoro)")
    ap.add_argument("--instruct-a", default=DEFAULT_INSTRUCT_A,
                    help="описание голоса, вариант A")
    ap.add_argument("--instruct-b", default=DEFAULT_INSTRUCT_B,
                    help="описание голоса, вариант B")
    ap.add_argument("--prefix", default="qwen3_vd",
                    help="префикс имён выходных файлов: <prefix>_A.wav, <prefix>_B.wav "
                         "(по умолчанию: %(default)s)")
    ap.add_argument("--model", default=DEFAULT_REPO_ID,
                    help="идентификатор модели (по умолчанию: %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("~/Desktop/kokoro-voices").expanduser(),
                    help="папка WAV рядом с bf_emma.wav (вне Git; по умолчанию: ~/Desktop/kokoro-voices)")
    args = ap.parse_args()

    phrases = args.phrases or DEFAULT_PHRASES
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    from mlx_audio.tts.utils import load_model  # импорт после argparse: --help без зависимостей
    model = load_model(args.model)
    t_load = time.perf_counter() - t0
    log(f"модель: {args.model}; загрузка: {t_load:.2f} c "
        f"(первый запуск — включая скачивание весов, далее — из кэша)")
    sample_rate = int(model.sample_rate)
    log(f"sample rate: {sample_rate} Hz; пауза между репликами: {PAUSE_SECONDS} c")

    synth_variant(model, sample_rate, "A", args.instruct_a, phrases,
                  out_dir / f"{args.prefix}_A.wav")
    synth_variant(model, sample_rate, "B", args.instruct_b, phrases,
                  out_dir / f"{args.prefix}_B.wav")

    log(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
