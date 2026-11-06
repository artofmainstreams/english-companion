#!/usr/bin/env python3
"""Сравнение голосов TTS: Kokoro-82M (MLX, british G2P) + macOS `say`.

Генерирует по одному WAV на голос: реплики из аргументов (по умолчанию —
4 учебные фразы) склеиваются с короткими паузами. Темп одинаковый для
всех голосов. Для сравнения с текущим TTS приложения можно добавить
образец голоса macOS `say` (по умолчанию Samantha) на тех же фразах.

Запуск — в ОТДЕЛЬНОМ экспериментальном окружении (НЕ в .venv проекта):

    python3.13 -m venv <env>
    <env>/bin/pip install mlx-audio misaki num2words spacy phonemizer soundfile
    <env>/bin/python -m spacy download en_core_web_sm
    brew install espeak-ng
    <env>/bin/python tools/kokoro_voice_samples.py --with-say

Полный список зависимостей эксперимента и примечания — в README
(раздел «Подбор голоса TTS: Kokoro-82M (MLX)»).

Пример (из корня проекта):

    /private/var/folders/hn/d7dw50jn3pb250mptnv1hmjc0000gn/T/opencode/kokoro-exp/venv/bin/python \
        tools/kokoro_voice_samples.py --with-say --out ~/Desktop/kokoro-voices \
        "Своя фраза 1." "Своя фраза 2."

Веса Kokoro-82M скачиваются с Hugging Face (mlx-community/Kokoro-82M-bf16) и
кэшируются вне репозитория; WAV-образцы попадают в --out (вне Git).
Приложение при этом остаётся на Samantha (companion/tts_synth.py) до
окончательного выбора голоса.

Отчёт: время загрузки модели, загрузки голоса, синтез каждой фразы и
повторный (тёплый) синтез первой фразы — отдельно, на stdout.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
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
DEFAULT_VOICES = ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"]
DEFAULT_REPO_ID = "mlx-community/Kokoro-82M-bf16"
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


def synth_voice(
    pipe,
    sample_rate: int,
    voice: str,
    phrases: list[str],
    speed: float,
    out_path: Path,
) -> float:
    t0 = time.perf_counter()
    pipe.load_voice(voice)  # lazy: скачивает/читает pack одного голоса
    t_voice = time.perf_counter() - t0
    log(f"[{voice}] загрузка голоса: {t_voice:.2f} c")

    segments: list[np.ndarray] = []
    for idx, phrase in enumerate(phrases):
        t0 = time.perf_counter()
        results = list(pipe(phrase, voice=voice, speed=speed))
        dt = time.perf_counter() - t0
        kind = "первый синтез" if idx == 0 else "синтез"
        log(f"[{voice}] {kind}: {dt:.2f} c | {phrase[:48]}")
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32).ravel() for r in results])
        if not audio.size:
            raise RuntimeError(f"{voice}: пустое аудио для «{phrase}»")
        segments.append(audio)
        if idx == 0:
            # повторный (тёплый) синтез той же фразы — модель и голос уже в памяти
            t0 = time.perf_counter()
            results = list(pipe(phrase, voice=voice, speed=speed))
            warm = time.perf_counter() - t0
            if not any(np.asarray(r.audio).size for r in results):
                raise RuntimeError(f"{voice}: пустое аудио (повтор) для «{phrase}»")
            log(f"[{voice}] повторный синтез (тёплый): {warm:.2f} c")

    write_wav(out_path, join_segments(segments, sample_rate, PAUSE_SECONDS), sample_rate)
    log(f"[{voice}] сохранено: {out_path}")
    return t_voice


def synth_say(
    say_voice: str,
    phrases: list[str],
    out_path: Path,
) -> None:
    segments: list[np.ndarray] = []
    sample_rate = 0
    import soundfile as sf  # чтение AIFF (aifc удалён в Python 3.13)
    with tempfile.TemporaryDirectory(prefix="kokoro-say-") as tmp:
        for idx, phrase in enumerate(phrases):
            aiff = Path(tmp) / f"say_{idx}.aiff"  # macOS say пишет AIFF (wav не поддерживается)
            t0 = time.perf_counter()
            subprocess.run(
                ["say", "-v", say_voice, "-f", "-", "-o", str(aiff)],
                input=phrase.encode("utf-8"),
                check=True,
            )
            dt = time.perf_counter() - t0
            pcm, sample_rate = sf.read(str(aiff), dtype="float32")
            segments.append(np.asarray(pcm, dtype=np.float32).ravel())
            log(f"[say:{say_voice}] синтез #{idx + 1}: {dt:.2f} c | {phrase[:48]}")
    write_wav(out_path, join_segments(segments, sample_rate, PAUSE_SECONDS), sample_rate)
    log(f"[say:{say_voice}] сохранено: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phrases", nargs="*", default=None,
                    help="фразы для синтеза (по умолчанию — 4 учебные)")
    ap.add_argument("--voices", nargs="*", default=DEFAULT_VOICES,
                    help="голоса Kokoro (по умолчанию: %(default)s)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="одинаковый темп для всех голосов (по умолчанию: %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("~/Desktop/kokoro-voices").expanduser(),
                    help="папка для WAV (вне репозитория; по умолчанию: ~/Desktop/kokoro-voices)")
    ap.add_argument("--model", default=DEFAULT_REPO_ID,
                    help="идентификатор модели (по умолчанию: %(default)s)")
    ap.add_argument("--with-say", action="store_true",
                    help="также сгенерировать образец macOS say для сравнения")
    ap.add_argument("--say-voice", default="Samantha",
                    help="голос say для сравнения (по умолчанию: %(default)s)")
    args = ap.parse_args()

    phrases = args.phrases or DEFAULT_PHRASES
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    from mlx_audio.tts.utils import load  # импорт после argparse: --help без зависимостей
    model = load(args.model)
    t_load = time.perf_counter() - t0
    log(f"модель: {args.model}; загрузка: {t_load:.2f} c "
        f"(первый запуск — включая скачивание весов, далее — из кэша)")

    from mlx_audio.tts.models.kokoro import KokoroPipeline

    # british G2P: lang_code="b" (алиас "en-gb") — британское произношение
    pipe = KokoroPipeline(lang_code="b", model=model, repo_id=args.model)
    sample_rate = int(model.sample_rate)
    log(f"sample rate: {sample_rate} Hz; темп: {args.speed}; пауза между репликами: {PAUSE_SECONDS} c")

    for voice in args.voices:
        synth_voice(pipe, sample_rate, voice, phrases, args.speed, out_dir / f"{voice}.wav")

    if args.with_say:
        synth_say(args.say_voice, phrases, out_dir / f"say_{args.say_voice.lower()}.wav")

    log(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
