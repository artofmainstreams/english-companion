#!/usr/bin/env python3
"""Сохранность голоса D (Qwen3-TTS) на новых репликах: два способа синтеза.

Для преподавателя выбран голос из образца ``qwen3_low_D.wav``
(Qwen3-TTS-12Hz-1.7B-VoiceDesign, описание варианта D). Задача — озвучивать
НОВЫЕ ответы тем же узнаваемым голосом. Проверяем два способа:

* ``voice_design`` — модель VoiceDesign + исходное описание голоса D
  (BASE_INSTRUCT + REGISTER_D из ``tools/qwen3_low_register_samples.py``):
  ``model.generate_voice_design(text=..., instruct=..., language=...)``;
* ``reference`` — модель Base (клонирование голоса) с одной ЧИСТОЙ репликой
  D в качестве референса и её точным текстом:
  ``model.generate(text=..., ref_audio=<wav>, ref_text=<транскрипция>)``.

Каждая новая реплика — отдельный вызов модели; на способ — один WAV с
одинаковыми паузами 0.6 c (и по файлу на реплику, для детального сравнения).
Высота — как у модели, без постобработки. Отчёт: время загрузки модели и
синтеза каждой реплики (короткая/длинная), плюс объективный ориентир —
медиана F0 результата против выбранного образца ``qwen3_low_D.wav``
(грубый автокорреляционный замер; финальное суждение — на слух).

Запуск — в экспериментальном окружении (НЕ в .venv проекта; один запуск =
один способ, модель подгружается в каждом процессе отдельно)::

    <env>/bin/python tools/qwen3_voice_consistency_check.py --method voice_design
    <env>/bin/python tools/qwen3_voice_consistency_check.py --method reference

По умолчанию рядом с результатом копируются выбранный образец
``qwen3_low_D.wav`` и чистая реплика-референс ``qwen3_D_ref_phrase.wav``
(создаётся разбивкой ``qwen3_low_D.wav`` по паузам; см. README).
Образцы — в --out (вне Git). Приложение не меняется: переключение TTS
делается отдельным изменением после прослушивания.

Модели (Hugging Face; веса в кэше ~/.cache/huggingface, вне репозитория;
первый запуск скачивает — Base ~4 GiB):
* VoiceDesign: mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
* Base:        mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
"""

from __future__ import annotations

import argparse
import shutil
import time
import wave
from pathlib import Path

import numpy as np

# Описание голоса D — ДОСЛОВНО из tools/qwen3_low_register_samples.py
# (вариант D: BASE_INSTRUCT + REGISTER_D).
BASE_INSTRUCT = (
    "A young adult woman with a natural southern English accent. "
    "Her voice is soft, warm and velvety, with clear diction and "
    "gentle, natural intonation. She speaks at a relaxed conversational pace, "
    "like a patient tutor talking to one student."
)
REGISTER_D = (
    "She speaks in a deep, naturally low alto register, "
    "with a full, warm resonance and a delicate, clear delivery."
)
D_INSTRUCT = BASE_INSTRUCT + " " + REGISTER_D

# Точный текст первой реплики D (для ref_text способа reference).
DEFAULT_REF_TEXT = "Hi! How was your day? Tell me one thing you enjoyed."
DEFAULT_REF_AUDIO = (
    Path("~/Desktop/qwen-voice-low-register/qwen3_D_ref_phrase.wav").expanduser()
)
DEFAULT_SELECTED = (
    Path("~/Desktop/qwen-voice-low-register/qwen3_low_D.wav").expanduser()
)

# Новые реплики (НЕ из исходного набора): короткая, короткая, средняя, длинная.
DEFAULT_PHRASES = [
    "Great job today!",
    "Tell me about your weekend.",
    "Your pronunciation of the past tense is getting clearer.",
    "That was a really good answer. You explained your idea, gave an example, "
    "and kept your rhythm natural.",
]

REPO_VOICE_DESIGN = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
REPO_BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
DEFAULT_OUT = Path("~/Desktop/qwen-voice-low-register").expanduser()
PAUSE_SECONDS = 0.6


def log(msg: str) -> None:
    print(msg, flush=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def join_segments(segments: list[np.ndarray], sample_rate: int,
                  pause_s: float) -> np.ndarray:
    silence = np.zeros(max(1, int(sample_rate * pause_s)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i:
            parts.append(silence)
        parts.append(np.asarray(seg, dtype=np.float32))
    return np.concatenate(parts)


def f0_median(audio: np.ndarray, sr: int, flen: float = 0.064,
              hop: float = 0.016) -> float:
    """Медиана F0 на энергетически сильных кадрах (автокорреляция).

    Грубый ориентир (±несколько Гц): защищает от октавного сброса
    (субгармоника 2L), но точной метрикой тембра не является —
    решение по голосу — на слух.
    """

    def f0_frame(seg: np.ndarray) -> float:
        s = seg - seg.mean()
        c = np.correlate(s, s, mode="full")[len(seg) - 1:]
        lo, hi = int(sr / 450), int(sr / 70)
        lag = int(np.argmax(c[lo:hi])) + lo
        if 2 * lag < hi and c[2 * lag] >= 0.9 * c[lag]:
            lag = 2 * lag
        return sr / lag

    n = int(flen * sr)
    step = int(hop * sr)
    cands: list[float] = []
    for i in range(0, len(audio) - n, step):
        seg = audio[i:i + n]
        if np.sqrt(np.mean(seg**2)) < 0.04:
            continue
        cands.append(f0_frame(seg))
    return float(np.median(cands)) if cands else float("nan")


def synth_phrase_voice_design(model, phrase: str) -> np.ndarray:
    results = list(
        model.generate_voice_design(
            text=phrase, instruct=D_INSTRUCT, language="English"
        )
    )
    audio = np.concatenate(
        [np.asarray(r.audio, dtype=np.float32).ravel() for r in results]
    )
    if not audio.size:
        raise RuntimeError(f"пустое аудио: {phrase!r}")
    return audio


def synth_phrase_reference(model, phrase: str, ref_audio: str,
                           ref_text: str) -> np.ndarray:
    results = list(
        model.generate(
            text=phrase,
            ref_audio=ref_audio,
            ref_text=ref_text,
            lang_code="english",
        )
    )
    audio = np.concatenate(
        [np.asarray(r.audio, dtype=np.float32).ravel() for r in results]
    )
    if not audio.size:
        raise RuntimeError(f"пустое аудио: {phrase!r}")
    return audio


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", required=True,
                    choices=("voice_design", "reference"),
                    help="способ синтеза нового голоса")
    ap.add_argument("phrases", nargs="*", default=None,
                    help="новые реплики (по умолчанию: 4 — короткая…длинная)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="папка результата (вне Git; по умолчанию: %(default)s)")
    ap.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO,
                    help="чистая реплика D для способа reference "
                         "(по умолчанию: %(default)s)")
    ap.add_argument("--ref-text", default=DEFAULT_REF_TEXT,
                    help="точный текст референса (по умолчанию: %(default)s)")
    ap.add_argument("--skip-copy", action="store_true",
                    help="не копировать выбранный образец D и реплику-референс")
    args = ap.parse_args()

    phrases = args.phrases or DEFAULT_PHRASES
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "qwen3_D_vd" if args.method == "voice_design" else "qwen3_D_ref"

    repo = REPO_VOICE_DESIGN if args.method == "voice_design" else REPO_BASE
    if args.method == "reference" and not args.ref_audio.is_file():
        raise SystemExit(f"нет файла референса: {args.ref_audio}")

    t0 = time.perf_counter()
    from mlx_audio.tts.utils import load_model  # лениво: --help без зависимостей
    model = load_model(repo)
    t_load = time.perf_counter() - t0
    sample_rate = int(model.sample_rate)
    log(f"способ: {args.method}; модель: {repo}")
    log(f"загрузка модели: {t_load:.2f} c "
        f"(первый запуск — включая скачивание весов, далее — из кэша)")
    if args.method == "reference":
        log(f"референс: {args.ref_audio.name} | текст: {args.ref_text!r}")
    log(f"sample rate: {sample_rate} Hz; пауза между репликами: {PAUSE_SECONDS} c; "
        "высота — как у модели (без постобработки)")

    segments: list[np.ndarray] = []
    timings: list[float] = []
    for idx, phrase in enumerate(phrases):
        synth = (
            (lambda p: synth_phrase_voice_design(model, p))
            if args.method == "voice_design"
            else (lambda p: synth_phrase_reference(
                model, p, str(args.ref_audio), args.ref_text))
        )
        t_p = time.perf_counter()
        audio = synth(phrase)
        dt = time.perf_counter() - t_p
        segments.append(audio)
        timings.append(dt)
        per = out_dir / f"{prefix}_p{idx + 1}.wav"
        write_wav(per, audio, sample_rate)
        log(f"реплика {idx + 1}/{len(phrases)}: {dt:.2f} c | {phrase[:52]} "
            f"→ {per.name}")

    combined = out_dir / f"{prefix}_new.wav"
    write_wav(combined, join_segments(segments, sample_rate, PAUSE_SECONDS),
              sample_rate)
    log(f"итог: {combined.name} — {sum(timings):.2f} c синтеза на "
        f"{len(phrases)} реплики")

    # Объективный ориентир: медиана F0 результата против образца D.
    out_audio, _ = read_wav(combined)
    try:
        selected_audio, _ = read_wav(DEFAULT_SELECTED)
        f0_new = f0_median(out_audio, sample_rate)
        f0_sel = f0_median(selected_audio, sample_rate)
        log(f"F0_median: новый = {f0_new:.1f} Hz, выбранный D = {f0_sel:.1f} Hz "
            "(грубый замер; решение — на слух)")
    except (OSError, wave.Error):
        log("F0-сравнение: выбранный образец D не найден — пропускаю")

    # Рядом: выбранный образец и реплика-референс (копии, без изменений).
    if not args.skip_copy:
        for ref in (DEFAULT_SELECTED, DEFAULT_REF_AUDIO):
            if ref.is_file():
                dest = out_dir / ref.name
                if not ref.samefile(dest):
                    shutil.copyfile(ref, dest)
                    log(f"референс: {ref.name} скопирован в {out_dir}")
            else:
                log(f"референс не найден, пропускаю: {ref}")

    log(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
