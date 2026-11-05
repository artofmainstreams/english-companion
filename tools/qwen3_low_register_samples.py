#!/usr/bin/env python3
"""Qwen3-TTS-12Hz-1.7B-VoiceDesign (MLX) — четыре низких «бархатных» женских голоса.

Эксперимент по подбору голоса преподавателя: мягкое, низкое, бархатное
звучание. Четыре варианта (A–D) = одно общее описание + разное описание
регистра. Все варианты — одна модель, одно окружение, одинаковые настройки
генерации. Высота сохраняется КАК ВЫДАЛА МОДЕЛЬ (без постобработки),
громкость партии выравнивается по RMS.

Для каждого варианта генерируются ОТДЕЛЬНО четыре реплики (по одной на
вызов generate_voice_design) — чтобы оценить сохранность голоса
(тембр, регистр, узнаваемость) между ответами. На вариант — один WAV
с одинаковыми короткими паузами между репликами.

Запуск — в экспериментальном окружении (НЕ в .venv проекта; доп. пакеты
не нужны — модуль qwen3_tts входит в mlx-audio):

    <env>/bin/python tools/qwen3_low_register_samples.py \
        --out ~/Desktop/qwen-voice-low-register

По умолчанию рядом с образцами копируется исходный Qwen-образец
qwen3_vd_A.wav (для сравнения); отключить: --skip-reference,
свой: --reference /путь/файл.wav (можно несколько).

Модель: mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 (Hugging Face;
веса в кэше ~/.cache/huggingface, вне репозитория). Образцы — в --out
(вне Git). Приложение при этом не меняется — TTS остаётся на macOS say
(Samantha) до выбора голоса.
"""

from __future__ import annotations

import argparse
import shutil
import time
import wave
from pathlib import Path

import numpy as np

BASE_INSTRUCT = (
    "A young adult woman with a natural southern English accent. "
    "Her voice is soft, warm and velvety, with clear diction and "
    "gentle, natural intonation. She speaks at a relaxed conversational pace, "
    "like a patient tutor talking to one student."
)
REGISTER_A = ("Her speaking pitch sits in the lower-middle register, "
              "with a light, rounded resonance.")
REGISTER_B = ("Her speaking voice has a distinctly low pitch, "
              "with a soft, rounded resonance and a clear, easy delivery.")
REGISTER_C = ("She has a low alto speaking voice, with a rich, velvety "
              "resonance and gentle rises and falls in pitch.")
REGISTER_D = ("She speaks in a deep, naturally low alto register, "
              "with a full, warm resonance and a delicate, clear delivery.")
VARIANTS: dict[str, str] = {
    "A": BASE_INSTRUCT + " " + REGISTER_A,
    "B": BASE_INSTRUCT + " " + REGISTER_B,
    "C": BASE_INSTRUCT + " " + REGISTER_C,
    "D": BASE_INSTRUCT + " " + REGISTER_D,
}
DEFAULT_PHRASES = [
    "Hi! How was your day? Tell me one thing you enjoyed.",
    "Of course. Would you like a table by the window?",
    "Try that again a little more slowly. Take your time.",
    "After class, take a glass of water and walk past the park.",
]
DEFAULT_REPO_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DEFAULT_OUT = Path("~/Desktop/qwen-voice-low-register").expanduser()
DEFAULT_REFERENCE = Path("~/Desktop/kokoro-voices/qwen3_vd_A.wav").expanduser()
PAUSE_SECONDS = 0.6


def log(msg: str) -> None:
    print(msg, flush=True)


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


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def join_segments(segments: list[np.ndarray], sample_rate: int, pause_s: float) -> np.ndarray:
    silence = np.zeros(max(1, int(sample_rate * pause_s)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i:
            parts.append(silence)
        parts.append(np.asarray(seg, dtype=np.float32))
    return np.concatenate(parts)


def synth_variant(model, sample_rate: int, name: str, instruct: str,
                  phrases: list[str], out_path: Path) -> float:
    log(f"[{name}] описание: {instruct[:64]}...")
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
            raise RuntimeError(f"{name}: пустое аудио для «{phrase}»")
        segments.append(audio)
        log(f"[{name}] реплика {idx + 1}/{len(phrases)}: {dt:.2f} c | {phrase[:48]}")
    write_wav(out_path, join_segments(segments, sample_rate, PAUSE_SECONDS), sample_rate)
    log(f"[{name}] синтез: {total:.2f} c; сохранено: {out_path}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phrases", nargs="*", default=None,
                    help="реплики (по умолчанию — те же 4, что в первом сравнении)")
    ap.add_argument("--model", default=DEFAULT_REPO_ID,
                    help="идентификатор модели (по умолчанию: %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="папка результата (вне Git; по умолчанию: %(default)s)")
    ap.add_argument("--reference", type=Path, action="append", default=None,
                    help="исходный Qwen-образец для сравнения — скопировать рядом "
                         "(можно несколько; по умолчанию: %(default)s)")
    ap.add_argument("--skip-reference", action="store_true",
                    help="не копировать исходный образец")
    args = ap.parse_args()

    phrases = args.phrases or DEFAULT_PHRASES
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    references = args.reference if args.reference else [DEFAULT_REFERENCE]

    t0 = time.perf_counter()
    from mlx_audio.tts.utils import load_model  # импорт после argparse: --help без зависимостей
    model = load_model(args.model)
    t_load = time.perf_counter() - t0
    log(f"модель: {args.model}; загрузка: {t_load:.2f} c "
        f"(первый запуск — включая скачивание весов, далее — из кэша)")
    sample_rate = int(model.sample_rate)
    log(f"sample rate: {sample_rate} Hz; пауза между репликами: {PAUSE_SECONDS} c; "
        "высота — как у модели (без постобработки)")

    # 1) синтез четырёх вариантов: по одной реплике на вызов
    for name, instruct in VARIANTS.items():
        synth_variant(model, sample_rate, name, instruct, phrases,
                      out_dir / f"qwen3_low_{name}.wav")

    # 2) выравнивание громкости партии: RMS ко самому громкому, пик ≤ 0.98
    files = [out_dir / f"qwen3_low_{n}.wav" for n in VARIANTS]
    audios = [read_wav(f) for f in files]
    rms = [float(np.sqrt(np.mean(a**2))) for a in audios]
    target = max(rms)
    for f, a, before in zip(files, audios, rms):
        gain = target / before if before > 1e-9 else 1.0
        after = before * gain
        write_wav(f, a * gain, sample_rate)
        peak = float(np.max(np.abs(np.clip(a * gain, -1.0, 1.0))))
        log(f"{f.name:20s} RMS {20*np.log10(before+1e-12):6.1f} → {20*np.log10(after+1e-12):6.1f} dBFS  пик {peak:.3f}")

    # 3) исходный Qwen-образец рядом для сравнения (копировать не меняя)
    if not args.skip_reference:
        for ref in references:
            if ref.is_file():
                shutil.copyfile(ref, out_dir / ref.name)
                log(f"референс: {ref.name} скопирован в {out_dir}")
            elif not args.reference:
                log(f"референс не найден, пропускаю: {ref}")

    log(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
