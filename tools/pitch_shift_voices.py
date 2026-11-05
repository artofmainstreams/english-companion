#!/usr/bin/env python3
"""Понижение высоты голосов с сохранением темпа, пауз и формантов (Rubber Band).

Для каждого входного WAV делает версии: исходная + высота ниже на заданное
число полутонов (по умолчанию 0, −1, −2). Обработка существующих записей —
без повторного синтеза.

Метод:
- Rubber Band, движок R3 (−3, более точный);
- −F — включено сохранение формантов (formant preservation), чтобы
  понижение не делало голос глухим;
- −t 1 — длительность сохраняется точно (темп и паузы без изменений),
  контур интонации смещается равномерно на заданные полутона (относительные
  отношения частот сохраняются);
- громкость партии выравнивается по RMS к самой громкой версии
  (линейное усиление, пик ≤ 0.98);
- БЕЗ эквализации и компрессии (это отдельная проба).

Требуется: `rubberband` в PATH (macOS: `brew install rubberband`) и
numpy (есть в экспериментальном окружении Kokoro/Qwen3). Запуск:

    <env>/bin/python tools/pitch_shift_voices.py \
        ~/Desktop/kokoro-voices/qwen3_vd_A.wav ~/Desktop/kokoro-voices/qwen3_vd_B.wav \
        --outdir ~/Desktop/qwen-voice-comparison

Или с другими полутонами/входными файлами:

    <env>/bin/python tools/pitch_shift_voices.py в1.wav в2.wav \
        --semitones 0,-1,-2 --outdir ~/Desktop/qwen-voice-comparison

Вывод: <база>_orig.wav, <база>_m1.wav, <база>_m2.wav (mN = ниже на N
полутонов) + отчёт (длительность, RMS до/после, пик) на stdout.
Файлы пишутся в --outdir (вне репозитория, не попадают в Git).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

DEFAULT_SEMITONES = [0.0, -1.0, -2.0]


def log(msg: str) -> None:
    print(msg, flush=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise RuntimeError(f"{path}: нужен mono 16-bit WAV")
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


def suffix_for(shift: float) -> str:
    if shift == 0:
        return "orig"
    return f"m{abs(shift):g}"


def shift_pitch(inp: Path, outp: Path, semitones: float) -> None:
    cmd = [
        "rubberband",
        "-3",      # R3 (finer) engine
        "-F",      # formant preservation (сохранение формантов)
        "-p", f"{semitones:g}",
        "-t", "1", # длительность точно как у исходника (темп и паузы)
        str(inp), str(outp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rubberband завершился с ошибкой:\n{proc.stderr.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="входные mono 16-bit WAV (например qwen3_vd_A.wav qwen3_vd_B.wav)")
    ap.add_argument("--semitones", default="0,-1,-2",
                    help="версии: список полутонов через запятую (по умолчанию: %(default)s)")
    ap.add_argument("--outdir", type=Path,
                    default=Path("~/Desktop/qwen-voice-comparison").expanduser(),
                    help="папка результата (вне Git; по умолчанию: %(default)s)")
    args = ap.parse_args()

    if shutil.which("rubberband") is None:
        raise SystemExit("не найден `rubberband` в PATH (macOS: brew install rubberband)")

    shifts = [float(x) for x in args.semitones.split(",") if x.strip()]
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) версии: orig — копия; остальные — Rubber Band R3, −F, −t 1
    outputs: list[tuple[Path, Path, float, np.ndarray, int]] = []  # (in, out, shift, audio, sr)
    for inp in args.inputs:
        if not inp.is_file():
            raise SystemExit(f"нет файла: {inp}")
        base = inp.stem
        src, sr = read_wav(inp)
        for sh in shifts:
            outp = outdir / f"{base}_{suffix_for(sh)}.wav"
            if sh == 0:
                shutil.copyfile(inp, outp)
            else:
                shift_pitch(inp, outp, sh)
            audio, out_sr = read_wav(outp)
            if out_sr != sr or len(audio) != len(src):
                raise SystemExit(
                    f"{outp.name}: длительность/rate изменились "
                    f"({len(src)} → {len(audio)} сэмплов) — это не ожидалось при −t 1"
                )
            outputs.append((inp, outp, sh, audio, sr))

    # 2) выравнивание громкости партии: RMS ко самому громкому, пик ≤ 0.98
    all_rms = [float(np.sqrt(np.mean(a**2))) for (_, _, _, a, _) in outputs]
    target = max(all_rms)
    for (inp, outp, sh, audio, sr), before in zip(outputs, all_rms):
        gain = target / before if before > 1e-9 else 1.0
        after = before * gain
        write_wav(outp, audio * gain, sr)
        peak = float(np.max(np.abs(np.clip(audio * gain, -1.0, 1.0))))
        log(f"{outp.name:32s} {sh:+g} полутон(а)  {len(audio)/sr:7.3f} s  "
            f"RMS {20*np.log10(before+1e-12):6.1f} → {20*np.log10(after+1e-12):6.1f} dBFS  пик {peak:.3f}")

    log(f"готово: {outdir}")
    log("метод: Rubber Band R3 (−3), −F (сохранение формантов), −t 1 (длительность/паузы); "
        "выравнивание по RMS без EQ и компрессии")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
