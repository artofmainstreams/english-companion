#!/usr/bin/env python
"""Пробный запуск СТРИМИНГА Qwen3 (reference, голос p4): речь частями.

Вопросы:
  * когда приходит ПЕРВЫЙ кусок аудио (time-to-first-sound, TTFS) — насколько
    раньше, чем «весь WAV» (текущее поведение приложения: ждёт полный синтез);
  * воспроизводится ли речь БЕЗ разрывов (underrun: следующий кусок должен
    быть готов ДО того, как закончится воспроизведение предыдущего);
  * как ``streaming_interval`` (размер куска в секундах) влияет на TTFS и
    на разрывы (меньше кусок → быстрее первый звук, но больше кустов).

Всё настоящее: реальная модель Qwen3 (reference, референс p4 из папки
данных приложения), реальное воспроизведение ``sounddevice`` (звук слышен).
Модель грузится ОДИН раз; TTFS меряется ПОСЛЕ загрузки (сравниваем
стриминг, а не загрузку).

Честные оговорки:
* TTFS меряется до момента СТАРТА воспроизведения первого куска; до
  физического звука — ещё несколько десятков мс (буфер вывода, проигрыватель);
* один короткий текст; тембр/естественность и слышимость разрывов —
  проверяешь на слух ты (числа ниже — инструментальные).

Запуск (из корня проекта)::

    .venv/bin/python real_tts_stream_trial.py
"""

import os
import queue
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from companion.config import AppConfig  # noqa: E402

TEXT = "Well done! Great job today."
INTERVALS = [2.0, 1.0, 0.5, 0.3]
SENTINEL = None


def run_full(model, ref, ref_text, sr):
    """Текущее поведение приложения: весь WAV, потом проигрываем."""
    t0 = time.monotonic()
    chunks = []
    for res in model.generate(
        text=TEXT, ref_audio=ref, ref_text=ref_text, lang_code="english"
    ):
        chunks.append(np.asarray(res.audio, dtype=np.float32).ravel())
    synth_t = time.monotonic() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return synth_t, len(audio) / sr


def run_stream(model, ref, ref_text, sr, interval):
    """Стриминг: producer генерирует куски, consumer играет их в реальном
    времени; меряем TTFS и разрывы (underrun)."""
    q: "queue.Queue" = queue.Queue(maxsize=32)
    err: list = []

    def producer():
        try:
            for res in model.generate(
                text=TEXT, ref_audio=ref, ref_text=ref_text, lang_code="english",
                stream=True, streaming_interval=interval,
            ):
                chunk = np.asarray(res.audio, dtype=np.float32).ravel()
                q.put((chunk, float(getattr(res, "real_time_factor", 0.0)),
                       int(getattr(res, "token_count", 0))))
        except Exception as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            q.put(SENTINEL)

    t_start = time.monotonic()
    th = threading.Thread(target=producer, daemon=True)
    th.start()

    first_arrive = None
    prev_end = None
    gaps: list[float] = []
    chunk_rows: list[tuple] = []
    total_samples = 0
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        chunk, rtf, tokens = item
        arrived = time.monotonic()
        if first_arrive is None:
            first_arrive = arrived
        if prev_end is not None and arrived > prev_end:
            gaps.append(arrived - prev_end)
        chunk_rows.append((len(chunk) / sr, rtf, tokens))
        total_samples += len(chunk)
        sd.play(chunk, sr, blocking=True)
        prev_end = time.monotonic()
    th.join(timeout=5)

    total_audio = total_samples / sr
    ttfs = (first_arrive - t_start) if first_arrive else float("nan")
    audible = [g for g in gaps if g > 0.05]
    return {
        "err": err, "ttfs": ttfs, "total_audio": total_audio,
        "n_chunks": len(chunk_rows),
        "gaps": gaps, "audible_gaps": audible,
        "max_gap": max(gaps) if gaps else 0.0,
        "total_gap": sum(gaps),
        "rows": chunk_rows,
        "mean_rtf": (sum(r[1] for r in chunk_rows) / len(chunk_rows)
                     if chunk_rows else 0.0),
    }


def main() -> int:
    config = AppConfig()
    ref = config.tts_qwen3_ref_audio
    if not os.path.exists(ref):
        print(f"ОШИБКА: нет референса: {ref}")
        return 1
    ref_text = config.tts_qwen3_ref_text
    print(f"референс:  {ref}")
    print(f"ref_text:  {ref_text!r}")
    print(f"текст:     {TEXT!r}")
    print(f"интервалы: {INTERVALS}\n")

    from mlx_audio.tts.utils import load_model
    t0 = time.monotonic()
    model = load_model(config.tts_qwen3_model_reference)
    sr = int(model.sample_rate)
    print(f"модель загружена за {time.monotonic() - t0:.1f} c; sr={sr}\n")

    # Прогрев устройства вывода (чтобы первая sd.play не включала время
    # открытия устройства в TTFS).
    sd.play(np.zeros(sr // 20, dtype=np.float32), sr, blocking=True)

    # --- Базовый (текущее поведение: весь WAV) ------------------------------
    full_t, full_dur = run_full(model, ref, ref_text, sr)
    print(f"=== БАЗА (весь WAV, текущее поведение приложения) ===")
    print(f"  синтез (весь файл): {full_t:.2f} c   аудио: {full_dur:.2f} c")
    print(f"  → сейчас первое слово слышно через ~{full_t:.2f} c\n")

    # --- Стриминг ------------------------------------------------------------
    print("=== СТРИМИНГ (речь частями) ===")
    results = {}
    for iv in INTERVALS:
        res = run_stream(model, ref, ref_text, sr, iv)
        if res["err"]:
            print(f"\n  interval={iv}: ОШИБКА {res['err'][0]!r}")
            continue
        results[iv] = res
        print(f"\n  --- streaming_interval = {iv} c ---")
        print(f"  TTFS (до старта 1-го куска): {res['ttfs']:.2f} c")
        print(f"  кусков: {res['n_chunks']}   суммарное аудио: "
              f"{res['total_audio']:.2f} c")
        print(f"  средний real-time-factor куска: {res['mean_rtf']:.2f} "
              f"(>1 = генерация быстрее реального времени → разрывов нет)")
        if res["audible_gaps"]:
            print(f"  РАЗРЫВЫ: {len(res['audible_gaps'])} (сумма "
                  f"{res['total_gap']:.2f} c, макс {res['max_gap']:.2f} c)")
            for k, (dur, rtf, tok) in enumerate(res["rows"][:8], 1):
                print(f"     кусок {k}: {dur:.2f} c аудио, rtf={rtf:.2f}, "
                      f"{tok} ток.")
        else:
            print("  разрывов нет (каждый следующий кусок был готов "
                  "до конца предыдущего)")

    # --- Итог ----------------------------------------------------------------
    print("\n=== ИТОГ ===")
    print(f"  БАЗА (весь WAV):              TTFS ~{full_t:.2f} c")
    best = None
    for iv in INTERVALS:
        if iv in results:
            r = results[iv]
            print(f"  стриминг interval={iv}:          TTFS {r['ttfs']:.2f} c, "
                  f"разрывов {len(r['audible_gaps'])}")
            if best is None or (len(r["audible_gaps"]) == 0 and
                               r["ttfs"] < results[best]["ttfs"]):
                best = iv
    if best is not None:
        r = results[best]
        print(f"\n  Лучший TTFS без разрывов: interval={best} → "
              f"{r['ttfs']:.2f} c (было ~{full_t:.2f} c на весь WAV)")

    print("\nГолос p4 — тембр/естественность и слышимость разрывов "
          "проверяешь на слух ты.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
