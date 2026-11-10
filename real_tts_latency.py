#!/usr/bin/env python
"""Замер задержки «короткий ответ» по этапам — ОДИН запуск приложения.

Цель: найти, где живая задержка до первого слышимого слова. Замеряет
ОТДЕЛЬНО 6 этапов каждого из 5 коротких циклов:

  1. STT           — реальное распознавание (mlx-whisper), «запись» студента
                     заменяет готовый английский аудио macOS ``say`` (Сэмента);
  2. ожидание      — имитация «ручной отправки» (фиксированная пауза — это
                     пользовательское действие, не вычисление; помечено честно);
  3. Ollama        — реальный запрос (QNetworkAccessManager) к модели;
  4. загрузка TTS  — ленивая загрузка модели Qwen3 (только в ПЕРВОМ цикле);
  5. синтез TTS    — синтез ГОЛОСОМ p4 (reference) ОДИНАКОВОГО короткого
                     текста в каждом цикле → первый вызов = загрузка + синтез,
                     повторные = синтез на загруженной модели (сравнимо);
  6. запуск плеера — от готовности WAV до входа плеера в PLAYING.

Честные оговорки (это диагностический запуск, а не живая сессия):
* «речь студента» — готовый аудио ``say``, не запись с микрофона;
* «ожидание отправки» — имитация паузы пользователя (фиксированная);
* TTS-этап синтезирует ОДИНАКОВЫЙ короткий текст (не точный ответ Ollama),
  чтобы загрузка vs синтез были сравниваемыми; ответ Ollama всё равно
  измеряется на своём этапе.

Запуск (из корня проекта; нужен туннель Ollama)::

    .venv/bin/python real_tts_latency.py
"""

import os
import subprocess
import sys
import tempfile
import time
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Скрипт лежит в корне; гарантируем импорт пакета ``companion``.
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.audio_controller import AudioController, build_tts  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.message import Message  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.stt_backends import build_backend  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402

# Короткие реплики студента (разные → реалистичный STT/Ollama).
STUDENT_TEXTS = [
    "Hi, how are you?",
    "I like apples.",
    "What is your name?",
    "Thank you!",
    "I went to the park yesterday.",
]

# ОДИНАКОВЫЙ короткий текст для TTS-этапа (чтобы первый вызов
# = загрузка + синтез, а повторные = синтез на загруженной модели).
TTS_SHORT = "Well done! Great job today."

# Имитация «ручной отправки» (пауза пользователя). Секунды.
SEND_DELAY_S = 1.5


def _wait(predicate, timeout: float, app: "QApplication") -> bool:
    """Ждём условие, качая цикл событий (доставка межпотоковых сигналов)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _pause(seconds: float, app: "QApplication") -> None:
    """Имитация паузы пользователя (кнопка «Отправить»), с качанием событий."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


def _wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _make_speech(dest_dir: str, sentence: str, name: str) -> str:
    """Готовый английский аудио macOS ``say`` (замена записи студента)."""
    path = os.path.join(dest_dir, name)
    subprocess.run(["say", "-v", "Samantha", "-o", path, sentence], check=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("say не создал файл речи")
    return path


def main() -> int:
    app = QApplication(sys.argv)
    config = AppConfig()

    print(f"Ollama: {config.ollama_base_url}")
    print(f"TTS: {config.tts_engine}/{config.tts_qwen3_method}, "
          f"референс: {config.tts_qwen3_ref_audio}")
    if not os.path.exists(config.tts_qwen3_ref_audio):
        print("ОШИБКА: нет референса. Восстанови: "
              ".venv/bin/python tools/install_tts_reference.py")
        return 1
    print(f"STT: {config.stt_backend} / {config.stt_mlx_model}")
    print(f"TTS-текст (одинаковый для всех циклов): {TTS_SHORT!r}\n")

    client = OllamaClient(config, app)
    audio = AudioController(parent=app, tts=build_tts(config))
    stt = SttEngine(build_backend(config), app)

    ready_paths: list[str] = []
    failed: list[str] = []
    audio.tts.reply_ready.connect(ready_paths.append)
    audio.tts.failed.connect(failed.append)
    chat: list[str] = []
    chat_err: list[str] = []
    client.chat_completed.connect(chat.append)
    client.chat_failed.connect(chat_err.append)
    models: list[str] = []
    model_err: list[str] = []
    client.models_loaded.connect(models.extend)
    client.models_failed.connect(model_err.append)

    # --- Модель Ollama -------------------------------------------------------
    client.load_models()
    if not _wait(lambda: models or model_err, 15.0, app):
        print("ОШИБКА: Ollama не ответила за 15 c")
        return 1
    if model_err:
        print(f"ОШИБКА: {model_err[0]}")
        return 1
    model = next((m for m in models if "qwen3" in m), models[0])
    print(f"модель: {model}\n")

    # --- Файлы «записи» (замена) + STT-прогрев -------------------------------
    tmp = tempfile.mkdtemp(prefix="english-companion-lat-")
    warm_audio = _make_speech(tmp, "hello", "warm.aiff")
    student_audio = [
        _make_speech(tmp, t, f"st{i}.aiff") for i, t in enumerate(STUDENT_TEXTS)
    ]

    t0 = time.monotonic()
    warm_text = stt.recognize_sync(warm_audio)
    stt_warm = time.monotonic() - t0
    print(f"STT прогрев (вкл. загрузку модели): {stt_warm:.2f} c → {warm_text!r}")

    history: list[Message] = []
    rows: list[dict] = []

    # --- 5 циклов ------------------------------------------------------------
    for i, (student, spath) in enumerate(zip(STUDENT_TEXTS, student_audio), 1):
        print(f"\n=== Цикл {i}/5 ===")
        print(f"студент (замена): {student!r}")

        # 1) STT (модель уже загружена после прогрева).
        t0 = time.monotonic()
        text = stt.recognize_sync(spath)
        stt_t = time.monotonic() - t0
        print(f"  1. STT            {stt_t:6.2f} c   распознано: {text!r}")
        if not text.strip():
            print("  ВНИМАНИЕ: STT вернул пустой текст (цикл продолжается "
                  "с исходной фразой)")
            text = student

        # 2) Ожидание ручной отправки (имитация паузы пользователя).
        t0 = time.monotonic()
        _pause(SEND_DELAY_S, app)
        send_t = time.monotonic() - t0
        print(f"  2. ожидание       {send_t:6.2f} c   (имитация ручной отправки)")

        # 3) Ollama (реальный запрос).
        t0 = time.monotonic()
        client.send_chat(model, history + [Message(role="user", content=text)])
        if not _wait(lambda: chat or chat_err, config.chat_timeout_ms / 1000, app):
            print("ОШИБКА: ответ Ollama не пришёл")
            return 1
        if chat_err:
            print(f"ОШИБКА: {chat_err.pop()}")
            return 1
        reply = chat.pop()
        ollama_t = time.monotonic() - t0
        print(f"  3. Ollama         {ollama_t:6.2f} c   "
              f"ответ {len(reply)} симв.: {reply[:80]!r}…")

        # 4+5) TTS (одинаковый короткий текст): первый = загрузка + синтез,
        #      повторные = синтез на загруженной модели.
        t_text = time.monotonic()
        n_ready = len(ready_paths)
        audio.speak_reply(TTS_SHORT)
        if not _wait(lambda: len(ready_paths) > n_ready or failed, 240.0, app):
            print("ОШИБКА: синтез не завершился за 240 c")
            return 1
        if failed:
            print(f"ОШИБКА синтеза: {failed.pop()}")
            return 1
        t_ready = time.monotonic()
        tts_t = t_ready - t_text
        load_note = "  (включая загрузку модели Qwen3)" if i == 1 else ""
        print(f"  4+5. TTS          {tts_t:6.2f} c{load_note}")

        # 6) Запуск плеера (готовность → PLAYING).
        if not _wait(lambda: audio.state == AudioController.PLAYING, 15.0, app):
            print("ОШИБКА: автозапуск не начал воспроизведение")
            return 1
        t_play = time.monotonic()
        player_t = t_play - t_ready
        print(f"  6. запуск плеера {player_t:6.2f} c")

        # Дождаться конца воспроизведения (звук должен быть слышен).
        if not _wait(lambda: audio.state == AudioController.IDLE, 120.0, app):
            print("ОШИБКА: воспроизведение не завершилось за 120 c")
            return 1
        path = ready_paths[-1]
        dur = _wav_seconds(path)

        total = stt_t + send_t + ollama_t + tts_t + player_t
        rows.append({
            "i": i, "stt": stt_t, "send": send_t, "ollama": ollama_t,
            "tts": tts_t, "player": player_t, "dur": dur, "total": total,
        })
        print(f"  ИТОГО цикл: {total:.2f} c   (аудио TTS {dur:.1f} c)")

        history += [
            Message(role="user", content=text),
            Message(role="assistant", content=reply),
        ]

    # --- Итог ----------------------------------------------------------------
    print("\n=== Итог (5 коротких циклов, один запуск) ===")

    stt_loaded = [r["stt"] for r in rows]
    stt_loaded_avg = sum(stt_loaded) / len(stt_loaded)
    stt_load = max(stt_warm - stt_loaded_avg, 0.0)

    oll_avg = sum(r["ollama"] for r in rows) / len(rows)
    first = rows[0]
    rest = rows[1:]
    tts_rest_avg = sum(r["tts"] for r in rest) / len(rest)
    tts_load = max(first["tts"] - tts_rest_avg, 0.0)
    player_avg = sum(r["player"] for r in rows) / len(rows)

    print(f"  STT:               загрузка ~{stt_load:.1f} c "
          f"(разово) | распознавание в среднем {stt_loaded_avg:.2f} c")
    print(f"  ожидание отправки: ~{SEND_DELAY_S:.1f} c (имитация, "
          f"действие пользователя)")
    print(f"  Ollama:            в среднем {oll_avg:.2f} c "
          f"(мин {min(r['ollama'] for r in rows):.2f} / "
          f"макс {max(r['ollama'] for r in rows):.2f})")
    print(f"  TTS:               загрузка ~{tts_load:.1f} c (разово, первый "
          f"цикл) | синтез короткого текста в среднем {tts_rest_avg:.2f} c "
          f"({first['dur']:.1f} c аудио)")
    print(f"  запуск плеера:     в среднем {player_avg:.2f} c")

    print(f"\n  ПЕРВЫЙ ответ (включая загрузку TTS): цикл {first['total']:.2f} c "
          f"(STT {first['stt']:.2f} + отправка {first['send']:.2f} "
          f"+ Ollama {first['ollama']:.2f} + TTS {first['tts']:.2f} "
          f"+ плеер {first['player']:.2f})")
    rest_total_avg = sum(r["total"] for r in rest) / len(rest)
    print(f"  ПОСЛЕДУЮЩИЕ (модель загружена):    в среднем {rest_total_avg:.2f} c "
          f"(мин {min(r['total'] for r in rest):.2f} / "
          f"макс {max(r['total'] for r in rest):.2f})")

    print("\nГде основная задержка короткого ответа (в среднем, модель "
          "загружена):")
    parts = {
        "STT (распознавание)": stt_loaded_avg,
        "ожидание отправки (имитация)": SEND_DELAY_S,
        "Ollama": oll_avg,
        f"TTS (синтез {first['dur']:.1f} c аудио)": tts_rest_avg,
        "запуск плеера": player_avg,
    }
    order = sorted(parts.items(), key=lambda kv: -kv[1])
    for name, val in order:
        print(f"   {val:6.2f} c   {name}")

    # Уборка своих файлов и корректное закрытие.
    audio.shutdown()
    stt.shutdown()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nГотово. Голос p4 — тембр/акцент проверяешь на слух ты.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
