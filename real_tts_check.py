#!/usr/bin/env python
"""Реальная многоцикловая проверка: запись → STT → отправка → ответ голосом → повтор.

Отличается от оффлайн-тестов: всё НАСТОЯЩЕЕ в одном запуске процесса:

* STT — реальный бэкенд из AppConfig (mlx-whisper large-v3, модель из кэша);
* Ollama — реальный запрос через ``OllamaClient`` (QNetworkAccessManager),
  модель выбирается из /api/tags;
* TTS — реальный синтезатор из ``build_tts(AppConfig())``: Qwen3-TTS
  reference-режим (Base-модель + референс p4 из папки данных приложения);
* воспроизведение — реальный ``AudioPlayer`` (звук слышен — прослушай).

Честные оговорки (это проверка, а не живая сессия):
* «запись» — готовый WAV/AIFF английский речи (по умолчанию генерируется
  macOS ``say``; можно передать свой файл через ``--audio``),
  не запись с микрофона;
* текст отправляется как есть (в приложении — та же механика: кнопка
  «Отправить»);
* время «текст → звук» меряется от момента, когда ответ Ollama получен
  (в приложении в этот момент он появляется в чате), до входа плеера в
  PLAYING.

Замеряет ОТДЕЛЬНО для первого ответа (в синтез входит загрузка модели)
и для последующего; проверяет «Повторить ответ» (тот же файл, без нового
синтеза) и закрытие ВО ВРЕМЯ синтеза (без зависаний и «оживания»).

Запуск (из корня проекта; нужен туннель Ollama)::

    .venv/bin/python real_tts_check.py
    .venv/bin/python real_tts_check.py --audio ~/Desktop/запись.m4a
    .venv/bin/python real_tts_check.py --cycles 3
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

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

# «Речь студента» для замены записи (заменяется файлом через --audio).
SENTENCE = (
    "Hi! How are you today? I finished my homework, "
    "and I want to tell you about my weekend."
)

# Длинный текст для проверки закрытия во время синтеза.
SHUTDOWN_TEXT = (
    "Great work today! Let me summarise what you did well. "
    "First, your pronunciation was much clearer than last time, "
    "especially the word 'schedule' and the difference between "
    "'ship' and 'sheep'. Second, you answered in full sentences, "
    "not just single words, which means you were thinking in "
    "English rather than translating in your head. Third, you "
    "used past simple correctly when you talked about your "
    "weekend — 'I went', 'I played', 'I saw' — and that is the "
    "hardest tense for most learners. One thing to keep "
    "practising is the question form: don't say 'You went to "
    "the park yesterday?', say 'Did you go to the park "
    "yesterday?'. Remember the helper verb 'did' and the base "
    "form of the verb after it. Overall, you are improving "
    "quickly, and I am proud of your progress. Keep talking "
    "every day, even for ten minutes, and soon long "
    "conversations will feel easy."
)


def _wait(predicate, timeout: float, app: "QApplication") -> bool:
    """Ждём условие, качая цикл событий (доставка межпотоковых сигналов)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _make_speech_file(dest_dir: str) -> str:
    """Генерирует «речь студента» macOS say (английская, Samantha)."""
    path = os.path.join(dest_dir, "student_speech.aiff")
    subprocess.run(
        ["say", "-v", "Samantha", "-o", path, SENTENCE],
        check=True,
    )
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("say не создал файл речи")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Реальная проверка полного цикла (STT + Ollama + Qwen3-TTS)."
    )
    parser.add_argument(
        "--audio", default="",
        help="файл «речи студента» (по умолчанию: генерирует macOS say)",
    )
    parser.add_argument(
        "--cycles", type=int, default=2,
        help="количество циклов запись→STT→ответ→повтор (по умолчанию 2)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    config = AppConfig()

    print(f"Туннель Ollama: {config.ollama_base_url}")
    print(f"TTS: {config.tts_engine} / {config.tts_qwen3_method} — "
          f"референс: {config.tts_qwen3_ref_audio}")
    if not os.path.exists(config.tts_qwen3_ref_audio):
        print("ОШИБКА: референс не найден. Восстанови: "
              ".venv/bin/python tools/install_tts_reference.py")
        return 1
    print(f"STT: {config.stt_backend} / {config.stt_mlx_model}\n")

    client = OllamaClient(config, app)
    audio = AudioController(parent=app, tts=build_tts(config))
    stt = SttEngine(build_backend(config), app)

    ready_paths: list[str] = []
    failed_reasons: list[str] = []
    audio.tts.reply_ready.connect(ready_paths.append)
    audio.tts.failed.connect(failed_reasons.append)
    chat_results: list[str] = []
    chat_errors: list[str] = []
    client.chat_completed.connect(chat_results.append)
    client.chat_failed.connect(chat_errors.append)
    models: list[str] = []
    model_errors: list[str] = []
    client.models_loaded.connect(models.extend)
    client.models_failed.connect(model_errors.append)

    failures: list[str] = []

    # --- Модель Ollama -------------------------------------------------------
    t0 = time.monotonic()
    client.load_models()
    if not _wait(lambda: models or model_errors, 15.0, app):
        print("ОШИБКА: Ollama не ответила за 15 c")
        return 1
    if model_errors:
        print(f"ОШИБКА: {model_errors[0]}")
        return 1
    model = next((m for m in models if "qwen3" in m), models[0])
    print(f"Ollama: {len(models)} моделей, выбрана {model!r} "
          f"({time.monotonic() - t0:.2f} c)\n")

    # --- Файл «записи» (замена) ---------------------------------------------
    own_temp = None
    if args.audio:
        speech = os.path.abspath(args.audio)
        if not os.path.exists(speech):
            print(f"ОШИБКА: файл не найден: {speech}")
            return 1
    else:
        own_temp = tempfile.mkdtemp(prefix="english-companion-check-")
        speech = _make_speech_file(own_temp)
    print(f"«Запись» (замена): {speech}\n")

    history: list[Message] = []
    cycle_stats: list[dict] = []

    # --- Циклы ---------------------------------------------------------------
    for n in range(1, args.cycles + 1):
        print(f"=== Цикл {n}/{args.cycles} ===")

        # 1) STT (реальная модель; в первом цикле — и загрузка).
        t0 = time.monotonic()
        text = stt.recognize_sync(speech)
        stt_time = time.monotonic() - t0
        first_note = " (включая загрузку модели)" if n == 1 else " (модель уже загружена)"
        print(f"  STT{first_note}: {stt_time:.2f} c")
        print(f"  распознано: {text!r}")
        if not text.strip():
            failures.append(f"цикл {n}: STT вернул пустой текст")

        # 2) Отправка (реальный запрос Ollama).
        t0 = time.monotonic()
        client.send_chat(model, history + [Message(role="user", content=text)])
        if not _wait(lambda: chat_results or chat_errors, config.chat_timeout_ms / 1000, app):
            print("ОШИБКА: ответ Ollama не пришёл")
            return 1
        if chat_errors:
            print(f"ОШИБКА: {chat_errors.pop()}")
            return 1
        reply = chat_results.pop()
        ollama_time = time.monotonic() - t0
        print(f"  Ollama: {ollama_time:.2f} c")
        print(f"  ответ: {reply[:120]!r}{'…' if len(reply) > 120 else ''}")

        # 3) Озвучивание (реальный синтез Qwen3 + реальное воспроизведение).
        t_text = time.monotonic()          # «текст появился в чате»
        n_ready_before = len(ready_paths)
        audio.speak_reply(reply)
        if not _wait(lambda: len(ready_paths) > n_ready_before or failed_reasons,
                     240.0, app):
            print("ОШИБКА: синтез не завершился за 240 c")
            return 1
        if failed_reasons:
            print(f"ОШИБКА: {failed_reasons.pop()}")
            return 1
        t_ready = time.monotonic()
        synth_time = t_ready - t_text
        load_note = " (включая загрузку модели Qwen3)" if n == 1 else ""
        print(f"  синтез готов{load_note}: {synth_time:.2f} c")

        if not _wait(lambda: audio.state == AudioController.PLAYING, 15.0, app):
            print("ОШИБКА: автозапуск не начал воспроизведение")
            return 1
        t_play = time.monotonic()
        text_to_audio = t_play - t_text
        print(f"  ТЕКСТ → ЗВУК: {text_to_audio:.2f} c "
              f"(готовность {synth_time:.2f} + старт плеера {t_play - t_ready:.2f})")
        path_n = ready_paths[-1]
        print(f"  файл: {path_n}")

        # Дождаться конца воспроизведения (звук должен быть слышен).
        if not _wait(lambda: audio.state == AudioController.IDLE, 120.0, app):
            print("ОШИБКА: воспроизведение не завершилось за 120 c")
            return 1
        print("  воспроизведение завершено")

        # 4) «Повторить ответ»: тот же файл, без нового синтеза.
        if not audio.can_replay:
            failures.append(f"цикл {n}: повтор недоступен после ответа")
        t0 = time.monotonic()
        audio.replay_reply()
        if not _wait(lambda: audio.state == AudioController.PLAYING, 15.0, app):
            print("ОШИБКА: повтор не начал воспроизведение")
            return 1
        t_replay_play = time.monotonic() - t0
        if not _wait(lambda: audio.state == AudioController.IDLE, 120.0, app):
            print("ОШИБКА: повтор не завершился за 120 c")
            return 1
        if len(ready_paths) != n_ready_before + 1:
            failures.append(f"цикл {n}: повтор запустил НОВЫЙ синтез")
        if audio.tts.last_reply_path != path_n:
            failures.append(f"цикл {n}: повтор проиграл ДРУГОЙ файл")
        print(f"  повтор: старт {t_replay_play:.2f} c, тот же файл, "
              f"нового синтеза нет")

        # История, как в приложении.
        history += [
            Message(role="user", content=text),
            Message(role="assistant", content=reply),
        ]
        cycle_stats.append({
            "n": n,
            "stt": stt_time,
            "ollama": ollama_time,
            "synth": synth_time,
            "text_to_audio": text_to_audio,
        })
        print()

    # --- Закрытие во время синтеза ------------------------------------------
    print("=== Закрытие во время синтеза ===")
    n_ready_before = len(ready_paths)
    n_failed_before = len(failed_reasons)
    audio.speak_reply(SHUTDOWN_TEXT)
    if not _wait(lambda: audio.tts.is_busy, 30.0, app):
        print("ОШИБКА: синтез длинного текста не начался")
        return 1
    time.sleep(2.0)  # гарантируем, что рабочий поток в синтезе
    t0 = time.monotonic()
    audio.shutdown()
    shutdown_time = time.monotonic() - t0
    app.processEvents()

    # Новый синтез после shutdown отклоняется компонентом (synthesize —
    # единственный путь, принимающий текст; фасад после shutdown молчит).
    audio.speak_reply("again")
    app.processEvents()
    checks = {
        "компонент остановлен": audio.tts.is_shutdown,
        "новый синтез отклонён": (
            audio.tts.synthesize("again") is False
            and audio.tts.is_shutdown
        ),
        "повтор отклонён": not audio.can_replay,
        "поздний результат не опубликован": len(ready_paths) == n_ready_before,
        "временных файлов нет": not audio.tts._temp_dir,
    }
    # Поздние сигналы (если рабочий поток всё ещё дописывает) не должны
    # «оживить» слой.
    time.sleep(3.0)
    app.processEvents()
    checks["слой не ожил за 3 c"] = len(ready_paths) == n_ready_before
    ok_all = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'OK ' if ok else 'FAIL'} {name}")
    print(f"  shutdown() вернулась за {shutdown_time:.2f} c "
          f"(ограничение join — 5 c)")
    if not ok_all:
        failures.append("закрытие во время синтеза: есть FAIL")

    stt.shutdown()

    # --- Итог ----------------------------------------------------------------
    print("\n=== Итог ===")
    first = next((c for c in cycle_stats if c["n"] == 1), None)
    second = next((c for c in cycle_stats if c["n"] > 1), None)
    if first:
        print(f"Первый ответ:    текст→звук {first['text_to_audio']:.2f} c "
              f"(синтез {first['synth']:.2f} c, вкл. загрузку модели)")
    if second:
        print(f"Последующий:     текст→звук {second['text_to_audio']:.2f} c "
              f"(синтез {second['synth']:.2f} c)")
    if first and second and second["synth"] > 0:
        est_load = max(first["synth"] - second["synth"], 0.0)
        print(f"Загрузка модели Qwen3 (оценка = разность синтезов): ~{est_load:.1f} c")
    if failed_reasons:
        print(f"Ошибки синтеза: {failed_reasons}")

    if own_temp:
        import shutil
        shutil.rmtree(own_temp, ignore_errors=True)

    if failures:
        print("\nЕсть FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nВсе проверки пройдены. Прослушай голос — тембр/акцент"
          " проверяешь на слух ты.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
