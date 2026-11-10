#!/usr/bin/env python
"""Потоковый Qwen3-TTS в ПРИЛОЖЕНИИ: реальный замер + сравнение с полным WAV.

Всё настоящее: реальная модель Qwen3 (reference, референс p4), реальный
``AudioController`` + ``Qwen3TtsSynth``, реальный воспроизводящий поток
(``sounddevice``/PortAudio — звук СЛЫШЕН). Замеряем и честно разделяем:

  * TTFC — время до ПЕРВОГО куска (стриминг) vs TTFS всего WAV (база);
  * старт вывода — когда плеер вошёл в PLAYING (для стриминга — после
    накопления стартового буфера, поэтому позже TTFC);
  * опустошение буфера — когда плеер вернулся в IDLE (весь ответ отыгран);
  * недочёты — сэмплы тишины, вставленные вместо звука (должно быть 0);
  * RTF по кускам (формула ниже); первый и последующие ответы (модель
    грузится один раз — сравнимо).

Формула RTF (real-time factor) на кусок:

    RTF_i = T_gen_i / T_audio_i
          = (arrival[i] - arrival[i-1]) / (samples[i] / sr)

    RTF < 1 → генерация быстрее реального времени → следующего куска
    хватает до конца предыдущего → недочётов нет;
    RTF > 1 → генерация медленнее → риск недочётов (тишина встык).

Сравнение стриминг vs «весь WAV» — на ОДИНАКОВЫХ текстах (короткий и
длинный). Полный голосовой цикл показываю ОТДЕЛЬНО: STT + пауза ручной
отправки (ИМИТАЦИЯ действия пользователя, не вычисление) + Ollama + TTS +
старт вывода. Проверки: Stop (красит куски, готовый не теряется), новый
ответ (сменяет), закрытие во время синтеза (ФАКТ смерти потока, отдельно
от возврата ``join(timeout)``). Тембр p4 и слышимость стыков — проверяешь
на слух ты (числа ниже — инструментальные).

Запуск (из корня проекта; нужен туннель Ollama для голосового цикла)::

    .venv/bin/python real_tts_stream_app.py
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion.audio_controller import AudioController, build_tts  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.message import Message  # noqa: E402
from companion.ollama_client import OllamaClient  # noqa: E402
from companion.stt_backends import build_backend  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402

SHORT = "Well done! Great job today."
LONG = (
    "That was a really strong answer. You explained your idea clearly, "
    "gave a concrete example, and kept your rhythm natural. "
    "One small thing to watch: try not to rush the ending. "
    "Overall, very clear and confident — well done, and keep it up."
)

# Имитация «ручной отправки» (пауза пользователя). Секунды.
SEND_DELAY_S = 1.5
STUDENT_SENTENCE = "I went to the park yesterday to read a book."


def _pump(app, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _wait(predicate, app, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class Recorder:
    """Хранит временные метки событий сигналами (доставка — в главном потоке)."""

    def __init__(self, audio: AudioController, sr: int) -> None:
        self.audio = audio
        self.sr = sr
        self.t_start = 0.0
        self.first_chunk = None
        self.chunks = []
        self.playing_at = None
        self.idle_at = None
        self.ready_at = None
        self.failed = None
        self.ready_path = None
        self.underrun_max = 0
        audio.tts.chunk_ready.connect(self._on_chunk)
        audio.player.state_changed.connect(self._on_state)
        audio.tts.reply_ready.connect(self._on_ready)
        audio.tts.failed.connect(self._on_failed)

    def reset(self) -> None:
        self.t_start = 0.0
        self.first_chunk = None
        self.chunks = []
        self.playing_at = None
        self.idle_at = None
        self.ready_at = None
        self.failed = None
        self.ready_path = None
        self.underrun_max = 0

    def _now(self) -> float:
        return time.monotonic()

    def _on_chunk(self, chunk, sample_rate, job_id) -> None:
        t = self._now()
        if self.first_chunk is None:
            self.first_chunk = t
        self.chunks.append((t, len(chunk)))

    def _on_state(self) -> None:
        st = self.audio.state
        if st == AudioController.PLAYING and self.playing_at is None:
            self.playing_at = self._now()
        elif st == AudioController.IDLE and self.playing_at is not None \
                and self.idle_at is None:
            self.idle_at = self._now()

    def _on_ready(self, path) -> None:
        self.ready_at = self._now()
        self.ready_path = path

    def _on_failed(self, reason) -> None:
        self.failed = reason

    def run(self, text: str, app, timeout: float = 180.0) -> None:
        self.reset()
        self.t_start = self._now()
        self.audio.speak_reply(text)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            # Недочёты читаем, пока поток жив (после teardown — уже 0).
            if self.audio.player.stream_active:
                u = self.audio.player.stream_underrun_samples()
                if u > self.underrun_max:
                    self.underrun_max = u
            if (self.failed is not None) or (
                self.ready_at is not None
                and self.audio.state == AudioController.IDLE
            ):
                break
            time.sleep(0.01)

    # --- Метрики -----------------------------------------------------------

    def metrics(self) -> dict:
        m = {"failed": self.failed, "chunks": len(self.chunks)}
        if self.first_chunk is not None:
            m["ttfc"] = self.first_chunk - self.t_start
        if self.ready_at is not None:
            m["ttfs"] = self.ready_at - self.t_start
        if self.playing_at is not None:
            m["out_start"] = self.playing_at - self.t_start
        if self.playing_at is not None and self.idle_at is not None:
            m["playback"] = self.idle_at - self.playing_at
        if self.idle_at is not None:
            m["to_idle"] = self.idle_at - self.t_start
        m["underrun"] = self.underrun_max
        # RTF по кускам (без первого — нет предыдущего прихода)
        r = self.audio.tts.streaming_interval
        rtfs = []
        for i in range(1, len(self.chunks)):
            t_gen = self.chunks[i][0] - self.chunks[i - 1][0]
            t_audio = self.chunks[i][1] / self.sr
            if t_audio > 0:
                rtfs.append(t_gen / t_audio)
        if rtfs:
            m["rtf_mean"] = sum(rtfs) / len(rtfs)
            m["rtf_max"] = max(rtfs)
        return m


def _make_speech(dest_dir: str, sentence: str, name: str) -> str:
    path = os.path.join(dest_dir, name)
    subprocess.run(["say", "-v", "Samantha", "-o", path, sentence], check=True)
    return path


def main() -> int:
    app = QApplication(sys.argv)
    config = AppConfig()
    ref = config.tts_qwen3_ref_audio
    if not os.path.exists(ref):
        print("ОШИБКА: нет референса p4. Восстанови: "
              ".venv/bin/python tools/install_tts_reference.py")
        return 1
    audio = AudioController(
        parent=app,
        tts=build_tts(config),
        stream_start_buffer_ms=config.tts_qwen3_start_buffer_ms,
    )
    synth = audio.tts
    sr = 24000
    rec = Recorder(audio, sr)
    print(f"референс p4: {ref}")
    print(f"интервал куска: {synth.streaming_interval} c; "
          f"стартовый буфер: {config.tts_qwen3_start_buffer_ms} мс")
    print(f"Ollama: {config.ollama_base_url}\n")

    def line(name, v, unit=""):
        print(f"  {name:<16}{v:>9.2f} {unit}" if v is not None
              else f"  {name:<16}{'—':>9}")

    def report(title, m):
        print(f"\n=== {title} ===")
        if m.get("failed"):
            print(f"  ОШИБКА: {m['failed']}")
            return
        line("куски", m["chunks"], "шт.")
        if "ttfc" in m:
            line("TTFC (1-й кусок)", m["ttfc"], "с")
        if "ttfs" in m:
            line("TTFS (готов)", m["ttfs"], "с")
        if "out_start" in m:
            line("старт вывода", m["out_start"], "с")
        if "playback" in m:
            line("длительность", m["playback"], "с")
        if "to_idle" in m:
            line("до IDLE", m["to_idle"], "с")
        line("недочёты", m["underrun"], "сэмпл.")
        if "rtf_mean" in m:
            line("RTF (средн.)", m["rtf_mean"], "")
            line("RTF (макс.)", m["rtf_max"], "")

    # --- Прогрев модели (разово) -------------------------------------------
    print("Прогрев модели (загрузка один раз)…")
    t0 = time.monotonic()
    audio.speak_reply(SHORT)
    if not _wait(lambda: rec.failed or rec.ready_at, app, 240.0):
        print("ОШИБКА: прогрев не завершился за 240 c"); return 1
    if rec.failed:
        print(f"ОШИБКА прогрева: {rec.failed}"); return 1
    _wait(lambda: audio.state == AudioController.IDLE, app, 60.0)
    print(f"  модель загружена + первый синтез за "
          f"{time.monotonic() - t0:.1f} c\n")

    # --- Сравнение: полный WAV vs стриминг (одинаковые тексты) --------------
    print("СРАВНЕНИЕ: «весь WAV» (текущее поведение) vs СТРИМИНГ "
          "(речь частями)")
    print(f"RTF_i = (приход[i]-приход[i-1]) / (сэмпл[i]/sr); "
          f"<1 — без недочётов\n")

    results = {}
    for label, text in (("короткий", SHORT), ("длинный", LONG)):
        # Полный WAV (стриминг выключен) — база.
        synth.streaming = False
        rec.run(text, app)
        m_full = rec.metrics()
        report(f"{label.upper()} · весь WAV (1-й ответ)", m_full)

        # Стриминг, первый ответ (модель уже загружена).
        synth.streaming = True
        rec.run(text, app)
        m_st1 = rec.metrics()
        report(f"{label.upper()} · стриминг (1-й ответ)", m_st1)

        # Стриминг, последующий ответ.
        rec.run(text, app)
        m_st2 = rec.metrics()
        report(f"{label.upper()} · стриминг (2-й ответ)", m_st2)

        results[label] = (m_full, m_st1, m_st2)

    # --- Подбор стартового буфера (500 vs 1000 мс) --------------------------
    print("\nПОДБОР СТАРТОВОГО БУФЕРА (влияет на «старт вывода», "
          "не на TTFC; короткий и длинный ответ):")
    for buf_ms in (500, 1000):
        audio.stream_start_buffer_ms = buf_ms
        for label, text in (("короткий", SHORT), ("длинный", LONG)):
            rec.run(text, app)
            m = rec.metrics()
            line(f"{label} · {buf_ms} мс → старт вывода",
                 m.get("out_start"), "с")
            line(f"{label} · {buf_ms} мс → недочёты",
                 m.get("underrun"), "сэмпл.")
    audio.stream_start_buffer_ms = config.tts_qwen3_start_buffer_ms

    # --- Полный голосовой цикл (STT + отправка + Ollama + TTS) -------------
    print("\nПОЛНЫЙ ГОЛОСОВОЙ ЦИКЛ (один запуск; пауза отправки — имитация):")
    _voice_cycle(app, config, audio, rec, sr)

    # --- Проверки: Stop / новый ответ / закрытие ---------------------------
    print("\nПРОВЕРКИ (реальный синтез):")
    print("  [Stop] …")
    ok_stop = _check_stop(app, audio, rec, LONG)
    print("  [новый ответ] …")
    ok_repl = _check_replace(app, audio, rec, SHORT, LONG)
    print("  [закрытие во время синтеза] …")
    ok_close = _check_shutdown(app, audio, rec, LONG)  # разрушает контроллер
    print(f"  Stop: {'OK' if ok_stop else 'FAIL'}; "
          f"новый ответ: {'OK' if ok_repl else 'FAIL'}; "
          f"закрытие: {'OK' if ok_close else 'FAIL'}")

    print("\nГолос p4 — тембр и слышимость стыков проверяешь на слух ты.")
    return 0


def _check_stop(app, audio, rec, text) -> bool:
    """Stop во время синтеза: куски отброшены, готовый (первый) жив."""
    # Первый ответ — успех.
    audio.tts.streaming = True
    rec.run(text, app)
    if rec.failed:
        return False
    _wait(lambda: audio.state == AudioController.IDLE, app, 60.0)
    first = audio.tts.last_reply_path
    # Второй: сразу Stop (модель ещё синтезирует длинный текст).
    audio.speak_reply(text)
    time.sleep(0.15)  # рабочий поток вошёл в синтез
    audio.stop_playback()
    _wait(lambda: audio.state == AudioController.IDLE, app, 30.0)
    time.sleep(1.0)   # дать «поздним» кускам прийти
    _pump(app, 0.3)
    last = audio.tts.last_reply_path
    no_late = rec.failed is None
    return first is not None and last == first and no_late


def _check_replace(app, audio, rec, short, long) -> bool:
    """Новый ответ сменяет незавершённый: поздний отброшен, свежий готов."""
    audio.tts.streaming = True
    audio.speak_reply(long)      # длинный — идёт долго
    time.sleep(0.2)             # вошёл в синтез
    audio.speak_reply(short)     # сменяет коротким
    ok = _wait(
        lambda: (rec.ready_at is not None
                 and audio.state == AudioController.IDLE)
        or rec.failed is not None,
        app, 60.0,
    )
    if rec.failed:
        return False
    _pump(app, 0.5)
    # Короткий ответ готов; длинный (старое задание) отброшен.
    return audio.tts.has_reply and ok


def _check_shutdown(app, audio, rec, text) -> bool:
    """Закрытие во время синтеза: поток мёртв, файлов нет, сигналов нет.

    Факт смерти потока проверяем ОТСЕЛЬНО: смотрим ``is_alive()`` по имени,
    а не только то, что ``join(timeout)`` вернулся.
    """
    audio.tts.streaming = True
    audio.speak_reply(text)
    time.sleep(0.2)             # вошёл в синтез
    rec.reset()
    t0 = time.monotonic()
    audio.shutdown()
    join_time = time.monotonic() - t0
    _pump(app, 0.3)
    time.sleep(1.0)
    workers = [t for t in threading.enumerate() if t.name == "tts-qwen3"]
    alive = any(t.is_alive() for t in workers)
    no_temp = not audio.tts._temp_dir
    no_late = (rec.failed is None and rec.ready_at is None)
    print(f"    join() вернулся за {join_time:.1f} c; "
          f"факт: поток {'ЖИВ' if alive else 'мёртв'}; "
          f"временных файлов: {'есть' if not no_temp else 'нет'}; "
          f"поздних сигналов: {'есть' if not no_late else 'нет'}")
    return (not alive) and no_temp and no_late


def _voice_cycle(app, config, audio, rec, sr) -> None:
    """STT + пауза ручной отправки (имитация) + Ollama + TTS (стриминг)."""
    client = OllamaClient(config, app)
    stt = SttEngine(build_backend(config), app)
    chat = []
    chat_err = []
    client.chat_completed.connect(chat.append)
    client.chat_failed.connect(chat_err.append)
    models = []
    model_err = []
    client.models_loaded.connect(models.extend)
    client.models_failed.connect(model_err.append)

    client.load_models()
    if not _wait(lambda: models or model_err, app, 15.0):
        print("  ОШИБКА: Ollama не ответила за 15 c (цикл пропущен)"); return
    if model_err:
        print(f"  ОШИБКА Ollama: {model_err[0]} (цикл пропущен)"); return
    model = next((m for m in models if "qwen3" in m), models[0])

    tmp = tempfile.mkdtemp(prefix="english-companion-sapp-")
    student_audio = _make_speech(tmp, STUDENT_SENTENCE, "student.aiff")
    print(f"  студент (замена): {STUDENT_SENTENCE!r}")
    print(f"  модель Ollama: {model}")
    audio.tts.streaming = True

    # 1) STT (реальное распознавание).
    t0 = time.monotonic()
    text = stt.recognize_sync(student_audio)
    stt_t = time.monotonic() - t0
    if not text.strip():
        text = STUDENT_SENTENCE
    print(f"  1. STT            {stt_t:6.2f} c   распознано: {text!r}")

    # 2) Ожидание ручной отправки (ИМИТАЦИЯ паузы пользователя).
    t0 = time.monotonic()
    _pump(app, SEND_DELAY_S)
    send_t = time.monotonic() - t0
    print(f"  2. отправка       {send_t:6.2f} c   (имитация действия "
          f"пользователя)")

    # 3) Ollama (реальный запрос).
    t0 = time.monotonic()
    client.send_chat(model, [Message(role="user", content=text)])
    if not _wait(lambda: chat or chat_err, app, config.chat_timeout_ms / 1000):
        print("  ОШИБКА: ответ Ollama не пришёл"); return
    if chat_err:
        print(f"  ОШИБКА Ollama: {chat_err.pop()}"); return
    reply = chat.pop()
    ollama_t = time.monotonic() - t0
    print(f"  3. Ollama         {ollama_t:6.2f} c   ответ {len(reply)} симв.")

    # 4+5) TTS (стриминг) + старт вывода на РЕАЛЬНОМ ответе Ollama.
    rec.reset()
    rec.t_start = time.monotonic()
    audio.speak_reply(reply)
    _wait(
        lambda: (rec.failed is not None)
        or (rec.ready_at is not None and audio.state == AudioController.IDLE),
        app, 180.0,
    )
    tts_t = (rec.ready_at or time.monotonic()) - rec.t_start
    m = rec.metrics()
    print(f"  4+5. TTS (стрим.) {tts_t:6.2f} c   до готовности; "
          f"куски: {m.get('chunks')}")
    if "ttfc" in m:
        print(f"  TTFC           {m['ttfc']:6.2f} c   до первого куска")
    if "out_start" in m:
        print(f"  старт вывода   {m['out_start']:6.2f} c")
    print(f"  недочёты       {m.get('underrun', 0):6d} сэмпл.")

    total = stt_t + send_t + ollama_t + tts_t
    print(f"\n  ИТОГО цикл: {total:.2f} c "
          f"(STT {stt_t:.2f} + отправка {send_t:.2f} "
          f"[имитация] + Ollama {ollama_t:.2f} + TTS {tts_t:.2f})")
    print(f"  Из них ПОЛУЧАЕТСЯ: вычисления без паузы пользователя = "
          f"{stt_t + ollama_t + tts_t:.2f} c")

    stt.shutdown()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    code = main()
    # Замеры (и проверки Stop/замены/закрытия) уже распечатаны. Осмысленный
    # код выхода фиксируем явно, а процесс завершаем через os._exit, минуя
    # разбор интерпретатора: на выходе может срабатывать недетерминированный
    # селфолт upstream MLX (разбор thread-local CompileCache в
    # CacheEntry::~CacheEntry, вне нашего кода и вне пути данных) — данные
    # при этом уже записаны. atexit-хендлеров у диагностики нет, stdout
    # печатался с flush=True.
    import sys as _s
    _s.stdout.flush()
    _s.stderr.flush()
    os._exit(code)
