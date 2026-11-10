#!/usr/bin/env python
"""Пробы ШТАТНОГО завершения (условия падения при закрытии).

Три сценария (по заданию):
  no_synth  — запуск и выход БЕЗ синтеза;
  speak     — выход ПОСЛЕ озвучивания (стриминг p4);
  stt_speak — выход ПОСЛЕ распознавания И озвучивания в одном процессе.

Везде — штатное завершение (порядок main.py: stt.shutdown → audio.shutdown),
БЕЗ os._exit: если процесс падает — это его РЕАЛЬНЫЙ код выхода. faulthandler
ловит SIGSEGV и печатает стек (состояние всех потоков).
"""
import os
import sys
import time
import faulthandler

# enable() сам регистрирует фатальные сигналы (SIGSEGV/SIGFPE/SIGABRT/...);
# явный register(SIGSEGV) на macOS запрещён — поэтому только enable().
faulthandler.enable(all_threads=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

MODE = os.environ.get("PROBE_MODE", "no_synth")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from companion.config import AppConfig  # noqa: E402
from companion.audio_controller import AudioController, build_tts  # noqa: E402
from companion.stt_engine import SttEngine  # noqa: E402
from companion.stt_backends import build_backend  # noqa: E402

SHORT = "Well done! Great job today."


def pump(app, seconds):
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def wait_idle(app, audio, timeout=60.0):
    d = time.monotonic() + timeout
    while time.monotonic() < d:
        pump(app, 0.05)
        if audio.state == AudioController.IDLE and not audio.tts.is_busy:
            return True
    return (audio.state == AudioController.IDLE and not audio.tts.is_busy)


def main() -> int:
    app = QApplication(sys.argv)
    config = AppConfig()
    audio = AudioController(
        parent=app,
        tts=build_tts(config),
        stream_start_buffer_ms=config.tts_qwen3_start_buffer_ms,
    )
    stt = SttEngine(build_backend(config), app)
    stt_name = type(getattr(stt, "backend", None)).__name__
    print(f"MODE={MODE} tts={type(audio.tts).__name__} stt={stt_name} "
          f"streaming={getattr(audio.tts, 'streaming', None)}", flush=True)

    if MODE in ("speak", "stt_speak"):
        if MODE == "stt_speak":
            import numpy as np
            import wave
            sr = 16000
            n = int(sr * 0.6)
            t = np.arange(n) / sr
            tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_probe_tone.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((tone * 32767).astype(np.int16).tobytes())
            print("STT recognize_sync (загрузка MLX-модели + инференс)…",
                  flush=True)
            text = stt.recognize_sync(path)
            print(f"STT result: {text!r}", flush=True)
            try:
                os.remove(path)
            except OSError:
                pass
        audio.speak_reply(SHORT)
        ok = wait_idle(app, audio)
        print(f"speak done: state={audio.state} tts_busy={audio.tts.is_busy} "
              f"has_reply={audio.tts.has_reply} ok={ok}", flush=True)

    print("=== штатное завершение (main.py): stt.shutdown → audio.shutdown ===",
          flush=True)
    stt.shutdown()
    audio.shutdown()
    print("SHUTDOWN_OK — процесс выходит штатно", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
