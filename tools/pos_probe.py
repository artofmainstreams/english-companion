"""Разовый пробо́вый скрипт: timeline QMediaPlayer.position().

Без микрофона/акустики: смотрим, С КОГОГО начинается воспроизведение.
* первая выборка position ≈ 0 → начало просто ЗАДЕРЖИВАЕТСЯ (буферизуется);
* первая выборка position > 0 → начало ОТСЕКАЕТСЯ (модели/engine выбрасывают).

1-я попытка — холодная загрузка источника, 2-я — кэш QMediaPlayer по URL
(ровно «первое» и «повторное» проигрывание пользователя).
"""
import math
import os
import struct
import sys
import tempfile
import time
import wave

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaDevices

app = QApplication(sys.argv)

DUR = 3.0
rate = 48000
n = int(rate * DUR)
f = os.path.join(tempfile.mkdtemp(prefix="posprobe-"), "f.wav")
with wave.open(f, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(b"".join(struct.pack("<h", int(20000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)))

out = QAudioOutput(QMediaDevices.defaultAudioOutput())
player = QMediaPlayer()
player.setAudioOutput(out)

from PySide6.QtCore import QUrl  # noqa: E402

def run_attempt(label: str) -> None:
    samples: list[tuple[float, int]] = []
    timer = QTimer()

    def tick() -> None:
        samples.append((time.monotonic(), player.position()))

    timer.timeout.connect(tick)
    timer.start(50)

    t0 = time.monotonic()
    player.setSource(QUrl.fromLocalFile(f))
    player.play()

    # Фиксированное окно 6 с: даём загрузиться, воспроизвестись и закончиться.
    end = time.monotonic() + 6.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.05)
    timer.stop()

    if samples:
        first_nz = next(((dt * 1000, p) for dt, p in ((s[0] - t0, s[1]) for s in samples) if p > 0), None)
        last = samples[-1]
        tail = samples[-6:]
        slope = (tail[-1][1] - tail[0][1]) / max(1e-9, (tail[-1][0] - tail[0][0]))
        print(f"{label}: первый ненулевой position: {first_nz} (dt мс, pos мс)")
        print(
            f"{label}: последний position={last[1]} мс; "
            f"скорость≈{slope:.2f} (мс позиции на мс времени)"
        )
        print(
            f"{label}: timeline(dt ms → pos ms): "
            + " ".join(f"{(s[0]-t0)*1000:.0f}:{s[1]}" for s in samples[::2])
        )
    else:
        print(f"{label}: нет samples")
    player.stop()


run_attempt("1-я (холодная загрузка):")
run_attempt("2-я (кэш по URL):")
print("DONE")
