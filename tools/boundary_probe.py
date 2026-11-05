"""Разовый пробо́вый скрипт: целость НАЧАЛА первой записи (чёткий критерий).

Схема (без неоднозначностей эха-онсета):
* тон ВКЛЮЧЁН в t=0, ВЫКЛЮЧЕН в t=2.0 (игрёт до и после окна записи);
* запись начинается в t=0.5, стоп в t=2.5 (файл 2.0 с);
* при intact-голове тон→тишина в файле на 1.5 с;
* если первые X с записи потеряны — переход на (1.5 − X), хвост тишины длиннее.

Приёмник — встроенный микрофон MacBook (set_input_device), звук — динамики
MacBook: сильная и стабильная связь «динамик → микрофон».

Запускать в СВЕЖЕМ процессе: запись 1 = холодное открытие устройства,
запись 2 = тёплое.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave

from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaDevices

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from companion.audio_controller import AudioController  # noqa: E402

app = QApplication(sys.argv)


def pump(s: float) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(s * 1000), loop.quit)
    loop.exec()


def tone_with_off(path: str, on: float = 0.0, off: float = 2.0, rate: int = 48000, amp: int = 32000) -> None:
    n = int(rate * (off + 0.5))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(n):
            t = i / rate
            frames.append(
                struct.pack("<h", int(amp * math.sin(2 * math.pi * 1000 * t / rate)))
                if on <= t < off
                else struct.pack("<h", 0)
            )
        w.writeframes(b"".join(frames))


def analyze(path: str):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "s16le", "-ar", "48000", "-ac", "1", "-"],
        capture_output=True, check=True,
    ).stdout
    s = struct.unpack(f"<{len(out) // 2}h", out)
    blk = 480
    env = []
    for k in range(0, len(s) - blk, blk):
        b = s[k : k + blk]
        env.append((k / 48000, math.sqrt(sum(x * x for x in b) / blk)))
    peak = max(r for _, r in env)
    floor = max([r for _, r in env[-8:]], default=10) or 10  # хвост = тишина
    thr = max(floor * 5, 60)
    hits = [t for t, r in env if r > thr]
    onset = hits[0] if hits else None
    tail = hits[-1] if hits else None
    return len(s) / 48000, onset, tail, peak, floor, thr, env


def main() -> int:
    # Встроенный микрофон — приёмник
    c = AudioController()
    ok = c.set_input_device("BuiltInMicrophoneDevice")
    print("выбор BuiltInMicrophoneDevice:", ok, c.status)
    if not ok:
        print("нет встроенного микрофона — прерываюсь")
        return 1

    tmp = tempfile.mkdtemp(prefix="boundary-")
    t1 = os.path.join(tmp, "tone1.wav")
    tone_with_off(t1)
    out = QAudioOutput(QMediaDevices.defaultAudioOutput())
    player = QMediaPlayer()
    player.setAudioOutput(out)
    player.setSource(QUrl.fromLocalFile(t1))
    pump(0.8)  # прогрев загрузки

    for i in (1, 2):
        player.play()  # тон играет: [0.0 .. 2.0]
        pump(0.5)      # тон уже 0.5 с в воздухе
        c.record()     # захват [0.5 .. 2.5]
        pump(2.0)
        c.stop_recording()
        pump(1.2)
        f = c.last_recording_path
        dur, onset, tail, peak, floor, thr, env = analyze(f)
        # переход тон→тишина = последний hit
        print(
            f"запись {i}: dur={dur:.3f} onset={onset if onset is None else round(onset, 2)} "
            f"тон-конец={tail if tail is None else round(tail, 2)} (идеал 1.50 при intact-голове) "
            f"→ потеря начала ≈ {round(1.5 - tail, 2) if tail is not None else '?'} с; "
            f"peak={peak:.0f} floor={floor:.0f} thr={thr:.0f}"
        )
        print("   env:", " ".join(f"{t:.2f}:{r:.0f}" for t, r in env))
        pump(1.0)
    c.shutdown()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
