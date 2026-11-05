"""Разовый пробо́вый скрипт: первый vs второй проигрыв того же файла.

Один процесс, ОДИН И ТОТ ЖЕ файл:
* проигрывание 1 — «первое» (холодная загрузка + первый старт вывода);
* проигрывание 2 — «повторное» (кэш по URL, вывод тёплый).

Критерий: длина тела тона (1.5 с) в захвате микрофоном.
Если начало ОТСЕКАЕТСЯ на первом проигрывании — тело будет короче.
Если просто ЗАДЕРЖИВАЕТСЯ старт — тела одинаковые.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
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


def body_file(path: str, t0: float = 0.3, dur: float = 1.5, rate: int = 48000, amp: int = 12000) -> None:
    total = t0 + dur + 0.6
    n = int(rate * total)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(n):
            t = i / rate
            frames.append(
                struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * t / rate)))
                if t0 <= t < t0 + dur
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
    floor = max([r for _, r in env if r < peak / 10], default=10) or 10
    thr = max(peak * 0.25, floor * 3, 50)
    hits = [t for t, r in env if r > thr]
    return len(s) / 48000, (hits[0] if hits else None), (hits[-1] if hits else None), peak, floor, thr, env


def main() -> int:
    c = AudioController()
    for dev_id, name in c.input_devices():
        if "MacBook" in name:
            c.set_input_device(dev_id)
            break

    tmp = tempfile.mkdtemp(prefix="firstplay-")
    f = os.path.join(tmp, "body.wav")
    body_file(f)

    out = QAudioOutput(QMediaDevices.defaultAudioOutput())
    player = QMediaPlayer()
    player.setAudioOutput(out)

    for i in (1, 2):
        c.record()
        pump(0.4)
        player.setSource(QUrl.fromLocalFile(f))
        player.play()
        pump(3.4)
        player.stop()
        pump(0.3)
        c.stop_recording()
        pump(1.0)
        cap = c.last_recording_path
        dur, onset, tail, peak, floor, thr, env = analyze(cap)
        body = round(tail - onset, 2) if (onset is not None and tail is not None) else None
        print(
            f"проигрывание {i}: onset={round(onset, 2) if onset is not None else None} "
            f"тело={body} с (идеал 1.5) peak={peak:.0f} thr={thr:.0f} floor={floor:.0f}"
        )
        print("   env:", " ".join(f"{t:.2f}:{r:.0f}" for t, r in env[::2]))
        pump(0.8)
    c.shutdown()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
