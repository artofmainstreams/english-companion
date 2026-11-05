"""Разовый пробо́вый скрипт: теряется ли начало ПЕРВОЙ записи процесса.

Ключ: клик-включается ПОСЛЕ старта записи. Если первые X с захвата
«пропали» (холодное открытие устройства), клик сдвинется на −X в файле
(или исчезнет). При intact-голове клик на своём месте.

S1 — первая запись в свежем процессе (холодное устройство);
S2 — вторая запись того же процесса (тёплое).
Клики: [0..0.1) тон, [0.1..1.5) тишина, [1.5..1.6) тон;
play() в +0.5 после record(); стоп в +2.3.
Ожидаемо: клик-вкл ≈ 0.5+lat, клик-выкл ≈ 2.0+lat (lat ≈ 0.1..0.3).
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


def clicks_file(path: str, rate: int = 48000, amp: int = 32000) -> None:
    total = 1.6
    n = int(rate * total)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(n):
            t = i / rate
            on = (0.0 <= t < 0.1) or (1.5 <= t < 1.6)
            frames.append(struct.pack("<h", int(amp * math.sin(2 * math.pi * 1000 * t / rate))) if on else struct.pack("<h", 0))
        w.writeframes(b"".join(frames))


def analyze(path: str):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "s16le", "-ar", "48000", "-ac", "1", "-"],
        capture_output=True, check=True,
    ).stdout
    s = struct.unpack(f"<{len(out) // 2}h", out)
    blk = 240  # 5 мс — клики короткие
    env = []
    for k in range(0, len(s) - blk, blk):
        b = s[k : k + blk]
        env.append((k / 48000, math.sqrt(sum(x * x for x in b) / blk)))
    peak = max(r for _, r in env)
    floor = max([r for _, r in env if r < peak / 8], default=10) or 10
    thr = max(peak * 0.3, 80)
    hits = [round(t, 2) for t, r in env if r > thr]
    return len(s) / 48000, hits, peak, floor, thr


def main() -> int:
    c = AudioController()
    print("вход:", c.selected_input_name() or "системный")
    pairs = c.input_devices()
    for dev_id, name in pairs:
        if "MacBook" in name or "BuiltIn" in dev_id:
            c.set_input_device(dev_id)
            break

    tmp = tempfile.mkdtemp(prefix="coldhead-")
    out = QAudioOutput(QMediaDevices.defaultAudioOutput())
    player = QMediaPlayer()
    player.setAudioOutput(out)

    for i in (1, 2):
        clicks = os.path.join(tmp, f"clicks{i}.wav")
        clicks_file(clicks)
        player.setSource(QUrl.fromLocalFile(clicks))
        pump(0.7)  # прогрев загрузки

        c.record()
        pump(0.5)
        player.play()  # клик-вкл в ~+0.5 (лат ~0.1-0.3)
        pump(1.8)
        c.stop_recording()
        pump(1.0)
        f = c.last_recording_path
        dur, hits, peak, floor, thr = analyze(f)
        on = next((h for h in hits if h < 1.2), None)
        off = next((h for h in hits if h >= 1.2), None)
        print(
            f"S{i}: dur={dur:.3f} (ожид ~1.8) клик-вкл={on} (идеал ~0.6-0.8) "
            f"клик-выкл={off} (идеал ~2.1-2.3); hits={hits[:10]} peak={peak:.0f} thr={thr:.0f}"
        )
        pump(1.0)
    c.shutdown()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
