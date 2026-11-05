"""Разовый пробо́вый скрипт: drop vs delay при первом/втором проигрывании.

Файл с двумя маркерами-тонами: A [0.0..0.5), B [1.5..2.0). Микрофон ловит
динамики. Если начало просто ЗАДЕРЖИВАЕТСЯ (start latency), интервал B−A
останется 1.5 с. Если начало ОТБРАСЫВАЕТСЯ (drop), A будет укорочено/
сдвинуто и интервал < 1.5 с.

В одном процессе играем ОДИН И ТОТ ЖЕ файл дважды:
* 1-й раз — холодная загрузка источника (как у пользователя «первое»);
* 2-й раз — кэш QMediaPlayer по URL (как «повторное»).
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


def two_tone(path: str, rate: int = 48000, amp: int = 32000) -> None:
    A0, A1, B0, B1, TOTAL = 0.0, 0.5, 1.5, 2.0, 3.2
    n = int(rate * TOTAL)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(n):
            t = i / rate
            if (A0 <= t < A1) or (B0 <= t < B1):
                frames.append(struct.pack("<h", int(amp * math.sin(2 * math.pi * 1000 * t / rate))))
            else:
                frames.append(struct.pack("<h", 0))
        w.writeframes(b"".join(frames))


def analyze(path: str):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "s16le", "-ar", "48000", "-ac", "1", "-"],
        capture_output=True, check=True,
    ).stdout
    s = struct.unpack(f"<{len(out) // 2}h", out)
    blk = 480  # 10 мс
    env = []
    for k in range(0, len(s) - blk, blk):
        b = s[k : k + blk]
        env.append((k / 48000, math.sqrt(sum(x * x for x in b) / blk)))
    peak = max(r for _, r in env)
    floor = max([r for _, r in env[:8]], default=10) or 10
    thr = max(floor * 4, 60)
    hits = [t for t, r in env if r > thr]
    return len(s) / 48000, hits, peak, floor, thr


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="dropdelay-")
    f = os.path.join(tmp, "markers.wav")
    two_tone(f)

    m = AudioController()
    out = QAudioOutput(QMediaDevices.defaultAudioOutput())
    player = QMediaPlayer()
    player.setAudioOutput(out)

    for attempt in (1, 2):
        m.record()
        pump(0.4)
        t_play = time.monotonic()
        player.setSource(QUrl.fromLocalFile(f))
        player.play()
        pump(3.8)
        player.stop()
        m.stop_recording()
        pump(1.0)
        cap = m.last_recording_path
        dur, hits, peak, floor, thr = analyze(cap)
        cap_start = t_play - 0.4
        if len(hits) >= 2:
            a1, a2 = hits[0], hits[1]
            # маркеры: A-тело начинается на hits[0]; B-тело — первый hit
            # после разрыва > 0.3 с
            b_start = None
            prev = a1
            for h in hits[1:]:
                if h - prev > 0.3:
                    b_start = h
                    break
                prev = h
            if b_start is not None:
                gap = b_start - a1
                delay_a = a1 - 0.4  # захват начался за 0.4 с до play()
                print(
                    f"попытка {attempt}: A={a1:.2f} B={b_start:.2f} (файл-время) | "
                    f"интервал B−A = {gap:.2f} с (идеал 1.5) | "
                    f"задержка A от play() ≈ {delay_a:.2f} с | peak={peak:.0f} thr={thr:.0f}"
                )
            else:
                print(f"попытка {attempt}: B не найден; hits={hits[:6]} dur={dur:.2f}")
        else:
            print(f"попытка {attempt}: мало маркеров hits={hits} dur={dur:.2f} peak={peak:.0f} floor={floor:.0f}")
    m.shutdown()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
