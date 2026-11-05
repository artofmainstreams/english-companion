"""Разовый пробо́вый скрипт: «холодный» первый проигрыватель vs прогретый.

Замер: тон (1 кГц, 1.0 с) проигрывается через динамик, микрофон ловит его
в запись. Сравниваем, когда тон появляется в записи относительно вызова
play() и сколько тела тона успело вписаться (drop vs delay).

WARM=0 — как сейчас в приложении: свежий QAudioOutput → setSource(тон) → play.
WARM=1 — с прогревом: свежий QAudioOutput → тишина 0.3 с → play → EndOfMedia → тон.

Каждый запуск — свежий процесс (холодный/тёплый звук macOS).
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


def wav(path: str, dur: float, amp: int, t0: float = 0.0, rate: int = 48000) -> None:
    total = int(rate * (t0 + dur + 0.2))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(total):
            t = i / rate
            if amp and t0 <= t <= t0 + dur:
                frames.append(struct.pack("<h", int(amp * math.sin(2 * math.pi * 1000 * (t - t0) / rate))))
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
    blk = 960  # 20 мс
    env = []
    for k in range(0, len(s) - blk, blk):
        b = s[k : k + blk]
        env.append((k / 48000, math.sqrt(sum(x * x for x in b) / blk)))
    peak = max(r for _, r in env)
    floor = max([r for _, r in env[:5]], default=10) or 10
    thr = max(floor * 4, 80)
    hits = [t for t, r in env if r > thr]
    onset = hits[0] if hits else None
    tail = hits[-1] if hits else None
    return len(s) / 48000, onset, tail, peak, floor


def main() -> int:
    warm = os.environ.get("WARM") == "1"
    tmp = tempfile.mkdtemp(prefix="playdrop-")
    tone = os.path.join(tmp, "tone.wav")
    wav(tone, dur=1.0, amp=32000)
    silence = os.path.join(tmp, "silence.wav")
    wav(silence, dur=0.3, amp=0)

    m = AudioController()  # «диктофон»

    # Точный flow приложения: свежий QAudioOutput → setSource → play.
    out = QAudioOutput(QMediaDevices.defaultAudioOutput())
    player = QMediaPlayer()
    player.setAudioOutput(out)

    m.record()
    pump(0.4)

    if warm:
        player.setSource(QUrl.fromLocalFile(silence))
        player.play()
        pump(1.0)  # тишина 0.3 с — прогрев устройства

    t_play = time.monotonic()
    player.setSource(QUrl.fromLocalFile(tone))
    player.play()
    pump(2.6)
    player.stop()

    m.stop_recording()
    pump(1.0)
    cap = m.last_recording_path
    dur, onset, tail, peak, floor = analyze(cap)
    cap_start = t_play - 0.4
    if onset is not None:
        onset_real = cap_start + onset
        print(
            f"{'WARM' if warm else 'COLD'}: onset в записи={onset:.2f} с → "
            f"задержка тона от play() ≈ {onset_real - t_play:.2f} с; "
            f"тело тона до {tail if tail is None else round(tail, 2)} с "
            f"(dur={dur:.2f}, peak={peak:.0f}, floor={floor:.0f})"
        )
    else:
        print(f"{'WARM' if warm else 'COLD'}: тон НЕ обнаружен (dur={dur:.2f})")
    m.shutdown()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
