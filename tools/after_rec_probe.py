"""Разовый пробо́вый скрипт: воспроизведение сразу после остановки записи.

Точно реплика сценария пользователя: запись → стоп → сразу «Прослушать».
Дискриминант: ТЕЛО маркерного тона (0.8 с) в захвате:
* delay  — onset сдвинут, тело полное (≈0.8 с);
* drop   — начало тона отсутствует, тело укорочено (< 0.8 с).

S1: воспроизведение через 0.2 с после stop_recording (сразу, как у пользователя);
S2: то же, но через 2.0 с после stop (повторное);
S3: воспроизведение без записи вообще (контроль).
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


def tone(path: str, t0: float = 0.0, dur: float = 0.8, rate: int = 48000, amp: int = 32000) -> None:
    n = int(rate * (t0 + dur + 0.3))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(n):
            t = i / rate
            frames.append(
                struct.pack("<h", int(amp * math.sin(2 * math.pi * 1000 * t / rate)))
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
    floor = max([r for _, r in env[:10]], default=10) or 10
    thr = max(floor * 4, 60)
    hits = [t for t, r in env if r > thr]
    body = (hits[-1] - hits[0]) if hits else None
    return len(s) / 48000, (hits[0] if hits else None), body, peak, floor, thr


def pos_timeline(player: QMediaPlayer, t0: float) -> list:
    return [(s[0] - t0, s[1]) for s in []]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="afterrec-")
    m = os.path.join(tmp, "marker.wav")
    tone(m)

    mic = AudioController()
    out = QAudioOutput(QMediaDevices.defaultAudioOutput())
    player = QMediaPlayer()
    player.setAudioOutput(out)
    url = QUrl.fromLocalFile(m)

    pos_samples: list[tuple[float, int]] = []
    timer = QTimer()
    t0_holder = [0.0]

    def tick() -> None:
        pos_samples.append((time.monotonic() - t0_holder[0], player.position()))

    def scenario(label: str, idle_after_stop: float) -> None:
        # «запись» (реальный микрофон)
        mic.record()
        pump(1.0)
        mic.stop_recording()
        pump(idle_after_stop)

        # захват + воспроизведение
        pos_samples.clear()
        timer.timeout.disconnect(tick)
        timer.timeout.connect(tick)
        timer.start(100)
        mic.record()
        pump(0.4)
        t0_holder[0] = time.monotonic()
        player.setSource(url)
        player.play()
        pump(2.0)
        player.stop()
        timer.stop()
        mic.stop_recording()
        pump(1.0)
        cap = mic.last_recording_path
        dur, onset, body, peak, floor, thr = analyze(cap)
        first_nz = next(((round(dt, 2), p) for dt, p in pos_samples if p > 0), None)
        print(
            f"{label}: onset={onset if onset is None else round(onset, 2)} с от старта захвата "
            f"(play через 0.4 с) → старт тона от play() ≈ "
            f"{round(onset - 0.4, 2) if onset is not None else None} с; "
            f"тело={round(body, 2) if body is not None else None} с (идеал 0.8); "
            f"player: первый pos>0 {first_nz} (с, мс); peak={peak:.0f} thr={thr:.0f}"
        )
        # кэш URL QMediaPlayer сбрасываем через новый источник? нет — оставляем,
        # чтобы S2/S3 играли «тот же файл» (как у пользователя)

    scenario("S1 (play через 0.2с после stop):", 0.2)
    scenario("S2 (play через 2.0с после stop):", 2.0)
    # S3: без записи — просто захват+play
    pos_samples.clear()
    timer.timeout.connect(tick)
    timer.start(100)
    mic.record()
    pump(0.4)
    t0_holder[0] = time.monotonic()
    player.setSource(url)
    player.play()
    pump(2.0)
    player.stop()
    timer.stop()
    mic.stop_recording()
    pump(1.0)
    cap = mic.last_recording_path
    dur, onset, body, peak, floor, thr = analyze(cap)
    first_nz = next(((round(dt, 2), p) for dt, p in pos_samples if p > 0), None)
    print(
        f"S3 (без записи): onset={onset if onset is None else round(onset, 2)} с → "
        f"старт тона от play() ≈ {round(onset - 0.4, 2) if onset is not None else None} с; "
        f"тело={round(body, 2) if body is not None else None} с (идеал 0.8); "
        f"player: первый pos>0 {first_nz}; peak={peak:.0f} thr={thr:.0f}"
    )

    mic.shutdown()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
