"""Диагностика «обрыв начала первого воспроизведения».

Фиксирует таймлайн событий контроллера (включая position) для двух сценариев,
чтобы различить причину:

  S1 — «первое проигрывание готового файла БЕЗ предварительной записи»
       (свежий процесс, записи не было). Если здесь начало тоже «падает» —
       причина в холодном BT-выходе. Если нет — причина в переходе
       «ввод(гарнитура) → выход(гарнитура)» сразу после записи.

  S2 — «сразу после записи» (запись → стоп → сразу play). Точная реплика
       сценария пользователя.

Позиция плеера сэмплируется каждые ~25 мс. Логгеры аудиослоя (INFO)
выдают дискретные события: record:stop-requested, record:finalized,
record:teardown, play:start, play:stop, mediaStatus, playbackState,
player_error.

ВАЖНО: этот скрипт показывает МЕХАНИКУ (последовательность, position, ошибки).
Он НЕ подтверждает слышимость начала — это проверяется только на наушниках
(см. ручную инструкцию).
"""

import logging
import math
import os
import struct
import sys
import tempfile
import time
import wave

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from companion.audio_controller import AudioController  # noqa: E402
from PySide6.QtMultimedia import QMediaDevices  # noqa: E402

app = QApplication(sys.argv)


def pump(s: float) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(s * 1000), loop.quit)
    loop.exec()


def make_recording(path: str, dur: float = 2.0, rate: int = 48000) -> None:
    """Файл «записи»: тон 440 Гц, чтобы у записи был реальный контент."""
    n = int(rate * dur)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(n)
            )
        )


def default_output_desc() -> str:
    d = QMediaDevices.defaultAudioOutput()
    return d.description() if not d.isNull() else "<none>"


class PositionSampler:
    def __init__(self, controller: AudioController, period_ms: int = 25) -> None:
        self.c = controller
        self.samples: list[tuple[int, int]] = []
        self.t0 = controller.player._t0  # та же база, что у _log_event плеера
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._period = period_ms

    def _tick(self) -> None:
        p = (
            self.c.player._player.position()
            if self.c.player._player is not None
            else -1
        )
        self.samples.append((int((time.monotonic() - self.t0) * 1000), p))

    def start(self) -> None:
        self.samples = []
        self._timer.start(self._period)

    def stop(self) -> None:
        self._timer.stop()


def summarize(samples: list[tuple[int, int]]) -> None:
    if not samples:
        print("  position: нет сэмплов")
        return
    first_nz = next(((t, p) for t, p in samples if p > 0), None)
    last = samples[-1]
    print(
        "  position: первый>0 "
        + (f"t={first_nz[0]}мс pos={first_nz[1]}мс" if first_nz else "НЕТ (остался 0)")
        + f" | конец pos={last[1]}мс"
    )
    # компактный график каждый ~200 мс
    print("  timeline(t→pos, мс): " + " ".join(f"{t}:{p}" for t, p in samples[::8]))


def scenario(label: str, capture_first: bool) -> None:
    print(f"\n===== {label} =====")
    print(f"  дефолтный вывод: {default_output_desc()}")
    c = AudioController()
    tmp = tempfile.mkdtemp(prefix="headdiag-")
    rec = os.path.join(tmp, "rec.wav")
    make_recording(rec)
    c.recorder._temp_dir = tmp
    c.recorder._record_file = rec
    # С1: играем «готовую» запись (без предшествующей записи).
    c.recorder._ready_recording = rec

    sampler = PositionSampler(c)

    if capture_first:
        # Сценарий S2: запись → стоп → сразу play
        c.record()
        pump(1.2)
        c.stop_recording()
        pump(1.0)
        print(f"  после записи: state={c.state} has_recording={c.has_recording}")

    # Плей
    c.play()
    sampler.start()
    pump(3.5)
    sampler.stop()
    c.stop_playback()
    pump(0.3)
    playback = (
        c.player._player.playbackState()
        if c.player._player is not None
        else "N/A"
    )
    print(f"  итог: state={c.state} playbackState={playback} has_recording={c.has_recording}")
    summarize(sampler.samples)

    c.shutdown()
    pump(0.3)


def main() -> int:
    inputs = [d.description() for d in QMediaDevices.audioInputs() if not d.isNull()]
    print("Входы:", inputs)
    print("Примечание: position=0 на старте не доказывает, что начало НЕ слышно —")
    print("это уровень учёта плеера. Слышимость проверяется только на наушниках.")

    scenario("S1: первое проигрывание ГОТОВОГО файла (без записи)", capture_first=False)
    scenario("S2: сразу ПОСЛЕ записи (реплика пользователя)", capture_first=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
