"""Локальное распознавание речи на faster-whisper.

Компонент отделён от интерфейса: ``recognize()`` запускает фоновый поток, в
котором лениво (один раз) грузится и переиспользуется модель Whisper, а
само распознавание и обход ленивого генератора ``segments`` выполняются вне
UI-потока. Результат возвращается в поток Qt через сигналы.

Тяжёлые вызовы вынесены в ``_load_whisper_model`` и ``_run_transcribe``:
обычные тесты подменяют их (или наследуют класс) и не касаются ни микрофона,
ни интернета, ни скачивания весов.
"""

import logging
import os
import threading

from PySide6.QtCore import QObject, Signal

from .config import AppConfig

logger = logging.getLogger(__name__)


class SttEngine(QObject):
    """Распознаёт речь локально и безопасно относительно UI-потока."""

    state_changed = Signal()
    recognized = Signal(str)
    failed = Signal(str)

    IDLE = "idle"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.state = self.IDLE
        self.status = "Распознавание: готов"

        # Модель грузится лениво один раз и переиспользуется.
        self._model = None
        self._model_lock = threading.Lock()
        # Защита состояния от гонок UI-поток ↔ рабочий поток.
        self._state_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._shutdown = False

    # --- Публичный API ------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        with self._state_lock:
            return self.state != self.IDLE

    @property
    def model_loaded(self) -> bool:
        with self._model_lock:
            return self._model is not None

    def recognize(self, audio_path: str | None) -> None:
        """Запускает распознавание файла в фоновом потоке.

        Вызов ожидается из UI-потока. Повторный запуск во время работы
        игнорируется (кнопка в окне тоже заблокирована). Путь передаётся из
        окна как публичный результат ``AudioController.last_recording_path``.
        """
        if self.is_busy:
            return
        if audio_path is None or not os.path.exists(audio_path):
            self.failed.emit("Файл аудио недоступен для распознавания.")
            return
        if not self._claim():
            return
        self._worker = threading.Thread(
            target=self._run, args=(audio_path,), daemon=True, name="stt-worker"
        )
        self._worker.start()

    def recognize_sync(self, audio_path: str) -> str:
        """Синхронное распознавание — для проверок без цикла событий.

        Использует ту же логику (ленивая загрузка модели + распознавание), но
        выполняется в вызывающем потоке.
        """
        return self._do_recognize(audio_path)

    def shutdown(self) -> None:
        """Останавливает приём новых работ и дожидается рабочего потока.

        Вычисление не прерывается намеренно: рабочий поток завершается сам
        (и лишь после этого файл можно удалять). Идемпотентна; безопасна из
        UI-потока и из ``aboutToQuit``.
        """
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True

        worker = self._worker
        if (
            worker is not None
            and worker is not threading.current_thread()
            and worker.is_alive()
        ):
            # Дожидаемся завершения: после этого файл больше не читается.
            worker.join()
        self._worker = None

        with self._state_lock:
            self.state = self.IDLE
            self.status = "Распознавание: остановлено"
        self.state_changed.emit()

    # --- Внутренняя оркестрация --------------------------------------------

    def _claim(self) -> bool:
        """Атомарно захватывает «занято». False — уже работаем или завершаем."""
        with self._state_lock:
            if self._shutdown or self.state != self.IDLE:
                return False
            self.state = self.LOADING
            self.status = "Загружаю модель распознавания…"
        # Сообщаем UI немедленно: кнопки блокируются, даже пока не стартовал
        # рабочий поток.
        self.state_changed.emit()
        return True

    def _run(self, audio_path: str) -> None:
        """Тело рабочего потока: загрузка модели + распознавание."""
        try:
            model = self._ensure_model()
            self._set_state(self.TRANSCRIBING, "Распознаю речь…")
            text = self._run_transcribe(model, audio_path).strip()
            self.recognized.emit(text)
            self._set_state(self.IDLE, "Распознавание: готов")
        except Exception as exc:  # noqa: BLE001 — сообщаем любую причину в UI
            logger.exception("Распознавание не удалось")
            self._set_state(self.IDLE, "Ошибка распознавания")
            self.failed.emit(f"Не удалось распознать речь: {exc}")

    def _do_recognize(self, audio_path: str) -> str:
        model = self._ensure_model()
        return self._run_transcribe(model, audio_path)

    def _ensure_model(self):
        """Возвращает модель, загрузив её лениво (один раз)."""
        with self._model_lock:
            if self._model is None:
                self._model = self._load_whisper_model()
            return self._model

    # --- Точка подмены для тестов ------------------------------------------

    def _load_whisper_model(self):
        """Создаёт Whisper-модель. Ленивый импорт: без скачивания весов.

        Тесты подменяют этот метод (и ``_run_transcribe``), чтобы не тянуть
        ни faster-whisper, ни интернет.
        """
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.config.stt_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
        )

    def _run_transcribe(self, model, audio_path: str) -> str:
        """Распознаёт файл и собирает текст. Выполняется в рабочем потоке."""
        segments, _info = model.transcribe(
            audio_path,
            language=self.config.stt_language,
            task=self.config.stt_task,
            vad_filter=self.config.stt_vad_filter,
        )
        # ``segments`` — ленивый генератор: обход идёт здесь, в рабочем
        # потоке, и не блокирует UI.
        parts = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _set_state(self, state: str, status: str) -> None:
        with self._state_lock:
            self.state = state
            self.status = status
        self.state_changed.emit()
