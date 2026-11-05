"""Локальное распознавание речи (Whisper).

``SttEngine`` управляет фоновой работой и жизненным циклом: фоновый поток,
сигналы в Qt, ленивая загрузка и переиспользование модели, блокировка
повторного запуска, общий sync/async затвор (не больше одного
распознавания одновременно), идемпотентный ``shutdown`` и отложенное
закрытие окна. Конкретные библиотеки живут в backend'е
(см. :mod:`companion.stt_backends`): движок знает только минимальный
контракт ``SttBackend`` (загрузка модели + распознавание файла) и не знает,
mlx-whisper это или faster-whisper.

Реализация backend'а выбирается при сборке приложения (``main.py`` →
``build_backend(config)``). Автоматического fallback нет: неизвестный
бэкенд или отсутствующие зависимости — понятная ошибка, а не тихая смена
движка.

Оркестрация проверяется с подменным backend'ом (``tests/stt_fakes.py``):
без микрофона, интернета и скачивания весов, и без наследования ради
подмены приватных методов.
"""

import logging
import os
import threading

from PySide6.QtCore import QObject, Signal

from .stt_backends import SttBackend

logger = logging.getLogger(__name__)


class SttEngine(QObject):
    """Распознаёт речь локально и безопасно относительно UI-потока."""

    state_changed = Signal()
    recognized = Signal(str)
    failed = Signal(str)

    IDLE = "idle"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"

    def __init__(self, backend: SttBackend, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.state = self.IDLE
        self.status = "Распознавание: готов"

        # Модель грузится лениво один раз и переиспользуется.
        self._model = None
        self._model_lock = threading.Lock()
        # Защита состояния от гонок UI-поток ↔ рабочий поток.
        self._state_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._shutdown = False
        # Общий «работает» затвор для sync/async: не больше одного
        # распознавания одновременно, независимо от пути. ``_inflight`` —
        # сколько работ в работе (0/1); ``_inflight_done`` — сброшен, пока
        # работа идёт, и устанавливается при её завершении, чтобы shutdown
        # мог чисто дождаться её (без busy-wait и terminate).
        self._inflight = 0
        self._inflight_done = threading.Event()
        self._inflight_done.set()

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

        Вызов ожидается из UI-потока. Повторный запуск во время работы и
        после ``shutdown`` отклоняются (кнопка в окне тоже заблокирована).
        Путь передаётся из окна как публичный результат
        ``AudioController.last_recording_path``.
        """
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

        Та же общая защита, что у асинхронного пути: после ``shutdown`` и
        во время другой работы отклоняется (``RuntimeError``); исключение из
        распознавания пробрасывается, но не оставляет движок «занятым».
        Выполняется в вызывающем потоке.
        """
        if audio_path is None or not os.path.exists(audio_path):
            raise RuntimeError("Файл аудио недоступен для распознавания.")
        if not self._claim():
            raise RuntimeError(
                "Распознавание недоступно: движок занят или остановлен."
            )
        release_status = "Распознавание: готов"
        try:
            return self._do_recognize(audio_path)
        except Exception:
            release_status = "Ошибка распознавания"
            raise
        finally:
            # И в успехе, и в сбое — отпускаем «работает» затвор, чтобы
            # движок не остался занятым.
            self._release(release_status)

    def shutdown(self) -> None:
        """Останавливает приём новых работ и дожидается уже начатых.

        Вычисление не прерывается намеренно (без ``terminate``): начатая работа
        завершается сама (и лишь после этого файл можно удалять), а ``shutdown``
        чисто дожидается её через ``_inflight_done`` (не busy-wait). Учитывает
        обе ветки — асинхронный поток и синхронный вызов в другом потоке.
        Идемпотентна; безопасна из UI-потока и из ``aboutToQuit``.

        Отзывчивость закрытия обеспечивает окно: пока распознавание идёт,
        ``closeEvent`` откладывает закрытие (``event.ignore``), и ``shutdown``
        вызывается после завершения работы.
        """
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True

        # Дожидаемся завершения уже начатых работ (sync/async). Если работ
        # нет — событие уже установлено и wait() возвращается сразу.
        self._inflight_done.wait()
        self._worker = None

        with self._state_lock:
            self.state = self.IDLE
            self.status = "Распознавание: остановлено"
        self.state_changed.emit()

    # --- Внутренняя оркестрация --------------------------------------------

    def _claim(self) -> bool:
        """Атомарно захватывает единственную «работает» ячейку (sync/async).

        Возвращает ``False``, если движок остановлен, уже идёт работа или
        состояние уже не ``IDLE``. При успехе устанавливает состояние
        ``LOADING``, помечает работу в работе и сбрасывает ``_inflight_done``.
        Блокировку состояния держит только на время этого обновления — тяжёлые
        вызовы (загрузка модели, распознавание) выполняются вне её.
        """
        with self._state_lock:
            if self._shutdown or self.state != self.IDLE:
                return False
            self._inflight += 1
            self._inflight_done.clear()
            self.state = self.LOADING
            self.status = "Загружаю модель распознавания…"
        # Сообщаем UI немедленно: кнопки блокируются, даже пока не стартовал
        # рабочий поток.
        self.state_changed.emit()
        return True

    def _release(self, status: str = "Распознавание: готов") -> None:
        """Отпускает «работает» ячейку и возвращает состояние ``IDLE``.

        Идемпотентна: повторный вызов не уводит счётчик ниже нуля и не
        сбрасывает ``_inflight_done`` повторно.
        """
        with self._state_lock:
            if self._inflight > 0:
                self._inflight -= 1
            if self._inflight == 0:
                self._inflight_done.set()
            self.state = self.IDLE
            self.status = status
        self.state_changed.emit()

    def _run(self, audio_path: str) -> None:
        """Тело рабочего потока: загрузка модели + распознавание."""
        try:
            text = self._do_recognize(audio_path)
            self.recognized.emit(text)
            self._release("Распознавание: готов")
        except Exception as exc:  # noqa: BLE001 — сообщаем любую причину в UI
            logger.exception("Распознавание не удалось")
            # Порядок важен: сначала отпускаем затвор (state_changed обновила
            # UI), затем failed — её сообщение становится ПОСЛЕДНИМ текстом
            # статусной строки окна (окно пишет status при state_changed).
            self._release("Ошибка распознавания")
            self.failed.emit(f"Не удалось распознать речь: {exc}")

    def _do_recognize(self, audio_path: str) -> str:
        """Загрузка модели + распознавание. Вызывается ПОСЛЕ успешного ``_claim``.

        Тяжёлые вызовы выполняются БЕЗ удержания ``_state_lock`` (блокировка
        состояния захватывается только короткими обновлениями состояния);
        сами тяжёлые вызовы (загрузка модели, распознавание) делает backend.
        """
        model = self._ensure_model()
        self._set_state(self.TRANSCRIBING, "Распознаю речь…")
        return self.backend.transcribe(model, audio_path).strip()

    def _ensure_model(self):
        """Возвращает модель, загрузив её лениво (один раз)."""
        with self._model_lock:
            if self._model is None:
                self._model = self.backend.load_model()
            return self._model

    def _set_state(self, state: str, status: str) -> None:
        with self._state_lock:
            self.state = state
            self.status = status
        self.state_changed.emit()
