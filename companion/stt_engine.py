"""Локальное распознавание речи (Whisper).

Бэкенд выбирается в ``AppConfig.stt_backend``:
  * ``"mlx"`` (по умолчанию) — mlx-whisper на Apple Silicon;
  * ``"faster-whisper"`` — альтернатива, явно выбираемая.

Автоматического fallback между бэкендами нет: неизвестный бэкенд или
отсутствующие зависимости — понятная ошибка, а не тихая смена движка.

Компонент отделён от интерфейса: ``recognize()`` запускает фоновый поток, в
котором лениво (один раз) грузится и переиспользуется модель, а само
распознавание выполняется вне UI-потока. Результат возвращается в поток Qt
через сигналы.

Тяжёлые вызовы вынесены в ``_load_model``/``_load_mlx_model``/
``_load_whisper_model`` и ``_run_transcribe``: обычные тесты подменяют их
(или наследуют класс) и не касаются ни микрофона, ни интернета, ни
скачивания весов.
"""

import logging
import os
import threading

from PySide6.QtCore import QObject, Signal

from .config import AppConfig

logger = logging.getLogger(__name__)


class SttEngine(QObject):
    """Распознаёт речь локально и безопасно относительно UI-потока."""

    # Поддерживаемые бэкенды. Других нет — это же сообщение попадёт в
    # ошибку при некорректном ``stt_backend``.
    SUPPORTED_BACKENDS = ("mlx", "faster-whisper")

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
        состояния захватывается только короткими обновлениями состояния).
        """
        model = self._ensure_model()
        self._set_state(self.TRANSCRIBING, "Распознаю речь…")
        return self._run_transcribe(model, audio_path).strip()

    def _ensure_model(self):
        """Возвращает модель, загрузив её лениво (один раз)."""
        with self._model_lock:
            if self._model is None:
                self._model = self._load_model()
            return self._model

    # --- Точки подмены для тестов ------------------------------------------

    def _pick_backend_loader(self):
        """Возвращает функцию создания модели выбранного бэкенда.

        Без fallback: неизвестный бэкенд — явная ошибка. Этот шаг обязателен
        и для подмен в тестах: фейк наследует проверку конфига.
        """
        backend = self.config.stt_backend
        if backend == "mlx":
            return self._load_mlx_model
        if backend == "faster-whisper":
            return self._load_whisper_model
        raise ValueError(
            f"Неизвестный STT-бэкенд: {backend!r}. "
            f"Выберите один из: {', '.join(self.SUPPORTED_BACKENDS)}."
        )

    def _load_model(self):
        """Создаёт модель выбранного бэкенда (точка подмены в тестах)."""
        return self._pick_backend_loader()()

    def _run_transcribe(self, model, audio_path: str) -> str:
        """Распознаёт файл бэкендом из конфига. Выполняется в рабочем потоке.

        Точка подмены в тестах. ``model`` — результат ``_load_model``
        (у faster-whisper имитируется и переиспользуется напрямую; у MLX
        модель уже кэширована в ModelHolder, см. ``_run_transcribe_mlx``).
        """
        if self.config.stt_backend == "mlx":
            return self._run_transcribe_mlx(model, audio_path)
        return self._run_transcribe_whisper(model, audio_path)

    # --- MLX (mlx-whisper) --------------------------------------------------

    @staticmethod
    def _require_mlx_imports() -> None:
        """Гарантирует наличие зависимостей MLX.

        При отсутствии — понятная ошибка с подсказкой по установке
        (без тихого перехода на другой бэкенд).
        """
        try:
            import mlx.core  # noqa: F401
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Бэкенд MLX недоступен: не установлены зависимости mlx / "
                "mlx-whisper. Установка (macOS, arm64): "
                "python -m pip install '.[mlx]' "
                "(плюс системный ffmpeg). "
                "Или выберите stt_backend='faster-whisper' в AppConfig."
            ) from exc

    def _load_mlx_model(self):
        """Создаёт MLX-модель (mlx-whisper). Ленивый импорт: без весов.

        ``ModelHolder`` кэширует модель по пути репозитория — повторная
        загрузка не происходит, ``_run_transcribe_mlx`` переиспользует уже
        загруженную модель. fp16 — стандартная точность Whisper
        (не квантизация); веса скачиваются только при первом запуске.
        """
        self._require_mlx_imports()
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        return ModelHolder.get_model(self.config.stt_mlx_model, mx.float16)

    def _run_transcribe_mlx(self, model, audio_path: str) -> str:
        """Распознаёт файл через mlx-whisper (в рабочем потоке).

        ``language``/``task`` берутся из AppConfig; текста за рамками самой
        Whisper-модели никаких исправлений не делается. Аудио декодируется
        системным ffmpeg (mlx-whisper вызывает его подпроцессом).
        """
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.config.stt_mlx_model,
            language=self.config.stt_language,
            task=self.config.stt_task,
        )
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return (text or "").strip()

    # --- faster-whisper -----------------------------------------------------

    def _load_whisper_model(self):
        """Создаёт Whisper-модель. Ленивый импорт: без скачивания весов.

        Тесты подменяют этот метод (и ``_run_transcribe``), чтобы не тянуть
        ни faster-whisper, ни интернет.
        """
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.config.stt_whisper_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
        )

    def _run_transcribe_whisper(self, model, audio_path: str) -> str:
        """Распознаёт файл faster-whisper. Выполняется в рабочем потоке."""
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
