"""Бэкенды локального распознавания речи (Whisper).

Минимальный типизированный контракт ``SttBackend`` (``typing.Protocol``):
загрузка/подготовка модели и распознавание аудиофайла. Конкретные
реализации знают свои параметры из ``AppConfig`` и лениво импортируют
опциональные библиотеки:

  * ``MlxBackend`` — mlx-whisper на Apple Silicon (по умолчанию);
  * ``FasterWhisperBackend`` — faster-whisper (CTranslate2), явно выбираемый.

Отсутствие зависимостей — понятная ошибка, а не тихая смена движка:
автоматического fallback нет. Реализация выбирается при сборке
приложения (``build_backend``), а не внутри оркестрации ``SttEngine``.
"""

from typing import Any, Protocol

from .config import AppConfig

# Поддерживаемые бэкенды. Других нет — это же список попадёт в ошибку
# при некорректном ``stt_backend``.
SUPPORTED_BACKENDS = ("mlx", "faster-whisper")


class SttBackend(Protocol):
    """Минимальный контракт backend'а STT.

    Реализация должна быть потокобезопасной в смысле «один вызов за раз»:
    ``SttEngine`` гарантирует, что одновременно выполняется не больше
    одного распознавания (общий sync/async затвор).
    """

    def load_model(self) -> Any:
        """Загружает/подготавливает модель. Вызывается лениво, один раз.

        При отсутствии зависимостей — понятная ошибка (без fallback).
        """
        ...

    def transcribe(self, model: Any, audio_path: str) -> str:
        """Распознаёт аудиофайл и возвращает текст.

        Выполняется в рабочем потоке ``SttEngine`` (вне UI-потока).
        """
        ...


class MlxBackend:
    """mlx-whisper на Apple Silicon (fp16 — стандартная точность Whisper)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def _require_imports() -> None:
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

    def load_model(self) -> Any:
        """Создаёт MLX-модель (mlx-whisper). Ленивый импорт: без весов.

        ``ModelHolder`` кэширует модель по пути репозитория — повторная
        загрузка не происходит, ``transcribe`` переиспользует уже
        загруженную модель. fp16 — стандартная точность Whisper
        (не квантизация); веса скачиваются только при первом запуске.
        """
        self._require_imports()
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        return ModelHolder.get_model(self.config.stt_mlx_model, mx.float16)

    def transcribe(self, model: Any, audio_path: str) -> str:
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


class FasterWhisperBackend:
    """faster-whisper (CTranslate2): параметры модели из AppConfig."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load_model(self) -> Any:
        """Создаёт Whisper-модель. Ленивый импорт: без скачивания весов."""
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.config.stt_whisper_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
        )

    def transcribe(self, model: Any, audio_path: str) -> str:
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


def build_backend(config: AppConfig) -> SttBackend:
    """Выбирает реализацию по ``AppConfig.stt_backend``.

    Вызывается при сборке приложения (main.py), а не в оркестрации
    движка. Без fallback: неизвестный бэкенд — явная ошибка, а не
    тихая смена движка.
    """
    if config.stt_backend == "mlx":
        return MlxBackend(config)
    if config.stt_backend == "faster-whisper":
        return FasterWhisperBackend(config)
    raise ValueError(
        f"Неизвестный STT-бэкенд: {config.stt_backend!r}. "
        f"Выберите один из: {', '.join(SUPPORTED_BACKENDS)}."
    )
