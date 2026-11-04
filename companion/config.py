from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_ms: int = 10_000
    chat_timeout_ms: int = 180_000

    # Локальное распознавание речи.
    # stt_backend: "mlx" (по умолчанию; macOS arm64 + системный ffmpeg) или
    # "faster-whisper". Автоматического fallback нет: недоступный бэкенд —
    # понятная ошибка, а не тихая смена движка.
    # Модель грузится лениво один раз и переиспользуется; при первой
    # загрузке нужны интернет-веса, далее распознавание полностью локальное.
    stt_backend: str = "mlx"

    # MLX-бэкенд (mlx-whisper; модель — репозиторий HuggingFace).
    stt_mlx_model: str = "mlx-community/whisper-large-v3-mlx"

    # faster-whisper-бэкенд (CTranslate2).
    stt_whisper_model: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"

    # Общие параметры распознавания.
    stt_language: str = "en"
    stt_task: str = "transcribe"
    # Только для faster-whisper (у MLX этого параметра нет).
    stt_vad_filter: bool = True
