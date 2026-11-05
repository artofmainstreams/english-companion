from dataclasses import dataclass

from .tts_synth import DEFAULT_VOICE


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_ms: int = 10_000
    chat_timeout_ms: int = 180_000

    # Озвучивание ответов преподавателя (TTS): английский голос macOS say.
    # Имя проверяется списком голосов: ``say -v '?'``. Если голоса на системе
    # нет, say молча возьмёт голос по умолчанию — это поведение macOS,
    # компонент синтеза на него не влияет.
    tts_voice: str = DEFAULT_VOICE

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
    # MULTILINGUAL модель (base), а не base.en: при автоматическом определении
    # языка (stt_language=None) модель должна уметь и русский, и английский.
    # .en-варианты — только английский: они ломают «русский остаётся русским».
    stt_whisper_model: str = "base"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"

    # Общие параметры распознавания.
    # stt_language: язык речи. По умолчанию None — Whisper определяет язык
    # САМ (по первому окну записи): русская речь -> русский текст,
    # английская -> английский. Язык можно зафиксировать явно ("en", "ru"),
    # но тогда речь на другом языке распознаётся «в лоб», без перевода.
    stt_language: str | None = None
    # transcribe — расшифровать речь как есть; НЕ translate (перевод).
    stt_task: str = "transcribe"
    # Только для faster-whisper (у MLX этого параметра нет).
    stt_vad_filter: bool = True
