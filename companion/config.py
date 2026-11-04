from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_ms: int = 10_000
    chat_timeout_ms: int = 180_000

    # Локальное распознавание речи (faster-whisper).
    # Модель грузится лениво один раз и кэшируется; при первой загрузке
    # нужны интернет-веса, далее распознавание идёт полностью локально.
    stt_model: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "en"
    stt_task: str = "transcribe"
    stt_vad_filter: bool = True
