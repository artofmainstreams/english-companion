from dataclasses import dataclass

from .app_data import tts_reference_path
from .tts_qwen3 import (
    DEFAULT_INSTRUCT,
    DEFAULT_MODEL_REFERENCE,
    DEFAULT_MODEL_VOICE_DESIGN,
    DEFAULT_REF_TEXT,
)
from .tts_synth import DEFAULT_VOICE


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_ms: int = 10_000
    chat_timeout_ms: int = 180_000

    # Озвучивание ответов преподавателя (TTS).
    # tts_engine: "qwen3" (по умолчанию; локальная Qwen3-TTS, MLX; синтез —
    # фоновый поток, голос — референс p4; см. README) или "say"
    # (macOS say, голос — tts_voice) как резервный движок.
    tts_engine: str = "qwen3"
    # Английский голос macOS say (только tts_engine="say"). Имя
    # проверяется списком голосов: ``say -v '?'``. Если голоса на системе
    # нет, say молча возьмёт голос по умолчанию — это поведение macOS,
    # компонент синтеза на него не влияет.
    tts_voice: str = DEFAULT_VOICE

    # Qwen3-TTS (tts_engine="qwen3"):
    # tts_qwen3_method: "reference" (по умолчанию: модель Base + одна
    # чистая реплика tts_qwen3_ref_audio и её точный текст
    # tts_qwen3_ref_text — одобренный голос p4) или "voice_design"
    # (модель + описание голоса tts_qwen3_instruct).
    # Weights — в кэше Hugging Face (~4 GiB), вне репозитория.

    tts_qwen3_method: str = "reference"
    tts_qwen3_model_voice_design: str = DEFAULT_MODEL_VOICE_DESIGN
    tts_qwen3_model_reference: str = DEFAULT_MODEL_REFERENCE
    # Описание голоса — выбранный голос D (образец qwen3_low_D.wav);
    # используется, только если tts_qwen3_method == "voice_design".
    tts_qwen3_instruct: str = DEFAULT_INSTRUCT
    # Референс способа "reference": файл кладётся в постоянную папку
    # данных приложения (переживает переустановку; установка —
    # tools/install_tts_reference.py). При отсутствии файла синтез
    # завершается понятной ошибкой (см. tts_qwen3.py).
    tts_qwen3_ref_audio: str = str(tts_reference_path())
    tts_qwen3_ref_text: str = DEFAULT_REF_TEXT

    # Потоковый вывод (речь частями; Qwen3TtsSynth по умолчанию):
    # tts_qwen3_streaming — включить (выключено → полный WAV, как раньше);
    # tts_qwen3_streaming_interval — размер куска в секундах (0.5 —
    # проверено: TTFS ~0.7 c, без недочётов; см. real_tts_stream_app.py);
    # tts_qwen3_start_buffer_ms — стартовый буфер (мс) вывода: вывод
    # стартует, когда накоплено не меньше этого аудио (защита от недочётов).
    tts_qwen3_streaming: bool = True
    tts_qwen3_streaming_interval: float = 0.5
    tts_qwen3_start_buffer_ms: int = 500

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
