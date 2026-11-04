"""Повторяемые подмены для тестов STT.

Никакого микрофона, интернета и скачивания весов: ``FakeSttEngine`` наследует
настоящий ``SttEngine`` (реальная оркестрация: ленивая загрузка модели,
фоновый поток, сигналы, блокировка повторного запуска, shutdown) и
подменяет только создание модели и распознавание — точки диспетчеризации
``_load_model``/``_run_transcribe``. Поэтому подмена одинаково работает для
обоих бэкендов (``mlx`` и ``faster-whisper``).

``FakeModel`` имитирует «модель»: ``transcribe()`` возвращает ленивый
генератор сегментов (API faster-whisper), а задержку/ошибку задаёт тест.
"""

import time

from companion.stt_engine import SttEngine


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    """Имитация Whisper-модели: ``transcribe() -> (итератор сегментов, None)``."""

    def __init__(
        self,
        result_text: str = "",
        delay: float = 0.0,
        raise_on_transcribe: BaseException | None = None,
    ) -> None:
        self._result_text = result_text
        self._delay = delay
        self._raise = raise_on_transcribe
        self.transcribe_calls = 0

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._raise is not None:
            raise self._raise

        def _segments():
            if self._result_text:
                yield FakeSegment(self._result_text)

        # Возвращаем ленивый генератор, как настоящий faster-whisper.
        return _segments(), None


class FakeSttEngine(SttEngine):
    """Re-использует реальную логику SttEngine, подменяя только модель.

    Нейтральна к бэкенду: подменяет ``_load_model`` (создание модели) и
    ``_run_transcribe`` (распознавание), поэтому работает и с mlx-конфигурацией
    по умолчанию, и с faster-whisper. Счётчики: ``model_load_calls`` — сколько
    раз реально создавалась модель; ``_fake_model.transcribe_calls`` — сколько
    раз выполнялось распознавание.
    """

    def __init__(
        self,
        config,
        parent=None,
        result_text: str = "",
        delay: float = 0.0,
        raise_on_transcribe: BaseException | None = None,
        raise_on_load: BaseException | None = None,
    ) -> None:
        super().__init__(config, parent)
        self.result_text = result_text
        self.delay = delay
        self.raise_on_transcribe = raise_on_transcribe
        self.raise_on_load = raise_on_load
        self._fake_model = None
        self.model_load_calls = 0

    def _load_model(self):
        # Валидация бэкенда из конфига обязательна и в тестах (no fallback).
        self._pick_backend_loader()
        self.model_load_calls += 1
        if self.raise_on_load is not None:
            raise self.raise_on_load
        if self._fake_model is None:
            self._fake_model = FakeModel(
                self.result_text, self.delay, self.raise_on_transcribe
            )
        return self._fake_model

    def _run_transcribe(self, model, audio_path):
        # Тот же обход ленивого генератора, что у faster-whisper.
        segments, _info = model.transcribe(audio_path)
        parts = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()
