"""Повторяемые подмены для тестов STT.

Никакого микрофона, интернета и скачивания весов: ``FakeSttEngine`` наследует
настоящий ``SttEngine`` (реальная оркестрация: ленивая загрузка модели,
фоновый поток, сигналы, блокировка повторного запуска) и подменяет только
создание Whisper-модели. ``FakeModel`` имитирует API ``faster-whisper``:
``transcribe()`` возвращает ленивый генератор сегментов.
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
    """Re-использует реальную логику SttEngine, подменяя только модель."""

    def __init__(
        self,
        config,
        parent=None,
        result_text: str = "",
        delay: float = 0.0,
        raise_on_transcribe: BaseException | None = None,
    ) -> None:
        super().__init__(config, parent)
        self.result_text = result_text
        self.delay = delay
        self.raise_on_transcribe = raise_on_transcribe
        self._fake_model = None

    def _load_whisper_model(self):
        if self._fake_model is None:
            self._fake_model = FakeModel(
                self.result_text, self.delay, self.raise_on_transcribe
            )
        return self._fake_model
