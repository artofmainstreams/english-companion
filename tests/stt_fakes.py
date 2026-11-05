"""Повторяемые подмены для тестов STT.

Никакого микрофона, интернета и скачивания весов: ``FakeBackend`` —
минимальная реализация контракта ``SttBackend`` (загрузка модели и
распознавание файла). Настоящий ``SttEngine`` работает с ним так же, как
с любым backend'ом: реальная оркестрация (ленивая загрузка модели и её
переиспользование, фоновый поток, сигналы, блокировка повторного запуска,
sync/async затвор, ``shutdown``) проверяется БЕЗ наследования ради подмены
приватных методов.
"""

import time


class FakeBackend:
    """Минимальный backend для тестов: ``load_model()`` + ``transcribe()``.

    Счётчики: ``load_calls`` — сколько раз реально «загружалась» модель;
    ``transcribe_calls`` — сколько раз выполнялось распознавание.
    Результат, задержку и ошибки задаёт тест (``result_text``, ``delay``,
    ``raise_on_transcribe``, ``raise_on_load``) — их можно менять после
    создания (оконные тесты так и делают).
    """

    def __init__(
        self,
        result_text: str = "",
        delay: float = 0.0,
        raise_on_transcribe: BaseException | None = None,
        raise_on_load: BaseException | None = None,
    ) -> None:
        self.result_text = result_text
        self.delay = delay
        self.raise_on_transcribe = raise_on_transcribe
        self.raise_on_load = raise_on_load
        self.load_calls = 0
        self.transcribe_calls = 0
        self.model = "fake-model"

    def load_model(self) -> object:
        """«Загружает модель»: считает вызовы, модель переиспользуется."""
        self.load_calls += 1
        if self.raise_on_load is not None:
            raise self.raise_on_load
        return self.model

    def transcribe(self, model: object, audio_path: str) -> str:
        self.transcribe_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        return self.result_text
