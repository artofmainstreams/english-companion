from PySide6.QtCore import QObject, Signal

from .lesson import Lesson
from .message import Message
from .ollama_client import OllamaClient
from .teacher_context import build_teacher_context


class SessionController(QObject):
    state_changed = Signal()
    message_added = Signal(object)

    def __init__(self, client: OllamaClient, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.models: list[str] = []
        self.selected_model = ""
        # Урок задаёт системный контекст преподавателя (level, темы,
        # фразы, слова). Без урока отправка недоступна.
        self.lesson: Lesson | None = None
        # Только успешно завершённые обмены попадают в контекст.
        self.history: list[Message] = []
        self.loading_models = False
        self.generating_reply = False
        self.status = "Готов"
        self._pending_message: Message | None = None
        self._request_model = ""
        client.models_loaded.connect(self._on_models_loaded)
        client.models_failed.connect(self._on_models_failed)
        client.chat_completed.connect(self._on_chat_completed)
        client.chat_failed.connect(self._on_chat_failed)

    @property
    def busy(self) -> bool:
        return self.loading_models or self.generating_reply

    @property
    def can_send(self) -> bool:
        return bool(self.selected_model) and self.lesson is not None and not self.busy

    def set_lesson(self, lesson: Lesson) -> None:
        """Выбранный урок — источник системного контекста преподавателя."""
        self.lesson = lesson
        self.state_changed.emit()

    def reset_history(self) -> None:
        """Новое занятие: история чата очищается (урок остаётся выбран)."""
        self.history = []
        self.status = "Готов"
        self.state_changed.emit()

    def load_models(self) -> None:
        if self.busy:
            return
        self.loading_models = True
        self.status = "Загружаю список моделей…"
        self.state_changed.emit()
        self.client.load_models()

    def select_model(self, model: str) -> None:
        if not self.busy and model in self.models and model != self.selected_model:
            self.selected_model = model
            self.state_changed.emit()

    def _on_models_loaded(self, models: list[str]) -> None:
        self.models = list(models)
        if self.selected_model not in models:
            self.selected_model = models[0] if models else ""
        self.loading_models = False
        self.status = (
            f"Готов. Доступно моделей: {len(models)}"
            if models else "В Ollama нет установленных моделей"
        )
        self.state_changed.emit()

    def _on_models_failed(self, error: str) -> None:
        self.loading_models = False
        self.status = error
        self.state_changed.emit()

    def send_message(self, text: str) -> bool:
        text = text.strip()
        if not text or not self.can_send:
            return False
        self._pending_message = Message("user", text)
        self._request_model = self.selected_model
        self.generating_reply = True
        self.status = f"Думаю · {self._request_model}…"
        self.message_added.emit(self._pending_message)
        self.state_changed.emit()
        # Системный контекст преподавателя (урок) уходит вместе с
        # историей, но в чат НЕ попадает — только в запрос к Ollama.
        self.client.send_chat(
            self._request_model,
            [*self.history, self._pending_message],
            system=self.system_context(),
        )
        return True

    def system_context(self) -> str:
        """Системный контекст для Ollama; пустая строка, если урока нет."""
        if self.lesson is None:
            return ""
        return build_teacher_context(self.lesson)

    def _on_chat_completed(self, text: str) -> None:
        if self._pending_message is None:
            return
        answer = Message("assistant", text, self._request_model)
        self.history.extend([self._pending_message, answer])
        self._pending_message = None
        self.generating_reply = False
        self.status = "Готов"
        self.message_added.emit(answer)
        self.state_changed.emit()

    def _on_chat_failed(self, error: str) -> None:
        self._pending_message = None
        self.generating_reply = False
        self.status = error + " Сообщение не добавлено в контекст; отправь его повторно."
        self.state_changed.emit()
