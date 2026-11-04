from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from .ollama_client import OllamaClient


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class SessionController(QObject):
    state_changed = Signal()
    message_added = Signal(str)

    def __init__(self, client: OllamaClient, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.models: list[str] = []
        self.selected_model = ""
        self.history: list[Message] = []
        self.loading = False
        self.status = "Готов"
        client.models_loaded.connect(self._on_models_loaded)
        client.models_failed.connect(self._on_models_failed)

    @property
    def can_send(self) -> bool:
        return bool(self.selected_model) and not self.loading

    def load_models(self) -> None:
        if self.loading:
            return
        self.loading = True
        self.status = "Загружаю список моделей…"
        self.state_changed.emit()
        self.client.load_models()

    def select_model(self, model: str) -> None:
        if not self.loading and model in self.models and model != self.selected_model:
            self.selected_model = model
            self.state_changed.emit()

    def _on_models_loaded(self, models: list[str]) -> None:
        self.models = list(models)
        if self.selected_model not in models:
            self.selected_model = models[0] if models else ""
        self.loading = False
        self.status = (
            f"Готов. Доступно моделей: {len(models)}"
            if models else "В Ollama нет установленных моделей"
        )
        self.state_changed.emit()

    def _on_models_failed(self, error: str) -> None:
        self.loading = False
        self.status = error
        self.state_changed.emit()

    def send_message(self, text: str) -> bool:
        message = text.strip()
        if not message or not self.can_send:
            return False
        self.history.append(Message("user", message))
        self.status = (
            f"Выбрана модель: {self.selected_model}. "
            "Отправка в Ollama ещё не подключена."
        )
        self.message_added.emit(message)
        self.state_changed.emit()
        return True
