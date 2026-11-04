import json

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from .config import AppConfig


class OllamaClient(QObject):
    models_loaded = Signal(list)
    models_failed = Signal(str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.network = QNetworkAccessManager(self)
        self._models_reply: QNetworkReply | None = None

    def load_models(self) -> None:
        if self._models_reply is not None:
            return
        request = QNetworkRequest(
            QUrl(f"{self.config.ollama_base_url.rstrip('/')}/api/tags")
        )
        request.setTransferTimeout(self.config.request_timeout_ms)
        self._models_reply = self.network.get(request)
        self._models_reply.finished.connect(self._on_models_loaded)

    @staticmethod
    def parse_models(payload: bytes) -> list[str]:
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
        models = data.get("models")
        if not isinstance(models, list):
            raise ValueError("отсутствует список models")
        names = []
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("неверный формат модели")
            name = model.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("неверное имя модели")
            names.append(name)
        return sorted(set(names), key=str.casefold)

    def _on_models_loaded(self) -> None:
        reply = self._models_reply
        if reply is None:
            return
        names = None
        error = None
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                error = f"Не удалось получить модели: {reply.errorString()}"
            else:
                names = self.parse_models(bytes(reply.readAll()))
        except (ValueError, UnicodeDecodeError) as exc:
            error = f"Некорректный ответ Ollama: {exc}"
        finally:
            self._models_reply = None
            reply.deleteLater()
        if error is not None:
            self.models_failed.emit(error)
        else:
            self.models_loaded.emit(names)
