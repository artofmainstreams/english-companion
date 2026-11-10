import json

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from .config import AppConfig
from .message import Message


class OllamaClient(QObject):
    chat_completed = Signal(str)
    chat_failed = Signal(str)
    models_loaded = Signal(list)
    models_failed = Signal(str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.network = QNetworkAccessManager(self)
        self._models_reply: QNetworkReply | None = None
        self._chat_reply: QNetworkReply | None = None

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

    def send_chat(self, model: str, messages: list[Message], system: str = "") -> None:
        """Отправить чат-запрос.

        ``system`` — системный контекст (например, из урока): уходит в
        поле ``system`` запроса и НЕ входит в историю сообщений.
        """
        if self._chat_reply is not None:
            raise RuntimeError("Запрос уже выполняется")
        request = QNetworkRequest(
            QUrl(f"{self.config.ollama_base_url.rstrip('/')}/api/chat")
        )
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )
        request.setTransferTimeout(self.config.chat_timeout_ms)
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
        }
        if system:
            payload["system"] = system
        self._chat_reply = self.network.post(
            request, json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        self._chat_reply.finished.connect(self._on_chat_completed)

    @staticmethod
    def parse_chat(payload: bytes) -> str:
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
        if data.get("error"):
            raise ValueError(str(data["error"]))
        message = data.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("отсутствует ответ assistant")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("пустой ответ модели")
        if data.get("done") is not True:
            raise ValueError("ответ модели не завершён")
        return content

    def _on_chat_completed(self) -> None:
        reply = self._chat_reply
        if reply is None:
            return
        content = ""
        error = None
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                error = f"Не удалось получить ответ: {reply.errorString()}"
            else:
                content = self.parse_chat(bytes(reply.readAll()))
        except (ValueError, UnicodeDecodeError) as exc:
            error = f"Некорректный ответ Ollama: {exc}"
        finally:
            self._chat_reply = None
            reply.deleteLater()
        if error is not None:
            self.chat_failed.emit(error)
        else:
            self.chat_completed.emit(content)
