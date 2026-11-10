from PySide6.QtCore import QObject, Signal

from .lesson import Lesson
from .message import Message
from .ollama_client import OllamaClient
from .teacher_context import build_teacher_context


class SessionController(QObject):
    state_changed = Signal()
    message_added = Signal(object)
    # История изменилась структурно (перезапись реплики / отмена): окно
    # перерисовывает чат целиком из истории + реплики в работе.
    history_changed = Signal()

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
        # Реплика, сохранённая для повтора запроса после сбоя: повтор
        # БЕЗ дублирования сообщения в чате.
        self._retry_message: Message | None = None
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

    @property
    def pending_message(self) -> Message | None:
        """Реплика пользователя, которая ждёт ответ (запрос в работе)."""
        return self._pending_message

    @property
    def retry_message(self) -> Message | None:
        """Реплика, готовая к повтору запроса после сбоя (без дубля в чате)."""
        return self._retry_message

    def set_lesson(self, lesson: Lesson) -> None:
        """Выбранный урок — источник системного контекста преподавателя."""
        self.lesson = lesson
        self.state_changed.emit()

    def reset_history(self) -> None:
        """Новое занятие: история чата очищается (урок остаётся выбран)."""
        self.history = []
        self.status = "Готов"
        self.state_changed.emit()

    # --- Перезапись последней реплики пользователя -------------------------

    def last_user_message(self) -> Message | None:
        """Последняя реплика пользователя — цель действия «Перезаписать».

        Пока после неё нет следующей реплики пользователя, она
        перезаписывается: ответ преподавателя после неё не препятствует
        (при замене он убирается вместе с ней).
        """
        if self._pending_message is not None:
            return self._pending_message
        for message in reversed(self.history):
            if message.role == "user":
                return message
        return None

    def discard_pending(self) -> bool:
        """Отменяет текущий запрос к Ollama: сеть, реплика, busy-состояние.

        Запоздалый результат отменённого запроса отбрасывается тихо (клиент
        отключает его ``finished``). Ответ НЕ перезапускается автоматически.
        Возвращает, был ли запрос в работе.
        """
        abort = getattr(self.client, "abort_chat", None)
        if callable(abort):
            abort()
        had = self._pending_message is not None or self.generating_reply
        self._pending_message = None
        self._retry_message = None
        self.generating_reply = False
        if had:
            self.status = "Готов"
            self.state_changed.emit()
        return had

    def replace_last_user_message(
        self, new_text: str, target: Message | None = None
    ) -> bool:
        """Перезапись: заменяет последнюю реплику пользователя (и связанный
        с ней ответ преподавателя) новой и запрашивает ответ заново.

        До вызова прежний обмен НЕ трогается. Запрос в работе отменяется
        (запоздалый результат отбрасывается). Окно перерисовывает чат по
        ``history_changed``; новый ответ придёт ``message_added``. Одна
        перезапись — максимум одна замена и один новый запрос.

        ``target`` — реплика, сохраняемая окном с момента клика «Перезаписать»
        (pending мог быть уже отменён — «остановить текущий ответ»). Без
        аргумента цель берётся из текущего состояния.

        Возвращает ``False``: нет цели, текст пуст или запрос нельзя
        отправить (модель/урок не выбраны).
        """
        new_text = (new_text or "").strip()
        if target is None:
            target = self.last_user_message()
        if not new_text or target is None:
            return False
        if not self.selected_model or self.lesson is None:
            return False
        index = None
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i] is target:
                index = i
                break
        if index is not None:
            # Убираем реплику и связанный с ней ответ преподавателя.
            base = self.history[:index]
        else:
            # Реплика ещё не в истории (ответ не пришёл) — история не чинится.
            base = list(self.history)
        self.discard_pending()
        replaced = Message("user", new_text)
        self.history = base
        self._pending_message = replaced
        self._retry_message = replaced
        self.generating_reply = True
        self.status = f"Думаю · {self.selected_model}…"
        self.history_changed.emit()
        self.state_changed.emit()
        self.client.send_chat(
            self.selected_model,
            [*base, replaced],
            system=self.system_context(),
        )
        return True

    def restore_pending(self, message: Message) -> None:
        """Отмена перезаписи: возвращает прежнюю реплику в «ждёт ответ»
        БЕЗ нового запроса (незавершённый ответ не перезапускается).

        Окно перерисовывает чат по ``history_changed``.
        """
        self._pending_message = message
        self._retry_message = message
        self.generating_reply = False
        self.status = "Перезапись отменена; можно повторить запрос."
        self.history_changed.emit()
        self.state_changed.emit()

    def retry_last_user_message(self) -> bool:
        """Повторяет запрос для сохранённой реплики БЕЗ её дублирования:
        реплика уже в чате — уходит только новый запрос к Ollama."""
        if self._retry_message is None or self.busy:
            return False
        if not self.selected_model or self.lesson is None:
            return False
        self._pending_message = self._retry_message
        self.generating_reply = True
        self.status = f"Думаю · {self.selected_model}…"
        self.state_changed.emit()
        self.client.send_chat(
            self.selected_model,
            [*self.history, self._retry_message],
            system=self.system_context(),
        )
        return True

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
        self._retry_message = None
        self.generating_reply = False
        self.status = "Готов"
        self.message_added.emit(answer)
        self.state_changed.emit()

    def _on_chat_failed(self, error: str) -> None:
        if self._pending_message is None:
            return  # поздний сигнал отменённого запроса — отбрасываем
        # Исправленная/отправленная реплика сохраняется для повтора запроса
        # БЕЗ её дублирования в чате (в историю она не попадала).
        self._retry_message = self._pending_message
        self._pending_message = None
        self.generating_reply = False
        self.status = error + " Реплика сохранена; повторите запрос."
        self.state_changed.emit()
