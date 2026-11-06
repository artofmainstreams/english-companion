"""Озвучивание ответов преподавателя: голосовой синтез macOS ``say``.

``TtsSynth`` — небольшой отдельный компонент синтеза. Он НЕ проигрывает
звук (это ``AudioPlayer``) и НЕ общается с Ollama (это ``SessionController``):
он превращает текст ответа в готовый AIFF-файл.

Механика:
* голос — английский, из конфига (``AppConfig.tts_voice``; проверить
  наличие: ``say -v '?'``); если голоса на системе нет, macOS ``say``
  молча возьмёт голос по умолчанию — это поведение системы;
* процесс запускается асинхронно через ``QProcess`` (цикл событий Qt,
  без фоновых потоков); текст передаётся как данные — через stdin
  (``-f -``), а не аргументом командной строки;
* на выход создаётся временный AIFF-файл (расширение ``.aiff`` —
  ``say`` выбирает формат по нему); этот формат проверен на целевой
  сборке Qt: QMediaPlayer декодирует и проигрывает его целиком;
* компонент — ЯВНЫЙ ВЛАДЕЛЕЦ своих временных файлов: создаёт каталог
  при первом синтезе и удаляет его в ``shutdown()``;
* новый ответ СМЕНЯЕТ незавершённый синтез: предыдущий процесс
  останавливается, частичный файл не публикуется;
* «готовым» ответом считается только файл успешного синтеза; повтор
  (``last_reply_path``) — тот же готовый файл, без нового синтеза.

Публичные сигналы: ``state_changed``, ``reply_ready(path)`` (файл
готов и существует), ``failed(reason)`` (состояние возвращается IDLE).
"""

import logging
import os
import shutil
import tempfile

from PySide6.QtCore import QProcess, QObject, Signal

logger = logging.getLogger(__name__)

# Перечисления связываем ВРЕМЯ ИМПОРТА (имена на уровне модуля): так код
# не зависит от подмены класса QProcess в тестах (у подмены нет вложенных
# перечислений, а имена ExitStatus/ProcessError/ProcessState не трогаются).
ExitStatus = QProcess.ExitStatus
ProcessError = QProcess.ProcessError
ProcessState = QProcess.ProcessState

# Английский голос macOS по умолчанию (есть в свежих системах;
# полный список: ``say -v '?'``).
DEFAULT_VOICE = "Samantha"


class TtsSynth(QObject):
    """Синтезирует текст ответа в временный AIFF без блокировки интерфейса."""

    state_changed = Signal()
    # Готовый голосовой ответ: путь к непустому AIFF-файлу.
    reply_ready = Signal(str)
    # Ошибка синтеза; причина — человекочитаемая строка.
    failed = Signal(str)

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"

    def __init__(self, voice: str = DEFAULT_VOICE, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.voice = voice
        self.state = self.IDLE
        self.status = "Озвучивание: готов"

        self._temp_dir: str | None = None
        # Каждый синтез получает новый файл (как в рекордере): плеер
        # кэширует источник по URL, запись поверх пути дала бы старое аудио.
        self._synth_seq = 0
        # Файл текущего (ещё незавершённого) синтеза.
        self._current_path: str | None = None
        # Последний УСПЕШНЫЙ синтез — то, что можно повторять.
        self._ready_path: str | None = None
        self._process: QProcess | None = None
        # Причина ошибки, полученная до finished (FailedToStart/Crashed).
        self._error_detail: str | None = None
        self._shutdown = False

    # --- Свойства для интерфейса ------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Идёт ли синтез прямо сейчас."""
        return self.state == self.SYNTHESIZING

    @property
    def has_reply(self) -> bool:
        """Есть ли готовый голосовой ответ (файл существует и непуст)."""
        return self._valid_file(self._ready_path)

    @property
    def last_reply_path(self) -> str | None:
        """Путь к последнему готовому ответу, либо ``None``."""
        if self.has_reply and self._ready_path is not None:
            return self._ready_path
        return None

    @property
    def is_shutdown(self) -> bool:
        """``True`` после ``shutdown()``: компонент больше не работает."""
        return self._shutdown

    # --- Публичные действия ------------------------------------------------

    def synthesize(self, text: str) -> bool:
        """Синтезирует текст в AIFF-файл (асинхронно, ``say``).

        Возвращает ``True``, если процесс запущен (готовность придёт
        сигналом ``reply_ready``/``failed``). ``False`` — синтез отклонён
        (остановка, пустой текст, не удалось запустить ``say``); причина
        в ``status``. Новый вызов во время синтеза СМЕНЯЕТ предыдущий:
        предыдущий процесс останавливается.
        """
        if self._shutdown:
            self.status = "Озвучивание недоступно: компонент остановлен."
            self.state_changed.emit()
            return False
        text = (text or "").strip()
        if not text:
            self.status = "Озвучивать нечего: пустой текст."
            self.state_changed.emit()
            return False
        if self.state == self.SYNTHESIZING:
            # Новый ответ сменяет незавершённый синтез.
            self._cancel_process()
        # Причина ошибки нового синтеза — только от нового процесса.
        self._error_detail = None

        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="english-companion-tts-")
        self._synth_seq += 1
        out_path = os.path.join(self._temp_dir, f"reply_{self._synth_seq}.aiff")
        self._current_path = out_path

        self.state = self.SYNTHESIZING
        self.status = "Озвучиваю ответ…"
        self.state_changed.emit()
        logger.info(
            "tts: start voice=%r seq=%d out=%s text_len=%d",
            self.voice, self._synth_seq, out_path, len(text),
        )

        process = QProcess(self)
        self._process = process
        process.finished.connect(self._on_process_finished)
        process.errorOccurred.connect(self._on_process_error)
        # Текст — данные через stdin; файл выхода задаём явно.
        process.start("say", ["-v", self.voice, "-f", "-", "-o", out_path])
        if not process.waitForStarted(5_000):
            self._fail("не удалось запустить say (команда недоступна?)")
            return False
        if not process.write(text.encode("utf-8")):
            self._fail("не удалось передать текст для синтеза")
            return False
        process.closeWriteChannel()
        process.waitForBytesWritten(5_000)
        return True

    def cancel(self) -> None:
        """Прерывает текущий (незавершённый) синтез; последний готовый
        ответ (повтор) сохраняется. Отдельно от ``shutdown()``: компонент
        остаётся живым. Частичный файл не публикуется. Идемпотентна.
        """
        if self._shutdown or self.state != self.SYNTHESIZING:
            return
        self._cancel_process()
        self._drop_file(self._current_path)
        self._current_path = None
        self.state = self.IDLE
        self.status = "Озвучивание: остановлено"
        self.state_changed.emit()

    def shutdown(self) -> None:
        """Останавливает синтез и удаляет СВОИ временные файлы. Идемпотентна.

        Порядок важен: сначала процесс ``say`` убивается (он может писать
        файл), потом удаляется временный каталог. После вызова компонент
        не работает: ``synthesize()`` отклоняется, поздние сигналы игнорируются.
        """
        if self._shutdown:
            return
        self._shutdown = True

        self._cancel_process()
        self._drop_file(self._current_path)
        self._current_path = None
        self._cleanup_temp_dir()
        self._ready_path = None

        self.state = self.IDLE
        self.status = "Озвучивание: остановлено"
        self.state_changed.emit()

    # --- Обработчики сигналов ----------------------------------------------

    def _on_process_error(self, error: ProcessError) -> None:
        # Запоминаем причину; решение (fail/ready) принимает _on_process_finished.
        if self._shutdown:
            return
        if error == ProcessError.FailedToStart:
            self._error_detail = "не удалось запустить say (команда недоступна?)"
        elif error == ProcessError.Crashed:
            self._error_detail = "say аварийно завершился"

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        if self._shutdown:
            return
        process = self._process
        self._process = None
        path = self._current_path
        self._current_path = None

        ok = (
            exit_code == 0
            and exit_status == ExitStatus.NormalExit
            and self._valid_file(path)
        )
        if ok:
            # Готовый ответ публикуем как «повторяемый».
            self._ready_path = path
            self.state = self.IDLE
            self.status = "Ответ озвучен."
            logger.info("tts: ready seq=%d out=%s", self._synth_seq, path)
            self.state_changed.emit()
            self.reply_ready.emit(path)
            return

        # Ошибка: частичный файл не «готовый ответ» — удаляем.
        self._drop_file(path)
        reason = self._error_detail
        self._error_detail = None
        if reason is None and process is not None:
            stderr = process.readAllStandardError().decode("utf-8", "replace").strip()
            if stderr:
                reason = stderr.splitlines()[-1]
        if reason is None:
            reason = f"say завершился с кодом {exit_code}"
        self.state = self.IDLE
        self.status = f"Не удалось озвучить ответ: {reason}"
        logger.warning("tts: failed seq=%d out=%s reason=%r", self._synth_seq, path, reason)
        self.state_changed.emit()
        self.failed.emit(reason)

    # --- Внутренние ---------------------------------------------------------

    def _fail(self, reason: str) -> None:
        """Сбой на этапе запуска: сбрасываем состояние и публикуем ошибку."""
        self._cancel_process()
        self._drop_file(self._current_path)
        self._current_path = None
        self.state = self.IDLE
        self.status = f"Не удалось озвучить ответ: {reason}"
        self.state_changed.emit()
        self.failed.emit(reason)

    def _cancel_process(self) -> None:
        """Останавливает процесс синтеза и разрывает связи с ним."""
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.finished.disconnect(self._on_process_finished)
            process.errorOccurred.disconnect(self._on_process_error)
        except (TypeError, RuntimeError):
            pass  # сигнал уже снят (повторный разбор) — нормально
        if process.state() != ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(2_000)
        process.deleteLater()

    def _drop_file(self, path: str | None) -> None:
        """Удаляет частичный/устаревший файл, если он существует."""
        if path is None:
            return
        try:
            if os.path.lexists(path):
                os.unlink(path)
        except OSError as exc:
            logger.warning("tts: не удалось удалить %s: %s", path, exc)

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is None:
            return
        # Компонент — единственный владелец каталога: внутри только
        # временные файлы синтеза (готовые ответы тоже временные).
        try:
            shutil.rmtree(self._temp_dir)
        except OSError as exc:
            logger.warning(
                "tts: не удалось удалить временную папку %s: %s",
                self._temp_dir, exc,
            )
        finally:
            self._temp_dir = None

    @staticmethod
    def _valid_file(path: str | None) -> bool:
        """Файл существует и непустой — пригоден как «готовый ответ»."""
        return (
            path is not None
            and os.path.exists(path)
            and os.path.getsize(path) > 0
        )
