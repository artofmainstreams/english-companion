"""Фасад аудиослоя: запись, воспроизведение и озвучивание ответов.

Аудиослой разбит на независимые компоненты:

  * :class:`companion.audio_recorder.AudioRecorder` — устройства ввода,
    запись, результат успешно завершённой записи; явный владелец своих
    временных файлов (создаёт и удаляет их);
  * :class:`companion.audio_player.AudioPlayer` — воспроизведение любого
    готового аудиофайла; файл ему НЕ принадлежит, и плеер его никогда
    не удаляет (это же позволяет проигрывать TTS-файлы ответов);
  * синтезатор ответов — компонент с единым публичным API
    (``synthesize``/``shutdown``, сигналы ``state_changed``/
    ``reply_ready``/``failed``, владелец своих временных файлов):
    :class:`companion.tts_synth.TtsSynth` (macOS ``say`` → AIFF) или
    :class:`companion.tts_qwen3.Qwen3TtsSynth` (Qwen3-TTS, MLX → WAV);
    какой — решает ``build_tts(config)`` по ``AppConfig.tts_engine``.

``AudioController`` — фасад с прежним публичным API плюс действия
озвучивания (``speak_reply``/``replay_reply``):

  * единое производное состояние ``IDLE/RECORDING/PLAYING``;
  * ``state``/``status`` синхронизируются из компонента, который
    последний раз подал ``state_changed``; ``devices_changed`` уходит
    с рекордера;
  * взаимное исключение записи и воспроизведения — молчаливый отказ,
    как раньше;
  * автозапуск готового ответа согласован с записью и воспроизведением:
    если аудио занято, ответ сохраняется для ручного «Повторить ответ»;
  * порядок ``shutdown()``: сначала плеер останавливается и отпускает
    файл и выход (файл при этом НЕ удаляется), потом синтез
    останавливается и удаляет СВОИ временные файлы, потом рекордер
    разбирает сессию; каждый компонент удаляет СВОИ временные файлы.

Фасад НЕ знает механизм синтеза (процесс ``say`` или фоновый поток
модели) — только публичный API синтезатора, поэтому смена движка
(``tts_engine``) не затрагивает остальной код.
"""

import os

from PySide6.QtCore import QObject, Signal

from .audio_player import AudioPlayer
from .audio_recorder import AudioRecorder
from .config import AppConfig
from .tts_qwen3 import Qwen3TtsSynth
from .tts_synth import DEFAULT_VOICE, TtsSynth


def build_tts(config: AppConfig, parent: QObject | None = None) -> QObject:
    """Выбирает движок озвучивания по конфигу (``tts_engine``).

    ``"qwen3"`` (по умолчанию) — :class:`companion.tts_qwen3.Qwen3TtsSynth`
    (mlx_audio импортируется лениво, в рабочем потоке первого синтеза);
    ``"say"`` — :class:`companion.tts_synth.TtsSynth` (резервный движок).
    Оба — с одинаковым публичным API: состояния, ``synthesize``/``shutdown``,
    сигналы ``state_changed``/``reply_ready``/``failed``. Неизвестный
    движок — явная ошибка (без тихой замены), как с STT-бэкендами;
    ``reference``-режим без пары «файл + текст» — тоже явная ошибка.
    """
    engine = (config.tts_engine or "say").strip().lower()
    if engine == "say":
        return TtsSynth(config.tts_voice, parent)
    if engine == "qwen3":
        method = (config.tts_qwen3_method or "voice_design").strip().lower()
        if method == "reference":
            if not (config.tts_qwen3_ref_audio and config.tts_qwen3_ref_text):
                # reference-режим без пары (файл + точный текст) не может
                # работать — явная ошибка, а не тихая смена на voice_design.
                raise ValueError(
                    "tts_qwen3_method='reference' требует непустых "
                    "tts_qwen3_ref_audio И tts_qwen3_ref_text"
                )
            model_id = config.tts_qwen3_model_reference
            ref_audio = config.tts_qwen3_ref_audio
            ref_text = config.tts_qwen3_ref_text
        else:
            model_id = config.tts_qwen3_model_voice_design
            # Пару референса НЕ передаём: режим синтезатора определяется
            # непустой парой — в voice_design она должна быть пуста.
            ref_audio = ""
            ref_text = ""
        return Qwen3TtsSynth(
            model_id=model_id,
            instruct=config.tts_qwen3_instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            streaming=config.tts_qwen3_streaming,
            streaming_interval=config.tts_qwen3_streaming_interval,
            parent=parent,
        )
    raise ValueError(
        f"неизвестный tts_engine: {config.tts_engine!r} "
        "(допустимо: 'say' или 'qwen3')"
    )


class AudioController(QObject):
    """Управляет записью, воспроизведением и озвучиванием ответов.

    Работает как отдельный слой (как SessionController/OllamaClient):
    MainWindow только показывает состояние и вызывает методы, сам Qt
    Multimedia живёт в компонентах и не мешает текстовому чату.

    Озвучивание ответов:
    * ``speak_reply(text)`` запускает синтез (``TtsSynth``); готовый файл
      автозапускается, только если аудио свободно (IDLE);
    * если аудио занято (запись/воспроизведение), готовый ответ сохраняется
      и ждёт ручного ``replay_reply()``;
    * ``replay_reply()`` проигрывает последний готовый ответ БЕЗ нового
      запроса к Ollama и без нового синтеза.

    Движок синтеза: готовый компонент с публичным API ``TtsSynth``
    (``tts=...``, например из ``build_tts(config)``) или по умолчанию
    ``TtsSynth(tts_voice)``; фасад не зависит от механизма синтеза.

    Жизненный цикл:
    * ``shutdown()`` идемпотентна — повторные вызовы безопасны;
    * после ``shutdown()`` ``record()``/``play()``/``speak_reply()``/
      ``replay_reply()`` отклоняются, а поздние сигналы Qt игнорируются
      (состояние больше не «оживает»);
    * разбор по порядку: остановка плеера → выход → остановка синтеза
      (процесс ``say`` / фоновый поток модели) → разбор сессии записи;
      каждый компонент удаляет СВОИ временные файлы.
    """

    state_changed = Signal()
    # Список устройств ввода изменился (подключение/отключение микрофона):
    # уходит с рекордера, интерфейс перечитывает список.
    devices_changed = Signal()
    # Успешно завершённая запись (уходит с рекордера): интерфейс по этому
    # сигналу автоматически запускает распознавание по готовому файлу.
    recording_completed = Signal(str)

    IDLE = "idle"
    RECORDING = "recording"
    PLAYING = "playing"

    def __init__(
        self,
        tts_voice: str = DEFAULT_VOICE,
        parent: QObject | None = None,
        tts: QObject | None = None,
        stream_start_buffer_ms: int = 500,
    ) -> None:
        super().__init__(parent)
        self.status = "Голос: готов"

        self.player = AudioPlayer(self)
        self.recorder = AudioRecorder(self)
        # Синтезатор: переданный компонент (build_tts/config.tts_engine)
        # или TtsSynth по умолчанию; родителя задаёт фасад.
        if tts is None:
            tts = TtsSynth(tts_voice, self)
        else:
            tts.setParent(self)
        self.tts = tts

        # Стартовый буфер потокового вывода (мс): вывод стартует, когда
        # накоплено не меньше этого аудио (защита от недочётов на старте).
        self.stream_start_buffer_ms = int(stream_start_buffer_ms)
        # Состояние потокового вывода: none → pending → active; off — sticky.
        self._stream = "none"
        self._stream_dropped = False
        self._stream_job = 0
        self._pending = []
        self._pending_samples = 0
        self._pending_sr = 0

        # Статус и состояние синхронизируются из компонента, который
        # последний раз подал state_changed.
        self.player.state_changed.connect(self._sync_from_player)
        self.recorder.state_changed.connect(self._sync_from_recorder)
        self.tts.state_changed.connect(self._sync_from_tts)
        # Готовый/сбойный синтез — отдельные сигналы: автозапуск и статус.
        self.tts.reply_ready.connect(self._on_reply_ready)
        self.tts.failed.connect(self._on_reply_failed)
        # Потоковый вывод (есть только у Qwen3TtsSynth): куски речи.
        chunk_sig = getattr(self.tts, "chunk_ready", None)
        if chunk_sig is not None:
            chunk_sig.connect(self._on_chunk)
        self.recorder.devices_changed.connect(self.devices_changed.emit)
        self.recorder.recording_completed.connect(
            self.recording_completed.emit
        )

        # Флаг shutdown фасада: после него он не реагирует на сигналы
        # компонентов и не пускает новые операции.
        self._shutdown = False
        # Готовый ответ временно недоступен для «Повторить ответ» (во время
        # перезаписи реплики — старый ответ не повторять). Снимается
        # ``unsuppress_reply()`` (отмена) или новым готовым ответом.
        self._reply_suppressed = False

    # --- Свойства для интерфейса ------------------------------------------

    @property
    def state(self) -> str:
        """Производное состояние: запись → воспроизведение → IDLE."""
        if self.recorder.state == AudioRecorder.RECORDING:
            return self.RECORDING
        if self.player.state == AudioPlayer.PLAYING:
            return self.PLAYING
        return self.IDLE

    @property
    def has_recording(self) -> bool:
        """Есть ли успешно завершённая запись (всегда у рекордера)."""
        return self.recorder.has_recording

    @property
    def can_record(self) -> bool:
        """Можно начать запись: сейчас не идёт ни запись, ни воспроизведение."""
        return self.state == self.IDLE and self.recorder.can_record

    @property
    def can_play(self) -> bool:
        """Можно прослушать готовую запись: сейчас ничего не идёт."""
        return (
            self.state == self.IDLE
            and self.recorder.has_recording
            and self.player.can_play
        )

    @property
    def is_shutdown(self) -> bool:
        """``True`` после ``shutdown()``: слой больше не работает."""
        return self._shutdown

    @property
    def last_recording_path(self) -> str | None:
        """Путь последней успешно завершённой записи (у рекордера)."""
        return self.recorder.last_recording_path

    @property
    def has_reply(self) -> bool:
        """Есть ли готовый голосовой ответ (у синтезатора)."""
        return self.tts.has_reply

    @property
    def can_replay(self) -> bool:
        """Можно «Повторить ответ»: аудио свободно, ответ готов и не отменён."""
        return (
            not self._shutdown
            and not self._reply_suppressed
            and self.state == self.IDLE
            and self.tts.has_reply
        )

    # --- Устройства ввода (делегирование рекордеру) ------------------------

    def input_devices(self) -> list[tuple[str, str]]:
        return self.recorder.input_devices()

    def input_device_names(self) -> list[str]:
        return self.recorder.input_device_names()

    def selected_input_id(self) -> str | None:
        return self.recorder.selected_input_id()

    def selected_input_name(self) -> str | None:
        return self.recorder.selected_input_name()

    def set_input_device(self, name_or_id: str | None) -> bool:
        return self.recorder.set_input_device(name_or_id)

    # --- Публичные действия ------------------------------------------------

    def record(self) -> None:
        """Начинает запись; запрещено, пока идёт воспроизведение."""
        if self._shutdown or self.state != self.IDLE:
            return
        self.recorder.record()

    def stop_recording(self) -> None:
        if self._shutdown:
            return
        self.recorder.stop_recording()

    def cancel_recording(self) -> None:
        """Отменяет текущую запись: файл отбрасывается,
        ``recording_completed`` не поднимается (STT не запускается)."""
        if self._shutdown:
            return
        self.recorder.cancel_recording()

    def play(self) -> None:
        """Проигрывает последнюю готовую запись; запрещено во время записи."""
        if self._shutdown or self.state != self.IDLE:
            return
        if not self.recorder.has_recording:
            self.status = "Сначала запиши речь."
            self.state_changed.emit()
            return
        path = self.recorder.last_recording_path
        if path is None:
            # has_recording == True, но файл исчез — публикуем отказ.
            self.status = "Нет аудиофайла для воспроизведения."
            self.state_changed.emit()
            return
        # Сам плеер дополнительно проверяет файл, выход и своё состояние.
        self.player.play(path)

    def stop_playback(self) -> None:
        """Останавливает текущий вывод и прерывает незавершённый синтез.

        Поток (речь частями): вывод останавливается, незавершённый синтез
        отменяется (куски отбрасываются, WAV не пишется). Полный WAV
        (QMediaPlayer): как раньше (``player.stop``). Последний готовый
        ответ сохраняется для «Повторить ответ» в обоих случаях.
        """
        if self._shutdown:
            return
        if self.player.stream_active:
            self.player.stop_stream()
        else:
            self.player.stop()
        # Незавершённый синтез: прерываем (для say — no-op, когда он уже
        # завершён и state == IDLE; для стриминга — отбросить куски).
        self._cancel_tts()
        self._reset_stream_state()

    def speak_reply(self, text: str) -> None:
        """Озвучивает ответ преподавателя (синтез → WAV/поток).

        Синтез не занимает микрофон и выход, поэтому запускается независимо
        от записи. Согласование — на шаге ВОСПРОИЗВЕДЕНИЯ: стриминг стартует
        только если аудио свободно (иначе куски отбрасываются, а ответ остаётся
        готовым для ручного ``replay_reply()``). Новый ответ сменяет текущий:
        активный поток останавливается, незавершённый синтез отменяется.
        После ``shutdown()`` отклоняется.
        """
        if self._shutdown:
            return
        # Новый ответ сменяет текущий: остановить вывод, сбросить состояние.
        if self.player.stream_active:
            self.player.stop_stream()
        self._reset_stream_state(drop=True)
        accepted = self.tts.synthesize(text)
        if accepted:
            # Номер задания нужен только для потокового вывода (Qwen3TtsSynth).
            job_id = getattr(self.tts, "current_job_id", None)
            if job_id is not None:
                self._stream_job = job_id
            self._stream_dropped = False

    def replay_reply(self) -> None:
        """Повторяет последний готовый голосовой ответ.

        Новый запрос к Ollama и новый синтез НЕ выполняются — проигрывается
        тот же готовый файл (``TtsSynth.last_reply_path``). Запрещено,
        пока идёт запись/воспроизведение, если готового ответа нет
        (причина публикуется в ``status``) или ответ отменён
        (``suppress_reply`` — во время перезаписи).
        """
        if self._shutdown or self.state != self.IDLE:
            return
        if self._reply_suppressed:
            self.status = "Повторять нечего: предыдущий ответ отменён."
            self.state_changed.emit()
            return
        path = self.tts.last_reply_path
        if path is None:
            self.status = "Повторять нечего: нет готового ответа."
            self.state_changed.emit()
            return
        if not os.path.exists(path):
            self.status = "Нет аудиофайла для воспроизведения."
            self.state_changed.emit()
            return
        # Файл принадлежит синтезатору; плеер его не удаляет.
        self.player.play(path)

    def suppress_reply(self) -> None:
        """Делает готовый ответ недоступным для «Повторить ответ».

        Применяется на время перезаписи реплики: старый голосовой ответ
        не повторяется ни во время перезаписи, ни после подтверждённой
        замены. Возвращает ``unsuppress_reply()``; снимается автоматически
        новым готовым ответом.
        """
        self._reply_suppressed = True
        self.state_changed.emit()

    def unsuppress_reply(self) -> None:
        """Возвращает «Повторить ответ», если готовый файл сохранился."""
        self._reply_suppressed = False
        self.state_changed.emit()

    def shutdown(self) -> None:
        """Останавливает запись, воспроизведение и синтез; разбирает слой.

        Идемпотентна: повторные вызовы (например, ``aboutToQuit`` и явный
        вызов из кода) безопасны. После вызова слой не работает:
        ``record()``/``play()``/``speak_reply()``/``replay_reply()``
        отклоняются, поздние сигналы Qt игнорируются.

        Порядок важен: сначала плеер останавливается и отпускает файл/выход
        (файл он НЕ удаляет — владелец файла это делает), потом синтез
        останавливается (процесс ``say`` или фоновый поток модели — он мог
        писать файл) и удаляет СВОИ временные файлы, потом рекордер
        разбирает сессию и удаляет СВОИ временные файлы.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # 1) Плеер: остановка, отсоединение выхода, отпускание источника.
        #    Переданный ранее файл при этом НЕ удаляется.
        self.player.shutdown()

        # 2) Синтез: остановка (процесс say / фоновый поток модели —
        #    до удаления временных файлов!), затем удаление его СВОИХ
        #    временных файлов.
        self.tts.shutdown()

        # 3) Рекордер: отключение hotplug, разбор графа, временные файлы.
        self.recorder.shutdown()

        self.status = "Голос: остановлен"
        self.state_changed.emit()

    # --- Внутренние ---------------------------------------------------------

    def _sync_from_player(self) -> None:
        """Плеер подал state_changed: обновляем фасадный статус и сигнал."""
        if self._shutdown:
            return  # фасад уже закрыт — поздние сигналы не «оживляют»
        self.status = self.player.status
        self.state_changed.emit()

    def _sync_from_recorder(self) -> None:
        """Рекордер подал state_changed: обновляем фасадный статус и сигнал."""
        if self._shutdown:
            return
        self.status = self.recorder.status
        self.state_changed.emit()

    def _sync_from_tts(self) -> None:
        """Синтезатор подал state_changed: обновляем фасадный статус и сигнал."""
        if self._shutdown:
            return
        self.status = self.tts.status
        self.state_changed.emit()

    def _on_chunk(self, chunk, sample_rate: int, job_id: int) -> None:
        """Очередной кусок речи (главный поток).

        Гейты: не наше задание / отброшено → молча отбрасываем.
        Поток уже активен — это НАШ собственный вывод, и state здесь
        ОЖИДАЕМ в PLAYING; куски пишем напрямую (не «занято»).
        Поток не активен (none/pending) — вывод стартует только если аудио
        свободно (не запись/полный WAV); иначе накопленное не пускаем —
        ответ останется готовым для ручного «Повторить ответ».
        """
        if self._shutdown or self._stream_dropped:
            return
        if job_id != self._stream_job:
            return  # кусок старого/чужого задания
        if self._stream == "active":
            # Активный поток — собственный вывод: state == PLAYING здесь норма,
            # поэтому куски пишем, а не отбрасываем (write_chunk безопасен:
            # если поток уже остановили, он становится no-op).
            self.player.write_chunk(chunk)
            return
        if self.state != self.IDLE:
            # Аудио занято записью или полным WAV (НЕ нашим потоком):
            # вывод не стартуем, накопленное отбрасываем — ответ готов для повтора.
            self._stream_dropped = True
            return
        # none/pending — накапливаем до стартового буфера.
        self._stream = "pending"
        self._pending_sr = sample_rate
        self._pending.append(chunk)
        self._pending_samples += chunk.size
        if self._pending_samples >= self._buffer_samples(sample_rate):
            self._begin_stream()

    def _on_reply_ready(self, path: str) -> None:
        """Синтез завершён: доиграть поток ИЛИ проиграть готовый WAV.

        Поток уже шёл (active) — завершаем (остаток опустится → IDLE);
        накопил, но не стартовал (pending, короткий ответ) — стартуем с тем,
        что есть, и завершаем. Стриминг не шёл (полный WAV / fallback /
        занятость) — классический автозапуск, если аудио свободно, иначе
        ответ остаётся готовым для ручного ``replay_reply()``.
        """
        if self._shutdown:
            return
        # Новый готовый ответ принадлежит текущему обмену: ограничение
        # «не повторять старый» снимается — повтор воспроизведёт НОВЫЙ.
        self._reply_suppressed = False
        if self._stream == "active":
            self.player.end_stream()
        elif self._stream == "pending":
            if self.state == self.IDLE:
                self._begin_stream()
                self.player.end_stream()
            else:
                self._stream_dropped = True
        else:
            if self.state == self.IDLE:
                self.player.play(path)
            else:
                self.status = (
                    "Ответ готов; аудио занято — нажми «Повторить ответ»."
                )
                self.state_changed.emit()
        self._reset_stream_state()
        self._stream_dropped = True  # кускам этого задания больше не вестись

    def _on_reply_failed(self, reason: str) -> None:
        """Синтез не удался: остановить (частичный) вывод, понятный статус.

        Если звук уже начался (поток активен) — вывод останавливаем, но
        НЕ запускаем заново. Текст ответа остаётся в чате.
        """
        if self._shutdown:
            return
        if self.player.stream_active:
            self.player.stop_stream()
        self._reset_stream_state()
        self._stream_dropped = True
        self.status = f"Не удалось озвучить ответ: {reason}"
        self.state_changed.emit()

    # --- Внутреннее: потоковый вывод ----------------------------------------

    def _buffer_samples(self, sr: int) -> int:
        """Стартовый буфер в сэмплах (из ``stream_start_buffer_ms``)."""
        return max(1, int(round(self.stream_start_buffer_ms * sr / 1000.0)))

    def _begin_stream(self) -> None:
        """Стартует вывод и выгружает накопленные куски (если удалось)."""
        self.player.start_stream(self._pending_sr)
        if self.player.state != self.player.PLAYING:
            # Не вышло (занято/нет устройства) — накопленное не пускаем.
            self._stream_dropped = True
            self._pending = []
            self._pending_samples = 0
            return
        self._stream = "active"
        for ch in self._pending:
            self.player.write_chunk(ch)
        self._pending = []
        self._pending_samples = 0

    def _reset_stream_state(self, drop: bool = False) -> None:
        """Сбрасывает учёт потокового вывода (для нового ответа/завершения)."""
        self._stream = "none"
        self._pending = []
        self._pending_samples = 0
        self._pending_sr = 0
        if drop:
            self._stream_dropped = True

    def _cancel_tts(self) -> None:
        """Прерывает незавершённый синтез (если движок умеет ``cancel``)."""
        cancel = getattr(self.tts, "cancel", None)
        if callable(cancel):
            cancel()
