"""Фасад аудиослоя: запись и воспроизведение в одном слое для UI.

Аудиослой разбит на два независимых компонента:

  * :class:`companion.audio_recorder.AudioRecorder` — устройства ввода,
    запись, результат успешно завершённой записи; явный владелец своих
    временных файлов (создаёт и удаляет их);
  * :class:`companion.audio_player.AudioPlayer` — воспроизведение любого
    готового аудиофайла; файл ему НЕ принадлежит, и плеер его никогда
    не удаляет (это же позволит проигрывать TTS-файлы ответов).

``AudioController`` — тонкий фасад с прежним публичным API, поэтому UI
(``MainWindow``) и сборка приложения (``main.py``) не меняются:

  * единое производное состояние ``IDLE/RECORDING/PLAYING``;
  * ``state``/``status`` синхронизируются из компонента, который
    последний раз подал ``state_changed``; ``devices_changed`` уходит
    с рекордера;
  * взаимное исключение записи и воспроизведения — молчаливый отказ,
    как раньше;
  * порядок ``shutdown()``: сначала плеер останавливается и отпускает
    файл и выход (файл при этом НЕ удаляется), потом рекордер разбирает
    сессию и удаляет СВОИ временные файлы.
"""

from PySide6.QtCore import QObject, Signal

from .audio_player import AudioPlayer
from .audio_recorder import AudioRecorder


class AudioController(QObject):
    """Управляет записью и воспроизведением речи без блокировки интерфейса.

    Работает как отдельный слой (как SessionController/OllamaClient):
    MainWindow только показывает состояние и вызывает методы, сам Qt
    Multimedia живёт в компонентах и не мешает текстовому чату.

    Жизненный цикл:
    * ``shutdown()`` идемпотентна — повторные вызовы безопасны;
    * после ``shutdown()`` ``record()``/``play()`` отклоняются, а поздние
      сигналы Qt игнорируются (состояние больше не «оживает»);
    * разбор по порядку: остановка плеера → выход → разбор сессии записи
      → удаление временных файлов рекордера.
    """

    state_changed = Signal()
    # Список устройств ввода изменился (подключение/отключение микрофона):
    # уходит с рекордера, интерфейс перечитывает список.
    devices_changed = Signal()

    IDLE = "idle"
    RECORDING = "recording"
    PLAYING = "playing"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.status = "Голос: готов"

        self.player = AudioPlayer(self)
        self.recorder = AudioRecorder(self)

        # Статус и состояние синхронизируются из компонента, который
        # последний раз подал state_changed.
        self.player.state_changed.connect(self._sync_from_player)
        self.recorder.state_changed.connect(self._sync_from_recorder)
        self.recorder.devices_changed.connect(self.devices_changed.emit)

        # Флаг shutdown фасада: после него он не реагирует на сигналы
        # компонентов и не пускает новые операции.
        self._shutdown = False

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
        if self._shutdown:
            return
        self.player.stop()

    def save_last_recording(self, dest_dir: str | None = None) -> str | None:
        return self.recorder.save_last_recording(dest_dir)

    def shutdown(self) -> None:
        """Останавливает запись/воспроизведение и разбирает оба компонента.

        Идемпотентна: повторные вызовы (например, ``aboutToQuit`` и явный
        вызов из кода) безопасны. После вызова слой не работает:
        ``record()``/``play()`` отклоняются, поздние сигналы Qt игнорируются.

        Порядок важен: сначала плеер останавливается и отпускает файл/выход
        (файл он НЕ удаляет — владелец файла это делает), потом рекордер
        разбирает сессию и удаляет СВОИ временные файлы.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # 1) Плеер: остановка, отсоединение выхода, отпускание источника.
        #    Переданный ранее файл при этом НЕ удаляется.
        self.player.shutdown()

        # 2) Рекордер: отключение hotplug, разбор графа, временные файлы.
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
