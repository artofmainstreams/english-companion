"""Автоматические проверки жизненного цикла AudioController.

Запуск (из корня проекта)::

    .venv/bin/python -m unittest discover -s tests -v

— безопасные тесты контроллера (идемпотентность shutdown, сторожевые
флаги); к устройствам звука они не обращаются и проходят без них.

Полный прогон с реальными записью и воспроизведением (нужны микрофон,
выход звука и разрешения macOS)::

    RUN_AUDIO_TESTS=1 .venv/bin/python -m unittest discover -s tests -v

Семантика результатов:
* RUN_AUDIO_TESTS не задан — реальные аудиотесты SKIP без запроса
  устройств;
* устройство отсутствует — допустимая причина SKIP;
* отказ записи/воспроизведения уже после начала сценария, пустой файл
  записи и неверное состояние — FAIL: это дефект контроллера, а не
  особенность окружения.
"""

import faulthandler
import os
import shutil
import sys
import tempfile
import unittest
import wave

faulthandler.enable()

# offscreen — чтобы тесты не зависели от дисплея; аудио при этом работает.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtMultimedia import QMediaDevices, QMediaRecorder
from PySide6.QtWidgets import QApplication

from companion.audio_controller import AudioController

_QAPP: QApplication | None = None


def _app() -> QApplication:
    global _QAPP
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication(sys.argv)
    return _QAPP


def _pump(seconds: float) -> None:
    """Пускаем цикл событий на ``seconds`` секунд (нужно для deleteLater)."""
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def _audio_tests_enabled() -> bool:
    """Реальные аудиотесты включаются только явным флагом."""
    return os.environ.get("RUN_AUDIO_TESTS") == "1"


def _skip_if_audio_tests_disabled() -> None:
    raise unittest.SkipTest(
        "RUN_AUDIO_TESTS не равен 1 — реальные аудиотесты отключены"
    )


def _has_audio_input() -> bool:
    _app()
    try:
        return not QMediaDevices.defaultAudioInput().isNull()
    except Exception:
        return False


def _has_audio_output() -> bool:
    _app()
    try:
        return not QMediaDevices.defaultAudioOutput().isNull()
    except Exception:
        return False


def _make_silent_wav(path: str, seconds: int = 1) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000 * seconds)


class TestControllerGuards(unittest.TestCase):
    """Проверки без оборудования: идемпотентность shutdown, сторожевые флаги."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()

    def tearDown(self) -> None:
        self.c.shutdown()
        del self.c
        _pump(0.1)

    def test_double_shutdown_cold(self) -> None:
        """Повторный shutdown безопасен и оставляет контроллер мёртвым."""
        self.c.shutdown()
        self.c.shutdown()
        _pump(0.2)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c._temp_dir)

    def test_actions_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.c.record()
        _pump(0.2)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c._temp_dir)
        self.c.play()
        self.assertEqual(self.c.state, AudioController.IDLE)

    def test_late_signals_do_not_revive_state(self) -> None:
        """Поздние сигналы Qt после shutdown не должны «оживлять» состояние."""
        self.c.shutdown()
        self.c._on_recorder_state(QMediaRecorder.RecorderState.RecordingState)
        self.c._on_player_playing_changed(True)
        self.c._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c._temp_dir)

    def test_capability_flags_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertFalse(self.c.can_record)
        self.assertFalse(self.c.can_play)


class TestSaveRecording(unittest.TestCase):
    """Сохранение последней записи для диагностики (без оборудования)."""

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-savetest-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"fake-audio-bytes")
        # «Завершённая запись» (в тесте допустимо трогать приватные поля;
        # окно и контроллер работают через публичный API).
        self.c._temp_dir = self.tmp
        self.c._record_file = self.rec
        self.dest_dir = tempfile.mkdtemp(prefix="english-companion-saved-")

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        for d in (self.tmp, self.dest_dir):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        del self.c

    def test_save_copies_recording_and_reports_path(self) -> None:
        dest = self.c.save_last_recording(self.dest_dir)
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"fake-audio-bytes")
        # Оригинал во временной папке до закрытия не тронут.
        self.assertTrue(os.path.exists(self.rec))
        self.assertIn("Запись сохранена", self.c.status)

    def test_saved_file_survives_shutdown(self) -> None:
        """После сохранения копия переживает закрытие; временные файлы уходят."""
        dest = self.c.save_last_recording(self.dest_dir)
        self.assertIsNotNone(dest)
        self.c.shutdown()
        _pump(0.2)
        self.assertTrue(
            os.path.exists(dest),
            f"сохранённая копия удалена при закрытии: {dest!r}",
        )
        self.assertFalse(os.path.isdir(self.tmp), "временная папка должна уйти")

    def test_saved_inside_temp_dir_survives_shutdown(self) -> None:
        """Даже копия внутри временной папки не удаляется при закрытии."""
        dest = self.c.save_last_recording(self.tmp)
        self.assertIsNotNone(dest)
        self.c.shutdown()
        _pump(0.2)
        self.assertTrue(
            os.path.exists(dest),
            f"копия во временной папке удалена при закрытии: {dest!r}",
        )

    def test_save_without_recording_returns_none(self) -> None:
        self.c._record_file = None
        self.assertIsNone(self.c.save_last_recording(self.dest_dir))

    def test_save_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertIsNone(self.c.save_last_recording(self.dest_dir))


class TestInputDeviceSelection(unittest.TestCase):
    """Выбор устройства ввода: выбор/сброс, hotplug-фолбэк, сторожевые флаги.

    Реальный захват звука не включается — проверяется логика выбора
    (разрешение устройства, статусы, сигналы), а не запись.
    """

    def setUp(self) -> None:
        _app()
        self.c = AudioController()

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        del self.c

    def test_default_is_system_device(self) -> None:
        """Без явного выбора записывается системное устройство по умолчанию."""
        if not _has_audio_input():
            self.skipTest("нет устройств ввода — выбрать нечего")
        self.assertIsNone(self.c._selected_input_id)
        self.assertIsNone(self.c.selected_input_name())
        device = self.c._selected_input_device()
        self.assertFalse(device.isNull())
        self.assertEqual(
            device.id(),
            QMediaDevices.defaultAudioInput().id(),
            "без выбора должен разрешаться системный микрофон",
        )

    def test_select_by_id_and_name(self) -> None:
        """Выбор работает и по id (путь UI), и по названию; ``None`` — системный."""
        pairs = self.c.input_devices()
        if not pairs:
            self.skipTest("нет устройств ввода — выбрать нечего")
        dev_id, name = pairs[0]
        self.assertTrue(self.c.set_input_device(dev_id), self.c.status)
        self.assertEqual(self.c.selected_input_id(), dev_id)
        self.assertEqual(self.c.selected_input_name(), name)
        # Разрешение: устройство с тем же идентификатором, что выбрано.
        device = self.c._selected_input_device()
        self.assertEqual(self.c._device_id(device), self.c._selected_input_id)

        # Совместимость: выбор по названию даёт тот же результат.
        self.assertTrue(self.c.set_input_device(name), self.c.status)
        self.assertEqual(self.c.selected_input_id(), dev_id)

        self.assertTrue(self.c.set_input_device(None), self.c.status)
        self.assertIsNone(self.c.selected_input_id())
        self.assertIn("системный", self.c.status)

    def test_unknown_device_rejected(self) -> None:
        """Несуществующий микрофон отклоняется; выбор не меняется."""
        self.c._selected_input_id = "sentinel-id"
        self.assertFalse(self.c.set_input_device("Нет-такого-микрофона"))
        self.assertIn("не найден", self.c.status)
        self.assertEqual(self.c._selected_input_id, "sentinel-id")

    def test_switch_rejected_while_recording(self) -> None:
        """Во время записи переключение запрещено (требование)."""
        self.c.state = self.c.RECORDING
        try:
            self.assertFalse(self.c.set_input_device("Любой"))
            self.assertIn("записи", self.c.status)
            self.assertIsNone(self.c._selected_input_id)
            # И сброс на системный тоже запрещён — это тоже переключение.
            self.assertFalse(self.c.set_input_device(None))
        finally:
            self.c.state = self.c.IDLE

    def test_reselect_falls_back_when_device_disappears(self) -> None:
        """Микрофон отключили: выбор сбрасывается на системный + сигнал."""
        self.c._selected_input_id = "b'gone-mic:input'"
        changed: list[int] = []
        self.c.devices_changed.connect(lambda: changed.append(1))

        self.c._reselect_if_needed()
        self.assertIsNone(self.c._selected_input_id)
        self.assertIn("системное устройство", self.c.status)

        # Полный обработчик hotplug: после фолбэка уходит devices_changed.
        self.c._on_input_devices_changed()
        self.assertEqual(changed, [1])
        self.assertIsNone(self.c._selected_input_id)

    def test_reselect_noop_when_system_selected(self) -> None:
        """Системный (None) выбранный: hotplug-фолбэк ничего не делает."""
        status_before = self.c.status
        self.c._selected_input_id = None
        self.c._reselect_if_needed()
        self.assertIsNone(self.c._selected_input_id)
        self.assertEqual(self.c.status, status_before)

    def test_set_input_device_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertFalse(self.c.set_input_device(None))
        self.assertFalse(self.c.set_input_device("Любой"))

    def test_hotplug_handler_is_noop_after_shutdown(self) -> None:
        self.c.shutdown()
        changed: list[int] = []
        self.c.devices_changed.connect(lambda: changed.append(1))
        self.c._on_input_devices_changed()
        self.assertEqual(changed, [])

    def test_hotplug_during_recording_resets_selection_for_next(self) -> None:
        """Отключение во время записи: выбор для следующей записи сброшен,
        UI получил devices_changed (комбо перестроится с «Системный»)."""
        changed: list[int] = []
        self.c.devices_changed.connect(lambda: changed.append(1))
        self.c._selected_input_id = "b'gone-mic:input'"
        self.c.state = self.c.RECORDING
        try:
            self.c._on_input_devices_changed()
            self.assertIsNone(self.c._selected_input_id)
            self.assertEqual(changed, [1])
        finally:
            self.c.state = self.c.IDLE


class TestRecordingError(unittest.TestCase):
    """Сбой сессии записи (например, отключение микрофона во время записи):
    частичный файл не считается готовой записью — в любом порядке событий.
    """

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-errtest-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"partial-bytes")
        # «Идёт запись» с фактическим файлом (без реального захвата).
        self.c._temp_dir = self.tmp
        self.c._record_file = self.rec
        self.c.state = self.c.RECORDING

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def _device_error(self) -> None:
        # В этой сборке нет Error.DeviceError — потеря устройства
        # приходит как ResourceError.
        self.c._on_recorder_error(
            QMediaRecorder.Error.ResourceError, "device lost"
        )

    def test_error_during_recording_invalidates_file(self) -> None:
        """Ошибка до Stopped: файл не помечается готовым, есть понятный статус."""
        self._device_error()
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertFalse(
            self.c.has_recording, "частичный файл помечен готовым"
        )
        self.assertIn("Ошибка записи", self.c.status)
        # Поздний Stopped не «оживляет» запись.
        self.c._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertFalse(self.c.has_recording)
        self.assertIn("Ошибка записи", self.c.status)

    def test_error_after_stop_invalidates_partial_file(self) -> None:
        """Ошибка после Stopped: «готовность» частичного файла отменяется."""
        # Сначала «нормальная» остановка (файл существует) → выглядит готовым…
        self.c._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertTrue(self.c.has_recording)
        # …потом приходит ошибка сессии: готовность отменяется.
        self._device_error()
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertFalse(
            self.c.has_recording, "частичный файл остался помечен готовым"
        )
        self.assertIn("Ошибка записи", self.c.status)

    def test_next_recording_after_failure_is_ready(self) -> None:
        """Флаг сбоя не переносится на следующую (успешную) запись."""
        self._device_error()
        self.c._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertFalse(self.c._session_failed, "флаг сбоя не сброшен")

        # Следующая запись: новый файл, та же последовательность без ошибки.
        rec2 = os.path.join(self.tmp, "recording_2.m4a")
        with open(rec2, "wb") as f:
            f.write(b"good-bytes")
        self.c._record_file = rec2
        self.c.state = self.c.RECORDING
        self.c._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertTrue(self.c.has_recording, "успешная запись не готова")
        self.assertIn("Запись готова", self.c.status)


class _FakePlayer:
    """Заглушка QMediaPlayer: фиксирует вызовы play/stop и состояние,
    без аудиооборудования. Позволяет проверить, что «Остановить» и ошибка
    действительно останавливают плеер, и что после этого звук не запускается
    сам (нет фоновых таймеров/колбэков, дёргающих ``play()``).
    """

    def __init__(self) -> None:
        self.play_calls = 0
        self.stop_calls = 0
        self._state = "Stopped"

    def play(self) -> None:
        self.play_calls += 1
        self._state = "Playing"

    def stop(self) -> None:
        self.stop_calls += 1
        self._state = "Stopped"

    def playbackState(self) -> str:
        return self._state

    def position(self) -> int:
        return 0

    def setSource(self, *args) -> None:
        pass

    def setAudioOutput(self, *args) -> None:
        pass

    def deleteLater(self) -> None:
        pass


class TestPlaybackLifecycle(unittest.TestCase):
    """Поведение пользователя при воспроизведении (без аудиооборудования):

    * «Остановить» останавливает плеер и переводит контроллер в IDLE;
    * ошибка воспроизведения возвращает контроллер в согласованное состояние;
    * после остановки/ошибки звук не запускается сам;
    * после ``shutdown`` новые операции (``record()``/``play()``) отклоняются.

    Плеер подменяется ``_FakePlayer``: проверяем взаимодействие контроллера с
    плеером, а не реальное устройство.
    """

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        # «Готовая запись», чтобы действия на воспроизведение дошли до логики.
        self.tmp = tempfile.mkdtemp(prefix="english-companion-playlifecyc-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"rec-bytes")
        self.c._temp_dir = self.tmp
        self.c._record_file = self.rec
        self.player = _FakePlayer()
        self.c._player = self.player

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def _simulate_playing(self) -> None:
        """Имитируем «идёт воспроизведение»: состояние PLAYING + плеер играет."""
        self.c.state = self.c.PLAYING
        self.c.status = "Воспроизведение…"
        self.player.play()

    def test_stop_stops_playback(self) -> None:
        """«Остановить» останавливает плеер и переводит контроллер в IDLE."""
        self._simulate_playing()
        self.c.stop_playback()
        self.assertGreaterEqual(self.player.stop_calls, 1, "stop() не вызван")
        self.assertEqual(self.player.playbackState(), "Stopped")
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIn("Голос: готов", self.c.status)

    def test_error_returns_consistent_state(self) -> None:
        """Ошибка воспроизведения возвращает контроллер в согласованное состояние."""
        self._simulate_playing()
        self.c._player_fail("не удалось открыть файл")
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIn("Ошибка воспроизведения", self.c.status)

    def test_no_spontaneous_playback_after_stop(self) -> None:
        """После «Остановить» звук не запускается сам (нет фоновых пусков)."""
        self._simulate_playing()
        self.c.stop_playback()
        # Пускаем цикл событий: если бы был фоновый таймер/колбэк, дёргающий
        # play(), play_calls вырос бы выше начального.
        _pump(0.4)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertEqual(self.player.playbackState(), "Stopped")
        self.assertEqual(
            self.player.play_calls, 1, "play() вызывался повторно сам"
        )

    def test_no_spontaneous_playback_after_error(self) -> None:
        """После ошибки звук не запускается сам."""
        self._simulate_playing()
        self.c._player_fail("сбой")
        # Пускаем цикл событий: фоновых механизмов, дёргающих play(), нет.
        _pump(0.4)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertEqual(
            self.player.play_calls, 1, "play() вызывался сам после ошибки"
        )

    def test_shutdown_rejects_new_operations(self) -> None:
        """После ``shutdown`` новые операции (record/play) отклоняются."""
        self.c.shutdown()
        _pump(0.1)
        self.c.record()
        self.c.play()
        _pump(0.1)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertFalse(self.c.can_record, "record() должен быть отклонён")
        self.assertFalse(self.c.can_play, "play() должен быть отклонён")
        self.assertTrue(self.c.is_shutdown)


class TestAudioPlayback(unittest.TestCase):
    """Воспроизведение: реальный выход (сценарий на тихом WAV)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _audio_tests_enabled():
            _skip_if_audio_tests_disabled()
        if not _has_audio_output():
            raise unittest.SkipTest("нет устройства вывода звука")
        super().setUpClass()

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-test-")
        self.wav = os.path.join(self.tmp, "silent.wav")
        _make_silent_wav(self.wav)
        self.c._temp_dir = self.tmp
        self.c._record_file = self.wav
        # Старт сценария. Если он не запускается, тест сам сообщит FAIL
        # (устройство же проверено в setUpClass) — а не пропустится.
        self.c.play()
        _pump(0.5)

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.2)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def test_shutdown_during_playback(self) -> None:
        """Закрытие окна во время воспроизведения: без падений, файлы удалены."""
        self.assertEqual(self.c.state, AudioController.PLAYING, self.c.status)
        self.c.shutdown()
        _pump(0.5)
        self.assertFalse(os.path.isdir(self.tmp), "папка с файлом осталась")
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.shutdown()  # идемпотентность
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.record()
        self.c.play()
        self.assertEqual(self.c.state, AudioController.IDLE)


class TestAudioRecording(unittest.TestCase):
    """Запись: реальный микрофон (и разрешения ОС)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _audio_tests_enabled():
            _skip_if_audio_tests_disabled()
        if not _has_audio_input():
            raise unittest.SkipTest(
                "нет микрофона — аудиотесты требуют оборудования"
            )
        super().setUpClass()

    def setUp(self) -> None:
        _app()
        self.c = AudioController()

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.2)
        temp_dir = self.c._temp_dir
        if temp_dir is not None and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        del self.c

    def test_shutdown_during_recording(self) -> None:
        """Закрытие окна во время записи: без падений, микрофон и файлы отпущены."""
        self.c.record()
        _pump(2.0)
        # Устройство проверено в setUpClass, значит «не стартовала» —
        # это отказ уже начавшегося сценария, а не SKIP.
        self.assertEqual(
            self.c.state,
            AudioController.RECORDING,
            f"запись не стартовала: {self.c.status!r}",
        )
        # Путь сохраняем до shutdown(): проверяем фактическое удаление
        # каталога, а не только None в поле контроллера.
        temp_dir = self.c._temp_dir
        record_file = self.c._record_file
        self.c.shutdown()
        _pump(0.5)
        self.assertFalse(
            os.path.isdir(temp_dir or ""),
            f"каталог записи не удалён: {temp_dir!r}",
        )
        if record_file is not None:
            self.assertFalse(
                os.path.exists(record_file),
                f"файл записи остался: {record_file!r}",
            )
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.c.shutdown()  # идемпотентность
        self.assertEqual(self.c.state, AudioController.IDLE)

    def test_record_play_cycles(self) -> None:
        """Несколько циклов record→stop→play→stop без висячих ссылок."""
        if not _has_audio_output():
            self.skipTest("нет устройства вывода — цикл record→play невозможен")
        for i in range(2):
            self.c.record()
            _pump(1.5)
            self.assertEqual(
                self.c.state,
                AudioController.RECORDING,
                f"цикл {i + 1}: запись не стартовала: {self.c.status!r}",
            )
            self.c.stop_recording()
            _pump(1.0)
            self.assertEqual(
                self.c.state,
                AudioController.IDLE,
                f"цикл {i + 1}: после stop_recording, status={self.c.status!r}",
            )
            # Пустой файл после успешной записи — это дефект, не SKIP.
            self.assertTrue(
                self.c.has_recording,
                f"цикл {i + 1}: файл записи пуст: {self.c.status!r}",
            )
            self.c.play()
            _pump(0.5)
            self.assertEqual(
                self.c.state,
                AudioController.PLAYING,
                f"цикл {i + 1}: play не стартовал: {self.c.status!r}",
            )
            self.c.stop_playback()
            _pump(0.5)
            self.assertEqual(
                self.c.state,
                AudioController.IDLE,
                f"цикл {i + 1}: после stop_playback, status={self.c.status!r}",
            )
        # Путь до shutdown(): проверяем, что каталог реально удалён.
        temp_dir = self.c._temp_dir
        self.c.shutdown()
        _pump(0.3)
        self.assertFalse(
            os.path.isdir(temp_dir or ""),
            f"каталог записи не удалён: {temp_dir!r}",
        )
