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

from unittest import mock

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtMultimedia import QMediaDevices, QMediaRecorder
from PySide6.QtWidgets import QApplication

from companion.audio_controller import AudioController
from companion.audio_player import AudioPlayer
from companion.audio_recorder import AudioRecorder

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
        self.assertIsNone(self.c.recorder._temp_dir)

    def test_actions_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.c.record()
        _pump(0.2)
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c.recorder._temp_dir)
        self.c.play()
        self.assertEqual(self.c.state, AudioController.IDLE)

    def test_late_signals_do_not_revive_state(self) -> None:
        """Поздние сигналы Qt после shutdown не должны «оживлять» состояние."""
        self.c.shutdown()
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.RecordingState
        )
        self.c.player._on_player_playing_changed(True)
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertIsNone(self.c.recorder._temp_dir)

    def test_capability_flags_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertFalse(self.c.can_record)
        self.assertFalse(self.c.can_play)


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
        self.assertIsNone(self.c.recorder._selected_input_id)
        self.assertIsNone(self.c.selected_input_name())
        device = self.c.recorder._selected_input_device()
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
        device = self.c.recorder._selected_input_device()
        self.assertEqual(
            self.c.recorder._device_id(device),
            self.c.recorder._selected_input_id,
        )

        # Совместимость: выбор по названию даёт тот же результат.
        self.assertTrue(self.c.set_input_device(name), self.c.status)
        self.assertEqual(self.c.selected_input_id(), dev_id)

        self.assertTrue(self.c.set_input_device(None), self.c.status)
        self.assertIsNone(self.c.selected_input_id())
        self.assertIn("системный", self.c.status)

    def test_unknown_device_rejected(self) -> None:
        """Несуществующий микрофон отклоняется; выбор не меняется."""
        self.c.recorder._selected_input_id = "sentinel-id"
        self.assertFalse(self.c.set_input_device("Нет-такого-микрофона"))
        self.assertIn("не найден", self.c.status)
        self.assertEqual(self.c.recorder._selected_input_id, "sentinel-id")

    def test_switch_rejected_while_recording(self) -> None:
        """Во время записи переключение запрещено (требование)."""
        self.c.recorder.state = AudioRecorder.RECORDING
        try:
            self.assertFalse(self.c.set_input_device("Любой"))
            self.assertIn("записи", self.c.status)
            self.assertIsNone(self.c.recorder._selected_input_id)
            # И сброс на системный тоже запрещён — это тоже переключение.
            self.assertFalse(self.c.set_input_device(None))
        finally:
            self.c.recorder.state = AudioRecorder.IDLE

    def test_reselect_falls_back_when_device_disappears(self) -> None:
        """Микрофон отключили: выбор сбрасывается на системный + сигнал."""
        self.c.recorder._selected_input_id = "b'gone-mic:input'"
        changed: list[int] = []
        self.c.devices_changed.connect(lambda: changed.append(1))

        self.c.recorder._reselect_if_needed()
        self.assertIsNone(self.c.recorder._selected_input_id)
        self.assertIn("системное устройство", self.c.status)

        # Полный обработчик hotplug: после фолбэка уходит devices_changed.
        self.c.recorder._on_input_devices_changed()
        self.assertEqual(changed, [1])
        self.assertIsNone(self.c.recorder._selected_input_id)

    def test_reselect_noop_when_system_selected(self) -> None:
        """Системный (None) выбранный: hotplug-фолбэк ничего не делает."""
        status_before = self.c.status
        self.c.recorder._selected_input_id = None
        self.c.recorder._reselect_if_needed()
        self.assertIsNone(self.c.recorder._selected_input_id)
        self.assertEqual(self.c.status, status_before)

    def test_set_input_device_rejected_after_shutdown(self) -> None:
        self.c.shutdown()
        self.assertFalse(self.c.set_input_device(None))
        self.assertFalse(self.c.set_input_device("Любой"))

    def test_hotplug_handler_is_noop_after_shutdown(self) -> None:
        self.c.shutdown()
        changed: list[int] = []
        self.c.devices_changed.connect(lambda: changed.append(1))
        self.c.recorder._on_input_devices_changed()
        self.assertEqual(changed, [])

    def test_hotplug_during_recording_resets_selection_for_next(self) -> None:
        """Отключение во время записи: выбор для следующей записи сброшен,
        UI получил devices_changed (комбо перестроится с «Системный»)."""
        changed: list[int] = []
        self.c.devices_changed.connect(lambda: changed.append(1))
        self.c.recorder._selected_input_id = "b'gone-mic:input'"
        self.c.recorder.state = AudioRecorder.RECORDING
        try:
            self.c.recorder._on_input_devices_changed()
            self.assertIsNone(self.c.recorder._selected_input_id)
            self.assertEqual(changed, [1])
        finally:
            self.c.recorder.state = AudioRecorder.IDLE


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
        self.c.recorder._temp_dir = self.tmp
        self.c.recorder._record_file = self.rec
        self.c.recorder.state = AudioRecorder.RECORDING

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def _device_error(self) -> None:
        # В этой сборке нет Error.DeviceError — потеря устройства
        # приходит как ResourceError.
        self.c.recorder._on_recorder_error(
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
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
        self.assertEqual(self.c.state, AudioController.IDLE)
        self.assertFalse(self.c.has_recording)
        self.assertIn("Ошибка записи", self.c.status)

    def test_error_after_stop_invalidates_partial_file(self) -> None:
        """Ошибка после Stopped: «готовность» частичного файла отменяется."""
        # Сначала «нормальная» остановка (файл существует) → выглядит готовым…
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
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
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
        self.assertFalse(
            self.c.recorder._session_failed, "флаг сбоя не сброшен"
        )

        # Следующая запись: новый файл, та же последовательность без ошибки.
        rec2 = os.path.join(self.tmp, "recording_2.m4a")
        with open(rec2, "wb") as f:
            f.write(b"good-bytes")
        self.c.recorder._record_file = rec2
        self.c.recorder.state = AudioRecorder.RECORDING
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
        self.assertTrue(self.c.has_recording, "успешная запись не готова")
        self.assertIn("Запись готова", self.c.status)


class TestRecordingReadiness(unittest.TestCase):
    """Готовность записи (регрессия дефекта 1): «пишущийся» файл не считается
    готовым; готовым запись становится только после подтверждения завершения
    рекордером; при ошибке частичная запись не публикуется как готовая.
    """

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-readiness-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"being-written-bytes")  # непустой файл «в записи»
        # Идёт запись: текущий файл непустой, готовой ещё нет.
        self.c.recorder._temp_dir = self.tmp
        self.c.recorder._record_file = self.rec
        self.c.recorder._ready_recording = None
        self.c.recorder.state = AudioRecorder.RECORDING

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def test_in_progress_file_not_ready(self) -> None:
        """Пока идёт запись, непустой файл НЕ готов (play/STT недоступны)."""
        self.assertFalse(self.c.has_recording, "файл в записи помечен готовым")
        self.assertIsNone(self.c.last_recording_path)

    def test_finalized_success_becomes_ready(self) -> None:
        """После подтверждения завершения (Stopped) запись становится готовой."""
        self.c.recorder._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
        self.assertTrue(self.c.has_recording, "успешная запись не готова")
        self.assertEqual(self.c.last_recording_path, self.rec)
        self.assertEqual(self.c.recorder._ready_recording, self.rec)

    def test_finalized_failure_not_ready(self) -> None:
        """Ошибка записи: частичный файл не публикуется как готовый."""
        self.c.recorder._on_recorder_error(
            QMediaRecorder.Error.ResourceError, "device lost"
        )
        self.assertFalse(self.c.has_recording, "частичный файл готов")
        self.assertIsNone(self.c.last_recording_path)
        self.assertIsNone(self.c.recorder._ready_recording)


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
        self.c.recorder._temp_dir = self.tmp
        self.c.recorder._record_file = self.rec
        self.c.recorder._ready_recording = self.rec
        self.player = _FakePlayer()
        self.c.player._player = self.player

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def _simulate_playing(self) -> None:
        """Имитируем «идёт воспроизведение»: состояние PLAYING + плеер играет."""
        self.c.player.state = AudioPlayer.PLAYING
        self.c.player.status = "Воспроизведение…"
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
        self.c.player._player_fail("не удалось открыть файл")
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
        self.c.player._player_fail("сбой")
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
        self.c.recorder._temp_dir = self.tmp
        self.c.recorder._record_file = self.wav
        self.c.recorder._ready_recording = self.wav
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
        temp_dir = self.c.recorder._temp_dir
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
        temp_dir = self.c.recorder._temp_dir
        record_file = self.c.recorder._record_file
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
        temp_dir = self.c.recorder._temp_dir
        self.c.shutdown()
        _pump(0.3)
        self.assertFalse(
            os.path.isdir(temp_dir or ""),
            f"каталог записи не удалён: {temp_dir!r}",
        )


class TestAudioPlayerBoundary(unittest.TestCase):
    """Граница AudioPlayer: play(path)/stop/ошибки/shutdown — в изоляции,
    без оборудования (выход и QAudioOutput подменяются) и без рекордера.

    Ключевое: плеер НЕ удаляет переданный файл — даже при shutdown.
    """

    def setUp(self) -> None:
        _app()
        # Выход и класс QAudioOutput подменяем в самом модуле плеера:
        # ни реального устройства, ни реального QAudioOutput не создаём.
        self.output_cls = mock.Mock(name="QAudioOutput-class")
        self.devices = mock.Mock(name="QMediaDevices")
        self.device = mock.Mock(name="output-device")
        self.device.isNull.return_value = False
        self.device.description.return_value = "fake output"
        self.devices.defaultAudioOutput.return_value = self.device
        self._p_output = mock.patch("companion.audio_player.QAudioOutput", self.output_cls)
        self._p_devices = mock.patch("companion.audio_player.QMediaDevices", self.devices)
        self._p_output.start()
        self._p_devices.start()

        self.player = AudioPlayer()
        self.fake = _FakePlayer()
        self.player._player = self.fake

        self.tmp = tempfile.mkdtemp(prefix="english-companion-playb-")
        self.audio_path = os.path.join(self.tmp, "tts_reply.m4a")
        with open(self.audio_path, "wb") as f:
            f.write(b"ready-audio-bytes")

    def tearDown(self) -> None:
        self.player.shutdown()
        _pump(0.1)
        self._p_output.stop()
        self._p_devices.stop()
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.player

    def test_play_with_explicit_path_starts_playback(self) -> None:
        """play(path) проигрывает ЛЮБОЙ готовый файл (не только запись)."""
        self.player.play(self.audio_path)
        self.assertEqual(self.player.state, AudioPlayer.PLAYING)
        self.assertIn("Воспроизведение", self.player.status)
        self.assertEqual(self.fake.play_calls, 1)
        self.output_cls.assert_called_once()

    def test_play_rejected_without_output(self) -> None:
        """Нет устройства вывода — понятный отказ, плеер не запускается."""
        self.device.isNull.return_value = True
        self.player.play(self.audio_path)
        self.assertEqual(self.player.state, AudioPlayer.IDLE)
        self.assertIn("Нет устройства вывода звука", self.player.status)
        self.assertEqual(self.fake.play_calls, 0)

    def test_play_rejected_missing_file(self) -> None:
        """Файла нет — понятный отказ, плеер не запускается."""
        self.player.play(os.path.join(self.tmp, "no-such-file.wav"))
        self.assertEqual(self.player.state, AudioPlayer.IDLE)
        self.assertIn("Нет аудиофайла", self.player.status)
        self.assertEqual(self.fake.play_calls, 0)

    def test_double_play_is_noop(self) -> None:
        """Повторный play во время воспроизведения игнорируется."""
        self.player.play(self.audio_path)
        self.player.play(self.audio_path)
        self.assertEqual(self.fake.play_calls, 1, "play() запущен дважды")

    def test_stop_returns_idle(self) -> None:
        """stop() останавливает плеер и возвращает IDLE."""
        self.player.play(self.audio_path)
        self.player.stop()
        self.assertEqual(self.fake.stop_calls, 1)
        self.assertEqual(self.player.state, AudioPlayer.IDLE)
        self.assertIn("Голос: готов", self.player.status)

    def test_player_error_returns_idle(self) -> None:
        """Ошибка воспроизведения возвращает согласованное состояние."""
        self.player.play(self.audio_path)
        self.player._player_fail("не удалось открыть файл")
        self.assertEqual(self.player.state, AudioPlayer.IDLE)
        self.assertIn("Ошибка воспроизведения", self.player.status)

    def test_shutdown_stops_and_rejects_new_play(self) -> None:
        """После shutdown плеер не работает и play отклоняется."""
        self.player.play(self.audio_path)
        self.player.shutdown()
        self.assertTrue(self.player.is_shutdown)
        self.assertIsNone(self.player._player)
        self.assertFalse(self.player.can_play)
        self.player.play(self.audio_path)
        self.assertEqual(self.fake.play_calls, 1, "play() после shutdown")
        self.player.stop()  # повторный stop безопасен
        self.assertEqual(self.player.state, AudioPlayer.IDLE)

    def test_played_file_survives_shutdown(self) -> None:
        """Плеер НЕ удаляет переданный файл (владелец файла — другой)."""
        self.player.play(self.audio_path)
        self.player.shutdown()
        _pump(0.2)
        self.assertTrue(
            os.path.exists(self.audio_path),
            "плеер удалил пользовательский файл",
        )


class TestAudioRecorderBoundary(unittest.TestCase):
    """Граница AudioRecorder: готовность только после Stopped, поздние
    сигналы после shutdown — no-op, временные файлы — его владелец.
    """

    def setUp(self) -> None:
        _app()
        self.r = AudioRecorder()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-recb-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"rec-bytes")

    def tearDown(self) -> None:
        self.r.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.r

    def test_ready_only_after_stopped(self) -> None:
        """«Пишущийся» файл не готов; готовым запись становится после Stopped."""
        self.r._temp_dir = self.tmp
        self.r._record_file = self.rec
        self.r.state = AudioRecorder.RECORDING
        self.assertFalse(
            self.r.has_recording, "файл в записи помечен готовым"
        )
        self.assertIsNone(self.r.last_recording_path)
        self.r._on_recorder_state(QMediaRecorder.RecorderState.StoppedState)
        self.assertTrue(self.r.has_recording)
        self.assertEqual(self.r.last_recording_path, self.rec)

    def test_record_while_recording_is_noop(self) -> None:
        """record() во время записи не стартует новую сессию."""
        self.r.state = AudioRecorder.RECORDING
        seq_before = self.r._record_seq
        self.r.record()
        self.assertEqual(
            self.r._record_seq, seq_before, "стартовала новая запись"
        )
        self.assertIsNone(
            self.r._record_file, "назначен новый файл записи"
        )
        self.assertEqual(self.r.state, AudioRecorder.RECORDING)

    def test_late_signals_noop_after_shutdown(self) -> None:
        """Поздние сигналы Qt после shutdown не «оживляют» состояние."""
        self.r.shutdown()
        self.r._on_recorder_state(
            QMediaRecorder.RecorderState.RecordingState
        )
        self.r._on_recorder_state(
            QMediaRecorder.RecorderState.StoppedState
        )
        self.r._on_recorder_error(
            QMediaRecorder.Error.ResourceError, "late error"
        )
        self.r._on_input_devices_changed()
        self.assertEqual(self.r.state, AudioRecorder.IDLE)
        self.assertIsNone(self.r._ready_recording)
        self.assertFalse(self.r.can_record)

    def test_temp_dir_cleaned_on_shutdown(self) -> None:
        """Временный каталог — владелец рекордер: shutdown удаляет его."""
        self.r._temp_dir = self.tmp
        self.r.shutdown()
        _pump(0.2)
        self.assertFalse(
            os.path.isdir(self.tmp), "временная папка рекордера осталась"
        )
        self.assertIsNone(self.r._temp_dir)
        self.assertIsNone(self.r._record_file)
        self.assertIsNone(self.r._ready_recording)


class TestFacadeMutualExclusion(unittest.TestCase):
    """Фасад: запись и воспроизведение взаимно исключают друг друга
    (молчаливый отказ — как раньше).
    """

    def setUp(self) -> None:
        _app()
        self.c = AudioController()
        self.tmp = tempfile.mkdtemp(prefix="english-companion-mutex-")
        self.rec = os.path.join(self.tmp, "recording_1.m4a")
        with open(self.rec, "wb") as f:
            f.write(b"rec-bytes")
        self.c.recorder._temp_dir = self.tmp
        self.c.recorder._record_file = self.rec
        self.c.recorder._ready_recording = self.rec
        self.fake = _FakePlayer()
        self.c.player._player = self.fake

    def tearDown(self) -> None:
        self.c.shutdown()
        _pump(0.1)
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)
        del self.c

    def test_play_rejected_while_recording(self) -> None:
        """Во время записи play() не запускает плеер (фасад отклоняет)."""
        self.c.recorder.state = AudioRecorder.RECORDING
        try:
            self.c.play()
            self.assertEqual(
                self.fake.play_calls, 0, "плеер запущен во время записи"
            )
            self.assertEqual(self.c.state, AudioController.RECORDING)
            self.assertEqual(self.c.player.state, AudioPlayer.IDLE)
        finally:
            self.c.recorder.state = AudioRecorder.IDLE

    def test_record_rejected_while_playing(self) -> None:
        """Во время воспроизведения record() не стартует сессию (фасад)."""
        self.c.player.state = AudioPlayer.PLAYING
        try:
            self.c.record()
            self.assertEqual(
                self.c.recorder._record_seq, 0, "запись стартовала в play"
            )
            self.assertIsNone(
                self.c.recorder._recorder, "создана сессия записи"
            )
            self.assertEqual(self.c.state, AudioController.PLAYING)
            self.assertEqual(self.c.recorder.state, AudioRecorder.IDLE)
        finally:
            self.c.player.state = AudioPlayer.IDLE
