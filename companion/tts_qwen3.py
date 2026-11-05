"""Озвучивание ответов преподавателя: Qwen3-TTS (MLX) — фоновый синтез.

``Qwen3TtsSynth`` — альтернатива :class:`companion.tts_synth.TtsSynth`
с тем же публичным API (состояния, сигналы, действия, владелец своих
временных файлов), но синтез выполняет локальная модель Qwen3-TTS
(mlx-audio), а не процесс ``say``:

* ``synthesize(text)`` — НЕБЛОКИРУЮЩАЯ: ставит задание в очередь
  рабочего потока и сразу возвращается; готовность придёт сигналом
  ``reply_ready(path)``/``failed(reason)`` — цикл событий Qt не стоит;
* рабочий поток лениво загружает модель при ПЕРВОМ синтезе (время
  загрузки — как отдельная часть первой задержки ответа), модель
  переиспользуется для всех следующих ответов;
* новый ответ СМЕНЯЕТ незавершённый: задание с меньшим номером
  отбрасывается (вычисления модели не прерываются, но её результат
  не публикуется и частичный файл не появляется);
* компонент — ЯВНЫЙ ВЛАДЕЛЕЦ своих временных WAV: каталог создаёт при
  первом готовом ответе и удаляет в ``shutdown()``;
* повтор (``last_reply_path``) — тот же готовый WAV, без нового синтеза.

Два способа синтеза (выбираются в ``AppConfig``; см. ``build_tts``):

* ``voice_design`` — модель VoiceDesign + описание голоса
  (``instruct``; по умолчанию — выбранный голос D, образец
  ``qwen3_low_D.wav``);
* ``reference`` — модель Base + одна чистая реплика голоса
  (``ref_audio`` + точный ``ref_text``) как референс клонирования.

Импорт ``mlx_audio`` ЛЕНИВЫЙ (внутри рабочего потока): приложение,
``build_tts`` и тесты работают без mlx-audio и без весов. Для тестов
можно передать готовый ``synthesizer`` — вызываемый
``(text) -> (audio: float-массив, sample_rate: int)`` — тогда модель
не нужна вовсе.

Публичные сигналы — как у ``TtsSynth``: ``state_changed``,
``reply_ready(path)`` (файл готов и существует), ``failed(reason)``
(состояние возвращается IDLE).
"""

import logging
import os
import queue
import shutil
import tempfile
import threading
import wave

import numpy as np

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

# Выбранный голос преподавателя — вариант D (образец qwen3_low_D.wav):
# общее описание + регистр, ДОСЛОВНО как в tools/qwen3_low_register_samples.py.
DEFAULT_INSTRUCT = (
    "A young adult woman with a natural southern English accent. "
    "Her voice is soft, warm and velvety, with clear diction and "
    "gentle, natural intonation. She speaks at a relaxed conversational pace, "
    "like a patient tutor talking to one student. "
    "She speaks in a deep, naturally low alto register, "
    "with a full, warm resonance and a delicate, clear delivery."
)
# Репозитории Hugging Face (веса — в кэше ~/.cache/huggingface, вне Git):
# voice_design — описание голоса; reference (Base) — клонирование по референсу.
DEFAULT_MODEL_VOICE_DESIGN = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DEFAULT_MODEL_REFERENCE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
# Точный текст первой реплики варианта D (референс способа reference).
DEFAULT_REF_TEXT = "Hi! How was your day? Tell me one thing you enjoyed."

# Метка очереди: «остановись» (shutdown).
_SENTINEL = object()


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Пишет 16-bit PCM WAV; пик ограничен 0.98 (без клиппинга)."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


class Qwen3TtsSynth(QObject):
    """Синтезирует текст ответа WAV-файлом (Qwen3-TTS, фоновый поток).

    Публичное поведение совпадает с ``TtsSynth``: ``synthesize(text)``,
    ``shutdown()``, ``has_reply``/``last_reply_path``/``is_busy``/
    ``is_shutdown``, ``state``/``status``, сигналы
    ``state_changed``/``reply_ready``/``failed``. Разница — в механизме:
    не процесс ``say`` в цикле событий Qt, а модель в daemon-потоке.
    """

    state_changed = Signal()
    # Готовый голосовой ответ: путь к непустому WAV-файлу.
    reply_ready = Signal(str)
    # Ошибка синтеза; причина — человекочитаемая строка.
    failed = Signal(str)

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_VOICE_DESIGN,
        instruct: str = DEFAULT_INSTRUCT,
        ref_audio: str = "",
        ref_text: str = "",
        synthesizer: "callable | None" = None,
        parent: "QObject | None" = None,
    ) -> None:
        super().__init__(parent)
        self.model_id = model_id
        self.instruct = instruct
        # reference-режим включается парой ref_audio + ref_text.
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.mode = "reference" if (ref_audio and ref_text) else "voice_design"
        # Тестовая/альтернативная точка синтеза: (text) -> (audio, sample_rate).
        self._synthesizer = synthesizer

        self.state = self.IDLE
        self.status = "Озвучивание: готов"

        self._job_queue: "queue.Queue" = queue.Queue()
        self._job_seq = 0
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

        self._temp_dir: str | None = None
        # Файл текущего (ещё незавершённого) синтеза.
        self._current_path: str | None = None
        # Последний УСПЕШНЫЙ синтез — то, что можно повторять.
        self._ready_path: str | None = None
        # Ленивая модель (загружается в рабочем потоке один раз).
        self._model = None
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
        """Синтезирует текст в WAV-файл (фоновый поток, Qwen3-TTS).

        Возвращает ``True``, если задание принято (готовность придёт
        сигналом ``reply_ready``/``failed``). ``False`` — синтез отклонён
        (остановка, пустой текст); причина в ``status``. Новый вызов
        во время синтеза СМЕНЯЕТ предыдущий: результат старого задания
        отбрасывается. Метод не блокирует интерфейс ни на время
        загрузки модели, ни на время синтеза.
        """
        with self._lock:
            if self._shutdown:
                rejected = True
            else:
                text = (text or "").strip()
                if not text:
                    rejected = True
                else:
                    self._job_seq += 1
                    job = (self._job_seq, text)
                    rejected = False
        if rejected:
            self.state = self.IDLE
            if self._shutdown:
                self.status = "Озвучивание недоступно: компонент остановлен."
            else:
                self.status = "Озвучивать нечего: пустой текст."
            self.state_changed.emit()
            return False

        self.state = self.SYNTHESIZING
        self.status = "Озвучиваю ответ…"
        self.state_changed.emit()
        logger.info(
            "tts-qwen3: start mode=%s job=%d model=%s text_len=%d",
            self.mode, job[0], self.model_id, len(job[1]),
        )

        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop, name="tts-qwen3", daemon=True
            )
            self._worker.start()
        self._job_queue.put(job)
        return True

    def shutdown(self) -> None:
        """Останавливает синтез и удаляет СВОИ временные файлы. Идемпотентна.

        Порядок важен: сначала рабочий поток останавливается (он может
        писать файл), потом удаляется временный каталог. После вызова
        компонент не работает: ``synthesize()`` отклоняется, поздние
        результаты отбрасываются. Если поток занят загрузкой модели,
        ``join`` ограничен по времени, поток — daemon и завершится сам
        (его результат в любом случае отбрасывается).
        """
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

        self._job_queue.put(_SENTINEL)
        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)

        self._drop_file(self._current_path)
        self._current_path = None
        self._ready_path = None
        self._cleanup_temp_dir()

        self.state = self.IDLE
        self.status = "Озвучивание: остановлено"
        self.state_changed.emit()

    # --- Рабочий поток -----------------------------------------------------

    def _worker_loop(self) -> None:
        """Обрабатывает задания последовательно; исключений наружу не гонит."""
        while True:
            job = self._job_queue.get()
            if job is _SENTINEL:
                return
            try:
                self._process_job(job)
            except Exception:  # поток не умирает из-за одного задания
                logger.exception("tts-qwen3: ошибка рабочего потока")

    def _process_job(self, job: "tuple[int, str]") -> None:
        job_id, text = job
        with self._lock:
            current = self._job_seq
        if job_id != current:
            # Задание сменили: результат не публикуем.
            logger.info("tts-qwen3: job=%d отброшен (заменён)", job_id)
            return
        if self._shutdown:
            return
        try:
            audio, sample_rate = self._make_audio(text)
        except Exception as exc:
            logger.warning("tts-qwen3: job=%d сбой: %s", job_id, exc)
            self._finish_failed(job_id, str(exc))
            return
        self._finish_ready(job_id, audio, sample_rate)

    def _make_audio(self, text: str) -> "tuple[np.ndarray, int]":
        """Синтез одной реплики; (float-массив, частота дискретизации)."""
        if self._synthesizer is not None:
            result = self._synthesizer(text)
            audio, sample_rate = result
            return np.asarray(audio, dtype=np.float32), int(sample_rate)
        model = self._model
        if model is None:
            model = self._load_model()
            self._model = model
        sample_rate = int(model.sample_rate)
        if self.mode == "reference":
            results = model.generate(
                text=text,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                lang_code="english",
            )
        else:
            results = model.generate_voice_design(
                text=text, instruct=self.instruct, language="English"
            )
        chunks = [np.asarray(r.audio, dtype=np.float32).ravel() for r in results]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if not audio.size:
            raise RuntimeError("модель вернула пустое аудио")
        return audio, sample_rate

    def _load_model(self):
        """Ленивая загрузка (в рабочем потоке): mlx_audio импортируется здесь."""
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "не найден mlx-audio (установите группу [qwen3-tts]: "
                "pip install '.[qwen3-tts]')"
            ) from exc
        return load_model(self.model_id)

    # --- Завершение задания ------------------------------------------------

    def _finish_ready(
        self, job_id: int, audio: np.ndarray, sample_rate: int
    ) -> None:
        with self._lock:
            if job_id != self._job_seq or self._shutdown:
                return
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(
                    prefix="english-companion-tts-"
                )
            out_path = os.path.join(self._temp_dir, f"reply_{job_id}.wav")
        _write_wav(out_path, audio, sample_rate)

        with self._lock:
            if job_id != self._job_seq or self._shutdown:
                self._drop_file(out_path)
                return
            # Готовый ответ публикуем как «повторяемый».
            self._ready_path = out_path
        self.state = self.IDLE
        self.status = "Ответ озвучен."
        logger.info("tts-qwen3: ready job=%d out=%s", job_id, out_path)
        # Сигналы из рабочего потока — Qt доставляет их в главный поток.
        self.state_changed.emit()
        self.reply_ready.emit(out_path)

    def _finish_failed(self, job_id: int, reason: str) -> None:
        with self._lock:
            if job_id != self._job_seq:
                return
        self._drop_file(self._current_path)
        self._current_path = None
        self.state = self.IDLE
        self.status = f"Не удалось озвучить ответ: {reason}"
        logger.warning("tts-qwen3: failed job=%d reason=%r", job_id, reason)
        self.state_changed.emit()
        self.failed.emit(reason)

    # --- Внутренние ---------------------------------------------------------

    def _drop_file(self, path: str | None) -> None:
        """Удаляет частичный/устаревший файл, если он существует."""
        if path is None:
            return
        try:
            if os.path.lexists(path):
                os.unlink(path)
        except OSError as exc:
            logger.warning("tts-qwen3: не удалось удалить %s: %s", path, exc)

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is None:
            return
        try:
            shutil.rmtree(self._temp_dir)
        except OSError as exc:
            logger.warning(
                "tts-qwen3: не удалось удалить временную папку %s: %s",
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
