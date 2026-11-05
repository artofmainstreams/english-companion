import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from companion.config import AppConfig
from companion.main_window import MainWindow
from companion.ollama_client import OllamaClient
from companion.audio_controller import AudioController
from companion.stt_backends import build_backend
from companion.stt_engine import SttEngine
from companion.session_controller import SessionController


def main() -> int:
    app = QApplication(sys.argv)
    config = AppConfig()
    client = OllamaClient(config, app)
    controller = SessionController(client, app)
    audio = AudioController(app)
    # Реализация STT выбирается при сборке приложения, а не в движке:
    # неизвестный stt_backend — понятная ошибка здесь, а не при первом
    # распознавании.
    stt = SttEngine(build_backend(config), app)
    window = MainWindow(controller, audio, stt)
    window.show()

    # Сеть-безопасность при выходе: STT сначала (ожидает завершение
    # рабочего потока и освобождает файл), потом AudioController (удаляет
    # временные файлы). Слоты окна (closeEvent) управляют основным порядком.
    app.aboutToQuit.connect(stt.shutdown)
    app.aboutToQuit.connect(audio.shutdown)

    QTimer.singleShot(0, controller.load_models)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
