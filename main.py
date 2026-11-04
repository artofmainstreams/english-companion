import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from companion.config import AppConfig
from companion.main_window import MainWindow
from companion.ollama_client import OllamaClient
from companion.audio_controller import AudioController
from companion.session_controller import SessionController


def main() -> int:
    app = QApplication(sys.argv)
    client = OllamaClient(AppConfig(), app)
    controller = SessionController(client, app)
    audio = AudioController(app)
    window = MainWindow(controller, audio)
    window.show()
    app.aboutToQuit.connect(audio.shutdown)
    QTimer.singleShot(0, controller.load_models)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
