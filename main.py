import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from companion.config import AppConfig
from companion.main_window import MainWindow
from companion.ollama_client import OllamaClient
from companion.session_controller import SessionController


def main() -> int:
    app = QApplication(sys.argv)
    client = OllamaClient(AppConfig(), app)
    controller = SessionController(client, app)
    window = MainWindow(controller)
    window.show()
    QTimer.singleShot(0, controller.load_models)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
