# services/dashboard_app/main.py
import sys
import threading

from PyQt6.QtWidgets import QApplication

from state_manager import StateManager
from ui import DashboardWindow
from grpc_server import serve_in_foreground


def _start_grpc_server(state_manager):
    # Blocking call; will run in a background thread
    serve_in_foreground(state_manager, port=50051)


def main():
    # Shared state between gRPC server and UI
    state_manager = StateManager()

    # Start gRPC server in a background thread
    server_thread = threading.Thread(
        target=_start_grpc_server,
        args=(state_manager,),
        daemon=True,
    )
    server_thread.start()

    # Start Qt application
    app = QApplication(sys.argv)
    window = DashboardWindow(state_manager)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()