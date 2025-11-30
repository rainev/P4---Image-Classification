# services/dashboard_app/state_manager.py
import threading


class StateManager:
    """
    Thread-safe shared state between the gRPC server (producer)
    and the Qt UI (consumer).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_batch = None
        self._loss_history = []   # list of (step, loss)
        self._last_step = None
        self._fps = 0.0
        self._queue_size = 0      # simple indicator (we overwrite latest)

    def update_from_batch(self, batch):
        """
        Called by the gRPC server thread whenever a new TrainingBatch arrives.
        """
        with self._lock:
            self._latest_batch = batch
            # We only keep the latest batch, so queue_size is effectively 1.
            self._queue_size = 1

            # Track loss history per step
            if batch.step != self._last_step:
                self._loss_history.append((batch.step, batch.loss))
                self._last_step = batch.step

    def get_latest_for_ui(self):
        """
        Called by the UI thread every frame to read the latest data.
        """
        with self._lock:
            batch = self._latest_batch
            loss_history_copy = list(self._loss_history)
            queue_size = self._queue_size
        return batch, loss_history_copy, queue_size

    def set_fps(self, fps: float):
        with self._lock:
            self._fps = fps

    def get_status(self):
        """
        Returns (fps, queue_size) for Ping() RPC.
        """
        with self._lock:
            return self._fps, self._queue_size