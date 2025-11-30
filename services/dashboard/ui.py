# services/dashboard_app/ui.py
import time

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QMainWindow,
)
import pyqtgraph as pg


class DashboardWindow(QMainWindow):
    def __init__(self, state_manager):
        super().__init__()
        self.state_manager = state_manager

        self.setWindowTitle("Image Classifier Dashboard")
        self.resize(1600, 900)

        # For FPS calculation
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0

        # Track last rendered step to avoid redundant redraws
        self._last_step_rendered = None

        # Build UI layout
        self._build_ui()

        # Timer for ~60 FPS render loop (16 ms)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(16)

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # 4x4 image grid
        self.image_labels = []
        image_grid_widget = QWidget()
        image_grid_layout = QGridLayout()
        for r in range(4):
            for c in range(4):
                lbl = QLabel()
                lbl.setFixedSize(150, 150)
                lbl.setStyleSheet(
                    "background-color: #222; border: 1px solid #555; color: #888;"
                )
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setText("No\nImage")
                image_grid_layout.addWidget(lbl, r, c)
                self.image_labels.append(lbl)
        image_grid_widget.setLayout(image_grid_layout)

        # 4x4 prediction/ground-truth label grid
        self.text_labels = []
        text_grid_widget = QWidget()
        text_grid_layout = QGridLayout()
        for r in range(4):
            for c in range(4):
                lbl = QLabel()
                lbl.setFixedSize(150, 60)
                lbl.setStyleSheet(
                    "background-color: #222; border: 1px solid #555; color: white;"
                )
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setText("— / —")
                text_grid_layout.addWidget(lbl, r, c)
                self.text_labels.append(lbl)
        text_grid_widget.setLayout(text_grid_layout)

        top_layout.addWidget(image_grid_widget)
        top_layout.addWidget(text_grid_widget)

        # Loss plot using pyqtgraph
        self.loss_plot_widget = pg.PlotWidget()
        self.loss_plot_widget.setBackground("k")
        self.loss_plot_widget.setLabel("left", "Loss")
        self.loss_plot_widget.setLabel("bottom", "Step")
        self.loss_curve = self.loss_plot_widget.plot([], [], pen="y")

        # FPS + latency labels
        status_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: white;")
        self.latency_label = QLabel("Latency: — ms")
        self.latency_label.setStyleSheet("color: white;")

        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.latency_label)
        status_widget = QWidget()
        status_widget.setLayout(status_layout)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.loss_plot_widget)
        main_layout.addWidget(status_widget)

        central.setStyleSheet("background-color: #111; color: white;")
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _on_frame(self):
        # --- FPS calculation ---
        now = time.perf_counter()
        dt = now - self._last_frame_time
        if dt > 0:
            # Exponential moving average for smoother FPS
            instant_fps = 1.0 / dt
            self._fps = (self._fps * 0.9) + (0.1 * instant_fps)
        self._last_frame_time = now
        self.fps_label.setText(f"FPS: {self._fps:5.1f}")

        # Update shared FPS so Ping() can report it
        self.state_manager.set_fps(self._fps)

        # --- Get latest batch from state manager ---
        batch, loss_history, queue_size = self.state_manager.get_latest_for_ui()
        if batch is None:
            return

        # Only re-render if new step arrived
        if batch.step == self._last_step_rendered:
            return
        self._last_step_rendered = batch.step

        # --- Update images & labels (4x4 tiles, i.e., 16 maximum) ---
        num_samples = min(16, len(batch.images))
        for i in range(16):
            if i < num_samples:
                img_bytes = batch.images[i]

                pred = (
                    batch.pred_labels[i]
                    if i < len(batch.pred_labels)
                    else "?"
                )
                true = (
                    batch.true_labels[i]
                    if i < len(batch.true_labels)
                    else "?"
                )

                # Decode bytes to pixmap
                pixmap = self._bytes_to_pixmap(img_bytes)
                if pixmap is not None:
                    self.image_labels[i].setPixmap(
                        pixmap.scaled(
                            self.image_labels[i].width(),
                            self.image_labels[i].height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    self.image_labels[i].setText("")  # clear placeholder text
                else:
                    self.image_labels[i].setPixmap(QPixmap())
                    self.image_labels[i].setText("Bad\nImage")

                # Color-code pred/true label
                correct = (pred == true)
                color = "#2ecc71" if correct else "#e74c3c"  # green/red
                self.text_labels[i].setStyleSheet(
                    f"background-color: #222; border: 1px solid #555; color: {color};"
                )
                self.text_labels[i].setText(f"{pred} / {true}")
            else:
                # Empty tiles
                self.image_labels[i].setPixmap(QPixmap())
                self.image_labels[i].setText("No\nImage")
                self.text_labels[i].setStyleSheet(
                    "background-color: #222; border: 1px solid #555; color: white;"
                )
                self.text_labels[i].setText("— / —")

        # --- Update loss plot ---
        if loss_history:
            steps, losses = zip(*loss_history)
            self.loss_curve.setData(steps, losses)

        # --- Latency (dashboard now - batch.sent_timestamp_ms) ---
        now_ms = int(time.time() * 1000)
        latency_ms = now_ms - batch.sent_timestamp_ms
        self.latency_label.setText(f"Latency: {latency_ms} ms")

    def _bytes_to_pixmap(self, img_bytes):
        if not img_bytes:
            return None
        qimg = QImage.fromData(img_bytes)
        if qimg.isNull():
            return None
        return QPixmap.fromImage(qimg)