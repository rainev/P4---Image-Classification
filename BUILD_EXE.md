## Building Standalone Executables

The project now ships with PyInstaller spec files for both parts of the
system: the dashboard (UI + embedded gRPC server) and the training client
that streams data to it.

### 1. Prerequisites

1. Use the platform that you want the executable to target (Windows builds
   must be produced on Windows; PyInstaller cannot cross-compile).
2. Install Python 3.10+ and create/activate a virtual environment.
3. Install the runtime dependencies plus PyInstaller:
   ```bash
   pip install -r requirements.txt pyinstaller
   ```

### 2. Dashboard build (UI + server)

```bash
pyinstaller --clean --noconfirm pyinstaller-dashboard.spec
```

Result: `dist/image_dashboard/image_dashboard.exe`

- This binary launches the PyQt6 dashboard and starts the gRPC server in a
  background thread.
- PyQtGraph resources are bundled automatically via the spec file.

### 3. Training client build

```bash
pyinstaller --clean --noconfirm pyinstaller-trainer.spec
```

Result: `dist/trainer/trainer.exe`

- Expect a large output (hundreds of MB) because PyTorch, torchvision, and
  torchaudio are statically collected.
- The executable downloads MNIST into a `data/` folder placed next to the
  executable if it is missing, so make sure the process has write access.

### 4. Running the pair

1. Launch `image_dashboard.exe`. The UI window confirms the server is
   accepting gRPC connections on port 50051.
2. In a separate terminal, launch `trainer.exe`. It connects to
   `localhost:50051`, starts the CNN training loop, and streams batches to
   the dashboard until it reaches the configured step count.

### 5. Notes and troubleshooting

- If you change Python source files, rebuild the executables to pick up the
  updates.
- Antivirus tools sometimes sandbox PyInstaller outputs; unblock the `.exe`
  files if Windows warns you.
- To inspect console logs from the dashboard build, run it from a terminal
  (`image_dashboard.exe`) so stdout/stderr stay visible.
- When distributing the executables, include the `data/` folder if you want
  to ship a pre-downloaded MNIST dataset; otherwise it will download on the
  first run.
