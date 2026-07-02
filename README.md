# [Project Name]

A robust, low-latency gesture-to-keyboard controller built with Python, MediaPipe, and OpenCV. This system allows you to control PC games and applications using natural hand gestures.

## Features
* **High Precision:** Uses EMA smoothing to eliminate webcam jitter.
* **Low Latency:** Asynchronous processing pipeline for responsive controls.
* **State-Aware:** Tracks "hold" vs "tap" states to prevent input spam.
* **Visual HUD:** Built-in telemetry dashboard (FPS, latency, active action).
* **Voice Feedback:** Optional non-blocking TTS for system notifications.

## Tech Stack
* **Vision:** OpenCV, MediaPipe
* **Controls:** Pynput, PyAutoGUI
* **UI:** Tkinter, CustomTkinter
* **Threading:** Multi-threaded architecture for voice/vision synchronization

## Getting Started
1. Clone the repository: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your bindings in `config/config.json`.
4. Run the application: `python main.py`

## License
[Choose a license, e.g., MIT]