import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from typing import Optional

from controller.game_controller import GameController
from vision.camera import Camera
from vision.hand_detector import HandDetector
from vision.gesture_recognition import GestureRecognizer
from vision.fps import FPSTracker
from ui.overlay import FrameOverlay


class DashboardApp:
    """
    The main desktop GUI application. Houses the video feed and controller interface.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GestureControl HUD")
        self.root.geometry("800x600")

        # Core Components
        self.camera = Camera()
        self.detector = HandDetector()
        self.recognizer = GestureRecognizer()
        self.overlay = FrameOverlay()
        self.controller = GameController()
        self.fps_tracker = FPSTracker()

        # UI Elements
        self.canvas = tk.Canvas(root, width=640, height=480, bg="black")
        self.canvas.pack(pady=20)
        
        self.btn_frame = ttk.Frame(root)
        self.btn_frame.pack()
        
        self.start_btn = ttk.Button(self.btn_frame, text="Start Controller", command=self._start)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(self.btn_frame, text="Stop Controller", command=self._stop)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        # Initialize Camera
        if self.camera.start():
            self._update_frame()
        else:
            print("Critical Error: Could not initialize camera.")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start(self) -> None:
        self.controller.toggle_active_state()

    def _stop(self) -> None:
        self.controller.toggle_active_state()

    def _update_frame(self) -> None:
        """The main loop processing the video pipeline."""
        self.fps_tracker.start_frame()
        
        success, frame_bgr, frame_rgb = self.camera.read()
        
        if success:
            # 1. Process AI
            results = self.detector.process_frame(frame_rgb)
            hands_data = self.detector.extract_hand_data(results)
            
            gesture = "NONE"
            if hands_data:
                # Use data from the first detected hand
                gesture = self.recognizer.classify(hands_data[0]["landmarks"], hands_data[0]["handedness"])
            
            # 2. Control System
            self.controller.process_gesture(gesture)
            
            # 3. Draw Overlay
            frame_bgr = self.overlay.draw_tracking_data(frame_bgr, hands_data)
            frame_bgr = self.overlay.draw_hud(
                frame_bgr, self.fps_tracker.fps, self.fps_tracker.latency_ms,
                gesture, self.controller.get_current_action(), self.controller.is_active
            )
            
            # 4. Display in Tkinter
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
            
            self.fps_tracker.end_frame()
            self.fps_tracker.update()

        self.root.after(10, self._update_frame)

    def _on_close(self) -> None:
        self.controller.stop()
        self.camera.release()
        self.root.destroy()