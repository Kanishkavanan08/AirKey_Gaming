import cv2
import numpy as np
from typing import Tuple, Optional

from utils.logger import AppLogger
from utils.helpers import ConfigManager


class Camera:
    """
    Manages the physical webcam interface utilizing OpenCV.
    Handles hardware initialization, frame capture, color-space conversion, 
    and automatic exposure compensation for low-light environments.
    """

    def __init__(self) -> None:
        """
        Initializes the Camera manager, fetching hardware preferences from the configuration.
        """
        self._logger = AppLogger.get_logger()
        
        # Load hardware constraints from config
        self.device_index: int = ConfigManager.get("camera", "device_index", 0)
        self.api_preference: int = ConfigManager.get("camera", "api_preference", cv2.CAP_ANY)
        self.target_width: int = ConfigManager.get("camera", "target_width", 640)
        self.target_height: int = ConfigManager.get("camera", "target_height", 480)
        self.target_fps: int = ConfigManager.get("camera", "target_fps", 30)
        self.auto_exposure: bool = ConfigManager.get("camera", "auto_exposure", True)
        self.brightness_comp: bool = ConfigManager.get("camera", "brightness_compensation", True)
        
        self._capture: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False

    def start(self) -> bool:
        """
        Initializes the physical webcam hardware and applies target configuration properties.
        
        Returns:
            bool: True if the camera initialized successfully, False otherwise.
        """
        if self._capture is not None and self._capture.isOpened():
            self._logger.warning("Camera is already running.")
            return True

        self._logger.info(f"Attempting to connect to Camera Index {self.device_index}...")
        self._capture = cv2.VideoCapture(self.device_index, self.api_preference)

        if not self._capture.isOpened():
            self._logger.error(f"Failed to open Camera Index {self.device_index}. Check permissions or USB connection.")
            return False

        # Apply hardware properties
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if self.auto_exposure:
            # Note: Exposure property support varies heavily by webcam manufacturer
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        # Verify actual resolution applied by the hardware
        actual_w: int = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h: int = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps: int = int(self._capture.get(cv2.CAP_PROP_FPS))
        
        self._logger.info(f"Camera successfully initialized. Hardware locked at {actual_w}x{actual_h} @ {actual_fps} FPS.")
        self.is_running = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Grabs the latest frame from the webcam buffer, applies mirroring, 
        and prepares the necessary color spaces.

        Returns:
            Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]: 
                - Success flag (False if camera disconnected).
                - BGR Frame (for UI drawing/OpenCV).
                - RGB Frame (for MediaPipe Neural Network).
        """
        if not self.is_running or self._capture is None:
            return False, None, None

        ret, frame = self._capture.read()
        
        if not ret or frame is None:
            self._logger.warning("Camera dropped a frame or abruptly disconnected.")
            return False, None, None

        # Flip horizontally to act as a mirror. Essential for intuitive gesture control.
        frame_bgr = cv2.flip(frame, 1)

        # Optional: Algorithmic brightness compensation for low-light rooms
        if self.brightness_comp:
            frame_bgr = self._apply_brightness_compensation(frame_bgr)

        # MediaPipe requires strictly RGB color space
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Marking the image as not writeable passes it by reference to MediaPipe, saving CPU cycles
        frame_rgb.flags.writeable = False

        return True, frame_bgr, frame_rgb

    def _apply_brightness_compensation(self, frame: np.ndarray) -> np.ndarray:
        """
        Dynamically analyzes the frame's luminosity and boosts it if the room is too dark,
        preventing MediaPipe from losing tracking due to shadows.

        Args:
            frame (np.ndarray): The raw BGR frame.

        Returns:
            np.ndarray: The adjusted BGR frame.
        """
        # Calculate average pixel intensity across the entire image matrix
        average_brightness: float = np.mean(frame)
        
        # If the frame is generally dark (average pixel value < 80 on a 0-255 scale)
        if average_brightness < 80:
            # Alpha controls contrast (1.0 = no change), Beta controls brightness (addition)
            alpha: float = 1.2
            beta: int = 40
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
        return frame

    def release(self) -> None:
        """
        Safely flushes the hardware buffer and releases the webcam back to the Operating System.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            
        self.is_running = False
        self._logger.info("Camera hardware successfully released.")