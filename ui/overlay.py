import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Tuple

from utils.constants import UIConstants
from utils.helpers import ConfigManager


class FrameOverlay:
    """
    Handles the rendering of the Heads Up Display (HUD) and 
    computer vision visualizers onto the live webcam feed.
    Optimized to minimize CPU overhead and prevent frame drops.
    """

    def __init__(self) -> None:
        """
        Initializes the overlay drawer, fetching typography and color themes 
        from the global configuration.
        """
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_hands = mp.solutions.hands

        # Load UI Preferences
        ui_config = ConfigManager.get("ui", default={})
        self.show_overlay: bool = ui_config.get("show_overlay", True)
        self.show_landmarks: bool = ui_config.get("show_landmarks", True)
        
        # Colors (BGR format for OpenCV)
        colors = ui_config.get("colors", {})
        self.c_bbox = tuple(colors.get("bounding_box", UIConstants.COLOR_BOUNDING_BOX))
        self.c_land = tuple(colors.get("landmark_points", UIConstants.COLOR_LANDMARK))
        self.c_conn = tuple(colors.get("landmark_connections", UIConstants.COLOR_CONNECTION))
        self.c_text = tuple(colors.get("text_primary", UIConstants.COLOR_TEXT))
        self.c_active = tuple(colors.get("status_active", (0, 255, 0)))
        self.c_inactive = tuple(colors.get("status_inactive", (0, 0, 255)))

        # Fonts
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_tracking_data(self, frame: np.ndarray, hands_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws the MediaPipe landmarks, joint connections, and bounding boxes for all detected hands.

        Args:
            frame (np.ndarray): The raw BGR frame from the webcam.
            hands_data (List[Dict[str, Any]]): The extracted hand data from HandDetector.

        Returns:
            np.ndarray: The modified frame with visualizers drawn.
        """
        if not self.show_landmarks or not hands_data:
            return frame

        h, w, _ = frame.shape

        for hand in hands_data:
            raw_landmarks = hand.get("raw_landmarks")
            handedness = hand.get("handedness", "Unknown")
            score = hand.get("score", 0.0)

            # 1. Draw Bounding Box and Hand Label
            if raw_landmarks:
                x_coords = [lm.x * w for lm in raw_landmarks.landmark]
                y_coords = [lm.y * h for lm in raw_landmarks.landmark]
                
                # Add padding to the bounding box
                padding = 20
                x_min, x_max = int(min(x_coords)) - padding, int(max(x_coords)) + padding
                y_min, y_max = int(min(y_coords)) - padding, int(max(y_coords)) + padding
                
                # Constrain coordinates to frame dimensions
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Draw the rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.c_bbox, 2)
                
                # Draw the Handedness and Confidence Score label
                label = f"{handedness} [{score:.2f}]"
                cv2.putText(
                    frame, label, (x_min, max(20, y_min - 10)), 
                    self.font, 0.5, self.c_bbox, 1, cv2.LINE_AA
                )

                # 2. Draw Landmark Points and Skeletal Connections
                # We use custom drawing specs to apply our configured theme colors
                landmark_spec = self._mp_drawing.DrawingSpec(color=self.c_land, thickness=2, circle_radius=2)
                connection_spec = self._mp_drawing.DrawingSpec(color=self.c_conn, thickness=2)
                
                self._mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=raw_landmarks,
                    connections=self._mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )

        return frame

    def draw_hud(
        self, 
        frame: np.ndarray, 
        fps: float, 
        latency: float, 
        current_gesture: str, 
        current_action: str, 
        is_active: bool
    ) -> np.ndarray:
        """
        Draws the main telemetry dashboard (HUD) on the top-left of the screen.

        Args:
            frame (np.ndarray): The BGR frame.
            fps (float): Current frames per second.
            latency (float): Current tracking latency in milliseconds.
            current_gesture (str): The actively detected gesture string.
            current_action (str): The actively executed game action.
            is_active (bool): Whether the controller is active or paused.

        Returns:
            np.ndarray: The modified frame with the HUD drawn.
        """
        if not self.show_overlay:
            return frame

        # Create a semi-transparent dark background block for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        # Apply alpha blending (0.6 opacity)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw State Indicator
        state_text = "SYSTEM ACTIVE" if is_active else "SYSTEM PAUSED"
        state_color = self.c_active if is_active else self.c_inactive
        cv2.putText(frame, state_text, (20, 35), self.font, 0.7, state_color, 2, cv2.LINE_AA)

        # Draw Telemetry Lines
        metrics = [
            f"FPS: {fps:.1f} | Latency: {latency:.1f}ms",
            f"Gesture: {current_gesture}",
            f"Action: {current_action}"
        ]

        y_offset = 70
        for text in metrics:
            cv2.putText(frame, text, (20, y_offset), self.font, 0.6, self.c_text, 1, cv2.LINE_AA)
            y_offset += 35

        # Draw hotkey reminder at the bottom of the HUD block
        cv2.putText(
            frame, "[F1] Start  [F2] Stop  [F3] HUD", 
            (20, y_offset), self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA
        )

        return frame