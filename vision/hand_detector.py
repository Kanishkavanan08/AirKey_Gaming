import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from utils.logger import AppLogger
from utils.helpers import ConfigManager
from utils.constants import MediaPipeConstants
from vision.smoothing import LandmarkSmoother


class HandDetector:
    """
    A robust wrapper for the MediaPipe Hands neural network.
    Handles lifecycle management, tracks multiple hands, and applies 
    spatial coordinate smoothing to eliminate webcam tracking jitter.
    """

    def __init__(self) -> None:
        """
        Initializes the MediaPipe AI model using parameters from the global configuration.
        Allocates independent spatial smoothers for up to the maximum configured hands.
        """
        self._logger = AppLogger.get_logger()
        
        # Load constraints from config, utilizing fallback constants if necessary
        self.max_hands: int = ConfigManager.get("detection", "max_num_hands", MediaPipeConstants.MAX_NUM_HANDS)
        self.min_det_conf: float = ConfigManager.get("detection", "min_detection_confidence", MediaPipeConstants.MIN_DETECTION_CONFIDENCE)
        self.min_track_conf: float = ConfigManager.get("detection", "min_tracking_confidence", MediaPipeConstants.MIN_TRACKING_CONFIDENCE)
        self.model_complexity: int = ConfigManager.get("detection", "model_complexity", 1)
        self.smoothing_alpha: float = 0.6  # Can be moved to config if dynamic adjustment is desired

        # Initialize MediaPipe Hands Solution
        self._mp_hands = mp.solutions.hands
        self._hands_model = self._mp_hands.Hands(
            static_image_mode=MediaPipeConstants.STATIC_IMAGE_MODE,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_det_conf,
            min_tracking_confidence=self.min_track_conf
        )

        # Allocate an independent EMA smoother for each potential hand 
        # to prevent cross-contamination of spatial coordinates.
        self._smoothers: List[LandmarkSmoother] = [
            LandmarkSmoother(alpha=self.smoothing_alpha) for _ in range(self.max_hands)
        ]
        
        self._previous_hand_count: int = 0
        self._logger.info(f"HandDetector initialized (Max Hands: {self.max_hands}, Complexity: {self.model_complexity})")

    def process_frame(self, frame_rgb: np.ndarray) -> Any:
        """
        Pushes an RGB frame through the MediaPipe neural network.

        Args:
            frame_rgb (np.ndarray): The captured frame in RGB color space.

        Returns:
            Any: The raw MediaPipe results object containing landmarks and handedness.
        """
        return self._hands_model.process(frame_rgb)

    def extract_hand_data(self, results: Any) -> List[Dict[str, Any]]:
        """
        Parses raw MediaPipe results, applies EMA smoothing, and packages the data.
        
        Args:
            results (Any): The output object from self.process_frame().

        Returns:
            List[Dict[str, Any]]: A list of dictionaries. Each dictionary contains:
                - 'handedness': str ('Left' or 'Right')
                - 'score': float (Confidence score)
                - 'landmarks': List[Tuple[float, float]] (Smoothed 2D coordinates)
                - 'raw_landmarks': The original MediaPipe landmark object (for UI drawing)
        """
        extracted_hands: List[Dict[str, Any]] = []

        # If no hands are detected, reset all smoothers to prevent visual artifacts
        # when a hand re-enters the frame at a completely different location.
        if not results.multi_hand_landmarks:
            if self._previous_hand_count > 0:
                for smoother in self._smoothers:
                    smoother.reset()
                self._previous_hand_count = 0
            return extracted_hands

        current_hand_count: int = len(results.multi_hand_landmarks)

        # If the number of hands changes abruptly, reset smoothers to avoid interpolating 
        # between Hand A leaving and Hand B entering.
        if current_hand_count != self._previous_hand_count:
            for smoother in self._smoothers:
                smoother.reset()

        self._previous_hand_count = current_hand_count

        for index, (hand_landmarks, hand_info) in enumerate(zip(
            results.multi_hand_landmarks, 
            results.multi_handedness
        )):
            # Safety bound: do not attempt to process more hands than configured
            if index >= self.max_hands:
                break

            # MediaPipe's classification (Left/Right) is based on the camera's perspective.
            # Because we flipped the frame horizontally in camera.py (mirroring),
            # we must invert MediaPipe's handedness string to match reality.
            raw_label = hand_info.classification[0].label
            actual_handedness = "Right" if raw_label == "Left" else "Left"
            confidence_score = hand_info.classification[0].score

            # Apply temporal smoothing to the raw X, Y coordinates
            smoothed_coords: List[Tuple[float, float]] = self._smoothers[index].update(hand_landmarks.landmark)

            extracted_hands.append({
                "handedness": actual_handedness,
                "score": confidence_score,
                "landmarks": smoothed_coords,
                "raw_landmarks": hand_landmarks  # Retained solely for MediaPipe's drawing utils in the UI
            })

        return extracted_hands

    def release(self) -> None:
        """
        Closes the MediaPipe AI model and frees allocated GPU/CPU memory.
        Must be called during application teardown.
        """
        self._hands_model.close()
        self._logger.info("HandDetector neural network resources successfully released.")