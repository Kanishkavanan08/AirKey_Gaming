from typing import List, Tuple

from utils.logger import AppLogger
from utils.constants import GestureType
from utils.helpers import MathUtils


class GestureRecognizer:
    """
    Analyzes mathematically smoothed 2D hand landmarks to classify static gestures.
    Utilizes radial distance from the wrist and vector comparisons to remain 
    robust against hand rotation and varying distances from the webcam.
    """

    def __init__(self) -> None:
        """
        Initializes the gesture classification engine.
        """
        self._logger = AppLogger.get_logger()
        
        # Standard MediaPipe Joint Indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.THUMB_MCP = 2
        
        # Finger Tip and PIP (Proximal Interphalangeal) joint indices
        self.FINGERS = {
            "INDEX": {"TIP": 8, "PIP": 6, "MCP": 5},
            "MIDDLE": {"TIP": 12, "PIP": 10, "MCP": 9},
            "RING": {"TIP": 16, "PIP": 14, "MCP": 13},
            "PINKY": {"TIP": 20, "PIP": 18, "MCP": 17}
        }

    def classify(self, landmarks: List[Tuple[float, float]], handedness: str) -> str:
        """
        Evaluates the spatial state of all five fingers and maps them to a defined GestureType.

        Args:
            landmarks (List[Tuple[float, float]]): The 21 smoothed (X, Y) coordinates.
            handedness (str): "Left" or "Right" hand (essential for thumb evaluation).

        Returns:
            str: The string representation of the identified GestureType.
        """
        if not landmarks or len(landmarks) < 21:
            return GestureType.UNKNOWN.value

        # 1. Determine the boolean state (Open/Closed) of every finger
        finger_states = self._get_finger_states(landmarks, handedness)
        thumb_open, index_open, middle_open, ring_open, pinky_open = finger_states

        # 2. Heuristic checks for specific, complex gestures

        # OK Sign: Thumb and Index tips are touching; other three fingers are fully extended.
        if middle_open and ring_open and pinky_open:
            pinch_distance = MathUtils.calculate_distance(landmarks[self.THUMB_TIP], landmarks[self.FINGERS["INDEX"]["TIP"]])
            # Since coordinates are normalized (0.0 to 1.0), 0.05 is roughly 5% of the frame width
            if pinch_distance < 0.05:
                return GestureType.OK_SIGN.value

        # 3. Standard Boolean Gestures
        if thumb_open and index_open and middle_open and ring_open and pinky_open:
            return GestureType.OPEN_PALM.value

        if not thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
            return GestureType.CLOSED_FIST.value

        if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
            # Ensure thumb is actually pointing UP (Y-axis tip is noticeably higher than the MCP)
            if landmarks[self.THUMB_TIP][1] < landmarks[self.THUMB_MCP][1]:
                return GestureType.THUMB_UP.value

        if not thumb_open and index_open and middle_open and not ring_open and not pinky_open:
            return GestureType.PEACE_SIGN.value

        if not thumb_open and index_open and not middle_open and not ring_open and pinky_open:
            return GestureType.ROCK_SIGN.value

        # 4. Vector-based Directional Pointing (Index open, others closed)
        if index_open and not middle_open and not ring_open and not pinky_open:
            index_tip = landmarks[self.FINGERS["INDEX"]["TIP"]]
            index_mcp = landmarks[self.FINGERS["INDEX"]["MCP"]]
            
            # Calculate the horizontal and vertical deltas
            delta_x = index_tip[0] - index_mcp[0]
            delta_y = index_tip[1] - index_mcp[1]
            
            # If the horizontal difference is significantly greater than the vertical difference,
            # the user is pointing laterally, not vertically.
            if abs(delta_x) > abs(delta_y) * 1.5:
                # X coordinates go from 0 (Left) to 1 (Right)
                if delta_x > 0:
                    return GestureType.POINT_RIGHT.value
                else:
                    return GestureType.POINT_LEFT.value

        return GestureType.NONE.value

    def _get_finger_states(self, landmarks: List[Tuple[float, float]], handedness: str) -> List[bool]:
        """
        Computes whether each individual finger is open (extended) or closed (curled).

        Args:
            landmarks (List[Tuple[float, float]]): The 21 hand landmarks.
            handedness (str): "Left" or "Right".

        Returns:
            List[bool]: A list of 5 booleans representing [Thumb, Index, Middle, Ring, Pinky].
                        True = Extended, False = Curled.
        """
        states: List[bool] = []
        wrist = landmarks[self.WRIST]

        # 1. Evaluate Thumb (Relies heavily on X-axis and handedness rather than Y-axis bending)
        # We compare the X-coordinate of the Thumb Tip to the Thumb IP joint.
        thumb_tip_x = landmarks[self.THUMB_TIP][0]
        thumb_ip_x = landmarks[self.THUMB_IP][0]
        
        # Because we applied a horizontal flip (mirror) to the camera feed:
        # A Right hand's thumb is extended to the LEFT of the screen (smaller X value).
        if handedness == "Right":
            states.append(thumb_tip_x < thumb_ip_x)
        else:
            states.append(thumb_tip_x > thumb_ip_x)

        # 2. Evaluate the remaining four fingers (Index, Middle, Ring, Pinky)
        # Using radial distance from the wrist is much more rotation-resistant than checking Y-coordinates.
        for finger_name in ["INDEX", "MIDDLE", "RING", "PINKY"]:
            tip_idx = self.FINGERS[finger_name]["TIP"]
            pip_idx = self.FINGERS[finger_name]["PIP"]
            
            dist_to_tip = MathUtils.calculate_distance(wrist, landmarks[tip_idx])
            dist_to_pip = MathUtils.calculate_distance(wrist, landmarks[pip_idx])
            
            # If the distance from the wrist to the finger TIP is greater than the distance 
            # to the knuckle (PIP), the finger is extended outward.
            is_extended = dist_to_tip > dist_to_pip
            states.append(is_extended)

        return states