from enum import Enum


class GestureType(Enum):
    """
    Strict enumeration of all detectable hand gestures within the system.
    Matches the naming conventions expected by the configuration mapper.
    """
    OPEN_PALM = "OPEN_PALM"
    CLOSED_FIST = "CLOSED_FIST"
    THUMB_UP = "THUMB_UP"
    PEACE_SIGN = "PEACE_SIGN"
    POINT_RIGHT = "POINT_RIGHT"
    POINT_LEFT = "POINT_LEFT"
    OK_SIGN = "OK_SIGN"
    ROCK_SIGN = "ROCK_SIGN"
    TWO_HANDS_OPEN = "TWO_HANDS_OPEN"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class ActionType(Enum):
    """
    Enumeration of all possible game actions to be triggered.
    These act as the intermediary layer between the gesture and the actual keyboard key.
    """
    ACCELERATE = "Accelerate"
    BRAKE = "Brake"
    NITRO_BOOST = "Nitro Boost"
    PAUSE = "Pause"
    STEER_RIGHT = "Steer Right"
    STEER_LEFT = "Steer Left"
    HORN = "Horn"
    RESTART = "Restart"
    EXIT = "Exit Safety Loop"
    IDLE = "Idle"


class KeyMode(Enum):
    """
    Defines how a keyboard key should be simulated via PyAutoGUI/Pynput.
    """
    HOLD = "hold"
    PRESS = "press"
    RELEASE = "release"


class MediaPipeConstants:
    """
    Fallback constants for MediaPipe Hand Tracking parameters.
    Used if config.json fails to load.
    """
    STATIC_IMAGE_MODE: bool = False
    MAX_NUM_HANDS: int = 2
    MIN_DETECTION_CONFIDENCE: float = 0.75
    MIN_TRACKING_CONFIDENCE: float = 0.75


class UIConstants:
    """
    Typography and fixed UI dimensions for the Dashboard and Overlay.
    """
    DEFAULT_FONT: tuple[str, int] = ("Helvetica", 12)
    HEADER_FONT: tuple[str, int, str] = ("Helvetica", 14, "bold")
    TITLE_FONT: tuple[str, int, str] = ("Helvetica", 20, "bold")
    
    # Fallback Colors (BGR format for OpenCV compatibility)
    COLOR_BOUNDING_BOX: tuple[int, int, int] = (0, 255, 0)
    COLOR_LANDMARK: tuple[int, int, int] = (0, 0, 255)
    COLOR_CONNECTION: tuple[int, int, int] = (220, 220, 220)
    COLOR_TEXT: tuple[int, int, int] = (255, 255, 255)