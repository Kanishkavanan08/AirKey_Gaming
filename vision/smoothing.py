import time
from typing import Dict, Tuple, List, Optional, Any


class ExponentialMovingAverage:
    """
    Applies an Exponential Moving Average (EMA) filter to a single scalar value.
    EMA reacts faster to sudden changes than a Simple Moving Average (SMA), 
    making it ideal for low-latency gaming inputs.
    """
    
    def __init__(self, alpha: float = 0.6) -> None:
        """
        Initializes the EMA filter.
        
        Args:
            alpha (float): The smoothing factor between 0.0 and 1.0. 
                           Higher = less smoothing/faster response. 
                           Lower = more smoothing/slower response.
        """
        self.alpha: float = alpha
        self._current_value: Optional[float] = None

    def update(self, value: float) -> float:
        """
        Updates the filter with a new raw data point.
        
        Args:
            value (float): The incoming raw value (e.g., an X or Y coordinate).
            
        Returns:
            float: The filtered, smoothed value.
        """
        if self._current_value is None:
            self._current_value = value
        else:
            self._current_value = (self.alpha * value) + ((1.0 - self.alpha) * self._current_value)
            
        return self._current_value

    def reset(self) -> None:
        """Flushes the filter's history."""
        self._current_value = None


class LandmarkSmoother:
    """
    Maintains independent EMA filters for all 21 MediaPipe hand landmarks (X and Y).
    This completely eliminates micro-jitter from webcam noise.
    """
    
    def __init__(self, alpha: float = 0.6) -> None:
        """
        Initializes the tracking filters for a single hand.
        
        Args:
            alpha (float): The EMA smoothing factor to apply to all spatial coordinates.
        """
        self.alpha: float = alpha
        # 21 landmarks * 2 coordinates (x, y) = 42 concurrent filters
        self.filters_x: Dict[int, ExponentialMovingAverage] = {
            i: ExponentialMovingAverage(alpha) for i in range(21)
        }
        self.filters_y: Dict[int, ExponentialMovingAverage] = {
            i: ExponentialMovingAverage(alpha) for i in range(21)
        }

    def update(self, landmarks: List[Any]) -> List[Tuple[float, float]]:
        """
        Processes a raw array of MediaPipe landmarks through the EMA filters.
        
        Args:
            landmarks (List[Any]): The raw normalized landmarks from MediaPipe.
            
        Returns:
            List[Tuple[float, float]]: A list of smoothed (X, Y) coordinate tuples.
        """
        smoothed_landmarks: List[Tuple[float, float]] = []
        
        for i, lm in enumerate(landmarks):
            # Guard against unexpected landmark list sizes
            if i > 20:
                break
                
            smooth_x: float = self.filters_x[i].update(lm.x)
            smooth_y: float = self.filters_y[i].update(lm.y)
            smoothed_landmarks.append((smooth_x, smooth_y))
            
        return smoothed_landmarks

    def reset(self) -> None:
        """
        Resets all coordinate filters. 
        Must be called when a hand leaves the frame to prevent interpolation artifacts 
        when the hand re-enters the frame at a different position.
        """
        for i in range(21):
            self.filters_x[i].reset()
            self.filters_y[i].reset()


class GestureDebouncer:
    """
    A temporal filter to prevent gesture flickering. 
    Requires a specific gesture to be held for a minimum time duration (debounce threshold) 
    before it is fully confirmed and dispatched to the game controller.
    """
    
    def __init__(self, debounce_time: float = 0.15) -> None:
        """
        Initializes the temporal debouncer.
        
        Args:
            debounce_time (float): Minimum seconds a gesture must be held to register.
        """
        self.debounce_time: float = debounce_time
        self._current_candidate: str = "UNKNOWN"
        self._candidate_start_time: float = 0.0
        self._confirmed_gesture: str = "UNKNOWN"

    def update(self, detected_gesture: str) -> str:
        """
        Evaluates the current raw gesture against the holding threshold.
        
        Args:
            detected_gesture (str): The raw classification from the current frame.
            
        Returns:
            str: The confirmed, debounced gesture.
        """
        current_time: float = time.perf_counter()

        if detected_gesture != self._current_candidate:
            # The gesture changed abruptly. Reset the timer lock.
            self._current_candidate = detected_gesture
            self._candidate_start_time = current_time
        else:
            # The gesture is being held. Check if the threshold is surpassed.
            elapsed: float = current_time - self._candidate_start_time
            if elapsed >= self.debounce_time:
                self._confirmed_gesture = self._current_candidate

        return self._confirmed_gesture