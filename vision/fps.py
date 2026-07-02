import time
from collections import deque
from typing import Optional


class FPSTracker:
    """
    High-precision performance metrics tracker for the computer vision pipeline.
    Computes a rolling-average Frames Per Second (FPS) and tracking latency in milliseconds.
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Initializes the performance tracker with a rolling window.

        Args:
            window_size (int): The number of historical frames to average over for stable metrics.
        """
        self._window_size: int = window_size
        self._frame_intervals: deque[float] = deque(maxlen=window_size)
        
        # Performance timestamps
        self._last_update_time: float = time.perf_counter()
        self._frame_start_time: Optional[float] = None
        self._current_latency_ms: float = 0.0

    def start_frame(self) -> None:
        """
        Marks the exact entry point of a complete frame execution loop.
        Call this immediately before fetching the latest video frame from the camera.
        """
        self._frame_start_time = time.perf_counter()

    def end_frame(self) -> None:
        """
        Marks the resolution point of a complete frame processing loop.
        Calculates the pure execution latency elapsed since start_frame() was run.
        """
        if self._frame_start_time is not None:
            end_time: float = time.perf_counter()
            self._current_latency_ms = (end_time - self._frame_start_time) * 1000.0
            self._frame_start_time = None

    def update(self) -> None:
        """
        Logs a completed processing iteration to advance the FPS calculation.
        Computes elapsed delta-time since the previous call to smooth visual tracking data.
        """
        current_time: float = time.perf_counter()
        delta_time: float = current_time - self._last_update_time
        
        # Avoid dividing by zero if tracking is called instantaneously or faster than system clock precision
        if delta_time > 0:
            self._frame_intervals.append(delta_time)
            
        self._last_update_time = current_time

    @property
    def fps(self) -> float:
        """
        Calculates the current rolling average Frames Per Second.

        Returns:
            float: The smoothed frame rate. Returns 0.0 if data is insufficient.
        """
        if not self._frame_intervals:
            return 0.0
        
        total_time: float = sum(self._frame_intervals)
        if total_time <= 0:
            return 0.0
            
        return len(self._frame_intervals) / total_time

    @property
    def latency_ms(self) -> float:
        """
        Retrieves the exact compute latency of the most recently processed frame sequence.

        Returns:
            float: Code execution time in milliseconds.
        """
        return self._current_latency_ms