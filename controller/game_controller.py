import threading
import queue
from typing import Optional

from utils.logger import AppLogger
from utils.helpers import ConfigManager
from controller.gesture_mapper import GestureMapper

# Optional pyttsx3 import for voice feedback
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class GameController:
    """
    The master facade for the game control subsystem.
    Routes confirmed gestures to the GestureMapper, manages global active/paused state,
    and handles asynchronous text-to-speech voice feedback.
    """

    def __init__(self) -> None:
        """
        Initializes the game controller, loads voice configurations, 
        and sets up the isolated text-to-speech worker thread.
        """
        self._logger = AppLogger.get_logger()
        self._mapper = GestureMapper()
        
        # State Management
        self.is_active: bool = False
        self._previous_action: str = "Idle"

        # Voice Feedback Configuration
        voice_config = ConfigManager.get("voice_feedback", default={})
        self._voice_enabled: bool = voice_config.get("enabled", False) and TTS_AVAILABLE
        self._speech_rate: int = voice_config.get("speech_rate", 150)
        self._volume: float = voice_config.get("volume", 0.8)

        # Threading for non-blocking TTS
        self._voice_queue: queue.Queue = queue.Queue()
        self._voice_thread: Optional[threading.Thread] = None
        self._stop_voice_thread: threading.Event = threading.Event()
        
        if self._voice_enabled:
            self._start_voice_worker()
        elif voice_config.get("enabled", False) and not TTS_AVAILABLE:
            self._logger.warning("Voice feedback enabled in config, but pyttsx3 is not installed.")

    def _start_voice_worker(self) -> None:
        """
        Spins up a background daemon thread to process text-to-speech requests.
        Initializing pyttsx3 inside the thread prevents COM/OS context errors.
        """
        self._stop_voice_thread.clear()
        self._voice_thread = threading.Thread(
            target=self._voice_worker_loop, 
            name="VoiceFeedbackThread", 
            daemon=True
        )
        self._voice_thread.start()
        self._logger.info("Asynchronous voice feedback worker started.")

    def _voice_worker_loop(self) -> None:
        """
        The continuous loop executed by the voice worker thread.
        Dequeues text strings and synthesizes speech.
        """
        try:
            # Initialize engine inside the thread for OS compatibility
            engine = pyttsx3.init()
            engine.setProperty('rate', self._speech_rate)
            engine.setProperty('volume', self._volume)
            
            while not self._stop_voice_thread.is_set():
                try:
                    # Block for 0.5 seconds, then loop back to check the stop event
                    text_to_speak: str = self._voice_queue.get(timeout=0.5)
                    engine.say(text_to_speak)
                    engine.runAndWait()
                    self._voice_queue.task_done()
                except queue.Empty:
                    continue
        except Exception as e:
            self._logger.error(f"Voice engine encountered a fatal error: {e}")

    def trigger_voice(self, text: str) -> None:
        """
        Safely queues a text string to be spoken by the background worker.

        Args:
            text (str): The phrase to speak.
        """
        if self._voice_enabled and not self._stop_voice_thread.is_set():
            # Clear queue to prevent a backlog of delayed speech if actions change rapidly
            with self._voice_queue.mutex:
                self._voice_queue.queue.clear()
            self._voice_queue.put(text)

    def process_gesture(self, gesture: str) -> None:
        """
        The main public entry point for routing vision data.
        Evaluates the gesture if the controller is active, and triggers voice feedback 
        on state changes.

        Args:
            gesture (str): The confirmed gesture string from the Vision AI.
        """
        if not self.is_active:
            # If paused, ensure no leftover keys are held, but don't map new gestures
            self._mapper.release_all()
            self._previous_action = "Idle"
            return

        # Route the gesture to the mapper
        self._mapper.process_gesture(gesture)

        # Evaluate if the resulting action changed to trigger voice feedback
        current_action: str = self._mapper.get_current_action_name()
        if current_action != self._previous_action:
            if current_action != "Idle":
                self.trigger_voice(f"{current_action}")
            self._previous_action = current_action

    def get_current_action(self) -> str:
        """
        Fetches the active game action being executed.

        Returns:
            str: The action name (e.g., 'Accelerate', 'Idle').
        """
        if not self.is_active:
            return "System Paused"
        return self._mapper.get_current_action_name()

    def toggle_active_state(self) -> bool:
        """
        Toggles the master active/paused state of the controller.
        Releases all held keys when pausing.

        Returns:
            bool: The new active state.
        """
        self.is_active = not self.is_active
        if self.is_active:
            self._logger.info("Game Controller is now ACTIVE. Listening for gestures.")
            self.trigger_voice("Controller Activated")
        else:
            self._logger.info("Game Controller is now PAUSED. Gestures ignored.")
            self.trigger_voice("Controller Paused")
            self._mapper.release_all()
            self._previous_action = "Idle"
            
        return self.is_active

    def stop(self) -> None:
        """
        Safely tears down the controller, releasing all OS keyboard hooks 
        and gracefully terminating the voice thread.
        """
        self.is_active = False
        self._mapper.release_all()
        
        if self._voice_enabled:
            self._stop_voice_thread.set()
            if self._voice_thread and self._voice_thread.is_alive():
                self._voice_thread.join(timeout=1.0)
                
        self._logger.info("Game Controller safely stopped and dismantled.")