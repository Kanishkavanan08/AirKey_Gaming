import time
from typing import Dict, Any, List

from utils.logger import AppLogger
from utils.helpers import ConfigManager
from utils.constants import GestureType
from controller.keyboard_controller import KeyboardController


class GestureMapper:
    """
    Translates confirmed AI hand gestures into executed game actions.
    Manages input state transitions, hold vs. press logic, and cooldown timers 
    to prevent accidental action spamming.
    """

    def __init__(self) -> None:
        """
        Initializes the mapper, loads key bindings from configuration, 
        and instantiates the low-level OS keyboard controller.
        """
        self._logger = AppLogger.get_logger()
        self._keyboard = KeyboardController()
        
        # Load mappings and threshold configurations
        self._bindings: Dict[str, Dict[str, Any]] = ConfigManager.get("key_bindings", default={})
        self._cooldown_duration: float = ConfigManager.get(
            "filtering", "cooldown_duration_seconds", 0.3
        )
        
        # State tracking
        self._current_gesture: str = GestureType.NONE.value
        self._last_action_timestamps: Dict[str, float] = {}

    def process_gesture(self, new_gesture: str) -> None:
        """
        Evaluates a newly confirmed gesture, releases previous held keys if necessary, 
        and triggers the newly bound OS keys based on their configured mode.

        Args:
            new_gesture (str): The confirmed gesture string from the Vision pipeline.
        """
        # If the gesture hasn't changed, maintain current state (holds remain held)
        if new_gesture == self._current_gesture:
            return

        self._logger.info(f"Gesture Transition: {self._current_gesture} -> {new_gesture}")

        # 1. Teardown phase: Release keys associated with the PREVIOUS gesture
        self._release_previous_gesture()

        # Update the active state
        self._current_gesture = new_gesture

        # 2. Execution phase: Trigger keys associated with the NEW gesture
        self._trigger_new_gesture()

    def _release_previous_gesture(self) -> None:
        """
        Looks up the previously active gesture and safely releases any keys 
        that were actively being held down.
        """
        if self._current_gesture in self._bindings:
            old_binding = self._bindings[self._current_gesture]
            mode: str = old_binding.get("mode", "hold")
            
            # We only need to manually release 'hold' mode keys.
            # 'press' mode keys were already released by the KeyboardController's tap method.
            if mode == "hold":
                keys_to_release: List[str] = old_binding.get("keys", [])
                for key in keys_to_release:
                    self._keyboard.release(key)

    def _trigger_new_gesture(self) -> None:
        """
        Looks up the newly active gesture, evaluates cooldowns, and executes 
        the associated keyboard commands.
        """
        if self._current_gesture not in self._bindings:
            return

        binding = self._bindings[self._current_gesture]
        action: str = binding.get("action", "Unknown Action")
        mode: str = binding.get("mode", "hold")
        keys: List[str] = binding.get("keys", [])
        
        current_time: float = time.perf_counter()

        if mode == "press":
            # For one-shot actions (like Pause or Horn), check the cooldown timer
            last_time: float = self._last_action_timestamps.get(action, 0.0)
            if current_time - last_time < self._cooldown_duration:
                self._logger.debug(f"Action '{action}' ignored (on cooldown).")
                return
            
            # Execute quick tap and reset the cooldown timer
            for key in keys:
                self._keyboard.tap(key)
            self._last_action_timestamps[action] = current_time
            self._logger.info(f"Executed TAP Action: {action} (Keys: {keys})")

        elif mode == "hold":
            # For continuous actions (like Accelerate or Brake), press and hold
            for key in keys:
                self._keyboard.press(key)
            self._logger.info(f"Executed HOLD Action: {action} (Keys: {keys})")

    def get_current_action_name(self) -> str:
        """
        Retrieves the human-readable action name for the currently active gesture.
        Useful for updating the UI Dashboard.

        Returns:
            str: The name of the action, or 'Idle' if none is active.
        """
        if self._current_gesture in self._bindings:
            return self._bindings[self._current_gesture].get("action", "Idle")
        return "Idle"

    def release_all(self) -> None:
        """
        Emergency teardown. Drops all tracking state and forces the KeyboardController 
        to execute a system-wide failsafe release sweep.
        """
        self._keyboard.release_all()
        self._current_gesture = GestureType.NONE.value
        self._logger.info("GestureMapper reset. All active actions cleared.")