import time
from typing import Dict, Set, Union
from pynput.keyboard import Controller, Key, KeyCode

from utils.logger import AppLogger


class KeyboardController:
    """
    A state-aware OS keyboard simulator utilizing pynput.
    Tracks currently pressed keys to prevent OS input spamming and 
    provides failsafes to guarantee keys are never permanently stuck down.
    """

    def __init__(self) -> None:
        """
        Initializes the keyboard controller and state tracking sets.
        """
        self._logger = AppLogger.get_logger()
        self._keyboard = Controller()
        self._active_keys: Set[Union[Key, KeyCode]] = set()

        # Mapping dictionary to translate JSON config strings into pynput Key enums
        self._special_keys: Dict[str, Key] = {
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "space": Key.space,
            "escape": Key.esc,
            "esc": Key.esc,
            "enter": Key.enter,
            "shift": Key.shift,
            "ctrl": Key.ctrl,
            "alt": Key.alt,
            "tab": Key.tab,
            "backspace": Key.backspace
        }

    def _get_key_object(self, key_str: str) -> Union[Key, KeyCode, None]:
        """
        Translates a string representation of a key into a pynput object.

        Args:
            key_str (str): The string key from config.json (e.g., 'up', 'space', 'w').

        Returns:
            Union[Key, KeyCode, None]: The pynput key object, or None if invalid.
        """
        key_lower = key_str.lower()
        if key_lower in self._special_keys:
            return self._special_keys[key_lower]
        elif len(key_lower) == 1:
            return KeyCode.from_char(key_lower)
        else:
            self._logger.error(f"Attempted to parse unknown or invalid key string: '{key_str}'")
            return None

    def press(self, key_str: str) -> None:
        """
        Presses and holds a key down. Ignored if the key is already physically pressed.

        Args:
            key_str (str): The string identifier of the key.
        """
        key_obj = self._get_key_object(key_str)
        if key_obj and key_obj not in self._active_keys:
            try:
                self._keyboard.press(key_obj)
                self._active_keys.add(key_obj)
            except Exception as e:
                self._logger.error(f"OS rejected simulated key press for '{key_str}': {e}")

    def release(self, key_str: str) -> None:
        """
        Releases a held key.

        Args:
            key_str (str): The string identifier of the key.
        """
        key_obj = self._get_key_object(key_str)
        if key_obj and key_obj in self._active_keys:
            try:
                self._keyboard.release(key_obj)
                self._active_keys.remove(key_obj)
            except Exception as e:
                self._logger.error(f"OS rejected simulated key release for '{key_str}': {e}")

    def tap(self, key_str: str, duration: float = 0.05) -> None:
        """
        Executes a rapid press-and-release sequence for a single key.
        Ideal for menu navigations, horns, or pause buttons.

        Args:
            key_str (str): The string identifier of the key.
            duration (float): Seconds to hold the key down before releasing.
        """
        key_obj = self._get_key_object(key_str)
        if key_obj:
            try:
                self._keyboard.press(key_obj)
                time.sleep(duration)
                self._keyboard.release(key_obj)
            except Exception as e:
                self._logger.error(f"Failed to tap key '{key_str}': {e}")

    def release_all(self) -> None:
        """
        Emergency failsafe. Instantly releases any and all keys currently 
        tracked as 'pressed' by the controller. 
        Must be called during application exit or when tracking is lost.
        """
        if not self._active_keys:
            return

        # Create a copy of the list to safely modify the original set during iteration
        for key_obj in list(self._active_keys):
            try:
                self._keyboard.release(key_obj)
                self._active_keys.remove(key_obj)
            except Exception as e:
                self._logger.error(f"Failed to release key '{key_obj}' during safety sweep: {e}")
                
        self._logger.info("Executed safety sweep: All virtual keys released.")