import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any


class AppLogger:
    """
    A thread-safe, centralized logging manager for the AI Gesture Controller Pro.
    Configures synchronized console and rolling file logging outputs.
    """
    _logger: Optional[logging.Logger] = None

    @classmethod
    def initialize(cls, logging_config: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """
        Initializes and returns the global application logger.
        
        Args:
            logging_config (Optional[Dict[str, Any]]): Logging configurations parsed from config.json.
                                                       Expects keys: 'level', 'log_to_file', 'log_file_path'.
        
        Returns:
            logging.Logger: The configured root-level application logger instance.
        """
        if cls._logger is not None:
            return cls._logger

        # Fallback defaults if configuration is completely missing or corrupted
        log_level_str: str = (logging_config or {}).get("level", "INFO").upper()
        log_to_file: bool = (logging_config or {}).get("log_to_file", True)
        log_file_path_str: str = (logging_config or {}).get("log_file_path", "logs/gesture_controller.log")

        # Map string level to logging numeric constants
        log_level: int = getattr(logging, log_level_str, logging.INFO)

        # Instantiate base logger instance
        logger: logging.Logger = logging.getLogger("AIGestureControllerPro")
        logger.setLevel(log_level)
        
        # Prevent log duplication issues across module re-imports
        if logger.hasHandlers():
            logger.handlers.clear()

        # Thread-safe unified formatting string
        log_format: logging.Formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Standard Console Output Handler
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # 2. Permanent File Output Handler (Optional based on configuration)
        if log_to_file:
            try:
                log_file: Path = Path(log_file_path_str).resolve()
                # Safely ensure target logging folder structure exists
                log_file.parent.mkdir(parents=True, exist_ok=True)

                file_handler: logging.FileHandler = logging.FileHandler(
                    filename=str(log_file), 
                    encoding="utf-8"
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(log_format)
                logger.addHandler(file_handler)
            except (OSError, PermissionError) as error:
                # Fallback to stdout notify if writing to disk filesystem fails abruptly
                logger.error(f"Failed to initialize file logger at {log_file_path_str}: {error}")

        cls._logger = logger
        cls._logger.info("Application logging subsystem successfully initialized.")
        return cls._logger

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Retrieves the global active application logger instance. 
        If not explicitly initialized prior, spins up a default-configured fallback instance.
        
        Returns:
            logging.Logger: The global active application logger instance.
        """
        if cls._logger is None:
            return cls.initialize(None)
        return cls._logger