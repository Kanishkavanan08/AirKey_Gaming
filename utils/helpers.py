import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from utils.logger import AppLogger


class ConfigManager:
    """
    Singleton-patterned configuration manager.
    Safely loads, caches, and dispenses configuration values from config.json.
    """
    _config: Dict[str, Any] = {}
    _is_loaded: bool = False
    _logger = AppLogger.get_logger()

    @classmethod
    def load_config(cls, config_path: str = "config/config.json") -> Dict[str, Any]:
        """
        Reads the JSON configuration file from disk and caches it in memory.

        Args:
            config_path (str): The relative or absolute path to the config file.

        Returns:
            Dict[str, Any]: The parsed configuration dictionary.
            
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the JSON file is malformed.
        """
        path: Path = Path(config_path).resolve()
        
        if not path.exists():
            cls._logger.error(f"Configuration file missing at {path}")
            raise FileNotFoundError(f"Required configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as file:
                cls._config = json.load(file)
            cls._is_loaded = True
            cls._logger.info(f"Configuration successfully loaded from {path}")
            return cls._config
        except json.JSONDecodeError as error:
            cls._logger.error(f"Malformed JSON in config file {path}: {error}")
            raise ValueError(f"Invalid JSON configuration: {error}")
        except Exception as error:
            cls._logger.error(f"Unexpected error while loading config: {error}")
            raise

    @classmethod
    def get(cls, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Safely retrieves a configuration value. If the config hasn't been loaded,
        it attempts to load the default config path automatically.

        Args:
            section (str): The top-level key in the JSON file (e.g., 'camera', 'detection').
            key (Optional[str]): The specific setting key. If None, returns the entire section.
            default (Any): The fallback value if the section or key does not exist.

        Returns:
            Any: The requested configuration value, or the default.
        """
        if not cls._is_loaded:
            cls._logger.warning("ConfigManager accessed before explicit load. Triggering auto-load.")
            try:
                cls.load_config()
            except Exception:
                return default

        section_data: Any = cls._config.get(section)
        
        if section_data is None:
            return default
            
        if key is None:
            return section_data
            
        if isinstance(section_data, dict):
            return section_data.get(key, default)
            
        return default


class MathUtils:
    """
    A collection of static geometric and mathematical utilities 
    specifically optimized for MediaPipe 2D landmark coordinates.
    """

    @staticmethod
    def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculates the Euclidean distance between two 2D points.

        Args:
            p1 (Tuple[float, float]): The first point (x, y).
            p2 (Tuple[float, float]): The second point (x, y).

        Returns:
            float: The Euclidean distance.
        """
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculates the angle (in degrees) formed by three points, where p2 is the vertex.
        Crucial for determining if fingers are curled or extended.

        Args:
            p1 (Tuple[float, float]): The start point (e.g., MCP joint).
            p2 (Tuple[float, float]): The vertex point (e.g., PIP joint).
            p3 (Tuple[float, float]): The end point (e.g., TIP joint).

        Returns:
            float: The internal angle in degrees (0.0 to 180.0).
        """
        # Vector 1 (p2 -> p1)
        v1_x, v1_y = p1[0] - p2[0], p1[1] - p2[1]
        # Vector 2 (p2 -> p3)
        v2_x, v2_y = p3[0] - p2[0], p3[1] - p2[1]

        dot_product = (v1_x * v2_x) + (v1_y * v2_y)
        magnitude_v1 = math.hypot(v1_x, v1_y)
        magnitude_v2 = math.hypot(v2_x, v2_y)

        # Prevent division by zero if points are identical
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0

        # Bound the cosine value to [-1.0, 1.0] to prevent math domain errors from floating point drift
        cos_angle = max(-1.0, min(1.0, dot_product / (magnitude_v1 * magnitude_v2)))
        
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)