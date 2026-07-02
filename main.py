import tkinter as tk
from utils.logger import AppLogger
from utils.helpers import ConfigManager
from ui.dashboard import DashboardApp

def main() -> None:
    """
    Application entry point. Initializes logging, loads configurations, 
    and launches the main UI thread.
    """
    # 1. Initialize Logger
    logger = AppLogger.get_logger()
    logger.info("GestureControl Application Starting...")

    try:
        # 2. Ensure Config is loaded before UI startup
        ConfigManager.load_config()

        # 3. Launch GUI
        root = tk.Tk()
        app = DashboardApp(root)
        
        logger.info("UI initialized. Entering main loop.")
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Application crashed during startup: {e}")
    finally:
        logger.info("Application closed.")

if __name__ == "__main__":
    main()