import os
import sys
import logging
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from .config_paths import PATHS


def log_configs(shared_config, approach_config=None, approach_label=None):
    logging.info("\n--- Shared Configuration ---")
    if hasattr(shared_config, "__dict__"):
        for key, value in vars(shared_config).items():
            if not key.startswith("_"):
                logging.info(f"{key:<28}: {value}")
    else:
        logging.info(str(shared_config))

    if approach_config is not None:
        label = approach_label or "Approach"
        logging.info(f"\n--- {label} Configuration ---")
        if hasattr(approach_config, "__dict__"):
            for key, value in vars(approach_config).items():
                if not key.startswith("_"):
                    logging.info(f"{key:<28}: {value}")
        else:
            logging.info(str(approach_config))


def setup_logging(mode="train", log_path=None):
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

    try:
        tehran_tz = ZoneInfo("Asia/Tehran")
    except Exception:
        print("[WARN] Could not find 'Asia/Tehran' timezone. Using system time.")
        tehran_tz = None

    now = datetime.now(tehran_tz)
    if log_path is None:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{mode}_{timestamp}.log"
        log_path = os.path.join(PATHS.LOGS_DIR, log_filename)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    logging.info(f"=== {mode.upper()} SESSION STARTED ===")
    logging.info(f"Log file: {log_path}")
    logging.info(f"Timestamp: {now}")
    return log_path
