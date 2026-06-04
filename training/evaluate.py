# Backward-compat: re-export evaluate and setup_logging.
from .evaluate_entry import evaluate  # noqa: F401
from shared.logging_utils import setup_logging  # noqa: F401 -- keeps calculate_k_factor.py import working
