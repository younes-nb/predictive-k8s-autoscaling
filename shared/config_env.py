import json
import os
import logging
from typing import Any, Optional, Type, Union

logger = logging.getLogger(__name__)


def _parse_json(val: str, default: Any) -> Any:
    try:
        return json.loads(val)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON for env var: {e}, using default: {default}")
        return default


def get_env(
    key: str,
    default: Any = None,
    type_cast: Type = str,
    *,
    is_list: bool = False,
    is_dict: bool = False,
) -> Any:
    """Read environment variable with type casting and logging.

    Args:
        key: Environment variable name (uppercase)
        default: Default value if env var not set
        type_cast: Type to cast the value to (str, int, float, bool)
        is_list: If True, parse as JSON list/array
        is_dict: If True, parse as JSON object/dict

    Returns:
        Parsed value or default
    """
    val = os.getenv(key)
    if val is None:
        return default

    if is_list or is_dict:
        result = _parse_json(val, default)
        logger.info(f"Config from env: {key}={result} (JSON)")
        return result

    if type_cast == bool:
        result = val.lower() in ("1", "true", "yes", "on")
    elif type_cast == int:
        result = int(val)
    elif type_cast == float:
        result = float(val)
    else:
        result = val

    logger.info(f"Config from env: {key}={result}")
    return result