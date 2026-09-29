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
