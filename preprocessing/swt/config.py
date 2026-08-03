from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SwtConfig:
    SWT_LEVEL: int = 5
    MEM_SWT_LEVEL: int = 5


CFG = SwtConfig()


def channel_dirs_for(swt_level: int, prefix: str = "") -> List[str]:
    dirs = [f"{prefix}A{swt_level}"]
    for lv in range(swt_level, 0, -1):
        dirs.append(f"{prefix}D{lv}")
    return dirs
