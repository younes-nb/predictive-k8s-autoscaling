from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SvConfig:
    SWT_LEVEL: int = 4
    MEM_SWT_LEVEL: int = 2
    VMD_K: int = 9
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7


CFG = SvConfig()


def channel_dirs_for(swt_level: int, vmd_k: int, prefix: str = "") -> List[str]:
    dirs: List[str] = [f"{prefix}vmd_mode_{k}" for k in range(vmd_k)]
    for lv in range(swt_level, 1, -1):
        dirs.append(f"{prefix}D{lv}")
    dirs.append(f"{prefix}A{swt_level}")
    return dirs
