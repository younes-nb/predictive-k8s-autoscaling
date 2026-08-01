from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SvConfig:
    SWT_LEVEL: int = 4
    MEM_SWT_LEVEL: int = 2
    VMD_K: int = 7
    MEM_VMD_K: int = 8
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7
    NO_VMD: bool = True
    VMD_SWT_LEVEL: int = 1
    MEM_VMD_SWT_LEVEL: int = 1


CFG = SvConfig()


def channel_dirs_for(
    swt_level: int,
    vmd_k: int,
    prefix: str = "",
    no_vmd: bool = False,
) -> List[str]:
    if no_vmd:
        dirs = [f"{prefix}A{swt_level}"]
        for lv in range(swt_level, 0, -1):
            dirs.append(f"{prefix}D{lv}")
        return dirs
    dirs = [f"{prefix}vmd_mode_{k}" for k in range(vmd_k)]
    for lv in range(swt_level, 1, -1):
        dirs.append(f"{prefix}D{lv}")
    dirs.append(f"{prefix}A{swt_level}")
    return dirs
