from dataclasses import dataclass


@dataclass(frozen=True)
class SvConfig:
    SWT_LEVEL: int = 2
    VMD_K: int = 9
    VMD_ALPHA: int = 2000
    VMD_TAU: float = 0.0
    VMD_DC: int = 0
    VMD_INIT: int = 1
    VMD_TOL: float = 1e-7


CFG = SvConfig()
