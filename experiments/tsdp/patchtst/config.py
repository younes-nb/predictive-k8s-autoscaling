from dataclasses import dataclass


@dataclass(frozen=True)
class PatchtstConfig:
    PATCH_LEN: int = 8
    PATCH_STRIDE: int = 4
    D_MODEL: int = 128
    N_HEADS: int = 4
    N_LAYERS: int = 3
    D_FF: int = 256
    DROPOUT: float = 0.2


CFG = PatchtstConfig()
