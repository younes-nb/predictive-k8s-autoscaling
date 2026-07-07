from dataclasses import dataclass


@dataclass(frozen=True)
class TcnBiGruConfig:
    TCN_KERNEL_SIZE: int = 3
    TCN_FILTERS: tuple = (256, 256)
    TCN_DILATIONS: tuple = (1, 2, 4)
    TCN_DROPOUT: float = 0.2
    BIGRU_HIDDEN: tuple = (64, 128)


CFG = TcnBiGruConfig()
