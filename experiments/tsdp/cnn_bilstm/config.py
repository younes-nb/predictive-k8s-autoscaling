from dataclasses import dataclass


@dataclass(frozen=True)
class CnnBiLSTMConfig:
    KERNEL_SIZES: tuple = (2, 4, 8)
    CONV1_OUT_CH: int = 32
    CONV2_OUT_CH: int = 64
    BILSTM_HIDDEN: tuple = (32, 64, 128)


CFG = CnnBiLSTMConfig()
