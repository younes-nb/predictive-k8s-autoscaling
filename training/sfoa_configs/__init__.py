from types import SimpleNamespace
from .rnn import SEARCH_SPACE as RNN_SEARCH_SPACE, DEFAULTS as RNN_DEFAULTS, build_model as rnn_build_model
from .bilstm import SEARCH_SPACE as BILSTM_SEARCH_SPACE, DEFAULTS as BILSTM_DEFAULTS, build_model as bilstm_build_model
from .cnn_bilstm import SEARCH_SPACE as CNN_SEARCH_SPACE, DEFAULTS as CNN_DEFAULTS, build_model as cnn_build_model
from .freq_tsmixer import SEARCH_SPACE as FREQ_TSMIXER_SEARCH_SPACE, DEFAULTS as FREQ_TSMIXER_DEFAULTS, build_model as freq_tsmixer_build_model
from .farms_cnn import SEARCH_SPACE as FARMS_CNN_SEARCH_SPACE, DEFAULTS as FARMS_CNN_DEFAULTS, build_model as farms_cnn_build_model
from .hfmixer import SEARCH_SPACE as HFMIXER_SEARCH_SPACE, DEFAULTS as HFMIXER_DEFAULTS, build_model as hfmixer_build_model

_REGISTRY = {
    "lstm":       ("rnn", RNN_SEARCH_SPACE, RNN_DEFAULTS, rnn_build_model),
    "gru":        ("rnn", RNN_SEARCH_SPACE, RNN_DEFAULTS, rnn_build_model),
    "bilstm":     ("bilstm", BILSTM_SEARCH_SPACE, BILSTM_DEFAULTS, bilstm_build_model),
    "bigrue":     ("rnn", RNN_SEARCH_SPACE, RNN_DEFAULTS, rnn_build_model),
    "cnn_bilstm": ("cnn_bilstm", CNN_SEARCH_SPACE, CNN_DEFAULTS, cnn_build_model),
    "freq_tsmixer": ("freq_tsmixer", FREQ_TSMIXER_SEARCH_SPACE, FREQ_TSMIXER_DEFAULTS, freq_tsmixer_build_model),
    "farms_cnn":    ("farms_cnn", FARMS_CNN_SEARCH_SPACE, FARMS_CNN_DEFAULTS, farms_cnn_build_model),
    "hfmixer":      ("hfmixer", HFMIXER_SEARCH_SPACE, HFMIXER_DEFAULTS, hfmixer_build_model),
}


def get_config(model_type):
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(_REGISTRY.keys())}")
    name, search_space, defaults, build_fn = _REGISTRY[model_type]
    return SimpleNamespace(
        NAME=name,
        SEARCH_SPACE=search_space,
        DEFAULTS=defaults,
        build_model=build_fn,
    )
