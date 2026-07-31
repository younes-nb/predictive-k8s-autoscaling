from types import SimpleNamespace
from .rnn import SEARCH_SPACE as RNN_SEARCH_SPACE, DEFAULTS as RNN_DEFAULTS, build_model as rnn_build_model
from .bilstm import SEARCH_SPACE as BILSTM_SEARCH_SPACE, DEFAULTS as BILSTM_DEFAULTS, build_model as bilstm_build_model
from .cnn_bilstm import SEARCH_SPACE as CNN_SEARCH_SPACE, DEFAULTS as CNN_DEFAULTS, build_model as cnn_build_model
from .cnn_bilstm_dualpath import SEARCH_SPACE as CNN_DP_SEARCH_SPACE, DEFAULTS as CNN_DP_DEFAULTS, build_model as cnn_dp_build_model

_REGISTRY = {
    "lstm":       ("rnn", RNN_SEARCH_SPACE, RNN_DEFAULTS, rnn_build_model),
    "gru":        ("rnn", RNN_SEARCH_SPACE, RNN_DEFAULTS, rnn_build_model),
    "bilstm":     ("bilstm", BILSTM_SEARCH_SPACE, BILSTM_DEFAULTS, bilstm_build_model),
    "bigrue":     ("rnn", RNN_SEARCH_SPACE, RNN_DEFAULTS, rnn_build_model),
    "cnn_bilstm": ("cnn_bilstm", CNN_SEARCH_SPACE, CNN_DEFAULTS, cnn_build_model),
    "cnn_bilstm_dualpath": ("cnn_bilstm_dualpath", CNN_DP_SEARCH_SPACE, CNN_DP_DEFAULTS, cnn_dp_build_model),
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
