import os

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://prometheus-stack-kube-prom-prometheus.monitoring.svc.cluster.local:9090",
)
NAMESPACE = os.getenv("TARGET_NAMESPACE", "default")
DEPLOYMENT = os.getenv(
    "HCPA_RESOURCE_NAME", os.getenv("TARGET_DEPLOYMENT", "fallback-name")
)
FEATURE_SET = os.getenv("FEATURE_SET", "cpu_mem_both")
MODEL_TYPE = os.getenv("MODEL_TYPE") or None
PREPROCESS_APPROACH = os.getenv("PREPROCESS_APPROACH", "none")
SWT_LEVEL = int(os.getenv("SWT_LEVEL", "5"))
MEM_SWT_LEVEL = int(os.getenv("MEM_SWT_LEVEL", "5"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "32"))
STABILIZATION_WINDOW_SECONDS = 300
TOLERANCE = 0.1


def _parse_pct(value, default):
    """Parse a percentage-or-fraction env value into a 0-1 fraction.

    Accepts "80" (percent) or "0.8" (fraction); values > 1 are /100.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v > 1.0:
        v = v / 100.0
    return v


# Model/prediction horizon (minutes per step-ahead block). The CPA eval
# interval is aligned to this: EVAL_INTERVAL_SECONDS defaults to HORIZON*60
# so each CPA cycle consumes exactly one horizon-ahead prediction.
HORIZON = int(os.getenv("HORIZON", "5"))
EVAL_INTERVAL_SECONDS = int(
    os.getenv("EVAL_INTERVAL_SECONDS", str(HORIZON * 60))
)

# --- Adaptive threshold (base 80%, range +/-R, driven by recent errors) ---
# BASE_THRESHOLD must be 80 (%). ADAPTIVE_THRESHOLD_RANGE sets how far the
# live threshold may deviate, e.g. 10 -> live threshold in [70, 90].
BASE_THRESHOLD = _parse_pct(os.getenv("BASE_THRESHOLD", "80"), 0.80)
ADAPTIVE_THRESHOLD_RANGE = _parse_pct(
    os.getenv("ADAPTIVE_THRESHOLD_RANGE", os.getenv("THRESHOLD_RANGE", "10")),
    0.10,
)
ADAPTIVE_THRESHOLD_MIN = max(0.01, BASE_THRESHOLD - ADAPTIVE_THRESHOLD_RANGE)
ADAPTIVE_THRESHOLD_MAX = min(0.99, BASE_THRESHOLD + ADAPTIVE_THRESHOLD_RANGE)
# Error window for the adaptive threshold. Defaults to the model input size
# (WINDOW_SIZE); override with ADAPTIVE_ERROR_WINDOW to decouple them.
# Newest errors weigh most via exp(-age/tau) with tau = N/3 (oldest weight
# ~= 5% of newest).
ADAPTIVE_ERROR_WINDOW = int(
    os.getenv("ADAPTIVE_ERROR_WINDOW", str(WINDOW_SIZE))
)
# Prometheus is the source of truth for recent errors (survives CPA pod
# restarts); pod-local state is only a cold-start fallback.
SCALE_UP_PERIOD_SECONDS = 15
SCALE_UP_MAX_PERCENT = 100.0
SCALE_UP_MAX_PODS = 4
MIN_REPLICAS = 1
MAX_REPLICAS = 10
MODEL_PATH = "/app/model.pt"

if FEATURE_SET in ["cpu_mem", "cpu_diff", "cpu_mem_both"]:
    RAW_INPUT_SIZE = 2
elif FEATURE_SET == "cpu_mem_http_rpc":
    RAW_INPUT_SIZE = 4
else:
    RAW_INPUT_SIZE = 1

if PREPROCESS_APPROACH == "swt":
    INPUT_SIZE = (SWT_LEVEL + 1) + (
        (MEM_SWT_LEVEL + 1) if FEATURE_SET == "cpu_mem_both" else 0
    )
else:
    INPUT_SIZE = RAW_INPUT_SIZE

NUM_TARGETS = 2 if FEATURE_SET in ["cpu_mem_both", "cpu_mem_http_rpc"] else 1
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
RNN_TYPE = "lstm"
BIDIRECTIONAL = False

STATE_FILE = os.getenv("STATE_FILE", "/tmp/cpa_state.json")
EXPERIMENT_METRICS_FILE = os.getenv(
    "EXPERIMENT_METRICS_FILE", "/tmp/experiment_metrics.csv"
)