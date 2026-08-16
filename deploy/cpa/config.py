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
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "128"))
STABILIZATION_WINDOW_SECONDS = 300
BASE_THRESHOLD = 0.80
TOLERANCE = 0.1
EVAL_INTERVAL_SECONDS = 60
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
HORIZON = 5
RNN_TYPE = "lstm"
BIDIRECTIONAL = False

# Conformal Prediction Configuration (replaces AR residual correction)
CONFORMAL_WINDOW = int(os.getenv("CONFORMAL_WINDOW", "500"))
CONFORMAL_TARGET_ALPHA = float(os.getenv("CONFORMAL_TARGET_ALPHA", "0.05"))
CONFORMAL_ETA = float(os.getenv("CONFORMAL_ETA", "0.01"))
CONFORMAL_ALPHA_MIN = float(os.getenv("CONFORMAL_ALPHA_MIN", "0.01"))
CONFORMAL_ALPHA_MAX = float(os.getenv("CONFORMAL_ALPHA_MAX", "0.20"))
SPIKE_THRESHOLD = float(os.getenv("SPIKE_THRESHOLD", "0.6099"))

STATE_FILE = os.getenv("STATE_FILE", "/tmp/cpa_state.json")
EXPERIMENT_METRICS_FILE = os.getenv(
    "EXPERIMENT_METRICS_FILE", "/tmp/experiment_metrics.csv"
)