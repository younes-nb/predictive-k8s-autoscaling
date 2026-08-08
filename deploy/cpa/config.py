import os

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://prometheus-stack-kube-prom-prometheus.monitoring.svc.cluster.local:9090",
)
NAMESPACE = os.getenv("TARGET_NAMESPACE", "default")
DEPLOYMENT = os.getenv(
    "HCPA_RESOURCE_NAME", os.getenv("TARGET_DEPLOYMENT", "fallback-name")
)
FEATURE_SET = os.getenv("FEATURE_SET", "cpu")
MODEL_TYPE = os.getenv("MODEL_TYPE") or None
PREPROCESS_APPROACH = os.getenv("PREPROCESS_APPROACH", "none")
SWT_LEVEL = int(os.getenv("SWT_LEVEL", "5"))
MEM_SWT_LEVEL = int(os.getenv("MEM_SWT_LEVEL", "5"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "128"))
STABILIZATION_WINDOW_SECONDS = 300
BASE_THRESHOLD = 0.75
MIN_REPLICAS = 1
MAX_REPLICAS = 20
MODEL_PATH = "/app/model.pt"
if FEATURE_SET in ["cpu_mem", "cpu_diff", "cpu_mem_both"]:
    RAW_INPUT_SIZE = 2
else:
    RAW_INPUT_SIZE = 1

if PREPROCESS_APPROACH == "swt":
    INPUT_SIZE = (SWT_LEVEL + 1) + (
        (MEM_SWT_LEVEL + 1) if FEATURE_SET == "cpu_mem_both" else 0
    )
else:
    INPUT_SIZE = RAW_INPUT_SIZE
NUM_TARGETS = 2 if FEATURE_SET == "cpu_mem_both" else 1
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
HORIZON = 5
RNN_TYPE = "lstm"
BIDIRECTIONAL = False
RESIDUAL = os.getenv("RESIDUAL", "false").lower() == "true"

RESIDUAL_CORRECTION = os.getenv("RESIDUAL_CORRECTION", "false").lower() == "true"
AR_ORDER = int(os.getenv("AR_ORDER", "2"))
FORGETTING_FACTOR = float(os.getenv("FORGETTING_FACTOR", "0.95"))
QUANTILE_ALPHA = float(os.getenv("QUANTILE_ALPHA", "0.9"))
RESIDUAL_WINDOW = int(os.getenv("RESIDUAL_WINDOW", str(WINDOW_SIZE)))
RLS_P0 = float(os.getenv("RLS_P0", "1.0"))
AR_ORDER = max(1, min(AR_ORDER, RESIDUAL_WINDOW))
RESIDUAL_WINDOW = max(AR_ORDER, RESIDUAL_WINDOW)

STATE_FILE = os.getenv("STATE_FILE", "/tmp/cpa_state.json")
EXPERIMENT_METRICS_FILE = os.getenv(
    "EXPERIMENT_METRICS_FILE", "/tmp/experiment_metrics.csv"
)
