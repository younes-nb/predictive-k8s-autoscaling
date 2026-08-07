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
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "adaptive")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "128"))
STABILIZATION_WINDOW_SECONDS = 300
UNCERTAINTY_INTERVAL_SECONDS = 600
BASE_THRESHOLD = 0.75
MIN_THRESHOLD = 0.60
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
MC_REPEATS = 25
K_FACTOR = 20.0
STATE_FILE = "/tmp/cpa_state.json"
EXPERIMENT_METRICS_FILE = "/tmp/experiment_metrics.csv"
