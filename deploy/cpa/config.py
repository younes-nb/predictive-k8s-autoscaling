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
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "adaptive")
WINDOW_SIZE = 60
STABILIZATION_WINDOW_SECONDS = 300
UNCERTAINTY_INTERVAL_SECONDS = 600
BASE_THRESHOLD = 0.75
MIN_THRESHOLD = 0.60
MIN_REPLICAS = 1
MAX_REPLICAS = 20
MODEL_PATH = "/app/model.pt"
if FEATURE_SET == "cpu_mem_traffic":
    INPUT_SIZE = 3
elif FEATURE_SET == "cpu_mem":
    INPUT_SIZE = 2
else:
    INPUT_SIZE = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
HORIZON = 5
RNN_TYPE = "lstm"
BIDIRECTIONAL = False
RESIDUAL = os.getenv("RESIDUAL", "false").lower() == "true"
MC_REPEATS = 25
K_FACTOR = 2.0
STATE_FILE = "/tmp/cpa_state.json"
EXPERIMENT_METRICS_FILE = "/tmp/experiment_metrics.csv"
