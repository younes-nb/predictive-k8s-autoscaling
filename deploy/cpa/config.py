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
WINDOW_SIZE = 60
STABILIZATION_WINDOW_SECONDS = 300
UNCERTAINTY_INTERVAL_SECONDS = 600
BASE_THRESHOLD = 0.75
MIN_THRESHOLD = 0.50
MAX_THRESHOLD = 0.95
MIN_REPLICAS = 1
MAX_REPLICAS = 20
MODEL_PATH = "/app/model.pt"
INPUT_SIZE = 2 if FEATURE_SET == "cpu_mem" else 1
HIDDEN_SIZE = 96
NUM_LAYERS = 3
DROPOUT = 0.4
HORIZON = 5
BIDIRECTIONAL = True
MC_REPEATS = 50
K_FACTOR = 5.0
STATE_FILE = "/tmp/cpa_state.json"
