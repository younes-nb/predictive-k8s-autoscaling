"""PBScaler configuration, adapted for the k8sautoscaledl lab cluster.

Upstream PBScaler hardcoded machine-specific paths and a fixed namespace.
Every knob here is env-overridable (PBS_* vars) so the same code runs:

  - locally on the runner (kubeconfig + port-forward to Mimir), and
  - in-cluster as a Deployment (ServiceAccount RBAC + in-cluster Mimir URL).

Defaults below match the live cluster:

  - TrainTicket lives in the ``train-ticket`` namespace, every stateless
    deployment has an HPA (min 1 / max 10, CPU 80%).
  - Metrics land in Mimir (Alloy remote-writes into it). Mimir serves the
    standard Prometheus HTTP API under the ``/prometheus`` prefix.
  - PDF: "The SLO values for ... Train Ticket were set to 200 ms"
    (Sec. IV-A4), upper bound 8 replicas per microservice in their cluster;
    ours is 10 to match the existing HPAs.
"""

import os


def get_env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def get_env_int(name: str, default: int) -> int:
    try:
        return int(get_env(name, str(default)))
    except ValueError:
        return default


def get_env_float(name: str, default: float) -> float:
    try:
        return float(get_env(name, str(default)))
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    raw = get_env(name, "true" if default else "false").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def getNowTime():
    import time
    return int(round(time.time()))


class Config():
    def __init__(self):
        # ---- workload: TrainTicket in its own namespace ------------------
        self.namespace = get_env("PBS_NAMESPACE", "train-ticket")

        # End-to-end front service (receives gateway traffic). Used as the
        # SLO reference for the SLO violation predictor label.
        self.front_svc = get_env("PBS_FRONT_SVC", "ts-ui-dashboard")
        # Istio ingress gateway workload name — filtered out of TopoRank's
        # abnormal subgraph (upstream used 'istio-ingressgateway').
        self.gateway_name = get_env("PBS_GATEWAY_NAME", "train-ticket-gateway-istio")

        self.SLO = get_env_float("PBS_SLO", 200.0)  # ms, paper Sec. IV-A4

        # Replica bounds (paper限制了8, ours aligned to the cluster's HPAs)
        self.max_pod = get_env_int("PBS_MAX_POD", 10)
        self.min_pod = get_env_int("PBS_MIN_POD", 1)

        # ---- k8s access --------------------------------------------------
        # Empty/missing -> KubernetesClient auto-detects: in-cluster config
        # when running as a pod, KUBECONFIG / ~/.kube/config otherwise.
        self.k8s_config = get_env(
            "PBS_K8S_CONFIG",
            os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config"))

        # ---- experiment duration / loop cadence --------------------------
        self.duration = get_env_int("PBS_DURATION", 20 * 60)  # seconds
        self.start = getNowTime()
        self.end = self.start + self.duration
        # Paper: monitoring interval is 5 s; Mimir exists hourly, step 5 s works
        self.step = get_env_int("PBS_PROM_STEP", 5)

        # ---- prometheus (Mimir) API --------------------------------------
        _mimir = get_env("PBS_MIMIR_URL", "")
        if not _mimir:
            # In-cluster default (k8s service DNS). For local runs set
            # PBS_MIMIR_URL=http://127.0.0.1:18080 together with:
            #   kubectl -n monitoring port-forward svc/mimir 18080:8080
            _mimir = "http://mimir.monitoring.svc.cluster.local:8080"
        self.prom_range_url = get_env("PBS_PROM_RANGE_URL", f"{_mimir}/prometheus/api/v1/query_range")
        self.prom_no_range_url = get_env("PBS_PROM_QUERY_URL", f"{_mimir}/prometheus/api/v1/query")

        # ---- SLO violation predictor -------------------------------------
        # RandomForest trained on the collector's train_data.csv
        # (simulation/RandomForestClassify.py). Relative to pbscaler/ root.
        self.slo_predictor_path = get_env(
            "PBS_SLO_PREDICTOR", "simulation/train_ticket/RandomForestClassify.model")

        # Companion svc-order list written next to the model by the training
        # script (<model>.svcs): the GA must feed the predictor the exact
        # feature row/order it was trained on.
        self.slo_predictor_svcs_path = get_env(
            "PBS_SLO_PREDICTOR_SVCS",
            os.path.splitext(self.slo_predictor_path)[0] + ".svcs")

        # ---- scaling integration -----------------------------------------
        # PBScaler patches Deployment replicas (paper: "commits optimized
        # strategies to the kubernetes-client"). The cluster's TrainTicket
        # HPAs must be out of the picture while PBScaler runs, otherwise
        # they overwrite the patch every 15 s.
        #   suspend_hpas_on_start: delete all HPAs in the namespace at start
        #     (state saved to hpa_state_file), restore them at exit
        #     (clean end, KeyboardInterrupt or SIGTERM).
        self.suspend_hpas_on_start = get_env_bool("PBS_SUSPEND_HPAS", True)
        self.restore_hpas_on_exit = get_env_bool("PBS_RESTORE_HPAS", True)
        self.hpa_state_file = get_env("PBS_HPA_STATE_FILE", "data/hpa_state.json")

        # dry_run: run every phase (detection, TopoRank, GA) but print the
        # action instead of patching replicas. Also avoids the need for a
        # trained predictor file (an identity stub is used).
        self.dry_run = get_env_bool("PBS_DRY_RUN", False)

        # ---- PBScaler loop constants (paper Sec. IV-A4) -------------------
        self.ab_check_interval = get_env_int("PBS_AB_CHECK_INTERVAL", 15)
        self.waste_check_interval = get_env_int("PBS_WASTE_CHECK_INTERVAL", 120)
        self.alpha = get_env_float("PBS_ALPHA", 0.2)
        self.beta = get_env_float("PBS_BETA", 0.9)
        self.k_top = get_env_int("PBS_K", 2)
        self.conf = get_env_float("PBS_CONF", 0.05)
