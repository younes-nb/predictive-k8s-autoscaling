"""AURORA-HPA configuration.

Every knob is env-overridable (``AURORA_*``) so the same code runs:

  * on the loadgen for training / dry-runs (kubeconfig + Mimir tunnel), and
  * in-cluster as a Deployment (ServiceAccount + in-cluster Mimir URL).

Timing defaults are the experiment settings from the spec:
telemetry 5 s, decision 15 s, 3 observations per decision.
"""

import os


def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _int(name, default):
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name, default):
    return str(_env(name, "true" if default else "false")).strip().lower() in (
        "1", "true", "yes", "y", "on")


class Config:
    def __init__(self):
        # ---- target -------------------------------------------------------
        self.namespace = _env("AURORA_NAMESPACE", "train-ticket")
        self.front_svc = _env("AURORA_FRONT_SVC", "ts-ui-dashboard")
        self.slo_ms = _float("AURORA_SLO_MS", 200.0)

        # ---- replica bounds (operational, not bottleneck thresholds) -------
        self.min_pod = _int("AURORA_MIN_POD", 1)
        self.max_pod = _int("AURORA_MAX_POD", 10)

        # ---- timing -------------------------------------------------------
        self.telemetry_step_s = _int("AURORA_TELEMETRY_STEP_S", 5)
        self.decision_interval_s = _int("AURORA_DECISION_INTERVAL_S", 15)
        self.history_len = _int("AURORA_HISTORY_LEN", 24)   # observations = 2 min
        # A tick is stale if its newest sample is older than this many intervals.
        self.stale_after_intervals = _int("AURORA_STALE_AFTER_INTERVALS", 2)
        # Missing-telemetry policy: "safe" = hold replicas, "skip" = skip the cycle.
        self.stale_policy = _env("AURORA_STALE_POLICY", "safe")

        # ---- duration -----------------------------------------------------
        self.duration_s = _int("AURORA_DURATION_S", 3900)

        # ---- k8s ----------------------------------------------------------
        self.k8s_config = _env(
            "AURORA_K8S_CONFIG",
            os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config"))
        self.suspend_hpas = _bool("AURORA_SUSPEND_HPAS", True)
        self.restore_hpas = _bool("AURORA_RESTORE_HPAS", True)
        self.hpa_state_file = _env("AURORA_HPA_STATE_FILE", "/var/lib/aurora/hpa_state.json")

        # ---- Mimir --------------------------------------------------------
        mimir = _env("AURORA_MIMIR_URL", "")
        if not mimir:
            mimir = "http://mimir.monitoring.svc.cluster.local:8080"
        self.mimir = mimir.rstrip("/")
        self.range_url = _env("AURORA_MIMIR_RANGE_URL", f"{self.mimir}/prometheus/api/v1/query_range")
        self.query_url = _env("AURORA_MIMIR_QUERY_URL", f"{self.mimir}/prometheus/api/v1/query")

        # ---- state --------------------------------------------------------
        self.state_dir = _env("AURORA_STATE_DIR", "/var/lib/aurora")
        self.profile_path = os.path.join(self.state_dir, "profiles.json")
        self.ledger_path = os.path.join(self.state_dir, "ledger.jsonl")
        self.registry_dir = _env("AURORA_REGISTRY_DIR", "/var/lib/aurora/registry")

        # ---- model --------------------------------------------------------
        self.model_dir = _env("AURORA_MODEL_DIR", "/opt/aurora/model")
        self.device = _env("AURORA_DEVICE", "cpu")
        self.uncertainty_scale = _float("AURORA_UNCERTAINTY_SCALE", 1.0)

        # ---- behaviour ----------------------------------------------------
        self.dry_run = _bool("AURORA_DRY_RUN", False)
        self.seed = _int("AURORA_SEED", 20)

    def __repr__(self):
        return (f"Config(ns={self.namespace} slo={self.slo_ms} "
                f"interval={self.decision_interval_s}s step={self.telemetry_step_s}s "
                f"dry_run={self.dry_run})")