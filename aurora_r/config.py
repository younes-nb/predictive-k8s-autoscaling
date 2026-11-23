"""Strict, explicit experiment configuration. No auto-discovery of writable targets."""
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class Target:
    deployment: str
    istio_workload: str
    min_replicas: int
    max_replicas: int
    service_name: str = ""  # Service exposing the Pods, for EndpointSlice serving confirmation
    def validate(self):
        if not self.deployment or not self.istio_workload:
            raise ValueError('deployment and istio_workload are required')
        if not (1 <= self.min_replicas <= self.max_replicas <= 64):
            raise ValueError(f'bad bounds for {self.deployment}: 1 <= min <= max <= 64')

@dataclass
class Config:
    namespace: str
    targets: list[Target]
    prometheus_url: str
    checkpoint: str
    mode: str = 'shadow'  # shadow, live
    # Where the user-facing SLO comes from.
    #   auto     prefer the independent gateway counters, fall back to Istio
    #   gateway  aurora_slo_* counters only (requires aurora_r.ingress_gateway)
    #   istio    derive the cohort from Istio's own request histogram
    # The Istio fallback exists so the controller can run on a workload whose
    # ingress it does not own; it is a completion-side approximation and cannot
    # see a request that never completes.
    slo_source: str = 'auto'
    # Where the replica proposal comes from.
    #   model  score candidates with the trained action-response ensemble
    #   rule   deterministic safety policy on measured resource pressure
    # ``rule`` exists because the response model is a bootstrap until a set of
    # settled, workload-matched interventions exists. Running the rule policy
    # is how that set is collected; it is not a learned scaling-benefit model.
    policy_source: str = 'model'
    # Telemetry is at 5 s; decisions are INDEPENDENTLY configurable.
    telemetry_step_s: int = 5
    decision_interval_s: float = 15.0
    history_points: int = 24
    telemetry_timeout_s: float = 7.0
    telemetry_workers: int = 8
    max_neighbors_per_caller: int = 12
    max_edges: int = 15000
    top_k_out: int = 5
    top_k_in: int = 5
    slo_window_s: int = 120
    # Frontend workload and latency budget for the Istio-derived SLO fallback.
    front_workload: str = ''
    slo_budget_ms: float = 200.0
    min_slo_eligible: int = 20
    max_slo_age_s: int = 20
    max_metric_age_s: int = 20
    # Safety constraints, not utilization thresholds.
    max_scale_out_step: int = 3
    max_scale_in_step: int = 2
    scale_in_health_s: int = 300
    min_settle_after_ready_s: int = 65
    max_rollout_s: int = 900
    experimental_actuation_ack: bool = False
    journal_path: str = '/var/lib/aurora-r/actions.sqlite3'
    adapter_path: str = '/var/lib/aurora-r/adapter.json'
    lease_name: str = 'aurora-r-writer'
    leader_identity: str = ''
    # Diagnosis thresholds come from held-out negative examples and target FPR,
    # never hand-coded CPU/memory or guessed network-probability cutoffs.
    max_diagnosis_fpr: float = 0.05
    conservative_z: float = 1.5
    min_relative_gain: float = 0.01
    cost_per_replica: float = 0.01
    action_churn_cost: float = 0.008
    uncertainty_penalty: float = 0.5
    capacity_failure_weight: float = 1.0
    max_slo_risk_scalein: float = 0.01
    max_workload_shift_for_learning: float = 0.20
    min_action_examples: int = 30
    # Deterministic rule policy evidence. These are *measured pressure*
    # thresholds, not SLO thresholds: the rule never reads the SLO as a
    # bottleneck signal, only as permission to act.
    rule_cpu_util_scaleout: float = 1.0
    rule_cpu_util_scalein: float = 0.30
    rule_throttle_scaleout: float = 0.05

    def __post_init__(self):
        if self.mode not in {'shadow', 'live'}:
            raise ValueError('mode must be shadow or live')
        if self.slo_source not in {'auto', 'gateway', 'istio'}:
            raise ValueError('slo_source must be auto, gateway or istio')
        if self.policy_source not in {'model', 'rule'}:
            raise ValueError('policy_source must be model or rule')
        if self.rule_cpu_util_scaleout < 0 or self.rule_cpu_util_scalein < 0:
            raise ValueError('rule CPU thresholds must be non-negative')
        if not self.namespace or not self.targets:
            raise ValueError('namespace and explicitly named targets required')
        if len({x.deployment for x in self.targets}) != len(self.targets):
            raise ValueError('duplicate Deployment names')
        if len({x.istio_workload for x in self.targets}) != len(self.targets):
            raise ValueError('ambiguous Istio workload names')
        if len({x.service_name or x.istio_workload for x in self.targets}) != len(self.targets):
            raise ValueError('ambiguous Service names for EndpointSlice serving evidence')
        for x in self.targets: x.validate()
        if self.slo_source != 'gateway' and not self.front_workload:
            raise ValueError('front_workload is required unless slo_source=gateway')
        if self.telemetry_step_s <= 0 or self.decision_interval_s <= 0 or self.history_points < 4:
            raise ValueError('invalid timing')
        if not 0 < self.max_diagnosis_fpr < 1:
            raise ValueError('max_diagnosis_fpr must be in (0,1)')
        if self.max_edges < 1 or self.max_neighbors_per_caller < 1:
            raise ValueError('invalid sparse graph bounds')
        if self.slo_window_s<=0 or self.min_slo_eligible<1 or self.max_slo_age_s<=0 or self.max_metric_age_s<=0:
            raise ValueError('invalid SLO/metric observation requirements')
        if not (0 < self.max_slo_risk_scalein < 1):
            raise ValueError('application SLO violation fraction must be in (0,1)')
        if self.scale_in_health_s<0 or self.max_rollout_s<=0 or self.min_settle_after_ready_s<0:
            raise ValueError('invalid lifecycle delays')
        if self.max_scale_in_step<1 or self.max_scale_out_step<1 or self.top_k_in<1 or self.top_k_out<1:
            raise ValueError('invalid action search bounds')
        if self.conservative_z<=0 or self.cost_per_replica<0 or self.uncertainty_penalty<0 or self.action_churn_cost<0:
            raise ValueError('invalid risk/cost weights')
        if self.mode == 'live' and not self.experimental_actuation_ack:
            raise ValueError('live actuation requires experimental_actuation_ack=true')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        data['targets'] = [Target(**x) for x in data['targets']]
        return cls(**data)

    def target_map(self):
        return {x.deployment: x for x in self.targets}