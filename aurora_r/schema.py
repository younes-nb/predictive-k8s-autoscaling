"""Stable data and feature contracts, serializable for intervention collection.

All windows use past observed measurements at 5 s steps; missing values have
explicit masks. No future resource or traffic values are model inputs.
"""
from dataclasses import dataclass, field
import numpy as np

SCHEMA_VERSION = 'aurora-r.graph.v4'
NODE_FEATURES = [
    'req_rate', 'source_req_rate', 'service_inflight', 'http_5xx_rate', 'http_503_rate', 'latency_p50_ms',
    'latency_p90_ms', 'latency_p99_ms', 'capacity_flag_rate',
    'network_flag_rate', 'http_4xx_rate', 'source_capacity_flag_rate',
    'source_network_flag_rate', 'cpu_cores', 'cpu_request_cores',
    'cpu_throttle_frac', 'mem_working_bytes', 'mem_request_bytes',
    'mem_limit_bytes', 'net_rx_bytes_s', 'net_tx_bytes_s',
    'fs_read_bytes_s', 'fs_write_bytes_s', 'oom_rate', 'restart_rate',
    'cpu_request_util', 'mem_request_util', 'mem_limit_util', 'req_per_ready',
]
EDGE_FEATURES = ['call_rate', 'error_5xx_rate', 'latency_p90_ms']
DEGRADATION = ['cpu','memory','network','storage','queue','dependency','application','infrastructure','unknown']
FAILURE_CAUSES = ['capacity', 'application', 'business', 'dependency', 'network', 'unknown']

# Known, dimensionless ratio features are not log-transformed. Large positive
# observations (bytes, latency, rates) receive log1p; train/live share this.
RATIO_FEATURES = {'cpu_throttle_frac','cpu_request_util','mem_request_util','mem_limit_util'}

def _num(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def node_vector(raw, ready):
    r = dict(raw)
    def ratio(a,b):
        x,y=_num(r.get(a)),_num(r.get(b))
        return None if x is None or y is None or y <= 0 else x/y
    r['cpu_request_util'] = ratio('cpu_cores','cpu_request_cores')
    r['mem_request_util'] = ratio('mem_working_bytes','mem_request_bytes')
    r['mem_limit_util'] = ratio('mem_working_bytes','mem_limit_bytes')
    q=_num(r.get('req_rate'))
    r['req_per_ready'] = None if ready <= 0 or q is None else q/ready
    values=[]; mask=[]
    for k in NODE_FEATURES:
        v=_num(r.get(k))
        if v is None or v < 0:
            values.append(0.0); mask.append(0.0)
        else:
            values.append(float(v) if k in RATIO_FEATURES else float(np.log1p(v)))
            mask.append(1.0)
    return values,mask

def edge_vector(raw):
    xs=[];mask=[]
    for k in EDGE_FEATURES:
        x=_num(raw.get(k))
        if x is None or x < 0:
            xs.append(0.0);mask.append(0.0)
        else:
            xs.append(float(np.log1p(x)));mask.append(1.0)
    return xs,mask

@dataclass
class ReplicaState:
    spec: int
    ready: int
    running: int
    available: int
    scheduled: int
    terminating: int
    request_pending: int
    rollout_complete: bool
    ready_serving: int
    pod_startup_seconds: list[float] = field(default_factory=list)
    def stable(self):
        return (self.spec == self.ready == self.available == self.ready_serving
                and self.request_pending == 0 and self.terminating == 0
                and self.rollout_complete)

@dataclass
class SLOState:
    eligible: float | None
    missed: float | None
    capacity_fail: float | None
    unresolved: float | None
    sampled_at: float
    window_s: int
    p99_ms: float | None = None
    def valid(self, now, max_age_s, min_eligible):
        if now - self.sampled_at > max_age_s or self.sampled_at > now + 5:
            return False
        vals=[self.eligible,self.missed,self.capacity_fail,self.unresolved]
        if any(v is None or not np.isfinite(v) or v<0 for v in vals):
            return False
        if self.eligible < min_eligible or self.missed > self.eligible + 1e-6:
            return False
        if self.capacity_fail > self.eligible + 1e-6:
            return False
        return True
    def upper_bernoulli_bound(self, misses, z=1.96):
        """Wilson one-sided upper bound for observed deadline/failure rates.

        Prometheus increase() extrapolates counters and may return fractional
        numbers; the resulting bound is an approximation, not exact per-request
        binomial confidence when requests are dependent.
        """
        if self.eligible is None or self.eligible<=0 or misses is None:return None
        n=float(self.eligible);p=min(1.,max(0.,float(misses)/n))
        denom=1+z*z/n
        return min(1.,(p+z*z/(2*n)+z*((p*(1-p)/n+z*z/(4*n*n))**.5))/denom)
    @property
    def miss_rate(self):
        return min(1.0, max(0.0, self.missed/self.eligible)) if self.eligible else None
    @property
    def capacity_failure_rate(self):
        return min(1.0, max(0.0, self.capacity_fail/self.eligible)) if self.eligible else None

@dataclass
class GraphSnapshot:
    deployment_id: str
    run_id: str
    timestamp: float
    services: list[str]
    node_x: np.ndarray          # [N,T,F] transformed, finite
    node_mask: np.ndarray       # [N,T,F] binary
    edge_index: np.ndarray      # [2,E] caller -> callee
    edge_x: np.ndarray          # [E,T,Fe]
    edge_mask: np.ndarray       # [E,T,Fe]
    replicas: dict[str,ReplicaState]
    slo: SLOState
    root_labels: dict = field(default_factory=dict)
    affected_labels: dict = field(default_factory=dict)
    remedy_labels: dict = field(default_factory=dict)
    scalein_labels: dict = field(default_factory=dict)
    degradation_labels: dict = field(default_factory=dict)
    failure_labels: dict = field(default_factory=dict)
    schema: str = SCHEMA_VERSION
    def validate(self):
        n=len(self.services)
        if self.schema!=SCHEMA_VERSION:raise ValueError('schema mismatch')
        if len(set(self.services))!=n:raise ValueError('duplicate services')
        if self.node_x.ndim!=3 or self.node_x.shape!=(n,self.node_x.shape[1],len(NODE_FEATURES)):
            raise ValueError('wrong node shape')
        if self.node_mask.shape!=self.node_x.shape or self.edge_index.shape[0]!=2:
            raise ValueError('wrong masks/edge index')
        e=self.edge_index.shape[1]
        if self.edge_x.shape!=(e,self.node_x.shape[1],len(EDGE_FEATURES)) or self.edge_mask.shape!=self.edge_x.shape:
            raise ValueError('wrong edge shape')
        if e and (self.edge_index.min()<0 or self.edge_index.max()>=n):
            raise ValueError('edge out of range')
        if self.node_mask.size and not np.isin(self.node_mask, [0.,1.]).all():
            raise ValueError('node mask not binary')
        if self.edge_mask.size and not np.isin(self.edge_mask, [0.,1.]).all():
            raise ValueError('edge mask not binary')
        for a in (self.node_x,self.node_mask,self.edge_x,self.edge_mask):
            if not np.isfinite(a).all(): raise ValueError('nonfinite tensor')
        for s in self.services:
            if s not in self.replicas:raise ValueError('missing replica state '+s)
        return self
    def to_dict(self):
        return {**{k:getattr(self,k) for k in ('deployment_id','run_id','timestamp','services','root_labels','affected_labels','remedy_labels','scalein_labels','degradation_labels','failure_labels','schema')},
            'node_x':self.node_x.tolist(),'node_mask':self.node_mask.tolist(),
            'edge_index':self.edge_index.tolist(),'edge_x':self.edge_x.tolist(),
            'edge_mask':self.edge_mask.tolist(),
            'replicas':{k:vars(v) for k,v in self.replicas.items()},'slo':vars(self.slo)}
    @classmethod
    def from_dict(cls, rec):
        kw={k:rec[k] for k in ('deployment_id','run_id','timestamp','services')}
        for k in ('root_labels','affected_labels','remedy_labels','scalein_labels','degradation_labels','failure_labels'):
            kw[k]=rec.get(k,{})
        kw['schema']=rec.get('schema','')
        kw['replicas']={k:ReplicaState(**v) for k,v in rec['replicas'].items()}
        kw['slo']=SLOState(**rec['slo'])
        for k,dtype in [('node_x',np.float32),('node_mask',np.float32),('edge_index',np.int64),('edge_x',np.float32),('edge_mask',np.float32)]:
            arr=np.asarray(rec[k],dtype=dtype)
            if k=='edge_index' and arr.size==0:arr=arr.reshape(2,0)
            if k in ('edge_x','edge_mask') and arr.size==0:
                arr=arr.reshape(0,kw['node_x'].shape[1],len(EDGE_FEATURES))
            kw[k]=arr
        return cls(**kw).validate()

@dataclass
class Intervention:
    """Outcome-backed, one-Deployment-at-a-time intervention, NOT a weak label.

    post metrics are recorded only after Ready+serving and a settled observation
    window; for causal analysis set matched_control=True and record concurrent
    control trends or use randomized workload-matched intervention order.
    """
    snapshot: GraphSnapshot
    service: str
    target_replicas: int
    post_eligible: float
    post_missed: float
    post_capacity_fail: float
    workload_shift: float
    settled: bool
    isolated: bool
    matched_control: bool
    confidence_tag: str = 'observational'
    control_adjustment_miss: float = 0.0
    control_adjustment_fail: float = 0.0
    @classmethod
    def from_dict(cls,r):
        return cls(snapshot=GraphSnapshot.from_dict(r['snapshot']),**{k:v for k,v in r.items() if k!='snapshot'})
    def to_dict(self):
        return {**{k:v for k,v in vars(self).items() if k!='snapshot'}, 'snapshot':self.snapshot.to_dict()}
    def eligible(self, max_shift=.2):
        return (self.settled and self.isolated and self.matched_control and
            self.confidence_tag in ('matched','randomized') and
            self.service in self.snapshot.services and self.post_eligible>=20 and
            0<=self.post_missed<=self.post_eligible and 0<=self.post_capacity_fail<=self.post_eligible and
            abs(self.workload_shift)<=max_shift and
            np.isfinite(self.control_adjustment_miss) and np.isfinite(self.control_adjustment_fail) and
            self.snapshot.slo.valid(self.snapshot.timestamp, 120, 20))