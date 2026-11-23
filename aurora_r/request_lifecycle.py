"""Independent, cohort-correct user-request deadline accounting.

The request is counted as SLO-eligible ONCE: on completion before deadline, or
at its deadline if still unfinished. An overdue request finishing later cannot
increment the numerator/denominator twice. Works outside the app runtime.
"""
from dataclasses import dataclass
import heapq,itertools,threading,time
from .schema import FAILURE_CAUSES

CAPACITY_ENVOY_FLAGS={'UO','OM'}    # overflow / overload manager
NETWORK_ENVOY_FLAGS={'UF','UC'}     # failures may be upstream dependency; abstain as network

@dataclass
class RequestState:
    token: int
    namespace: str
    operation: str
    start: float
    deadline: float
    eligible: bool=False
    missed: bool=False

class RequestObserver:
    def __init__(self,deadline_s,namespace,registry=None,clock=time.monotonic):
        if deadline_s<=0:raise ValueError('deadline must be positive')
        self.deadline_s=deadline_s;self.namespace=namespace
        self.clock=clock;self.lock=threading.Lock()
        self.next_id=itertools.count(1);self.pending={};self.expirations=[]
        self.event_counts={}
        self.registered_operations=set()
        self.prom=None
        try:
            from prometheus_client import Counter,Gauge
            kw={'registry':registry} if registry is not None else {}
            self.prom={
                'attempt':Counter('aurora_requests_started_total','Valid user request attempts', ['namespace','operation'],**kw),
                'eligible':Counter('aurora_slo_eligible_total','Completed or deadline-reached valid requests, once each',['namespace','operation'],**kw),
                'missed':Counter('aurora_slo_deadline_violations_total','Valid request deadline misses counted at deadline',['namespace','operation'],**kw),
                'failure':Counter('aurora_request_failures_total','Outcome-attributed request failures',['namespace','operation','cause'],**kw),
                'inflight':Gauge('aurora_requests_inflight','Still outstanding valid request attempts',['namespace','operation'],**kw),
                'heartbeat':Gauge('aurora_observer_heartbeat_seconds','Latest instrumentation watchdog tick',['namespace'],**kw),
                'late':Counter('aurora_requests_late_outcome_total','Late completions after SLO deadline, by classified cause',['namespace','operation','cause'],**kw),
            }
        except ImportError:pass
        self.ensure_operation('default')

    def ensure_operation(self,operation):
        """Register zero counters: absence of failures is known zero, not missing."""
        if operation in self.registered_operations:return
        self.registered_operations.add(operation)
        if self.prom:
            for name in ('attempt','eligible','missed'):
                self.prom[name].labels(self.namespace,operation).inc(0)
            for name in FAILURE_CAUSES:
                self.prom['failure'].labels(self.namespace,operation,name).inc(0)
                self.prom['late'].labels(self.namespace,operation,name).inc(0)
            self.prom['inflight'].labels(self.namespace,operation).set(0)
            self.prom['heartbeat'].labels(self.namespace).set(time.time())

    def _event(self,name,operation,cause=None,delta=1):
        key=(name,operation,cause)
        self.event_counts[key]=self.event_counts.get(key,0)+delta
        if self.prom:
            if name in ('failure','late'):self.prom[name].labels(self.namespace,operation,cause).inc(delta)
            elif name=='inflight':self.prom[name].labels(self.namespace,operation).inc(delta) if delta>0 else self.prom[name].labels(self.namespace,operation).dec(-delta)
            elif name=='heartbeat':self.prom[name].labels(self.namespace).set(time.time())
            else:self.prom[name].labels(self.namespace,operation).inc(delta)
    def start(self,operation='default',at=None):
        at=self.clock() if at is None else at
        with self.lock:
            self.ensure_operation(operation)
            tok=next(self.next_id)
            s=RequestState(tok,self.namespace,operation,at,at+self.deadline_s)
            self.pending[tok]=s
            heapq.heappush(self.expirations,(s.deadline,tok))
            self._event('attempt',operation);self._event('inflight',operation,delta=1)
            return tok
    def sweep(self,at=None):
        at=self.clock() if at is None else at
        with self.lock:
            count=0
            self._event('heartbeat','default')
            while self.expirations and self.expirations[0][0]<=at:
                _,tok=heapq.heappop(self.expirations)
                s=self.pending.get(tok)
                if s and not s.eligible:
                    s.eligible=True;s.missed=True;count+=1
                    self._event('eligible',s.operation)
                    self._event('missed',s.operation)
            return count
    def finish(self,token,cause='none',at=None):
        at=self.clock() if at is None else at
        # Even if the sweep thread was delayed, finishing after a deadline must
        # count the request as missed, not as a fast success.
        self.sweep(at)
        with self.lock:
            s=self.pending.pop(token,None)
            if s is None:return False
            self._event('inflight',s.operation,delta=-1)
            late=s.eligible
            if not s.eligible:
                s.eligible=True;self._event('eligible',s.operation)
            if cause in FAILURE_CAUSES:
                # Failure and eligibility increments must share the same event
                # population for a rate to be meaningful. Late completions are
                # tracked separately, rather than corrupting the SLO cohort.
                self._event('late' if late else 'failure',s.operation,cause)
            return True

def classify_failure(status=None,headers=None,transport=None,application_error=False):
    """Evidence-based classifier; HTTP status alone NEVER means capacity.

    A dedicated trustworthy upstream header (if deployed) may assert an
    application/intentional rejection; Envoy overflow flags support capacity.
    Unknown is an important first-class outcome.
    """
    headers={str(k).lower():str(v) for k,v in (headers or {}).items()}
    cause=headers.get('x-aurora-failure-cause','').strip().lower()
    if cause in FAILURE_CAUSES and cause!='unknown':return cause
    flag=set(headers.get('x-envoy-response-flags','').replace(',',' ').split())
    if flag&CAPACITY_ENVOY_FLAGS:return 'capacity'
    if flag&NETWORK_ENVOY_FLAGS:return 'network'
    if headers.get('x-application-rejection','').lower() in ('true','business'):
        return 'business'
    if application_error:return 'application'
    if transport=='dns' or transport=='connection':return 'network'
    # 2xx/3xx responses are completed and non-failing. Timeouts, 429,
    # 500/503/504 require context; UNKNOWN failures remain counted separately.
    if status is not None and 200<=int(status)<400:return 'none'
    return 'unknown'