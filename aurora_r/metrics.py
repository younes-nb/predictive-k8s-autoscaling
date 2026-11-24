"""Prometheus/Mimir ingestion with native 5-second history and sparse call graph.

One cluster-wide query per metric family, NOT one request per service. Unseen
series are masked rather than fabricated as zero. Caller/callee edges are
bounded by observed rate (top neighbors), never an NxN adjacency matrix.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import dataclass
import math,time,requests,numpy as np
from .schema import (GraphSnapshot,SLOState,NODE_FEATURES,EDGE_FEATURES,
                     node_vector,edge_vector)

@dataclass(frozen=True)
class Query:
    key: str
    expr: str
    scope: str   # pod / workload / edge / global

class Prometheus:
    """Mimir/Prometheus HTTP client with bounded retry.

    A concurrent collection re-issues many range queries at once and Mimir
    answers 429 (limiter) under that burst. A single 429 used to drop the whole
    metric family for the cycle; retry with backoff instead.
    """
    def __init__(self,url,timeout_s=7.0,retries=3):
        self.url=url.rstrip('/')
        self.timeout=timeout_s
        self.retries=max(1,int(retries))
    def get(self,path,params):
        last=None
        for attempt in range(self.retries):
            try:
                rsp=requests.get(self.url+path,params=params,timeout=self.timeout)
                if rsp.status_code in (429,500,502,503,504):
                    last=RuntimeError(f'HTTP {rsp.status_code} for {path}')
                    if attempt+1<self.retries:
                        time.sleep(min(8.0,0.5*(2**attempt)))
                        continue
                rsp.raise_for_status()
                payload=rsp.json()
                if payload.get('status')!='success':
                    raise RuntimeError('Prometheus query failed: '+str(payload.get('error')))
                return payload['data']['result']
            except requests.RequestException as exc:
                last=exc
                if attempt+1>=self.retries:break
                time.sleep(min(8.0,0.5*(2**attempt)))
        raise last if last else RuntimeError('query failed')
    def range(self,expr,start,end,step):
        return self.get('/api/v1/query_range',{'query':expr,'start':start,'end':end,'step':step})
    def instant(self,expr,at=None):
        params={'query':expr}
        if at is not None:params['time']=at
        return self.get('/api/v1/query',params)

# Only app containers, no proxy sidecars; different runtimes share this schema.
def queries(ns):
    sel=f'namespace="{ns}",container!="",container!="POD",container!="istio-proxy"'
    rq=f'istio_requests_total{{reporter="destination",destination_workload_namespace="{ns}"}}'
    rh=f'istio_request_duration_milliseconds_bucket{{reporter="destination",destination_workload_namespace="{ns}"}}'
    edge=f'source_workload_namespace="{ns}",destination_workload_namespace="{ns}",reporter="destination"'
    erq=f'istio_requests_total{{{edge}}}'
    src_edge=edge.replace('reporter="destination"','reporter="source"')
    erh=f'istio_request_duration_milliseconds_bucket{{{edge}}}'
    wg=lambda expr:f'sum by (destination_workload) ({expr})'
    pod=lambda expr:f'sum by (pod) ({expr})'
    egroup=lambda expr:f'sum by (source_workload,destination_workload) ({expr})'
    return [
        Query('req_rate',wg(f'rate({rq}[1m])'),'workload'),
        Query('source_req_rate',wg(f'rate(istio_requests_total{{reporter="source",destination_workload_namespace="{ns}"}}[1m])'),'workload'),
        # Optional source-side active-request exporter. Istio's standard request
        # counter is completion-oriented, so this cannot be synthesized from it.
        Query('service_inflight',f'max by (destination_workload) (aurora_service_inflight{{namespace="{ns}"}})','workload'),
        Query('http_5xx_rate',wg(f'rate(istio_requests_total{{reporter="destination",destination_workload_namespace="{ns}",response_code=~"5.."}}[1m])'),'workload'),
        Query('http_503_rate',wg(f'rate(istio_requests_total{{reporter="destination",destination_workload_namespace="{ns}",response_code="503"}}[1m])'),'workload'),
        Query('http_4xx_rate',wg(f'rate(istio_requests_total{{reporter="destination",destination_workload_namespace="{ns}",response_code=~"4.."}}[1m])'),'workload'),
        Query('capacity_flag_rate',wg(f'rate(istio_requests_total{{reporter="destination",destination_workload_namespace="{ns}",response_flags=~".*(UO|OM).*"}}[1m])'),'workload'),
        Query('network_flag_rate',wg(f'rate(istio_requests_total{{reporter="destination",destination_workload_namespace="{ns}",response_flags=~".*(UF|UC).*"}}[1m])'),'workload'),
        Query('source_capacity_flag_rate',wg(f'rate(istio_requests_total{{reporter="source",destination_workload_namespace="{ns}",response_flags=~".*(UO|OM).*"}}[1m])'),'workload'),
        Query('source_network_flag_rate',wg(f'rate(istio_requests_total{{reporter="source",destination_workload_namespace="{ns}",response_flags=~".*(UF|UC).*"}}[1m])'),'workload'),
        Query('observed_istio_ts',f'max by (destination_workload) (timestamp({rq}))','workload'),
        Query('observed_source_ts',f'max by (destination_workload) (timestamp(istio_requests_total{{reporter="source",destination_workload_namespace="{ns}"}}))','workload'),
        *[Query(f'latency_p{p}_ms',f'histogram_quantile({p/100:.2f},sum by (destination_workload,le) (rate({rh}[1m])))','workload') for p in (50,90,99)],
        Query('cpu_cores',pod(f'rate(container_cpu_usage_seconds_total{{{sel}}}[1m])'),'pod'),
        Query('cpu_request_cores',pod(f'kube_pod_container_resource_requests{{namespace="{ns}",resource="cpu",unit="core",container!="",container!="istio-proxy"}}'),'pod'),
        Query('cpu_throttled',pod(f'rate(container_cpu_cfs_throttled_periods_total{{{sel}}}[1m])'),'pod'),
        Query('cpu_periods',pod(f'rate(container_cpu_cfs_periods_total{{{sel}}}[1m])'),'pod'),
        Query('mem_working_bytes',pod(f'container_memory_working_set_bytes{{{sel}}}'),'pod'),
        Query('mem_request_bytes',pod(f'kube_pod_container_resource_requests{{namespace="{ns}",resource="memory",unit="byte",container!="",container!="istio-proxy"}}'),'pod'),
        Query('mem_limit_bytes',pod(f'kube_pod_container_resource_limits{{namespace="{ns}",resource="memory",unit="byte",container!="",container!="istio-proxy"}}'),'pod'),
        Query('net_rx_bytes_s',pod(f'rate(container_network_receive_bytes_total{{namespace="{ns}",pod!=""}}[1m])'),'pod'),
        Query('net_tx_bytes_s',pod(f'rate(container_network_transmit_bytes_total{{namespace="{ns}",pod!=""}}[1m])'),'pod'),
        Query('fs_read_bytes_s',pod(f'rate(container_fs_reads_bytes_total{{{sel}}}[1m])'),'pod'),
        Query('fs_write_bytes_s',pod(f'rate(container_fs_writes_bytes_total{{{sel}}}[1m])'),'pod'),
        Query('oom_rate',pod(f'rate(container_oom_events_total{{{sel}}}[1m])'),'pod'),
        Query('restart_rate',pod(f'rate(kube_pod_container_status_restarts_total{{namespace="{ns}",container!="istio-proxy"}}[1m])'),'pod'),
        Query('call_rate',egroup(f'rate({erq}[1m])'),'edge'),
        Query('source_call_rate',egroup(f'rate(istio_requests_total{{{src_edge}}}[1m])'),'edge'),
        Query('error_5xx_rate',egroup(f'rate(istio_requests_total{{{edge},response_code=~"5.."}}[1m])'),'edge'),
        Query('edge_latency_p90_ms',f'histogram_quantile(0.90,sum by (source_workload,destination_workload,le) (rate({erh}[1m])))','edge'),
    ]

# Istio's default histogram buckets, in milliseconds. Used to pick a bucket
# boundary at or below the SLO deadline: the deadline itself is usually not a
# boundary, and querying a non-existent ``le`` returns an empty series, which
# would otherwise make every request look missed.
ISTIO_BUCKETS_MS = (0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000,
                    2500, 5000, 10000, 20000, 30000, 60000)


def _largest_bucket_at_most(budget_ms):
    """Largest Istio bucket boundary <= ``budget_ms`` (0.5 if none)."""
    ok = [b for b in ISTIO_BUCKETS_MS if b <= float(budget_ms)]
    return max(ok) if ok else ISTIO_BUCKETS_MS[0]


def _samples(item):
    out=[]
    for ts,x in item.get('values',[]):
        try:
            x=float(x)
            if math.isfinite(x) and x>=0:out.append((float(ts),x))
        except (TypeError,ValueError):pass
    return out

def _index(ts,grid,step):
    if not grid:return None
    j=int(round((ts-grid[0])/step))
    if 0<=j<len(grid) and abs(grid[j]-ts)<=step*.48:return j
    return None

class TelemetryCollector:
    def __init__(self,cfg,client=None):
        self.cfg=cfg
        self.client=client or Prometheus(cfg.prometheus_url,cfg.telemetry_timeout_s)
        self.last_query_timings={}
        # Query-result cache: preserves the full model history while fetching
        # only new 5s bins (+ a short overlap) after the first control cycle.
        self._cache={}
        self._cache_end=None
        self._cache_run=None
    def _query(self,q,start,end):
        begin=time.perf_counter()
        try:
            value=self.client.range(q.expr,start,end,self.cfg.telemetry_step_s)
            return q,value,None,time.perf_counter()-begin
        except Exception as ex:
            return q,[],str(ex),time.perf_counter()-begin
    def read_slo(self,now):
        """User-facing deadline state, from the configured source.

        ``gateway`` prefers the independent aurora_slo_* counters, which count an
        eligible request exactly once -- at completion before the deadline, or at
        the deadline if it is still outstanding. ``istio`` derives the same
        cohort from Istio's request histogram; that is a completion-side
        approximation and is used only when the controller does not own the
        ingress.

        A missing or stale source is ``unknown``, never ``met``.
        """
        if self.cfg.slo_source == 'gateway':
            return self._read_slo_gateway(now)
        if self.cfg.slo_source == 'istio':
            return self._read_slo_istio(now)
        state = self._read_slo_gateway(now)
        if state.valid(now, self.cfg.max_slo_age_s, self.cfg.min_slo_eligible):
            return state
        return self._read_slo_istio(now)

    def _read_slo_istio(self, now):
        """Derive the SLO cohort from Istio's own request telemetry.

        eligible = completed requests to the frontend workload
        missed   = those that took longer than the budget
        capacity = those the sidecar attributed to overflow / overload

        The window is evaluated at ``now`` so ``SLOState.valid`` compares like
        with like; a request still in flight is invisible here, which is why the
        gateway counter is preferred whenever it exists.
        """
        w = int(self.cfg.slo_window_s)
        ns = self.cfg.namespace
        front = self.cfg.front_workload
        budget = self.cfg.slo_budget_ms
        common = (f'reporter="destination",destination_workload_namespace="{ns}",'
                  f'destination_workload="{front}"')
        # Istio's histogram has fixed bucket boundaries (0.5,1,5,10,25,50,100,
        # 250,500,...). A deadline of 200 ms is not a boundary, so querying
        # le="200" returns nothing and every request looks missed. Use the
        # largest boundary at or below the deadline: conservative (it can only
        # understate the share of requests inside the budget), never optimistic.
        boundary = _largest_bucket_at_most(budget)
        expr = {
            'count': f'sum(increase(istio_request_duration_milliseconds_count{{{common}}}[{w}s]))',
            'fast': (f'sum(increase(istio_request_duration_milliseconds_bucket'
                     f'{{{common},le="{boundary}"}}[{w}s]))'),
            'capacity': (f'sum(increase(istio_requests_total{{{common},'
                      f'response_flags=~".*(UO|OM).*"}}[{w}s]))'),
        }

        def one(key, query):
            try:
                data = self.client.instant(query, now)
                val = float(data[0]['value'][1]) if data else None
                return val if val is not None and math.isfinite(val) and val >= 0 else None
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=min(3, self.cfg.telemetry_workers)) as pool:
            r = dict(pool.map(lambda kv: (kv[0], one(*kv)), expr.items()))
        total, fast, capacity = r.get('count'), r.get('fast'), r.get('capacity')
        if total is None or total < self.cfg.min_slo_eligible:
            return SLOState(None, None, None, None, 0, w)
        fast = min(total, fast if fast is not None else 0.0)
        capacity = min(total, capacity or 0.0)
        return SLOState(total, total - fast, capacity, 0.0, now, w)

    def _read_slo_gateway(self, now):
        """Independent gateway counters; missing samples fail closed."""
        w=int(self.cfg.slo_window_s)
        ns=self.cfg.namespace
        filt=f'namespace="{ns}"'
        expr={
            'eligible':f'sum(increase(aurora_slo_eligible_total{{{filt}}}[{w}s]))',
            'missed':f'sum(increase(aurora_slo_deadline_violations_total{{{filt}}}[{w}s]))',
            'capacity':f'sum(increase(aurora_request_failures_total{{{filt},cause="capacity"}}[{w}s]))',
            'unresolved':f'sum(aurora_requests_inflight{{{filt}}})',
        }
        expr['heartbeat']=f'max(aurora_observer_heartbeat_seconds{{{filt}}})'
        def one(key,query):
            try:
                data=self.client.instant(query,now)
                val=float(data[0]['value'][1]) if data else None
                return key, val if val is not None and math.isfinite(val) else None
            except Exception:
                return key,None
        # Five queries concurrently: an unavailable gateway must not push the
        # control cycle past the configured interval with sequential timeouts.
        with ThreadPoolExecutor(max_workers=min(5,self.cfg.telemetry_workers)) as pool:
            r=dict(pool.map(lambda kv:one(*kv),expr.items()))
        hb=r.get('heartbeat')
        if hb is None or now-hb>self.cfg.max_slo_age_s or hb>now+5:
            return SLOState(None,None,None,None,0,w)
        return SLOState(r['eligible'],r['missed'],r['capacity'],r['unresolved'],hb,w)

    def collect(self,run_id,states,now=None,labels=None):
        """states dict Deployment -> (Target, ReplicaState, pod_names)."""
        end=(math.floor((now or time.time())/self.cfg.telemetry_step_s)*self.cfg.telemetry_step_s)
        T=self.cfg.history_points;step=self.cfg.telemetry_step_s
        grid=[end-(T-1-j)*step for j in range(T)]
        if self._cache_run != run_id or (self._cache_end is not None and
               (end<self._cache_end or end-self._cache_end>T*step)):
            self._cache.clear();self._cache_end=None
        self._cache_run=run_id
        # Prometheus rate() automatically reads its own [1m] underlying samples;
        # query_range need not refetch those evaluations. Re-read 2 bins to
        # correct for scrape/Mimir ingestion latency. Cold start fetches all T.
        start=(grid[0] if self._cache_end is None else
               max(grid[0],self._cache_end-2*step))
        podmap={p:name for name,(_,_,pods) in states.items() for p in pods}
        workload_map={tg.istio_workload:name for name,(tg,_,_) in states.items()}
        services=sorted(states)
        result={};errors=[]
        with ThreadPoolExecutor(max_workers=self.cfg.telemetry_workers) as pool:
            fs=[pool.submit(self._query,q,start,end) for q in queries(self.cfg.namespace)]
            for f in as_completed(fs):
                q,val,err,secs=f.result()
                self.last_query_timings[q.key]=secs
                if err:errors.append(f'{q.key}: {err}')
                if err:
                    # Do not reuse the previous cycle as a fabricated current
                    # measurement. Previously cached timestamps remain present
                    # only at their original timestamps and will age out.
                    source=self._cache.get(q.key,{})
                else:
                    source=self._cache.setdefault(q.key,{})
                    for item in val:
                        labelset=tuple(sorted((item.get('metric') or {}).items()))
                        samples=source.setdefault(labelset,{})
                        for ts,value in _samples(item):samples[ts]=value
                    for labelset in list(source):
                        series=source[labelset]
                        for ts in list(series):
                            if ts < grid[0]-step:series.pop(ts,None)
                        if not series:source.pop(labelset,None)
                result[q.key]=(q,[{'metric':dict(labelset),
                                   'values':sorted(points.items())}
                                  for labelset,points in source.items()])
        self._cache_end=end
        if len(errors)>len(result)//2:
            raise RuntimeError('most Mimir metric families failed: '+';'.join(errors[:5]))
        node_raw={s:[{} for _ in grid] for s in services}
        edge_raw={}
        for key,(q,series) in result.items():
            for item in series:
                meta=item.get('metric') or {}
                if q.scope=='pod':
                    svc=podmap.get(meta.get('pod'))
                elif q.scope=='workload':
                    svc=workload_map.get(meta.get('destination_workload'))
                else:svc=None
                if q.scope in ('pod','workload'):
                    if svc is None:continue
                    for ts,x in _samples(item):
                        j=_index(ts,grid,step)
                        if j is None:continue
                        raw=node_raw[svc][j]
                        if q.scope=='pod':raw[key]=raw.get(key,0.0)+x
                        else:raw[key]=x
                elif q.scope=='edge':
                    a=workload_map.get(meta.get('source_workload'))
                    b=workload_map.get(meta.get('destination_workload'))
                    if not a or not b or a==b:continue
                    rawseq=edge_raw.setdefault((a,b),[{} for _ in grid])
                    subkey=('latency_p90_ms' if key=='edge_latency_p90_ms' else
                            'call_rate' if key=='source_call_rate' else key)
                    for ts,x in _samples(item):
                        j=_index(ts,grid,step)
                        if j is None:continue
                        # For source/destination reporters, union evidence via
                        # max rather than adding the same completed call twice.
                        rawseq[j][subkey]=max(rawseq[j].get(subkey,0.),x)
        N=len(services); F=len(NODE_FEATURES)
        x=np.zeros((N,T,F),np.float32); mask=np.zeros_like(x)
        for i,s in enumerate(services):
            ready=states[s][1].ready
            for j,raw in enumerate(node_raw[s]):
                # Prometheus instant/range results can remain queryable after a
                # source stops scraping. Verify raw Istio-counter timestamps.
                dest_ts=raw.pop('observed_istio_ts',None)
                src_ts=raw.pop('observed_source_ts',None)
                if dest_ts is None or grid[j]-dest_ts>self.cfg.max_metric_age_s:
                    for k in ('req_rate','http_5xx_rate','http_503_rate','http_4xx_rate',
                              'latency_p50_ms','latency_p90_ms','latency_p99_ms',
                              'capacity_flag_rate','network_flag_rate'):
                        raw.pop(k,None)
                if src_ts is None or grid[j]-src_ts>self.cfg.max_metric_age_s:
                    for k in ('source_req_rate','source_capacity_flag_rate',
                              'source_network_flag_rate'):
                        raw.pop(k,None)
                t=raw.get('cpu_throttled'); d=raw.get('cpu_periods')
                if t is not None and d is not None and d>0:
                    raw['cpu_throttle_frac']=min(1.0,max(0.0,t/d))
                v,m=node_vector(raw,ready)
                x[i,j],mask[i,j]=v,m
        # Sparse, risk-aware edge retention. Top-rate-only selection discarded
        # low-volume/high-tail-latency calls, which may dominate user impact.
        # Retain a union of rate, latency burden and error evidence per caller.
        # Complexity O(E_observed log E_observed) selection, O(E_kept) inference.
        by_caller={}
        for (a,b),rs in edge_raw.items():
            rate=max((r.get('call_rate',0.) for r in rs),default=0.)
            err=max((r.get('error_5xx_rate',0.) for r in rs),default=0.)
            lat=max((r.get('latency_p90_ms',0.) for r in rs),default=0.)
            if rate>0. or err>0.:
                by_caller.setdefault(a,[]).append((b,rate,rate*math.log1p(lat),err))
        selected=[]
        for a,choices in by_caller.items():
            k=self.cfg.max_neighbors_per_caller
            selected_b=set()
            metrics=(1,2,3)
            # Round robin across ranking criteria supports sparse graphs in
            # which traffic rate and criticality are not the same signal.
            ranked=[sorted(choices,key=lambda item:item[j],reverse=True) for j in metrics]
            counters=[0,0,0]
            while len(selected_b)<min(k,len(choices)):
                changed=False
                for j,items in enumerate(ranked):
                    while counters[j]<len(items) and items[counters[j]][0] in selected_b:
                        counters[j]+=1
                    if counters[j]<len(items) and len(selected_b)<k:
                        selected_b.add(items[counters[j]][0]);counters[j]+=1;changed=True
                if not changed:break
            meta={item[0]:item for item in choices}
            selected.extend((a,b,meta[b][1]) for b in selected_b)
        if len(selected)>self.cfg.max_edges:
            selected=sorted(selected,key=lambda item:item[2],reverse=True)[:self.cfg.max_edges]
        selected.sort(key=lambda item:(item[0],item[1]))
        idx={s:i for i,s in enumerate(services)}
        E=len(selected)
        ei=np.zeros((2,E),dtype=np.int64)
        ex=np.zeros((E,T,len(EDGE_FEATURES)),np.float32); em=np.zeros_like(ex)
        for e,(a,b,_) in enumerate(selected):
            ei[:,e]=[idx[a],idx[b]]
            for j,r in enumerate(edge_raw[(a,b)]):
                v,m=edge_vector(r);ex[e,j],em[e,j]=v,m
        snap=GraphSnapshot(
            deployment_id=self.cfg.namespace,run_id=run_id,timestamp=end,
            services=services,node_x=x,node_mask=mask,edge_index=ei,
            edge_x=ex,edge_mask=em,
            replicas={s:states[s][1] for s in services},
            slo=self.read_slo(end),**(labels or {}))
        return snap.validate(),{'queries':len(result),'range_start':start,
                                'range_end':end,'incremental':start>grid[0],
                                'errors':errors,'edges_before':len(edge_raw),'edges_retained':E,
                                'max_metric_query_s':max(self.last_query_timings.values(),default=0.0)}