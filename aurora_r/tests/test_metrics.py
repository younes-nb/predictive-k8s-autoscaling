from aurora_r.metrics import TelemetryCollector
from aurora_r.schema import SLOState,NODE_FEATURES,ReplicaState

class FakeCollector(TelemetryCollector):
    def read_slo(self,now):return SLOState(100.,15.,3.,1.,now,60)
    def _query(self,q,start,end):
        ts=end
        def row(labels,value):return {'metric':labels,'values':[[ts,str(value)]]}
        if q.key=='req_rate':
            data=[row({'destination_workload':s},5.) for s in ('frontend','down1','down2','down3')]
        elif q.key=='observed_istio_ts':
            data=[row({'destination_workload':s},ts) for s in ('frontend','down1','down2','down3')]
        elif q.key=='call_rate':
            data=[row({'source_workload':'frontend','destination_workload':s},5.+i)
                  for i,s in enumerate(('down1','down2','down3'))]
        else:data=[]
        return q,data,None,0.

def test_collector_preserves_directed_sparse_edges_and_missing_masks(configuration):
    from aurora_r.config import Target
    configuration.targets=[Target(s,s,1,8,s) for s in ('frontend','down1','down2','down3')]
    configuration.max_neighbors_per_caller=2
    collector=FakeCollector(configuration)
    st={s:(configuration.target_map()[s],ReplicaState(
        2,2,2,2,2,0,0,True,2),[]) for s in configuration.target_map()}
    ss,info=collector.collect('test',st,now=1000.)
    assert ss.edge_index.shape[1]==2
    assert info['edges_before']==3
    assert ss.node_mask[ss.services.index('frontend'),-1,NODE_FEATURES.index('req_rate')]==1.
    assert ss.node_mask[ss.services.index('frontend'),-1,NODE_FEATURES.index('cpu_request_cores')]==0.
    assert all(ss.services[a]=='frontend' for a in ss.edge_index[0])

def test_no_istio_timestamp_is_not_a_zero_request_rate(configuration):
    class NoTimestamp(FakeCollector):
        def _query(self,q,start,end):
            if q.key=='observed_istio_ts':return q,[],None,0.
            return super()._query(q,start,end)
    c=NoTimestamp(configuration)
    t=configuration.targets[0]
    state={t.deployment:(t,ReplicaState(2,2,2,2,2,0,0,True,2),[])}
    snap,_=c.collect('test',state,now=1000.)
    assert snap.node_mask[0,-1,NODE_FEATURES.index('req_rate')]==0.

def test_metric_cache_fetches_only_recent_bins_and_run_boundary_resets(configuration):
    t=configuration.targets[0]
    state={t.deployment:(t,ReplicaState(2,2,2,2,2,0,0,True,2),[])}
    collector=FakeCollector(configuration)
    first,info1=collector.collect('runA',state,now=1000.)
    second,info2=collector.collect('runA',state,now=1015.)
    assert info1['incremental'] is False
    assert info2['incremental'] is True
    assert info2['range_start']>=990. and info2['range_start']<1015.
    assert second.node_mask[0,-1,NODE_FEATURES.index('req_rate')]==1
    third,info3=collector.collect('runB',state,now=1030.)
    assert info3['incremental'] is False

def test_slo_fallback_uses_a_real_bucket_boundary():
    # Istio's histogram has no le="200" series; a naive query would return an
    # empty series and every request would look missed.
    from aurora_r.metrics import _largest_bucket_at_most
    assert _largest_bucket_at_most(200) == 100
    assert _largest_bucket_at_most(100) == 100
    assert _largest_bucket_at_most(250) == 250
    assert _largest_bucket_at_most(0.2) == 0.5

def test_istio_slo_counts_fast_requests_below_the_deadline(configuration):
    class FakeSloClient:
        def instant(self, query, at=None):
            # count, fast (le=100 bucket), capacity
            if '_count{' in query:
                return [{'value': [at, '1000']}]
            if '_bucket{' in query:
                assert 'le="100"' in query, query
                return [{'value': [at, '940']}]
            if 'response_flags' in query:
                return [{'value': [at, '0']}]
            return []
    configuration.slo_source = 'istio'
    configuration.front_workload = 'frontend'
    configuration.slo_budget_ms = 200.0
    c = TelemetryCollector(configuration, client=FakeSloClient())
    slo = c.read_slo(1000.0)
    assert slo.eligible == 1000.0
    assert abs(slo.miss_rate - 0.06) < 1e-9
