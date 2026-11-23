from aurora_r.request_lifecycle import RequestObserver,classify_failure
from prometheus_client import CollectorRegistry
from aurora_r.journal import ActionJournal
from aurora_r.schema import SLOState

def test_unfinished_request_counts_deadline_once_and_late_failure_separately():
    o=RequestObserver(.2,'research',clock=lambda:0,registry=CollectorRegistry())
    x=o.start('browse',at=0)
    assert o.sweep(at=.19)==0
    assert o.sweep(at=.20)==1
    assert o.sweep(at=.21)==0
    assert o.finish(x,cause='capacity',at=.5)
    assert o.event_counts[('eligible','browse',None)]==1
    assert o.event_counts[('missed','browse',None)]==1
    assert ('failure','browse','capacity') not in o.event_counts
    assert o.event_counts[('late','browse','capacity')]==1
    assert o.event_counts[('inflight','browse',None)]==0
    assert not o.finish(x,at=.7)

def test_finished_request_counts_failure_with_eligible_same_event():
    o=RequestObserver(.2,'research',clock=lambda:0,registry=CollectorRegistry())
    x=o.start(at=0);o.finish(x,cause='capacity',at=.1)
    assert o.event_counts[('eligible','default',None)]==1
    assert o.event_counts[('failure','default','capacity')]==1
    assert ('missed','default',None) not in o.event_counts

def test_status_alone_never_capacity_label():
    assert classify_failure(503)=='unknown'
    assert classify_failure(429)=='unknown'
    assert classify_failure(500)=='unknown'
    assert classify_failure(503,{'x-envoy-response-flags':'UO'})=='capacity'
    assert classify_failure(503,{'x-envoy-response-flags':'OM'})=='capacity'
    assert classify_failure(400,{'x-application-rejection':'business'})=='business'
    assert classify_failure(500,application_error=True)=='application'
    assert classify_failure(504,transport='connection')=='network'

def test_slo_no_unobserved_completion_is_not_healthy():
    assert not SLOState(None,None,None,None,1000.,60).valid(1000,20,20)
    assert not SLOState(19.,0.,0.,0.,1000.,60).valid(1000,20,20)
    assert not SLOState(100.,30.,0.,0.,950.,60).valid(1000,20,20)
    assert SLOState(100.,30.,0.,0.,1000.,60).miss_rate==.3

def test_journal_retains_unresolved_and_resets_ready_at(tmp_path):
    f=str(tmp_path/'j.db');j=ActionJournal(f)
    aid=j.create('frontend',3,5,{'before':3},now=100)
    j.reconcile_created({'frontend':type('Rep',(),{'spec':5})()})
    assert j.get(aid)['status']=='applied'
    j.update(aid,'ready',now=110)
    assert j.get(aid)['ready_at']==110
    j.update(aid,'applied',now=120,detail='readiness regressed')
    assert j.get(aid)['ready_at'] is None
    j.close()
    j=ActionJournal(f)
    assert len(j.unresolved())==1
    j.update(aid,'inconclusive',detail='api uncertain')
    try:j.operator_resolve(aid,'')
    except ValueError:pass
    else:assert False
    j.operator_resolve(aid,'manual cluster inspection confirms current count')
    assert len(j.unresolved())==0
    j.close()

def test_zero_failure_and_zero_deadline_counters_are_registered():
    from prometheus_client import generate_latest
    reg=CollectorRegistry()
    o=RequestObserver(.2,'research',registry=reg,clock=lambda:0)
    o.ensure_operation('browse')
    text=generate_latest(reg).decode()
    assert 'aurora_slo_deadline_violations_total{namespace="research",operation="browse"} 0.0' in text
    assert 'aurora_request_failures_total{cause="capacity",namespace="research",operation="browse"} 0.0' in text
    assert 'aurora_requests_inflight{namespace="research",operation="browse"} 0.0' in text

def test_unknown_errors_are_counted_but_never_as_capacity():
    from prometheus_client import CollectorRegistry
    o=RequestObserver(.2,'research',registry=CollectorRegistry(),clock=lambda:0)
    x=o.start(at=0)
    assert classify_failure(status=200)=='none'
    assert classify_failure(status=500)=='unknown'
    o.finish(x,at=.1,cause=classify_failure(status=500))
    assert o.event_counts[('failure','default','unknown')]==1
    assert ('failure','default','capacity') not in o.event_counts


def test_gateway_eligibility_policy_is_route_based_not_http_status():
    from aurora_r.ingress_gateway import eligible_path
    assert eligible_path('/api/order',('/api/',),('/api/health',))
    assert not eligible_path('/api/health',('/api/',),('/api/health',))
    assert not eligible_path('/static/file',('/api/',),())
    assert eligible_path('/api/checkout',(),('/static',))