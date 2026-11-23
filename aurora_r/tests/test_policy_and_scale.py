import torch
from conftest import snapshot
from aurora_r.policy import DecisionPolicy

class DummyGraph:
    def eval(self):return self

class DummyResponse:
    def eval(self):return self

def output(root=1.0,remed=1.0,scalein=.9,n=1):
    return {'valid':torch.ones(n,dtype=torch.bool),
            'embedding':torch.zeros((n,8)),
            'root':torch.full((n,),root),
            'remediable':torch.full((n,),remed),
            'scalein_safe':torch.full((n,),scalein)}

def test_remediability_veto_and_root_gate(configuration):
    snap=snapshot(missed=30)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    p.infer=lambda s:output(root=.99,remed=.01)
    dec,reason,_=p.choose(snap,now=snap.timestamp)
    assert dec is None and 'no appropriately supported' in reason
    p.infer=lambda s:output(root=.01,remed=.99)
    dec,reason,_=p.choose(snap,now=snap.timestamp)
    assert dec is None

def test_robust_effect_target_selection(configuration):
    s=snapshot(missed=40)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    p.infer=lambda s:output(root=.99,remed=.99)
    p._batch_eval=lambda snap,out,i,targets,now:([[.15,0.] for _ in targets],
                                              [[.002,.002] for _ in targets])
    result,_,_=p.choose(s,now=s.timestamp)
    assert result is not None and result.direction=='out'
    assert result.service=='frontend' and result.target in (4,5,6)

def test_ready_pending_suppresses_scaleout(configuration):
    s=snapshot(missed=40)
    s.replicas['frontend'].ready=2
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    p.infer=lambda s:output(root=.99,remed=.99)
    p._batch_eval=lambda snap,out,i,targets,now:([[.1,0.] for _ in targets],[[0,0] for _ in targets])
    result,reason,_=p.choose(s,now=s.timestamp)
    assert result is None

def test_unknown_slo_cannot_trigger_scale_in(configuration):
    s=snapshot(missed=0)
    s.slo.eligible=None
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    result,reason,_=p.choose(s,now=s.timestamp)
    assert result is None and 'ingress SLO' in reason

def test_unvalidated_checkpoint_blocks_live_metadata(configuration):
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{'intervention_validated':True})
    assert not p.eligible_for_live()


def test_zero_misses_with_small_sample_is_inconclusive_not_scale_out(configuration):
    s=snapshot(missed=0)
    # Zero missed requests in a small but valid SLO sample is insufficient for
    # safe scale-in; it is NOT evidence that the application is degraded.
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    p.infer=lambda snap:output(root=.99,remed=.99,scalein=.99)
    p._batch_eval=lambda *args,**kwargs: (_ for _ in ()).throw(
        AssertionError('no action should be evaluated on inconclusive SLO'))
    result,reason,_=p.choose(s,now=s.timestamp)
    assert result is None and 'inconclusive' in reason

def test_timing_does_not_carry_over_when_slo_missing(configuration):
    s=snapshot(missed=10)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    p.last_diag_ms=123.
    s.slo.eligible=None
    result,_,_=p.choose(s,now=s.timestamp)
    assert result is None and p.last_diag_ms==0

def _pressure_snapshot(configuration, cpu_util, throttle=0.0, missed=40., eligible=100.):
    from aurora_r.schema import NODE_FEATURES
    s=snapshot(missed=missed, eligible=eligible)
    iu=NODE_FEATURES.index('cpu_request_util')
    it=NODE_FEATURES.index('cpu_throttle_frac')
    s.node_x[0,-1,iu]=cpu_util
    s.node_x[0,-1,it]=throttle
    s.node_mask[0,-1,iu]=1.
    s.node_mask[0,-1,it]=1.
    return s

def test_rule_policy_scales_out_on_measured_cpu_request_pressure(configuration):
    configuration.policy_source='rule'
    s=_pressure_snapshot(configuration, 1.4, throttle=.2, missed=40.)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    dec,why,_=p.choose(s,now=s.timestamp)
    assert dec is not None and dec.direction=='out', why
    assert dec.service=='frontend' and dec.target==4

def test_rule_policy_holds_when_not_pressured_even_if_slo_missed(configuration):
    configuration.policy_source='rule'
    s=_pressure_snapshot(configuration, 0.2, throttle=0.0, missed=40.)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    dec,why,_=p.choose(s,now=s.timestamp)
    assert dec is None and 'no service met the measured-pressure evidence' in why

def test_rule_policy_scale_in_needs_sustained_health(configuration):
    configuration.policy_source='rule'
    configuration.scale_in_health_s=300
    # A large clean sample is needed before zero misses is evidence of health;
    # a Wilson bound on 100 requests cannot certify a 1% SLO.
    s=_pressure_snapshot(configuration, 0.1, throttle=0.0, missed=0., eligible=5000.)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    dec,why,_=p.choose(s,now=s.timestamp)
    assert dec is None and 'healthy window has not elapsed' in why
    # Each cycle builds a fresh snapshot, so refresh the sampling time too.
    s.slo.sampled_at=s.timestamp+301
    dec,why,_=p.choose(s,now=s.timestamp+301)
    assert dec is not None and dec.direction=='in', why
    assert dec.target==2

def test_rule_policy_never_reads_slo_as_a_bottleneck_signal(configuration):
    configuration.policy_source='rule'
    # SLO violated but CPU cool -> no scale-out: the rule keys on pressure only.
    s=_pressure_snapshot(configuration, 0.1, throttle=0.0, missed=90.)
    p=DecisionPolicy(configuration,DummyGraph(),DummyResponse(),{})
    dec,why,_=p.choose(s,now=s.timestamp)
    assert dec is None
