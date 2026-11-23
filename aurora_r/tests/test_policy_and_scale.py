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