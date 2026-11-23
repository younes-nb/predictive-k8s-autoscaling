import time
from conftest import snapshot
from aurora_r.controller import AuroraController
from aurora_r.schema import ReplicaState
from aurora_r.journal import ActionJournal
from aurora_r.adapter import OnlineAdapter
from aurora_r.model import SparseTemporalGraph,ActionResponseModel,checkpoint_dict
import torch
import pytest

class FakeKube:
    def __init__(self,config):self.cfg=config
    def read_states(self):
        t=self.cfg.targets[0]
        return {t.deployment:(t,ReplicaState(3,3,3,3,3,0,0,True,3),[])}
    def assert_single_writer(self):return True
    def write_scale(self,*args):raise AssertionError('shadow mode must never patch')

class FakeCollector:
    def collect(self,run_id,states,now=None):
        s=snapshot(timestamp=now or time.time())
        return s,{'synthetic':True}

def test_shadow_mode_runs_one_cycle_without_scale_writer(configuration):
    g=SparseTemporalGraph(dim=12);r=ActionResponseModel(dim=12)
    torch.save(checkpoint_dict(g,r,{'untrained':True,'outcome_target':'delta_rates'}),configuration.checkpoint)
    c=AuroraController(configuration,kube=FakeKube(configuration),collector=FakeCollector())
    result=c.cycle()
    assert result['mode']=='shadow'
    assert result['write']=='shadow'
    assert result['timing']['complete_cycle_ms']>=0
    assert 'graph_inference_ms' in result['timing']
    c.journal.close()

def test_live_mode_refuses_unvalidated_models_before_api_calls(configuration):
    g=SparseTemporalGraph(dim=12);r=ActionResponseModel(dim=12)
    torch.save(checkpoint_dict(g,r,{'untrained':True,'outcome_target':'delta_rates'}),configuration.checkpoint)
    configuration.mode='live';configuration.experimental_actuation_ack=True
    with pytest.raises(RuntimeError,match='not intervention-validated'):
        AuroraController(configuration,kube=FakeKube(configuration),collector=FakeCollector())

def test_settlement_waits_for_post_ready_metric_window(configuration):
    cfg=type('Config',(),{'settle_window_s':60,'slo_window_s':120,
         'min_settle_after_ready_s':65,'telemetry_step_s':5,
         'max_rollout_s':900,'max_slo_age_s':20,'min_slo_eligible':20,
         'max_workload_shift_for_learning':.20})()
    c=AuroraController.__new__(AuroraController)
    c.cfg=cfg
    c.journal=ActionJournal('/tmp/aurora-test-controller-settle-'+str(time.time_ns())+'.sqlite3')
    c.adapter=OnlineAdapter('/tmp/aurora-test-adapter-'+str(time.time_ns())+'.json')
    target='frontend'
    s=snapshot(timestamp=1010)
    aid=c.journal.create(target,3,5,{'slo_miss_rate':.4,
         'slo_capacity_failure_rate':0.,'service_req_rate':.64872,
         'direction':'out','prediction':{'mu_miss':.2,'mu_fail':0.}},now=1000)
    c.journal.update(aid,'applied',now=1000)
    not_ready=ReplicaState(5,3,3,3,3,0,2,False,3)
    c._settle({target:(None,not_ready,[])},s,now=1010)
    assert c.journal.get(aid)['status']=='applied'
    ready=ReplicaState(5,5,5,5,5,0,0,True,5)
    s.timestamp=1020;s.slo.sampled_at=1020
    c._settle({target:(None,ready,[])},s,now=1020)
    assert c.journal.get(aid)['status']=='ready'
    s.timestamp=1090;s.slo.sampled_at=1090
    c._settle({target:(None,ready,[])},s,now=1090)
    assert c.journal.get(aid)['status']=='ready'  # SLO interval crosses readiness
    s.timestamp=1160;s.slo.sampled_at=1160
    c._settle({target:(None,ready,[])},s,now=1160)
    assert c.journal.get(aid)['status']=='settled'
    c.journal.close()