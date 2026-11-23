import numpy as np
import pytest
from aurora_r.schema import GraphSnapshot,ReplicaState,SLOState,NODE_FEATURES,EDGE_FEATURES
from aurora_r.config import Config,Target

@pytest.fixture
def configuration(tmp_path):
    return Config(namespace='research',targets=[Target('frontend','frontend',1,8,'front')], front_workload='frontend',
                  prometheus_url='http://prom',checkpoint=str(tmp_path/'model.pt'),
                  journal_path=str(tmp_path/'actions.sqlite3'),adapter_path=str(tmp_path/'adapter.json'),
                  scale_in_health_s=0,min_slo_eligible=20)

def snapshot(deployment='a',services=('frontend',),T=5,nodes=True,missed=40.,eligible=100.,timestamp=1_000.):
    n=len(services);f=len(NODE_FEATURES);e=0
    x=np.zeros((n,T,f),dtype=np.float32);m=np.zeros_like(x)
    if nodes:
        m[:,:,:]=1.;x[:,:,:]=.5
    r={s:ReplicaState(spec=3,ready=3,running=3,available=3,scheduled=3,
                      terminating=0,request_pending=0,rollout_complete=True,ready_serving=3) for s in services}
    return GraphSnapshot(deployment_id=deployment,run_id='run1',timestamp=timestamp,
         services=list(services),node_x=x,node_mask=m,edge_index=np.zeros((2,0),np.int64),
         edge_x=np.zeros((e,T,len(EDGE_FEATURES)),np.float32),
         edge_mask=np.zeros((e,T,len(EDGE_FEATURES)),np.float32),replicas=r,
         slo=SLOState(eligible,missed,0.,0.,timestamp,60),
         root_labels={s:float(i==0) for i,s in enumerate(services)},
         affected_labels={s:float(i>0) for i,s in enumerate(services)},
         remedy_labels={s:float(i==0) for i,s in enumerate(services)},
         scalein_labels={s:float(i>0) for i,s in enumerate(services)})