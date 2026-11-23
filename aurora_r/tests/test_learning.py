import numpy as np
import torch
from conftest import snapshot
from aurora_r.model import SparseTemporalGraph,load_checkpoint,tensors,action_vector
from aurora_r.schema import EDGE_FEATURES,Intervention
from aurora_r.train.train import train_all,action_effect
from aurora_r.policy import DecisionPolicy

def test_graph_sparse_variable_topology_and_missingness():
    n=128;T=5
    s=snapshot('deployment-large',tuple(f's{i}' for i in range(n)),T=T)
    src=np.arange(n-1,dtype=np.int64)
    s.edge_index=np.stack((src,src+1))
    s.edge_x=np.ones((n-1,T,len(EDGE_FEATURES)),np.float32)
    s.edge_mask=np.ones_like(s.edge_x)
    s.validate()
    g=SparseTemporalGraph(dim=16).eval()
    with torch.inference_mode():y=g(*tensors(s))
    assert y['root'].shape==(n,)
    assert y['remediable'].shape==(n,)
    assert y['embedding'].shape==(n,16)
    assert not any('s0' in k for k in g.state_dict())

def test_missing_node_never_becomes_valid():
    s=snapshot(nodes=False)
    out=SparseTemporalGraph(dim=16)(*tensors(s))
    assert not bool(out['valid'].any())

def test_action_vector_current_and_target_are_not_weak_replica_labels():
    s=snapshot()
    v=action_vector(s,'frontend',5,s.timestamp)
    assert v.shape[0]==10
    assert v[0]>0
    assert action_vector(s,'frontend',2,s.timestamp)[0]<0

def test_training_only_uses_evidence_and_heldout_split(tmp_path):
    a=snapshot('alpha'); b=snapshot('beta')
    ia=Intervention(a,'frontend',4,100,20,0,0,True,True,True,'matched')
    ib=Intervention(b,'frontend',2,100,55,1,0,True,True,True,'randomized')
    assert ia.eligible() and ib.eligible()
    assert torch.allclose(action_effect(ia),torch.tensor([-.2,0.]))
    out=tmp_path/'model.pt'
    meta=train_all([a,b],[ia,ib],{'beta'},str(out),graph_epochs=1,effect_epochs=1,dim=12)
    assert meta['training_deployments']==['alpha']
    assert meta['validation_deployments']==['beta']
    assert meta['heldout_interventions']==1
    assert not meta['intervention_validated']
    g,m,loaded=load_checkpoint(out)
    assert loaded['outcome_target']=='delta_rates'
    assert not DecisionPolicy(type('C',(),{'min_action_examples':1})(),g,m,loaded).eligible_for_live()

def test_unmatched_or_observational_interventions_cannot_train():
    a=snapshot()
    obs=Intervention(a,'frontend',4,100,10,0,0,True,True,False,'observational')
    assert not obs.eligible()
    obs.matched_control=True
    assert not obs.eligible()