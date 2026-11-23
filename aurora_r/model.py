"""Sparse O(T*(N+E)*D + (N+E)*D^2) temporal graph diagnosis.

No NxN adjacency matrix, random walk, all-pairs distance, or node-specific
weights. A fixed maximum number of observed caller neighbors protects latency.
Diagnosis and action-response supervision are intentionally distinct.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .schema import NODE_FEATURES, EDGE_FEATURES, DEGRADATION, FAILURE_CAUSES, SCHEMA_VERSION

MODEL_VERSION='aurora-r.model.v6'
GLOBAL_FIELDS=['deadline_miss_rate','capacity_failure_rate','inflight_over_eligible','slo_observed']
ACTION_FIELDS=[
    'delta_relative','target_relative','log_ready','pending_frac',
    'observed_miss_rate','observed_capacity_fail_rate',
    'log_service_rate','log_target','log_spec','recent_action_flag',
]

class MaskedTemporalGRU(nn.Module):
    def __init__(self,features,dim):
        super().__init__()
        self.proj=nn.Linear(2*features,dim)
        self.cell=nn.GRUCell(dim,dim)
        self.norm=nn.LayerNorm(dim)
    def forward(self,values,masks):
        # [B,T,F]; missing timesteps do NOT advance the recurrent state.
        b,t,_=values.shape
        h=values.new_zeros((b,self.cell.hidden_size))
        for ti in range(t):
            mk=masks[:,ti,:]
            inp=self.proj(torch.cat([values[:,ti,:]*mk,mk],dim=-1))
            nxt=self.cell(inp,h)
            available=mk.any(-1,keepdim=True)
            h=torch.where(available,nxt,h)
        return self.norm(h)

class SparseTemporalGraph(nn.Module):
    """Bidirectional directed-edge propagation, degree-normalized and O(E)."""
    def __init__(self,dim=64,layers=2):
        super().__init__()
        self.dim=dim
        # Training-only, robust per-feature statistics persisted in the state.
        # Frozen during inference so novel deployments never distort the encoder.
        self.register_buffer('node_center',torch.zeros(len(NODE_FEATURES)))
        self.register_buffer('node_scale',torch.ones(len(NODE_FEATURES)))
        self.register_buffer('edge_center',torch.zeros(len(EDGE_FEATURES)))
        self.register_buffer('edge_scale',torch.ones(len(EDGE_FEATURES)))
        self.node=MaskedTemporalGRU(len(NODE_FEATURES),dim)
        self.global_proj=nn.Linear(len(GLOBAL_FIELDS),dim)
        self.edge=MaskedTemporalGRU(len(EDGE_FEATURES),dim)
        self.forward_layers=nn.ModuleList(nn.Linear(dim*2,dim) for _ in range(layers))
        self.backward_layers=nn.ModuleList(nn.Linear(dim*2,dim) for _ in range(layers))
        self.merge=nn.ModuleList(nn.Linear(dim*3,dim) for _ in range(layers))
        self.norms=nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.root=nn.Linear(dim,1)
        self.affected=nn.Linear(dim,1)
        self.remediable=nn.Linear(dim,1)
        self.scalein_safe=nn.Linear(dim,1)
        self.degradation=nn.Linear(dim,len(DEGRADATION))
        self.failure=nn.Linear(dim,len(FAILURE_CAUSES))
        self.pretrain_reconstruction=nn.Linear(dim,len(NODE_FEATURES))
    def normalized_nodes(self,x):
        return ((x-self.node_center)/self.node_scale.clamp_min(.5)).clamp(-8,8)
    def normalized_edges(self,x):
        return ((x-self.edge_center)/self.edge_scale.clamp_min(.5)).clamp(-8,8)
    def forward(self,nodes,node_masks,edges,edge_features,edge_masks,global_context=None):
        n=nodes.shape[0]
        if n==0:raise ValueError('empty graph')
        h=self.node(self.normalized_nodes(nodes)*node_masks,node_masks)
        # Current observed end-to-end SLO context; no future resource values.
        if global_context is not None:
            h=h+self.global_proj(global_context.to(h.dtype)).reshape(1,-1)
        node_valid=node_masks.any(-1).any(-1)
        if edges.shape[1]:
            ef=self.edge(self.normalized_edges(edge_features)*edge_masks,edge_masks)
            src,dst=edges[0],edges[1]
            keep=node_valid[src]&node_valid[dst]&edge_masks.any((1,2))
            src,dst,ef=src[keep],dst[keep],ef[keep]
        else:
            src=dst=torch.empty((0,),dtype=torch.long,device=h.device)
            ef=torch.empty((0,h.shape[1]),device=h.device,dtype=h.dtype)
        for fw,bw,merge,norm in zip(self.forward_layers,self.backward_layers,self.merge,self.norms):
            fin=h.new_zeros(h.shape); bout=h.new_zeros(h.shape)
            d_in=h.new_zeros((n,1)); d_out=h.new_zeros((n,1))
            if src.numel():
                # caller -> callee and callee -> caller are not interchangeable
                fin.index_add_(0,dst,F.gelu(fw(torch.cat([h[src],ef],-1))))
                bout.index_add_(0,src,F.gelu(bw(torch.cat([h[dst],ef],-1))))
                ones=torch.ones((src.numel(),1),device=h.device,dtype=h.dtype)
                d_in.index_add_(0,dst,ones); d_out.index_add_(0,src,ones)
            h=norm(h+F.gelu(merge(torch.cat([h,fin/d_in.clamp_min(1),bout/d_out.clamp_min(1)],-1))))
            h=h*node_valid[:,None].float()
        return {
            'embedding':h,'valid':node_valid,
            'root':torch.sigmoid(self.root(h)).squeeze(-1),
            'affected':torch.sigmoid(self.affected(h)).squeeze(-1),
            'remediable':torch.sigmoid(self.remediable(h)).squeeze(-1),
            'scalein_safe':torch.sigmoid(self.scalein_safe(h)).squeeze(-1),
            'degradation':torch.sigmoid(self.degradation(h)),
            'failure':F.softmax(self.failure(h),dim=-1),
            'reconstruction':self.pretrain_reconstruction(h),
        }

class ActionResponseModel(nn.Module):
    """Candidate-conditioned signed changes in miss and capacity-failure rates.

    This estimates *effects of a replica intervention under observed workload*,
    not future resource utilization or traffic. Prediction needs true intervention
    training examples and confidence validation, not weak replica-count labels.
    """
    def __init__(self,dim=64,ensemble=3):
        super().__init__()
        self.ensemble=nn.ModuleList(nn.Sequential(
            nn.Linear(dim+len(ACTION_FIELDS),64),nn.GELU(),
            nn.Linear(64,32),nn.GELU(),nn.Linear(32,2)) for _ in range(ensemble))
    def forward(self,embedding,action_features):
        x=torch.cat([embedding,action_features],-1)
        return torch.stack([torch.tanh(net(x)) for net in self.ensemble],0) # K,B,2 signed delta
    def estimate(self,embedding,action_features,calibrated_residual=0.0):
        pred=self.forward(embedding,action_features)
        mu=pred.mean(0)
        epistemic=pred.std(0,unbiased=False)
        # Held-out residual covers unmodeled noise; never infer "zero uncertainty"
        # merely because separately initialized models agree.
        sigma=torch.sqrt(epistemic.square()+float(calibrated_residual)**2)
        return mu,sigma

def action_vector(snapshot,service,target,now,recent_action=False):
    """EXACT same constructor in training and live; no zero-filled scalars."""
    import math
    from .schema import NODE_FEATURES
    rep=snapshot.replicas[service]
    idx=snapshot.services.index(service)
    # Prefer destination-side observations, then source-side call observations.
    # Both are completion-side measurements, never a traffic forecast.
    service_rate=0.0
    for name in ('req_rate','source_req_rate'):
        qcol=NODE_FEATURES.index(name)
        if snapshot.node_mask[idx,-1,qcol]:
            service_rate=float(snapshot.node_x[idx,-1,qcol]);break
    valid=snapshot.slo.valid(now,10**9,1)
    miss=float(snapshot.slo.miss_rate or 0.0) if valid else 0.0
    fail=float(snapshot.slo.capacity_failure_rate or 0.0) if valid else 0.0
    cur=max(1,rep.spec)
    result=[(target-cur)/cur,target/cur,math.log1p(rep.ready),
            max(0,rep.spec-rep.ready)/cur,miss,fail,service_rate,
            math.log1p(target),math.log1p(cur),float(recent_action)]
    return torch.tensor(result,dtype=torch.float32)

def tensors(snap,device='cpu'):
    """No dense adjacency, same shapes used during training and inference."""
    observed=snap.slo.valid(snap.timestamp,120,20)
    miss=float(snap.slo.miss_rate or 0) if observed else 0.
    fail=float(snap.slo.capacity_failure_rate or 0) if observed else 0.
    inflight=(min(1.,max(0.,float(snap.slo.unresolved or 0)/max(float(snap.slo.eligible or 0),1.)))
              if observed else 0.)
    global_context=torch.tensor([miss,fail,inflight,float(observed)],dtype=torch.float32,device=device)
    return (torch.as_tensor(snap.node_x,dtype=torch.float32,device=device),
            torch.as_tensor(snap.node_mask,dtype=torch.float32,device=device),
            torch.as_tensor(snap.edge_index,dtype=torch.long,device=device),
            torch.as_tensor(snap.edge_x,dtype=torch.float32,device=device),
            torch.as_tensor(snap.edge_mask,dtype=torch.float32,device=device),
            global_context)

def checkpoint_dict(graph,response,meta):
    return {'model_version':MODEL_VERSION,'feature_schema':SCHEMA_VERSION,
            'node_order':NODE_FEATURES,'edge_order':EDGE_FEATURES,
            'action_fields':ACTION_FIELDS,'global_order':GLOBAL_FIELDS,
            'hidden_dim':graph.dim,
            'graph_state':graph.state_dict(),'response_state':response.state_dict(),
            'metadata':meta, 'response_target':'delta_miss_and_capacity_failure'}

def load_checkpoint(path,device='cpu'):
    b=torch.load(path,map_location=device,weights_only=True)
    if (b.get('model_version')!=MODEL_VERSION or b.get('feature_schema')!=SCHEMA_VERSION or
        b.get('node_order')!=NODE_FEATURES or b.get('edge_order')!=EDGE_FEATURES or
        b.get('action_fields')!=ACTION_FIELDS or
        b.get('global_order')!=GLOBAL_FIELDS or
        b.get('response_target')!='delta_miss_and_capacity_failure'):
        raise ValueError('checkpoint version, feature or action schema mismatch')
    graph=SparseTemporalGraph(dim=int(b['hidden_dim'])).to(device)
    response=ActionResponseModel(dim=int(b['hidden_dim'])).to(device)
    graph.load_state_dict(b['graph_state']);response.load_state_dict(b['response_state'])
    graph.eval();response.eval()
    return graph,response,b['metadata']