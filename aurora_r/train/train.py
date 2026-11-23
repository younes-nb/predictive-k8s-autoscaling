"""Evidence-aware two-stage training, grouped by Deployment to avoid leakage.

Training is deliberately unavailable from weak utilization-derived 'ground truth'.
Stage 1: masked self-supervision and only *explicitly labeled* root, propagated,
remediability, scale-in and failure diagnoses. Stage 2: frozen shared encoder,
action-response ensemble trained on independently settled, workload-matched,
controlled replica interventions (signed change in deadline-miss/failure rates).

A checkpoint without deployment-held-out intervention and head calibration is
SHADOW ONLY. Supplying flags does not manufacture experimental validation.
"""
from collections import defaultdict
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from ..model import SparseTemporalGraph,ActionResponseModel,tensors,action_vector,checkpoint_dict
from ..schema import GraphSnapshot,Intervention,DEGRADATION,FAILURE_CAUSES

HEAD_LABELS={'root':'root_labels','affected':'affected_labels',
             'remediable':'remedy_labels','scalein_safe':'scalein_labels'}

def read_jsonl(paths,cls):
    result=[]
    for p in paths or []:
        with open(p,encoding='utf-8') as fh:
            for i,line in enumerate(fh,1):
                if not line.strip():continue
                try:result.append(cls.from_dict(json.loads(line)))
                except Exception as exc:raise ValueError(f'{p}:{i}: {exc}') from exc
    return result

def action_effect(item):
    # Interventions are usable whatever produced the action (rule or model), as
    # long as they carry the evidence fields: settled post-Ready SLO, a
    # workload-matched rate, and a declared control. The treatment *label* is
    # the observed replica change; no causal claim is made without a control.
    if not item.eligible():raise ValueError('intervention missing evidence')
    pre=item.snapshot.slo
    # Matched-control trend corrections are declared by the experiment. Under
    # controlled randomization the corrections can be zero. Never infer a causal
    # improvement from an unpaired pre/post comparison.
    return torch.tensor([
        (item.post_missed/item.post_eligible)-(pre.miss_rate or 0)-item.control_adjustment_miss,
        (item.post_capacity_fail/item.post_eligible)-(pre.capacity_failure_rate or 0)-item.control_adjustment_fail],
        dtype=torch.float32).clamp(-1.,1.)

def head_losses(graph,snapshot,mask_prob=.20,supervised_weight=1.0):
    x,m,ei,ex,em,gctx=tensors(snapshot)
    latest=m[:,-1].bool()
    corruption=(torch.rand_like(m[:,-1])<mask_prob)&latest
    # Mask only the last timepoint, so the model can use past context and graph
    # relationships to reconstruct absent values without a forecasting target.
    xm=x.clone();mm=m.clone()
    xm[:,-1][corruption]=0.;mm[:,-1][corruption]=0.
    out=graph(xm,mm,ei,ex,em,gctx)
    losses={}
    if corruption.any():
        losses['masked_reconstruction']=F.smooth_l1_loss(
            out['reconstruction'][corruption],graph.normalized_nodes(x)[:,-1][corruption])
    for head,field in HEAD_LABELS.items():
        labels=getattr(snapshot,field)
        ids=[i for i,s in enumerate(snapshot.services) if s in labels and bool(out['valid'][i])]
        if ids:
            y=torch.tensor([float(labels[snapshot.services[i]]) for i in ids],dtype=torch.float32)
            if not torch.isfinite(y).all() or y.min()<0 or y.max()>1:
                raise ValueError(f'invalid {field}: expected [0,1]')
            losses[head]=supervised_weight*F.binary_cross_entropy(
                out[head][ids].clamp(1e-6,1-1e-6),y)
    ids=[i for i,s in enumerate(snapshot.services) if s in snapshot.degradation_labels and bool(out['valid'][i])]
    if ids:
        y=torch.zeros((len(ids),len(DEGRADATION)))
        for j,i in enumerate(ids):
            labels=snapshot.degradation_labels[snapshot.services[i]]
            if isinstance(labels,str):labels=[labels]
            for name in labels:
                if name not in DEGRADATION:raise ValueError('unknown degradation '+name)
                y[j,DEGRADATION.index(name)]=1.
        losses['degradation']=supervised_weight*F.binary_cross_entropy(
            out['degradation'][ids].clamp(1e-6,1-1e-6),y)
    ids=[i for i,s in enumerate(snapshot.services) if s in snapshot.failure_labels and bool(out['valid'][i])]
    if ids:
        labs=[snapshot.failure_labels[snapshot.services[i]] for i in ids]
        if any(name not in FAILURE_CAUSES for name in labs):raise ValueError('unknown failure cause')
        y=torch.tensor([FAILURE_CAUSES.index(name) for name in labs],dtype=torch.long)
        losses['failure']=supervised_weight*F.nll_loss(out['failure'][ids].clamp_min(1e-8).log(),y)
    return losses

def fit_robust_scalers(graph,snapshots):
    """Robust medians/scales from training deployments only, without leakage."""
    def fit(feature_count,extract):
        by_feature=[[] for _ in range(feature_count)]
        for snap in snapshots:
            values,mask=extract(snap)
            for j in range(feature_count):
                observed=values[...,j][mask[...,j]>0]
                if len(observed):by_feature[j].extend(observed.astype(float).tolist())
        center=[];scale=[]
        for vals in by_feature:
            if len(vals)<3:center.append(0.);scale.append(1.);continue
            q25,median,q75=np.quantile(vals,[.25,.5,.75])
            center.append(float(median))
            scale.append(float(max(.5,(q75-q25)/1.349)))
        return torch.tensor(center),torch.tensor(scale)
    nc,ns=fit(len(graph.node_center),lambda s:(s.node_x,s.node_mask))
    ec,es=fit(len(graph.edge_center),lambda s:(s.edge_x,s.edge_mask))
    with torch.no_grad():
        graph.node_center.copy_(nc);graph.node_scale.copy_(ns)
        graph.edge_center.copy_(ec);graph.edge_scale.copy_(es)

def train_graph(graph,snapshots,epochs=6,lr=8e-4,seed=20):
    if not snapshots:return []
    torch.manual_seed(seed)
    optimizer=torch.optim.AdamW(graph.parameters(),lr=lr,weight_decay=1e-4)
    graph.train();history=[]
    for epoch in range(epochs):
        random.Random(seed+epoch).shuffle(snapshots)
        total=0.;seen=0
        for snap in snapshots:
            optimizer.zero_grad()
            parts=head_losses(graph,snap)
            if not parts:continue
            loss=sum(parts.values())
            loss.backward();torch.nn.utils.clip_grad_norm_(graph.parameters(),2.)
            optimizer.step()
            total+=float(loss.detach());seen+=1
        history.append({'epoch':epoch+1,'loss':total/max(seen,1),'steps':seen})
    graph.eval()
    return history

def train_effect(graph,model,items,epochs=12,lr=8e-4,seed=21):
    if not items:return []
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    model.train();history=[]
    # Graph is frozen. Each Deployment has one shared representation across
    # replica choices; none of these samples create root-cause pseudo labels.
    graph.eval()
    for epoch in range(epochs):
        rng=random.Random(seed+epoch)
        order=items.copy();rng.shuffle(order)
        total=0.;seen=0
        for intervention in order:
            snap=intervention.snapshot
            with torch.no_grad():
                embedding=graph(*tensors(snap))['embedding'][snap.services.index(intervention.service)]
            a=action_vector(snap,intervention.service,intervention.target_replicas,snap.timestamp).unsqueeze(0)
            target=action_effect(intervention).unsqueeze(0)
            opt.zero_grad()
            predictions=model(embedding.unsqueeze(0),a)
            # Poisson(1) bootstrap weights decorrelate ensemble members; this
            # does NOT on its own guarantee calibrated uncertainty.
            weight=torch.poisson(torch.ones((predictions.shape[0],1,1)))
            if weight.sum()==0:weight[0]=1.
            raw=F.smooth_l1_loss(predictions,target.unsqueeze(0).expand_as(predictions),
                                 reduction='none')
            loss=(raw*weight).sum()/weight.sum().clamp_min(1.)/predictions.shape[-1]
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step();total+=float(loss.detach());seen+=1
        history.append({'epoch':epoch+1,'loss':total/max(seen,1),'steps':seen})
    model.eval();return history

def evaluate_effect(graph,model,items):
    if not items:return {'examples':0,'mae':None,'q95_abs_error':None,'out':0,'in':0}
    errs=[];by_direction=defaultdict(int)
    with torch.no_grad():
        for item in items:
            s=item.snapshot;i=s.services.index(item.service)
            e=graph(*tensors(s))['embedding'][i].unsqueeze(0)
            a=action_vector(s,item.service,item.target_replicas,s.timestamp).unsqueeze(0)
            estimate=model(e,a).mean(0)[0]
            errs.extend((estimate-action_effect(item)).abs().tolist())
            by_direction['out' if item.target_replicas>s.replicas[item.service].spec else 'in']+=1
    return {'examples':len(items),'mae':float(np.mean(errs)),
            'q95_abs_error':float(np.quantile(errs,.95)),
            'out':by_direction['out'],'in':by_direction['in']}

def _labels_for_head(graph,snapshots,head,field):
    vals=[];labels=[]
    with torch.no_grad():
        for snap in snapshots:
            out=graph(*tensors(snap))
            for i,svc in enumerate(snap.services):
                if svc in getattr(snap,field) and bool(out['valid'][i]):
                    vals.append(float(out[head][i]))
                    labels.append(float(getattr(snap,field)[svc]))
    return vals,labels

def _temperature(probs,truth):
    """Select a scalar temperature on disjoint held-out deployments."""
    if len(truth)<20 or len(set(truth))<2:return None,None
    logits=torch.tensor(probs).clamp(1e-5,1-1e-5).logit()
    labels=torch.tensor(truth)
    grid=np.geomspace(.4,5.,40)
    best=min(grid,key=lambda t:float(F.binary_cross_entropy_with_logits(logits/float(t),labels)))
    calibrated=torch.sigmoid(logits/float(best))
    brier=float(((calibrated-labels)**2).mean())
    return float(best),brier

def calibrated_cutoff(probs,truth,temperature,target_fpr=.05):
    """Held-out negative-class quantile with finite-sample FPR check.

    This is a validation-derived decision threshold, not an arbitrary feature
    cut-off. It deliberately abstains when negative or positive support is small.
    """
    pos=sum(y>.5 for y in truth);neg=sum(y<=.5 for y in truth)
    if temperature is None or pos<10 or neg<20:return None
    clipped=np.clip(np.asarray(probs,dtype=float),1e-6,1-1e-6)
    adjusted=1/(1+np.exp(-np.log(clipped/(1-clipped))/temperature))
    negatives=adjusted[np.asarray(truth)<=.5]
    # At most empirical FPR target; strict > to resolve tied probabilities.
    cutoff=float(max(.5,np.quantile(negatives,1-target_fpr)+1e-6))
    realized=float((negatives>=cutoff).mean())
    return cutoff if realized<=target_fpr else None

def ood_stats(graph,snapshots):
    vectors=[]
    with torch.no_grad():
        for snap in snapshots:
            out=graph(*tensors(snap))
            vals=out['embedding'][out['valid']]
            if len(vals):vectors.append(vals.numpy())
    if not vectors:return None,None
    z=np.concatenate(vectors,axis=0)
    center=np.mean(z,axis=0)
    norm=np.linalg.norm(z-center,axis=1)
    # A training quantile is a heuristic OOD gate, not a guarantee of coverage.
    return center.tolist(),float(np.quantile(norm,.99)*1.3+1e-3)

def train_all(snapshots,interventions,heldout_deployments,output,
              graph_epochs=8,effect_epochs=14,dim=48,seed=21,target_fpr=.05):
    if not 0<target_fpr<1:raise ValueError('diagnosis FPR must lie in (0,1)')
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    train_snap=[s for s in snapshots if s.deployment_id not in heldout_deployments]
    val_snap=[s for s in snapshots if s.deployment_id in heldout_deployments]
    evidence=[it for it in interventions if it.eligible()]
    train_effects=[i for i in evidence if i.snapshot.deployment_id not in heldout_deployments]
    val_effects=[i for i in evidence if i.snapshot.deployment_id in heldout_deployments]
    if not train_snap and train_effects:
        train_snap=list({id(i.snapshot):i.snapshot for i in train_effects}.values())
    if not train_snap:raise ValueError('no training snapshots, create observations before training')
    if not heldout_deployments:
        # Training is possible for shadow inference, but cannot claim a
        # deployment-independent test or permit live actuation.
        val_snap=[];val_effects=[]
    graph=SparseTemporalGraph(dim=dim)
    response=ActionResponseModel(dim=dim)
    fit_robust_scalers(graph,train_snap)
    gh=train_graph(graph,train_snap,epochs=graph_epochs,seed=seed)
    eh=train_effect(graph,response,train_effects,epochs=effect_epochs,seed=seed+1)
    eval_result=evaluate_effect(graph,response,val_effects)
    temps={};briers={};thresholds={}
    for head,field in HEAD_LABELS.items():
        probs,truth=_labels_for_head(graph,val_snap,head,field)
        t,brier=_temperature(probs,truth)
        if t is not None:
            temps[head]=t;briers[head]=brier
            cutoff=calibrated_cutoff(probs,truth,t,target_fpr=target_fpr)
            if cutoff is not None:thresholds[head]=cutoff
    center,dist=ood_stats(graph,train_snap)
    train_deployments=sorted({s.deployment_id for s in train_snap})
    validation_deployments=sorted({s.deployment_id for s in val_snap}|
                                  {i.snapshot.deployment_id for i in val_effects})
    validated=(len(train_deployments)>=1 and len(validation_deployments)>=1
               and len(val_effects)>=10 and eval_result['out']>=3
               and eval_result['in']>=3 and eval_result['mae'] is not None
               and eval_result['mae']<.10)
    calibrated=(all(h in thresholds for h in ('root','remediable','scalein_safe')) and
                eval_result['examples']>=10)
    meta={
        'model_id':'aurora-r-'+str(seed), 'training_seed':seed,
        'outcome_target':'delta_rates', 'diagnosis_target_fpr':target_fpr,
        'training_deployments':train_deployments,
        'validation_deployments':validation_deployments,
        'train_snapshots':len(train_snap),'heldout_snapshots':len(val_snap),
        'usable_interventions':len(train_effects),
        'heldout_interventions':len(val_effects),
        'intervention_validated':bool(validated),
        'heldout_deployment_validation':bool(validated),
        'probability_calibrated':bool(calibrated),
        'head_temperatures':temps,'decision_thresholds':thresholds,
        'brier':briers,
        'ood_center':center,'ood_threshold':dist,
        'action_residual_rmse':max(.02,eval_result['q95_abs_error'] or .25),
        'effect_validation':eval_result,
        'graph_training_history':gh,'effect_training_history':eh,
        'warnings':[
            'The q95 validation residual is a heuristic uncertainty bound; not a causal confidence interval.',
            'Causal interpretation requires randomized or defensible matched-control interventions.',
            'Held-out deployment validation does not imply safety in every unseen deployment.',
            'Do not enable live mode until independent cluster-specific shadow tests pass.',
        ],
    }
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint_dict(graph,response,meta),path)
    return meta

def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshots',action='append',default=[],help='JSONL GraphSnapshot; repeatable')
    ap.add_argument('--interventions',action='append',default=[],help='JSONL Intervention; repeatable')
    ap.add_argument('--heldout',action='append',default=[],help='Deployment ID never used in training')
    ap.add_argument('--output',required=True)
    ap.add_argument('--graph-epochs',type=int,default=8)
    ap.add_argument('--effect-epochs',type=int,default=14)
    ap.add_argument('--dimension',type=int,default=48)
    ap.add_argument('--seed',type=int,default=21)
    ap.add_argument('--diagnosis-fpr',type=float,default=.05)
    args=ap.parse_args(argv)
    snaps=read_jsonl(args.snapshots,GraphSnapshot)
    items=read_jsonl(args.interventions,Intervention)
    meta=train_all(snaps,items,set(args.heldout),args.output,
                   args.graph_epochs,args.effect_epochs,args.dimension,args.seed,
                   args.diagnosis_fpr)
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()