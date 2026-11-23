"""Deployment-agnostic, budgeted, action-conditioned HPA policy.

WHY: separated diagnosis and outcome models are independently testable; action
selection uses the signed effect of a replica intervention instead of imitating
the replica counts in historical telemetry. Prediction targets are *performance
consequences under observed conditions*, never traffic/resource forecasts.

Complexity: one sparse GNN forward, plus at most K * bounded-step actions.
Every proposal may be vetoed by telemetry validity and lifecycle constraints.
"""
from dataclasses import dataclass, asdict
import heapq
import math
import time
import torch
from .model import tensors, action_vector

@dataclass
class Decision:
    service: str
    expected: int
    target: int
    direction: str
    score: float
    rationale: str
    p_root: float=0.
    p_remed: float=0.
    mu_miss: float=0.
    sigma_miss: float=0.
    mu_fail: float=0.
    sigma_fail: float=0.
    hold_mu_miss: float=0.
    valid: bool=True
    def to_dict(self):
        return asdict(self)

class DecisionPolicy:
    def __init__(self,cfg,graph_model,action_model,meta,adapter=None):
        self.cfg=cfg
        self.graph=graph_model.eval()
        self.action=action_model.eval()
        self.meta=meta or {}
        self.adapter=adapter
        self.healthy_since=None
        self.last_diag_ms=0.
        self.last_action_ms=0.
        self.last_rankings=[]

    def eligible_for_live(self):
        """Not an accreditation: a guard that prevents accidental weak-model writes."""
        m=self.meta
        return (m.get('intervention_validated') is True and
                int(m.get('usable_interventions',0))>=self.cfg.min_action_examples and
                m.get('heldout_deployment_validation') is True and
                m.get('probability_calibrated') is True and
                m.get('outcome_target')=='delta_rates' and
                float(m.get('diagnosis_target_fpr',1.0))<=self.cfg.max_diagnosis_fpr and
                m.get('ood_threshold') is not None and
                all(k in m.get('decision_thresholds',{}) for k in ('root','remediable','scalein_safe')) and
                m.get('action_residual_rmse') is not None)

    def infer(self,snapshot):
        t=time.perf_counter()
        with torch.inference_mode():
            out=self.graph(*tensors(snapshot))
        self.last_diag_ms=(time.perf_counter()-t)*1000
        return out

    def _valid_candidates(self,snap,out):
        ok=out['valid'].detach().cpu().tolist()
        # A CPU sample without recent request observations is not enough to
        # assert either a workload bottleneck or idle capacity.
        from .schema import NODE_FEATURES
        seen=[NODE_FEATURES.index(k) for k in ('req_rate','source_req_rate','service_inflight')]
        ok=[bool(v and any(snap.node_mask[i,-1,j] for j in seen)) for i,v in enumerate(ok)]
        center=self.meta.get('ood_center')
        distance=self.meta.get('ood_threshold')
        if center is not None and distance is not None:
            center=torch.tensor(center,dtype=out['embedding'].dtype,device=out['embedding'].device)
            if center.numel()==out['embedding'].shape[-1]:
                dist=torch.linalg.vector_norm(out['embedding']-center,dim=-1).detach().cpu().tolist()
                ok=[v and d<=float(distance) for v,d in zip(ok,dist)]
            else:
                ok=[False]*len(ok)
        elif self.cfg.mode=='live':
            ok=[False]*len(ok)
        return ok

    def _batch_eval(self,snap,out,index,targets,now):
        """Vectorize all candidate replica outcomes for one service."""
        t=time.perf_counter()
        service=snap.services[index]
        actions=torch.stack([action_vector(snap,service,n,now) for n in targets])
        embed=out['embedding'][index].detach().unsqueeze(0).expand(len(targets),-1)
        with torch.inference_mode():
            delta,sd=self.action.estimate(embed,actions,
                calibrated_residual=float(self.meta.get('action_residual_rmse',0.25)))
        correction=[0.,0.]
        if self.adapter:
            direction='out' if targets[0]>snap.replicas[service].spec else 'in'
            correction=self.adapter.correction(service,direction)
        baseline=torch.tensor([snap.slo.miss_rate,snap.slo.capacity_failure_rate],dtype=delta.dtype)
        absolute=(delta+baseline+torch.tensor(correction,dtype=delta.dtype)).clamp(0,1)
        self.last_action_ms+=(time.perf_counter()-t)*1000
        return absolute.detach().cpu().tolist(),sd.detach().cpu().tolist()

    def _score(self,mu,sd,current,target):
        c=self.cfg
        return (mu[0] + c.capacity_failure_weight*mu[1] +
                c.uncertainty_penalty*c.conservative_z*(sd[0]+sd[1]) +
                c.cost_per_replica*target + c.action_churn_cost*abs(target-current))

    def _calibrated_head(self,out,key):
        probs=out[key].detach().clamp(1e-5,1-1e-5)
        temperature=float(self.meta.get('head_temperatures',{}).get(key,1.))
        return torch.sigmoid(torch.logit(probs)/max(.1,temperature)).cpu().tolist()

    def choose(self,snap,now=None,blocked=False):
        now=time.time() if now is None else now
        self.last_action_ms=0.
        self.last_diag_ms=0.
        self.last_rankings=[]
        if not snap.slo.valid(now,self.cfg.max_slo_age_s,self.cfg.min_slo_eligible):
            self.healthy_since=None
            return None,'independent ingress SLO absent, stale, or too few requests',None
        if blocked:
            self.healthy_since=None
            return None,'unresolved action: wait for Ready, serving, post-action observation',None
        out=self.infer(snap)
        ok=self._valid_candidates(snap,out)
        if not any(ok):
            return None,'no valid in-distribution service telemetry',out
        root=self._calibrated_head(out,'root')
        remedy=self._calibrated_head(out,'remediable')
        scalein=self._calibrated_head(out,'scalein_safe')
        self.last_rankings=sorted([
            {'service':s,'p_root':float(root[i]),'p_remediable':float(remedy[i]),
             'p_scalein_safe':float(scalein[i]),'model_valid':bool(ok[i])}
            for i,s in enumerate(snap.services)],
            key=lambda d:d['p_root']*d['p_remediable'],reverse=True)[:min(len(snap.services),10)]
        if self.cfg.mode=='live' and not self.eligible_for_live():
            return None,'model lacks independently validated intervention effects',out
        miss=snap.slo.miss_rate
        failures=snap.slo.capacity_failure_rate
        # Fail closed for scale-in on small samples: zero violations in twenty
        # attempts is NOT evidence of a 99% SLO. Wilson is a conservative proxy
        # (not exact under correlated requests).
        # Separate an observed SLO violation from insufficient statistical
        # evidence of health. The latter must HOLD, not trigger scale-out.
        # This is an application SLO limit, not a fixed resource threshold.
        degraded=(miss>self.cfg.max_slo_risk_scalein or
                  failures>self.cfg.max_slo_risk_scalein)
        healthy=(snap.slo.upper_bernoulli_bound(snap.slo.missed)<=self.cfg.max_slo_risk_scalein and
                 snap.slo.upper_bernoulli_bound(snap.slo.capacity_fail)<=self.cfg.max_slo_risk_scalein)
        if healthy:
            if self.healthy_since is None:self.healthy_since=now
        else:
            self.healthy_since=None
        cutoffs=self.meta.get('decision_thresholds',{})
        # In shadow, a neutral 0.5 is an exploration default only; live
        # requires data-derived thresholds for every supervised decision head.
        root_cut=cutoffs.get('root',.5)
        remedy_cut=cutoffs.get('remediable',.5)
        scalein_cut=cutoffs.get('scalein_safe',.5)
        if degraded:
            ids=heapq.nlargest(self.cfg.top_k_out,
                (i for i in range(len(snap.services)) if ok[i] and
                 root[i]>=root_cut and remedy[i]>=remedy_cut),
                key=lambda i:root[i]*remedy[i])
            direction='out'
        elif healthy:
            # A continuously healthy window is necessary, but independently
            # Ready/serving stabilization is checked per Deployment below.
            if now-self.healthy_since<self.cfg.scale_in_health_s:
                return None,'scale-in: healthy window has not elapsed',out
            ids=heapq.nlargest(self.cfg.top_k_in,
                (i for i in range(len(snap.services)) if ok[i] and
                 scalein[i]>=scalein_cut),key=lambda i:scalein[i])
            direction='in'
        else:
            return None,'SLO inconclusive: no measured breach and not enough evidence for scale-in',out
        if not ids:
            return None,'no appropriately supported '+direction+' candidates',out
        best=None
        for i in ids:
            svc=snap.services[i]
            rep=snap.replicas[svc]
            if not rep.stable():continue
            current=rep.spec
            bounds=self.cfg.target_map()[svc]
            lo,hi=bounds.min_replicas,bounds.max_replicas
            targets=(list(range(current+1,min(hi,current+self.cfg.max_scale_out_step)+1))
                     if direction=='out' else
                     list(range(max(lo,current-self.cfg.max_scale_in_step),current)))
            if not targets:continue
            vals,sds=self._batch_eval(snap,out,i,targets,now)
            hold_score=(miss+self.cfg.capacity_failure_weight*failures +
                        self.cfg.cost_per_replica*current)
            for target,mu,sd in zip(targets,vals,sds):
                if not all(math.isfinite(z) for z in mu+sd):continue
                score=self._score(mu,sd,current,target)
                if direction=='out':
                    # Must improve observed service-level harm, not just provide
                    # apparent savings in replica costs.
                    baseline=miss+self.cfg.capacity_failure_weight*failures
                    candidate=mu[0]+self.cfg.capacity_failure_weight*mu[1]
                    robust_gain=baseline-candidate-self.cfg.conservative_z*(sd[0]+sd[1])
                    if robust_gain<self.cfg.min_relative_gain:continue
                else:
                    if mu[0]+self.cfg.conservative_z*sd[0]>self.cfg.max_slo_risk_scalein:continue
                    if mu[1]+self.cfg.conservative_z*sd[1]>self.cfg.max_slo_risk_scalein:continue
                if score>=hold_score:continue
                if best is None or score<best.score:
                    best=Decision(svc,current,target,direction,score,
                        'counterfactual effect conditional on current observed telemetry; '
                        'conservative uncertainty and replica-cost gate',
                        root[i],remedy[i],mu[0],sd[0],mu[1],sd[1],miss)
        if best is None:
            return None,'all candidate actions failed conservative benefit/risk gates',out
        return best,'action selected',out