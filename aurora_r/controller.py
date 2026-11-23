"""Resident 5-second-telemetry / configurable-interval horizontal controller."""
import json,math,signal,time,traceback
from .schema import NODE_FEATURES
from .metrics import TelemetryCollector
from .policy import DecisionPolicy
from .journal import ActionJournal
from .adapter import OnlineAdapter
from .kube import KubernetesReader,LeaseManager
from .model import load_checkpoint

class AuroraController:
    def __init__(self,cfg,kube=None,collector=None,journal=None,leader=None):
        self.cfg=cfg
        self.graph,self.response,self.metadata=load_checkpoint(cfg.checkpoint)
        self.adapter=OnlineAdapter(cfg.adapter_path)
        self.policy=DecisionPolicy(cfg,self.graph,self.response,self.metadata,self.adapter)
        self.kube=kube or KubernetesReader(cfg)
        self.collector=collector or TelemetryCollector(cfg)
        self.journal=journal or ActionJournal(cfg.journal_path)
        self.leader=leader
        self.run_id=str(int(time.time()))
        self.stopped=False
        self.timings=[]
        if cfg.mode=='live' and cfg.policy_source=='model' and not self.policy.eligible_for_live():
            raise RuntimeError('model not intervention-validated/heldout/calibrated; live scaling refused')
        if cfg.mode=='live' and cfg.policy_source=='rule':
            # The rule policy is deterministic and carries no learned
            # confidence, so the model accreditation gate does not apply. It is
            # still bound by every other guard: single writer, per-service
            # bounds, SLO validity, lifecycle stability.
            print('[aurora-r] live mode with the deterministic rule policy; '
                  'the learned response model is not in the decision path',
                  flush=True)
    @staticmethod
    def _rate(snap,svc):
        idx=snap.services.index(svc)
        f=NODE_FEATURES.index('req_rate')
        if not snap.node_mask[idx,-1,f]:return None
        return math.expm1(float(snap.node_x[idx,-1,f]))

    def _settle(self,states,snap,now):
        """Needs spec convergence, readiness, serving, AND entirely post-Ready SLO window."""
        st={name:val[1] for name,val in states.items()}
        self.journal.reconcile_created(st)
        for a in self.journal.unresolved():
            if a['status']=='inconclusive':continue
            rep=st.get(a['service'])
            if rep is None:continue
            if rep.spec!=a['target']:
                if now-a['created']>self.cfg.max_rollout_s:
                    self.journal.update(a['id'],'inconclusive',detail='Scale spec did not converge')
                continue
            converged=rep.stable() and rep.ready==a['target']
            if a['status']=='applied':
                if converged:self.journal.update(a['id'],'ready',now=now,detail='Ready and serving verified')
                elif now-a['created']>self.cfg.max_rollout_s:
                    self.journal.update(a['id'],'inconclusive',detail='Pod rollout/serving never converged')
                continue
            if a['status']=='ready':
                if not converged:
                    # If Pods regress, do NOT treat old ready_at as evidence.
                    self.journal.update(a['id'],'applied',detail='ready/serving regressed; rechecking')
                    continue
                delay=max(self.cfg.min_settle_after_ready_s,
                          self.cfg.slo_window_s+2*self.cfg.telemetry_step_s)
                if now-(a['ready_at'] or now)<delay:continue
                # Ensure the entire SLO increase() interval follows readiness,
                # even when Mimir has ingestion lag.
                if snap.slo.sampled_at < (a['ready_at'] or now)+self.cfg.slo_window_s:continue
                if not snap.slo.valid(now,self.cfg.max_slo_age_s,self.cfg.min_slo_eligible):continue
                post_req=self._rate(snap,a['service'])
                pre_req=a['before'].get('service_req_rate')
                workload_shift=(abs(post_req-pre_req)/max(1.,pre_req)
                                if post_req is not None and pre_req is not None else float('inf'))
                matched=workload_shift<=self.cfg.max_workload_shift_for_learning
                after={'post_miss_rate':snap.slo.miss_rate,
                       'post_capacity_failure_rate':snap.slo.capacity_failure_rate,
                       'post_eligible':snap.slo.eligible,
                       'service_req_rate':post_req,'workload_shift':workload_shift,
                       'workload_matched':matched,'settled_at':now,
                       'replicas':rep.spec,'ready':rep.ready,'ready_serving':rep.ready_serving}
                self.journal.update(a['id'],'settled',now=now,after=after,
                                    detail='observed rollout and post-convergence SLO')
                # Only observational residual calibration; no causal effect claim.
                pred=a['before'].get('prediction',{})
                pre=a['before']
                if all(k in pred for k in ('mu_miss','mu_fail')) and all(
                    pre.get(k) is not None for k in ('slo_miss_rate','slo_capacity_failure_rate')):
                    # Learn only *residual errors in measured action effects*, not
                    # raw post rates; matched state avoids obvious confounding.
                    measured_delta=[after['post_miss_rate']-pre['slo_miss_rate'],
                                    after['post_capacity_failure_rate']-pre['slo_capacity_failure_rate']]
                    predicted_delta=[pred['mu_miss']-pre['slo_miss_rate'],
                                     pred['mu_fail']-pre['slo_capacity_failure_rate']]
                    self.adapter.update(a['service'],a['before'].get('direction','out'),
                        predicted_delta, measured_delta, matched=matched,settled=True)

    def cycle(self):
        tic=time.perf_counter();now=time.time();ms={}
        t=time.perf_counter();states=self.kube.read_states();ms['kube_read_ms']=(time.perf_counter()-t)*1000
        t=time.perf_counter();snap,info=self.collector.collect(self.run_id,states,now)
        ms['telemetry_ms']=(time.perf_counter()-t)*1000
        t=time.perf_counter();self._settle(states,snap,time.time())
        unresolved=self.journal.unresolved();ms['reconcile_ms']=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        decision,why,diag=self.policy.choose(snap,now=time.time(),blocked=bool(unresolved))
        ms['inference_and_policy_ms']=(time.perf_counter()-t)*1000
        ms['graph_inference_ms']=self.policy.last_diag_ms
        ms['action_inference_ms']=self.policy.last_action_ms
        record={'ts':now,'mode':self.cfg.mode,'decision':decision.to_dict() if decision else None,
                'reason':why,'telemetry':info,'actions_unresolved':len(unresolved),
                'diagnostic_rankings':self.policy.last_rankings}
        if decision and self.cfg.mode=='live':
            if not self.leader or not self.leader.authorized():
                record['write']='hold: no current Lease authority'
            else:
                t=time.perf_counter()
                # Check immediately before the write; no conflicting native HPAs.
                self.kube.assert_single_writer()
                if not self.leader.authorized():
                    record['write']='hold: leadership lost'
                else:
                    before={
                        # Full pre-treatment observation is indispensable for
                        # intervention-conditioned learning; summary alone cannot
                        # reconstruct the treatment context later.
                        'snapshot':snap.to_dict(),
                        'slo_eligible':snap.slo.eligible,
                        'slo_miss_rate':snap.slo.miss_rate,
                        'slo_capacity_failure_rate':snap.slo.capacity_failure_rate,
                        'service_req_rate':self._rate(snap,decision.service),
                        'direction':decision.direction,
                        'prediction':{'mu_miss':decision.mu_miss,'mu_fail':decision.mu_fail},
                        'model':self.metadata.get('model_id','unknown')}
                    aid=self.journal.create(decision.service,decision.expected,decision.target,before)
                    try:
                        self.kube.write_scale(decision.service,decision.expected,decision.target)
                        self.journal.update(aid,'applied',detail='Scale API responded success')
                        record['write']='applied';record['action_id']=aid
                    except Exception as exc:
                        # Patch could have reached apiserver despite a timeout.
                        # Keep the action unresolved, do not automatically retry.
                        self.journal.update(aid,'inconclusive',detail=f'ambiguous Scale API: {exc}')
                        record['write']=f'inconclusive: {exc}'
                ms['scale_write_ms']=(time.perf_counter()-t)*1000
        else:record['write']='shadow' if self.cfg.mode=='shadow' else 'hold'
        ms['complete_cycle_ms']=(time.perf_counter()-tic)*1000
        record['timing']=ms
        self.timings.append(ms)
        if len(self.timings)>1000:self.timings.pop(0)
        print(json.dumps(record,allow_nan=False),flush=True)
        return record

    def run(self,once=False):
        if self.cfg.mode=='live':
            self.kube.assert_single_writer()
            if self.leader is None:self.leader=LeaseManager(self.kube,self.cfg.lease_name)
            self.leader.start()
        def stop(sig,frame):self.stopped=True
        if not once:
            signal.signal(signal.SIGTERM,stop)
            signal.signal(signal.SIGINT,stop)
        try:
            while not self.stopped:
                start=time.monotonic()
                try:self.cycle()
                except Exception as exc:
                    print(json.dumps({'cycle_error':str(exc),'trace':traceback.format_exc()}),flush=True)
                if once:break
                remaining=self.cfg.decision_interval_s-(time.monotonic()-start)
                if remaining>0:time.sleep(remaining)
                # Overruns are logged as a full cycle, never overlapping writes.
        finally:
            if self.leader:self.leader.stop()
            if self.timings:
                total=sorted(x['complete_cycle_ms'] for x in self.timings)
                p=lambda q:total[min(len(total)-1,int((len(total)-1)*q))]
                print(json.dumps({'cycles':len(total),'cycle_latency_p50_ms':p(.5),
                                  'cycle_latency_p95_ms':p(.95),
                                  'cycle_latency_p99_ms':p(.99),'cycle_latency_max_ms':total[-1]}))