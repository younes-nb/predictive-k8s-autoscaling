"""Kubernetes Scale-only actuation; explicit target ownership, no HPA deletion.

Requires operator removal/disablement of conflicting native HPAs before live
mode. Writes use the Scale subresource with optimistic resourceVersion checks.
Only one elected writer may issue writes. Lease alone is not a hard fencing token.
"""
from datetime import datetime,timezone
import os,socket,threading,time
from .schema import ReplicaState

class KubernetesReader:
    def __init__(self,config):
        from kubernetes import client,config as kc
        try:kc.load_incluster_config()
        except kc.ConfigException:kc.load_kube_config()
        self.cfg=config
        self.apps=client.AppsV1Api()
        self.pods=client.CoreV1Api()
        self.autos=client.AutoscalingV2Api()
        self.discovery=client.DiscoveryV1Api()
        self.coordination=client.CoordinationV1Api()
        self.client=client

    def read_states(self):
        """One list for Deployments, one for Pods, one for EndpointSlices."""
        ns=self.cfg.namespace
        names=self.cfg.target_map()
        deps={x.metadata.name:x for x in self.apps.list_namespaced_deployment(ns).items if x.metadata.name in names}
        missing=set(names)-set(deps)
        if missing:raise RuntimeError('managed Deployment absent: '+','.join(sorted(missing)))
        # Map Pod -> ReplicaSet -> Deployment owner references in O(P+R),
        # instead of testing every Pod selector against every Deployment (O(P*N)).
        # The owning RS is authoritative for a Deployment-managed Pod.
        replicasets=self.apps.list_namespaced_replica_set(ns).items
        rs_to_deploy={}
        for rs in replicasets:
            for owner in (rs.metadata.owner_references or []):
                if owner.kind=='Deployment' and owner.name in deps:
                    dep=deps[owner.name]
                    if dep.metadata.uid and owner.uid!=dep.metadata.uid:continue
                    rs_to_deploy[rs.metadata.name]=owner.name
        all_pods=self.pods.list_namespaced_pod(ns).items
        matched={s:[] for s in deps}
        for p in all_pods:
            owners={rs_to_deploy.get(ref.name) for ref in (p.metadata.owner_references or [])
                    if ref.kind=='ReplicaSet' and ref.name in rs_to_deploy}
            owners.discard(None)
            if len(owners)>1:raise RuntimeError('Pod owned by multiple managed Deployments')
            if owners:matched[next(iter(owners))].append(p)
        try:
            slices=self.discovery.list_namespaced_endpoint_slice(ns).items
        except Exception:
            slices=[]    # no EndpointSlice telemetry -> serving unknown -> hold
        serving={s:set() for s in deps}
        service_index={t.service_name or t.istio_workload:s for s,t in names.items()}
        for ep_slice in slices:
            svc_name=(ep_slice.metadata.labels or {}).get('kubernetes.io/service-name')
            dep_name=service_index.get(svc_name)
            if dep_name is None:continue
            for ep in (ep_slice.endpoints or []):
                conditions=ep.conditions
                if not (conditions and conditions.ready is True and
                        getattr(conditions,'serving',None) is True):continue
                ref=ep.target_ref
                if ref is not None and ref.kind=='Pod' and ref.name:
                    serving[dep_name].add(ref.name)
        states={}
        for s,d in deps.items():
            ps=matched[s]
            spec=int(d.spec.replicas or 0)
            ready=running=scheduled=terminating=0
            startups=[];alive=[]
            for p in ps:
                if p.metadata.deletion_timestamp:
                    terminating+=1;continue
                alive.append(p.metadata.name)
                conditions={c.type:c for c in p.status.conditions or []}
                if conditions.get('PodScheduled') and conditions['PodScheduled'].status=='True':scheduled+=1
                if p.status.phase=='Running':running+=1
                if conditions.get('Ready') and conditions['Ready'].status=='True':
                    ready+=1
                    if p.metadata.creation_timestamp and conditions['Ready'].last_transition_time:
                        dt=(conditions['Ready'].last_transition_time-p.metadata.creation_timestamp).total_seconds()
                        if dt>=0:startups.append(float(dt))
            status=d.status
            rollout=(int(status.observed_generation or 0)>=int(d.metadata.generation or 0)
                     and int(status.updated_replicas or 0)==spec and int(status.unavailable_replicas or 0)==0)
            rep=ReplicaState(spec=spec,ready=ready,running=running,
                available=int(status.available_replicas or 0),scheduled=scheduled,
                terminating=terminating,request_pending=max(0,spec-len(alive)),
                rollout_complete=rollout,ready_serving=len(serving[s]&set(alive)),
                pod_startup_seconds=startups)
            states[s]=(names[s],rep,alive)
        return states

    def assert_single_writer(self):
        targets=set(self.cfg.target_map())
        hpas=self.autos.list_namespaced_horizontal_pod_autoscaler(self.cfg.namespace).items
        conflict=[h.metadata.name for h in hpas if h.spec.scale_target_ref.kind=='Deployment'
                  and h.spec.scale_target_ref.name in targets]
        if conflict:raise RuntimeError('refusing to compete with native HPAs: '+','.join(conflict))

    def write_scale(self,service,expected_spec,target):
        if service not in self.cfg.target_map():raise ValueError('unmanaged Deployment')
        t=self.cfg.target_map()[service]
        if not t.min_replicas<=target<=t.max_replicas:raise ValueError('out of replica bounds')
        obj=self.apps.read_namespaced_deployment_scale(service,self.cfg.namespace)
        actual=int(obj.spec.replicas or 0)
        if actual!=expected_spec:
            raise RuntimeError(f'Scale.spec race for {service}: observed {actual} expected {expected_spec}')
        obj.spec.replicas=int(target)
        # A replace carries metadata.resourceVersion, rejecting a stale writer.
        return self.apps.replace_namespaced_deployment_scale(service,self.cfg.namespace,obj)

class LeaseManager:
    def __init__(self,kube,lease_name,renew_s=5,lease_seconds=35):
        self.kube=kube;self.cfg=kube.cfg;self.name=lease_name
        self.lease_seconds=lease_seconds;self.renew_s=renew_s
        self.identity=self.cfg.leader_identity or os.getenv('HOSTNAME') or (socket.gethostname()+'-'+str(os.getpid()))
        self.leader=False;self.last_ok=0.0
        self._lock=threading.Lock();self._stopped=threading.Event();self._thread=None
    def _renew_once(self):
        from kubernetes.client.rest import ApiException
        now=datetime.now(timezone.utc)
        try:
            try:obj=self.kube.coordination.read_namespaced_lease(self.name,self.cfg.namespace)
            except ApiException as exc:
                if exc.status!=404:raise
                obj=None
            if obj is not None:
                spec=obj.spec
                holder=spec.holder_identity or ''
                renew=spec.renew_time
                # Clock skew margin: takeover only if clearly past expiry.
                expired=renew is not None and (now-renew).total_seconds()>self.lease_seconds+5
                if holder and holder!=self.identity and not expired:return False
                if holder!=self.identity:spec.acquire_time=now
                spec.lease_transitions=int(spec.lease_transitions or 0)+(1 if holder and holder!=self.identity else 0)
                spec.holder_identity=self.identity
                spec.renew_time=now;spec.lease_duration_seconds=self.lease_seconds
                self.kube.coordination.replace_namespaced_lease(self.name,self.cfg.namespace,obj)
            else:
                c=self.kube.client
                new=c.V1Lease(metadata=c.V1ObjectMeta(name=self.name,namespace=self.cfg.namespace),
                    spec=c.V1LeaseSpec(holder_identity=self.identity,lease_duration_seconds=self.lease_seconds,
                        acquire_time=now,renew_time=now,lease_transitions=0))
                self.kube.coordination.create_namespaced_lease(self.cfg.namespace,new)
            return True
        except Exception:
            return False
    def _work(self):
        while not self._stopped.is_set():
            okay=self._renew_once()
            with self._lock:
                self.leader=okay
                if okay:self.last_ok=time.monotonic()
            self._stopped.wait(self.renew_s)
    def start(self):
        if self._thread:return
        self._thread=threading.Thread(target=self._work,daemon=True)
        self._thread.start()
    def authorized(self):
        with self._lock:
            return self.leader and (time.monotonic()-self.last_ok)<self.lease_seconds/2
    def stop(self):
        self._stopped.set()
        if self._thread:self._thread.join(timeout=2)
        # Do not blindly release; natural expiry protects leadership races.