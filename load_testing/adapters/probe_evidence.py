#!/usr/bin/env python3
"""Strict interval evidence. Site supplies explicit query policy (no guessed labels).
Each check has exact expected label sets, value bounds, and source-age expression.
Missing samples, stale source data, failed API, or unexpected series fail closed.
"""
import argparse, json, math, os, pathlib, sys, urllib.request, urllib.parse
from urllib.request import build_opener, ProxyHandler

REQUIRED={'generator_cpu','generator_memory','generator_network','broker_resources','broker_memory',
          'cluster_cpu','cluster_memory','cluster_network','application_throttling',
          'database','storage','gateway','pending_pods'}
GENERATOR={'generator_cpu','generator_memory','generator_network','broker_resources','broker_memory'}

def query_range(base, expr, start, end, step, tenant=None):
    url=base.rstrip('/')+'/prometheus/api/v1/query_range?'+urllib.parse.urlencode(dict(query=expr,start=start,end=end,step=step))
    req=urllib.request.Request(url,headers={'X-Scope-OrgID':tenant} if tenant else {})
    with build_opener(ProxyHandler({})).open(req,timeout=30) as r: body=json.load(r)
    if body.get('status')!='success' or body.get('warnings') or body.get('data',{}).get('resultType')!='matrix': raise ValueError('query_range failed or returned warnings')
    return body['data']['result']

def labels(x): return tuple(sorted((k,str(v)) for k,v in x.items() if k!='__name__'))

def matrix_values(matrix, expected, start, end, step):
    targets={labels(x) for x in expected}
    if not targets or len(targets)!=len(expected): raise ValueError('empty or duplicate expected series')
    series={}; count=round((end-start)/step)+1
    for row in matrix:
        key=labels(row['metric']); vals=row['values']
        if key in series or len(vals)!=count: raise ValueError('incomplete or duplicate samples')
        values=[]
        for i,(t,v) in enumerate(vals):
            if abs(float(t)-(start+i*step))>.01 or not math.isfinite(float(v)): raise ValueError('invalid sample time/value')
            values.append(float(v))
        series[key]=values
    if set(series)!=targets: raise ValueError('unexpected/missing series')
    return [v for row in series.values() for v in row]

def evaluate(check, query, start, end, step):
    try:
        values=matrix_values(query(check['expr'],start,end,step),check['expected_labels'],start,end,step)
        ages=matrix_values(query(check['age_expr'],start,end,step),check['expected_labels'],start,end,step)
        max_age=float(check['max_age_seconds'])
        if not math.isfinite(max_age) or max_age<=0 or min(ages)<-1 or max(ages)>max_age: raise ValueError('stale source samples')
        lo=check.get('min'); hi=check.get('max')
        if lo is None and hi is None and check.get('observe_only') is not True: raise ValueError('no bounds or explicit observe_only')
        if any(not math.isfinite(float(v)) for v in (lo,hi) if v is not None): raise ValueError('nonfinite bounds')
        ok=(lo is None or min(values)>=lo) and (hi is None or max(values)<=hi)
        return {'verified':True,'within_bounds':ok,'min':min(values),'max':max(values),'samples':len(values),'max_age_seconds':max(ages)}
    except Exception as e:
        return {'verified':False,'within_bounds':False,'error_class':type(e).__name__,'reason':str(e) if isinstance(e,(ValueError,KeyError)) else 'query failed'}

def interval_evidence(ns,policy,query,start,duration,vector):
    """Shared strict interval evaluator for business and authentication probes."""
    step=int(policy['step_seconds'])
    if not math.isfinite(float(policy['replica_max_age_seconds'])) or float(policy['replica_max_age_seconds'])<=0: raise ValueError('invalid replica freshness limit')
    if start is None or not math.isfinite(start) or not math.isfinite(duration) or step<1 or duration<2*step: raise ValueError('invalid probe interval')
    # Align the last grid point inside the actual measurement, not drain or post-run.
    end=start+math.floor(duration/step)*step
    if end>=start+duration: end-=step
    if end<=start: raise ValueError('too few interval samples')
    checks=policy['checks']; required=REQUIRED
    if not required<=set(checks): raise ValueError('required telemetry categories missing: '+str(sorted(required-set(checks))))
    # The session broker is the local Python process, not the RabbitMQ service.
    for name in ('broker_resources','broker_memory'):
        if 'rabbitmq' in (checks[name]['expr']+' '+checks[name]['age_expr']).lower(): raise ValueError('wrong broker metric: expected local session broker, not RabbitMQ')
    # Cluster totals cannot establish that each participating node has headroom.
    for name in ('cluster_cpu','cluster_memory','cluster_network'):
        if any(not (row.get('instance') or row.get('node')) for row in checks[name]['expected_labels']): raise ValueError('per-node telemetry required: '+name)
    for name in GENERATOR|{'database','storage','gateway','pending_pods','cluster_cpu','cluster_memory','cluster_network'}:
        if checks[name].get('observe_only') or not any(k in checks[name] for k in ('min','max')): raise ValueError('missing headroom bounds: '+name)
    results={n:evaluate(c,query,start,end,step) for n,c in checks.items()}
    if not isinstance(vector,dict) or not vector: raise ValueError('missing expected replica vector')
    replicas={}
    expected=[{'namespace':ns,'deployment':name} for name in vector]
    for metric in ('kube_deployment_spec_replicas','kube_deployment_status_replicas_ready','kube_deployment_status_replicas'):
        sel=metric+'{namespace='+json.dumps(ns)+'}'
        try:
            # Subtract the expected count per deployment after checking all series.
            # Aggregate scrape duplicates using both min and max to detect disagreement.
            age='max by(namespace,deployment) (max_over_time((time() - timestamp('+sel+'))[95s:5s]))'
            ages=matrix_values(query(age,start,end,step),expected,start,end,step)
            if min(ages)<-1 or max(ages)>float(policy['replica_max_age_seconds']): raise ValueError('stale replica evidence')
            for agg in ('min','max'):
                rows=query(agg+' by(namespace,deployment) ('+sel+')',start,end,step)
                matrix_values(rows,expected,start,end,step)
                for row in rows:
                    name=row['metric']['deployment']; vals=[float(v) for _,v in row['values']]
                    replicas[name+'/'+metric+'/'+agg]={'verified':True,'within_bounds':all(v==vector[name] for v in vals),'min':min(vals),'max':max(vals),'expected':vector[name]}
        except Exception as exc:
            replicas[metric]={'verified':False,'within_bounds':False,'error_class':type(exc).__name__}
    def passed(rs): return all(x['verified'] and x['within_bounds'] for x in rs)
    return {'generator_healthy':passed(results[n] for n in GENERATOR),'runtime_headroom':passed(results[n] for n in set(results)-GENERATOR-{'application_throttling'}), 'replicas_stable':passed(replicas.values()) and passed([results['pending_pods']]),'telemetry_verified':all(r['verified'] for r in [*results.values(),*replicas.values()]),'interval':{'start':start,'end':end,'step':step,'measurement_seconds':duration},'checks':results,'replicas':replicas}

def evidence(ns,run,policy,query):
    from analyze_run import summarize
    run=pathlib.Path(run); summary=summarize(run)
    if not summary['complete']: raise ValueError('incomplete business probe')
    vector=json.loads((run.parent/(run.name+'.replicas.json')).read_text())
    result=interval_evidence(ns,policy,query,summary['measurement_origin'],summary['planned_seconds'],vector)
    m=json.loads((run/'measurement_summary.json').read_text())['metrics']
    def value(name,field):
        item=m.get(name,{}); raw=item.get(field,item.get('values',{}).get(field)); return float(raw) if raw is not None else None
    peak=value('vus','max'); maximum=value('vus_max','value'); dropped=summary['dropped']
    vu=bool(dropped>0 and peak is not None and maximum is not None and math.isfinite(peak) and math.isfinite(maximum) and maximum>0 and peak>=.99*maximum)
    result.update(vu_exhaustion=vu,vu_peak=peak,vu_max=maximum,dropped=dropped)
    return result

def main():
    sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
    p=argparse.ArgumentParser(); p.add_argument('namespace');p.add_argument('run');p.add_argument('--policy',required=True);a=p.parse_args()
    policy=json.loads(pathlib.Path(a.policy).read_text()); base=os.environ.get('MIMIR') or policy['base_url'];tenant=os.environ.get('MIMIR_TENANT')
    raw=pathlib.Path(a.run)/'interval_telemetry.jsonl'
    with raw.open('x') as log:
        def recorded(expr,s,e,step):
            rows=query_range(base,expr,s,e,step,tenant)
            log.write(json.dumps({'expr':expr,'start':s,'end':e,'step':step,'series':rows})+'\n');log.flush()
            return rows
        result=evidence(a.namespace,a.run,policy,recorded)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
