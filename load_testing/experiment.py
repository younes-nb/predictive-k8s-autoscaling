#!/usr/bin/env python3
"""Run on the dedicated load-generator node. No controller or database mutation here."""
import argparse, csv, hashlib, json, math, os, pathlib, subprocess, sys, time
from urllib.request import Request, urlopen
from session_manager import Broker
from contracts import validate
from trace_design import read_trace, digest
HERE=pathlib.Path(__file__).resolve().parent

def plan_trace(path, start, minutes, reference_ips):
    ts,x,observed=read_trace(path)
    if start<0 or minutes<=0 or start+minutes>len(x) or not all(observed[start:start+minutes]): raise ValueError('invalid slice or unobserved minutes; no wrapping')
    if not math.isfinite(reference_ips) or reference_ips<=0: raise ValueError('invalid reference rate')
    return {'mode':'trace','canonical_sha256':digest(path),'start_min':start,'test_min':minutes,'reference_ips':reference_ips,'buckets':[{'seconds':60,'rate_per_min':math.floor(v*reference_ips*60+.5)} for v in x[start:start+minutes]]}

def constant(rate, seconds):
    if rate<=0 or seconds<=0: raise ValueError('invalid constant plan')
    return {'mode':'constant','buckets':[{'seconds':seconds,'rate_per_min':math.floor(rate*60+.5)}]}

def run_k6(plan, phase, env, out, k6):
    p=out/(phase+'_plan.json'); p.write_text(json.dumps(plan,indent=2)+'\n')
    e=dict(env,RATE_PLAN=str(p),PHASE=phase)
    cmd=[k6,'run','--include-system-env-vars=true','--summary-export',str(out/(phase+'_summary.json')),'-o','json='+str(out/(phase+'_samples.json.gz'))]
    if e.get('K6_RW_SERVER_URL'): cmd+=['-o','experimental-prometheus-rw']
    cmd+=[str(HERE/'k6/trainticket.js')]
    return subprocess.run(cmd,env=e,check=False).returncode

def probe(host,path,body=None,token=None):
    headers={'Content-Type':'application/json'}
    if token: headers['Authorization']='Bearer '+token
    req=Request(host+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    with urlopen(req,timeout=15) as r: return json.load(r)

def preflight(host,date,broker,profile,fixtures):
    from datetime import date as D,timedelta
    from urllib.parse import quote
    from contracts import validate
    if D.fromisoformat(date)<=D.today(): raise ValueError('travel date must be future')
    if profile!='business': raise ValueError('coverage not yet semantically validated; use separate development harness')
    for fixture in fixtures:
        service,origin,dest,trip=(fixture[k] for k in ('service','origin','destination','trip_id'))
        for offset in range(3):
            day=str(D.fromisoformat(date)+timedelta(days=offset)); body={'startPlace':origin,'endPlace':dest,'departureTime':day}
            validate(probe(host,f'/api/v1/{service}/trips/left',body),'travel',trip)
        validate(probe(host,f'/api/v1/{service}/trips/{quote(trip,safe="")}'),'trip',trip)
    for user in broker.users[:min(3,len(broker.users))]:
        token=broker.sessions[user['key']]['token']; aid=user['accountId']; path_id=quote(str(aid),safe='')
        validate(probe(host,f'/api/v1/userservice/users/{quote(user["username"],safe="")}',token=token),'user',aid)
        validate(probe(host,f'/api/v1/userservice/users/id/{path_id}',token=token),'user',aid)
        validate(probe(host,f'/api/v1/contactservice/contacts/account/{path_id}',token=token),'list')
        for endpoint in ('orderservice/order/query','orderOtherService/orderOther/query'):
            validate(probe(host,f'/api/v1/{endpoint}',{'loginId':aid},token),'list')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True); p.add_argument('--out',required=True); a=p.parse_args(); cfg=json.load(open(a.config)); out=pathlib.Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    host=cfg['host'].rstrip('/'); seed=pathlib.Path(cfg['seed_file']).resolve(); raw=json.load(seed.open()); vus=int(cfg['vus']); session_count=int(cfg.get('session_count',vus)); extra_count=int(cfg.get('new_login_accounts',0)); auth=cfg['auth']; profile=cfg.get('workload_profile','business');
    if vus<1 or session_count<vus or extra_count<0 or len(raw)<session_count+extra_count: raise ValueError('insufficient unique seed accounts')
    if auth['login_rate']>0 and extra_count<1: raise ValueError('new login accounts required')
    selected=raw[:session_count+extra_count];
    if any(not x.get('username') or not x.get('accountId') for x in selected) or len({x['username'] for x in selected})!=len(selected) or len({x['accountId'] for x in selected})!=len(selected): raise ValueError('invalid or duplicate seed identity')
    if profile=='coverage' and any(not u.get('contactsId') for u in selected[:session_count]): raise ValueError('coverage requires valid contacts')
    pw=os.environ.get('TT_USER_PASSWORD'); apw=os.environ.get('TT_ADMIN_PASSWORD'); admin_name=os.environ.get('TT_ADMIN_USER','admin')
    if not pw or not apw: raise ValueError('set TT_USER_PASSWORD and TT_ADMIN_PASSWORD in environment')
    if admin_name in {x['username'] for x in selected}: raise ValueError('admin must be distinct')
    users=[dict(username=x['username'],accountId=x['accountId'],password=pw,key=str(i)) for i,x in enumerate(selected[:session_count])]; extra=[dict(username=x['username'],accountId=x['accountId'],password=pw,key='extra-'+str(i)) for i,x in enumerate(selected[session_count:])]; admin={'username':admin_name,'password':apw,'key':'admin'}
    fixtures=cfg['travel_fixtures'];
    from datetime import date as D
    if D.fromisoformat(cfg['travel_date'])<=D.today(): raise ValueError('travel_date must be in the future')
    if profile!='business' and not cfg.get('development_only',False): raise ValueError('unvalidated coverage profile prohibited in official runs')
    if not fixtures or any(f.get('service') not in ('travelservice','travel2service') or not all(f.get(k) for k in ('origin','destination','trip_id')) for f in fixtures): raise ValueError('provide validated travel fixture list')
    w=cfg['workload'];
    if w['mode']=='constant': plan=constant(w['ips'],w['seconds'])
    elif w['mode']=='trace':
        if 'slice_manifest' in w:
            m=json.load(open(w['slice_manifest'])); role=w['slice_role']; s=m[role]
            if digest(w['csv'])!=m['canonical_sha256']: raise ValueError('trace hash mismatch')
            if m['training']['start_min']+m['training']['test_min']>m['comparison']['start_min']: raise ValueError('training overlaps comparison')
            start,mins=s['start_min'],s['test_min']
        else: start,mins=w['start_min'],w['test_min']
        plan=plan_trace(w['csv'],start,mins,w['reference_ips'])
    else: raise ValueError('invalid workload mode')
    # Write only non-secret configuration and sanitized seeds. Avoid retaining seed JWTs.
    os.chmod(out,0o700); safe_seed=out/'business_seeds.json'; safe_seed.write_text(json.dumps([{k:x.get(k) for k in ('username','accountId','contactsId')} for x in selected[:session_count]])); (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n'); (out/'manifest.json').write_text(json.dumps({'seed_sha256':digest(seed),'plan':plan,'vus':vus,'session_count':session_count,'new_login_accounts':extra_count},indent=2)+'\n')
    fixture_path=out/'travel_fixtures.json'; fixture_path.write_text(json.dumps(fixtures))
    with (out/'auth_events.jsonl').open('w') as log:
        broker=Broker(host,users,admin,extra,log,**auth); server=None; rc=None; activation_attempted=False; error=None
        try:
            broker.prepare(); server=broker.server();
            if cfg.get('preflight',True): preflight(host,cfg['travel_date'],broker,profile,fixtures)
            elif not cfg.get('development_only',False): raise ValueError('preflight bypass requires development_only=true')
            env=dict(os.environ,HOST=host,BROKER_URL='http://127.0.0.1:'+str(server.server_port),BROKER_KEY=broker.secret,SEED_FILE=str(safe_seed),PRE_ALLOCATED_VUS=str(vus),MAX_VUS=str(vus),TRAVEL_DATE=cfg['travel_date'],TRAVEL_FIXTURES=str(fixture_path),WORKLOAD_SEED=str(cfg.get('workload_seed',42)),WORKLOAD_PROFILE=profile,BUSINESS_SLO_MS=str(cfg.get('business_slo_ms',2000)),DRAIN=cfg.get('drain','120s'),RUNID=out.name,TRACE_COMPRESS='1')
            env.pop('TT_USER_PASSWORD',None); env.pop('TT_ADMIN_PASSWORD',None)
            warm=cfg.get('warmup',{'ips':5,'seconds':120}); k6=cfg.get('k6_bin','k6')
            if warm['seconds']>0 and run_k6(constant(warm['ips'],warm['seconds']),'warmup',env,out,k6): raise RuntimeError('warmup k6 failed')
            if not broker.ready(): raise RuntimeError('initial tokens cannot cover first renewal slots')
            if cfg.get('before_measurement_command'):
                if not cfg.get('after_measurement_command'): raise ValueError('controller activation requires cleanup hook')
                activation_attempted=True
                subprocess.run(cfg['before_measurement_command'],check=True,timeout=cfg.get('hook_timeout_seconds',120))
            if not broker.ready(): raise RuntimeError('initial tokens cannot cover first renewal slots')
            rc=run_k6(plan,'measurement',env,out,k6); return rc
        except BaseException as e:
            error=type(e).__name__; raise
        finally:
            cleanup_error=None
            try:
                if activation_attempted:
                    subprocess.run(cfg['after_measurement_command'],check=True,timeout=cfg.get('hook_timeout_seconds',120))
            except BaseException as e:
                cleanup_error=type(e).__name__; raise
            finally:
                (out/'exit_status.json').write_text(json.dumps({'k6_exit':rc,'measurement_started_at':broker.t0,'finished_at':time.time(),'error':error,'cleanup_error':cleanup_error}))
                broker.close()
                if server: server.shutdown(); server.server_close()
if __name__=='__main__': sys.exit(main())
