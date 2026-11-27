// Merged benchmark: independent auth broker; open-loop minute buckets only.
import http from 'k6/http';
import { check } from 'k6';
import { Counter, Gauge, Trend } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import exec from 'k6/execution';
import { classify } from './contracts.js';
const HOST=__ENV.HOST, BROKER=__ENV.BROKER_URL, KEY=__ENV.BROKER_KEY;
if (!HOST || !BROKER || !KEY) throw new Error('Use experiment.py / launch_k6.sh');
function integer(name, fallback, min=0) { const n=Number(__ENV[name]||fallback); if(!Number.isSafeInteger(n)||n<min) throw new Error('invalid '+name); return n; }
const PRE=integer('PRE_ALLOCATED_VUS',0,1), MAX=integer('MAX_VUS',PRE,1);
if(MAX!==PRE) throw new Error('fixed VU budget required');
const PLAN=JSON.parse(open(__ENV.RATE_PLAN));
const SEEDS=new SharedArray('seeds',()=>JSON.parse(open(__ENV.SEED_FILE)));
if(SEEDS.length<PRE) throw new Error('insufficient seed accounts');
const FIXTURES=JSON.parse(open(__ENV.TRAVEL_FIXTURES));
const DATE=__ENV.TRAVEL_DATE;
if(!DATE || !/^\d{4}-\d{2}-\d{2}$/.test(DATE)) throw new Error('TRAVEL_DATE required');
function travelDate(offset){const d=new Date(DATE+'T00:00:00Z');d.setUTCDate(d.getUTCDate()+offset);return d.toISOString().slice(0,10);}
const stages=[]; let expected=0, peak=0;
for(const bucket of PLAN.buckets){ if(!Number.isSafeInteger(bucket.rate_per_min)||bucket.rate_per_min<0||!Number.isSafeInteger(bucket.seconds)||bucket.seconds<=0)throw new Error('invalid rate bucket'); stages.push({duration:'0s',target:bucket.rate_per_min},{duration:`${bucket.seconds}s`,target:bucket.rate_per_min}); expected+=bucket.rate_per_min*bucket.seconds/60; peak=Math.max(peak,bucket.rate_per_min); }
export const options={scenarios:{load:{executor:'ramping-arrival-rate',timeUnit:'1m',startRate:PLAN.buckets[0].rate_per_min,stages,preAllocatedVUs:PRE,maxVUs:MAX,gracefulStop:__ENV.DRAIN||'120s'}},systemTags:['proto','status','method','name','group','check','error_code','scenario'],summaryTrendStats:['avg','min','med','max','p(90)','p(95)','p(99)']};
const started=new Counter('iterations_started_total'), finished=new Counter('business_finished_total'), outcome=new Counter('business_outcome_total'), actionStarted=new Counter('business_action_started_total'), duration=new Trend('business_duration_ms',true), appDuration=new Trend('app_http_duration_ms',true), appOutcomes=new Counter('app_http_outcome_total'), expectedGauge=new Gauge('scheduled_iterations_expected'), peakGauge=new Gauge('scheduled_peak_per_min'), workloadFallback=new Counter('workload_fallback_total'), brokerErrors=new Counter('broker_errors_total');
const correlationPresent=new Counter('correlation_mesh_id_present_total'), correlationMissing=new Counter('correlation_mesh_id_missing_total');
const labAttempted=new Counter('lab_attempted'), labGood=new Counter('lab_good'), labSlow=new Counter('lab_slow_success'), labFailed=new Counter('lab_failed'), labTimeout=new Counter('lab_timeout');
let current=null, rng=1, seq=0, ST=null;
function random(){rng=(rng+0x6D2B79F5)|0;let t=Math.imul(rng^(rng>>>15),1|rng);t=(t+Math.imul(t^(t>>>7),61|t))^t;return ((t^(t>>>14))>>>0)/4294967296;}
function pick(a){return a[Math.floor(random()*a.length)];}
function skip(reason){if(current && !current.skip) current.skip=reason;}
function readSession(key){const r=http.get(BROKER+'/session/'+key,{headers:{'X-Broker-Key':KEY},tags:{name:'local-session-broker',traffic:'broker'},timeout:'2s'});if(r.status!==200){brokerErrors.add(1);return null;}try{return r.json().token||null;}catch(e){brokerErrors.add(1);return null;}}
function ensureLogin(st){return !!(st.sess&&st.sess.token);}
function H(st){const token=st&&(st.token||(st.sess&&st.sess.token));return token?{Authorization:'Bearer '+token}:{};}
function request(method,path,body,st,tag,contract,expectedValue){
 const id=`${__ENV.RUNID||'run'}-${__VU}-${__ITER}-${++seq}`; const name=tag||path; const begin=Date.now();
 const r=http.request(method,HOST+path,body===null?null:JSON.stringify(body),{headers:Object.assign({'Content-Type':'application/json','X-Request-ID':id},H(st)),tags:{name,traffic:'application'},timeout:'30s',redirects:0});
 const mesh=r.headers['X-Request-Id']||r.headers['X-Request-ID'];if(mesh)correlationPresent.add(1);else correlationMissing.add(1);
 if(__ENV.CORRELATION_LOG==='1')console.log(`CORREL name=${name} client_id=${id} mesh_id=${mesh||'none'} status=${r.status}`);
 let payload=null,parseOK=true;try{payload=r.json();}catch(e){parseOK=false;}
 const type=classify(r.status,payload,contract,expectedValue,parseOK);
 appDuration.add(Date.now()-begin,{name});appOutcomes.add(1,{name,outcome:type});
 if(current&&type!=='success'&&!current.failure)current.failure=type;return r;
}
function get(path,st,tag,contract,expected){return request('GET',path,null,st,tag,contract,expected);}
function post(path,body,st,tag,contract,expected){return request('POST',path,body,st,tag,contract,expected);}
const TRIPS = ['G1234', 'G1235', 'G1236'];
const ODS = [
  ['shanghai', 'taiyuan'],
  ['nanjing', 'shanghai'],
  ['suzhou', 'shanghai'],
];
const ROUTES = ['0b23bd3e-876a-4af3-b920-c50a90c90b04'];

function a_travel(st) {
  const f=pick(FIXTURES), date=travelDate(Math.floor(random()*3));
  post(`/api/v1/${f.service}/trips/left`,{startPlace:f.origin,endPlace:f.destination,departureTime:date},st,f.service==='travelservice'?'travel-query':'travel2-query','travel',f.trip_id);
  get(`/api/v1/${f.service}/trips/${f.trip_id}`,st,'trip-detail','trip',f.trip_id);
}
function a_static(st) {
  get(`/api/v1/routeservice/routes/${pick(ROUTES)}`, st, 'route-detail');
  get('/api/v1/trainservice/trains', st);
  get('/api/v1/stationservice/stations/name/shanghai', st);
  get('/api/v1/priceservice/prices', st);
  get('/api/v1/configservice/configs', st);
  get('/api/v1/basicservice/welcome', st);
  get('/api/v1/routeplanservice/welcome', st);
  get('/api/v1/travelplanservice/welcome', st);
}
function a_menus(st) {
  if (!ensureLogin(st)) { skip('no_session'); return; }
  get('/api/v1/assuranceservice/assurances', st);
  get('/api/v1/assuranceservice/assurances/types', st);
  get('/api/v1/foodservice/welcome', st);
  get('/api/v1/foodservice/orders', st);
  // FIX(stale-date): normalize to the same future window the rest of the
  // workload uses (a_preserve/a_pay send 2027-05-2x). Verified: FoodServiceImpl
  // .getAllFood does NOT filter by date (it fans out to trainfood / travel /
  // stationfood by tripId+stations; the date is only logged) and the past date
  // returned 200 when cold — the run-time 500s on this path were transient
  // downstream failures inside that REST fan-out, not a date bug.
  get(`/api/v1/foodservice/foods/${travelDate(0)}/shanghai/taiyuan/G1234`, st, 'food-by-trip');
  // FIX(4xx-drift): GET /welcome hits the ui nginx SPA root, which has no such
  // static file -> 404 on every a_menus iteration (~6% of all edge requests).
  // /index.html is the same SPA entry and returns 200.
  get('/index.html', st);
  get('/api/v1/consignservice/welcome', st);
  get('/api/v1/consignpriceservice/welcome', st);
  get('/api/v1/securityservice/securityConfigs', st);
  if (st.account) get(`/api/v1/contactservice/contacts/account/${encodeURIComponent(st.account)}`, st, 'user-contacts','list');
}
function a_leaf(st) {
  if (!ensureLogin(st)) { skip('no_session'); return; }
  get('/api/v1/seatservice/welcome', st);
  get('/api/v1/paymentservice/welcome', st);
  get('/api/v1/paymentservice/payment', st);
  get('/api/v1/executeservice/welcome', st);
  get('/api/v1/rebookservice/welcome', st);
  get('/office/getAll', st);
  get('/office/getRegionList', st);
  get('/api/v1/newsservice/welcome', st);
  // FIX(4xx-drift): POST /api/v1/avatar lands on the ui nginx (the gateway has
  // no avatarservice route in this deployment; the SPA itself never calls it)
  // and nginx answers 400 -> removed. Verified live: /api/v1/avatarservice/*
  // is not routable either (404).
  get('/api/v1/verifycode/generate/', st);
  get('/', st);
}
function a_admin(st) {
  // FIX(4xx-drift): the admin* reads are gated by Spring Security
  // (SecurityConfig: .hasRole("ADMIN")) — the original flow sent the regular
  // user's ROLE_USER JWT -> 403 on every admintravel/adminroute/adminusers/
  // adminorder call (~2.8% of all requests). Use the seeded admin account.
  // The admin session is renewed from its own exp claim with independent
  // jitter, so admin renewals never synchronize with each other or with users.
  if (!st.admin.token) { skip('no_admin_session'); return; }
  const s = { token: st.admin.token };
  const r1 = get('/api/v1/admintravelservice/admintravel', s, 'admintravel');
  if (r1.status === 403) { st.admin.token = null; return; }
  get('/api/v1/adminrouteservice/adminroute', s, 'adminroute');
  get('/api/v1/adminuserservice/users', s, 'adminusers');
  get('/api/v1/adminorderservice/adminorder', s, 'adminorder');
  get('/api/v1/adminbasicservice/adminbasic/contacts', s, 'adminbasic-contacts');
  get('/api/v1/adminbasicservice/adminbasic/stations', s, 'adminbasic-stations');
}
function a_orders(st) {
  if (!ensureLogin(st)) { skip('no_session'); return; }
  post('/api/v1/orderservice/order/query', { loginId: st.account }, st, 'orders-query','list');
  const r = post('/api/v1/orderOtherService/orderOther/query', { loginId: st.account }, st, 'orderother-query','list');
  if (r.status === 200) {
    try {
      const data = r.json().data || [];
      data.slice(0, 3).forEach((o) => { if (o && o.id && !st.orders.includes(o.id)) st.orders.push(o.id); }); st.orders = st.orders.slice(-100);
    } catch (e) { }
  }
}
function a_preserve(st) {
  if (!ensureLogin(st) || !st.contacts) {
    workloadFallback.add(1, { action: 'a_preserve', reason: st.contacts ? 'no_session' : 'no_contacts' });
    skip('missing_reservation_prerequisite'); return;
  }
  const [o, d] = pick(ODS.slice(0, 3));
  const r = post('/api/v1/preserveservice/preserve', {
    accountId: st.account, contactsId: st.contacts, tripId: pick(TRIPS.slice(0, 3)),
    seatType: pick([2, 3]), date: travelDate(Math.floor(random()*3)),
    from: o, to: d, assurance: 0,
  }, st, 'preserve');
  if (r.status === 200) a_orders(st);
}
function a_pay(st) {
  if (!ensureLogin(st) || !st.orders.length) {
    workloadFallback.add(1, { action: 'a_pay', reason: st.orders.length ? 'no_session' : 'no_orders' });
    skip('missing_order_prerequisite'); return;
  }
  const oid = pick(st.orders);
  let price = '100.0';
  const pr = get(`/api/v1/orderservice/order/price/${oid}`, st, 'order-price');
  if (pr.status === 200) { try { price = String(pr.json().data || price); } catch (e) { } }
  post('/api/v1/inside_pay_service/inside_payment', { userId: st.account, orderId: oid, tripId: pick(TRIPS.slice(0, 3)), price }, st, 'inside-payment');
  get(`/api/v1/executeservice/execute/collected/${oid}`, st, 'execute-collected');
  get(`/api/v1/executeservice/execute/execute/${oid}`, st, 'execute-execute');
  if (random() < 0.5) get(`/api/v1/cancelservice/cancel/${oid}/${st.account}`, st, 'cancel');
  if (random() < 0.3 && st.orders.length) {
    post('/api/v1/rebookservice/rebook', { orderId: pick(st.orders), oldTripId: 'G1234', tripId: 'G1235', seatType: 2, date: travelDate(Math.floor(random()*3)) }, st, 'rebook');
  }
}


function a_voucher(st) {
  // FIX(stage-12k): the /getVoucher API contract is {orderId, voucherId} — the
  // original call sent {userId, voucherId} which always raised KeyError('orderId')
  // -> 500 at ~4.5/s for the whole previous run. Gate like a_pay/a_menus and send
  // a real order id once the VU has logged in and listed orders.
  if (!ensureLogin(st) || !st.orders.length) { skip('missing_order_prerequisite'); return; }
  post('/getVoucher', { orderId: pick(st.orders), voucherId: 'VOUCHER1' }, st, 'getVoucher');
}

function a_preserve_other(st) {
  if (!ensureLogin(st) || !st.contacts) {
    workloadFallback.add(1, { action: 'a_preserve_other', reason: st.contacts ? 'no_session' : 'no_contacts' });
    skip('missing_reservation_prerequisite'); return;
  }
  const [o, d] = pick(ODS);
  const r = post('/api/v1/preserveotherservice/preserveOther', {
    accountId: st.account, contactsId: st.contacts, tripId: pick(TRIPS),
    ticketType: 2, date: travelDate(Math.floor(random()*3)),
    from: o, to: d,
  }, st, 'preserve-other');
  if (r.status === 200) a_orders(st);
}

function a_user_profile(st) {
  // FIX(4xx-drift): /api/v1/userservice/user/{info,profile,contacts} don't
  // exist in this deployment (UserController maps /users/…, contacts live in
  // ts-contact-service) -> all three 404'd on every call. Verified live 200:
  // GET /userservice/users/{userName}, GET /userservice/users/id/{accountId},
  // GET /contactservice/contacts/account/{accountId}.
  if (!ensureLogin(st)) { skip('no_session'); return; }
  if (st.username) get(`/api/v1/userservice/users/${encodeURIComponent(st.username)}`, st, 'user-info','user',st.account);
  if (st.account) {
    get(`/api/v1/userservice/users/id/${encodeURIComponent(st.account)}`, st, 'user-profile','user',st.account);
    get(`/api/v1/contactservice/contacts/account/${encodeURIComponent(st.account)}`, st, 'user-contacts','list');
  }
}

function a_cancel(st) {
  if (!ensureLogin(st) || !st.orders.length) { skip('missing_order_prerequisite'); return; }
  const oid = pick(st.orders);
  get(`/api/v1/cancelservice/cancel/${oid}/${st.account}`, st, 'cancel');
}

function a_verifycode(st) {
  get('/api/v1/verifycode/generate/', st);
  // FIX(stage-12k): dropped GET /api/v1/verifycode/verify/ — that endpoint does
  // not exist on ts-verification-code-service (returns Spring 404), so the
  // request fell through ui nginx to the static root and 404'd at ~4.6/s all run.
}

function a_inside_payment(st) {
  if (!ensureLogin(st) || !st.orders.length) { skip('missing_order_prerequisite'); return; }
  const oid = pick(st.orders);
  let price = '100.0';
  const pr = get(`/api/v1/orderservice/order/price/${oid}`, st, 'order-price');
  if (pr.status === 200) { try { price = String(pr.json().data || price); } catch (e) { } }
  post('/api/v1/inside_pay_service/inside_payment', { userId: st.account, orderId: oid, tripId: pick(TRIPS.slice(0, 3)), price }, st, 'inside-payment');
}



const coverage=[[a_travel,10],[a_static,6],[a_menus,5],[a_leaf,4],[a_orders,5],[a_preserve,3],[a_pay,3],[a_admin,2],[a_voucher,2],[a_preserve_other,2],[a_user_profile,2],[a_cancel,2],[a_verifycode,2],[a_inside_payment,2]];
// Read-heavy default avoids state drift and unvalidated mutation paths. Coverage is explicit experimental mode.
const business=[[a_travel,55],[a_orders,25],[a_user_profile,20]];
const profile=__ENV.WORKLOAD_PROFILE||'business';if(profile!=='business')throw new Error('coverage profile requires separate endpoint contracts and fixture validation before use');
const BAG=[];for(const [fn,w] of (profile==='business'?business:coverage))for(let i=0;i<w;i++)BAG.push(fn);
const publicActions=new Set([a_travel,a_static,a_verifycode]);
export function setup(){labAttempted.add(0);labGood.add(0);labSlow.add(0);labFailed.add(0);labTimeout.add(0);expectedGauge.add(expected);peakGauge.add(peak);if(__ENV.PHASE==='measurement'){const r=http.post(BROKER+'/begin',JSON.stringify({duration:PLAN.buckets.reduce((s,b)=>s+b.seconds,0)}),{headers:{'X-Broker-Key':KEY,'Content-Type':'application/json'},tags:{name:'local-session-broker',traffic:'broker'},timeout:'5s'});if(r.status!==200)throw new Error('broker could not begin independent schedule');}return {};}
export default function(){
 if(!ST){const s=SEEDS[exec.vu.idInTest-1];if(!s)throw new Error('VU outside session pool');ST={username:s.username,account:s.accountId,contacts:s.contactsId,orders:[],sess:{token:null},admin:{token:null}};}
 rng=(Math.imul(exec.scenario.iterationInTest+1,2654435761)^integer('WORKLOAD_SEED',42))|0;
 const action=pick(BAG), name=action.name;current={failure:null,skip:null};const begin=Date.now();started.add(1);actionStarted.add(1,{action:name});ST.sess.token=null;ST.admin.token=null;
 try{if(action===a_admin){ST.admin.token=readSession('admin');if(!ST.admin.token)skip('no_admin_session');}else if(!publicActions.has(action)){ST.sess.token=readSession(String(exec.vu.idInTest-1));if(!ST.sess.token)skip('no_session');}if(!current.skip)action(ST);}catch(e){current.failure=current.failure||'script_exception';}
 finally{const elapsed=Date.now()-begin;const result=current.skip?'skipped_'+current.skip:current.failure|| (elapsed>Number(__ENV.BUSINESS_SLO_MS||2000)?'late':'on_time');outcome.add(1,{action:name,outcome:result});duration.add(elapsed,{action:name});finished.add(1);labAttempted.add(1);if(result==='on_time')labGood.add(1);else if(result==='late')labSlow.add(1);else if(!result.startsWith('skipped_')){labFailed.add(1);if(result.toLowerCase().includes('timeout'))labTimeout.add(1);}current=null;}
}
