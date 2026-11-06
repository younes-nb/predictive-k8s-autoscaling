import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

const HOST = __ENV.HOST || 'http://10.10.1.2:30080';
const MAXR = parseInt(__ENV.MAX_REQUESTS || '5000', 10);
const POOL = parseInt(__ENV.USER_POOL || '500', 10);
const START_MIN = parseInt(__ENV.START_MIN || '0', 10);
const TEST_MIN = parseInt(__ENV.TEST_MIN || '0', 10);






const COMPRESS = parseFloat(__ENV.TRACE_COMPRESS || '1', 10);
const PW = __ENV.PW || 'loadtest123';





const OPEN_LOOP = __ENV.OPEN_LOOP === '1';
const PRE_ALLOC = parseInt(__ENV.PRE_ALLOCATED_VUS || '100', 10);



const MCR = new SharedArray('mcr', function () {
  const txt = open(__ENV.MCR_CSV);
  const lines = txt.trim().split('\n').slice(1);
  const vals = lines.map((l) => parseFloat(l.split(',')[2]));
  const peak = Math.max(...vals);
  return vals.map((v) => Math.max(0, Math.round((v / peak) * MAXR)));
});
const SEEDS = new SharedArray('seeds', function () {
  try {
    return JSON.parse(open(__ENV.SEED_FILE));
  } catch (e) {
    return [];
  }
});

const NMIN = TEST_MIN > 0 ? TEST_MIN : Math.ceil(MCR.length / COMPRESS);




function buildArrivalStages() {
  const rows = TEST_MIN > 0 ? TEST_MIN : Math.ceil(MCR.length / COMPRESS);
  const stageSec = 60 / Math.max(COMPRESS, 0.0001);
  const stages = [];
  for (let m = 0; m < rows; m++) {
    const idx = (START_MIN + Math.floor(m * COMPRESS)) % MCR.length;
    const rps = (MCR[idx] * COMPRESS) / 60;
    stages.push({ duration: `${stageSec.toFixed(3)}s`, target: Math.round(rps) });
  }
  return stages;
}
export const options = OPEN_LOOP ? {
  scenarios: {
    // Trace replay only: offered rps = MCR[idx] * TRACE_COMPRESS / 60 where
    // MCR[idx] = round(raw_mcr / peak_raw * MAX_REQUESTS). FLAT_RPS (constant-
    // arrival bypassing the trace) was removed by user directive: the official
    // load axis is the jul13/jul14 NASA day-slice replay.
    load: {
      executor: 'ramping-arrival-rate',
      timeUnit: '1s',
      stages: buildArrivalStages(),
      startRate: 0,
      preAllocatedVUs: PRE_ALLOC,
      maxVUs: POOL,
    },
  },

  systemTags: ['proto', 'status', 'method', 'name', 'group', 'check', 'error', 'error_code', 'tls_version'],
  thresholds: {
    http_req_failed: ['rate<0.35'],
    http_req_duration: ['p(95)<8000'],
  },
} : {
  vus: POOL,
  duration: `${NMIN}m`,

  systemTags: ['proto', 'status', 'method', 'name', 'group', 'check', 'error', 'error_code', 'tls_version'],
  thresholds: {
    http_req_failed: ['rate<0.35'],
    http_req_duration: ['p(95)<8000'],
  },
};


const TRIPS = ['G1234', 'G1235', 'G1236'];
const ODS = [
  ['shanghai', 'taiyuan'],
  ['nanjing', 'shanghai'],
  ['suzhou', 'shanghai'],
];
const ROUTES = ['0b23bd3e-876a-4af3-b920-c50a90c90b04'];

function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

// Pre-flight data validation - runs in setup()
function verifyClusterData() {
  const checks = [
    { name: 'station-shanghai', url: `${HOST}/api/v1/stationservice/stations/id/shanghai`, expect: 200 },
    { name: 'route-shanghai-taiyuan', url: `${HOST}/api/v1/routeservice/routes/0b23bd3e-876a-4af3-b920-c50a90c90b04`, expect: 200 },
    { name: 'travel-query', url: `${HOST}/api/v1/travelservice/trips/left`, method: 'POST', body: JSON.stringify({startPlace:'shanghai',endPlace:'taiyuan',departureTime:'2027-05-04'}), expect: 200 },
    { name: 'basic-welcome', url: `${HOST}/api/v1/basicservice/welcome`, expect: 200 },
  ];
  
  for (const c of checks) {
    const params = {
      method: c.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      timeout: '15s'
    };
    if (c.body) params.body = c.body;
    
    const r = http.request(c.method || 'GET', c.url, c.body || null, params);
    if (r.status !== c.expect) {
      throw new Error(`Pre-flight check failed: ${c.name} returned ${r.status} (expected ${c.expect})`);
    }
    console.log(`Pre-flight OK: ${c.name}`);
  }
  console.log('All pre-flight data checks passed');
}

function logFailedRequest(r, tag, body) {
  if (r.status >= 500 || r.status === 0) {
    console.error(`FAILED ${tag}: status=${r.status} url=${r.url} body=${JSON.stringify(body).slice(0,200)} error=${r.error || r.error_code || 'none'}`);
  }
}

function login(st) {
  const r = http.post(
    `${HOST}/api/v1/users/login`,
    JSON.stringify({ username: st.username, password: PW }),
    { headers: { 'Content-Type': 'application/json' }, timeout: '30s' },
  );
  if (r.status === 200) {
    try {
      const d = r.json().data;
      st.token = d.token; st.account = d.userId;
      return true;
    } catch (e) {  }
  }
  return false;
}
function H(st) { return st.token ? { Authorization: `Bearer ${st.token}` } : {}; }
function ensureLogin(st, force) {
  if (st.token && !force) return true;
  return login(st);
}
function get(path, st, tag) {
  const r = http.get(`${HOST}${path}`, { headers: Object.assign({ 'Content-Type': 'application/json' }, st ? H(st) : {}), tags: { name: tag || path }, timeout: '30s' });
  logFailedRequest(r, tag);
  return r;
}
function post(path, body, st, tag) {
  const r = http.post(`${HOST}${path}`, JSON.stringify(body), { headers: Object.assign({ 'Content-Type': 'application/json' }, st ? H(st) : {}), tags: { name: tag || path }, timeout: '30s' });
  logFailedRequest(r, tag, body);
  return r;
}

function a_travel(st) {
  const [o, d] = pick(ODS);
  const day = 27 + Math.floor(Math.random() * 3);
  const b = { startPlace: o, endPlace: d, departureTime: `2027-05-${String(day).padStart(2, '0')}` };
  const r1 = post('/api/v1/travelservice/trips/left', b, st, 'travel-query');
  check(r1, { 'travel-query ok': r => r.status === 200 && r.json().status === 1 });
  const r2 = post('/api/v1/travel2service/trips/left', b, st);
  check(r2, { 'travel2-query ok': r => r.status === 200 });
  const r3 = get(`/api/v1/travelservice/trips/${pick(TRIPS)}`, st);
  check(r3, { 'trip-detail ok': r => r.status === 200 });
}
function a_static(st) {
  get(`/api/v1/routeservice/routes/${pick(ROUTES)}`, st);
  get('/api/v1/trainservice/trains', st);
  get('/api/v1/stationservice/stations/name/shanghai', st);
  get('/api/v1/priceservice/prices', st);
  get('/api/v1/configservice/configs', st);
  get('/api/v1/basicservice/welcome', st);
  get('/api/v1/routeplanservice/welcome', st);
  get('/api/v1/travelplanservice/welcome', st);
}
function a_menus(st) {
  if (!ensureLogin(st)) return;
  get('/api/v1/assuranceservice/assurances', st);
  get('/api/v1/assuranceservice/assurances/types', st);
  get('/api/v1/foodservice/welcome', st);
  get('/api/v1/foodservice/orders', st);
  get('/api/v1/foodservice/foods/2026-09-27/shanghai/taiyuan/G1234', st);
  get('/welcome', st);
  get('/api/v1/consignservice/welcome', st);
  get('/api/v1/consignpriceservice/welcome', st);
  get('/api/v1/securityservice/securityConfigs', st);
  if (st.account) get(`/api/v1/contactservice/contacts/account/${st.account}`, st);
}
function a_leaf(st) {
  if (!ensureLogin(st)) return;
  get('/api/v1/seatservice/welcome', st);
  get('/api/v1/paymentservice/welcome', st);
  get('/api/v1/paymentservice/payment', st);
  post('/api/v1/paymentservice/payment', { userId: st.account || 'x', orderId: st.orders.length ? pick(st.orders) : 'x', tripId: 'G1234', price: '100.0' }, st);
  get('/api/v1/executeservice/welcome', st);
  get('/api/v1/rebookservice/welcome', st);
  get('/office/getAll', st);
  get('/office/getRegionList', st);
  get('/api/v1/newsservice/welcome', st);
  post('/api/v1/avatar', { img: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==' }, st);
  get('/api/v1/verifycode/generate/', st);
  get('/', st);
}
function a_admin(st) {
  if (!ensureLogin(st)) return;
  get('/api/v1/admintravelservice/admintravel', st);
  get('/api/v1/adminrouteservice/adminroute', st);
  get('/api/v1/adminuserservice/users', st);
  get('/api/v1/adminorderservice/adminorder', st);
  get('/api/v1/adminbasicservice/adminbasic/contacts', st);
  get('/api/v1/adminbasicservice/adminbasic/stations', st);
}
function a_orders(st) {
  if (!ensureLogin(st)) return;
  post('/api/v1/orderservice/order/query', { loginId: st.account }, st);
  const r = post('/api/v1/orderOtherService/orderOther/query', { loginId: st.account }, st);
  if (r.status === 200) {
    try {
      const data = r.json().data || [];
      data.slice(0, 3).forEach((o) => { if (o && o.id) st.orders.push(o.id); });
    } catch (e) {  }
  }
}
function a_preserve(st) {
  if (!ensureLogin(st) || !st.contacts) { a_travel(st); return; }
  const [o, d] = pick(ODS.slice(0, 3));
  const r = post('/api/v1/preserveservice/preserve', {
    accountId: st.account, contactsId: st.contacts, tripId: pick(TRIPS.slice(0, 3)),
    seatType: pick([2, 3]), date: `2027-05-${27 + Math.floor(Math.random() * 3)}`,
    from: o, to: d, assurance: 0,
  }, st);
  if (r.status === 200) a_orders(st);
}
function a_pay(st) {
  if (!ensureLogin(st) || !st.orders.length) { a_orders(st); return; }
  const oid = pick(st.orders);
  let price = '100.0';
  const pr = get(`/api/v1/orderservice/order/price/${oid}`, st);
  if (pr.status === 200) { try { price = String(pr.json().data || price); } catch (e) {  } }
  post('/api/v1/inside_pay_service/inside_payment', { userId: st.account, orderId: oid, tripId: pick(TRIPS.slice(0, 3)), price }, st);
  get(`/api/v1/executeservice/execute/collected/${oid}`, st);
  get(`/api/v1/executeservice/execute/execute/${oid}`, st);
  if (Math.random() < 0.5) get(`/api/v1/cancelservice/cancel/${oid}/${st.account}`, st);
  if (Math.random() < 0.3 && st.orders.length) {
    post('/api/v1/rebookservice/rebook', { orderId: pick(st.orders), oldTripId: 'G1234', tripId: 'G1235', seatType: 2, date: `2027-05-${27 + Math.floor(Math.random() * 3)}` }, st);
  }
}


function a_notifications(st) {
  if (!ensureLogin(st)) return;
  get('/api/v1/notificationservice/notifications', st);
  get('/api/v1/notificationservice/notifications/types', st);
  get(`/api/v1/notificationservice/notifications/account/${st.account}`, st);
}

function a_voucher(st) {
  
  
  
  post('/getVoucher', { userId: st.account, voucherId: 'VOUCHER1' }, st);
}

function a_preserve_other(st) {
  if (!ensureLogin(st) || !st.contacts) { a_travel(st); return; }
  const [o, d] = pick(ODS);
  const r = post('/api/v1/preserveotherservice/preserveOther', {
    accountId: st.account, contactsId: st.contacts, tripId: pick(TRIPS),
    ticketType: 2, date: `2027-05-${27 + Math.floor(Math.random() * 3)}`,
    from: o, to: d,
  }, st);
  if (r.status === 200) a_orders(st);
}

function a_user_profile(st) {
  if (!ensureLogin(st)) return;
  get('/api/v1/userservice/user/info', st);
  get('/api/v1/userservice/user/profile', st);
  get('/api/v1/userservice/user/contacts', st);
}

function a_cancel(st) {
  if (!ensureLogin(st) || !st.orders.length) return;
  const oid = pick(st.orders);
  get(`/api/v1/cancelservice/cancel/${oid}/${st.account}`, st);
}

function a_verifycode(st) {
  get('/api/v1/verifycode/generate/', st);
  get('/api/v1/verifycode/verify/', st);
}

function a_inside_payment(st) {
  if (!ensureLogin(st) || !st.orders.length) return;
  const oid = pick(st.orders);
  let price = '100.0';
  const pr = get(`/api/v1/orderservice/order/price/${oid}`, st);
  if (pr.status === 200) { try { price = String(pr.json().data || price); } catch (e) {  } }
  post('/api/v1/inside_pay_service/inside_payment', { userId: st.account, orderId: oid, tripId: pick(TRIPS.slice(0, 3)), price }, st);
}


const ACTS = [
  [a_travel, 10], [a_static, 6], [a_menus, 5], [a_leaf, 4], [a_orders, 5],
  [a_preserve, 3], [a_pay, 3], [a_admin, 2],
  [a_notifications, 2], [a_voucher, 2], [a_preserve_other, 2],
  [a_user_profile, 2], [a_cancel, 2], [a_verifycode, 2], [a_inside_payment, 2],
];
const BAG = [];
ACTS.forEach(([f, w]) => { for (let i = 0; i < w; i++) BAG.push(f); });

export function setup() {
  verifyClusterData();
  return { t0: Date.now() };
}
function state() {
  const s = SEEDS.length ? SEEDS[__VU % SEEDS.length] : {};
  return {
    username: s.username || `ltpool${__VU}`,
    account: s.accountId || null,
    contacts: s.contactsId || null,
    token: null, orders: [],
  };
}
let ST = null;
export default function (data) {
  if (!ST) {
    ST = state();
    sleep(Math.random() * 2);
    ensureLogin(ST);
  }
  
  get('/', ST);

  if (OPEN_LOOP) {
    
    
    if (Math.random() < 0.03) ensureLogin(ST, true);
    pick(BAG)(ST);
    return;
  }
  const elapsedMin = Math.floor((Date.now() - data.t0) / 60000);
  const traceIdx = Math.floor((START_MIN + elapsedMin * COMPRESS) % MCR.length);
  const count = Math.round(MCR[traceIdx] * COMPRESS) || 0;
  if (count <= 0) { sleep(1); return; }
  const perVU = count / POOL;
  if (Math.random() < 0.03) ensureLogin(ST, true);
  pick(BAG)(ST);
  sleep(Math.max(0.05, 60 / Math.max(1, perVU) - 0.5));
}