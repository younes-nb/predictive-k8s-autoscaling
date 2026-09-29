import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

const HOST = __ENV.HOST || 'https://train-ticket.younesnb.linkpc.net';
const MAXR = parseInt(__ENV.MAX_REQUESTS || '5000', 10);
const POOL = parseInt(__ENV.USER_POOL || '500', 10);
const START_MIN = parseInt(__ENV.START_MIN || '0', 10);
const TEST_MIN = parseInt(__ENV.TEST_MIN || '0', 10);
const PW = __ENV.PW || 'loadtest123';

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

const NMIN = TEST_MIN > 0 ? TEST_MIN : MCR.length;
export const options = {
  vus: POOL,
  duration: `${NMIN}m`,


  systemTags: ['proto', 'status', 'method', 'name', 'group', 'check', 'error', 'error_code', 'tls_version'],
  thresholds: {
    http_req_failed: ['rate<0.35'],
    http_req_duration: ['p(95)<8000'],
  },
};

const TRIPS = ['G1234', 'G1235', 'G1236', 'D1345'];
const ODS = [['nanjing', 'shanghai'], ['nanjing', 'suzhou'], ['suzhou', 'shanghai'], ['wuxi', 'shanghai']];
const ROUTES = ['0b23bd3e-876a-4af3-b920-c50a90c90b04', '92708982-77af-4318-be25-57ccb0ff69ad'];

function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

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
  return http.get(`${HOST}${path}`, { headers: Object.assign({ 'Content-Type': 'application/json' }, st ? H(st) : {}), tags: { name: tag || path }, timeout: '30s' });
}
function post(path, body, st, tag) {
  return http.post(`${HOST}${path}`, JSON.stringify(body), { headers: Object.assign({ 'Content-Type': 'application/json' }, st ? H(st) : {}), tags: { name: tag || path }, timeout: '30s' });
}

function a_travel(st) {
  const [o, d] = pick(ODS);
  const day = 27 + Math.floor(Math.random() * 3);
  const b = { startPlace: o, endPlace: d, departureTime: `2026-09-${String(day).padStart(2, '0')}` };
  post('/api/v1/travelservice/trips/left', b, st);
  post('/api/v1/travel2service/trips/left', b, st);
  get(`/api/v1/travelservice/trips/${pick(TRIPS)}`, st);
  post('/api/v1/travel2service/trips/left', b, st);
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
    seatType: pick([2, 3]), date: `2026-09-${27 + Math.floor(Math.random() * 3)}`,
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
  if (Math.random() < 0.25) get(`/api/v1/cancelservice/cancel/${oid}/${st.account}`, st);
  if (Math.random() < 0.15 && st.orders.length) {
    post('/api/v1/rebookservice/rebook', { orderId: pick(st.orders), oldTripId: 'G1234', tripId: 'G1235', seatType: 2, date: `2026-09-${27 + Math.floor(Math.random() * 3)}` }, st);
  }
}

const ACTS = [
  [a_travel, 10], [a_static, 6], [a_menus, 5], [a_leaf, 4], [a_orders, 5],
  [a_preserve, 2], [a_pay, 3], [a_admin, 1],
];
const BAG = [];
ACTS.forEach(([f, w]) => { for (let i = 0; i < w; i++) BAG.push(f); });

export function setup() {
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
    sleep(Math.random() * 10);
    ensureLogin(ST);
  }
  const elapsedMin = Math.floor((Date.now() - data.t0) / 60000);
  const count = MCR[(START_MIN + elapsedMin) % MCR.length] || 0;
  if (count <= 0) { sleep(1); return; }
  const perVU = count / POOL;
  if (Math.random() < 0.03) ensureLogin(ST, true);
  pick(BAG)(ST);
  sleep(Math.max(0.05, 60 / Math.max(1, perVU) - 0.5));
}
export function handleSummary(data) {
  return { stdout: textSummary(data, { compact: true }) };
}
