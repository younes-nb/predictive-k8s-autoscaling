import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Gauge, Trend } from 'k6/metrics';
import { SharedArray } from 'k6/data';

// ===========================================================================
// TrainTicket load generator — NASA-shaped replay for HPA vs PBScaler.
//
// Official mode (default): open-loop `ramping-arrival-rate`. The NASA MCR
// trace is replayed as stepwise bucket holds: every sampled trace minute holds
// its own integer arrival rate for exactly one (compressed) minute, then jumps
// to the next bucket. No synthetic warm-up and no linear ramps between
// buckets.
//
// Closed-loop mode (explicit EXEC_MODE=closed or OPEN_LOOP=0 only): fixed VUs,
// achieved iteration rate depends on iteration completion time. It is NOT
// equivalent to the open-loop replay and must not be used for the official
// HPA-vs-PBScaler comparison.
//
// VU allocation vs authentication:
//   preAllocatedVUs VUs are created up front as execution capacity. Their
//   default function does not run until the arrival-rate scheduler assigns
//   them an iteration, so pre-allocation performs no login. Each VU creates
//   its session lazily on its first scheduled iteration and reuses it,
//   refreshing once the 45-min refresh threshold is crossed.
//
// Env vars:
//   MCR_CSV SEED_FILE HOST
//   MAX_REQUESTS        peak trace rate in iterations/minute (normalizer peak)
//   START_MIN           first trace row (wraps at CSV end, as before)
//   TEST_MIN            trace rows to replay (0 = whole trace)
//   TRACE_COMPRESS      time compression C: rate x C, hold x 1/C
//   USER_POOL           account-pool / max-VU budget (closed loop: fixed VUs)
//   PRE_ALLOCATED_VUS   open-loop preAllocatedVUs (execution capacity)
//   MAX_VUS             open-loop maxVUs (default = PRE_ALLOCATED_VUS)
//   EXEC_MODE=open|closed ; OPEN_LOOP=1|0 (legacy; unset = open)
//   PREFLIGHT=1         opt-in read-only preflight in setup() (default off)
//   ALLOW_SEED_REUSE=1  permit VU->account sharing (default: hard fail)
//   LOGIN_RETRIES       bounded retries per logical login (default 2)
//   LOGIN_COOLDOWN_MS   per-VU cooldown after a failed login (default 30000)
//   PW ADMIN_USER ADMIN_PW
// ===========================================================================

const HOST = __ENV.HOST || 'http://10.10.1.2:30080';
const PW = __ENV.PW || 'loadtest123';
// FIX(4xx-drift): seeded admin for the a_admin flow — ts-auth-service InitUser
// seeds admin/222222 with ROLE_ADMIN (the regular loadtest users are ROLE_USER).
const ADMIN_USER = __ENV.ADMIN_USER || 'admin';
const ADMIN_PW = __ENV.ADMIN_PW || '222222';

// ---------------------------------------------------------------------------
// Environment parsing (strict: fail in init, before any traffic is sent)
// ---------------------------------------------------------------------------
function envRaw(name) {
  const v = __ENV[name];
  return v === undefined || v === '' ? null : v;
}

function envInt(name, dflt, min) {
  const raw = envRaw(name);
  if (raw === null) return dflt;
  if (!/^-?[0-9]+$/.test(raw)) {
    throw new Error(`${name}='${raw}' is not an integer`);
  }
  const n = parseInt(raw, 10);
  if (!isFinite(n) || n < min) {
    throw new Error(`${name}=${n} is invalid (must be an integer >= ${min})`);
  }
  return n;
}

function envNum(name, dflt, minExclusive) {
  const raw = envRaw(name);
  if (raw === null) return dflt;
  const n = Number(raw);
  if (!isFinite(n) || n <= minExclusive) {
    throw new Error(`${name}='${raw}' is invalid (must be a finite number > ${minExclusive})`);
  }
  return n;
}

// Mode selection. Unset OPEN_LOOP resolves to OPEN (the official benchmark),
// never to closed loop. An explicit EXEC_MODE wins, and a contradictory
// legacy OPEN_LOOP value is a hard error rather than a silent override.
const EXEC_MODE_RAW = (envRaw('EXEC_MODE') || '').toLowerCase();
const OPEN_LOOP_ENV = envRaw('OPEN_LOOP');
let OPEN_LOOP;
if (EXEC_MODE_RAW) {
  if (EXEC_MODE_RAW !== 'open' && EXEC_MODE_RAW !== 'closed') {
    throw new Error(`EXEC_MODE='${EXEC_MODE_RAW}' is invalid (use 'open' or 'closed')`);
  }
  OPEN_LOOP = EXEC_MODE_RAW === 'open';
  if (OPEN_LOOP_ENV !== null && (OPEN_LOOP_ENV === '1') !== OPEN_LOOP) {
    throw new Error(`EXEC_MODE='${EXEC_MODE_RAW}' contradicts OPEN_LOOP='${OPEN_LOOP_ENV}'`);
  }
} else if (OPEN_LOOP_ENV !== null) {
  if (OPEN_LOOP_ENV !== '0' && OPEN_LOOP_ENV !== '1') {
    throw new Error(`OPEN_LOOP='${OPEN_LOOP_ENV}' is invalid (use 1 or 0)`);
  }
  OPEN_LOOP = OPEN_LOOP_ENV === '1';
} else {
  OPEN_LOOP = true;
}

const MAXR = envInt('MAX_REQUESTS', 5000, 1);
const POOL = envInt('USER_POOL', 500, 1);
const START_MIN = envInt('START_MIN', 0, 0);
const TEST_MIN = envInt('TEST_MIN', 0, 0);
const COMPRESS = envNum('TRACE_COMPRESS', 1, 0);
const PRE_ALLOC = envInt('PRE_ALLOCATED_VUS', 100, 1);
const MAX_VUS = envInt('MAX_VUS', PRE_ALLOC, 1);
const PREFLIGHT = __ENV.PREFLIGHT === '1';
const ALLOW_SEED_REUSE = __ENV.ALLOW_SEED_REUSE === '1';
const LOGIN_RETRIES = envInt('LOGIN_RETRIES', 2, 0);
const LOGIN_COOLDOWN_MS = envInt('LOGIN_COOLDOWN_MS', 30 * 1000, 0);
// CORRELATION_LOG=1 emits one CORREL line per request with BOTH the k6 client
// id and the mesh-issued X-Request-ID. Off by default: at benchmark volume a
// per-request console line is prohibitive. The correlation test uses it.
const CORRELATION_LOG = __ENV.CORRELATION_LOG === '1';

if (LOGIN_RETRIES > 5) {
  throw new Error(`LOGIN_RETRIES=${LOGIN_RETRIES} is too high (max 5 bounded attempts)`);
}
if (MAX_VUS < PRE_ALLOC) {
  throw new Error(`MAX_VUS=${MAX_VUS} < PRE_ALLOCATED_VUS=${PRE_ALLOC} (k6 requires maxVUs >= preAllocatedVUs)`);
}
if (MAX_VUS > POOL) {
  throw new Error(`MAX_VUS=${MAX_VUS} > USER_POOL=${POOL} (would exceed the declared account/VU budget)`);
}
// COMPRESS < 1 is rejected rather than silently producing a longer run.
// With the old formula (rows = TEST_MIN, idx = floor(m*C), hold = 60/C) a
// compress of 0.5 kept TEST_MIN buckets but doubled each hold, so TEST_MIN=60
// produced a 120-minute run (verified with k6 inspect: holds 2m0s), and only
// half the trace rows were ever visited. Stretch semantics are ambiguous, so
// fail closed; the official benchmark uses COMPRESS=1.
if (COMPRESS < 1) {
  throw new Error(
    `TRACE_COMPRESS=${COMPRESS} is not supported (must be >= 1). ` +
    `Values < 1 used to stretch each bucket's hold to 60/COMPRESS seconds, ` +
    `silently doubling the run duration (TEST_MIN=60, COMPRESS=0.5 -> 120 min).`);
}

// ---------------------------------------------------------------------------
// NASA trace (strict parsing; malformed rows abort the run)
// ---------------------------------------------------------------------------
const MCR = new SharedArray('mcr', function () {
  const path = envRaw('MCR_CSV');
  if (!path) throw new Error('MCR_CSV is required (NASA MCR CSV path)');
  const txt = open(path);
  const lines = txt.split(/\r?\n/);
  const rows = [];
  let headerSeen = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === '') continue;
    const cols = line.split(',');
    if (!headerSeen) {
      headerSeen = true;
      if (cols.length < 3) {
        throw new Error(`MCR_CSV line ${i + 1}: header must have >= 3 columns, got ${cols.length}`);
      }
      const h = cols[2].trim().toLowerCase();
      if (h !== 'http_mcr') {
        throw new Error(`MCR_CSV line ${i + 1}: expected http_mcr in column 3, got '${cols[2].trim()}'`);
      }
      continue;
    }
    if (cols.length < 3) {
      throw new Error(`MCR_CSV line ${i + 1}: malformed row (${cols.length} columns, expected >= 3)`);
    }
    const raw = cols[2].trim();
    if (!/^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$/.test(raw)) {
      throw new Error(`MCR_CSV line ${i + 1}: non-numeric http_mcr '${raw}'`);
    }
    const v = Number(raw);
    if (!isFinite(v) || v < 0) {
      throw new Error(`MCR_CSV line ${i + 1}: invalid http_mcr '${raw}' (must be finite and >= 0)`);
    }
    rows.push(v);
  }
  if (rows.length === 0) throw new Error('MCR_CSV has no data rows');
  let peak = 0;
  for (let i = 0; i < rows.length; i++) if (rows[i] > peak) peak = rows[i];
  if (!(peak > 0)) throw new Error('MCR_CSV peak http_mcr is 0 (cannot normalize)');
  // Preserve the existing normalization: round(raw/peak * MAX_REQUESTS),
  // clamped at 0. MAX_REQUESTS is therefore the peak scheduled rate in
  // iterations per minute.
  const out = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    out[i] = Math.max(0, Math.round((rows[i] / peak) * MAXR));
  }
  return out;
});

// ---------------------------------------------------------------------------
// Seed accounts (strict: unique usernames/account ids, no silent ltpool)
// ---------------------------------------------------------------------------
const SEEDS = new SharedArray('seeds', function () {
  const path = envRaw('SEED_FILE');
  if (!path) throw new Error('SEED_FILE is required (JSON array of {username, accountId, contactsId})');
  const data = JSON.parse(open(path));
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error(`SEED_FILE '${path}' must be a non-empty JSON array`);
  }
  const seenUsers = Object.create(null);
  const seenAccounts = Object.create(null);
  const out = new Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const s = data[i];
    if (!s || typeof s.username !== 'string' || s.username === '') {
      throw new Error(`SEED_FILE entry ${i}: missing username`);
    }
    if (typeof s.accountId !== 'string' || s.accountId === '') {
      throw new Error(`SEED_FILE entry ${i} (${s.username}): missing accountId`);
    }
    if (seenUsers[s.username]) {
      throw new Error(`SEED_FILE entry ${i}: duplicate username '${s.username}'`);
    }
    if (seenAccounts[s.accountId]) {
      throw new Error(`SEED_FILE entry ${i}: duplicate accountId '${s.accountId}'`);
    }
    seenUsers[s.username] = true;
    seenAccounts[s.accountId] = true;
    out[i] = {
      username: s.username,
      accountId: s.accountId,
      contactsId: s.contactsId || null,
    };
  }
  return out;
});

const ACTIVE_VUS = OPEN_LOOP ? MAX_VUS : POOL;
if (PRE_ALLOC > POOL) {
  throw new Error(`PRE_ALLOCATED_VUS=${PRE_ALLOC} > USER_POOL=${POOL} (account budget)`);
}
if (ACTIVE_VUS > SEEDS.length) {
  if (!ALLOW_SEED_REUSE) {
    throw new Error(
      `seed pool too small: SEED_FILE has ${SEEDS.length} unique accounts but max active VUs = ${ACTIVE_VUS} ` +
      `(PRE_ALLOCATED_VUS=${PRE_ALLOC}, MAX_VUS=${MAX_VUS}, USER_POOL=${POOL}). ` +
      `Lower USER_POOL/MAX_VUS/PRE_ALLOCATED_VUS to <= ${SEEDS.length}, or set ALLOW_SEED_REUSE=1 ` +
      `to explicitly allow account sharing.`
    );
  }
}

// ---------------------------------------------------------------------------
// Stepwise arrival stages
//
// One source trace bucket -> two k6 stages:
//   { duration: '0', target }   instantaneous jump at the bucket boundary
//   { duration: hold, target }  hold the bucket rate
//
// Rate precision: timeUnit='1m' and an integer target of
// round(MCR[idx] * TRACE_COMPRESS) iterations/minute. Quantization is at most
// +-0.5 iteration/min (+-0.0083 it/s) per bucket, exact when TRACE_COMPRESS is
// an integer. The old timeUnit='1s' + Math.round(rps) quantized to +-0.5 it/s,
// 60x coarser.
//
// Compression semantics: every TRACE_COMPRESS-th trace row is replayed, rate
// x COMPRESS, hold / COMPRESS. COMPRESS >= 1 only (see the validation below);
// COMPRESS > 1 samples every COMPRESS-th row rather than aggregating the rows
// it skips, so it is a faster replay of a subset, not an exact replay of every
// original bucket.
// ---------------------------------------------------------------------------
function buildArrivalStages() {
  const rows = TEST_MIN > 0 ? TEST_MIN : Math.ceil(MCR.length / COMPRESS);
  const holdSec = 60 / COMPRESS;
  const stages = [];
  let expected = 0;
  let peakPerMin = 0;
  let firstIdx = 0;
  let lastIdx = 0;
  for (let m = 0; m < rows; m++) {
    const idx = (START_MIN + Math.floor(m * COMPRESS)) % MCR.length;
    if (m === 0) firstIdx = idx;
    lastIdx = idx;
    const perMin = Math.round(MCR[idx] * COMPRESS);
    stages.push({ duration: '0', target: perMin });
    stages.push({ duration: `${holdSec.toFixed(6)}s`, target: perMin });
    // perMin iterations/min held for (60/COMPRESS)s = (1/COMPRESS)min
    expected += perMin / COMPRESS;
    if (perMin > peakPerMin) peakPerMin = perMin;
  }
  return {
    stages: stages,
    rows: rows,
    holdSec: holdSec,
    expected: expected,
    peakPerMin: peakPerMin,
    firstIdx: firstIdx,
    lastIdx: lastIdx,
    firstRate: stages[0].target,
  };
}

const STAGE_INFO = buildArrivalStages();
const STAGES = STAGE_INFO.stages;
const NMIN = TEST_MIN > 0 ? TEST_MIN : Math.ceil(MCR.length / COMPRESS);

if (!OPEN_LOOP) {
  console.warn('CLOSED-LOOP mode selected: fixed VUs; achieved rate depends on iteration completion time and falls when the application slows. Not the official benchmark mode.');
}

// ---------------------------------------------------------------------------
// k6 options
// ---------------------------------------------------------------------------
const COMMON_OPTIONS = {
  systemTags: ['proto', 'status', 'method', 'name', 'group', 'check', 'error', 'error_code', 'tls_version'],
  thresholds: {
    http_req_failed: ['rate<0.35'],
    http_req_duration: ['p(95)<8000'],
  },
};

export const options = OPEN_LOOP ? {
  scenarios: {
    // Official workload: offered rate = MCR[idx] * TRACE_COMPRESS / 60
    // iterations/s, held bucket by bucket. FLAT_RPS (constant-arrival bypass)
    // stays removed: the official load axis is the NASA day-slice replay.
    load: {
      executor: 'ramping-arrival-rate',
      timeUnit: '1m',
      startRate: STAGE_INFO.firstRate,
      stages: STAGES,
      preAllocatedVUs: PRE_ALLOC,
      maxVUs: MAX_VUS,
      gracefulStop: '30s',
      tags: { exec_mode: 'open' },
    },
  },
  ...COMMON_OPTIONS,
} : {
  scenarios: {
    load: {
      executor: 'constant-vus',
      vus: POOL,
      duration: `${NMIN}m`,
      gracefulStop: '30s',
      tags: { exec_mode: 'closed' },
    },
  },
  ...COMMON_OPTIONS,
};

// ---------------------------------------------------------------------------
// Custom metrics (low cardinality: no usernames/accounts/order ids/request ids)
// ---------------------------------------------------------------------------
const iterationsStarted = new Counter('iterations_started_total');
// auth_login_attempts_total is the ALL-LOGINS superset (user + admin, initial
// + renewal). The unambiguous breakdowns are auth_initial_login_* and
// auth_renew_*. Kept for continuity with existing dashboards.
const loginAttempts = new Counter('auth_login_attempts_total');
// Initial authentication only: a renewal must never land in these.
const loginInitialAttempts = new Counter('auth_initial_login_attempts_total');
const loginInitialSuccess = new Counter('auth_initial_login_success_total');
const loginInitialFailures = new Counter('auth_initial_login_failures_total');
const loginRefreshSuccess = new Counter('auth_refresh_success_total');
const loginFailures = new Counter('auth_login_failures_total');
const loginRetries = new Counter('auth_login_retries_total');
const authSkippedIterations = new Counter('auth_skipped_iterations_total');
const renewAttempts = new Counter('auth_renew_attempts_total');
const renewSuccess = new Counter('auth_renew_success_total');
const renewFailure = new Counter('auth_renew_failures_total');
const renewRetries = new Counter('auth_renew_retries_total');
const renewScheduled = new Counter('auth_renew_scheduled_total');
const expiredTokens = new Counter('auth_expired_token_total');
const renewLatency = new Trend('auth_renew_latency_ms');
const authUnavailable = new Counter('auth_unavailable_actions_total');
// Iterations whose selected action needs no session at all. Separating these
// from auth_skipped_iterations_total is what makes the auth counters readable:
// before, every iteration with a failed login incremented the "skipped" counter
// even when the chosen action never needed a token.
const authUnauthIterations = new Counter('auth_unauth_iterations_total');
// Counted when an auth-preferred action (a_preserve/a_pay/a_preserve_other)
// degrades to its unauthenticated fallback.
const workloadFallback = new Counter('workload_fallback_total');
const adminLoginAttempts = new Counter('auth_admin_login_attempts_total');
const adminLoginFailures = new Counter('auth_admin_login_failures_total');
const adminSkipped = new Counter('auth_admin_skipped_total');
const scheduledGauge = new Gauge('scheduled_iterations_expected');
const peakGauge = new Gauge('scheduled_peak_per_min');
const traceRowsGauge = new Gauge('trace_rows_replayed');
const preflightGauge = new Gauge('preflight_enabled');
// Configured pool size, NOT live utilization. Live VU usage is k6's built-in
// `vus` gauge and `vus_max`; the previous name (vu_allocation) read as if it
// measured allocated/running VUs.
const vuConfigGauge = new Gauge('vu_pool_configured');
// P6 request correlation: whether the response carried the mesh-issued
// X-Request-ID. IDs are never used as metric tags (cardinality), only counted.
const correlationPresent = new Counter('correlation_mesh_id_present_total');
const correlationMissing = new Counter('correlation_mesh_id_missing_total');
const correlationLogged = new Counter('correlation_records_total');

// ---------------------------------------------------------------------------
// Workload data (unchanged)
// ---------------------------------------------------------------------------
const TRIPS = ['G1234', 'G1235', 'G1236'];
const ODS = [
  ['shanghai', 'taiyuan'],
  ['nanjing', 'shanghai'],
  ['suzhou', 'shanghai'],
];
const ROUTES = ['0b23bd3e-876a-4af3-b920-c50a90c90b04'];

function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

// Pre-flight data validation — read-only, opt-in (PREFLIGHT=1) only.
function verifyClusterData() {
  const checks = [
    { name: 'station-shanghai', url: `${HOST}/api/v1/stationservice/stations/id/shanghai`, expect: 200 },
    { name: 'route-shanghai-taiyuan', url: `${HOST}/api/v1/routeservice/routes/0b23bd3e-876a-4af3-b920-c50a90c90b04`, expect: 200 },
    { name: 'travel-query', url: `${HOST}/api/v1/travelservice/trips/left`, method: 'POST', body: JSON.stringify({ startPlace: 'shanghai', endPlace: 'taiyuan', departureTime: '2027-05-04' }), expect: 200 },
    { name: 'basic-welcome', url: `${HOST}/api/v1/basicservice/welcome`, expect: 200 },
  ];

  for (const c of checks) {
    const params = {
      method: c.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': generateRequestId('preflight'),
      },
      timeout: '15s',
      tags: { name: `preflight-${c.name}` },
    };
    if (c.body) params.body = c.body;

    const r = http.request(c.method || 'GET', c.url, c.body || null, params);
    if (r.status !== c.expect) {
      console.error(`PREFLIGHT-WARN: ${c.name} returned ${r.status} (expected ${c.expect}) - continuing`);
      continue;
    }
    console.log(`Pre-flight OK: ${c.name}`);
  }
  console.log('All pre-flight data checks passed');
}

// Unique request id per HTTP call. Module scope is per-VU in k6, and __ITER
// disambiguates iterations, so ids do not repeat across VUs or iterations.
let RID_SEQ = 0;
function generateRequestId(scope) {
  RID_SEQ += 1;
  return `k6-v${__VU}-i${__ITER}-${scope}-${RID_SEQ.toString(36)}-${Math.random().toString(36).substring(2, 10)}`;
}

// Envoy generates/replaces X-Request-ID at the ingress, so the k6 client id
// never reaches the application. The mesh id is what ties the k6 ledger to the
// gateway and destination-sidecar access logs.
function meshRequestId(r) {
  if (!r || !r.headers) return null;
  const h = r.headers;
  return h['X-Request-Id'] || h['X-Request-ID'] || h['x-request-id'] ||
         h['x-request-id '] || null;
}

// Per-request correlation record. Cheap by default (two counter increments);
// emits a console line only when CORRELATION_LOG=1.
function observeResponse(r, rid, tag, path, method) {
  const mesh = meshRequestId(r);
  if (mesh) correlationPresent.add(1); else correlationMissing.add(1);
  if (!CORRELATION_LOG) return;
  let dur = '-';
  try { dur = r.timings.duration.toFixed(1); } catch (e) { /* no timings */ }
  correlationLogged.add(1);
  console.log(`CORREL vu=${__VU} iter=${__ITER} name=${tag || path || '-'} ` +
    `method=${method || '-'} path=${path || '-'} status=${r.status} ` +
    `dur_ms=${dur} client_id=${rid || 'none'} mesh_id=${mesh || 'none'}`);
}

function logFailedRequest(r, tag, body) {
  if (r.status >= 500 || r.status === 0) {
    let dur = '-';
    try { dur = r.timings.duration.toFixed(1); } catch (e) { /* no timings */ }
    let m = '-';
    try { m = r.request.method; } catch (e) { /* older k6 */ }
    console.error(`FAILED ${tag}: status=${r.status} method=${m} url=${r.url} ` +
      `dur_ms=${dur} mesh_id=${meshRequestId(r) || 'none'} vu=${__VU} iter=${__ITER} ` +
      `body=${body !== undefined && body !== null ? JSON.stringify(body).slice(0, 200) : 'none'} ` +
      `error=${r.error || r.error_code || 'none'}`);
  }
}

// ---------------------------------------------------------------------------
// Authentication and JWT session lifecycle (per VU)
//
// TrainTicket issues self-contained HS256 JWTs with a 60-minute `exp` claim and
// NO refresh-token endpoint: `ts-auth-service` JWTProvider uses
// validityInMilliseconds = 3600000, and POST /api/v1/users/login returns
// {userId, username, token} with no `expires_in` and no refresh grant. Renewal
// is therefore re-authentication through the supported login endpoint (RFC 9700
// §4.14 refresh-token protection does not apply because no refresh token
// exists).
//
// Each VU owns an independent session. Renewal is derived from the token's own
// `exp` claim (decoded for scheduling only, never modified):
//
//     renewAt = exp - safety - U(0, jitter)
//
// with independent per-VU jitter, so VUs that logged in together do not renew
// together. There is no global 45-minute interval and no background timer: the
// check runs when an authenticated action starts.
// ---------------------------------------------------------------------------
const AUTH_NEW = 'new';
const AUTH_OK = 'ok';
const AUTH_FAILED = 'failed';
const AUTH_UNAVAILABLE = 'unavailable';

let RUN_T0 = 0; // scenario start (ms), set on the first iteration

const B64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

// k6 v2.3.0 has no atob(); the JWT payload is ASCII JSON, so a plain
// base64url decoder is sufficient. Used for scheduling only.
function b64urlDecode(s) {
  s = s.replace(/-/g, '+').replace(/_/g, '/');
  // Count explicit '=' plus any padding we add: each one means one trailing
  // byte in the output is padding garbage, not data. JWT payloads are usually
  // unpadded, so the added count is what matters.
  let pad = (s.match(/=*$/) || [''])[0].length;
  while (s.length % 4) { s += '='; pad += 1; }
  let out = '';
  for (let i = 0; i < s.length; i += 4) {
    const a = B64_ALPHABET.indexOf(s[i]);
    const b = B64_ALPHABET.indexOf(s[i + 1]);
    const c = s[i + 2] === '=' ? 0 : B64_ALPHABET.indexOf(s[i + 2]);
    const d = s[i + 3] === '=' ? 0 : B64_ALPHABET.indexOf(s[i + 3]);
    const n = (a << 18) | (b << 12) | (c << 6) | d;
    out += String.fromCharCode((n >> 16) & 255, (n >> 8) & 255, n & 255);
  }
  // Bytes produced from '=' padding are not real data; leaving them in adds a
  // trailing NUL that makes JSON.parse fail ("Unexpected token at the end").
  if (pad > 0) out = out.slice(0, out.length - pad);
  return out;
}

function jwtExpSeconds(token) {
  if (typeof token !== 'string') return null;
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  let claims;
  try {
    claims = JSON.parse(b64urlDecode(parts[1]));
  } catch (e) {
    return null;
  }
  const exp = claims && claims.exp;
  if (typeof exp !== 'number' || !isFinite(exp) || exp <= 0) return null;
  return exp;
}

function newSession() {
  return {
    token: null, exp: 0, iat: 0, renewAt: 0,
    failures: 0, retryAt: 0, state: AUTH_NEW,
  };
}

// isRenew distinguishes a renewal from the initial issuance so that
// auth_renew_scheduled_total counts renewal scheduling only.
function sessionSchedule(sess, isRenew) {
  const now = Date.now() / 1000;
  const ttl = sess.exp - now;
  if (!(ttl > 0)) return false;
  // Safety margin: at least 60 s where the lifetime permits, at most 5 % of it.
  const safety = Math.min(60, Math.max(5, ttl * 0.05));
  // Jitter: up to 10 min or 20 % of the lifetime, whichever is smaller.
  const jitter = Math.min(600, ttl * 0.2);
  sess.renewAt = sess.exp - safety - Math.random() * jitter;
  if (isRenew) {
    const bucket = RUN_T0 ? Math.floor((sess.renewAt * 1000 - RUN_T0) / 600000) : 0;
    renewScheduled.add(1, { bucket: String(bucket) });
  }
  return true;
}

// Validate a freshly issued token and schedule its renewal. The JWT is never
// modified; only its exp claim is read, for scheduling.
function sessionAccept(sess, token, kind, isRenew) {
  const exp = jwtExpSeconds(token);
  const now = Date.now() / 1000;
  if (!exp) {
    console.error(`TOKEN-ERR vu=${__VU} kind=${kind} reason=no-exp-claim`);
    return false;
  }
  const ttl = exp - now;
  if (ttl <= 0) {
    console.error(`TOKEN-ERR vu=${__VU} kind=${kind} reason=already-expired ttl=${ttl.toFixed(0)}`);
    return false;
  }
  if (ttl > 6 * 3600) {
    console.error(`TOKEN-ERR vu=${__VU} kind=${kind} reason=implausible-ttl ttl=${ttl.toFixed(0)}`);
    return false;
  }
  sess.token = token;
  sess.exp = exp;
  sess.iat = now;
  sess.state = AUTH_OK;
  sess.failures = 0;
  sess.retryAt = 0;
  return sessionSchedule(sess, isRenew);
}

function sessionUsable(sess) {
  return !!(sess.token && sess.state === AUTH_OK && Date.now() / 1000 < sess.exp);
}

function H(st) {
  if (!st) return {};
  const tok = st.sess ? st.sess.token : st.token;
  return tok ? { Authorization: `Bearer ${tok}` } : {};
}

function backoffSeconds(failures) {
  return Math.min(300, 30 * Math.pow(2, Math.min(4, Math.max(0, failures - 1))));
}

function sessionFail(sess, now) {
  sess.failures += 1;
  sess.retryAt = now + backoffSeconds(sess.failures);
}

// Renewal state machine. Returns true when a usable session is available.
// Bounded attempts happen inside loginRequest(); the per-VU retry deadline here
// prevents every iteration from starting a new retry sequence.
function sessionEnsure(st, sess, kind, loginFn) {
  const now = Date.now() / 1000;
  // Recovery from AUTH_UNAVAILABLE is driven by the per-iteration preamble, so
  // an outage ends as soon as the backoff allows and the service is back.
  if (sess.state === AUTH_UNAVAILABLE) return false;
  if (!sess.token || now >= sess.exp) {
    if (sess.retryAt && now < sess.retryAt) return false;
    if (loginFn(st, sess, kind, false)) return true;
    sessionFail(sess, now);
    authUnavailable.add(1);
    return false;
  }
  // A usable token exists.
  if (now < sess.renewAt) return true;
  if (sess.retryAt && now < sess.retryAt) return true;
  if (loginFn(st, sess, kind, true)) return true;
  sessionFail(sess, now);
  // The old token remains usable until its real exp, so authenticated work can
  // continue while renewal is retried.
  return true;
}

// Called at the start of every iteration, for every VU and every session it
// already holds. It renews a session only when a token exists: an
// unauthenticated iteration must not create a session (that was the herd bug),
// but an existing session must not be allowed to expire just because the next
// few actions happen to be public (or admin actions are rare).
function sessionPreambleOne(st, sess, kind) {
  if (!sess) return;
  const now = Date.now() / 1000;
  if (sess.state === AUTH_UNAVAILABLE) {
    // Try to recover, but only on the bounded backoff schedule.
    if (sess.retryAt && now < sess.retryAt) return;
    if (loginRequest(st, sess, kind, false)) return;
    sessionFail(sess, now);
    return;
  }
  if (!sess.token) return; // no session yet: sessions are created lazily
  if (now >= sess.exp) {
    expiredTokens.add(1);
    console.error(`TOKEN-EXPIRED vu=${__VU} kind=${kind} latexp=${sess.exp.toFixed(0)} now=${now.toFixed(0)}`);
    sess.token = null;
    sess.state = AUTH_UNAVAILABLE;
    authUnavailable.add(1);
    return;
  }
  if (now < sess.renewAt) return;
  if (sess.retryAt && now < sess.retryAt) return;
  if (loginRequest(st, sess, kind, true)) return;
  sessionFail(sess, now);
}

function sessionPreamble(st) {
  sessionPreambleOne(st, st.sess, 'user');
  sessionPreambleOne(st, st.admin, 'admin');
}

function loginRequest(st, sess, kind, isRefresh) {
  let lastStatus = 0;
  let lastError = '';
  const maxAttempts = 1 + LOGIN_RETRIES;
  // Per-event counters: the st.* counters are cumulative for the VU's lifetime,
  // so logging them directly made a 3-attempt policy look like a 6-attempt one.
  let eventAttempts = 0;
  let eventRetries = 0;
  const started = Date.now();
  // Initial vs renewal must be counted apart. The old code called
  // loginInitialSuccess.add(1) on EVERY successful user login, so renewals
  // inflated it (150-min run: 15 = 5 initial + 10 renewals).
  if (isRefresh) renewAttempts.add(1);
  else loginInitialAttempts.add(1);
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (attempt > 0) {
      eventRetries += 1;
      st.loginRetries += 1;
      loginRetries.add(1);
      if (isRefresh) renewRetries.add(1);
      // Bounded exponential-ish backoff, capped at 1 s. Runs only on failure.
      sleep(Math.min(1.0, 0.4 + Math.random() * 0.6));
    }
    const rid = generateRequestId(kind === 'admin' ? 'admin-login' : 'login');
    const body = kind === 'admin'
      ? JSON.stringify({ username: ADMIN_USER, password: ADMIN_PW })
      : JSON.stringify({ username: st.username, password: PW });
    loginAttempts.add(1);
    if (kind === 'admin') adminLoginAttempts.add(1);
    if (isRefresh) renewAttempts.add(0); // no-op keeps the counter declaration honest
    st.loginAttempts += 1;
    eventAttempts += 1;
    const r = http.post(`${HOST}/api/v1/users/login`, body, {
      headers: { 'Content-Type': 'application/json', 'X-Request-ID': rid },
      tags: { name: kind === 'admin' ? 'admin-login' : 'login' },
      timeout: '10s',
    });
    lastStatus = r.status;
    lastError = r.error || r.error_code || '';
    // Instrumentation fix: login failures used to be counted in
    // http_req_failed but never logged, so ~15k failed requests had no
    // request-level record. Log every non-200 attempt like any other action.
    if (r.status !== 200) logFailedRequest(r, kind === 'admin' ? 'admin-login' : 'login', body);
    if (r.status === 200) {
      let token = null;
      try {
        const d = r.json().data;
        if (!d || !d.token) throw new Error('login response missing data.token');
        token = d.token;
        if (kind === 'user') {
          st.account = d.userId || st.account;
          st.loginSuccesses += 1;
        }
        if (isRefresh) {
          // counted below, once the token is accepted
        } else {
          loginInitialSuccess.add(1);
        }
      } catch (e) {
        lastError = e.message || 'unparseable login response';
      }
      if (token && sessionAccept(sess, token, kind, isRefresh)) {
        if (isRefresh) {
          renewSuccess.add(1);
          renewLatency.add(Date.now() - started);
          console.log(`RENEW-OK vu=${__VU} kind=${kind} exp_in=${(sess.exp - Date.now() / 1000).toFixed(0)} renew_at=${sess.renewAt.toFixed(0)}`);
        }
        return true;
      }
      lastError = 'token rejected by client validation';
    }
  }

  sess.state = AUTH_FAILED;
  st.lastLoginFailureAt = Date.now();
  st.loginFailures += 1;
  loginFailures.add(1);
  if (isRefresh) renewFailure.add(1);
  else loginInitialFailures.add(1);
  if (kind === 'admin') adminLoginFailures.add(1);
  console.error(`AUTH-FAIL vu=${__VU} kind=${kind} user=${kind === 'admin' ? ADMIN_USER : st.username} attempts_this_event=${eventAttempts} retries_this_event=${eventRetries} attempts_total=${st.loginAttempts} retries_total=${st.loginRetries} last_status=${lastStatus} error=${lastError}`);
  return false;
}

// Backward-compatible name: the user session is st.sess.
function ensureLogin(st) {
  return sessionEnsure(st, st.sess, 'user', loginRequest);
}

function get(path, st, tag) {
  const headers = Object.assign(
    { 'Content-Type': 'application/json', 'X-Request-ID': generateRequestId('get') },
    st ? H(st) : {}
  );
  // FIX(request-id): the previous version built `headers` (with X-Request-ID)
  // and then passed a different object to http.get, dropping the id.
  const rid = headers['X-Request-ID'];
  let r = http.get(`${HOST}${path}`, {
    headers: headers,
    tags: { name: tag || path },
    timeout: '30s',
  });
  observeResponse(r, rid, tag, path, 'GET');
  // 401 before the scheduled renewal: the token was rejected early (rotation,
  // revocation or a shorter-than-expected lifetime). Invalidate it, renew once,
  // and replay the GET exactly once — GET is safe to repeat.
  if (r.status === 401 && st && st.sess && st.sess.token) {
    expiredTokens.add(1);
    console.error(`TOKEN-401 vu=${__VU} path=${path} action=relogin-replay-get`);
    st.sess.token = null;
    st.sess.state = AUTH_NEW;
    st.sess.exp = 0;
    st.sess.renewAt = 0;
    st.sess.retryAt = 0;
    if (sessionEnsure(st, st.sess, 'user', loginRequest)) {
      const h2 = Object.assign(
        { 'Content-Type': 'application/json', 'X-Request-ID': generateRequestId('get') },
        H(st)
      );
      const rid2 = h2['X-Request-ID'];
      const r2 = http.get(`${HOST}${path}`, {
        headers: h2,
        tags: { name: tag || path },
        timeout: '30s',
      });
      observeResponse(r2, rid2, tag, path, 'GET');
      logFailedRequest(r2, tag || path);
      return r2;
    }
  }
  logFailedRequest(r, tag || path);
  return r;
}

function post(path, body, st, tag) {
  const headers = Object.assign(
    { 'Content-Type': 'application/json', 'X-Request-ID': generateRequestId('post') },
    st ? H(st) : {}
  );
  const rid = headers['X-Request-ID'];
  const r = http.post(`${HOST}${path}`, JSON.stringify(body), {
    headers: headers,
    tags: { name: tag || path },
    timeout: '30s',
  });
  observeResponse(r, rid, tag, path, 'POST');
  // A 401 invalidates the session for subsequent requests, but a POST is never
  // replayed automatically: it may mutate business state and TrainTicket offers
  // no idempotency guarantee.
  if (r.status === 401 && st && st.sess && st.sess.token) {
    expiredTokens.add(1);
    console.error(`TOKEN-401 vu=${__VU} path=${path} action=no-replay-post`);
    st.sess.token = null;
    st.sess.state = AUTH_NEW;
    st.sess.exp = 0;
    st.sess.renewAt = 0;
    st.sess.retryAt = 0;
  }
  logFailedRequest(r, tag || path, body);
  return r;
}

// ---------------------------------------------------------------------------
// Actions (endpoint paths, payloads, checks and ACTS weights unchanged;
// dynamic-path requests now carry stable, low-cardinality `name` tags)
// ---------------------------------------------------------------------------
function a_travel(st) {
  const [o, d] = pick(ODS);
  const day = 27 + Math.floor(Math.random() * 3);
  const b = { startPlace: o, endPlace: d, departureTime: `2027-05-${String(day).padStart(2, '0')}` };
  const r1 = post('/api/v1/travelservice/trips/left', b, st, 'travel-query');
  // FIX(check-bug): trips/left returns a bare ArrayList<TripResponse> (even its
  // error path returns an array — TravelController.queryInfo). There is no
  // {status:1} envelope, so the old check (r.json().status === 1) evaluated
  // undefined===1 and failed 100% of iterations (smoke: ✗223) with zero HTTP
  // failures. Validate the response shape only; data presence is not asserted.
  check(r1, { 'travel-query ok': r => r.status === 200 && Array.isArray(r.json()) });
  const r2 = post('/api/v1/travel2service/trips/left', b, st, 'travel2-query');
  check(r2, { 'travel2-query ok': r => r.status === 200 });
  const r3 = get(`/api/v1/travelservice/trips/${pick(TRIPS)}`, st, 'trip-detail');
  check(r3, { 'trip-detail ok': r => r.status === 200 });
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
  if (!ensureLogin(st)) return;
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
  get('/api/v1/foodservice/foods/2027-05-27/shanghai/taiyuan/G1234', st, 'food-by-trip');
  // FIX(4xx-drift): GET /welcome hits the ui nginx SPA root, which has no such
  // static file -> 404 on every a_menus iteration (~6% of all edge requests).
  // /index.html is the same SPA entry and returns 200.
  get('/index.html', st);
  get('/api/v1/consignservice/welcome', st);
  get('/api/v1/consignpriceservice/welcome', st);
  get('/api/v1/securityservice/securityConfigs', st);
  if (st.account) get(`/api/v1/contactservice/contacts/account/${st.account}`, st, 'user-contacts');
}
function a_leaf(st) {
  if (!ensureLogin(st)) return;
  get('/api/v1/seatservice/welcome', st);
  get('/api/v1/paymentservice/welcome', st);
  get('/api/v1/paymentservice/payment', st);
  post('/api/v1/paymentservice/payment', { userId: st.account || 'x', orderId: st.orders.length ? pick(st.orders) : 'x', tripId: 'G1234', price: '100.0' }, st, 'payment');
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
  if (!sessionEnsure(st, st.admin, 'admin', loginRequest)) {
    adminSkipped.add(1);
    return;
  }
  const s = { token: st.admin.token };
  const r1 = get('/api/v1/admintravelservice/admintravel', s, 'admintravel');
  if (r1.status === 403) { st.admin.token = null; st.admin.state = AUTH_NEW; return; }
  get('/api/v1/adminrouteservice/adminroute', s, 'adminroute');
  get('/api/v1/adminuserservice/users', s, 'adminusers');
  get('/api/v1/adminorderservice/adminorder', s, 'adminorder');
  get('/api/v1/adminbasicservice/adminbasic/contacts', s, 'adminbasic-contacts');
  get('/api/v1/adminbasicservice/adminbasic/stations', s, 'adminbasic-stations');
}
function a_orders(st) {
  if (!ensureLogin(st)) return;
  post('/api/v1/orderservice/order/query', { loginId: st.account }, st, 'orders-query');
  const r = post('/api/v1/orderOtherService/orderOther/query', { loginId: st.account }, st, 'orderother-query');
  if (r.status === 200) {
    try {
      const data = r.json().data || [];
      data.slice(0, 3).forEach((o) => { if (o && o.id) st.orders.push(o.id); });
    } catch (e) { }
  }
}
function a_preserve(st) {
  if (!ensureLogin(st) || !st.contacts) {
    workloadFallback.add(1, { action: 'a_preserve', reason: st.contacts ? 'no_session' : 'no_contacts' });
    a_travel(st); return;
  }
  const [o, d] = pick(ODS.slice(0, 3));
  const r = post('/api/v1/preserveservice/preserve', {
    accountId: st.account, contactsId: st.contacts, tripId: pick(TRIPS.slice(0, 3)),
    seatType: pick([2, 3]), date: `2027-05-${27 + Math.floor(Math.random() * 3)}`,
    from: o, to: d, assurance: 0,
  }, st, 'preserve');
  if (r.status === 200) a_orders(st);
}
function a_pay(st) {
  if (!ensureLogin(st) || !st.orders.length) {
    workloadFallback.add(1, { action: 'a_pay', reason: st.orders.length ? 'no_session' : 'no_orders' });
    a_orders(st); return;
  }
  const oid = pick(st.orders);
  let price = '100.0';
  const pr = get(`/api/v1/orderservice/order/price/${oid}`, st, 'order-price');
  if (pr.status === 200) { try { price = String(pr.json().data || price); } catch (e) { } }
  post('/api/v1/inside_pay_service/inside_payment', { userId: st.account, orderId: oid, tripId: pick(TRIPS.slice(0, 3)), price }, st, 'inside-payment');
  get(`/api/v1/executeservice/execute/collected/${oid}`, st, 'execute-collected');
  get(`/api/v1/executeservice/execute/execute/${oid}`, st, 'execute-execute');
  if (Math.random() < 0.5) get(`/api/v1/cancelservice/cancel/${oid}/${st.account}`, st, 'cancel');
  if (Math.random() < 0.3 && st.orders.length) {
    post('/api/v1/rebookservice/rebook', { orderId: pick(st.orders), oldTripId: 'G1234', tripId: 'G1235', seatType: 2, date: `2027-05-${27 + Math.floor(Math.random() * 3)}` }, st, 'rebook');
  }
}


function a_notifications(st) {
  // FIX(4xx-drift): REMOVED from ACTS. The deployed ts-notifyservice only
  // exposes side-effectful test endpoints (test_send_mq/test_send_mail —
  // NotificationController.java) — there is no readable notifications API in
  // this deployment, so the three GETs 404'd on every call. Calling the test
  // endpoints at load rate would send real MQ/email, so the flow is dropped
  // rather than re-pointed.
  if (!ensureLogin(st)) return;
  get('/api/v1/notifyservice/test_send_mq', st);
  get('/api/v1/notifyservice/test_send_mail', st);
}

function a_voucher(st) {
  // FIX(stage-12k): the /getVoucher API contract is {orderId, voucherId} — the
  // original call sent {userId, voucherId} which always raised KeyError('orderId')
  // -> 500 at ~4.5/s for the whole previous run. Gate like a_pay/a_menus and send
  // a real order id once the VU has logged in and listed orders.
  if (!ensureLogin(st) || !st.orders.length) return;
  post('/getVoucher', { orderId: pick(st.orders), voucherId: 'VOUCHER1' }, st, 'getVoucher');
}

function a_preserve_other(st) {
  if (!ensureLogin(st) || !st.contacts) {
    workloadFallback.add(1, { action: 'a_preserve_other', reason: st.contacts ? 'no_session' : 'no_contacts' });
    a_travel(st); return;
  }
  const [o, d] = pick(ODS);
  const r = post('/api/v1/preserveotherservice/preserveOther', {
    accountId: st.account, contactsId: st.contacts, tripId: pick(TRIPS),
    ticketType: 2, date: `2027-05-${27 + Math.floor(Math.random() * 3)}`,
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
  if (!ensureLogin(st)) return;
  if (st.username) get(`/api/v1/userservice/users/${st.username}`, st, 'user-info');
  if (st.account) {
    get(`/api/v1/userservice/users/id/${st.account}`, st, 'user-profile');
    get(`/api/v1/contactservice/contacts/account/${st.account}`, st, 'user-contacts');
  }
}

function a_cancel(st) {
  if (!ensureLogin(st) || !st.orders.length) return;
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
  if (!ensureLogin(st) || !st.orders.length) return;
  const oid = pick(st.orders);
  let price = '100.0';
  const pr = get(`/api/v1/orderservice/order/price/${oid}`, st, 'order-price');
  if (pr.status === 200) { try { price = String(pr.json().data || price); } catch (e) { } }
  post('/api/v1/inside_pay_service/inside_payment', { userId: st.account, orderId: oid, tripId: pick(TRIPS.slice(0, 3)), price }, st, 'inside-payment');
}


const ACTS = [
  [a_travel, 10], [a_static, 6], [a_menus, 5], [a_leaf, 4], [a_orders, 5],
  [a_preserve, 3], [a_pay, 3], [a_admin, 2],
  [a_voucher, 2], [a_preserve_other, 2],
  [a_user_profile, 2], [a_cancel, 2], [a_verifycode, 2], [a_inside_payment, 2],
];
const BAG = [];
ACTS.forEach(([f, w]) => { for (let i = 0; i < w; i++) BAG.push(f); });

// ---------------------------------------------------------------------------
// Setup (no traffic unless PREFLIGHT=1)
// ---------------------------------------------------------------------------
export function setup() {
  scheduledGauge.add(STAGE_INFO.expected);
  peakGauge.add(STAGE_INFO.peakPerMin);
  traceRowsGauge.add(STAGE_INFO.rows);
  preflightGauge.add(PREFLIGHT ? 1 : 0);
  vuConfigGauge.add(MAX_VUS);

  const wraps = START_MIN + Math.floor((STAGE_INFO.rows - 1) * COMPRESS) >= MCR.length;
  console.log(
    `CONFIG mode=${OPEN_LOOP ? 'open' : 'closed'} host=${HOST} trace_rows=${MCR.length} ` +
    `replay_rows=${STAGE_INFO.rows} trace_idx=${STAGE_INFO.firstIdx}..${STAGE_INFO.lastIdx} wrap=${wraps} ` +
    `start_min=${START_MIN} test_min=${TEST_MIN} compress=${COMPRESS} max_requests=${MAXR} ` +
    `hold_s=${STAGE_INFO.holdSec.toFixed(3)} expected_iterations=${STAGE_INFO.expected.toFixed(1)} ` +
    `peak_iters_per_min=${STAGE_INFO.peakPerMin} peak_iters_per_s=${(STAGE_INFO.peakPerMin / 60).toFixed(3)} ` +
    `pre_allocated_vus=${PRE_ALLOC} max_vus=${MAX_VUS} user_pool=${POOL} seeds=${SEEDS.length} ` +
    `seed_reuse=${ALLOW_SEED_REUSE ? 'allowed' : 'forbidden'} preflight=${PREFLIGHT ? 'ON' : 'OFF'}`
  );

  if (PREFLIGHT) {
    console.log(`Preflight performed at ${new Date().toISOString()} (opt-in; these requests are outside the measured arrival schedule)`);
    verifyClusterData();
  } else {
    console.log('Preflight disabled (official benchmark default): no pre-scenario traffic');
  }
  return { t0: Date.now() };
}

// ---------------------------------------------------------------------------
// Per-VU state (created lazily on first scheduled iteration)
// ---------------------------------------------------------------------------
function state() {
  const vu = __VU > 0 ? __VU : 1;
  const idx = (vu - 1) % SEEDS.length; // one-based VU -> zero-based seed
  const s = SEEDS[idx];
  return {
    vu: vu,
    seedIndex: idx,
    username: s.username,
    account: s.accountId,
    contacts: s.contactsId,
    // Independent per-VU sessions. The admin session is renewed on its own
    // schedule with its own jitter, even though all VUs share the admin account.
    sess: newSession(),
    admin: newSession(),
    authState: AUTH_NEW,
    loginAttempts: 0,
    loginSuccesses: 0,
    loginFailures: 0,
    loginRetries: 0,
    lastLoginFailureAt: 0,
    orders: [],
  };
}

let ST = null;

// Authentication requirement per action.
//   AUTH_ONLY   - must never run without a session
//   AUTH_PREFER - needs a session, but degrades to an unauthenticated fallback
//   (neither)   - needs no session at all
// Selecting the action first is what stops unauthenticated iterations from
// generating login traffic they do not need.
const AUTH_ONLY = new Set([a_menus, a_leaf, a_orders, a_user_profile, a_cancel,
                           a_voucher, a_inside_payment]);
const AUTH_PREFER = new Set([a_preserve, a_pay, a_preserve_other]);

function runAction(action, st) {
  if (!AUTH_ONLY.has(action) && !AUTH_PREFER.has(action)) {
    authUnauthIterations.add(1);
    action(st);
    return;
  }
  if (ensureLogin(st)) {
    action(st);
    return;
  }
  authSkippedIterations.add(1);
  if (AUTH_PREFER.has(action)) {
    // Preserve the intended action mix: the action's own guard now takes its
    // unauthenticated fallback (a_travel / a_orders).
    action(st);
  }
  // AUTH_ONLY actions are deliberately not executed: sending them without a
  // token is the defect this replaces.
}

export default function (data) {
  if (!RUN_T0) RUN_T0 = data.t0;
  if (!ST) ST = state(); // lazy per-VU init; no sleep, no login here
  iterationsStarted.add(1);

  get('/', ST);

  // Renewal is proactive and per-VU: only VUs that already hold a token renew
  // here, so unauthenticated iterations still generate no login traffic.
  sessionPreamble(ST);

  // Choose the action BEFORE touching auth. The previous version called
  // ensureLogin(ST) unconditionally, so the ~37% of iterations that picked an
  // unauthenticated action (a_travel/a_static/a_verifycode) still attempted a
  // login — pure amplification during the start-of-run herd.
  const action = pick(BAG);

  if (OPEN_LOOP) {
    // No end-of-iteration sleep: the arrival-rate executor paces iterations.
    runAction(action, ST);
    return;
  }

  // Closed-loop, fixed-VU mode (explicit opt-in only). Achieved rate depends
  // on iteration completion time and falls when the application slows.
  const elapsedMin = Math.floor((Date.now() - data.t0) / 60000);
  const traceIdx = Math.floor((START_MIN + elapsedMin * COMPRESS) % MCR.length);
  const count = Math.round(MCR[traceIdx] * COMPRESS) || 0;
  if (count <= 0) { sleep(1); return; }
  const perVU = count / POOL;
  runAction(action, ST);
  sleep(Math.max(0.05, 60 / Math.max(1, perVU) - 0.5));
}
