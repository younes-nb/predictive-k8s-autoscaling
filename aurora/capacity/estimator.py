"""Service state and capacity estimation.

Replaces the previous single-scalar ``capacity = goodput / replicas``.

The service's sustainable capacity is now estimated **per binding signal** and
the binding capacity is the minimum across signals:

    cap_cpu      CPU-limited arrivals per replica
    cap_memory   memory-limited arrivals per replica
    cap_latency  arrivals per replica before latency degrades vs the service's
                 own baseline (never the SLO)
    cap_throttle arrivals per replica before CFS throttling dominates
    cap_queue    arrivals per replica before in-flight work exceeds the
                 service's own baseline
    cap_goodput  observed completions per replica -- reported as a feature and
                 used as a gate, deliberately NOT part of the minimum

The argmin is the bottleneck, so bottleneck identification falls out of the
capacity estimate instead of being a separate guess.

Goodput is an input, not the objective. No SLO value enters this module.

Partial pooling: each signal's log-capacity is shrunk toward a per-signal global
mean, so a service with few observations (or a brand new deployment) still gets
a usable estimate.
"""

import math

SIGNALS = ["cpu", "memory", "latency", "throttle", "queue"]
ALL_SIGNALS = SIGNALS + ["goodput"]

EPS = 1e-6

# Defaults used only until the service has enough of its own evidence.
DEFAULT_U_CPU = 0.85
DEFAULT_U_MEM = 0.90
THROTTLE_TOLERANCE = 0.20      # 20% of CFS periods throttled is the ceiling


def _log(x):
    return math.log(max(x, EPS))


class ServiceStateEstimator:
    def __init__(self, cfg, prior_strength=10.0):
        self.cfg = cfg
        self.prior_strength = prior_strength
        # signal -> {"mean": float, "n": float} global prior
        self.global_stats = {}
        # service -> signal -> {"mean", "m2", "n"}
        self.local = {}
        # service -> adaptive constants
        self.adapt = {}
        self.last_state = {}
        self.warming = {}

    # ------------------------------------------------------------------
    def _prior(self, signal):
        g = self.global_stats.setdefault(signal, {"mean": 0.0, "n": 0.0})
        return g["mean"]

    def _update_global(self, signal, value):
        g = self.global_stats.setdefault(signal, {"mean": 0.0, "n": 0.0})
        g["n"] += 1.0
        g["mean"] += (value - g["mean"]) / g["n"]

    # ------------------------------------------------------------------
    def _adapt(self, service):
        a = self.adapt.setdefault(service, {
            "u_cpu": DEFAULT_U_CPU, "u_mem": DEFAULT_U_MEM,
            "lat_base": None, "inflight_base": None,
        })
        return a

    def _observe_adaptive(self, service, st):
        """Update the service's own sustainable-utilisation and baseline estimates."""
        a = self._adapt(service)
        clp = st.get("cpu_limit_pct")
        thr = st.get("throttled_pct")
        if clp is not None and thr is not None and thr < 0.05 and clp > 0:
            # highest utilisation observed while NOT throttling
            a["u_cpu"] = max(0.5, min(0.95, 0.9 * a["u_cpu"] + 0.1 * clp))
        mlp = st.get("mem_limit_pct")
        pf = st.get("page_faults")
        if mlp is not None and mlp > 0 and (pf is None or pf <= 0):
            a["u_mem"] = max(0.6, min(0.98, 0.9 * a["u_mem"] + 0.1 * mlp))
        p50 = st.get("lat_p50")
        if p50 is not None and p50 > 0:
            # running low envelope = the service's own idea of normal
            a["lat_base"] = p50 if a["lat_base"] is None else min(a["lat_base"], p50)
            a["lat_base"] = max(a["lat_base"], 1.0)
        ifl = st.get("inflight_per_replica")
        if ifl is not None and ifl > 0:
            a["inflight_base"] = (ifl if a["inflight_base"] is None
                                  else 0.95 * a["inflight_base"] + 0.05 * ifl)
        return a

    # ------------------------------------------------------------------
    def is_warming(self, st):
        """True when the service's *ready* pods are still mostly booting.

        Per-Pod, not blanket: a service with four mature Ready pods and one new
        one is NOT warming, and its capacity must still be learned.
        """
        mf = st.get("mature_fraction")
        if mf is not None:
            return mf < 0.5
        age = st.get("pod_age_s")
        startup = st.get("startup_p95_s")
        if age is None or not startup:
            return False
        return age < startup

    def observe(self, service, st, update_posterior=True):
        """Fold one tick into the estimate.

        ``st`` keys: replicas, req_rate, completed, cpu_cores, cpu_limit,
        mem_working_set, mem_limit, lat_p50, lat_p90, throttled_pct,
        inflight_per_replica, latency_available, blindness_ratio,
        cpu_limit_pct, mem_limit_pct, page_faults.
        """
        reps = st.get("replicas") or 0
        if reps <= 0:
            return
        self.last_state[service] = st
        if self.is_warming(st) or not update_posterior:
            # Startup-contaminated samples must not enter the long-term
            # posterior: a pod that is booting shows high latency, throttling
            # and queueing that say nothing about sustainable capacity.
            self.warming[service] = self.is_warming(st)
            return
        # A booting pod has high latency and queues that mean nothing about
        # capacity. While the service is still inside its own startup envelope,
        # latency and queue must not be capacity signals -- otherwise every cold
        # service looks saturated at t=0 and the best-effort floor pins the
        # whole namespace at max (measured: 35/42 deployments at 10 replicas).
        age = st.get("pod_age_s")
        startup = st.get("startup_p95_s")
        if age is not None and startup:
            self.warming[service] = age < startup
        a = self._observe_adaptive(service, st)
        caps = self._capacities(service, st, a)
        loc = self.local.setdefault(service, {})
        for sig, cap in caps.items():
            if cap is None or not math.isfinite(cap) or cap <= 0:
                continue
            v = _log(cap)
            self._update_global(sig, v)
            d = loc.setdefault(sig, {"mean": v, "m2": 0.0, "n": 0.0})
            d["n"] += 1.0
            delta = v - d["mean"]
            d["mean"] += delta / d["n"]
            d["m2"] += delta * (v - d["mean"])

    # ------------------------------------------------------------------
    def _capacities(self, service, st, a):
        """Return {signal: arrivals per replica that signal can sustain}."""
        reps = float(st.get("replicas") or 0)
        if reps <= 0:
            return {}
        arr = st.get("req_rate")
        if arr is None or arr <= 0:
            return {}
        arr_rep = arr / reps

        out = {}

        # --- CPU -------------------------------------------------------
        cpu_lim = st.get("cpu_limit")
        cpu_use = st.get("cpu_cores")
        if cpu_lim and cpu_use is not None and cpu_use > 0:
            cpu_lim_rep = cpu_lim / reps
            cpu_use_rep = cpu_use / reps
            out["cpu"] = (cpu_lim_rep * a["u_cpu"]) / (cpu_use_rep / arr_rep)

        # --- memory ----------------------------------------------------
        # Memory is a stock, not a rate: express the capacity as the arrival
        # rate the service can take before working set reaches u_mem of its
        # limit. Dividing bytes by arrivals (the earlier form) produced values
        # ~1e8 and a meaningless spread.
        mem_pct = st.get("mem_limit_pct")
        if mem_pct is None:
            mem_use, mem_lim = st.get("mem_working_set"), st.get("mem_limit")
            if mem_use and mem_lim:
                mem_pct = mem_use / mem_lim
        if mem_pct and mem_pct > 0:
            out["memory"] = arr_rep * (a["u_mem"] / mem_pct)

        # --- latency ---------------------------------------------------
        # Excluded entirely when the telemetry is blind: missing latency must
        # not be read as "fast".
        lat_avail = st.get("latency_available", True)
        blind = st.get("blindness_ratio", 0.0) or 0.0
        p90 = st.get("lat_p90")
        if lat_avail and blind < 0.5 and p90 and p90 > 0 and a["lat_base"]:
            out["latency"] = arr_rep * (a["lat_base"] / p90)

        # --- throttling -------------------------------------------------
        thr = st.get("throttled_pct")
        if thr is not None:
            # canonical representation is a fraction in [0,1]
            frac = max(0.0, min(1.0, float(thr)))
            out["throttle"] = arr_rep * (THROTTLE_TOLERANCE /
                                         max(frac, THROTTLE_TOLERANCE))

        # --- queue / in-flight ------------------------------------------
        ifl = st.get("inflight_per_replica")
        if ifl is not None and ifl > 0 and a["inflight_base"]:
            out["queue"] = arr_rep * (a["inflight_base"] / ifl)

        # --- goodput (reported, not part of the minimum) -----------------
        comp = st.get("completed")
        if comp is not None and comp > 0:
            out["goodput"] = comp / reps
        return out

    # ------------------------------------------------------------------
    def _pooled(self, service, signal):
        loc = self.local.get(service, {}).get(signal)
        if not loc or loc["n"] <= 0:
            return self._prior(signal), 1.0
        n = loc["n"]
        w = self.prior_strength
        pm = self._prior(signal)
        mean = (n * loc["mean"] + w * pm) / (n + w)
        var = (loc["m2"] / max(n - 1, 1)) if n > 1 else 1.0
        # shrink variance toward 1.0 (one e-fold) as well
        var = (n * var + w * 1.0) / (n + w)
        return mean, math.sqrt(max(var, 1e-4))

    def capacities(self, service):
        """{signal: (capacity, sigma_log, n)}.

        ``sigma_log`` is the standard deviation of log-capacity -- a
        dimensionless relative uncertainty, which is what the decision engine
        needs. Reporting it in the natural scale would make it grow with the
        capacity itself and be meaningless as a risk term.
        """
        out = {}
        for sig in ALL_SIGNALS:
            m, sd = self._pooled(service, sig)
            loc = self.local.get(service, {}).get(sig, {})
            out[sig] = (math.exp(m), sd, loc.get("n", 0.0))
        return out

    def _attribute(self, service, usable, st):
        """Decide which signal truly binds.

        Latency alone is a symptom: at saturation it explodes even when the
        resource that causes it has room, so letting it bind would conclude
        "nothing can help" and freeze the replica count (measured: the first
        rebuild bound on latency with cap~0.001 and held at 4 replicas while
        the hot tier was throttled 76 % of periods).

        When latency is anomalous *and* CPU is under pressure, attribute to the
        resource instead. Latency binds on its own only when the resources are
        calm -- that is the queue/dependency case.
        """
        ordered = sorted(usable.items(), key=lambda kv: kv[1][0])
        sig = ordered[0][0]
        if sig != "latency":
            return ordered
        clp = st.get("cpu_limit_pct") or 0.0
        thr = st.get("throttled_pct") or 0.0
        if clp > 0.35 or thr > 0.20:
            resources = [(k, v) for k, v in ordered if k in ("cpu", "throttle", "queue", "memory")]
            if resources:
                return resources
        return ordered

    def binding(self, service, st=None):
        """The binding capacity and which signal binds.

        Returns dict(signal, capacity, sigma, second_signal, spread).
        ``spread`` is the ratio between the largest and smallest signal; a
        large spread means the signals disagree and the estimate is uncertain.
        """
        caps = self.capacities(service)
        usable = {k: v for k, v in caps.items()
                  if k in SIGNALS and v[2] > 0 and v[0] > 0}
        if self.warming.get(service):
            # startup: latency/queue describe warmup, not sustainable capacity
            usable = {k: v for k, v in usable.items()
                      if k not in ("latency", "queue", "throttle")}
        if not usable:
            # Unknown capacity is NOT "zero capacity". Returning 0 here made the
            # decision engine compute an infinite requirement and saturate at
            # maxReplicas -- the opposite of a safe fallback.
            return {"signal": None, "capacity": 0.0, "sigma": 1.0,
                    "second_signal": None, "spread": 1.0, "signals": caps,
                    "valid": False}
        ordered = self._attribute(service, usable, st or self.last_state.get(service, {}))
        sig, (cap, sig_sigma, _) = ordered[0]
        second = ordered[1][0] if len(ordered) > 1 else None
        biggest = ordered[-1][1][0]
        st = self.last_state.get(service, {})
        return {
            "signal": sig,
            "capacity": cap,
            "sigma": sig_sigma,
            "second_signal": second,
            "spread": biggest / cap if cap > 0 else 1.0,
            "signals": caps,
            "cpu_limit_pct": st.get("cpu_limit_pct"),
            "throttled_pct": st.get("throttled_pct"),
            "valid": True,
        }

    # ------------------------------------------------------------------
    def snapshot(self):
        return {
            "global": self.global_stats,
            "local": {s: {k: v for k, v in d.items()} for s, d in self.local.items()},
            "adapt": self.adapt,
        }

    def load(self, snap):
        self.global_stats = snap.get("global", {})
        self.local = snap.get("local", {})
        self.adapt = snap.get("adapt", {})