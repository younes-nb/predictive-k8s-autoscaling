"""Staged adaptation.

Level 1 -- baseline adaptation: the running normalisation statistics in
``WindowManager`` update every cycle (Welford), no model change.

Level 2 -- local adaptation: per-service prototypes and capacity posteriors,
updated only from *validated* observations. This is what runs online.

Level 3 -- global refresh: offline only. Produces a candidate model version; it
is never trained inside the control loop and never auto-promoted.

Level 4 -- validation and controlled rollout: the registry keeps the previous
version, ``publish`` is atomic, ``rollback`` repoints ``current``. Promotion is a
manual/scripted step, not a drift reaction.
"""

import json
import math
import os
import time


class AdaptationManager:
    def __init__(self, cfg, drift, ledger, prototype_ema=0.2,
                 confidence_floor=0.6, min_validated=5):
        self.cfg = cfg
        self.drift = drift
        self.ledger = ledger
        self.prototype_ema = prototype_ema
        self.confidence_floor = confidence_floor
        self.min_validated = min_validated
        self.validated = {}     # service -> count of validated examples
        self.last_global_refresh = None

    # ------------------------------------------------------------------
    def level1(self, service):
        """Baseline adaptation is implicit: the window manager updates its
        running statistics on every ingest. Returns what changed."""
        return {"level": 1, "service": service, "applied": True}

    # ------------------------------------------------------------------
    def level2(self, service, embedding, bottleneck, confidence, outcome_ok):
        """Update the local prototype only when the action was validated.

        An unvalidated action must never become a label: ``outcome_ok`` is set
        by the ledger once post-action measurements exist.
        """
        if not outcome_ok:
            return {"applied": False, "reason": "outcome not validated"}
        if confidence < self.confidence_floor:
            return {"applied": False, "reason": "confidence below floor"}
        n = self.validated.get(service, 0)
        if n < self.min_validated:
            self.validated[service] = n + 1
            return {"applied": False, "reason": f"collecting ({n + 1}/{self.min_validated})"}
        self.validated[service] = n + 1
        return {"applied": True, "class": bottleneck, "ema": self.prototype_ema}

    # ------------------------------------------------------------------
    def level3_prepare(self, registry, model, feature_schema, validation, notes=""):
        """Offline hand-off: publish a candidate. The caller decides whether to
        promote it; this method does not switch ``current``."""
        os.makedirs(self.cfg.registry_dir, exist_ok=True)
        from models.registry import ModelRegistry
        reg = registry or ModelRegistry(self.cfg.registry_dir)
        version = f"v{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        reg.publish(model, version, feature_schema, validation, notes)
        self.last_global_refresh = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return version

    # ------------------------------------------------------------------
    def drift_report(self):
        return {
            "missing_ticks": dict(self.drift.missing_ticks),
            "cusum_pos": {k: round(v, 3) for k, v in self.drift.cusum_pos.items()},
            "capacity_drift": [s for s in self.drift.cusum_pos
                               if self.drift.capacity_drift(s)],
            "config_hash": self.drift.config_hash,
            "topology_hash": self.drift.topology_hash,
        }