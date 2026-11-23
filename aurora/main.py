"""AURORA-HPA entry point (two-model bundle).

    cd aurora
    python main.py                     # resident: run until SIGTERM
    python main.py --dry-run           # decide, print, never patch

Loads and validates the bundle BEFORE touching Kubernetes: if the model is
missing, or its node/edge schema does not match the running transform, the
controller aborts and leaves the native HPAs in place.

HPA suspension is treated as a recovery problem, not a happy path:

  * the saved state is authoritative, not "how many HPAs this process deleted";
  * a restart that finds the HPAs absent inherits the obligation to restore them
    (C02);
  * startup/shutdown is protected from the first deletion onward (C04);
  * bounds come from the live HPAs or, when they are suspended, from the saved
    manifests (O01/O02).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config                                        # noqa: E402
from control.loop import AuroraLoop                              # noqa: E402
from features.groups import FEATURE_ORDER                        # noqa: E402
from k8s.leader import LeaderElection                            # noqa: E402
from k8s.scale_client import ScaleClient                         # noqa: E402
from telemetry.mimir import MimirClient                           # noqa: E402

# The bundle schema is validated against this. A bundle with the same feature
# *count* but a different order or units would load and silently misbehave.
FEATURE_SCHEMA_ID = "aurora.features.v1"


def load_bundle(cfg):
    """Load and validate the diagnosis + replica bundle."""
    import numpy as np
    import torch
    from models.graph import (DEGRADATION_CLASSES, PROVISIONING_CLASSES,
                              GraphDiagnosisModel)
    from models.replica import ACTIONS, ReplicaChangeModel

    path = os.path.join(cfg.registry_dir, "bundle.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no model bundle at {path}; run aurora/train/train_models.py")
    b = torch.load(path, map_location="cpu", weights_only=False)

    # ---- schema validation (10.3) -------------------------------------
    problems = []
    if b.get("n_node_features") != len(FEATURE_ORDER):
        problems.append(
            f"bundle expects {b.get('n_node_features')} node features but the "
            f"running transform has {len(FEATURE_ORDER)}")
    if b.get("feature_order") not in (None, list(FEATURE_ORDER)):
        problems.append("bundle feature_order does not match the running transform")
    if b.get("edge_feature_order") not in (None, ["rate", "err", "p50", "p90", "p99"]):
        problems.append("bundle edge_feature_order does not match the running transform")
    if b.get("feature_schema") not in (None, FEATURE_SCHEMA_ID):
        problems.append(f"bundle feature_schema {b.get('feature_schema')!r} "
                        f"!= {FEATURE_SCHEMA_ID!r}")
    if b.get("label_taxonomy", {}).get("degradation") not in (None, DEGRADATION_CLASSES):
        problems.append("bundle degradation taxonomy does not match the running model")
    if b.get("label_taxonomy", {}).get("provisioning") not in (None, PROVISIONING_CLASSES):
        problems.append("bundle provisioning taxonomy does not match the running model")
    if b.get("label_taxonomy", {}).get("actions") not in (None, list(ACTIONS)):
        problems.append("bundle action vocabulary does not match the running model")
    if b.get("n_edge_features") not in (None, 5):
        problems.append(f"bundle expects {b.get('n_edge_features')} edge features, "
                        f"running transform has 5")
    if "diagnosis" not in b or "replica" not in b:
        problems.append("bundle is missing model weights")
    if problems:
        raise SystemExit("incompatible bundle:\n  - " + "\n  - ".join(problems))

    diag = GraphDiagnosisModel(b["n_node_features"], n_edge_features=5,
                               d_model=b["d_model"],
                               n_degradation=b["n_degradation"])
    rep = ReplicaChangeModel(b["d_model"], n_degradation=b["n_degradation"])
    diag.load_state_dict(b["diagnosis"]); diag.eval()
    rep.load_state_dict(b["replica"]); rep.eval()

    adj = np.asarray(b["edges"], dtype=np.float32)
    print(f"[aurora] bundle loaded: services={len(b['services'])} "
          f"edges={int((adj > 0).sum())} features={b['n_node_features']} "
          f"schema={b.get('feature_schema', 'unset')}")
    return {"diagnosis": diag, "replica": rep, "services": b["services"],
            "adj": adj, "norm": b.get("norm"),
            "ood_centroid": b.get("ood_centroid"),
            "ood_threshold": b.get("ood_threshold")}


def _bounds_from_live(scale, services):
    """Per-service (min, max) from the live HPAs.

    Raises when a managed service has no readable bound: falling back to the
    global min/max would silently authorise counts the Deployment never allowed
    (O02). Callers fall back to the saved manifests only when the HPAs are
    genuinely absent, not when a single lookup failed.
    """
    bounds = {}
    live = {}
    try:
        for h in scale.autoscaling.list_namespaced_horizontal_pod_autoscaler(
                scale.cfg.namespace).items:
            ref = h.spec.scale_target_ref
            if ref.kind == "Deployment":
                live[ref.name] = (int(h.spec.min_replicas or 1),
                                  int(h.spec.max_replicas or 1))
    except Exception as exc:                                # noqa: BLE001
        raise RuntimeError(f"cannot list HPAs for bounds: {exc}") from exc
    missing = [s for s in services if s not in live]
    if missing:
        raise RuntimeError(
            "no HPA bound for managed service(s): " + ", ".join(missing))
    for s in services:
        bounds[s] = live[s]
    return bounds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-hpa-suspend", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.dry_run:
        cfg.dry_run = True
    if args.no_hpa_suspend:
        cfg.suspend_hpas = False

    # Validate the model BEFORE any Kubernetes change.
    try:
        bundle = load_bundle(cfg)
    except (FileNotFoundError, SystemExit) as exc:
        print(f"[aurora] FATAL: {exc}", file=sys.stderr)
        print("[aurora] refusing to suspend native HPAs", file=sys.stderr)
        return 2

    scale = ScaleClient(cfg)
    mimir = MimirClient(cfg)

    services = [s for s in scale.scalable_services() if s != "aurora"]
    print(f"[aurora] managed services: {len(services)}")

    # ---- bounds: live first, saved manifests when suspended (O01) -------
    bounds = {}
    try:
        bounds = _bounds_from_live(scale, services)
        source = "live HPAs"
    except RuntimeError as exc:
        print(f"[aurora] {exc}")
        bounds = scale.bounds_from_saved_state()
        source = "saved HPA manifests"
        missing = [s for s in services if s not in bounds]
        if missing:
            print(f"[aurora] FATAL: no bounds for {missing} in the saved state "
                  f"either; refusing to guess", file=sys.stderr)
            return 2
    print(f"[aurora] per-service bounds from {source}: {len(bounds)}")

    election = LeaderElection(cfg)

    # ---- recovery obligation (C02) --------------------------------------
    # The saved state, not this process's delete count, decides whether HPAs
    # must be restored. A restart that finds the HPAs already gone inherits the
    # previous process's obligation instead of dropping it.
    inherited = False
    if cfg.suspend_hpas and not cfg.dry_run:
        inherited = scale.pending_recovery()
        if inherited:
            print("[aurora] HPAs are absent but a saved manifest exists -- "
                  "restoring before starting (inherited recovery)")

    suspended = []
    try:
        if cfg.suspend_hpas and not cfg.dry_run and election.try_acquire():
            suspended = scale.suspend_hpas(services)
            print(f"[aurora] suspended {len(suspended)} HPAs")

        loop = AuroraLoop(cfg, bundle, scale, mimir,
                          model_version="bundle", services=services,
                          bounds=bounds,
                          norm=bundle.get("norm") or
                               {"mean": (bundle.get("norm") or {}).get("mean"),
                                "var": (bundle.get("norm") or {}).get("var")})
        loop.election = election if election.is_leader else None
        loop.run()
    finally:
        # Runs on every exit path after the first deletion, including an
        # exception during AuroraLoop construction (C04).
        if cfg.restore_hpas and (suspended or inherited):
            try:
                n = scale.restore_hpas()
                print(f"[aurora] restored {n} HPAs", flush=True)
            except Exception as exc:                        # noqa: BLE001
                print(f"[aurora] FATAL: HPA restore failed: {exc}; the saved "
                      f"state is kept for tools/restore_hpas.py", file=sys.stderr)
        if election.is_leader:
            election.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())