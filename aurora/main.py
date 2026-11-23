"""AURORA-HPA entry point (two-model bundle).

    cd aurora
    python main.py                     # run for AURORA_DURATION_S
    python main.py --dry-run           # decide, print, never patch

Loads and validates the bundle BEFORE touching Kubernetes: if the model is
missing, or its node/edge schema does not match the running transform, the
controller aborts and leaves the native HPAs in place.
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


def load_bundle(cfg):
    """Load and validate the diagnosis + replica bundle."""
    import numpy as np
    import torch
    from models.graph import GraphDiagnosisModel
    from models.replica import ReplicaChangeModel

    path = os.path.join(cfg.registry_dir, "bundle.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no model bundle at {path}; run aurora/train/train_models.py")
    b = torch.load(path, map_location="cpu", weights_only=False)

    if b.get("n_node_features") != len(FEATURE_ORDER):
        raise SystemExit(
            f"bundle expects {b.get('n_node_features')} node features but the "
            f"running transform has {len(FEATURE_ORDER)}; refusing to load")
    if "diagnosis" not in b or "replica" not in b:
        raise SystemExit("bundle is missing model weights")

    diag = GraphDiagnosisModel(b["n_node_features"], n_edge_features=1,
                               d_model=b["d_model"],
                               n_degradation=b["n_degradation"])
    rep = ReplicaChangeModel(b["d_model"],
                             n_degradation=b["n_degradation"])
    diag.load_state_dict(b["diagnosis"]); diag.eval()
    rep.load_state_dict(b["replica"]); rep.eval()

    import numpy as np
    adj = np.asarray(b["edges"], dtype=np.float32)
    print(f"[aurora] bundle loaded: services={len(b['services'])} "
          f"edges={int((adj > 0).sum())} features={b['n_node_features']}")
    return {"diagnosis": diag, "replica": rep, "services": b["services"],
            "adj": adj}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-hpa-suspend", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.duration is not None:
        cfg.duration_s = args.duration
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

    election = LeaderElection(cfg)
    suspended = 0
    if cfg.suspend_hpas and not cfg.dry_run and election.try_acquire():
        suspended = len(scale.suspend_hpas())
        print(f"[aurora] suspended {suspended} HPAs")

    loop = AuroraLoop(cfg, bundle, scale, mimir,
                      model_version="bundle", services=services)
    loop.election = election if election.is_leader else None

    def _shutdown(signum, frame):
        print(f"[aurora] signal {signum}: restoring HPAs", flush=True)
        if cfg.restore_hpas and suspended:
            print(f"[aurora] restored {scale.restore_hpas()} HPAs")
        election.release()
        sys.exit(0)

    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run()
    finally:
        if cfg.restore_hpas and suspended:
            print(f"[aurora] restored {scale.restore_hpas()} HPAs", flush=True)
        if election.is_leader:
            election.release()


if __name__ == "__main__":
    main()