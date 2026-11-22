"""AURORA-HPA entry point.

    cd aurora
    python main.py                      # run for AURORA_DURATION_S
    python main.py --dry-run            # full loop, prints actions, no patches
    python main.py --duration 900

The controller owns replicas only for the workloads it manages, and only while
it runs: HPAs are suspended at start and restored on any clean exit.
"""

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config                                   # noqa: E402
from control.loop import AuroraLoop                          # noqa: E402
from k8s.scale_client import ScaleClient                     # noqa: E402
from telemetry.mimir import MimirClient, MimirError          # noqa: E402


def build_model(cfg):
    """Load the published model from the registry, or return None if absent."""
    import torch
    from models.encoder import TS2VecEncoder
    from models.proto import AddressabilityHead, AuroraModel, PrototypicalHead
    from models.registry import ModelRegistry

    reg = ModelRegistry(cfg.registry_dir)
    try:
        payload, meta = reg.load()
    except FileNotFoundError:
        print("[aurora] no published model found; running in hold-only mode")
        return None, None

    enc = TS2VecEncoder(meta["n_features"], d_model=meta["d_model"])
    bn = PrototypicalHead(meta["d_model"], classes=meta["classes"],
                          temperature=meta.get("temperature") or 0.1)
    ad = AddressabilityHead(meta["d_model"])
    model = AuroraModel(enc, bn, ad)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if meta.get("dist_threshold") is not None:
        model.bottleneck._dist_threshold = meta["dist_threshold"]
    print(f"[aurora] model {meta['version']} loaded "
          f"(classes={len(meta['classes'])}, threshold={meta.get('dist_threshold')})")
    return model, meta


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

    model, meta = build_model(cfg)
    scale = ScaleClient(cfg)
    mimir = MimirClient(cfg)

    # Capture the managed set BEFORE suspending the HPAs. Afterwards
    # scalable_services() finds no HPAs and falls back to every deployment,
    # which would include AURORA itself and all 22 mongos (measured: 65
    # services instead of 42).
    services = [s for s in scale.scalable_services() if s != "aurora"]
    print(f"[aurora] managed services: {len(services)}")

    suspended = 0
    if cfg.suspend_hpas and not cfg.dry_run:
        suspended = len(scale.suspend_hpas())
        print(f"[aurora] suspended {suspended} HPAs")

    loop = AuroraLoop(cfg, model, scale, mimir, norm=(meta or {}).get("norm"),
                      services=services)

    def _shutdown(signum, frame):
        print(f"[aurora] signal {signum}: restoring HPAs and exiting", flush=True)
        if cfg.restore_hpas and suspended:
            n = scale.restore_hpas()
            print(f"[aurora] restored {n} HPAs")
        loop._persist_state()
        loop._report()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run()
    finally:
        if cfg.restore_hpas and suspended:
            n = scale.restore_hpas()
            print(f"[aurora] restored {n} HPAs", flush=True)


if __name__ == "__main__":
    main()