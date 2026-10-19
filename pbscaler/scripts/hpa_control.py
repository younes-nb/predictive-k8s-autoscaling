#!/usr/bin/env python3
"""Suspend / restore / status of the TrainTicket HPAs.

TrainTicket ships an HPA per stateless deployment (CPU 80%, min 1/max 10).
PBScaler sets Deployment .spec.replicas directly (paper behavior), which an
active HPA would rewrite within its 15 s sync. Options chosen for this lab:

  * suspend : delete all HPAs in the namespace, saving their manifests to
              PBS_HPA_STATE_FILE (default data/hpa_state.json)
  * restore : re-create them from the saved manifests
  * status  : list HPAs with current/min/max replicas

Runoptions (PBScaler.main does this automatically at start/exit when
PBS_SUSPEND_HPAS is true):

    cd pbscaler && ./.venv/bin/python scripts/hpa_control.py suspend
    cd pbscaler && ./.venv/bin/python scripts/hpa_control.py restore
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.Config import Config          # noqa: E402
from util.KubernetesClient import KubernetesClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['suspend', 'restore', 'status'])
    ap.add_argument('--state-file', default=None)
    args = ap.parse_args()

    config = Config()
    state_file = args.state_file or config.hpa_state_file
    k8s_util = KubernetesClient(config)

    if args.action == 'suspend':
        k8s_util.suspend_hpas(state_file)
    elif args.action == 'restore':
        k8s_util.restore_hpas(state_file)
    else:
        rows = k8s_util.hpa_status()
        print(f"{'HPA':40s}  {'current':>7s}  {'min':>3s}  {'max':>3s}")
        for name, cur, mn, mx in rows:
            print(f"{name:40s}  {cur!s:>7s}  {mn!s:>3s}  {mx!s:>3s}")
        if not rows:
            print("(no HPAs present — suspended?)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
