"""Restore the namespace HPAs from the saved AURORA state.

Used by the POST_RUN_HOOK when the controller died before restoring them, so a
crashed run can never leave the namespace without HPAs.

This is a thin wrapper around the SAME restoration path the controller uses
(``ScaleClient.restore_hpas``), which recreates each HPA from its complete
saved manifest. There is deliberately no second implementation here: the old
tool rebuilt a minimal autoscaling/v1 HPA with only target/min/max, silently
discarding metrics, behaviour policies and every other field of the original.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                              # noqa: E402
from k8s.scale_client import ScaleClient               # noqa: E402


def main():
    cfg = Config()
    sc = ScaleClient(cfg)
    items = sc.load_saved_state()
    if not items:
        print("[restore] no saved HPA state; nothing to do")
        return 0
    restored = sc.restore_hpas()
    print(f"[restore] restored {restored}/{len(items)} HPAs "
          f"(complete manifests, same path as the controller)")
    return 0 if restored >= len(items) else 1


if __name__ == "__main__":
    sys.exit(main())