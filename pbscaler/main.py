"""Run PBScaler against the live TrainTicket cluster.

    # from repo/pbscaler/:
    source .venv/bin/activate
    python main.py                 # 20 min run, suspends + restores HPAs
    python main.py --dry-run       # full loop, prints actions, no patches
    python main.py --duration 900

Env overrides are documented in config/Config.py (PBS_*) — e.g. for local
runs against a Mimir port-forward:
    PBS_MIMIR_URL=http://127.0.0.1:18080 python main.py --dry-run

NOTE (training): the SLO violation predictor must exist at
config.slo_predictor_path (default simulation/train_ticket/
RandomForestClassify.model) unless --dry-run, which uses an identity stub.

The upstream baseline controllers (KHPA / MicroScaler / SHOWAR / random /
None) from others/ are intentionally NOT vendored — this repo brings only the
PBScaler code itself.
"""

import argparse
import os
import sys

from config.Config import Config
from PBScaler import PBScaler
import warnings

warnings.filterwarnings("ignore")


class DenseStubModel:
    """dry-run placeholder so the GA can score candidate actions without a
    trained RandomForest (treats every strategy as SLO-safe)."""

    def predict(self, x):
        return [1 for _ in range(len(x))]


def load_predictor(config: Config):
    path = config.slo_predictor_path
    if path and os.path.exists(path):
        import joblib
        print(f"loaded SLO violation predictor: {path}")
        return joblib.load(path)
    if config.dry_run:
        print("WARNING: no trained predictor found; using identity stub "
              "(dry-run only)")
        return DenseStubModel()
    sys.exit(
        f"SLO violation predictor not found at '{path}'.\n"
        f"Collect training data (pbscaler/scripts/collect_training_data.py),\n"
        f"train it (pbscaler/simulation/RandomForestClassify.py) or use --dry-run.")


def initController(config: Config):
    return PBScaler(config, load_predictor(config))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration', type=int, default=None,
                    help='experiment duration in seconds (PBS_DURATION)')
    ap.add_argument('--dry-run', action='store_true',
                    help='run every phase but only print scaling actions')
    ap.add_argument('--no-hpa-suspend', action='store_true',
                    help='do not delete/restore the namespace HPAs this run')
    args = ap.parse_args()

    config = Config()
    if args.duration is not None:
        config.duration = args.duration
    if args.dry_run:
        config.dry_run = True
    if args.no_hpa_suspend:
        config.suspend_hpas_on_start = False

    controller = initController(config)
    controller.start()
