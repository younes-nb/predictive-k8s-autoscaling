"""Train the PBScaler SLO-violation predictor on TrainTicket data.

PAPER MODE (IEEE TSC 2024): per-service feature row [i, qps, count] with the
sklearn RandomForest default class weight (NO balancing and NO derived delta
features). A previous deviation here — class_weight='balanced_subsample' plus
[-1-step deltas] features (215 = 5x43) — was reverted on 2026-10-21 per the
paper-faithfulness directive; cold-start contribution is instead absorbed by
a 1-h k6 warmup that precedes data collection (scripts/smoke_load.sh) and is
excluded from the dataset.

Lab adaptations kept (they do not touch the paper's feature/label semantics):
  * the exact training svc ORDER is derived from the training CSV columns and
    dumped next to the model ('<model>.svcs') so PBScaler's GA util builds the
    runtime feature row against the identical order at prediction time.

Stage 7.26 hardening (methodology corrections; feature/label semantics still
unchanged — [i, qps, count] rows, default class weight):
  * chronological holdout split: the most recent --test-frac of the labeled
    rows is the test set. The previous global random.shuffle + unseeded
    train_test_split leaked future history into training and made the reported
    metrics optimistic; it is removed.
  * timestamp embargo: test rows whose timestamp is less than
    --embargo-seconds (default 60, hard floor 60) after the last training row
    are dropped. The collector samples qps (irate) and front P90 over [1m]
    PromQL windows, so a test row inside 60 s of the training tail shares
    metric-window data with training rows (label/feature bleed).
  * strict labels: rows with slo_reward not exactly 0 or 1 (the collector
    writes -1 = "no front traffic in interval") are REJECTED and counted —
    never coerced, never silently dropped.
  * strict features: rows with a non-finite <svc>&qps / <svc>&count value are
    REJECTED and counted; the previous fillna(0) fabricated zero traffic.
  * optional measured-window filter (--window-start/--window-end, ISO-8601 or
    epoch seconds): rows outside the window (e.g. the k6 warm-up, which the
    paper excludes from training data) are rejected and counted — the
    "warm-up excluded" rule is enforced mechanically, not by convention.
  * schema guards: every <svc>&qps column must have its <svc>&count partner
    and vice versa; timestamps must be present and non-decreasing; the fitted
    model must have n_features_in_ == 3 * len(svcs); the dumped .svcs file is
    read back and compared; optional --expect-svcs hard-fails on any mismatch
    with the training svc order (re-bake safety: never deploy a baked model
    whose feature row does not match the current trainer's).
  * support envelope + OOD: support_envelope.json records the per-feature
    [min, max] training envelope; the model card reports the fraction of test
    rows with any feature outside that envelope (empirical out-of-support
    assessment for the predictor the GA will query).
  * determinism: RandomForestClassifier(random_state=--seed); repeated runs
    on the same CSV produce identical artifacts.

Input — written by pbscaler/scripts/collect_training_data.py during a k6 run
(NASA replay) with the namespace's native HPAs performing elastic scaling
(paper Sec. IV-C2: historical metric data generated with the K8s default
autoscaler) and labeled with the paper's istio front P90 (Sec. IV-C2):

    <data>/train_data.csv   columns: timestamp, ..., <svc>&qps, <svc>&count,
                            front_p90, slo_reward (1 = no SLO violation,
                            0 = violation)

Output — model + stats to simulation/<bench>/ (RandomForestClassify.model,
rf.pkl, RandomForestClassify.svcs, support_envelope.json, model_card.json).
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             recall_score, roc_auc_score, roc_curve)

TRAINER_VERSION = "rf-train-2.0-stage726"

# reference lists (documented; the active list is ALWAYS re-derived from the
# training CSV's <svc>&qps columns — a CSV without them is a hard error now)
TT_SVCS = ['ts-admin-basic-info-service', 'ts-admin-order-service', 'ts-admin-route-service',
           'ts-admin-travel-service', 'ts-admin-user-service', 'ts-assurance-service',
           'ts-auth-service', 'ts-avatar-service', 'ts-basic-service', 'ts-cancel-service',
           'ts-config-service', 'ts-consign-price-service', 'ts-consign-service',
           'ts-contacts-service', 'ts-delivery-service', 'ts-execute-service',
           'ts-food-map-service', 'ts-food-service', 'ts-inside-payment-service',
           'ts-news-service', 'ts-notification-service', 'ts-order-other-service',
           'ts-order-service', 'ts-payment-service', 'ts-preserve-other-service',
           'ts-preserve-service', 'ts-price-service', 'ts-rebook-service',
           'ts-route-plan-service', 'ts-route-service', 'ts-seat-service',
           'ts-security-service', 'ts-station-service', 'ts-ticket-office-service',
           'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-plan-service',
           'ts-travel-service', 'ts-travel2-service', 'ts-ui-dashboard',
           'ts-user-service', 'ts-verification-code-service', 'ts-voucher-service']
BOUTIQUE_SVCS = ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 'emailservice',
                 'frontend', 'paymentservice', 'productcatalogservice', 'recommendationservice',
                 'shippingservice']

EMBARGO_FLOOR_SECONDS = 60  # collector qps/P90 PromQL windows are [1m]


class TrainerError(Exception):
    """Raised for any hard training-data / schema / contract failure."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def svcs_from_csv(df: pd.DataFrame) -> list:
    qps_cols = sorted(c for c in df.columns if c.endswith('&qps'))
    return [c[:-4] for c in qps_cols]


def validate_schema(df: pd.DataFrame) -> list:
    """Hard schema checks; returns the derived svc list on success."""
    for required in ('timestamp', 'slo_reward'):
        if required not in df.columns:
            raise TrainerError(f"missing required column '{required}' in training CSV")
    qps_svcs = {c[:-4] for c in df.columns if c.endswith('&qps')}
    cnt_svcs = {c[:-6] for c in df.columns if c.endswith('&count')}
    unpaired_qps = sorted(qps_svcs - cnt_svcs)
    unpaired_cnt = sorted(cnt_svcs - qps_svcs)
    if unpaired_qps or unpaired_cnt:
        raise TrainerError(
            'unpaired qps/count columns: '
            f"qps without count={unpaired_qps}, count without qps={unpaired_cnt}")
    if not qps_svcs:
        raise TrainerError('no <svc>&qps columns in training CSV')
    return sorted(qps_svcs)


def parse_timestamps(series: pd.Series) -> np.ndarray:
    """Timestamps -> epoch seconds (float). Accepts ISO-8601 strings (what the
    collector writes: pd.Timestamp.now(tz='UTC').isoformat()) or numeric
    epochs (median < 1e11 interpreted as seconds, otherwise milliseconds)."""
    s = series.astype(str)
    numeric = pd.to_numeric(s, errors='coerce')
    if numeric.notna().all():
        med = float(numeric.median())
        vals = numeric.to_numpy(dtype=float)
        return vals / 1000.0 if med >= 1e11 else vals
    try:
        ts = pd.to_datetime(s, utc=True, errors='raise')
    except Exception as exc:
        raise TrainerError(f'cannot parse timestamp column as ISO/epoch: {exc}')
    return np.array([t.timestamp() for t in ts], dtype=float)


def parse_single_ts(value: str) -> float:
    """One CLI timestamp: epoch seconds (<1e11), epoch milliseconds, or ISO-8601."""
    try:
        v = float(value)
        return v / 1000.0 if v >= 1e11 else v
    except ValueError:
        pass
    try:
        return pd.to_datetime(value, utc=True).timestamp()
    except Exception as exc:
        raise TrainerError(f"cannot parse timestamp '{value}' as epoch/ISO: {exc}")


def load_dataset(path: str, window_start=None, window_end=None):
    """Load + validate the collector CSV. Order is preserved (chronological).

    Returns dict with: svcs, ts (epoch s, kept rows), X (float, n x 3k),
    y (int 0/1), rejects {invalid_label, nonfinite_feature}, n_rows_total,
    n_rejected, data_sha256. Rows with slo_reward not exactly 0/1 or with any
    non-finite qps/count are rejected and counted (reasons may overlap)."""
    df = pd.read_csv(path)
    svcs = validate_schema(df)
    ts = parse_timestamps(df['timestamp'])
    if len(ts) and not np.all(np.diff(ts) >= 0):
        bad = int(np.argmax(np.diff(ts) < 0)) + 1
        raise TrainerError(
            f'timestamps not non-decreasing at row {bad} '
            f'({ts[bad - 1]:.0f} -> {ts[bad]:.0f}); chronological split requires order')

    y_raw = pd.to_numeric(df['slo_reward'], errors='coerce')
    valid_label = y_raw.isin([0, 1]).to_numpy()

    feat_cols = [f'{s}&qps' for s in svcs] + [f'{s}&count' for s in svcs]
    feats = df[feat_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    valid_feat = np.isfinite(feats).all(axis=1)

    keep = valid_label & valid_feat
    rejects = {
        'invalid_label': int((~valid_label).sum()),
        'nonfinite_feature': int((~valid_feat).sum()),
    }
    if window_start is not None or window_end is not None:
        in_win = np.ones(len(ts), dtype=bool)
        if window_start is not None:
            in_win &= ts >= window_start
        if window_end is not None:
            in_win &= ts <= window_end
        rejects['outside_window'] = int((~in_win).sum())
        keep = keep & in_win

    n_kept = int(keep.sum())
    X = np.empty((n_kept, 3 * len(svcs)), dtype=float)
    for i, svc in enumerate(svcs):
        X[:, 3 * i] = float(i)
        X[:, 3 * i + 1] = feats[keep, i]                # <svc>&qps  (first block)
        X[:, 3 * i + 2] = feats[keep, len(svcs) + i]    # <svc>&count (second block)
    y = y_raw.to_numpy()[keep].astype(int)

    return {
        'svcs': svcs,
        'ts': ts[keep],
        'X': X,
        'y': y,
        'rejects': rejects,
        'n_rows_total': int(len(df)),
        'n_rejected': int(len(df) - n_kept),
        'data_sha256': sha256_file(path),
    }


def chronological_split(ts: np.ndarray, y: np.ndarray, test_frac: float,
                        embargo_seconds: float, min_test_violations: int = 0):
    """Hold out the most recent test_frac of rows; drop test rows inside the
    embargo window after the last training timestamp.

    min_test_violations guards against a degenerate holdout. A chronological
    split takes the *tail* of the run as the test set, so if the violations all
    happened early (e.g. one cold-start episode) the test set can end up with
    zero violations and every metric (recall, AUC) becomes meaningless. When that
    happens the split point is moved earlier until the test set contains enough
    violations, or the run is rejected outright (2026-11-21).
    """
    n = len(ts)
    if n < 2:
        raise TrainerError(f'need >= 2 labeled rows, got {n}')

    def split_at(k):
        last_train_ts = float(ts[k - 1])
        embargo_end = last_train_ts + float(embargo_seconds)
        test_start = int(np.searchsorted(ts, embargo_end, side='left'))
        return k, test_start, last_train_ts, embargo_end

    k = min(max(int(np.floor(n * (1.0 - test_frac))), 1), n - 1)
    k, test_start, last_train_ts, embargo_end = split_at(k)
    if test_start >= n:
        raise TrainerError(
            f'embargo of {embargo_seconds:g}s removes the entire test set '
            f'(last train ts={last_train_ts:.0f}, embargo end={embargo_end:.0f}, '
            f'last row ts={ts[-1]:.0f})')

    requested_k = k
    n_test_viol = int((y[test_start:] == 0).sum())
    if min_test_violations > 0:
        while n_test_viol < min_test_violations and k > 1:
            k -= 1
            k2, ts2, lt, ee = split_at(k)
            if ts2 >= n:
                break
            k, test_start, last_train_ts, embargo_end = k2, ts2, lt, ee
            n_test_viol = int((y[test_start:] == 0).sum())
        if n_test_viol < min_test_violations:
            raise TrainerError(
                f'test set has only {n_test_viol} violation(s), fewer than '
                f'--min-test-violations={min_test_violations}. The run does not '
                f'contain enough spread-out violations for a chronological '
                f'holdout; pick a load level or window that violates SLO outside '
                f'the cold-start minutes.')

    return (np.arange(0, k), np.arange(test_start, n),
            {'last_train_ts_epoch_s': last_train_ts,
             'embargo_end_epoch_s': embargo_end,
             'test_rows_dropped_by_embargo': test_start - k,
             'requested_split_index': int(requested_k),
             'used_split_index': int(k),
                'test_violations': int(n_test_viol),
                'min_test_violations': int(min_test_violations)})


def feature_bounds(X: np.ndarray, svcs: list) -> list:
    bounds = []
    for i, svc in enumerate(svcs):
        bounds.append({'feature': f'{svc}#index', 'slot': 'index', 'col': 3 * i,
                       'min': float(X[:, 3 * i].min()), 'max': float(X[:, 3 * i].max())})
        bounds.append({'feature': f'{svc}&qps', 'slot': 'qps', 'col': 3 * i + 1,
                       'min': float(X[:, 3 * i + 1].min()), 'max': float(X[:, 3 * i + 1].max())})
        bounds.append({'feature': f'{svc}&count', 'slot': 'count', 'col': 3 * i + 2,
                       'min': float(X[:, 3 * i + 2].min()), 'max': float(X[:, 3 * i + 2].max())})
    return bounds


def ood_fraction(X: np.ndarray, bounds: list) -> float:
    """Fraction of rows with ANY feature outside the training envelope."""
    if len(X) == 0:
        return 0.0
    outside = np.zeros(len(X), dtype=bool)
    for b in bounds:
        col = X[:, b['col']]
        outside |= (col < b['min']) | (col > b['max'])
    return float(outside.mean())


def class_balance(y: np.ndarray) -> dict:
    return {'0': int((y == 0).sum()), '1': int((y == 1).sum())}


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    try:
        auc = float(roc_auc_score(y_test, y_pred))
    except ValueError:
        auc = None  # single-class test set
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'recall_violation_pos_label_0': float(recall_score(y_test, y_pred, pos_label=0,
                                                           zero_division=0)),
        'roc_auc': auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(y_test, y_pred, digits=3,
                                                       zero_division=0),
    }


def write_artifacts(out_dir, model, svcs, summary, args) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'RandomForestClassify.model')
    joblib.dump(model, model_path)

    # paper-metric ROC pickle (kept from the previous trainer revision)
    fpr, tpr, thresholds = roc_curve(summary['y_test'], model.predict(summary['X_test']),
                                     pos_label=None, sample_weight=None,
                                     drop_intermediate=True)
    pickle.dump({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
                open(os.path.join(out_dir, 'rf.pkl'), 'wb'))

    # bake the exact training svc ORDER next to the model + round-trip verify
    svc_path = os.path.join(out_dir, 'RandomForestClassify.svcs')
    json.dump(svcs, open(svc_path, 'w'))
    with open(svc_path) as f:
        if json.load(f) != svcs:
            raise TrainerError('.svcs round-trip mismatch after dump')

    # support envelope: per-feature [min, max] of the training rows
    envelope = {
        'created_utc': utc_now_iso(),
        'trainer_version': TRAINER_VERSION,
        'data_path': args.data,
        'data_sha256': summary['data_sha256'],
        'svcs': svcs,
        'n_features': 3 * len(svcs),
        'n_train': summary['n_train'],
        'n_test': summary['n_test'],
        'test_frac': args.test_frac,
        'embargo_seconds': args.embargo_seconds,
        'split': summary['split'],
        'rejects': summary['rejects'],
        'n_rows_total': summary['n_rows_total'],
        'n_rejected': summary['n_rejected'],
        'class_balance': summary['class_balance'],
        'feature_bounds': summary['bounds'],
    }
    env_path = os.path.join(out_dir, 'support_envelope.json')
    json.dump(envelope, open(env_path, 'w'), indent=1)

    card = {
        'created_utc': utc_now_iso(),
        'trainer_version': TRAINER_VERSION,
        'sklearn_version': sklearn.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'joblib_version': joblib.__version__,
        'seed': args.seed,
        'data_path': args.data,
        'data_sha256': summary['data_sha256'],
        'svcs': svcs,
        'n_features': 3 * len(svcs),
        'model_n_features_in': int(model.n_features_in_),
        'split': summary['split'],
        'rejects': summary['rejects'],
        'n_rows_total': summary['n_rows_total'],
        'n_rejected': summary['n_rejected'],
        'class_balance': summary['class_balance'],
        'metrics': summary['metrics'],
        'test_ood_fraction': summary['test_ood_fraction'],
        'ood_definition': ('fraction of test rows with any [i, qps, count] feature '
                           'outside the per-feature [min, max] of the training rows'),
        'notes': [
            'paper feature semantics: per-service [i, qps, count], default class weight',
            'chronological holdout (most recent test_frac), embargo >= 60 s hard floor',
            'labels strictly 0/1; collector -1 (no traffic) rows rejected and counted',
            'non-finite qps/count rows rejected and counted (no fillna)',
            'RandomForestClassifier(random_state=seed) for reproducible artifacts',
        ],
    }
    json.dump(card, open(os.path.join(out_dir, 'model_card.json'), 'w'), indent=1)


def train_from_args(args) -> dict:
    window = {}
    if args.window_start is not None:
        window['window_start_epoch_s'] = parse_single_ts(args.window_start)
    if args.window_end is not None:
        window['window_end_epoch_s'] = parse_single_ts(args.window_end)
    if 'window_start_epoch_s' in window and 'window_end_epoch_s' in window \
            and window['window_start_epoch_s'] >= window['window_end_epoch_s']:
        raise TrainerError('--window-start must be strictly before --window-end')
    data = load_dataset(args.data,
                        window_start=window.get('window_start_epoch_s'),
                        window_end=window.get('window_end_epoch_s'))
    svcs = data['svcs']
    print(f'using {len(svcs)} services from {args.data}')
    print(f"rows: {data['n_rows_total']} total, {data['n_rejected']} rejected "
          f"(invalid_label={data['rejects']['invalid_label']}, "
          f"nonfinite_feature={data['rejects']['nonfinite_feature']}), "
          f"{len(data['y'])} usable")
    if args.expect_svcs:
        expected = [s.strip() for s in args.expect_svcs.split(',') if s.strip()]
        if expected != svcs:
            raise TrainerError(
                '--expect-svcs mismatch: expected '
                f'{expected!r}, training CSV derived {svcs!r}')
    if len(data['y']) == 0:
        raise TrainerError('no usable labeled rows after rejection')

    train_idx, test_idx, split_info = chronological_split(
        data['ts'], data['y'], args.test_frac, args.embargo_seconds,
        args.min_test_violations)
    X_train, y_train = data['X'][train_idx], data['y'][train_idx]
    X_test, y_test = data['X'][test_idx], data['y'][test_idx]
    print(f"samples: {len(y_train)} train / {len(y_test)} test "
          f"(violations train={int((y_train == 0).sum())}, test={int((y_test == 0).sum())})")
    ts_to_iso = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
    split_info['ts_train_utc'] = [ts_to_iso(data['ts'][train_idx].min()),
                                  ts_to_iso(data['ts'][train_idx].max())]
    split_info['ts_test_utc'] = [ts_to_iso(data['ts'][test_idx].min()),
                                 ts_to_iso(data['ts'][test_idx].max())]

    model = RandomForestClassifier(random_state=args.seed)
    model.fit(X_train, y_train)
    if model.n_features_in_ != 3 * len(svcs):
        raise TrainerError(
            f'model n_features_in_={model.n_features_in_} != 3*len(svcs)={3 * len(svcs)}')

    metrics = compute_metrics(model, X_test, y_test)
    print('acc', metrics['accuracy'])
    print('recall', metrics['recall_violation_pos_label_0'])
    print('auc', metrics['roc_auc'])
    print(metrics['classification_report'])

    bounds = feature_bounds(X_train, svcs)
    ood = ood_fraction(X_test, bounds)
    print(f'test OOD fraction (outside training support envelope): {ood:.4f}')

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.bench)
    summary = {
        'y_test': y_test, 'X_test': X_test,
        'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
        'split': dict(split_info, test_frac=args.test_frac,
                      embargo_seconds=args.embargo_seconds,
                      n_train=int(len(y_train)), n_test=int(len(y_test))),
        'window': window,
        'rejects': data['rejects'],
        'n_rows_total': data['n_rows_total'],
        'n_rejected': data['n_rejected'],
        'class_balance': {'train': class_balance(y_train), 'test': class_balance(y_test)},
        'data_sha256': data['data_sha256'],
        'bounds': bounds,
        'test_ood_fraction': ood,
        'metrics': metrics,
    }
    write_artifacts(out_dir, model, svcs, summary, args)
    print(f'model written: {os.path.join(out_dir, "RandomForestClassify.model")}\n'
          f'(pickled ROC: {os.path.join(out_dir, "rf.pkl")}; '
          f'predictor svc order: {os.path.join(out_dir, "RandomForestClassify.svcs")}; '
          f'support envelope: {os.path.join(out_dir, "support_envelope.json")}; '
          f'model card: {os.path.join(out_dir, "model_card.json")})')
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='../data/train_ticket/train_data.csv',
                    help='path to train_data.csv from the collector')
    ap.add_argument('--bench', default='train_ticket', choices=['train_ticket', 'boutique'])
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--test-frac', type=float, default=0.2,
                    help='fraction of the most recent labeled rows held out for test')
    ap.add_argument('--embargo-seconds', type=float, default=EMBARGO_FLOOR_SECONDS,
                    help=f'embargo between train and test (hard floor '
                         f'{EMBARGO_FLOOR_SECONDS}s = collector [1m] PromQL windows)')
    ap.add_argument('--min-test-violations', type=int, default=30,
                    help='reject a chronological holdout whose test set has fewer '
                         'violations than this (default 30; 0 disables the guard)')
    ap.add_argument('--expect-svcs',
                    help='comma-separated svc order that MUST match the CSV-derived '
                         'order exactly (hard fail otherwise; re-bake safety)')
    ap.add_argument('--window-start',
                    help='ISO-8601 or epoch timestamp; rows before it are rejected '
                         '(use the measured-window start to exclude the k6 warm-up)')
    ap.add_argument('--window-end',
                    help='ISO-8601 or epoch timestamp; rows after it are rejected '
                         '(e.g. exclude the recovery tail from training data)')
    ap.add_argument('--out-dir',
                    help='output dir (default: simulation/<bench> next to this file)')
    return ap


def run_cli(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.test_frac <= 0 or args.test_frac >= 1:
        print('error: --test-frac must be in (0, 1)', file=sys.stderr)
        return 2
    if args.embargo_seconds < EMBARGO_FLOOR_SECONDS:
        print(f'error: --embargo-seconds must be >= {EMBARGO_FLOOR_SECONDS} '
              '(collector qps/P90 use [1m] windows; smaller embargoes leak)',
              file=sys.stderr)
        return 2
    try:
        train_from_args(args)
    except TrainerError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


def setup_seed(seed: int):
    np.random.seed(seed)


if __name__ == '__main__':
    args = build_parser().parse_args()
    setup_seed(args.seed)
    sys.exit(run_cli())
