"""Stage 7.26 offline tests for pbscaler/simulation/RandomForestClassify.py.

Synthetic collector CSVs only — no network, no cluster, no real datasets.
Run with the pinned PBScaler env:

    pbscaler/.venv/bin/python -m unittest discover -s pbscaler/tests -v
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PY = os.path.join(_REPO, 'pbscaler', 'simulation', 'RandomForestClassify.py')

_spec = importlib.util.spec_from_file_location('rfc_stage726', _MODEL_PY)
rfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfc)

SVCS = ['svc-a', 'svc-b', 'svc-c']          # already alphabetical (= derived order)
BASE_TS = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
SPACING_S = 300.0                            # >> 60 s embargo floor


def iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()


def write_csv(path, n_rows, ood_tail=0, bad_labels=(), nan_rows=(),
              drop_count_cols=(), include_timestamp=True, ts_mode='iso'):
    """Synthetic collector CSV. ood_tail: last k rows get qps far outside the
    training envelope; bad_labels: row indexes with slo_reward not in {0,1};
    nan_rows: row indexes with a NaN qps."""
    cols = (['timestamp'] if include_timestamp else []) + \
        ['front_p90', 'front_avg', 'front_source', 'slo_reward', 'total_qps']
    for s in SVCS:
        cols += [f'{s}&qps', f'{s}&count']
    if drop_count_cols:
        cols = [c for c in cols if c not in drop_count_cols]
    rows = []
    for i in range(n_rows):
        row = {}
        if include_timestamp:
            t = BASE_TS + i * SPACING_S
            row['timestamp'] = t if ts_mode == 'epoch_s' else (t * 1000.0 if ts_mode == 'epoch_ms' else iso(t))
        row['front_p90'] = 20.0 + (i % 5)
        row['front_avg'] = 10.0
        row['front_source'] = 'destination'
        if i in bad_labels:
            row['slo_reward'] = 2 if i % 2 else -1
        else:
            row['slo_reward'] = 1 if i % 5 else 0
        row['total_qps'] = 30.0 + i
        qps_base = 0.5 + (i % 10) * 0.01               # periodic: test rows stay in-envelope
        ood = i >= n_rows - ood_tail
        for j, s in enumerate(SVCS):
            q = (500.0 + j) if ood else qps_base + j * 0.1
            row[f'{s}&qps'] = ('' if i in nan_rows else q)  # '' parses to NaN
            row[f'{s}&count'] = 5 + (i % 7)
        rows.append(row)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


class TempDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='rfc726_')
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def csv(self, name='train_data.csv', **kw):
        return write_csv(os.path.join(self.tmp, name), **kw)

    def quiet_train(self, argv):
        out, errout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errout):
            rc = rfc.run_cli(argv)
        return rc, out.getvalue() + errout.getvalue()


class TestChronologicalSplit(TempDirs):
    def test_split_is_chronological_with_embargo(self):
        csv = self.csv(n_rows=100)
        out_dir = os.path.join(self.tmp, 'out')
        rc, _ = self.quiet_train(['--data', csv, '--out-dir', out_dir, '--seed', '7'])
        self.assertEqual(rc, 0)
        env = load_json(os.path.join(out_dir, 'support_envelope.json'))
        self.assertLess(env['split']['last_train_ts_epoch_s'] + 60.0,
                        self._first_test_ts(env))
        # train strictly before test
        tr_lo, tr_hi = env['split']['ts_train_utc']
        te_lo, te_hi = env['split']['ts_test_utc']
        self.assertLess(tr_hi, te_lo)

    @staticmethod
    def _first_test_ts(env):
        return datetime.fromisoformat(env['split']['ts_test_utc'][0]).timestamp()

    def test_embargo_drops_rows_inside_window(self):
        ts = np.arange(80, dtype=float) * SPACING_S          # n=100 -> k=80 train rows
        tail = ts[79] + 6.0 * np.arange(1, 21)               # test candidates +6..+120 s
        ts_full = np.concatenate([ts, tail])
        train_idx, test_idx, info = rfc.chronological_split(ts_full, 0.2, 60)
        self.assertEqual(info['test_rows_dropped_by_embargo'], 9)   # +6..+54 s dropped
        self.assertEqual(test_idx[0], 89)                    # first row >= ts[79]+60
        self.assertAlmostEqual(ts_full[89], ts[79] + 60.0)
        self.assertEqual(train_idx[-1], 79)

    def test_embargo_consumes_test_set_fails(self):
        ts = np.arange(100, dtype=float)                      # 1 s spacing
        with self.assertRaises(rfc.TrainerError):
            rfc.chronological_split(ts, 0.2, 60)

    def test_embargo_below_floor_rejected_by_cli(self):
        csv = self.csv(n_rows=50)
        rc, err = self.quiet_train(['--data', csv, '--embargo-seconds', '30',
                                    '--out-dir', os.path.join(self.tmp, 'o')])
        self.assertEqual(rc, 2)
        self.assertIn('embargo', err)


class TestEmbargoAtCollectorSpacing(TempDirs):
    """Stage 7.28 reconciliation: the collector samples every 5 s and every
    feature/label query uses a [1m] PromQL window, so adjacent rows overlap by
    55 s of metric-window data. These tests pin the embargo at that real
    spacing (the other tests use a 300 s spacing that cannot reproduce the
    overlap regime)."""

    COLLECTOR_SPACING_S = 5.0

    def _ts(self, n):
        return np.array([BASE_TS + i * self.COLLECTOR_SPACING_S for i in range(n)])

    def test_embargo_makes_train_and_test_windows_disjoint(self):
        n = 360                                   # cal10500 shape: 30 min at 5 s
        ts = self._ts(n)
        train_idx, test_idx, info = rfc.chronological_split(ts, 0.2, 60)
        self.assertEqual(train_idx[0], 0)
        self.assertEqual(train_idx[-1], 287)      # first 80 %, contiguous block
        self.assertEqual(test_idx[-1], n - 1)     # test = last block, contiguous
        gap = ts[test_idx[0]] - ts[train_idx[-1]]
        self.assertAlmostEqual(gap, 60.0)         # first row past the embargo
        self.assertEqual(info['test_rows_dropped_by_embargo'], 11)  # +5..+55 s
        # [1m] PromQL windows (T-60, T] of the boundary rows share no samples
        self.assertGreaterEqual(ts[test_idx[0]] - 60.0, ts[train_idx[-1]])

    def test_split_is_deterministic_and_not_random(self):
        ts = self._ts(360)
        a = rfc.chronological_split(ts, 0.2, 60)
        b = rfc.chronological_split(ts, 0.2, 60)
        self.assertEqual(a[0].tolist(), b[0].tolist())
        self.assertEqual(a[1].tolist(), b[1].tolist())
        self.assertTrue(np.all(np.diff(a[0]) == 1) and np.all(np.diff(a[1]) == 1))

    def test_full_train_at_5s_spacing_records_embargoed_split(self):
        old = SPACING_S
        globals()['SPACING_S'] = self.COLLECTOR_SPACING_S
        try:
            csv = self.csv(n_rows=360)
            out_dir = os.path.join(self.tmp, 'out')
            rc, _ = self.quiet_train(['--data', csv, '--out-dir', out_dir, '--seed', '7'])
            self.assertEqual(rc, 0)
            env = load_json(os.path.join(out_dir, 'support_envelope.json'))
            split = env['split']
            self.assertEqual(split['embargo_seconds'], 60.0)
            self.assertEqual((split['n_train'], split['n_test']), (288, 61))
            first_test = datetime.fromisoformat(split['ts_test_utc'][0]).timestamp()
            self.assertAlmostEqual(first_test - split['last_train_ts_epoch_s'], 60.0)
            self.assertLess(split['ts_train_utc'][1], split['ts_test_utc'][0])
            # two-way chronological split: no separate validation period exists
            self.assertNotIn('n_val', split)
        finally:
            globals()['SPACING_S'] = old


class TestStrictRejection(TempDirs):
    def test_invalid_labels_rejected_and_counted(self):
        csv = self.csv(n_rows=60, bad_labels=(3, 10, 40))     # -> 2, -1, 2
        data = rfc.load_dataset(csv)
        self.assertEqual(data['rejects']['invalid_label'], 3)
        self.assertEqual(data['n_rejected'], 3)
        self.assertTrue(set(np.unique(data['y'])).issubset({0, 1}))
        self.assertEqual(len(data['y']), 57)

    def test_nonfinite_features_rejected_not_filled(self):
        csv = self.csv(n_rows=60, nan_rows=(5,))
        data = rfc.load_dataset(csv)
        self.assertEqual(data['rejects']['nonfinite_feature'], 1)
        self.assertEqual(data['n_rejected'], 1)
        self.assertTrue(np.isfinite(data['X']).all())         # no fillna(0) leakage
        # the rejected row is gone entirely (57 = 60 - 1)
        self.assertEqual(len(data['y']), 59)

    def test_nan_label_counts_as_invalid(self):
        csv = self.csv(n_rows=30, bad_labels=(7,))
        data = rfc.load_dataset(csv)
        self.assertEqual(data['rejects']['invalid_label'], 1)


class TestMeasuredWindowFilter(TempDirs):
    def test_window_excludes_rows_and_counts(self):
        csv = self.csv(n_rows=100)                      # ts: BASE_TS + i*300
        ws = BASE_TS + 40 * SPACING_S                   # keep rows 40..99
        data = rfc.load_dataset(csv, window_start=ws)
        self.assertEqual(data['rejects']['outside_window'], 40)
        self.assertEqual(len(data['y']), 60)
        self.assertGreaterEqual(data['ts'].min(), ws)

    def test_window_epoch_ms_and_iso_accepted_via_cli(self):
        csv = self.csv(n_rows=100)
        out = os.path.join(self.tmp, 'w')
        rc, _ = self.quiet_train(['--data', csv, '--out-dir', out,
                                  '--window-start', str((BASE_TS + 40 * SPACING_S) * 1000.0)])
        self.assertEqual(rc, 0)
        env = load_json(os.path.join(out, 'support_envelope.json'))
        self.assertEqual(env['rejects']['outside_window'], 40)

    def test_window_reversed_rejected(self):
        csv = self.csv(n_rows=50)
        rc, err = self.quiet_train(['--data', csv, '--out-dir', os.path.join(self.tmp, 'z'),
                                    '--window-start', '2026-11-02T00:00:00+00:00',
                                    '--window-end', '2026-11-01T00:00:00+00:00'])
        self.assertEqual(rc, 1)
        self.assertIn('strictly before', err)


class TestSchemaGuards(TempDirs):
    def test_unpaired_qps_column_fails(self):
        csv = self.csv(n_rows=30, drop_count_cols=('svc-b&count',))
        with self.assertRaises(rfc.TrainerError) as ctx:
            rfc.load_dataset(csv)
        self.assertIn('unpaired', str(ctx.exception))

    def test_unpaired_count_column_fails(self):
        csv = self.csv(n_rows=30, drop_count_cols=('svc-c&qps',))
        with self.assertRaises(rfc.TrainerError):
            rfc.load_dataset(csv)

    def test_missing_timestamp_fails(self):
        csv = self.csv(n_rows=30, include_timestamp=False)
        with self.assertRaises(rfc.TrainerError) as ctx:
            rfc.load_dataset(csv)
        self.assertIn('timestamp', str(ctx.exception))

    def test_decreasing_timestamps_fail(self):
        path = os.path.join(self.tmp, 'dec.csv')
        write_csv(path, n_rows=40)
        df = pd.read_csv(path)
        df.loc[20, 'timestamp'] = iso(BASE_TS)                # jump backwards
        df.to_csv(path, index=False)
        with self.assertRaises(rfc.TrainerError) as ctx:
            rfc.load_dataset(path)
        self.assertIn('non-decreasing', str(ctx.exception))

    def test_epoch_second_and_ms_timestamps_accepted(self):
        d1 = rfc.load_dataset(self.csv(n_rows=20, ts_mode='epoch_s'))
        d2 = rfc.load_dataset(self.csv(n_rows=20, ts_mode='epoch_ms'))
        self.assertTrue(np.allclose(d1['ts'], d2['ts']))


class TestArtifactContract(TempDirs):
    def setUp(self):
        super().setUp()
        self.csv_path = self.csv(n_rows=100, ood_tail=3)
        self.out_dir = os.path.join(self.tmp, 'out')
        rc, _ = self.quiet_train(['--data', self.csv_path, '--out-dir', self.out_dir,
                                  '--seed', '11'])
        assert rc == 0, 'training run failed in setUp'
        self.card = load_json(os.path.join(self.out_dir, 'model_card.json'))
        self.env = load_json(os.path.join(self.out_dir, 'support_envelope.json'))

    def test_n_features_matches_3x_svcs(self):
        self.assertEqual(self.card['model_n_features_in'], 3 * len(SVCS))
        self.assertEqual(self.card['n_features'], 3 * len(SVCS))

    def test_svcs_roundtrip_file(self):
        dumped = load_json(os.path.join(self.out_dir, 'RandomForestClassify.svcs'))
        self.assertEqual(dumped, SVCS)
        self.assertEqual(self.env['svcs'], SVCS)

    def test_support_envelope_bounds(self):
        bounds = self.env['feature_bounds']
        self.assertEqual(len(bounds), 3 * len(SVCS))
        by_feat = {b['feature']: b for b in bounds}
        self.assertEqual(by_feat['svc-a#index']['min'], 0.0)
        self.assertEqual(by_feat['svc-c#index']['max'], float(len(SVCS) - 1))
        # qps envelope must exclude the OOD tail rows (train rows only)
        self.assertLess(by_feat['svc-a&qps']['max'], 10.0)
        self.assertGreater(by_feat['svc-a&qps']['min'], 0.0)
        self.assertEqual(by_feat['svc-a&count']['min'], 5.0)

    def test_ood_fraction_detected(self):
        # 3 of the last 20 test rows are OOD -> 0.15
        self.assertGreater(self.card['test_ood_fraction'], 0.0)
        self.assertAlmostEqual(self.card['test_ood_fraction'], 0.15, places=6)

    def test_expect_svcs_mismatch_hard_fails(self):
        rc, err = self.quiet_train(['--data', self.csv_path, '--out-dir', os.path.join(self.tmp, 'x'),
                                    '--seed', '11', '--expect-svcs', 'wrong-one,svc-b,svc-c'])
        self.assertEqual(rc, 1)
        self.assertIn('mismatch', err)

    def test_expect_svcs_match_passes(self):
        rc, _ = self.quiet_train(['--data', self.csv_path, '--out-dir', os.path.join(self.tmp, 'y'),
                                  '--seed', '11', '--expect-svcs', ','.join(SVCS)])
        self.assertEqual(rc, 0)

    def test_reproducible_with_same_seed(self):
        out2 = os.path.join(self.tmp, 'out2')
        rc, _ = self.quiet_train(['--data', self.csv_path, '--out-dir', out2, '--seed', '11'])
        self.assertEqual(rc, 0)
        card2 = load_json(os.path.join(out2, 'model_card.json'))
        self.assertEqual(self.card['metrics']['accuracy'], card2['metrics']['accuracy'])
        self.assertEqual(self.card['metrics']['roc_auc'], card2['metrics']['roc_auc'])
        env2 = load_json(os.path.join(out2, 'support_envelope.json'))
        self.assertEqual(self.env['class_balance'], env2['class_balance'])
        self.assertEqual(self.env['split']['ts_train_utc'], env2['split']['ts_train_utc'])

    def test_model_and_roc_files_written(self):
        for name in ('RandomForestClassify.model', 'rf.pkl',
                     'RandomForestClassify.svcs', 'support_envelope.json',
                     'model_card.json'):
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, name)), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
