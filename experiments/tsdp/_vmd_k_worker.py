#!/usr/bin/env python3
"""
Worker: processes one service file.
Extracts windows (INPUT_LEN=60, STRIDE=5), SWT->D1, VMD K=1..max_k.
Outputs per-K per-mode metrics as JSON.
"""

import json
import os
import sys
import time

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis
import pywt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.tsdp.config import CFG


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def compute_center_frequency(mode, fs=1.0):
    if len(mode) < 2:
        return 0.0
    freqs, psd = sig.welch(mode, fs=fs, nperseg=min(256, len(mode)))
    return min(freqs[np.argmax(psd)], 0.5)


def vmd_decompose_safe(sig_arr, K, alpha, tau, DC, init, tol):
    from vmdpy import VMD
    sig_arr = np.asarray(sig_arr, dtype=np.float64)
    pad = len(sig_arr) % 2
    padded = np.append(sig_arr, sig_arr[-1]) if pad else sig_arr
    u, _, _ = VMD(padded, alpha, tau, K, DC, init, tol)
    if pad:
        u = u[:, :-1]
    return u[:, :len(sig_arr)]


def process_service(service_path, max_k, vmd_params):
    signal_data = np.load(service_path).astype(np.float64)
    n = len(signal_data)
    T = n - CFG.INPUT_LEN - CFG.PRED_HORIZON + 1
    if T <= 0:
        return None

    # Per-K accumulators: list of per-mode values across windows
    # For each K, store list of lists: [[cf_mode0, cf_mode1, ...], ...]
    per_k_cf = {K: [] for K in range(1, max_k + 1)}
    per_k_kt = {K: [] for K in range(1, max_k + 1)}
    per_k_cr = {K: [] for K in range(1, max_k + 1)}

    window_count = 0
    for i in range(0, T, CFG.STRIDE):
        window = signal_data[i: i + CFG.INPUT_LEN]
        if np.std(window) < 1e-12:
            continue

        try:
            swt_coeffs = pywt.swt(window, 'sym4', level=1, norm=True, trim_approx=True)
            _, d1 = swt_coeffs
        except Exception:
            continue

        if np.std(d1) < 1e-12:
            continue

        for K in range(1, max_k + 1):
            try:
                modes = vmd_decompose_safe(
                    d1, K,
                    vmd_params['alpha'], vmd_params['tau'],
                    vmd_params['DC'], vmd_params['init'], vmd_params['tol'],
                )
            except Exception:
                continue

            cf_list = []
            kt_list = []
            cr_list = []
            for mi in range(min(K, modes.shape[0])):
                mode = modes[mi, :]
                cf_list.append(compute_center_frequency(mode))
                kt_list.append(float(kurtosis(mode, fisher=False)))
                if np.std(mode) > 1e-12 and np.std(d1) > 1e-12:
                    cr_list.append(float(np.corrcoef(mode, d1)[0, 1]))
                else:
                    cr_list.append(0.0)

            per_k_cf[K].append(cf_list)
            per_k_kt[K].append(kt_list)
            per_k_cr[K].append(cr_list)

        window_count += 1

    if window_count == 0:
        return None

    # Aggregate: mean across windows for each mode
    result = {}
    for K in range(1, max_k + 1):
        if not per_k_cf[K]:
            continue

        # Pad shorter mode lists to same length
        max_modes = max(len(s) for s in per_k_cf[K])
        cf_padded = [s + [np.nan] * (max_modes - len(s)) for s in per_k_cf[K]]
        kt_padded = [s + [np.nan] * (max_modes - len(s)) for s in per_k_kt[K]]
        cr_padded = [s + [np.nan] * (max_modes - len(s)) for s in per_k_cr[K]]

        cf_arr = np.array(cf_padded)
        kt_arr = np.array(kt_padded)
        cr_arr = np.array(cr_padded)

        per_mode = []
        for mi in range(max_modes):
            cf_vals = cf_arr[:, mi]
            kt_vals = kt_arr[:, mi]
            cr_vals = cr_arr[:, mi]
            cf_vals = cf_vals[~np.isnan(cf_vals)]
            kt_vals = kt_vals[~np.isnan(kt_vals)]
            cr_vals = cr_vals[~np.isnan(cr_vals)]
            per_mode.append({
                'cf': float(np.mean(cf_vals)) if len(cf_vals) > 0 else 0.0,
                'kt': float(np.mean(kt_vals)) if len(kt_vals) > 0 else 0.0,
                'cr': float(np.mean(cr_vals)) if len(cr_vals) > 0 else 0.0,
            })

        result[f'K{K}'] = {'per_mode': per_mode}

    result['_windows'] = window_count
    return result


def main():
    if len(sys.argv) < 3:
        print("RESULT:False:ERROR: usage: worker.py <service_path> <max_k> [json_out]",
              file=sys.stdout)
        sys.exit(1)

    service_path = sys.argv[1]
    max_k = int(sys.argv[2])
    json_out = sys.argv[3] if len(sys.argv) > 3 else None

    vmd_params = {
        'alpha': CFG.VMD_ALPHA, 'tau': CFG.VMD_TAU,
        'DC': CFG.VMD_DC, 'init': CFG.VMD_INIT, 'tol': CFG.VMD_TOL,
    }

    t0 = time.time()
    result = process_service(service_path, max_k, vmd_params)
    elapsed = time.time() - t0

    if result is None:
        print("RESULT:True:no_data")
        sys.exit(0)

    if json_out:
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, 'w') as f:
            json.dump(result, f)
        print(f"RESULT:True:{result['_windows']} windows in {elapsed:.1f}s")
    else:
        print(f"RESULT:True:{json.dumps(result)}")

    sys.exit(0)


if __name__ == "__main__":
    main()
