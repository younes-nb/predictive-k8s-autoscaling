
import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cvcbm.config import CFG, set_seed
from experiments.cvcbm.decomposition import decompose_service_signal

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

MAX_IMFS = 15

def main() -> None:
    set_seed(CFG.SEED)
    ms_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    no_clustering = len(sys.argv) > 4 and sys.argv[4] == "1"
    _t0 = _time.time()

    dir_prefix = "raw_imf" if no_clustering else "co_imf"
    done_marker = os.path.join(out_dir, f"service_{idx:05d}.done")
    if os.path.exists(done_marker):
        if no_clustering:
            files_exist = os.path.exists(os.path.join(out_dir, "raw_imf_0", f"service_{idx:05d}.npy"))
        else:
            files_exist = all(
                os.path.exists(os.path.join(out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy"))
                for k in range(CFG.N_CLUSTERS)
            )
        if files_exist:
            print(f"RESULT:True:already done")
            sys.exit(0)
        else:
            os.remove(done_marker)

    try:
        signal = np.load(
            os.path.join(out_dir, "original", f"service_{idx:05d}.npy")
        ).astype(np.float32)

        # Split signal into sliding windows of INPUT_LEN; each window is decomposed independently.
        # T = number of valid starting positions for which a target (next value) also exists.
        T = len(signal) - CFG.INPUT_LEN - CFG.PRED_HORIZON + 1
        if T <= 0:
            print(f"RESULT:True:too short ({len(signal)})")
            sys.exit(0)

        n_windows = len(range(0, T, CFG.STRIDE))
        _log(f"[{ms_name}] {len(signal)} steps, {n_windows} windows, "
             f"INPUT_LEN={CFG.INPUT_LEN}, STRIDE={CFG.STRIDE}")

        window_imfs = [[] for _ in range(20)]
        window_vmd_modes = [[] for _ in range(CFG.VMD_K)]
        windows = []
        _last_pct = 0
        for wi, i in enumerate(range(0, T, CFG.STRIDE)):
            pct = (wi + 1) * 100 // n_windows
            if pct >= _last_pct + 10:
                _last_pct = pct - (pct % 10)
                _log(f"[{ms_name}] {_last_pct}% ({wi+1}/{n_windows}, {_time.time()-_t0:.0f}s)")

            window = signal[i : i + CFG.INPUT_LEN]
            windows.append(window)
            decomposed = decompose_service_signal(window.astype(np.float64), CFG, return_raw_imfs=no_clustering)
            if no_clustering:
                for k in range(len(decomposed)):
                    window_imfs[k].append(np.asarray(decomposed[k], dtype=np.float32))
            else:
                co_imfs, vmd_modes = decomposed
                if vmd_modes is not None:
                    for mode_k in range(vmd_modes.shape[0]):
                        window_vmd_modes[mode_k].append(vmd_modes[mode_k].astype(np.float32))
                for k in range(CFG.N_CLUSTERS):
                    window_imfs[k].append(np.asarray(co_imfs[k], dtype=np.float32))

        n_windows = len(windows)
        n_actual = MAX_IMFS if no_clustering else CFG.N_CLUSTERS

        # Pad IMF slots to have exactly n_windows entries each
        for k in range(n_actual):
            n_present = len(window_imfs[k])
            if n_present < n_windows:
                pad_shape = (CFG.INPUT_LEN,) if n_present > 0 else (CFG.INPUT_LEN,)
                template = window_imfs[k][0] if n_present > 0 else np.zeros(pad_shape, dtype=np.float32)
                pad_cnt = n_windows - n_present
                window_imfs[k].extend([np.zeros_like(template) for _ in range(pad_cnt)])
        windows_arr = np.stack(windows, axis=0)
        stacked = []

        if not no_clustering:
            has_vmd_modes = any(len(lst) > 0 for lst in window_vmd_modes)
            if has_vmd_modes:
                for mode_k in range(CFG.VMD_K):
                    if len(window_vmd_modes[mode_k]) > 0:
                        vmd_arr = np.stack(window_vmd_modes[mode_k], axis=0)
                    else:
                        vmd_arr = np.zeros((n_windows, CFG.INPUT_LEN), dtype=np.float32)
                    vmd_path = os.path.join(out_dir, f"co_imf_0", f"vmd_mode_{mode_k}_service_{idx:05d}.npy")
                    for attempt in range(3):
                        try:
                            np.save(vmd_path, vmd_arr)
                            break
                        except OSError as e:
                            if attempt == 2 or "Read-only" not in str(e):
                                raise
                            _time.sleep(5 * (attempt + 1))
            n_save = CFG.N_CLUSTERS
        else:
            n_save = n_actual

        for k in range(n_save):
            if len(window_imfs[k]) > 0:
                arr = np.stack(window_imfs[k], axis=0)
            else:
                arr = np.zeros((n_windows, CFG.INPUT_LEN), dtype=np.float32)
            stacked.append(arr)
            imf_path = os.path.join(out_dir, f"{dir_prefix}_{k}", f"service_{idx:05d}.npy")
            for attempt in range(3):
                try:
                    np.save(imf_path, arr)
                    break
                except OSError as e:
                    if attempt == 2 or "Read-only" not in str(e):
                        raise
                    _time.sleep(5 * (attempt + 1))

        total_rec = sum(stacked)
        rec_mae = float(np.mean(np.abs(total_rec - windows_arr)))

        with open(os.path.join(out_dir, f"service_{idx:05d}.meta.txt"), "w") as f:
            f.write(f"{ms_name}\n")

        open(done_marker, "a").close()
        _elapsed = _time.time() - _t0
        _log(f"[{ms_name}] 100% ({n_windows}/{n_windows}, {_elapsed:.0f}s)")
        print(f"RESULT:True:ok (MAE={rec_mae:.8f})")
        sys.exit(0)

    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
