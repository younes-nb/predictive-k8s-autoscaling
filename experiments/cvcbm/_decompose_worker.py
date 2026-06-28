
import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cvcbm.config import CFG
from experiments.cvcbm.decomposition import decompose_service_signal

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def main() -> None:
    ms_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    _t0 = _time.time()

    done_marker = os.path.join(out_dir, f"service_{idx:05d}.done")
    if os.path.exists(done_marker):
        co_imf_files_exist = all(
            os.path.exists(os.path.join(out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy"))
            for k in range(CFG.N_CLUSTERS)
        )
        if co_imf_files_exist:
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

        window_imfs = [[] for _ in range(CFG.N_CLUSTERS)]
        windows = []
        _last_pct = 0
        for wi, i in enumerate(range(0, T, CFG.STRIDE)):
            pct = (wi + 1) * 100 // n_windows
            if pct >= _last_pct + 10:
                _last_pct = pct - (pct % 10)
                _log(f"[{ms_name}] {_last_pct}% ({wi+1}/{n_windows}, {_time.time()-_t0:.0f}s)")

            window = signal[i : i + CFG.INPUT_LEN]
            windows.append(window)
            co_imfs = decompose_service_signal(window.astype(np.float64), CFG)
            for k in range(CFG.N_CLUSTERS):
                window_imfs[k].append(np.asarray(co_imfs[k], dtype=np.float32))

        # Save each Co-IMF as a single (num_windows, INPUT_LEN) array per service.
        windows_arr = np.stack(windows, axis=0)
        stacked = []
        for k in range(CFG.N_CLUSTERS):
            arr = np.stack(window_imfs[k], axis=0)
            stacked.append(arr)
            imf_path = os.path.join(out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy")
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
