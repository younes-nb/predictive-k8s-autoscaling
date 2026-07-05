import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.sdtnet.config import CFG
from experiments.sdtnet.decomposition import decompose_window


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


CHANNEL_DIRS = [f"mode_{i}" for i in range(CFG.SVMD2_MAX_MODES)] + ["lowfreq_0"]
N_CHANNELS = len(CHANNEL_DIRS)


def main() -> None:
    ms_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    _t0 = _time.time()

    done_marker = os.path.join(out_dir, "mode_0", f"service_{idx:05d}.done")
    if os.path.exists(done_marker):
        all_exist = all(
            os.path.exists(os.path.join(out_dir, d, f"service_{idx:05d}.npy"))
            for d in CHANNEL_DIRS
        )
        if all_exist:
            print("RESULT:True:already done")
            sys.exit(0)
        else:
            os.remove(done_marker)

    try:
        signal = np.load(
            os.path.join(out_dir, "original", f"service_{idx:05d}.npy")
        ).astype(np.float32)

        T = len(signal) - CFG.INPUT_LEN - CFG.PRED_HORIZON + 1
        if T <= 0:
            print(f"RESULT:True:too short ({len(signal)})")
            sys.exit(0)

        n_windows = len(range(0, T, CFG.STRIDE))
        _log(f"[{ms_name}] {len(signal)} steps, {n_windows} windows, "
             f"INPUT_LEN={CFG.INPUT_LEN}, STRIDE={CFG.STRIDE}")

        accum = [[] for _ in range(N_CHANNELS)]
        _last_pct = 0
        for wi, i in enumerate(range(0, T, CFG.STRIDE)):
            pct = (wi + 1) * 100 // n_windows
            if pct >= _last_pct + 10:
                _last_pct = pct - (pct % 10)
                _log(f"[{ms_name}] {_last_pct}% ({wi+1}/{n_windows}, {_time.time()-_t0:.0f}s)")

            window = signal[i: i + CFG.INPUT_LEN]
            channels = decompose_window(window, CFG)
            for c in range(N_CHANNELS):
                accum[c].append(channels[c])

        n_windows = len(accum[0])
        for c in range(N_CHANNELS):
            arr = np.stack(accum[c], axis=0)
            ch_dir = os.path.join(out_dir, CHANNEL_DIRS[c])
            os.makedirs(ch_dir, exist_ok=True)
            fpath = os.path.join(ch_dir, f"service_{idx:05d}.npy")
            for attempt in range(3):
                try:
                    np.save(fpath, arr)
                    break
                except OSError as e:
                    if attempt == 2 or "Read-only" not in str(e):
                        raise
                    _time.sleep(5 * (attempt + 1))

        open(done_marker, "a").close()
        _elapsed = _time.time() - _t0
        _log(f"[{ms_name}] 100% ({n_windows}/{n_windows}, {_elapsed:.0f}s)")
        print(f"RESULT:True:ok")
        sys.exit(0)

    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
