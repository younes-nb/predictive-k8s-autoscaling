import os
import sys
import time as _time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocessing.sv.config import CFG
from preprocessing.sv.decomposition import decompose_window
from shared.config_preprocessing_defaults import PREPROCESSING


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


CHANNEL_DIRS = [f"vmd_mode_{k}" for k in range(10)] + ["D2", "A2"]
MEM_CHANNEL_DIRS = [f"mem_vmd_mode_{k}" for k in range(10)] + ["mem_D2", "mem_A2"]
N_CHANNELS = len(CHANNEL_DIRS)


def _decompose_and_save(
    signal: np.ndarray, channel_dirs: list[str], out_dir: str, idx: int,
    prefix: str, ms_name: str,
) -> None:
    T = len(signal) - PREPROCESSING.INPUT_LEN - PREPROCESSING.PRED_HORIZON + 1
    if T <= 0:
        _log(f"[{ms_name}/{prefix}] too short ({len(signal)} steps)")
        return

    n_windows = len(range(0, T, PREPROCESSING.STRIDE))
    _log(f"[{ms_name}/{prefix}] {len(signal)} steps, {n_windows} windows")

    accum = [[] for _ in range(N_CHANNELS)]
    _last_pct = 0
    _t0 = _time.time()
    for wi, i in enumerate(range(0, T, PREPROCESSING.STRIDE)):
        pct = (wi + 1) * 100 // n_windows
        if pct >= _last_pct + 10:
            _last_pct = pct - (pct % 10)
            _log(f"[{ms_name}/{prefix}] {_last_pct}% ({wi+1}/{n_windows}, {_time.time()-_t0:.0f}s)")

        window = signal[i: i + PREPROCESSING.INPUT_LEN]
        channels = decompose_window(window, CFG)
        for c in range(N_CHANNELS):
            accum[c].append(channels[c])

    n_windows = len(accum[0])
    for c in range(N_CHANNELS):
        arr = np.stack(accum[c], axis=0)
        ch_dir = os.path.join(out_dir, channel_dirs[c])
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

    _elapsed = _time.time() - _t0
    _log(f"[{ms_name}/{prefix}] 100% ({n_windows}/{n_windows}, {_elapsed:.0f}s)")


def main() -> None:
    ms_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]
    feature_set = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    has_mem = feature_set == "cpu_mem_both"
    _t0 = _time.time()

    all_dirs = CHANNEL_DIRS + (MEM_CHANNEL_DIRS if has_mem else [])
    done_marker = os.path.join(out_dir, CHANNEL_DIRS[0], f"service_{idx:05d}.done")
    if os.path.exists(done_marker):
        all_exist = all(
            os.path.exists(os.path.join(out_dir, d, f"service_{idx:05d}.npy"))
            for d in all_dirs
        )
        if all_exist:
            print("RESULT:True:already done")
            sys.exit(0)
        else:
            os.remove(done_marker)

    try:
        cpu_path = os.path.join(out_dir, "original", f"service_{idx:05d}.npy")
        if not os.path.exists(cpu_path):
            print(f"RESULT:False:ERROR: CPU signal not found at {cpu_path}")
            sys.exit(1)
        cpu_signal = np.load(cpu_path).astype(np.float32)

        cpu_T = len(cpu_signal) - PREPROCESSING.INPUT_LEN - PREPROCESSING.PRED_HORIZON + 1
        if cpu_T <= 0:
            print(f"RESULT:True:too short ({len(cpu_signal)})")
            sys.exit(0)

        _decompose_and_save(cpu_signal, CHANNEL_DIRS, out_dir, idx, "cpu", ms_name)

        if has_mem:
            mem_path = os.path.join(out_dir, "mem_original", f"service_{idx:05d}.npy")
            if not os.path.exists(mem_path):
                print(f"RESULT:False:ERROR: memory signal not found at {mem_path}")
                sys.exit(1)
            mem_signal = np.load(mem_path).astype(np.float32)
            mem_T = len(mem_signal) - PREPROCESSING.INPUT_LEN - PREPROCESSING.PRED_HORIZON + 1
            if mem_T <= 0:
                _log(f"[{ms_name}] memory signal too short ({len(mem_signal)}), writing zeros")
                mem_signal = np.zeros_like(cpu_signal)
            _decompose_and_save(mem_signal, MEM_CHANNEL_DIRS, out_dir, idx, "mem", ms_name)

        open(done_marker, "a").close()
        _elapsed = _time.time() - _t0
        _log(f"[{ms_name}] all done ({_elapsed:.0f}s)")
        print(f"RESULT:True:ok")
        sys.exit(0)

    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
