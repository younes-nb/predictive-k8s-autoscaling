
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cvcbm.config import CFG
from experiments.cvcbm.decomposition import decompose_service_signal
import numpy as np

def main() -> None:
    ms_name = sys.argv[1]
    idx = int(sys.argv[2])
    out_dir = sys.argv[3]

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

        if len(signal) < CFG.MIN_SIGNAL_LEN:
            print(f"RESULT:True:too short ({len(signal)})")
            sys.exit(0)

        co_imfs = decompose_service_signal(signal.astype(np.float64), CFG)

        rec = np.sum([np.asarray(c) for c in co_imfs], axis=0)
        rec_mae = float(np.mean(np.abs(rec.astype(np.float32) - signal)))

        for k, co_imf in enumerate(co_imfs):
            imf_path = os.path.join(out_dir, f"co_imf_{k}", f"service_{idx:05d}.npy")
            for attempt in range(3):
                try:
                    np.save(imf_path, np.asarray(co_imf, dtype=np.float32))
                    break
                except OSError as e:
                    if attempt == 2 or "Read-only" not in str(e):
                        raise
                    import time
                    time.sleep(5 * (attempt + 1))

        with open(os.path.join(out_dir, f"service_{idx:05d}.meta.txt"), "w") as f:
            f.write(f"{ms_name}\n")

        open(done_marker, "a").close()
        print(f"RESULT:True:ok (MAE={rec_mae:.8f})")
        sys.exit(0)

    except Exception as exc:
        print(f"RESULT:False:ERROR: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
