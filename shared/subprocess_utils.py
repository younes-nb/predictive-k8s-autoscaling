import time
import subprocess


def run(cmd, title: str):
    print(f"\n=== {title} ===")
    print("Running:", " ".join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"=== {title} completed in {elapsed:.2f}s ===")
    return elapsed
