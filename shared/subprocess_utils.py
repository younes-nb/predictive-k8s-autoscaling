import os
import time
import subprocess


def run(cmd, title: str, env: dict | None = None):
    print(f"\n=== {title} ===")
    print("Running:", " ".join(cmd))
    start = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, check=True, env=full_env)
    elapsed = time.time() - start
    print(f"=== {title} completed in {elapsed:.2f}s ===")
    return elapsed
