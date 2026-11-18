"""Compare BPE strategies in separate processes on the same training data.

Run from the assignment root: uv run python -m tests.benchmark_bpe
"""

import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from tests.adapters import run_train_bpe

STRATEGIES = ("baseline", "rebuild_index", "incremental", "heap")
INPUT = Path(__file__).parent / "fixtures" / "tinystories_sample_5M.txt"


def run_one(strategy):
    start = time.perf_counter()
    vocab, merges = run_train_bpe(INPUT, 10_000, ["<|endoftext|>"], strategy=strategy)
    elapsed = time.perf_counter() - start
    # On macOS ru_maxrss is bytes; Linux reports KiB.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    canonical = json.dumps(
        [sorted((key, value.hex()) for key, value in vocab.items()),
         [(left.hex(), right.hex()) for left, right in merges]],
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    print(json.dumps({"strategy": strategy, "seconds": elapsed,
                      "peak_mib": peak_mib, "digest": digest}))


def main():
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
        return
    results = []
    for strategy in STRATEGIES:
        print(f"Running {strategy}...", flush=True)
        output = subprocess.check_output([sys.executable, "-m", "tests.benchmark_bpe", strategy], text=True)
        results.append(json.loads(output))
    print("\nStrategy          Time (s)    Peak process memory (MiB)")
    for result in results:
        print(f"{result['strategy']:<17} {result['seconds']:>8.2f} {result['peak_mib']:>20.1f}")
    print("Identical vocabulary and merges:", len({r['digest'] for r in results}) == 1)


if __name__ == "__main__":
    main()
