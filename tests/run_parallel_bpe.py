"""Run the full TinyStories experiment using eight pre-tokenization processes.

Run from the assignment root: uv run python -m tests.run_parallel_bpe
"""

import json
import resource
import time
from pathlib import Path


def main():
    # Keep this import inside main so worker processes do not import PyTorch.
    from tests.adapters import run_train_bpe

    print("Training with eight pre-tokenization processes...", flush=True)
    start = time.perf_counter()
    vocab, merges = run_train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt", 10_000, ["<|endoftext|>"],
        strategy="heap", num_processes=8,
    )
    elapsed = time.perf_counter() - start
    parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    output = Path("data/tinystories_bpe_heap_parallel.json")
    with output.open("w", encoding="utf-8") as f:
        json.dump({"vocab": {str(k): v.hex() for k, v in vocab.items()},
                   "merges": [[a.hex(), b.hex()] for a, b in merges],
                   "encoding": "hex"}, f)
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Parent process peak memory: {parent_peak:.1f} MiB (excludes workers)")
    print(f"Vocabulary size: {len(vocab)}; merges: {len(merges)}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
