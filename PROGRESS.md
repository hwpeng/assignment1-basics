# CS336 Assignment 1: Progress

Personal study notes and implementation progress. For the full learning review and step-by-step optimization history, see [BPE_REVIEW.md](BPE_REVIEW.md). Based on the [Stanford CS336 assignment1-basics repository](https://github.com/stanford-cs336/assignment1-basics). See the original [README](README.md) and [handout](cs336_assignment1_basics.pdf) for setup and requirements.

## Progress

- [x] Watch Lecture 2.
- [x] Implement BPE training and pass the three supplied BPE training tests.
- [x] Compare four merge strategies on the 5 MiB TinyStories sample.
- [x] Train BPE on the full TinyStories training dataset, save the vocabulary and merges, and compare single-process, eight-process, and sixteen-process pre-tokenization.
- [x] Profile the final parallel training run and record the bottleneck.
- [x] Section 2.6: implement encoding, decoding, and iterable encoding; 23 provided functional tests passed. Two memory tests skipped on macOS; arbitrary streaming boundaries remain unverified.
- [x] Section 2.7: run small TinyStories-tokenizer compression experiments on both samples and measure throughput.
- [ ] Encode training and validation data into `uint16` arrays for later model training.
- [ ] Finish the relevant lecture, then start Section 3.
- [ ] OpenWebText BPE experiment (deferred for now).

## TinyStories experiment

Training uses the 2,227,753,162-byte TinyStories training file, a maximum vocabulary size of 10,000, and `<|endoftext|>` as the special token. The full dataset is a separate download and is not included here.

| Method | Training wall time | Peak memory |
| --- | ---: | --- |
| Heap merges, single-process pre-tokenization | 120.35 s | 3,834 MiB, whole process |
| Heap merges, eight-process pre-tokenization | 29.03 s | 213.6 MiB, parent only; workers excluded |
| Heap merges, sixteen-process pre-tokenization | 25.10 s | 217.6 MiB, parent only; workers excluded |

All three full runs produced identical vocabularies and ordered merges. The longest token was `b' accomplishment'` (15 bytes); a word with a leading space is plausible given the pre-tokenization pattern. The memory numbers are not directly comparable. These are single runs on one machine with 10 reported logical CPUs; 16 processes share those CPUs.

The saved outputs are `data/tinystories_bpe_heap.json` and `data/tinystories_bpe_heap_parallel.json`. Their byte values are encoded as hexadecimal strings. To rerun the parallel experiment, from the repository root:

```sh
uv run python -m tests.run_parallel_bpe     # eight processes
uv run python -m tests.run_parallel_bpe 16  # sixteen processes
```

## Profiling the parallel trainer

In a separate full-data run with `cProfile` enabled for the parent and all eight workers, the full run took 47.99 s and the parallel pre-tokenization stage took 45.22 s wall time, including process setup, file reading, counting, and result collection. Profiling adds overhead, so these times should not be compared directly with the unprofiled 29.03 s run. Worker profiles showed that pre-token matching and counting dominated their work. Summed worker times overlap in parallel and are not wall time. Thus, even after parallelization, pre-tokenization remains the largest measured stage in this profile. Every occurrence in the input must be read and counted, while the merge stage works on distinct pre-tokens and their frequencies; repeated pre-tokens are represented once.

Reproduce with `uv run python -m tests.profile_parallel_bpe`. The script writes the raw profiles to `/tmp`.

## Merge strategies on the 5 MiB sample

| Strategy | Time | Peak process memory |
| --- | ---: | ---: |
| Baseline | 28.39 s | 221.9 MiB |
| Rebuild index each round | 23.54 s | 224.4 MiB |
| Maintain pair counts and index | 2.53 s | 228.4 MiB |
| Maintain counts and index, use a heap | 0.44 s | 232.5 MiB |

All four produced identical vocabularies and merges. These are single unprofiled runs; peak memory includes the whole Python process. Repeat with `uv run python -m tests.benchmark_bpe`.

## Code map

- `tests/adapters.py`: trainer, original baseline and rebuilt-index implementations, and strategy selection.
- `tests/bpe_strategies.py`: incremental and heap merge strategies.
- `tests/bpe_parallel.py`: safe chunk boundaries and parallel pre-token counting for the TinyStories marker.
- `tests/benchmark_bpe.py`: compares strategies on the small sample.
- `tests/run_parallel_bpe.py`: runs the full TinyStories parallel experiment.
- `tests/profile_parallel_bpe.py`: profiles the parent and worker processes.

Run the existing checks with `uv run pytest tests/test_train_bpe.py -q`. The parallel pre-token counts, vocabulary and merges were also checked against the single-process output on the sample and on the full training file.

## Encoding experiments and optimizations

The corrected TinyStories tokenizer produced 4.10 bytes/token on 10 TinyStories documents (separately encoded, 7,844 bytes / 1,912 IDs) and 3.28 bytes/token on the first 10 OpenWebText documents (48,275 bytes / 14,707 IDs). This is a small sample, not a representative dataset-wide comparison. Our initial encoder ignored merge priority; its old compression measurements should not be used.

| Encoder change | Workload | Result |
| --- | --- | --- |
| Scan all merge rules in priority order | 10 OpenWebText docs repeated 24 times | ~43,800 bytes/s average over 3 runs |
| Look up adjacent-pair priorities | Same workload | ~1,276,000 bytes/s average over 3 runs; same ID count |
| Cache pre-token merge results | Continuous ~20 MB TinyStories | 14.44 s → 2.87 s; process peaks 1,039.7 → 680.5 MiB, single runs |
| Parallel pre-tokenization with cache | Same ~20 MB TinyStories | 1 process 6.86 MB/s; 8 processes 11.51 MB/s; 16 processes 11.96 MB/s, identical output hashes |

Memory readings for the cache experiment are whole-process peaks, and worker memory for the parallel encoder was not measured. See [BPE_REVIEW.md](BPE_REVIEW.md) for the reasoning, profiles, caveats, and the full trainer optimization history.

- `cs336_basics/tokenizer.py`: encoder, decoder, iterable interface, caching, and optional parallel pre-tokenization.
- Run `uv run pytest tests/test_tokenizer.py -q` for the supplied tokenizer tests.
