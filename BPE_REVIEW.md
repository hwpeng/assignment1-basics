# Section 2: BPE review and learning notes

These notes capture the concepts, questions, corrections, and measurements from our BPE trainer and tokenizer work. They are a review aid, not a claim that every Section 2 deliverable is finished. See [PROGRESS.md](PROGRESS.md) for current status.

## 1. What are we building?

A byte-level BPE **trainer** learns a vocabulary and an ordered list of merge rules from a text corpus. An **encoder** uses those learned rules to turn new text into token IDs. A **decoder** turns token IDs back into text. Training and encoding are different workloads, so their bottlenecks can differ.

A vocabulary maps integer IDs to token bytes. It starts with 256 one-byte tokens, one for each possible byte value. The trainer adds special tokens and then tokens formed by merges, up to the requested maximum vocabulary size. If there are no adjacent pairs left, it can stop early.

## 2. Bytes, UTF-8, and vocabulary

A byte is a value from 0 to 255. UTF-8 maps a Unicode character to one or more bytes; those 256 byte values are not a list of the 256 most common characters. For example, `bytes([65])` creates the single byte with value 65, while `bytes(65)` creates 65 zero bytes. A single byte need not represent a complete character.

A pre-token such as `"abab"` can initially be represented as `(b"a", b"b", b"a", b"b")`. As merging proceeds, tuple elements can become multi-byte tokens such as `b"ab"`. Tuples are useful as dictionary keys because they are immutable.

Special tokens such as `<|endoftext|>` receive a dedicated vocabulary ID. In training they mark hard boundaries: do not count pairs across them, and do not include their contents in merge statistics. In encoding, preserve them as single tokens at their original positions. There may be multiple kinds of special token, not just an end marker.

## 3. Pre-tokenization and counting

Pre-tokenization uses the assignment's regex to divide ordinary text into word-like pieces, including spaces, punctuation, and numbers. BPE could be defined without it, but this assignment restricts merges to within pre-tokens. The regex affects which byte sequences can ever become tokens.

The trainer splits around special tokens, pre-tokenizes each segment, and aggregates counts across **all** segments. We initially considered training each segment separately; the correction is that the full corpus supplies global pair frequencies and one shared merge sequence. `regex.finditer` produces matches one at a time, and `match.group()` extracts the matched text. `"|".join(re.escape(token) for token in special_tokens)` builds a regex alternative from literal special-token strings.

A pre-token frequency dictionary stores repeated text once with a count. For example, if `abab` occurs 3 times, `(a, b)` occurs twice in that pre-token and contributes 6 to its global pair count; `(b, a)` contributes 3. Pair counting includes overlapping adjacent positions. Replacement is different: it scans left to right and consumes positions once, so merging `(a, a)` in `(a, a, a)` yields `(aa, a)`.

At each training step, choose the globally most frequent pair. If counts tie, choose the lexicographically greatest pair, as required by the handout. Save the original pair in the ordered `merges` list and assign the concatenated bytes a new vocabulary ID.

## 4. Trainer optimization: step by step

Measurements below are experiments on this machine, not universal speed claims. Profiling changes timings. The 5 MiB measurements used vocabulary size 10,000 and one special token; each was a single unprofiled run. All four versions produced identical vocabulary and merges for that workload.

1. **Baseline: recount and scan everything.** Every iteration rebuilds pair frequencies from all unique pre-token sequences and scans the sequences again to apply the chosen merge. Approximate repeated work is `O(M × U × L)` for `M` merge rounds, `U` unique sequences, and average length `L`, ignoring hashing and changing lengths. Measured **28.39 s**, peak process memory **221.9 MiB**.
2. **Rebuild a pair-to-sequence index each round.** Record which sequences contain each pair; after selecting a pair, only merge its affected sequences. But rebuilding both counts and index still repeats much work. Measured **23.54 s**, **224.4 MiB**.
3. **Keep counts and the index between rounds.** User insight: most sequences don't change during a merge. Build `pair_counts` and `pair_to_seq` once, then update statistics and memberships for affected sequences. Counts must reflect all adjacent occurrences weighted by pre-token frequency; an index set needs only one membership per sequence. Measured **2.53 s**, **228.4 MiB**.
4. **Use a heap to find the highest count.** Persistent counts still required scanning all pairs with `max` each round. A heap helps find the maximum; entries whose stored counts no longer match the dictionary are discarded as stale. Preserve lexicographic tie-breaking. Measured **0.44 s**, **232.5 MiB**.

We discussed storing every pair's positions within each sequence. Merges shift indices and overlapping pairs require care; because pre-token sequences are short, we chose to index affected sequences and rescan those sequences rather than keep all positions. We also compared multiprocessing with indexing: they optimize different work, so they can be combined.

### Full TinyStories training and multiprocessing

The training file was 2,227,753,162 bytes. With 10,000 vocabulary entries and `<|endoftext|>`, the heap trainer produced 9,743 merges. Longest learned token: `b" accomplishment"` (15 bytes), plausible as a frequent word with a leading space.

| Pre-tokenization | Wall time | Peak process memory |
| --- | ---: | --- |
| Single process | 120.35 s | 3,834 MiB (whole process) |
| 8 processes | 29.03 s | 213.6 MiB (parent only; excludes workers) |
| 16 processes | 25.10 s | 217.6 MiB (parent only; excludes workers) |

The three runs produced identical vocabulary and merges. The parent-only memory figures are **not** comparable to the single-process peak. Mac reported 10 logical CPUs; more processes than CPUs can run, but they share resources and may not be faster.

The chunking insight is to choose approximate positions and move to a special-token boundary. Independent workers pre-tokenize and count their chunks; their counts are summed before global BPE merging. The file still has to be read and processed. In a profiled 8-process run, the pre-tokenization stage took 45.22 s of a 47.99 s profiled wall time, including worker setup and collection. Worker matching and counting dominated worker work. Profiling adds overhead, and summed worker times overlap; do not add them to obtain wall time.

## 5. Encoding and decoding: key correction

The decoder uses IDs to look up byte tokens, concatenates the bytes, and decodes UTF-8; invalid sequences are replaced with `�` as required by the handout. The encoder needs the vocabulary, special tokens, and **ordered merge rules**. The input text is supplied when calling `encode`.

A merge rule `(left, right)` replaces adjacent tokens with `left + right`. The vocabulary alone cannot tell us which pair formed a token or resolve competing merges. For example, if the rules prioritize `(b, c)` over `(a, b)`, encoding `a b c` starts by merging `b c`, even if `ab` also exists in the vocabulary. Smaller rule-list index means higher priority.

Our first encoder merged any matching pair from left to right. This was a mistake: it ignored rule priority. Round-trip checks still passed because a decoder can reconstruct the original text from several different sequences of valid IDs. Reference-tokenizer comparisons exposed the issue. The corrected logic finds the highest-priority applicable pair, merges non-overlapping occurrences, and repeats until no rule applies within the pre-token.

We also initially removed special tokens during splitting and failed to put them back. Unlike training, encoding must preserve their position and identity. We then returned both segments and the special tokens between them. For overlapping markers, try longer alternatives first: regex alternatives at the same input position are tried in pattern order.

`from_files` currently reads our saved JSON representation, where bytes are encoded as hexadecimal strings. Our vocabulary and merges are together in one JSON file, so pass that path for both arguments. The supplied tests load their GPT-2 reference files separately and pass the resulting Python objects through `get_tokenizer`; they do not directly validate our `from_files` format.

## 6. Encoder optimization: step by step

1. **Initial scan and correction.** The original left-to-right merge implementation was invalid because it ignored rule order. Its early compression and speed numbers are not reliable BPE results. After correcting merge order, the 10-document samples gave **4.10 bytes/token** for TinyStories (7,844 bytes, 1,912 IDs, documents encoded separately without boundary markers) and **3.28 bytes/token** for OpenWebText (48,275 bytes, 14,707 IDs). The OpenWebText sample is the first ten documents, not a representative random sample. Both round-tripped, but round-trip alone is not enough to prove correct BPE.
2. **Scan the rules in priority order.** Build the pre-token's adjacent-pair set, scan the merge list for the first matching rule, apply it, and repeat. This follows priority but repeatedly scans the long rule list. On the OpenWebText sample repeated 24 times, three runs averaged about **43,800 bytes/s**. Profiling one pass showed the rule-search expression dominating `merge_segments`.
3. **Pair-to-priority dictionary.** User insight: a pre-token has far fewer adjacent pairs than the ~9,700 rules. Build a dictionary once, look up only the pairs present, and select the smallest priority. The same repeated workload averaged **1,276,000 bytes/s**, about **29×** the preceding result; each run produced the same 352,968 IDs. The supplied suite still passed its functional tests. These after-runs were under a second each, so treat the factor as an estimate.
4. **Cache merged pre-token results.** The same pre-token often appears many times. Cache its merged byte-token sequence so a repeated occurrence doesn't repeat merge search. This doesn't add entries to the tokenizer vocabulary. The cache accepts up to 100,000 distinct inputs; once full, new entries are not added. On one continuous ~20 MB TinyStories workload, baseline **14.44 s / 1,039.7 MiB** versus cached **2.87 s / 680.5 MiB**, with the same ID count. These single-run process peaks include the whole process; identical counts alone are not proof of identical outputs. The lower memory is an observation, not a fully isolated causal measurement. Functional tests passed.
5. **Parallel pre-tokenization in encoding.** Preserve order: split around special tokens, group consecutive complete segments into roughly equal chunks, and send those chunks to workers. Encoding retains pre-token order instead of aggregating counts as training did. On the same ~20 MB input, 1 process: **2.92 s, 6.86 MB/s**; 8: **1.74 s, 11.51 MB/s**; 16: **1.67 s, 11.96 MB/s**. The three output ID sequences had identical hashes. These were single runs and did not measure total worker memory. A single large segment cannot be split among workers by this method.

The last 16-process profile took 3.13 s wall time with profiling. Parent wait/shutdown accounted for 1.55 s cumulative, and the 16 worker profiles showed 15.25 s summed in pre-tokenization; worker times overlap and cannot be interpreted as elapsed time. The profile includes regex matching and per-match byte tuple creation together. Roughly 11.96 MB/s gives an 825 GB estimate of about 19 hours at constant throughput; this is not a full-Pile measurement.

## 7. Streaming and Python details

An iterable supplies pieces one by one; a generator yields token IDs as requested. `encode_iterable` aims to avoid holding the entire file and its output at once. A chunk boundary can cut through a pre-token or special token. We discussed keeping a buffer and waiting until a pre-token appears complete before yielding. The current approach passes the provided iterable functional tests, but arbitrary input boundaries and bounded memory are not fully verified. The provided memory tests are skipped on macOS due to their Linux-specific `rlimit` checks.

`cProfile` reports where function execution time is spent but adds overhead. Its cumulative time includes called functions and can overlap with other entries. `resource.getrusage(...).ru_maxrss` is an inexpensive peak measurement; on macOS the value is bytes, but parent-only measurements exclude worker processes. Unprofiled runs are preferable for throughput comparisons.

## 8. What remains

- OpenWebText tokenizer training and the complete Section 2.5/2.7 comparison were intentionally deferred. We downloaded only ten OpenWebText documents for a small TinyStories-tokenizer experiment.
- Encoding the training and validation datasets as `uint16` arrays remains for later model training. A 10,000-entry or 32,000-entry vocabulary fits in the 0–65,535 range of `uint16`.
- The 23 provided tokenizer functional tests passed; two memory tests were skipped. The special-token and streaming behavior beyond the tested cases should not be assumed verified.
- Next: finish the relevant lecture, then begin Section 3. Avoid further open-ended tokenizer optimization unless it blocks later work.
