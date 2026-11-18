"""Alternative BPE merge strategies for comparing training performance."""

import heapq


def build_index(token_counts):
    """Count every adjacent occurrence and record which sequences contain it."""
    pair_counts = {}
    pair_to_seq = {}
    for tokens, count in token_counts.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_seq.setdefault(pair, set()).add(tokens)
    return pair_counts, pair_to_seq


def merge_sequences(token_counts, pair):
    """Merge non-overlapping occurrences, from left to right."""
    result = {}
    merged = pair[0] + pair[1]
    for tokens, count in token_counts.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        key = tuple(new_tokens)
        result[key] = result.get(key, 0) + count
    return result


class ReversePair:
    """Break equal-count heap ties in descending byte-pair order."""

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def incremental_merges(token_counts, use_heap=False):
    """Yield merges while maintaining counts and the sequence index in place."""
    pair_counts, pair_to_seq = build_index(token_counts)
    heap = []
    if use_heap:
        heap = [(-count, ReversePair(pair)) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

    while pair_counts:
        if use_heap:
            # Older heap entries are harmless: the dictionary holds current counts.
            while heap:
                negative_count, wrapped_pair = heapq.heappop(heap)
                best_pair = wrapped_pair.pair
                if pair_counts.get(best_pair) == -negative_count:
                    break
            else:
                raise RuntimeError("Heap and pair counts are inconsistent")
        else:
            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        affected = {seq: token_counts[seq] for seq in pair_to_seq[best_pair]}
        merged_counts = merge_sequences(affected, best_pair)
        changed = set()

        # Remove old sequences before adding new ones, including on key collisions.
        for tokens, count in affected.items():
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] -= count
                pair_to_seq[pair].discard(tokens)
                changed.add(pair)
            del token_counts[tokens]

        for tokens, count in merged_counts.items():
            token_counts[tokens] = token_counts.get(tokens, 0) + count
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_to_seq.setdefault(pair, set()).add(tokens)
                changed.add(pair)

        for pair in changed:
            if pair_counts[pair] == 0:
                del pair_counts[pair]
                del pair_to_seq[pair]
            elif use_heap:
                heapq.heappush(heap, (-pair_counts[pair], ReversePair(pair)))
        yield best_pair
