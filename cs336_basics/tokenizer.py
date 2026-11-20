"""Helpers for encoding text with a byte-level BPE tokenizer."""

import json
from concurrent.futures import ProcessPoolExecutor

import regex as re

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(text: str) -> list[tuple[bytes, ...]]:
    """Split text into ordered pre-tokens made of individual UTF-8 bytes."""
    pre_tokens = []
    for match in re.finditer(PATTERN, text):
        byte_tokens = tuple(bytes([value]) for value in match.group().encode("utf-8"))
        pre_tokens.append(byte_tokens)
    return pre_tokens


def split_special_tokens(text: str, special_tokens: list[str]) -> tuple[list[str], list[str]]:
    """Return ordered text segments and the special tokens between them."""
    if not special_tokens:
        return [text], []
    ordered_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in ordered_tokens)
    parts = re.split(f"({pattern})", text)
    return parts[::2], parts[1::2]


def pre_tokenize_segments(segments: list[str]) -> list[list[tuple[bytes, ...]]]:
    """Pre-tokenize each segment, keeping the results grouped in order."""
    return [pre_tokenize(segment) for segment in segments]


def group_segments_into_chunks(segments: list[str], num_chunks: int) -> list[list[str]]:
    """Group consecutive segments into roughly equal chunks without splitting them."""
    target_size = max(1, sum(len(segment) for segment in segments) // num_chunks)
    chunks = []
    current = []
    current_size = 0
    for segment in segments:
        current.append(segment)
        current_size += len(segment)
        if current_size >= target_size and len(chunks) < num_chunks - 1:
            chunks.append(current)
            current = []
            current_size = 0
    if current:
        chunks.append(current)
    return chunks


def merge_segments(
    segments: list[list[tuple[bytes, ...]]],
    merge_priorities: dict[tuple[bytes, bytes], int],
    cache: dict[tuple[bytes, ...], tuple[bytes, ...]],
) -> list[list[tuple[bytes, ...]]]:
    """Apply merge rules to each pre-token, preserving segment order."""
    result = []
    for segment in segments:
        merged_segment = []
        for tokens in segment:
            original = tokens
            if original in cache:
                merged_segment.append(cache[original])
                continue
            while True:
                adjacent_pairs = set(zip(tokens, tokens[1:]))
                best_pair = min(
                    (pair for pair in adjacent_pairs if pair in merge_priorities),
                    key=merge_priorities.get,
                    default=None,
                )
                if best_pair is None:
                    break

                updated = []
                i = 0
                while i < len(tokens):
                    if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                        updated.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        updated.append(tokens[i])
                        i += 1
                tokens = tuple(updated)
            if len(cache) < 100_000:
                cache[original] = tokens
            merged_segment.append(tokens)
        result.append(merged_segment)
    return result


class Tokenizer:
    """BPE tokenizer interface from the assignment handout."""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merges = merges
        self.merge_cache = {}
        self.merge_priorities = {pair: priority for priority, pair in reversed(list(enumerate(merges)))}
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            vocab_data = json.load(vocab_file)
        with open(merges_filepath, encoding="utf-8") as merges_file:
            merges_data = json.load(merges_file)

        vocab = {int(token_id): bytes.fromhex(token) for token_id, token in vocab_data["vocab"].items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_data["merges"]]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str, num_processes: int = 1) -> list[int]:
        segments, found_special_tokens = split_special_tokens(text, self.special_tokens)
        if num_processes == 1:
            pre_tokens = pre_tokenize_segments(segments)
        else:
            chunks = group_segments_into_chunks(segments, num_processes)
            with ProcessPoolExecutor(max_workers=num_processes) as pool:
                chunk_results = pool.map(pre_tokenize_segments, chunks)
                pre_tokens = [segment for chunk in chunk_results for segment in chunk]
        merged_segments = merge_segments(pre_tokens, self.merge_priorities, self.merge_cache)
        token_ids = []
        for index, segment in enumerate(merged_segments):
            for pre_token in segment:
                for token in pre_token:
                    token_ids.append(self.token_to_id[token])
            if index < len(found_special_tokens):
                special_token = found_special_tokens[index].encode("utf-8")
                token_ids.append(self.token_to_id[special_token])
        return token_ids

    def encode_iterable(self, iterable):
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            while buffer:
                special_matches = [
                    (buffer.find(token), token)
                    for token in self.special_tokens
                    if token and buffer.find(token) >= 0
                ]
                if special_matches:
                    position, special = min(special_matches, key=lambda item: (item[0], -len(item[1])))
                    yield from self.encode(buffer[:position])
                    yield self.token_to_id[special.encode("utf-8")]
                    buffer = buffer[position + len(special):]
                    continue

                matches = list(re.finditer(PATTERN, buffer))
                if len(matches) < 2:
                    break
                boundary = matches[0].end()
                yield from self.encode(buffer[:boundary])
                buffer = buffer[boundary:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
