"""Parallel pre-tokenization for the TinyStories end-of-document marker."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import regex as re

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MARKER = b"<|endoftext|>"


def chunk_boundaries(path, num_chunks):
    """Find approximately equal byte ranges, beginning at document boundaries."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        boundaries = [0]
        for i in range(1, num_chunks):
            f.seek(size * i // num_chunks)
            # Keep an overlap so a marker across two reads is not missed.
            offset = f.tell()
            tail = b""
            while block := f.read(65536):
                data = tail + block
                pos = data.find(MARKER)
                if pos >= 0:
                    boundaries.append(offset - len(tail) + pos)
                    break
                tail = data[-(len(MARKER) - 1):]
                offset += len(block)
            else:
                boundaries.append(size)
        boundaries.append(size)
    return sorted(set(boundaries))


def count_chunk(args):
    """Read one chunk and count its pre-tokens without crossing markers."""
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    counts = Counter()
    for segment in text.split(MARKER.decode("ascii")):
        counts.update(match.group() for match in re.finditer(PATTERN, segment))
    return counts


def parallel_pre_token_counts(path, num_processes=8):
    """Count chunks independently, then add them into one global dictionary."""
    boundaries = chunk_boundaries(path, num_processes)
    chunks = [(path, start, end) for start, end in zip(boundaries, boundaries[1:])]
    counts = Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        for chunk_counts in pool.map(count_chunk, chunks):
            counts.update(chunk_counts)
    return counts
