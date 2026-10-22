"""Shape buckets for captured CUDA graphs.

A captured graph serves one shape, so whatever is replayed from graphs -- the
ranking loss's beam search, the frozen context tower -- is padded to a bounded
set of sizes first.
"""

from __future__ import annotations


def bucket(size: int) -> int:
    """*size* rounded up to the next of four steps an octave: 8, 10, 12, 14, 16, 20, ...

    Padding every captured call to these sizes bounds the graphs a run records
    by a few an octave, at a quarter of padding at most.
    """
    if size <= 8:
        return 8
    quantum = 1 << (size.bit_length() - 3)
    return -(-size // quantum) * quantum
