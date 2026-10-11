"""Split operations into prompt-sized batches so large specs fit small local models."""

from collections.abc import Sequence

from api_shakedown.spec.models import Operation

DEFAULT_MAX_CHARS = 12_000  # roughly 3k tokens of operation JSON


def chunk_operations(
    operations: Sequence[Operation], max_chars: int = DEFAULT_MAX_CHARS
) -> list[list[Operation]]:
    """Greedy batching in tag order so related operations tend to share a chunk.

    An operation larger than `max_chars` on its own gets a chunk to itself.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1.")
    ordered = sorted(operations, key=lambda op: op.tags[0] if op.tags else "")
    chunks: list[list[Operation]] = []
    current: list[Operation] = []
    size = 0
    for op in ordered:
        op_size = len(op.to_prompt())
        if current and size + op_size > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(op)
        size += op_size
    if current:
        chunks.append(current)
    return chunks
