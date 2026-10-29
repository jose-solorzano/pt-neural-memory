"""Sizing rules for a dense RLS-cell addressed memory.

With ``n`` blocks of ``2**m`` cells, key dimension ``d`` and value dimension
``d_v``, every cell carries a ``d_v x d`` map and a ``d x d`` covariance, so the
float count is

    n * (2**m * (d_v * d + d**2) + m * d),

and the byte count is that times ``bytes_per_float``.  The covariance is the
reason the table is wide: it costs ``d**2`` per cell no matter how narrow the
value is.

The capacity is set by how many writers one cell can interpolate.  A stored
pair trains ``m + 1`` cells per block (radius 1), so the writers at a cell are

    c = N * (m + 1) / 2**m,

and the map stops being well-conditioned once ``c`` approaches the key
dimension.  Measurements at ``d = 32`` put the 1% clean-error point at
``c ~ 26``, about ``0.8 d``; key noise moves that point far lower, so a
deployment should pass the fraction it measured rather than trust the default.
Capacity is therefore

    N = f * d * 2**m / (m + 1),

with ``f = 0.8`` by default.  Sizing a budget means choosing ``m`` and ``n`` to
maximise that subject to the byte constraint; the search is a one-dimensional
enumeration over ``m`` because the block count is ``min(max_blocks, budget //
block_bytes)``.

The block count is not free either.  Capacity per byte is proportional to
``c_max(n) / n``, where ``c_max(n)`` is the measured writers per cell that meet
the target error, and ``c_max`` saturates at three blocks
(``src/pt_neural_memory/benchmarks/balance_bench.py``).  ``balanced_layout`` therefore picks two or
three blocks and spends everything else on address bits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# Measured clean-key writers per cell at 1% error, as a fraction of key_dim.
# src/pt_neural_memory/benchmarks/addressed_bench.py is the source: at d=32, N=8192, m=12, the
# occupancy is c=26 and the own-address error is 5e-3.
WRITERS_PER_CELL_FRACTION = 0.8
BYTES_PER_FLOAT = 4

# Admissible writers per cell against the block count, at sigma=0.25, d=32,
# verified radius 1.  From src/pt_neural_memory/benchmarks/balance_bench.py: the error grid is
# interpolated to the largest occupancy whose measured error meets the target.
# The block axis saturates by three blocks and then slowly degrades, because
# the variance score prefers a crowded cell and more blocks give it more
# chances.  These tables are what turns "a memory of a certain size" into an
# address width and a block count.
ADMISSIBLE_WRITERS_5PCT: dict[int, float] = {
    1: 0.222,
    2: 5.51,
    3: 5.65,
    4: 5.21,
    5: 5.04,
    8: 4.60,
}
ADMISSIBLE_WRITERS_2PCT: dict[int, float] = {
    1: 0.089,
    2: 1.36,
    3: 1.97,
    4: 1.66,
    5: 1.45,
    8: 1.17,
}


def writers_per_cell(
    key_dim: int, fraction: float = WRITERS_PER_CELL_FRACTION
) -> float:
    """Writers one cell can hold at the target error, default 1% clean."""
    if key_dim < 1:
        raise ValueError(f"key_dim must be positive, got {key_dim}")
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction}")
    return fraction * key_dim


def pairs_per_cell(
    n_bits: int, key_dim: int, fraction: float = WRITERS_PER_CELL_FRACTION
) -> float:
    """Associations per cell: every pair trains ``n_bits + 1`` cells."""
    if n_bits < 1:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    return writers_per_cell(key_dim, fraction) / (n_bits + 1)


def capacity(
    n_bits: int,
    n_blocks: int,
    key_dim: int,
    fraction: float = WRITERS_PER_CELL_FRACTION,
) -> int:
    """Associations the memory holds at the target error."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    return int(pairs_per_cell(n_bits, key_dim, fraction) * 2**n_bits)


def state_parts(
    n_bits: int,
    n_blocks: int,
    value_dim: int,
    key_dim: int = 64,
    bytes_per_float: int = BYTES_PER_FLOAT,
) -> dict[str, int]:
    """Bytes in each part of the state."""
    if n_bits < 1 or n_blocks < 1 or value_dim < 1 or key_dim < 1:
        raise ValueError("dimensions and block counts must be positive")
    cells = 1 << n_bits
    return {
        "maps": bytes_per_float * n_blocks * cells * value_dim * key_dim,
        "covariances": bytes_per_float * n_blocks * cells * key_dim * key_dim,
        "projection": bytes_per_float * n_blocks * n_bits * key_dim,
    }


def state_bytes(
    n_bits: int,
    n_blocks: int,
    value_dim: int,
    key_dim: int = 64,
    bytes_per_float: int = BYTES_PER_FLOAT,
) -> int:
    """Total state bytes for one configuration."""
    return sum(
        state_parts(n_bits, n_blocks, value_dim, key_dim, bytes_per_float).values()
    )


@dataclass(frozen=True)
class MemorySizing:
    """A configuration chosen to fit a byte budget."""

    n_bits: int
    n_blocks: int
    capacity: int
    writers_per_cell: float
    bytes_by_part: dict[str, int]
    probes_per_query: int

    @property
    def cells_per_block(self) -> int:
        return 1 << self.n_bits

    @property
    def total_cells(self) -> int:
        return self.n_blocks * self.cells_per_block

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_part.values())

    @property
    def bytes_per_pair(self) -> float:
        return self.total_bytes / self.capacity

    def summary(self) -> str:
        return (
            f"{self.n_blocks} blocks x 2**{self.n_bits} cells, "
            f"capacity {self.capacity:,} at {self.writers_per_cell:.1f} writers/cell, "
            f"{self.total_bytes / 2**30:.2f} GiB "
            f"({self.bytes_per_pair / 1024:.1f} KiB/pair), "
            f"{self.probes_per_query} probes/query"
        )


def size_for_budget(
    budget_bytes: int,
    value_dim: int,
    key_dim: int = 64,
    fraction: float = WRITERS_PER_CELL_FRACTION,
    min_blocks: int = 2,
    max_blocks: int = 8,
    bytes_per_float: int = BYTES_PER_FLOAT,
) -> MemorySizing:
    """Maximise capacity within a byte budget.

    Enumerates the address width and, for each, uses as many blocks as the
    budget allows up to ``max_blocks``.  ``min_blocks`` keeps the ensemble
    meaningful; ``max_blocks`` bounds the read cost (``n * (m + 1)`` probes).

    Raises:
        ValueError: if the budget cannot hold ``min_blocks`` blocks at any
            width.
    """
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    if min_blocks < 1 or max_blocks < min_blocks:
        raise ValueError(
            f"need 1 <= min_blocks <= max_blocks, got {min_blocks}, {max_blocks}"
        )
    best: MemorySizing | None = None
    for n_bits in range(2, 31):
        if n_bits > key_dim:
            break
        one_block = state_bytes(n_bits, 1, value_dim, key_dim, bytes_per_float)
        n_blocks = min(budget_bytes // one_block, max_blocks)
        if n_blocks < min_blocks:
            continue
        candidate = MemorySizing(
            n_bits=n_bits,
            n_blocks=int(n_blocks),
            capacity=capacity(n_bits, int(n_blocks), key_dim, fraction),
            writers_per_cell=writers_per_cell(key_dim, fraction),
            bytes_by_part=state_parts(
                n_bits, int(n_blocks), value_dim, key_dim, bytes_per_float
            ),
            probes_per_query=int(n_blocks) * (n_bits + 1),
        )
        if best is None or candidate.capacity > best.capacity:
            best = candidate
    if best is None:
        raise ValueError(
            f"{budget_bytes} bytes cannot hold {min_blocks} blocks at any width"
        )
    return best


def balanced_layout(
    budget_bytes: int,
    value_dim: int,
    key_dim: int = 64,
    admissible: Mapping[int, float] | None = None,
    max_blocks: int = 8,
    bytes_per_float: int = BYTES_PER_FLOAT,
) -> MemorySizing:
    """Split a budget between address bits and blocks by measured occupancy.

    For each measured block count the budget fixes the address width, because
    ``n * 2**m * cell_bytes <= budget``, and the measured admissible occupancy
    ``c_max(n)`` fixes the capacity, ``c_max(n) * 2**m / (m + 1)``.  The split
    with the largest capacity wins.

    Capacity per byte is proportional to ``c_max(n) / n``, and ``c_max``
    saturates at three blocks, so the answer is two or three blocks: two when
    dropping a block buys an extra address bit, three otherwise.  More blocks
    only add probes.

    Args:
        admissible: measured writers per cell keyed by block count.  Defaults
            to ``ADMISSIBLE_WRITERS_5PCT``, which is what
            ``src/pt_neural_memory/benchmarks/balance_bench.py`` measures at ``sigma=0.25``.
        max_blocks: largest block count to consider.

    Raises:
        ValueError: if the budget cannot hold one cell at two blocks.
    """
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    if value_dim < 1 or key_dim < 1:
        raise ValueError("value_dim and key_dim must be positive")
    table = ADMISSIBLE_WRITERS_5PCT if admissible is None else admissible
    per_cell = bytes_per_float * (value_dim * key_dim + key_dim * key_dim)
    best: MemorySizing | None = None
    for n_blocks, occupancy in table.items():
        if n_blocks < 1 or n_blocks > max_blocks:
            continue
        cells = budget_bytes // (n_blocks * per_cell)
        if cells < 2:
            continue
        n_bits = min(int(math.floor(math.log2(cells))), key_dim, 20)
        # The projection is not part of the cell table, so a width can still
        # overshoot the budget by n * m * key_dim numbers.
        while n_bits >= 2 and (
            state_bytes(n_bits, n_blocks, value_dim, key_dim, bytes_per_float)
            > budget_bytes
        ):
            n_bits -= 1
        if n_bits < 2:
            continue
        candidate = MemorySizing(
            n_bits=n_bits,
            n_blocks=int(n_blocks),
            capacity=int(occupancy * 2**n_bits / (n_bits + 1)),
            writers_per_cell=occupancy,
            bytes_by_part=state_parts(
                n_bits, int(n_blocks), value_dim, key_dim, bytes_per_float
            ),
            probes_per_query=int(n_blocks) * (n_bits + 1),
        )
        if best is None or candidate.capacity > best.capacity:
            best = candidate
    if best is None:
        raise ValueError(
            f"{budget_bytes} bytes cannot hold one block at any block count"
        )
    return best


def bits_for_capacity(
    target: int,
    key_dim: int,
    fraction: float = WRITERS_PER_CELL_FRACTION,
) -> int:
    """Smallest address width that holds ``target`` associations."""
    if target < 1:
        raise ValueError(f"target must be positive, got {target}")
    n_bits = 1
    while capacity(n_bits, 1, key_dim, fraction) < target:
        n_bits += 1
    return n_bits


def main() -> None:
    budget = 4 * 2**30
    print(f"4 GiB budget ({budget:,} bytes), d=64, float32, 1% clean error")
    print()
    print("| value dim | blocks | bits | cells/block | capacity | state GiB | KiB/pair |")
    print("|---|---|---|---|---|---|---|")
    for value_dim in (16, 64, 256, 768):
        sizing = size_for_budget(budget, value_dim, key_dim=64)
        print(
            f"| {value_dim} | {sizing.n_blocks} | {sizing.n_bits} "
            f"| {sizing.cells_per_block:,} | {sizing.capacity:,} "
            f"| {sizing.total_bytes / 2**30:.2f} "
            f"| {sizing.bytes_per_pair / 1024:.1f} |"
        )
    print()
    print("Balanced split at sigma=0.25, target 5%:")
    print()
    print("| value dim | blocks | bits | capacity | state GiB | probes/read |")
    print("|---|---|---|---|---|---|")
    for value_dim in (16, 64, 256, 768):
        sizing = balanced_layout(budget, value_dim, key_dim=64)
        print(
            f"| {value_dim} | {sizing.n_blocks} | {sizing.n_bits} "
            f"| {sizing.capacity:,} | {sizing.total_bytes / 2**30:.2f} "
            f"| {sizing.probes_per_query} |"
        )


if __name__ == "__main__":
    main()
