"""Balance of address bits and blocks for a memory of a given size.

The load variable of the design is the number of writers per cell,
``c = N (m + 1) / 2**m``.  Two things trade against each other when a fixed byte
budget is split:

* **address bits** buy cells, hence capacity, but each cell costs
  ``d_v d + d^2`` bytes, so ``m`` and the block count compete for the same
  budget: ``n * 2**m * cell_bytes = budget``;
* **blocks** buy redundancy against a noisy cue, but they duplicate the table,
  and each block is another copy of the same writes.

This module measures the error against ``(c, n)`` at a fixed address width,
turns it into an admissible occupancy ``c_max(n)`` for a target error, and then
solves the budget split:

    capacity(n) = c_max(n) * 2**m / (m + 1),   m = floor(log2(budget / (n * cell)))

so the only free variable is the block count.  Capacity per byte is
proportional to ``c_max(n) / n``, which is why the recommendation is "as many
blocks as it takes for ``c_max`` to saturate, and no more".

Run from the repository root::

    python -m pt_neural_memory.benchmarks.balance_bench
"""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from pt_neural_memory.benchmarks.addressed_bench import (
    D,
    DV,
    Table,
    build,
    format_table,
    mean,
    noisy,
    pairs,
    relative_squared_error,
    sample,
)

SIGMA = 0.25
BASE_M = 12
OCCUPANCIES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BLOCKS = (1, 2, 3, 4, 5, 8)
TARGETS = (0.02, 0.05)
BUDGETS = (2**24, 2**26, 2**28, 2**30)
SEEDS = (0, 1, 2)
RESULTS_PATH = Path("results/balance_bench.json")

_LAW: dict[tuple[int, float], float] = {}


def associations(writers: float, n_bits: int = BASE_M) -> int:
    return max(1, round(writers * 2**n_bits / (n_bits + 1)))


def measure(
    n_bits: int,
    n_blocks: int,
    writers: float,
    sigma: float = SIGMA,
    readout: str = "best",
    seeds: tuple[int, ...] = SEEDS,
) -> float:
    """Verified radius-1 error at one (m, n, occupancy, noise) point."""
    errors = []
    for seed in seeds:
        n = associations(writers, n_bits)
        keys, values = pairs(n, seed)
        memory = build(n_bits, n_blocks, seed)
        for start in range(0, n, 128):
            memory.store(keys[start : start + 128], values[start : start + 128])
        pick = sample(n, seed)
        read = memory.read_ball(noisy(keys[pick], sigma, seed), radius=1)
        prediction = read.best() if readout == "best" else read.mean()
        errors.append(relative_squared_error(prediction, values[pick]))
    return mean(errors)


def law(n_bits: int = BASE_M) -> dict[tuple[int, float], float]:
    """Cached error grid over (blocks, occupancy) at the base address width."""
    global _LAW
    if not _LAW:
        _LAW = {
            (n_blocks, writers): measure(n_bits, n_blocks, writers)
            for n_blocks in BLOCKS
            for writers in OCCUPANCIES
        }
    return _LAW


def law_table(n_bits: int = BASE_M) -> Table:
    grid = law(n_bits)
    series = {f"n={n}": [grid[(n, c)] for c in OCCUPANCIES] for n in BLOCKS}
    return Table(
        title=(
            f"Retrieval error at sigma={SIGMA} against writers per cell "
            f"(d={D}, m={n_bits}, verified r1)"
        ),
        x_label="writers per cell c",
        x=list(OCCUPANCIES),
        series=series,
    )


def address_table(writers: float = 2.0, blocks: int = 3) -> Table:
    """Error against address width at fixed occupancy and block count."""
    bit_counts = (8, 10, 12, 13)
    series = {"best block": [], "mean block": []}
    for n_bits in bit_counts:
        series["best block"].append(measure(n_bits, blocks, writers, readout="best"))
        series["mean block"].append(measure(n_bits, blocks, writers, readout="mean"))
    return Table(
        title=(
            f"Error against address width at c={writers}, n={blocks}, "
            f"sigma={SIGMA}"
        ),
        x_label="address bits m",
        x=list(bit_counts),
        series=series,
    )


def blocks_table(writers: float = 2.0) -> Table:
    """Error against block count at fixed occupancy."""
    counts = (1, 2, 3, 5, 9)
    series = {"best block": [], "mean block": []}
    for n_blocks in counts:
        series["best block"].append(measure(BASE_M, n_blocks, writers, readout="best"))
        series["mean block"].append(measure(BASE_M, n_blocks, writers, readout="mean"))
    return Table(
        title=f"Error against block count at c={writers}, m={BASE_M}, sigma={SIGMA}",
        x_label="blocks n",
        x=list(counts),
        series=series,
    )


def admissible_occupancy(target: float, n_blocks: int) -> float:
    """Largest c whose measured error is at most ``target``, log-log interpolated."""
    points = [(c, law()[(n_blocks, c)]) for c in OCCUPANCIES]
    if points[0][1] > target:
        return points[0][0] * target / points[0][1]
    for (c0, e0), (c1, e1) in zip(points, points[1:]):
        if e0 <= target < e1:
            weight = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(c0) + weight * (math.log(c1) - math.log(c0)))
    return points[-1][0]


def cmax_table() -> Table:
    series = {f"error<={target:g}": [] for target in TARGETS}
    for n_blocks in BLOCKS:
        for target in TARGETS:
            series[f"error<={target:g}"].append(
                admissible_occupancy(target, n_blocks)
            )
    return Table(
        title=f"Admissible writers per cell at sigma={SIGMA}",
        x_label="blocks n",
        x=list(BLOCKS),
        series=series,
    )


def cell_bytes(value_dim: int, key_dim: int, bytes_per_float: int = 4) -> int:
    return bytes_per_float * (value_dim * key_dim + key_dim**2)


def balance(budget: int, value_dim: int, key_dim: int, target: float) -> dict:
    """Best block count and address width for a budget, from the measured law."""
    per_cell = cell_bytes(value_dim, key_dim)
    rows = []
    for n_blocks in BLOCKS:
        cells = budget // (n_blocks * per_cell)
        if cells < 2:
            continue
        n_bits = min(int(math.floor(math.log2(cells))), key_dim, 20)
        occupancy = admissible_occupancy(target, n_blocks)
        capacity = int(occupancy * 2**n_bits / (n_bits + 1))
        rows.append(
            {
                "n_blocks": n_blocks,
                "n_bits": n_bits,
                "occupancy": occupancy,
                "capacity": capacity,
                "probes": n_blocks * (n_bits + 1),
                "bytes": n_blocks * per_cell * 2**n_bits,
            }
        )
    best = max(rows, key=lambda row: row["capacity"])
    return {"budget": budget, "target": target, "rows": rows, "best": best}


def balance_table(target: float, key_dim: int = 64, value_dim: int = 16) -> Table:
    series = {"capacity": [], "blocks": [], "bits": [], "probes/query": []}
    for budget in BUDGETS:
        best = balance(budget, value_dim, key_dim, target)["best"]
        series["capacity"].append(best["capacity"])
        series["blocks"].append(best["n_blocks"])
        series["bits"].append(best["n_bits"])
        series["probes/query"].append(best["probes"])
    return Table(
        title=(
            f"Best split at error<={target:g}, sigma={SIGMA}, d={key_dim}, "
            f"d_v={value_dim}"
        ),
        x_label="budget GiB",
        x=[budget / 2**30 for budget in BUDGETS],
        series=series,
    )


def validation(target: float = 0.05) -> list[dict[str, object]]:
    """Measure the recommended configuration instead of predicting it."""
    rows = []
    for budget in (2**24, 2**26):
        best = balance(budget, 16, 64, target)["best"]
        rows.append(
            {
                "budget_GiB": budget / 2**30,
                "n_bits": best["n_bits"],
                "n_blocks": best["n_blocks"],
                "occupancy": best["occupancy"],
                "capacity": best["capacity"],
                "target": target,
                "measured": measure(
                    best["n_bits"],
                    best["n_blocks"],
                    best["occupancy"],
                    seeds=(0, 1),
                ),
            }
        )
    return rows


def format_validation(rows: list[dict[str, object]]) -> str:
    lines = [
        "**Validation: predicted occupancy versus measured error**",
        "",
        "| budget GiB | m | n | c | capacity | target | measured |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['budget_GiB']:.2f} | {row['n_bits']} | {row['n_blocks']} "
            f"| {row['occupancy']:.2f} | {row['capacity']:,} "
            f"| {row['target']:g} | {row['measured']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    tables = [
        law_table(),
        address_table(),
        blocks_table(),
        cmax_table(),
        balance_table(0.05),
        balance_table(0.02),
    ]
    for table in tables:
        print(format_table(table))
        print()
    rows = validation()
    print(format_validation(rows))
    print()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "protocol": {
                    "key_dim": D,
                    "value_dim": DV,
                    "sigma": SIGMA,
                    "base_n_bits": BASE_M,
                    "occupancies": list(OCCUPANCIES),
                    "blocks": list(BLOCKS),
                    "seeds": list(SEEDS),
                },
                "tables": [asdict(table) for table in tables],
                "validation": rows,
                "balance": [balance(budget, 16, 64, 0.05) for budget in BUDGETS],
            },
            indent=2,
        )
    )
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
