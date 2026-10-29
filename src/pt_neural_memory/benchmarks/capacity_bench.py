"""How much can one cell hold, and how much a whole addressed memory?

A cell's map is ``d_v x d``, so it has at most ``d`` independent key directions
and can interpolate at most ``d`` associations exactly; beyond that the keys are
linearly dependent and the values generally inconsistent.  The table multiplies
that by the number of cells, but pays for delivery: every pair is trained into
``m + 1`` cells per block at radius 1, and blocks are replicas rather than extra
storage.

This module measures:

* a single cell's exact-recall capacity against the number of stored pairs;
* the table's capacity at several address widths, and that the block count does
  not change it;
* the equal-state comparison: one large cell with the same number of floats as
  a small table.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.capacity_bench
"""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from pt_neural_memory.benchmarks.addressed_bench import Table, format_table, mean
from pt_neural_memory.addressed_memory import AddressedMemory
from pt_neural_memory.mem_rls import RLSMemCell

D = 32
DV = 16
LAMBDA = 0.01
SEEDS = (0, 1, 2)
EXACT = 1e-4
RESULTS_PATH = Path("results/capacity_bench.json")


def per_pair_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().sum(dim=-1) / target.square().sum(dim=-1)


def scored(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    errors = per_pair_error(pred, target)
    return errors.mean().item(), (errors < EXACT).float().mean().item()


def cell_capacity(ns: tuple[int, ...] = (8, 16, 24, 32, 40, 64)) -> Table:
    series = {"exact fraction": [], "mean error": []}
    for n in ns:
        fractions, errors = [], []
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            keys = torch.randn(n, D, generator=generator)
            values = torch.randn(n, DV, generator=generator)
            cell = RLSMemCell(D, DV, rr_lambda=LAMBDA)
            cell.store(keys, values)
            error, fraction = scored(cell(keys), values)
            fractions.append(fraction)
            errors.append(error)
        series["exact fraction"].append(mean(fractions))
        series["mean error"].append(mean(errors))
    return Table(
        title=f"One cell, d={D}, d_v={DV}: exact recall against stored pairs",
        x_label="associations N",
        x=list(ns),
        series=series,
    )


def table_capacity(n_bits: int, n_blocks: int = 1) -> list[dict[str, object]]:
    """Exact fraction against occupancy for one table configuration."""
    rows = []
    for writers in (8, 16, 24, 32, 40):
        n = max(1, round(writers * 2**n_bits / (n_bits + 1)))
        fractions, errors = [], []
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            keys = torch.randn(n, D, generator=generator)
            values = torch.randn(n, DV, generator=generator)
            memory = AddressedMemory(
                D,
                DV,
                n_blocks=n_blocks,
                n_bits=n_bits,
                write_radius=1,
                rr_lambda=LAMBDA,
                seed=seed,
            )
            for start in range(0, n, 128):
                memory.store(keys[start : start + 128], values[start : start + 128])
            error, fraction = scored(memory.read_blocks(keys)[0], values)
            fractions.append(fraction)
            errors.append(error)
        rows.append(
            {
                "n_bits": n_bits,
                "n_blocks": n_blocks,
                "cells": n_blocks * 2**n_bits,
                "writers": writers,
                "N": n,
                "exact": mean(fractions),
                "error": mean(errors),
            }
        )
    return rows


def table_tables(bit_counts: tuple[int, ...] = (8, 10, 12)) -> list[Table]:
    rows = [row for n_bits in bit_counts for row in table_capacity(n_bits)]
    tables = []
    for n_bits in bit_counts:
        series = {
            "exact fraction": [
                row["exact"] for row in rows if row["n_bits"] == n_bits
            ],
            "mean error": [
                row["error"] for row in rows if row["n_bits"] == n_bits
            ],
        }
        tables.append(
            Table(
                title=(
                    f"Table at m={n_bits} ({2**n_bits} cells), d={D}, "
                    f"one block"
                ),
                x_label="writers per cell c",
                x=[8, 16, 24, 32, 40],
                series=series,
            )
        )
    return tables


def block_table() -> Table:
    """The block count is redundancy, not storage."""
    rows = [row for n_blocks in (1, 3) for row in table_capacity(10, n_blocks)]
    series = {
        "exact, 1 block": [row["exact"] for row in rows if row["n_blocks"] == 1],
        "exact, 3 blocks": [row["exact"] for row in rows if row["n_blocks"] == 3],
    }
    return Table(
        title="Exact recall at m=10 with one and three blocks",
        x_label="writers per cell c",
        x=[8, 16, 24, 32, 40],
        series=series,
    )


def equal_state_dim(n_bits: int) -> int:
    """Input width of one cell whose state matches a one-block table at m."""
    state = 2**n_bits * (DV * D + D**2)
    return int((-DV + math.sqrt(DV**2 + 4 * state)) / 2)


def equal_state_table(bit_counts: tuple[int, ...] = (8, 10)) -> Table:
    """Capacity at 99% exact for equal-state contestants, by table width."""
    series: dict[str, list[float]] = {"one cell": [], "table (c=16)": []}
    for n_bits in bit_counts:
        width = equal_state_dim(n_bits)
        capacity = 0
        for share in (0.8, 0.9, 0.95, 1.0):
            n = round(share * width)
            fractions = []
            for seed in (0, 1):
                generator = torch.Generator().manual_seed(seed)
                keys = torch.randn(n, width, generator=generator)
                values = torch.randn(n, DV, generator=generator)
                cell = RLSMemCell(width, DV, rr_lambda=LAMBDA)
                cell.store(keys, values)
                fractions.append(scored(cell(keys), values)[1])
            if mean(fractions) >= 0.99:
                capacity = n
            else:
                break
        series["one cell"].append(capacity)
        rows = table_capacity(n_bits)
        exact_rows = [row for row in rows if row["exact"] >= 0.99]
        series["table (c=16)"].append(
            max(row["N"] for row in exact_rows) if exact_rows else 0
        )
    return Table(
        title=(
            "Equal state, 99% exact: one cell of width sqrt(state) versus the "
            "table, same float count"
        ),
        x_label="table address bits",
        x=list(bit_counts),
        series=series,
    )


def main() -> None:
    torch.set_num_threads(4)
    tables = [cell_capacity(), *table_tables(), block_table(), equal_state_table()]
    for table in tables:
        print(format_table(table))
        print()
    rows = table_capacity(12)
    print(
        "Capacity at m=12, one block, d=32: "
        f"{rows[1]['N']:,} pairs at c=16 ({rows[1]['exact']:.2f} exact), "
        f"a cell of width 32 holds 24."
    )
    print(
        "Ratio to one cell, same d: "
        f"{rows[1]['N'] / 24:.0f}x at m=12 (2^m/(m+1) = {2**12 / 13:.0f})."
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "protocol": {
                    "key_dim": D,
                    "value_dim": DV,
                    "rr_lambda": LAMBDA,
                    "seeds": list(SEEDS),
                    "exact_threshold": EXACT,
                },
                "tables": [asdict(table) for table in tables],
                "equal_state_dims": {
                    str(n_bits): equal_state_dim(n_bits) for n_bits in (8, 10, 12)
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
