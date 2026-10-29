"""Scaling sweeps for the RLS-cell addressed memory.

Two questions that the load sweep in ``addressed_bench`` does not isolate:

* **occupancy.** Error against the number of writers per cell ``c``.  ``c`` is
  the design's real load variable: a cell interpolates until its map becomes
  ill-conditioned, which happens near ``c = key_dim``, not near ``c = 1`` as it
  did for a cell that stored a mean.
* **blocks.** Error against the number of independent blocks at fixed address
  width.  Blocks are replicas: each block's map is fitted to the same writes,
  so the ensemble reduces the read variance without lowering occupancy.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.scaling_bench
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from pt_neural_memory.benchmarks.addressed_bench import (
    D,
    DV,
    LAMBDA,
    M,
    NB,
    Table,
    build,
    error,
    format_table,
    mean,
    noisy,
    pairs,
    read_ball,
    read_own,
    relative_squared_error,
    sample,
)

SEEDS = (0, 1, 2)
OCCUPANCIES = (0.2, 0.8, 3.2, 6.5, 13.0, 26.0, 52.0)
BLOCK_COUNTS = (1, 2, 3, 5, 9)
RESULTS_PATH = Path("results/scaling_bench.json")


def associations_for(writers: float, n_bits: int = M) -> int:
    """Number of stored pairs that produce ``writers`` writers per cell."""
    return max(1, round(writers * 2**n_bits / (n_bits + 1)))


def occupancy_table() -> Table:
    series = {
        "own sigma0": [],
        "verified r1 sigma0": [],
        "own sigma0.3": [],
        "verified r1 sigma0.3": [],
    }
    for writers in OCCUPANCIES:
        n = associations_for(writers)
        series["own sigma0"].append(error(read_own, n, 0.0))
        series["verified r1 sigma0"].append(error(read_ball, n, 0.0))
        series["own sigma0.3"].append(error(read_own, n, 0.3))
        series["verified r1 sigma0.3"].append(error(read_ball, n, 0.3))
    return Table(
        title=(
            f"Error against writers per cell (d={D}, m={M}, n={NB}, "
            f"lambda={LAMBDA}, radius 1)"
        ),
        x_label="writers per cell c",
        x=list(OCCUPANCIES),
        series=series,
    )


def block_table(
    block_counts: tuple[int, ...] = BLOCK_COUNTS,
    associations: int = 1024,
) -> Table:
    series = {"clean": [], "sigma0.3": []}
    for n_blocks in block_counts:
        for sigma in (0.0, 0.3):
            errors = []
            for seed in SEEDS:
                keys, values = pairs(associations, seed)
                memory = build(M, n_blocks, seed)
                for start in range(0, associations, 128):
                    memory.store(
                        keys[start : start + 128], values[start : start + 128]
                    )
                pick = sample(associations, seed)
                errors.append(
                    relative_squared_error(
                        memory.read_ball(noisy(keys[pick], sigma, seed), radius=1).best(),
                        values[pick],
                    )
                )
            series["clean" if sigma == 0.0 else "sigma0.3"].append(mean(errors))
    return Table(
        title=f"Blocks at fixed 2**{M} cells/block, N={associations}, verified r1",
        x_label="blocks n",
        x=list(block_counts),
        series=series,
    )


def main() -> None:
    tables = [occupancy_table(), block_table()]
    for table in tables:
        print(format_table(table))
        print()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "protocol": {
                    "key_dim": D,
                    "value_dim": DV,
                    "n_bits": M,
                    "n_blocks": NB,
                    "rr_lambda": LAMBDA,
                    "seeds": list(SEEDS),
                },
                "tables": [asdict(table) for table in tables],
            },
            indent=2,
        )
    )
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
