"""Benchmark suite for the RLS-cell addressed memory.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.addressed_bench

The memory is ``n_blocks`` blocks of ``2**n_bits`` cells.  A key's address in
a block is the sign pattern of a random projection; a store trains the cell at
that address and (with ``write_radius=1``) every one-bit neighbour.  A cell is
an ``RLSMemCell``: a ridge linear map of the keys written to it, with its exact
posterior covariance.  A plain read returns the map at the query's own address;
``VerifiedMemory`` probes the neighbourhood and keeps the cell with the smallest
posterior variance.

The suite separates six questions:

* ``load``: how does error grow with the number of stored pairs?
* ``noise``: how many corrupted address bits does the design survive?
* ``lambda``: how much ridge, and what does it buy?
* ``read``: what does verification add over the query's own address?
* ``cost``: microseconds per write and read, and the resident state.
* ``prior``: the archived numbers of the mean-cell implementation.

Results are printed as Markdown and written to ``results/addressed_bench.json``.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import torch

from pt_neural_memory.addressed_memory import AddressedMemory, VerifiedMemory

D = 32
DV = 16
M = 12
NB = 3
LAMBDA = 0.01
SEEDS = (0, 1, 2)
SAMPLE = 512
NOISE_OFFSET = 1000
SAMPLE_OFFSET = 5000
RESULTS_PATH = Path("results/addressed_bench.json")
PRIOR_PATH = Path("results/mean_cell_bench.json")

Reader = Callable[[AddressedMemory, torch.Tensor], torch.Tensor]


@dataclass
class Table:
    """One sweep: an x-axis and one error series per variant."""

    title: str
    x_label: str
    x: list[float | str]
    series: dict[str, list[float]]


def pairs(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, D, generator=generator),
        torch.randn(n, DV, generator=generator),
    )


def noisy(keys: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add i.i.d. Gaussian key noise; the same draw is reused across variants."""
    if sigma == 0.0:
        return keys
    generator = torch.Generator().manual_seed(seed + NOISE_OFFSET)
    return keys + sigma * torch.randn(keys.shape, generator=generator)


def sample(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + SAMPLE_OFFSET)
    return torch.randperm(n, generator=generator)[: min(SAMPLE, n)]


def relative_squared_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of target energy left unexplained; 1 is a zero answer."""
    return (pred - target).square().sum().item() / target.square().sum().item()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build(n_bits: int, n_blocks: int, seed: int, **kwargs: object) -> AddressedMemory:
    options = {
        "write_radius": 1,
        "read_radius": 1,
        "rr_lambda": LAMBDA,
        "seed": seed,
        **kwargs,
    }
    return VerifiedMemory(D, DV, n_blocks, n_bits, **options)


def read_own(memory: AddressedMemory, query: torch.Tensor) -> torch.Tensor:
    return memory.read_blocks(query).mean(dim=0)


def read_ball(memory: AddressedMemory, query: torch.Tensor) -> torch.Tensor:
    return memory.read_ball(query, radius=1).best()


def read_ball2(memory: AddressedMemory, query: torch.Tensor) -> torch.Tensor:
    return memory.read_ball(query, radius=2).best()


def read_guided(memory: AddressedMemory, query: torch.Tensor) -> torch.Tensor:
    assert isinstance(memory, VerifiedMemory)
    return memory.read_guided(query, search_bits=6).best()


def error(
    reader: Reader,
    n: int,
    sigma: float,
    n_bits: int = M,
    n_blocks: int = NB,
    seeds: tuple[int, ...] = SEEDS,
    **kwargs: object,
) -> float:
    """Mean relative squared error over seeds, evaluated on a query sample."""
    errors = []
    for seed in seeds:
        keys, values = pairs(n, seed)
        memory = build(n_bits, n_blocks, seed, **kwargs)
        for start in range(0, n, 128):
            memory.store(keys[start : start + 128], values[start : start + 128])
        pick = sample(n, seed)
        errors.append(
            relative_squared_error(
                reader(memory, noisy(keys[pick], sigma, seed)), values[pick]
            )
        )
    return mean(errors)


def writers_per_cell(n: int, n_bits: int = M) -> float:
    """Expected writers at a cell, for write radius 1."""
    return n * (n_bits + 1) / 2**n_bits


def load_table(
    ns: tuple[int, ...] = (64, 256, 1024, 4096, 8192),
) -> list[Table]:
    tables = []
    for sigma in (0.0, 0.3):
        series = {"own address": [], "verified r1": []}
        for n in ns:
            series["own address"].append(error(read_own, n, sigma))
            series["verified r1"].append(error(read_ball, n, sigma))
        tables.append(
            Table(
                title=(
                    f"Load, sigma={sigma} (d={D}, m={M}, n={NB}, "
                    f"lambda={LAMBDA}, write radius 1)"
                ),
                x_label="associations N",
                x=list(ns),
                series=series,
            )
        )
    return tables


def noise_table(n: int = 64) -> Table:
    sigmas = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    readers = {
        "own address": read_own,
        "verified r1": read_ball,
        "verified r2": read_ball2,
        "guided6": read_guided,
    }
    series = {
        name: [error(reader, n, sigma) for sigma in sigmas]
        for name, reader in readers.items()
    }
    return Table(
        title=f"Noisy cues, N={n} (d={D}, m={M}, n={NB}, lambda={LAMBDA})",
        x_label="key noise sigma",
        x=list(sigmas),
        series=series,
    )


def lambda_table(
    lams: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001), n: int = 1024
) -> Table:
    series = {
        "own sigma0": [],
        "verified r1 sigma0": [],
        "own sigma0.3": [],
        "verified r1 sigma0.3": [],
    }
    for lam in lams:
        series["own sigma0"].append(error(read_own, n, 0.0, rr_lambda=lam))
        series["verified r1 sigma0"].append(
            error(read_ball, n, 0.0, rr_lambda=lam)
        )
        series["own sigma0.3"].append(error(read_own, n, 0.3, rr_lambda=lam))
        series["verified r1 sigma0.3"].append(
            error(read_ball, n, 0.3, rr_lambda=lam)
        )
    return Table(
        title=f"Ridge lambda, N={n} (d={D}, m={M}, n={NB})",
        x_label="rr_lambda",
        x=list(lams),
        series=series,
    )


def read_table(n: int = 1024) -> Table:
    """Write replication versus read probes, at two noise levels."""
    variants: dict[str, Callable[[int, float], float]] = {
        "w1 r0": lambda nn, s: error(read_own, nn, s),
        "w1 r1": lambda nn, s: error(read_ball, nn, s),
        "w1 r2": lambda nn, s: error(read_ball2, nn, s),
        "w1 guided6": lambda nn, s: error(read_guided, nn, s),
    }
    series = {name: [run(n, sigma) for sigma in (0.0, 0.3)] for name, run in variants.items()}
    return Table(
        title=f"Read strategy at N={n}, one pass, two noise levels",
        x_label="strategy",
        x=["sigma 0", "sigma 0.3"],
        series=series,
    )


def measure_cost(
    n: int = 1024,
    batch: int = 64,
    passes: int = 3,
) -> dict[str, object]:
    """Median microseconds per write and read, plus resident state."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        keys, values = pairs(n, 0)
        memory = build(M, NB, 0)
        for start in range(0, n, 128):
            memory.store(keys[start : start + 128], values[start : start + 128])
        query = keys[:64]
        memory.read_ball(query)
        read_times = []
        for _ in range(passes):
            start = time.perf_counter()
            memory.read_ball(query)
            read_times.append((time.perf_counter() - start) * 1e6 / query.shape[0])
        own_times = []
        for _ in range(passes):
            start = time.perf_counter()
            memory.read_blocks(query)
            own_times.append((time.perf_counter() - start) * 1e6 / query.shape[0])
        write_times = []
        for repeat in range(passes):
            fresh = build(M, NB, repeat)
            start = time.perf_counter()
            for start_row in range(0, n, batch):
                fresh.store(
                    keys[start_row : start_row + batch],
                    values[start_row : start_row + batch],
                )
            write_times.append((time.perf_counter() - start) * 1e6 / n)
        state = sum(
            buffer.numel() * buffer.element_size() for buffer in memory.buffers()
        )
        return {
            "n_bits": M,
            "n_blocks": NB,
            "key_dim": D,
            "value_dim": DV,
            "n": n,
            "batch": batch,
            "write_us_per_pair": median(write_times),
            "read_own_us_per_query": median(own_times),
            "read_ball_us_per_query": median(read_times),
            "state_mib": state / 2**20,
            "writes_per_pair": NB * (M + 1),
            "probes_per_query": NB * (M + 1),
            "rounds_note": "duplicate write addresses are applied in sequence",
        }
    finally:
        torch.set_num_threads(threads)


def prior_tables() -> list[Table]:
    """The archived mean-cell numbers, if the baseline run is present."""
    if not PRIOR_PATH.exists():
        return []
    archived = json.loads(PRIOR_PATH.read_text())
    tables = []
    for table in archived["tables"]:
        if table["title"].startswith("Capacity, one pass"):
            x = table["x"]
            keep = [i for i, n in enumerate(x) if 256 <= n <= 1024]
            if not keep:
                continue
            tables.append(
                Table(
                    title="Archived mean-cell implementation (verified r1, clean)",
                    x_label="associations N",
                    x=[x[i] for i in keep],
                    series={
                        "verified r1 (prior)": [
                            table["series"]["verified r1"][i] for i in keep
                        ]
                    },
                )
            )
    return tables


def format_table(table: Table) -> str:
    names = list(table.series)
    lines = [
        f"**{table.title}**",
        "",
        "| " + table.x_label + " | " + " | ".join(names) + " |",
        "|" + "---|" * (len(names) + 1),
    ]
    for i, x in enumerate(table.x):
        label = f"{x:g}" if isinstance(x, float) else str(x)
        cells = [f"{table.series[name][i]:.2e}" for name in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    tables = [*load_table(), noise_table(), read_table(), lambda_table(), *prior_tables()]
    for table in tables:
        print(format_table(table))
        print()
    cost = measure_cost()
    print(
        f"**Cost (d={cost['key_dim']}, m={cost['n_bits']}, n={cost['n_blocks']}, "
        f"{cost['n']} pairs, one CPU thread)**"
    )
    print()
    print("| write us/pair | read own us/query | read verified r1 us/query | state MiB |")
    print("|---|---|---|---|")
    print(
        f"| {cost['write_us_per_pair']:.1f} | {cost['read_own_us_per_query']:.1f} "
        f"| {cost['read_ball_us_per_query']:.1f} | {cost['state_mib']:.0f} |"
    )
    print()
    print(
        f"Writes per pair: {cost['writes_per_pair']}; "
        f"probes per query: {cost['probes_per_query']}"
    )
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
                    "sample_per_seed": SAMPLE,
                    "write_radius": 1,
                },
                "tables": [asdict(table) for table in tables],
                "cost": cost,
            },
            indent=2,
        )
    )
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
