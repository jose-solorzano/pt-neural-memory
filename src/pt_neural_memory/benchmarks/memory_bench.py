"""Benchmark suite for the recursive-least-squares memory cell.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.memory_bench

Every task writes a stream of associations into the cell and reports the
relative squared error (RSE) of the read path.  RSE is
``||pred - value||² / ||value||²``: 0 is perfect recall and 1 is what a cell
that always answers zero scores.

The tasks characterize the properties an associative memory is supposed to
have:

* ``capacity``: one pass over ``α * d`` random pairs, then recall all of them.
* ``forgetting``: recall set A after set B has been written.
* ``correlated_keys``: keys share a direction, so writes interfere.
* ``noise``: recall from noisy cues.
* ``drift``: values rotate every epoch, so the stream is non-stationary.

Results are printed as Markdown and written to ``results/memory_bench.json``.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import torch

from pt_neural_memory.mem_rls import RLSMemCell

D = 64
M = 64
SEEDS = (0, 1, 2, 3, 4)
RESULTS_PATH = Path("results/memory_bench.json")


@dataclass
class Table:
    """One sweep: an x-axis (numeric or categorical) and one series per cell."""

    title: str
    x_label: str
    x: list[float | str]
    series: dict[str, list[float]]


def random_pairs(
    n: int, d: int = D, m: int = M, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``n`` i.i.d. Gaussian pairs in a random order."""
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randn(n, d, generator=gen)
    values = torch.randn(n, m, generator=gen)
    order = torch.randperm(n, generator=gen)
    return keys[order], values[order]


def make_cell(
    d: int = D, m: int = M, rr_lambda: float = 1.0, decay: float = 1.0
) -> RLSMemCell:
    """Build an empty cell; the constructor starts the map at zero."""
    return RLSMemCell(d, m, rr_lambda=rr_lambda, decay=decay)


def relative_squared_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of target energy left unexplained (SST around zero)."""
    return (pred - target).square().sum().item() / target.square().sum().item()


def write_stream(cell: RLSMemCell, keys: torch.Tensor, values: torch.Tensor) -> None:
    """Feed one pair at a time, the online protocol the cell targets."""
    for key, value in zip(keys, values):
        cell.store(key.unsqueeze(0), value.unsqueeze(0))


def recall_error(
    cell: RLSMemCell, keys: torch.Tensor, values: torch.Tensor
) -> float:
    """RSE of ``cell`` on the given associations."""
    return relative_squared_error(cell(keys), values)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def capacity_table(
    alphas: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    seeds: tuple[int, ...] = SEEDS,
) -> Table:
    """One pass over ``α * d`` pairs, then recall of the whole stream."""
    errors = []
    for alpha in alphas:
        n = max(1, round(alpha * D))
        runs = []
        for seed in seeds:
            keys, values = random_pairs(n, seed=seed)
            cell = make_cell()
            write_stream(cell, keys, values)
            runs.append(recall_error(cell, keys, values))
        errors.append(mean(runs))
    return Table(
        title=f"Capacity, one pass (d={D}, m={M}, lambda=1, {len(seeds)} seeds)",
        x_label="load alpha = n/d",
        x=list(alphas),
        series={"rls": errors},
    )


def forgetting_table(
    n_a: int = 16, n_b: int = 16, seeds: tuple[int, ...] = SEEDS
) -> Table:
    """Write A, measure it; write B; measure A and B again."""
    runs = {"before": [], "after": [], "new": []}
    for seed in seeds:
        keys_a, values_a = random_pairs(n_a, seed=seed)
        keys_b, values_b = random_pairs(n_b, seed=seed + 100)
        cell = make_cell()
        write_stream(cell, keys_a, values_a)
        runs["before"].append(recall_error(cell, keys_a, values_a))
        write_stream(cell, keys_b, values_b)
        runs["after"].append(recall_error(cell, keys_a, values_a))
        runs["new"].append(recall_error(cell, keys_b, values_b))
    return Table(
        title=f"Retention across a task boundary (n={n_a}+{n_b}, {len(seeds)} seeds)",
        x_label="phase",
        x=["A before B", "A after B", "B after B"],
        series={"rls": [mean(values) for values in runs.values()]},
    )


def correlated_keys_table(
    rhos: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 0.99),
    n: int = 32,
    seeds: tuple[int, ...] = SEEDS,
) -> Table:
    """Keys are ``√ρ c + √(1-ρ) ε`` rescaled to a constant norm."""
    errors = []
    for rho in rhos:
        runs = []
        for seed in seeds:
            gen = torch.Generator().manual_seed(seed)
            direction = torch.randn(1, D, generator=gen)
            direction = direction / direction.norm()
            noise = torch.randn(n, D, generator=gen)
            keys = rho**0.5 * direction + (1.0 - rho) ** 0.5 * noise
            keys = keys * (D**0.5 / keys.norm(dim=1, keepdim=True))
            values = torch.randn(n, M, generator=gen)
            cell = make_cell()
            write_stream(cell, keys, values)
            runs.append(recall_error(cell, keys, values))
        errors.append(mean(runs))
    return Table(
        title=f"Correlated keys (n={n}, d={D}, lambda=1, {len(seeds)} seeds)",
        x_label="key correlation rho",
        x=list(rhos),
        series={"rls": errors},
    )


def noise_table(
    sigmas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    n: int = 32,
    seeds: tuple[int, ...] = SEEDS,
) -> Table:
    """Recall from cues corrupted by ``σ * N(0, I)``."""
    runs = {sigma: [] for sigma in sigmas}
    for seed in seeds:
        keys, values = random_pairs(n, seed=seed)
        noise_gen = torch.Generator().manual_seed(seed + 7)
        noise = torch.randn(keys.shape, generator=noise_gen)
        cell = make_cell()
        write_stream(cell, keys, values)
        for sigma in sigmas:
            runs[sigma].append(recall_error(cell, keys + sigma * noise, values))
    return Table(
        title=f"Noisy cues (n={n}, d={D}, lambda=1, {len(seeds)} seeds)",
        x_label="cue noise sigma",
        x=list(sigmas),
        series={"rls": [mean(runs[sigma]) for sigma in sigmas]},
    )


def drift_table(
    n: int = 32,
    epochs: int = 4,
    decays: tuple[float, ...] = (1.0, 0.98, 0.95),
    seeds: tuple[int, ...] = SEEDS,
) -> Table:
    """Values are rotated by an orthogonal map after every epoch."""
    names = [f"rls decay={decay:g}" for decay in decays]
    errors: dict[str, list[list[float]]] = {name: [] for name in names}
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        keys = torch.randn(n, D, generator=gen)
        values = torch.randn(n, M, generator=gen)
        rotation, _ = torch.linalg.qr(torch.randn(M, M, generator=gen))
        cells = [make_cell(decay=decay) for decay in decays]
        per_seed = {name: [] for name in names}
        for _ in range(epochs):
            for name, cell in zip(names, cells):
                write_stream(cell, keys, values)
                per_seed[name].append(recall_error(cell, keys, values))
            values = values @ rotation
        for name in names:
            errors[name].append(per_seed[name])
    return Table(
        title=f"Drift: rotating values (n={n}, {epochs} epochs, {len(seeds)} seeds)",
        x_label="epoch",
        x=list(range(1, epochs + 1)),
        series={
            name: [
                sum(run[e] for run in errors[name]) / len(seeds)
                for e in range(epochs)
            ]
            for name in names
        },
    )


def measure_cost(
    d: int = 256,
    m: int = 256,
    batch_sizes: tuple[int, ...] = (1, 16),
    passes: int = 10,
    repeats: int = 3,
) -> dict[str, object]:
    """Report median microseconds per write and buffer size in KiB."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        timing = {}
        for batch in batch_sizes:
            times = []
            for repeat in range(repeats):
                gen = torch.Generator().manual_seed(repeat)
                keys = torch.randn(batch, d, generator=gen)
                values = torch.randn(batch, m, generator=gen)
                cell = make_cell(d, m)
                cell.store(keys, values)  # warm-up
                start = time.perf_counter()
                for _ in range(passes):
                    cell.store(keys, values)
                times.append((time.perf_counter() - start) / passes * 1e6)
            timing[f"batch={batch}"] = median(times)
        cell = make_cell(d, m)
        state_kib = sum(
            buf.numel() * buf.element_size() for buf in cell.buffers()
        ) / 1024
    finally:
        torch.set_num_threads(threads)
    return {"d": d, "m": m, "write_us": timing, "state_kib": state_kib}


def format_table(table: Table) -> str:
    """Render a table as Markdown."""
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
    tables = [
        capacity_table(),
        forgetting_table(),
        correlated_keys_table(),
        noise_table(),
        drift_table(),
    ]
    cost = measure_cost()
    for table in tables:
        print(format_table(table))
        print()
    print(f"**Write cost (d={cost['d']}, m={cost['m']}, 1 CPU thread)**")
    print()
    print("| batch | rls us/write |")
    print("|---|---|")
    for batch, microseconds in cost["write_us"].items():
        print(f"| {batch} | {microseconds:.1f} |")
    print()
    print(f"State: {cost['state_kib']:.0f} KiB")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"tables": [asdict(table) for table in tables], "cost": cost},
            indent=2,
        )
    )
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
