"""Wall-clock fit time for every method on the ten-part stream.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.training_cost

Protocol: CPU, float32, four threads, seed 0, three fresh fits per method, and
training only (the ten ``train_task`` calls). Data generation, model
construction, and the 10 x 10 evaluation are excluded. The reported time is the
median of the three fits; a single cold run is often slower, so use repeated
measurements when comparing machines.

The first table is the continual-learning protocol: each method sees the parts
in order, and the MLP methods take a fixed 300 steps per part (the joint
reference takes 800). The second table is a different baseline: one MLP trained
from scratch on all 5,120 pairs pooled, with no task boundaries and no
forgetting. It is the offline oracle that has all the data, not a continual
learner.
"""

from __future__ import annotations

import json
import platform
import time
from math import comb
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

from pt_neural_memory.benchmarks.forgetting_bench import (
    DEFAULT_PROTOCOL,
    LEARNERS,
    MEMORY_OPTIONS,
    MLP,
    r2_score,
)
from pt_neural_memory.forgetting_data import regression_stream

SEED = 0
REPEATS = 3
METHODS: tuple[str, ...] = ("naive", "ewc", "si", "replay", "joint", "addressed")
FROM_SCRATCH_STEPS: tuple[int, ...] = (300, 1000, 3000, 8000, 16000)
THREADS = 4
RESULTS_PATH = Path("results/training_cost.json")


def fit_time(method: str, stream, seed: int) -> tuple[float, object]:
    """Seconds to train one method on one stream, and the fitted learner."""
    dim = stream[0].train_inputs.shape[1]
    options = MEMORY_OPTIONS if method == "addressed" else {}
    learner = LEARNERS[method](dim, seed, **options)
    start = time.perf_counter()
    for task in stream:
        learner.train_task(task)
    return time.perf_counter() - start, learner


def resident_state(learner: object) -> int:
    """Bytes of the learned/persistent state, including the addressed table."""
    model = getattr(learner, "model", None)
    if model is not None and hasattr(model, "state_bytes"):
        return int(model.state_bytes())
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in learner.params
        )
    )


def train_from_scratch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    stream,
    steps: int,
    seed: int,
) -> tuple[float, float, list[float]]:
    """Fit one MLP on the pooled pairs and return seconds, ACC and per-part R^2."""
    model = MLP(
        inputs.shape[1],
        DEFAULT_PROTOCOL.hidden,
        DEFAULT_PROTOCOL.depth,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_PROTOCOL.lr)
    generator = torch.Generator().manual_seed(seed + 10_007)
    start = time.perf_counter()
    for _ in range(steps):
        indices = torch.randint(
            inputs.shape[0], (DEFAULT_PROTOCOL.batch_size,), generator=generator
        )
        loss = F.mse_loss(model(inputs[indices]), targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        scores = [
            r2_score(model(task.test_inputs), task.test_targets) for task in stream
        ]
    return elapsed, sum(scores) / len(scores), scores


def main() -> None:
    torch.set_num_threads(THREADS)
    stream = regression_stream(seed=SEED)
    rows = []
    for method in METHODS:
        times = []
        state_bytes = 0
        for _ in range(REPEATS):
            elapsed, learner = fit_time(method, stream, SEED)
            times.append(elapsed)
            state_bytes = resident_state(learner)
        rows.append(
            {
                "method": method,
                "median_seconds": median(times),
                "min_seconds": min(times),
                "per_part_ms": 1000.0 * median(times) / len(stream),
                "runs": times,
                "state_bytes": state_bytes,
            }
        )

    print("| method | fit median (s) | per part (ms) | model state |")
    print("|---|---:|---:|---:|")
    for row in rows:
        state_mib = row["state_bytes"] / 2**20
        state = (
            f"{row['state_bytes'] / 1024:.1f} KiB"
            if state_mib < 1
            else f"{state_mib:.1f} MiB"
        )
        print(
            f"| {row['method']} | {row['median_seconds']:.2f} "
            f"| {row['per_part_ms']:.0f} | {state} |"
        )

    inputs = torch.cat([task.train_inputs for task in stream])
    targets = torch.cat([task.train_targets for task in stream])
    scratch_rows = []
    print("\nFrom-scratch MLP on all parts pooled (no task boundaries):")
    print("\n| steps | fit median (s) | ACC |")
    print("|---:|---:|---:|")
    for steps in FROM_SCRATCH_STEPS:
        times = []
        scores: list[float] = []
        for _ in range(REPEATS):
            elapsed, acc, scores = train_from_scratch(
                inputs, targets, stream, steps, SEED
            )
            times.append(elapsed)
        acc = sum(scores) / len(scores)
        scratch_rows.append(
            {
                "steps": steps,
                "median_seconds": median(times),
                "min_seconds": min(times),
                "runs": times,
                "acc": acc,
                "scores": scores,
            }
        )
        print(f"| {steps:,} | {median(times):.2f} | {acc:+.3f} |")

    n_tasks = len(stream)
    stored_pairs = int(inputs.shape[0])
    writes_per_pair = int(MEMORY_OPTIONS["n_blocks"]) * sum(
        comb(int(MEMORY_OPTIONS["n_bits"]), j)
        for j in range(int(MEMORY_OPTIONS["write_radius"]) + 1)
    )
    print(
        f"\n{n_tasks} parts, {stored_pairs:,} stored pairs, "
        f"{writes_per_pair} cell writes per pair, "
        f"{torch.get_num_threads()} threads, torch {torch.__version__}."
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "repeats": REPEATS,
                "threads": torch.get_num_threads(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "n_tasks": n_tasks,
                "stored_pairs": stored_pairs,
                "writes_per_pair": writes_per_pair,
                "rows": rows,
                "from_scratch": scratch_rows,
            },
            indent=2,
        )
    )
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
