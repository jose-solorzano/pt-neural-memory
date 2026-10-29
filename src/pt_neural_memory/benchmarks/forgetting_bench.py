"""Catastrophic forgetting on a ten-part regression stream.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.forgetting_bench

Scenario
--------
The model is a single MLP with one scalar output and **no task identifier**.
It is trained on part 0, then part 1, ..., and after each part it is evaluated
on every part, seen or not.  This is the domain-incremental scenario (van de
Ven & Tolias, 2020): the input distribution changes from part to part, the
task identity is unavailable at test time, and one set of weights has to serve
every part.

One method (``addressed``) is not an MLP: it is a table of RLS cells addressed
by random features of the input, fitted in closed form instead of by gradient
descent.  It is the data-side limit of the same question -- what if the model
keeps every pair it is given?  The other five methods are the MLP protocol.

Scores
------
Each entry of the accuracy matrix is R^2 on a held-out sample of that part,

    a[t][j] = 1 - MSE(predictions on part j after training part t) / Var(y_j),

so 0 is the score of a predictor that always answers the per-part mean and 1 is
perfect.  Because the test targets are noiseless (see
:mod:`pt_neural_memory.forgetting_data`), a value below the joint-training ceiling is
attributable to the learner, not to label noise.

From the matrix ``R`` with ``n`` parts and per-part random-init scores ``b``:

* ``ACC``  = mean_j R[n-1][j] -- final score, all parts.
* ``BWT``  = mean_{j<n-1} (R[n-1][j] - R[j][j]) -- negative means forgetting.
* ``LA``   = mean_j R[j][j] -- how well each part was learned when current.
* ``FWT``  = mean_{j>0} (R[j-1][j] - b[j]) -- zero-shot transfer to the next
  part, which is hopeless here: the parts have disjoint support.

The first three follow Lopez-Paz & Ranzato (2017) with accuracy replaced by
R^2.  The five MLP methods share one architecture, one optimiser and one step
budget, except the joint-training reference, which is an upper bound and is
given a larger budget.  The addressed memory has no optimiser.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from statistics import fmean
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pt_neural_memory.addressed_regressor import AddressedRegressor
from pt_neural_memory.forgetting_data import Task, regression_stream

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2)
RESULTS_PATH = Path("results/forgetting_bench.json")


@dataclass(frozen=True)
class Protocol:
    """Optimisation budget and architecture, shared by every method."""

    steps_per_task: int = 300
    batch_size: int = 128
    lr: float = 1e-3
    hidden: int = 64
    depth: int = 2


DEFAULT_PROTOCOL = Protocol()
# The joint reference sees all data and is only an upper bound, so it is given
# a converged fit rather than the sequential budget.
JOINT_PROTOCOL = Protocol(steps_per_task=800, lr=3e-3)


class MLP(nn.Module):
    """ReLU regressor with ``depth`` hidden layers and one scalar output.

    Initialisation is drawn from ``generator`` (uniform Glorot-style bound,
    zero bias) so that every method and every seed starts from the same
    weights regardless of other random draws.
    """

    def __init__(
        self,
        inputs: int,
        hidden: int,
        depth: int,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = inputs
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                bound = layer.in_features**-0.5
                with torch.no_grad():
                    layer.weight.copy_(
                        (2.0 * torch.rand(layer.weight.shape, generator=generator) - 1.0)
                        * bound
                    )
                    layer.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Learner:
    """What the benchmark needs from a method: learn a part, predict inputs.

    ``SequentialLearner`` implements it by gradient descent on one MLP;
    ``MemoryLearner`` implements it by storing each part in an addressed memory.
    """

    name = "learner"

    def train_task(self, task: Task) -> None:
        raise NotImplementedError

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SequentialLearner(Learner):
    """Fine-tunes one MLP on the parts in order; subclasses add a defence.

    Subclasses override the hooks: ``_batch`` (what a step is trained on),
    ``_penalty`` (an extra loss term), ``_begin_task`` / ``_end_task`` (state
    consolidation), and ``_after_step`` (per-step state, used by SI).
    """

    name = "naive"
    tracks_path = False

    def __init__(
        self, dim: int, seed: int, protocol: Protocol = DEFAULT_PROTOCOL
    ) -> None:
        self.protocol = protocol
        self.model = MLP(
            dim,
            protocol.hidden,
            protocol.depth,
            generator=torch.Generator().manual_seed(seed),
        )
        self.params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=protocol.lr)
        self._generator = torch.Generator().manual_seed(seed + 10_007)

    def train_task(self, task: Task) -> None:
        """Fit the current part for the protocol's step budget."""
        self._begin_task(task)
        for _ in range(self.protocol.steps_per_task):
            inputs, targets = self._batch(task)
            loss = F.mse_loss(self.model(inputs), targets) + self._penalty()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            before = self._snapshot() if self.tracks_path else None
            grads = (
                [parameter.grad.detach().clone() for parameter in self.params]
                if before is not None
                else None
            )
            self.optimizer.step()
            if before is not None and grads is not None:
                self._after_step(before, grads)
        self._end_task(task)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predictions for ``inputs`` with the model in eval mode."""
        self.model.eval()
        predictions = self.model(inputs)
        self.model.train()
        return predictions

    def _batch(self, task: Task) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sample(task, self.protocol.batch_size)

    def _sample(self, task: Task, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            task.train_inputs.shape[0], (n,), generator=self._generator
        )
        return task.train_inputs[indices], task.train_targets[indices]

    def _snapshot(self) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in self.params]

    def _penalty(self) -> torch.Tensor:
        return torch.zeros(())

    def _begin_task(self, task: Task) -> None:
        pass

    def _end_task(self, task: Task) -> None:
        pass

    def _after_step(
        self, before: list[torch.Tensor], grads: list[torch.Tensor]
    ) -> None:
        pass


class EWCLearner(SequentialLearner):
    """Elastic weight consolidation: diagonal-Fisher quadratic anchors.

    One anchor is kept per past part, with the Fisher estimated on that part's
    training set by averaging squared mini-batch gradients.
    """

    name = "ewc"

    def __init__(
        self,
        dim: int,
        seed: int,
        protocol: Protocol = DEFAULT_PROTOCOL,
        strength: float = 100.0,
        fisher_batches: int = 16,
    ) -> None:
        super().__init__(dim, seed, protocol)
        self.strength = strength
        self.fisher_batches = fisher_batches
        self._anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        self._fisher_generator = torch.Generator().manual_seed(seed + 20_011)

    def _penalty(self) -> torch.Tensor:
        total = torch.zeros(())
        for params, fisher in self._anchors:
            for parameter, weight, anchor in zip(self.params, fisher, params):
                total = total + (weight * (parameter - anchor).square()).sum()
        return 0.5 * self.strength * total

    def _end_task(self, task: Task) -> None:
        self._anchors.append((self._snapshot(), self._fisher(task)))

    def _fisher(self, task: Task) -> list[torch.Tensor]:
        fisher = [torch.zeros_like(parameter) for parameter in self.params]
        for _ in range(self.fisher_batches):
            indices = torch.randint(
                task.train_inputs.shape[0], (64,), generator=self._fisher_generator
            )
            loss = F.mse_loss(
                self.model(task.train_inputs[indices]), task.train_targets[indices]
            )
            grads = torch.autograd.grad(loss, self.params)
            for estimate, gradient in zip(fisher, grads):
                estimate += gradient.square() / self.fisher_batches
        return fisher


class SILearner(SequentialLearner):
    """Synaptic intelligence: online path-integral importance per parameter."""

    name = "si"
    tracks_path = True

    def __init__(
        self,
        dim: int,
        seed: int,
        protocol: Protocol = DEFAULT_PROTOCOL,
        strength: float = 1.0,
        damping: float = 0.1,
    ) -> None:
        super().__init__(dim, seed, protocol)
        self.strength = strength
        self.damping = damping
        self._path: list[torch.Tensor] = []
        self._start: list[torch.Tensor] = []
        self._anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    def _begin_task(self, task: Task) -> None:
        self._path = [torch.zeros_like(parameter) for parameter in self.params]
        self._start = self._snapshot()

    def _after_step(
        self, before: list[torch.Tensor], grads: list[torch.Tensor]
    ) -> None:
        # Path integral of the loss along the task's trajectory: omega += -g * dtheta.
        for omega, gradient, previous, parameter in zip(
            self._path, grads, before, self.params
        ):
            omega -= gradient * (parameter.detach() - previous)

    def _end_task(self, task: Task) -> None:
        importance = []
        for omega, start, parameter in zip(self._path, self._start, self.params):
            displacement = (parameter.detach() - start).square()
            importance.append(omega.clamp(min=0.0) / (displacement + self.damping))
        self._anchors.append((self._snapshot(), importance))

    def _penalty(self) -> torch.Tensor:
        total = torch.zeros(())
        for params, importance in self._anchors:
            for parameter, weight, anchor in zip(self.params, importance, params):
                total = total + (weight * (parameter - anchor).square()).sum()
        return 0.5 * self.strength * total


class ReplayLearner(SequentialLearner):
    """Experience replay: half of every batch is drawn from a fixed store.

    ``buffer_per_task`` samples of each part are retained (a uniform subsample,
    not reservoir sampling, because the stream is fixed).
    """

    name = "replay"

    def __init__(
        self,
        dim: int,
        seed: int,
        protocol: Protocol = DEFAULT_PROTOCOL,
        buffer_per_task: int = 32,
    ) -> None:
        super().__init__(dim, seed, protocol)
        self.buffer_per_task = buffer_per_task
        self._buffer_inputs = torch.empty(0, dim)
        self._buffer_targets = torch.empty(0, 1)

    def _batch(self, task: Task) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.protocol.batch_size // 2
        if len(self._buffer_inputs) == 0:
            return self._sample(task, self.protocol.batch_size)
        current_inputs, current_targets = self._sample(task, half)
        indices = torch.randint(
            len(self._buffer_inputs), (half,), generator=self._generator
        )
        return (
            torch.cat([current_inputs, self._buffer_inputs[indices]]),
            torch.cat([current_targets, self._buffer_targets[indices]]),
        )

    def _end_task(self, task: Task) -> None:
        count = min(self.buffer_per_task, task.train_inputs.shape[0])
        indices = torch.randperm(
            task.train_inputs.shape[0], generator=self._generator
        )[:count]
        self._buffer_inputs = torch.cat(
            [self._buffer_inputs, task.train_inputs[indices]]
        )
        self._buffer_targets = torch.cat(
            [self._buffer_targets, task.train_targets[indices]]
        )


class JointLearner(SequentialLearner):
    """Train on every part seen so far: the multi-task upper bound.

    This is replay with an unbounded store.  It is not a continual-learning
    method, because it is allowed to re-read all past data.
    """

    name = "joint"

    def __init__(
        self, dim: int, seed: int, protocol: Protocol = JOINT_PROTOCOL
    ) -> None:
        super().__init__(dim, seed, protocol)
        self._pool_inputs = torch.empty(0, dim)
        self._pool_targets = torch.empty(0, 1)

    def _begin_task(self, task: Task) -> None:
        self._pool_inputs = torch.cat([self._pool_inputs, task.train_inputs])
        self._pool_targets = torch.cat([self._pool_targets, task.train_targets])

    def _batch(self, task: Task) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            len(self._pool_inputs),
            (self.protocol.batch_size,),
            generator=self._generator,
        )
        return self._pool_inputs[indices], self._pool_targets[indices]


class MemoryLearner(Learner):
    """Addressed memory on random features: a store, not a gradient-trained net.

    Every pair of a part is written once into a table of RLS cells, addressed by
    random features of the input.  There is no optimiser and no shared parameter
    for a later part to overwrite, so the finished model contains every part it
    has seen.  The price is that it stores the data -- as local maps, each with
    a dense covariance -- rather than compressing it into a few thousand
    weights.

    ``samples_per_task`` keeps only that many rows of each part, which makes the
    memory directly comparable with the replay store: the same question, how
    many points per part are enough, with the readout fitted in closed form
    instead of by gradient descent.
    """

    name = "addressed"

    def __init__(
        self,
        dim: int,
        seed: int,
        protocol: Protocol = DEFAULT_PROTOCOL,
        samples_per_task: int | None = None,
        **options: object,
    ) -> None:
        if samples_per_task is not None and samples_per_task < 1:
            raise ValueError(
                f"samples_per_task must be positive or None, got {samples_per_task}"
            )
        self.protocol = protocol
        self.samples_per_task = samples_per_task
        self.model = AddressedRegressor(dim, 1, seed=seed, **options)
        self._generator = torch.Generator().manual_seed(seed + 30_013)
        self.stored_pairs = 0

    def train_task(self, task: Task) -> None:
        inputs, targets = task.train_inputs, task.train_targets
        if self.samples_per_task is not None:
            count = min(self.samples_per_task, inputs.shape[0])
            indices = torch.randperm(
                inputs.shape[0], generator=self._generator
            )[:count]
            inputs, targets = inputs[indices], targets[indices]
        self.model.store(inputs, targets)
        self.stored_pairs += inputs.shape[0]

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


MEMORY_OPTIONS: dict[str, object] = {
    "kind": "rff",
    "width": 64,
    "scale": 2.0,
    "n_blocks": 3,
    "n_bits": 10,
    "write_radius": 1,
    "rr_lambda": 0.01,
    "samples_per_task": None,
}
"""The addressed-memory configuration, mirroring ``MemoryLearner``'s defaults."""

METHODS: tuple[str, ...] = ("naive", "ewc", "si", "replay", "joint", "addressed")

LEARNERS: dict[str, type[Learner]] = {
    "naive": SequentialLearner,
    "ewc": EWCLearner,
    "si": SILearner,
    "replay": ReplayLearner,
    "joint": JointLearner,
    "addressed": MemoryLearner,
}


@dataclass
class Run:
    """One method trained on one stream, with the full accuracy matrix."""

    method: str
    seed: int
    matrix: list[list[float]]
    baseline: list[float]


@dataclass
class Aggregate:
    """Several runs of one method, averaged cell by cell."""

    method: str
    matrix: list[list[float]]
    baseline: list[float]
    summary: dict[str, float]
    spread: dict[str, float]


@dataclass
class Sweep:
    """One x-axis, one series per method, several metrics per series."""

    title: str
    x_label: str
    x: list[str]
    series: dict[str, dict[str, list[float]]]


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Coefficient of determination; 0 is the per-part mean predictor."""
    variance = targets.var(unbiased=False)
    if variance <= 0:
        raise ValueError("targets have no variance, so R^2 is undefined")
    return float(1.0 - (predictions - targets).square().mean() / variance)


def summarize(matrix: Sequence[Sequence[float]], baseline: Sequence[float]) -> dict[str, float]:
    """ACC, BWT, LA and FWT from one accuracy matrix."""
    n = len(matrix)
    final = matrix[n - 1]
    return {
        "acc": fmean(final),
        "bwt": fmean(final[j] - matrix[j][j] for j in range(n - 1)),
        "la": fmean(matrix[j][j] for j in range(n)),
        "fwt": fmean(matrix[j - 1][j] - baseline[j] for j in range(1, n)),
    }


def benchmark(
    method: str,
    seed: int,
    stream_options: dict[str, object] | None = None,
    task_order: Sequence[int] | None = None,
    **method_options: object,
) -> Run:
    """Train ``method`` on one stream and return its accuracy matrix."""
    if method not in LEARNERS:
        raise ValueError(
            f"unknown method {method!r}, expected one of {sorted(LEARNERS)}"
        )
    tasks = regression_stream(seed=seed, **(stream_options or {}))
    if task_order is not None:
        tasks = [tasks[index] for index in task_order]
    dim = tasks[0].train_inputs.shape[1]
    learner = LEARNERS[method](dim, seed, **method_options)
    baseline = [r2_score(learner.predict(t.test_inputs), t.test_targets) for t in tasks]
    n = len(tasks)
    matrix = [[0.0] * n for _ in range(n)]
    for t, task in enumerate(tasks):
        learner.train_task(task)
        for j, other in enumerate(tasks):
            matrix[t][j] = r2_score(
                learner.predict(other.test_inputs), other.test_targets
            )
    return Run(method=method, seed=seed, matrix=matrix, baseline=baseline)


def aggregate(runs: Sequence[Run]) -> Aggregate:
    """Average runs cell by cell and average their summary metrics."""
    n = len(runs[0].matrix)
    matrix = [
        [fmean(run.matrix[t][j] for run in runs) for j in range(n)] for t in range(n)
    ]
    baseline = [fmean(run.baseline[j] for run in runs) for j in range(n)]
    per_run = [summarize(run.matrix, run.baseline) for run in runs]
    summary = {key: fmean(run[key] for run in per_run) for key in per_run[0]}
    spread = {
        key: (
            (sum((run[key] - summary[key]) ** 2 for run in per_run) / (len(per_run) - 1))
            ** 0.5
            if len(per_run) > 1
            else float("nan")
        )
        for key in per_run[0]
    }
    return Aggregate(
        method=runs[0].method,
        matrix=matrix,
        baseline=baseline,
        summary=summary,
        spread=spread,
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def format_summary(aggregates: Sequence[Aggregate]) -> str:
    rows = []
    for result in aggregates:
        summary = result.summary
        rows.append(
            [
                result.method,
                f"{summary['acc']:.3f}",
                f"{summary['bwt']:+.3f}",
                f"{summary['la']:.3f}",
                f"{summary['fwt']:+.3f}",
            ]
        )
    return markdown_table(["method", "ACC", "BWT", "LA", "FWT"], rows)


def format_matrix(matrix: Sequence[Sequence[float]], title: str) -> str:
    """Render an accuracy matrix; rows are the stage, columns the part."""
    n = len(matrix)
    headers = ["after / part"] + [f"t{j}" for j in range(n)]
    rows = [[f"t{t}"] + [f"{matrix[t][j]:+.2f}" for j in range(n)] for t in range(n)]
    return f"**{title}**\n\n" + markdown_table(headers, rows)


def format_sweep(sweep: Sweep, metrics: Sequence[str] = ("acc", "bwt")) -> str:
    methods = list(sweep.series)
    headers = [sweep.x_label] + [
        f"{method} {metric}" for method in methods for metric in metrics
    ]
    rows = []
    for index, label in enumerate(sweep.x):
        cells = [label]
        for method in methods:
            for metric in metrics:
                value = sweep.series[method][metric][index]
                cells.append(f"{value:+.3f}" if metric != "la" else f"{value:.3f}")
        rows.append(cells)
    return markdown_table(headers, rows)


def main_experiment(
    seeds: Sequence[int] = SEEDS,
    methods: Sequence[str] = METHODS,
    stream_options: dict[str, object] | None = None,
    task_order: Sequence[int] | None = None,
    method_options: dict[str, dict[str, object]] | None = None,
) -> dict[str, Aggregate]:
    """Run each method on each seed's stream."""
    method_options = method_options or {}
    aggregates = {}
    for method in methods:
        runs = [
            benchmark(
                method,
                seed,
                stream_options,
                task_order,
                **method_options.get(method, {}),
            )
            for seed in seeds
        ]
        aggregates[method] = aggregate(runs)
    return aggregates


def sweep_experiment(
    title: str,
    x_label: str,
    values: Sequence[object],
    build_method_options: Callable[[object], dict[str, object]],
    methods: Sequence[str],
    seeds: Sequence[int] = SWEEP_SEEDS,
    metric_names: Sequence[str] = ("acc", "bwt"),
    stream_options: dict[str, object] | None = None,
    protocol: Protocol = DEFAULT_PROTOCOL,
) -> Sweep:
    """Run several methods along one axis and collect their summary metrics.

    ``build_method_options(value)`` returns per-method keyword arguments for a
    value; it may also carry a stream override under the key ``"stream"``,
    which is popped and passed to the stream generator.
    """
    series: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in metric_names} for method in methods
    }
    for value in values:
        options = dict(build_method_options(value))
        overrides = options.pop("stream", {})
        merged = dict(stream_options or {})
        merged.update(overrides)
        method_options = {
            method: {**options.get(method, {}), "protocol": protocol}
            for method in methods
        }
        results = main_experiment(
            seeds,
            methods,
            merged,
            method_options=method_options,
        )
        for method in methods:
            for metric in metric_names:
                series[method][metric].append(results[method].summary[metric])
    return Sweep(
        title=title,
        x_label=x_label,
        x=[str(value) for value in values],
        series=series,
    )


def separation_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """How much does raising the parts' overlap reduce forgetting?

    ``center_radius`` shrinks the ring the part means sit on; at small radii
    the parts cover the same input region and the same stretch of the target.
    """
    return sweep_experiment(
        "Part overlap (smaller radius = more overlap)",
        "center_radius",
        (3.0, 1.6, 0.9, 0.6),
        lambda radius: {"stream": {"center_radius": float(radius)}},
        methods=("naive", "ewc", "replay"),
        seeds=seeds,
        protocol=protocol,
        stream_options=stream_options,
    )


def buffer_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """Replay store size, as samples kept per part out of 512."""
    return sweep_experiment(
        "Replay store size",
        "samples per part",
        (4, 8, 16, 32, 64, 128),
        lambda size: {"replay": {"buffer_per_task": int(size)}},
        methods=("replay",),
        seeds=seeds,
        protocol=protocol,
        stream_options=stream_options,
    )


def strength_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """How sensitive are the two parameter-space defences to their weight?"""
    return sweep_experiment(
        "Regularisation strength",
        "strength",
        (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
        lambda strength: {
            "ewc": {"strength": float(strength)},
            "si": {"strength": float(strength)},
        },
        methods=("ewc", "si"),
        seeds=seeds,
        protocol=protocol,
        stream_options=stream_options,
    )


def target_shift_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """Shared target versus a per-part rotated copy of it."""
    return sweep_experiment(
        "Target shift",
        "target",
        ("shared", "rotated"),
        lambda mode: {"stream": {"rotated_targets": mode == "rotated"}},
        methods=("naive", "ewc", "replay"),
        seeds=seeds,
        protocol=protocol,
        stream_options=stream_options,
    )


def order_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """Is the result an artefact of one particular part order?"""
    n = int((stream_options or {}).get("n_tasks", 10))
    shuffled = torch.randperm(
        n, generator=torch.Generator().manual_seed(0)
    ).tolist()
    orders = [
        ("forward", list(range(n))),
        ("reverse", list(range(n - 1, -1, -1))),
        ("shuffled", shuffled),
    ]
    series: dict[str, dict[str, list[float]]] = {
        method: {"acc": [], "bwt": []} for method in ("naive", "replay")
    }
    for _, order in orders:
        results = main_experiment(
            seeds,
            ("naive", "replay"),
            stream_options,
            task_order=order,
            method_options={
                method: {"protocol": protocol} for method in ("naive", "replay")
            },
        )
        for method in ("naive", "replay"):
            for metric in ("acc", "bwt"):
                series[method][metric].append(results[method].summary[metric])
    return Sweep(
        title="Part order",
        x_label="order",
        x=[label for label, _ in orders],
        series=series,
    )


def memory_capacity_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """Address width, and with it the table's state, against accuracy."""
    return sweep_experiment(
        "Memory size",
        "n_bits",
        (1, 4, 6, 8, 10, 12),
        lambda bits: {"addressed": {"n_bits": int(bits)}},
        methods=("addressed",),
        seeds=seeds,
        metric_names=("acc", "bwt", "la"),
        protocol=protocol,
        stream_options=stream_options,
    )


def memory_store_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """How many stored pairs per part does the memory need?"""
    return sweep_experiment(
        "Memory store size",
        "samples per part",
        (4, 8, 16, 32, 64, 128, 512),
        lambda size: {"addressed": {"samples_per_task": int(size)}},
        methods=("addressed",),
        seeds=seeds,
        metric_names=("acc", "bwt", "la"),
        protocol=protocol,
        stream_options=stream_options,
    )


MEMORY_DESIGN: dict[str, dict[str, object]] = {
    "default": {},
    "2 cells/block": {"n_bits": 1},
    "width 32": {"width": 32},
    "no raw input": {"use_raw": False},
    "lambda 0.1": {"rr_lambda": 0.1},
    "lambda 0.001": {"rr_lambda": 1e-3},
    "verified read": {"read": "verified"},
    "no write radius": {"write_radius": 0},
}


def memory_design_sweep(
    seeds: Sequence[int] = SWEEP_SEEDS,
    protocol: Protocol = DEFAULT_PROTOCOL,
    stream_options: dict[str, object] | None = None,
) -> Sweep:
    """One design choice per row, with the rest held at the default."""
    return sweep_experiment(
        "Memory design",
        "variant",
        tuple(MEMORY_DESIGN),
        lambda name: {"addressed": MEMORY_DESIGN[str(name)]},
        methods=("addressed",),
        seeds=seeds,
        metric_names=("acc", "bwt", "la"),
        protocol=protocol,
        stream_options=stream_options,
    )


def main() -> None:
    torch.set_num_threads(4)
    print("# Forgetting on the ten-part regression stream\n")

    main_results = main_experiment(method_options={"addressed": MEMORY_OPTIONS})
    print("## Main comparison (10 parts, 5 seeds)\n")
    print(format_summary(list(main_results.values())))
    for method in ("naive", "ewc", "replay", "joint", "addressed"):
        print()
        print(format_matrix(main_results[method].matrix, f"R^2 matrix, {method}"))

    sweeps = {
        "separation": separation_sweep(),
        "buffer": buffer_sweep(),
        "strength": strength_sweep(),
        "target_shift": target_shift_sweep(),
        "order": order_sweep(),
        "memory_capacity": memory_capacity_sweep(),
        "memory_store": memory_store_sweep(),
        "memory_design": memory_design_sweep(),
    }
    for name, sweep in sweeps.items():
        metrics = (
            ("acc", "bwt", "la") if name.startswith("memory") else ("acc", "bwt")
        )
        print(f"\n## {sweep.title}\n")
        print(format_sweep(sweep, metrics=metrics))

    stream = regression_stream(seed=0)
    dim = stream[0].train_inputs.shape[1]
    memory = MemoryLearner(dim, 0, **MEMORY_OPTIONS).model
    state_mib = memory.state_bytes() / 2**20
    writes_per_pair = int(MEMORY_OPTIONS["n_blocks"]) * sum(
        comb(int(MEMORY_OPTIONS["n_bits"]), j)
        for j in range(int(MEMORY_OPTIONS["write_radius"]) + 1)
    )
    stored_pairs = sum(len(task.train_inputs) for task in stream)
    print(
        f"\n## Addressed memory\n\n"
        f"{state_mib:.0f} MiB state, {writes_per_pair} cell writes per pair, "
        f"{stored_pairs} pairs stored per stream, one probe per block per query."
    )

    results = {
        "protocol": asdict(DEFAULT_PROTOCOL),
        "joint_protocol": asdict(JOINT_PROTOCOL),
        "addressed": {
            **MEMORY_OPTIONS,
            "state_mib": state_mib,
            "writes_per_pair": writes_per_pair,
            "stored_pairs": stored_pairs,
        },
        "main": {
            method: {
                "matrix": result.matrix,
                "baseline": result.baseline,
                "summary": result.summary,
                "spread": result.spread,
            }
            for method, result in main_results.items()
        },
        "sweeps": {name: asdict(sweep) for name, sweep in sweeps.items()},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
