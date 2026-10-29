"""Fit the ten-part continual-regression stream and report ACC / BWT.

Run from the repository root::

    python examples/continual_regression.py

The memory stores every training pair once, fits each cell in closed form, and
is evaluated on every part after every part.  No gradient step is taken.  On
seed 0 the expected summary is approximately ``ACC 0.948, BWT -0.010``; the
five-seed means in the report are ``0.954`` and ``-0.006``.
"""

import torch

from pt_neural_memory import AddressedRegressor, regression_stream
from pt_neural_memory.benchmarks.forgetting_bench import summarize


def r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    variance = target.var(unbiased=False)
    return float(1.0 - (prediction - target).square().mean() / variance)


def main() -> None:
    stream = regression_stream(seed=0)
    input_dim = stream[0].train_inputs.shape[1]
    model = AddressedRegressor(
        input_dim,
        1,
        kind="rff",
        width=64,
        scale=2.0,
        n_blocks=3,
        n_bits=10,
        write_radius=1,
        rr_lambda=0.01,
        seed=0,
    )

    # The benchmark's baseline is the untrained model's R^2 on each part.
    baseline = [
        r2_score(model(part.test_inputs), part.test_targets) for part in stream
    ]
    matrix: list[list[float]] = []
    for task in stream:
        model.store(task.train_inputs, task.train_targets)
        matrix.append(
            [
                r2_score(model(part.test_inputs), part.test_targets)
                for part in stream
            ]
        )

    summary = summarize(matrix, baseline)
    print(
        f"{len(stream)} parts, {model.memory.n_blocks} x 2**{model.memory.n_bits} "
        f"cells, {model.state_bytes() / 2**20:.0f} MiB state"
    )
    print(
        f"ACC {summary['acc']:.3f}  BWT {summary['bwt']:+.3f}  "
        f"LA {summary['la']:.3f}  FWT {summary['fwt']:+.3f}"
    )
    print("final row (R^2 per part):")
    print("  " + "  ".join(f"{value:+.2f}" for value in matrix[-1]))


if __name__ == "__main__":
    main()
