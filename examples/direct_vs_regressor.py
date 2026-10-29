"""Compare direct ``AddressedMemory`` with the feature-keyed regressor.

Run from the repository root::

    python examples/direct_vs_regressor.py

``AddressedMemory`` is already a regression system: storing ``(x, y)`` fits a
piecewise-linear model of ``y`` in ``x``.  ``AddressedRegressor`` is the same
memory preceded by a random feature map.  The table printed below is the
comparison quoted in the README: on the ten-part stream the raw 6-D key has too
few address bits, while the 71-D feature key holds the nonlinear target.
"""

import torch

from pt_neural_memory import AddressedMemory, AddressedRegressor, regression_stream
from pt_neural_memory.benchmarks.forgetting_bench import summarize


def r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - (prediction - target).square().mean() / target.var(unbiased=False))


def evaluate(model, stream) -> tuple[float, float, float]:
    """Fit the stream in order and return ACC, BWT and resident MiB."""
    matrix = []
    for task in stream:
        model.store(task.train_inputs, task.train_targets)
        matrix.append(
            [r2_score(model(part.test_inputs), part.test_targets) for part in stream]
        )
    summary = summarize(matrix, [0.0 for _ in stream])
    state_mib = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    ) / 2**20
    return summary["acc"], summary["bwt"], state_mib


def main() -> None:
    stream = regression_stream(seed=0)
    configs = {
        "direct memory, raw 6-D": AddressedMemory(
            6, 1, n_blocks=3, n_bits=6, write_radius=1, rr_lambda=0.01, seed=0
        ),
        "direct + intercept": AddressedRegressor(
            6,
            1,
            kind="raw",
            use_bias=True,
            n_blocks=3,
            n_bits=7,
            write_radius=1,
            rr_lambda=0.01,
            seed=0,
        ),
        "feature-keyed regressor": AddressedRegressor(
            6,
            1,
            kind="rff",
            width=64,
            scale=2.0,
            n_blocks=3,
            n_bits=10,
            write_radius=1,
            rr_lambda=0.01,
            seed=0,
        ),
    }

    print("| model | ACC | BWT | state MiB |")
    print("|---|---:|---:|---:|")
    for name, model in configs.items():
        acc, bwt, state_mib = evaluate(model, stream)
        print(f"| {name} | {acc:+.3f} | {bwt:+.3f} | {state_mib:.2f} |")

    # A linear target is consistent across every writer, so even the crowded
    # raw-key table fits it exactly.
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(5120, 6, generator=generator)
    weights = torch.randn(6, 1, generator=generator)
    targets = inputs @ weights
    linear = AddressedMemory(
        6, 1, n_blocks=3, n_bits=6, write_radius=1, rr_lambda=0.01, seed=0
    )
    linear.store(inputs, targets)
    rse = (linear(inputs) - targets).square().sum() / targets.square().sum()
    print(f"\nlinear 6-D target, direct memory: RSE {float(rse):.2e}")


if __name__ == "__main__":
    main()
