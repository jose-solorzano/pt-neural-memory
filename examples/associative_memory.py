"""Store random associations and measure clean and noisy recall.

Run from the repository root::

    python examples/associative_memory.py

At ``d=32, d_v=16, m=12, n=3`` the clean read is exact to roughly ``1e-6``.
A query with Gaussian key noise of ``sigma=0.3`` is where the design is weakest;
the verified read recovers most of the gap by probing one-bit neighbours.
"""

import torch

from pt_neural_memory import AddressedMemory, VerifiedMemory


def relative_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of target energy left unexplained: 0 perfect, 1 zero answer."""
    return float(
        (prediction - target).square().sum() / target.square().sum()
    )


def main() -> None:
    generator = torch.Generator().manual_seed(0)
    n, key_dim, value_dim = 1024, 32, 16
    keys = torch.randn(n, key_dim, generator=generator)
    values = torch.randn(n, value_dim, generator=generator)
    noise = torch.randn(n, key_dim, generator=generator)

    own = AddressedMemory(
        key_dim,
        value_dim,
        n_blocks=3,
        n_bits=12,
        write_radius=1,
        rr_lambda=0.01,
        seed=0,
    )
    own.store(keys, values)

    verified = VerifiedMemory(
        key_dim,
        value_dim,
        n_blocks=3,
        n_bits=12,
        write_radius=1,
        read_radius=1,
        rr_lambda=0.01,
        seed=0,
    )
    verified.store(keys, values)

    noisy = keys + 0.3 * noise
    print(f"stored pairs: {n}")
    print(f"clean, own address:      {relative_squared_error(own(keys), values):.2e}")
    print(f"noisy 0.3, own address:  {relative_squared_error(own(noisy), values):.2e}")
    print(
        "noisy 0.3, verified r1:  "
        f"{relative_squared_error(verified.read_ball(noisy).best(), values):.2e}"
    )


if __name__ == "__main__":
    main()
