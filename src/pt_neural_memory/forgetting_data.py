"""A ten-part regression stream whose parts differ in input distribution.

One nonlinear target function ``f`` maps ``R^dim`` to ``R``.  Part ``t`` draws
inputs from its own ellipsoidal distribution around a centre ``c_t``: a
different mean, a different covariance orientation, and a marginal shape that
cycles through Gaussian, uniform and Laplace.  Centres are mutually far apart
relative to the spread, so the parts have almost disjoint support: a model
trained on one part has seen no data anywhere near another part's inputs.

This is the multi-variable regression analogue of permuted MNIST.  A single
MLP with one output head and no task identifier must represent every part in
the same weights, so each part learned later overwrites the function's
behaviour on the earlier parts' regions.

The generator is deterministic given ``seed``.  Test targets are the noiseless
function values, so a reported score measures how well the *function* was
retained rather than how lucky the noise draw was; noise is added to the
training targets only.
"""

from dataclasses import dataclass

import torch

SHAPES: tuple[str, ...] = ("gaussian", "uniform", "laplace")


@dataclass(frozen=True)
class Task:
    """One part of the stream: a regression sample from one input distribution."""

    index: int
    name: str
    train_inputs: torch.Tensor  # (n_train, dim) float32
    train_targets: torch.Tensor  # (n_train, 1) float32
    test_inputs: torch.Tensor  # (n_test, dim) float32
    test_targets: torch.Tensor  # (n_test, 1) float32


def target_function(inputs: torch.Tensor) -> torch.Tensor:
    """The shared target: oscillatory, interacting and mildly quadratic in x.

    Uses coordinates 0-3 for the oscillatory and interaction terms and every
    coordinate for the radial term, so extra dimensions are not inert.

    Args:
        inputs: (n, dim) tensor with ``dim >= 4``.

    Returns:
        (n, 1) tensor.
    """
    oscillatory = torch.sin(1.5 * inputs[:, 0]) * torch.cos(1.5 * inputs[:, 1])
    interaction = 0.6 * torch.tanh(inputs[:, 2] * inputs[:, 3])
    radial = 0.3 * ((inputs.square().mean(dim=1) - 9.0) / 4.0)
    return (oscillatory + interaction + radial).unsqueeze(1)


def regression_stream(
    n_tasks: int = 10,
    dim: int = 6,
    n_train: int = 512,
    n_test: int = 256,
    center_radius: float = 3.0,
    spread: float = 0.35,
    anisotropy: float = 2.0,
    noise: float = 0.1,
    rotated_targets: bool = False,
    seed: int = 0,
) -> list[Task]:
    """Build the stream.

    Args:
        n_tasks: number of parts; each is trained on in order 0, 1, ..., n-1.
        dim: number of input variables.
        n_train, n_test: samples per part for training and evaluation.
        center_radius: distance of every part's mean from the origin.
        spread: nominal per-coordinate standard deviation within a part.
        anisotropy: ratio between the largest and smallest covariance axis.
            ``1.0`` gives isotropic parts.
        noise: standard deviation of the target noise on the training set, as a
            fraction of that part's target standard deviation.
        rotated_targets: give each part its own rotated copy of ``f`` instead of
            the shared ``f``.  This turns the stream into a function shift,
            which no single shared function can fit.
        seed: seed for centres, covariances, samples, rotations and noise.

    Returns:
        A list of ``n_tasks`` tasks in training order.

    Raises:
        ValueError: on any argument outside its documented range.
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be positive, got {n_tasks}")
    if dim < 4:
        raise ValueError(f"dim must be at least 4 for the target function, got {dim}")
    if n_train < 1 or n_test < 1:
        raise ValueError(f"n_train and n_test must be positive, got {n_train}, {n_test}")
    if center_radius <= 0:
        raise ValueError(f"center_radius must be positive, got {center_radius}")
    if spread <= 0:
        raise ValueError(f"spread must be positive, got {spread}")
    if anisotropy < 1.0:
        raise ValueError(f"anisotropy must be at least 1, got {anisotropy}")
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"noise must lie in [0, 1), got {noise}")

    generator = torch.Generator().manual_seed(_seed(seed, 0, 0))
    centers = _task_centers(n_tasks, dim, center_radius, generator)

    tasks = []
    for index in range(n_tasks):
        factor = _task_factor(dim, anisotropy, _seed(seed, index, 1))
        shape = SHAPES[index % len(SHAPES)]
        rotation = (
            _random_rotation(dim, _seed(seed, index, 4)) if rotated_targets else None
        )
        train_inputs = _draw(
            centers[index], factor, spread, shape, n_train, _seed(seed, index, 2)
        )
        test_inputs = _draw(
            centers[index], factor, spread, shape, n_test, _seed(seed, index, 3)
        )
        train_targets = _targets(train_inputs, rotation)
        test_targets = _targets(test_inputs, rotation)

        if noise > 0.0:
            noise_generator = torch.Generator().manual_seed(_seed(seed, index, 5))
            jitter = torch.randn(train_targets.shape, generator=noise_generator)
            train_targets = train_targets + noise * train_targets.std() * jitter

        tasks.append(
            Task(
                index=index,
                name=f"t{index:02d}",
                train_inputs=train_inputs,
                train_targets=train_targets,
                test_inputs=test_inputs,
                test_targets=test_targets,
            )
        )
    return tasks


def _seed(seed: int, task: int, slot: int) -> int:
    """Independent generator seed for one (stream, task, purpose) triple."""
    return seed * 100_000 + task * 100 + slot


def _task_centers(
    n_tasks: int,
    dim: int,
    radius: float,
    generator: torch.Generator,
    pool_size: int = 2_048,
) -> torch.Tensor:
    """Pick well-separated centres by farthest-point sampling on the sphere.

    Greedy farthest-point sampling is used instead of rejection sampling so
    that the parts' separation is as large as the radius allows for any
    ``n_tasks``, without a threshold that can become infeasible.
    """
    candidates = torch.randn(pool_size + n_tasks, dim, generator=generator)
    candidates = radius * candidates / candidates.norm(dim=1, keepdim=True)
    chosen = [0]
    nearest = torch.cdist(candidates, candidates[chosen[:1]]).squeeze(1)
    for _ in range(n_tasks - 1):
        index = int(nearest.argmax())
        chosen.append(index)
        distance = torch.cdist(candidates, candidates[index : index + 1]).squeeze(1)
        nearest = torch.minimum(nearest, distance)
    return candidates[chosen]


def _task_factor(dim: int, anisotropy: float, seed: int) -> torch.Tensor:
    """A ``dim x dim`` scale-and-rotate factor with geometric-mean scale 1."""
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    exponent = torch.log10(torch.tensor(anisotropy)).item()
    scales = torch.logspace(-exponent, exponent, dim)
    return rotation * scales


def _random_rotation(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    return rotation


def _standard_sample(
    shape: str, n: int, dim: int, generator: torch.Generator
) -> torch.Tensor:
    """``n`` samples with identity covariance from the named marginal shape."""
    if shape == "gaussian":
        return torch.randn(n, dim, generator=generator)
    if shape == "uniform":
        return (torch.rand(n, dim, generator=generator) - 0.5) * 12**0.5
    if shape == "laplace":
        # Laplace(0, b) has variance 2b^2, so b = 2^-1/2 for unit variance.
        uniform = torch.rand(n, dim, generator=generator) - 0.5
        magnitude = -torch.log1p(-2.0 * uniform.abs()) * 2**-0.5
        return torch.sign(uniform) * magnitude
    raise ValueError(f"unknown shape {shape!r}, expected one of {SHAPES}")


def _draw(
    center: torch.Tensor,
    factor: torch.Tensor,
    spread: float,
    shape: str,
    n: int,
    seed: int,
) -> torch.Tensor:
    """Sample ``n`` inputs from one part's distribution."""
    generator = torch.Generator().manual_seed(seed)
    standard = _standard_sample(shape, n, center.numel(), generator)
    return center + spread * standard @ factor.transpose(0, 1)


def _targets(inputs: torch.Tensor, rotation: torch.Tensor | None) -> torch.Tensor:
    """Apply the shared target, optionally through a part-specific rotation."""
    rotated = inputs if rotation is None else inputs @ rotation.transpose(0, 1)
    return target_function(rotated)
