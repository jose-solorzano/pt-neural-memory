"""Addressed memory as a nonlinear regressor.

A cell is a linear map of its key, so a nonlinear model of an input ``x`` needs
a key map ``phi``.  The read is then ``W_c phi(x)`` for the cell that
``phi(x)`` addresses: a piecewise-linear function of the key, and a nonlinear
function of the input.  ``RandomFeatureMap`` supplies three key maps --

* ``rff``: random Fourier features of a Gaussian kernel with length scale
  ``scale``, ``sqrt(2/width) * cos(x Wᵀ + phase)``;
* ``relu``: the hidden layer of a random one-hidden-layer network,
  ``sqrt(2/width) * relu(x Wᵀ + bias)``;
* ``raw``: the input itself.

Each may be followed by a constant (an intercept) and by the raw input (a
linear base); both are on by default, so the key is
``[cos or relu features, 1, x]``.  ``AddressedRegressor`` composes such a map
with ``AddressedMemory``: ``store`` writes ``(phi(x), y)`` and ``forward``
reads the addressed cell's local ridge map.

The feature weights are drawn once from ``seed`` and are never trained.  All
fitting happens in the memory's closed-form posterior updates, so a stored pair
is never overwritten: the model's state is exactly the set of pairs written,
encoded as a table of local maps.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pt_neural_memory.addressed_memory import AddressedMemory, VerifiedMemory

FEATURE_SEED_OFFSET = 7_777
"""Added to the map's seed so that features and projections are independent draws."""

KINDS: tuple[str, ...] = ("rff", "relu", "raw")
"""Key maps: random Fourier features, random ReLU features, the raw input."""


class RandomFeatureMap(nn.Module):
    """A random, untrained map from an input to a memory key.

    The map is a concatenation of

    * ``width`` random features (absent for ``kind="raw"``),
    * a constant ``1``, if ``use_bias``,
    * the raw input, if ``use_raw``.

    ``rff`` draws ``W`` with i.i.d. normal entries of standard deviation
    ``1 / scale``, so ``scale`` is the kernel length scale: features of inputs
    closer than ``scale`` are correlated.  ``relu`` draws ``W`` with standard
    deviation ``scale / sqrt(input_dim)`` and a normal bias of standard
    deviation ``scale``.  Features are scaled by ``sqrt(2 / width)`` so that a
    fresh key has order-unit norm from the random part alone.

    Args:
        input_dim: dimension of the inputs.
        kind: one of ``KINDS``.
        width: number of random features; ignored for ``kind="raw"``.
        scale: length scale (``rff``) or gain (``relu``).
        use_raw: append the raw input to the key.
        use_bias: append a constant feature to the key.
        seed: seed of the feature draw.
        dtype: key dtype, promoted to at least float32.
    """

    weights: torch.Tensor | None
    offset: torch.Tensor | None

    def __init__(
        self,
        input_dim: int,
        kind: str = "rff",
        width: int = 64,
        scale: float = 2.0,
        use_raw: bool = True,
        use_bias: bool = True,
        seed: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if kind != "raw" and width < 1:
            raise ValueError(f"width must be positive, got {width}")
        if kind != "raw" and scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if kind == "raw" and not use_raw:
            raise ValueError("kind='raw' with use_raw=False would give an empty key")
        self.input_dim = input_dim
        self.kind = kind
        self.width = 0 if kind == "raw" else width
        self.scale = scale
        self.use_raw = use_raw
        self.use_bias = use_bias
        self.key_dim = self.width + int(use_bias) + (input_dim if use_raw else 0)
        self.dtype = torch.promote_types(dtype or torch.float32, torch.float32)

        generator = torch.Generator().manual_seed(seed + FEATURE_SEED_OFFSET)
        weights: torch.Tensor | None = None
        offset: torch.Tensor | None = None
        if kind == "rff":
            weights = torch.randn(
                width, input_dim, generator=generator, dtype=self.dtype
            ) / scale
            offset = (
                torch.rand(width, generator=generator, dtype=self.dtype)
                * 2.0
                * math.pi
            )
        elif kind == "relu":
            weights = (
                torch.randn(width, input_dim, generator=generator, dtype=self.dtype)
                * scale
                / input_dim**0.5
            )
            offset = (
                torch.randn(width, generator=generator, dtype=self.dtype) * scale
            )
        self.register_buffer("weights", weights)
        self.register_buffer("offset", offset)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``inputs`` of shape ``(batch, input_dim)`` to ``(batch, key_dim)``."""
        if inputs.dim() != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape (batch, {self.input_dim}), "
                f"got {tuple(inputs.shape)}"
            )
        state = inputs.to(self.dtype)
        parts = []
        if self.weights is not None and self.offset is not None:
            projection = state @ self.weights.transpose(0, 1) + self.offset
            norm = (2.0 / self.width) ** 0.5
            if self.kind == "rff":
                parts.append(norm * torch.cos(projection))
            else:
                parts.append(norm * torch.relu(projection))
        if self.use_bias:
            parts.append(
                torch.ones(state.shape[0], 1, dtype=state.dtype, device=state.device)
            )
        if self.use_raw:
            parts.append(state)
        return torch.cat(parts, dim=1)


class AddressedRegressor(nn.Module):
    """Read the local ridge map of the cell that a feature key addresses.

    A store writes the pairs ``(phi(x), y)`` into the memory; a read returns the
    addressed cell's prediction in each block, averaged over blocks (or the
    least-uncertain block, with ``read="verified"``).  By default nothing is
    trained by gradient descent — a store is no-grad — so there is no shared
    parameter for a later batch to overwrite.  Pass ``grad_enabled=True`` to
    ``store`` to keep the graph through a write.

    Args:
        input_dim: dimension of the inputs.
        value_dim: dimension of the targets.
        kind, width, scale, use_raw, use_bias: the key map; see
            ``RandomFeatureMap``.
        n_blocks: number of independent memory blocks.
        n_bits: address width per block; the block has ``2**n_bits`` cells.
        write_radius: Hamming radius of the write neighbourhood.
        rr_lambda: ridge prior of every cell.
        read: ``"own"`` averages the blocks at the query's own address;
            ``"verified"`` probes every address within one bit flip and keeps
            the cell with the smallest posterior variance.
        seed: seed of the feature map and of the projections.
        dtype: state dtype, promoted to at least float32.
    """

    def __init__(
        self,
        input_dim: int,
        value_dim: int = 1,
        kind: str = "rff",
        width: int = 64,
        scale: float = 2.0,
        use_raw: bool = True,
        use_bias: bool = True,
        n_blocks: int = 3,
        n_bits: int = 10,
        write_radius: int = 1,
        rr_lambda: float = 0.01,
        read: str = "own",
        seed: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if read not in ("own", "verified"):
            raise ValueError(
                f"read must be 'own' or 'verified', got {read!r}"
            )
        self.input_dim = input_dim
        self.value_dim = value_dim
        self.read = read
        self.features = RandomFeatureMap(
            input_dim,
            kind=kind,
            width=width,
            scale=scale,
            use_raw=use_raw,
            use_bias=use_bias,
            seed=seed,
            dtype=dtype,
        )
        memory = VerifiedMemory if read == "verified" else AddressedMemory
        self.memory = memory(
            self.features.key_dim,
            value_dim,
            n_blocks=n_blocks,
            n_bits=n_bits,
            write_radius=write_radius,
            rr_lambda=rr_lambda,
            seed=seed,
            dtype=dtype,
        )

    def store(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        grad_enabled: bool = False,
    ) -> None:
        """Write a batch of ``(x, y)`` pairs into the addressed cells.

        The feature map and the memory update are no-grad by default; pass
        ``grad_enabled=True`` to keep the graph through the inputs and targets.

        Args:
            inputs: ``(batch, input_dim)`` inputs.
            targets: ``(batch, value_dim)`` targets.
            grad_enabled: keep the autograd graph for the update.
        """
        if targets.dim() != 2 or targets.shape[1] != self.value_dim:
            raise ValueError(
                f"targets must have shape (batch, {self.value_dim}), "
                f"got {tuple(targets.shape)}"
            )
        with torch.set_grad_enabled(mode=grad_enabled):
            keys = self.features(inputs)
            if keys.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"batch sizes differ: {keys.shape[0]} inputs, "
                    f"{targets.shape[0]} targets"
                )
            self.memory.store(
                keys,
                targets.to(keys.dtype),
                grad_enabled=grad_enabled,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict targets for ``inputs``, shape ``(batch, value_dim)``."""
        keys = self.features(inputs)
        if self.read == "own":
            return self.memory.read_blocks(keys).mean(dim=0)
        return self.memory.read_ball(keys).best()

    def state_bytes(self) -> int:
        """Resident bytes of the feature map, projections and cell table."""
        return sum(
            buffer.numel() * buffer.element_size() for buffer in self.buffers()
        )
