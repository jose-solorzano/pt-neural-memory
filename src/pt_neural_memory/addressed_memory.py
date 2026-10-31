"""Addressed memory: a table of RLS cells selected by projection bits.

Each of the ``n_blocks`` blocks owns an independent random projection of the
key.  A key's address in a block is the sign pattern of its projected
coordinates read as an ``n_bits``-bit integer, so the block has ``2**n_bits``
cells.  A cell is a recursive-least-squares memory cell: it fits the local
linear map ``value = weights @ key`` from the pairs written to it, and carries
the exact ridge posterior that ``RLSMemCell`` maintains.

``store`` writes each pair not only at the key's address but at every address
within ``write_radius`` bit flips of it.  Radius 1 therefore trains
``n_blocks * (n_bits + 1)`` cells per pair instead of ``n_blocks``, and lets a
query whose address was corrupted in one bit still find a cell that was trained
on it.  Because a cell is a regression rather than a mean, the copies are not
wasted redundancy even on clean keys: each neighbour cell learns its own local
map from every pair that touches it.

``forward`` reads the cell at the query's own address in each block and
averages the blocks.  ``VerifiedMemory`` instead probes the addresses within
``read_radius`` flips and reads the cell that explains the query best.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import torch
from torch import nn

from pt_neural_memory.mem_rls import RLSCellBank


def _flip_masks(n_bits: int, radius: int) -> list[tuple[int, ...]]:
    """Every set of bit positions of size at most ``radius``.

    Each tuple names one address in the write neighbourhood: the key's own
    address for the empty tuple, otherwise the address reached by flipping
    those bits.
    """
    return [
        positions
        for size in range(radius + 1)
        for positions in combinations(range(n_bits), size)
    ]


def _mask_value(mask: tuple[int, ...]) -> int:
    """Pack bit positions into an integer XOR mask."""
    value = 0
    for position in mask:
        value |= 1 << position
    return value


class AddressedMemory(nn.Module):
    """``n_blocks`` blocks of ``2**n_bits`` RLS cells, addressed by projection signs.

    The block projections have orthonormal rows, so for a key with i.i.d.
    standard-normal coordinates each bit is an independent fair coin; the
    address of a random key is uniform on ``2**n_bits`` cells and two random
    keys share a block address with probability ``2**-n_bits``.

    A cell holds the ridge posterior of every pair written to it: its map is
    the exact least-squares fit ``value = weights @ key`` regularised by
    ``rr_lambda``, and its covariance says how well-constrained that map is.
    One cell can therefore recall up to ``key_dim`` independent associations
    exactly, where a cell that stores a mean can recall only one.  An unwritten
    cell is the prior and answers zero.  Writes are no-grad by default; pass
    ``grad_enabled=True`` to ``store`` to keep the graph through the written
    keys and values.  Address selection is discrete, so gradients do not flow
    through the choice of cell.

    Args:
        key_dim: dimension of the keys the memory is addressed with.
        value_dim: dimension of the values stored in a cell.
        n_blocks: number of independent blocks.
        n_bits: address width per block; the block has ``2**n_bits`` cells.
            Capped at 20 so a typo cannot allocate a billion cells, and it must
            not exceed ``key_dim``: the address bits are independent only while
            the projection rows are orthonormal, and a ``key_dim``-dimensional
            space admits at most ``key_dim`` of them.
        write_radius: Hamming radius of the write neighbourhood.  ``0`` writes
            only the key's own address; ``1`` also writes every one-bit
            neighbour.
        rr_lambda: ridge prior of every cell; see ``RLSMemCell``.
        seed: seed of the projection generator, not of the data.
        dtype: state dtype, promoted to at least float32.
    """

    projection: torch.Tensor
    cells: RLSCellBank

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        n_blocks: int = 3,
        n_bits: int = 12,
        write_radius: int = 1,
        rr_lambda: float = 1.0,
        seed: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if key_dim < 1 or value_dim < 1:
            raise ValueError(
                f"key_dim and value_dim must be positive, got {key_dim} and {value_dim}"
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}")
        if not 1 <= n_bits <= 20:
            raise ValueError(f"n_bits must lie in [1, 20], got {n_bits}")
        if n_bits > key_dim:
            raise ValueError(
                f"n_bits must not exceed key_dim, got n_bits={n_bits} and "
                f"key_dim={key_dim}: the address bits are independent only "
                f"while the projection rows are orthonormal, and {key_dim} "
                f"dimensions admit at most {key_dim} of them"
            )
        if not 0 <= write_radius <= n_bits:
            raise ValueError(
                f"write_radius must lie in [0, {n_bits}], got {write_radius}"
            )
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.n_blocks = n_blocks
        self.n_bits = n_bits
        self.n_cells = 1 << n_bits
        self.write_radius = write_radius
        state_dtype = torch.promote_types(dtype or torch.float32, torch.float32)
        self.register_buffer("projection", self._make_projection(seed, state_dtype))
        self.cells = RLSCellBank(
            key_dim,
            value_dim,
            n_blocks * self.n_cells,
            rr_lambda=rr_lambda,
            dtype=dtype,
        )

    def _make_projection(self, seed: int, dtype: torch.dtype) -> torch.Tensor:
        """One ``n_bits x key_dim`` block projection with orthonormal rows."""
        generator = torch.Generator().manual_seed(seed)
        blocks = []
        for _ in range(self.n_blocks):
            raw = torch.randn(
                self.key_dim, self.n_bits, generator=generator, dtype=torch.float64
            )
            orthogonal, _ = torch.linalg.qr(raw)
            blocks.append(orthogonal[:, : self.n_bits].T.contiguous())
        return torch.stack(blocks).to(dtype)

    def projections(self, key: torch.Tensor) -> torch.Tensor:
        """Signed projection coordinates, shape ``(batch, n_blocks, n_bits)``.

        A row has unit norm, so for a key with standard-normal coordinates the
        coordinate is standard normal and its absolute value is the address
        margin: how much key noise that bit tolerates before it flips.
        """
        self._check_key(key)
        return torch.einsum("bd,nmd->bnm", key, self.projection)

    def bit_patterns(self, key: torch.Tensor) -> torch.Tensor:
        """Address bits, shape ``(batch, n_blocks, n_bits)``; ``True`` is 1."""
        return self.projections(key) >= 0

    def addresses(self, key: torch.Tensor) -> torch.Tensor:
        """Packed addresses, shape ``(batch, n_blocks)``, dtype int64."""
        bits = self.bit_patterns(key).to(torch.int64)
        powers = 1 << torch.arange(self.n_bits, device=key.device)
        return (bits * powers).sum(dim=-1)

    def _check_key(self, key: torch.Tensor) -> None:
        if key.dim() != 2 or key.shape[1] != self.key_dim:
            raise ValueError(
                f"key must have shape (batch, {self.key_dim}), got {tuple(key.shape)}"
            )

    def _flat_indices(self, address: torch.Tensor) -> torch.Tensor:
        """Flatten ``(block, cell)`` pairs into one index per block row."""
        blocks = torch.arange(self.n_blocks, device=address.device)
        return address + blocks * self.n_cells

    def _write_addresses(
        self, key: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Addresses to write, with an optional per-entry validity mask.

        ``None`` means every row is written; a boolean mask of shape
        ``(batch, n_blocks)`` selects the entries of an address tensor that
        are written.  The mask is what lets a subclass skip neighbours.
        """
        base = self.addresses(key)
        return [
            (base ^ _mask_value(mask), None)
            for mask in _flip_masks(self.n_bits, self.write_radius)
        ]

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        grad_enabled: bool = False,
    ) -> None:
        """Train the write neighbourhood of a batch of associations.

        Each pair trains the cell at its address and every cell within
        ``write_radius`` flips of it, in every block. The update is no-grad by
        default; pass ``grad_enabled=True`` to keep the autograd graph through
        the key and value.

        Args:
            key: ``(batch, key_dim)`` keys to address.
            value: ``(batch, value_dim)`` values to associate with them.
            grad_enabled: keep the autograd graph for the update.
        """
        if key.dim() != 2 or value.dim() != 2:
            raise ValueError(
                "key and value must be 2D (batch, features), got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        if key.shape[0] != value.shape[0]:
            raise ValueError(
                f"batch sizes differ: {key.shape[0]} keys, {value.shape[0]} values"
            )
        if key.shape[1] != self.key_dim:
            raise ValueError(
                f"key must have shape (batch, {self.key_dim}), got {tuple(key.shape)}"
            )
        if value.shape[1] != self.value_dim:
            raise ValueError(
                f"value must have shape (batch, {self.value_dim}), "
                f"got {tuple(value.shape)}"
            )
        if key.shape[0] == 0:
            return
        with torch.set_grad_enabled(mode=grad_enabled):
            expanded_key = (
                key[:, None, :]
                .expand(-1, self.n_blocks, -1)
                .reshape(-1, self.key_dim)
            )
            expanded_value = (
                value[:, None, :]
                .expand(-1, self.n_blocks, -1)
                .reshape(-1, self.value_dim)
            )
            # One bank call for the whole neighbourhood: the masks name disjoint
            # cells for a given pair, and the bank sequences any repeats itself.
            indices = []
            keys = []
            values = []
            for address, valid in self._write_addresses(key):
                flat = self._flat_indices(address).reshape(-1)
                if valid is None:
                    indices.append(flat)
                    keys.append(expanded_key)
                    values.append(expanded_value)
                else:
                    keep = valid.reshape(-1)
                    indices.append(flat[keep])
                    keys.append(expanded_key[keep])
                    values.append(expanded_value[keep])
            self.cells.store(
                torch.cat(indices),
                torch.cat(keys),
                torch.cat(values),
                grad_enabled=grad_enabled,
            )

    def read_blocks(self, key: torch.Tensor) -> torch.Tensor:
        """Per-block cell predictions, shape ``(n_blocks, batch, value_dim)``."""
        self._check_key(key)
        flat = self._flat_indices(self.addresses(key))
        keys = key[:, None, :].expand(-1, self.n_blocks, -1)
        return self.cells.predict(flat, keys).transpose(0, 1)

    def read_variances(self, key: torch.Tensor) -> torch.Tensor:
        """Per-block posterior variance of the query, shape ``(n_blocks, batch)``.

        The cell's own confidence signal: ``‖k‖² / λ`` at an unwritten cell,
        about one at a cell whose map was fitted to this key.
        """
        self._check_key(key)
        flat = self._flat_indices(self.addresses(key))
        keys = key[:, None, :].expand(-1, self.n_blocks, -1)
        return self.cells.variance(flat, keys).transpose(0, 1)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        """Read the cell at the query address, averaged over blocks."""
        return self.read_blocks(key).mean(dim=0)


@dataclass
class CellRead:
    """The cell selected in each block, and the evidence for the choice."""

    values: torch.Tensor
    """Selected cell prediction per block, shape ``(n_blocks, batch, value_dim)``."""
    scores: torch.Tensor
    """Posterior variance of the query under the selected cell; lower is better."""
    indices: torch.Tensor
    """Flat ids of the selected cells, shape ``(n_blocks, batch)``, int64."""

    def best(self) -> torch.Tensor:
        """Value of the best-scoring block, shape ``(batch, value_dim)``."""
        pick = self.scores.argmin(dim=0)
        rows = torch.arange(pick.shape[0], device=pick.device)
        return self.values[pick, rows]

    def mean(self) -> torch.Tensor:
        """Mean of the selected block values, shape ``(batch, value_dim)``."""
        return self.values.mean(dim=0)


class VerifiedMemory(AddressedMemory):
    """Addressed memory that probes neighbouring addresses before reading.

    A read no longer trusts the query's own address alone.  It scans the
    addresses within ``read_radius`` bit flips and asks each candidate cell how
    well it explains the query, using the cell's own posterior variance:

    $$ s(c) = q^\\top P_c q . $$

    The smallest variance wins in each block, the smallest across blocks is
    returned, and the value is that cell's prediction ``W_c q``.  The signal is
    the RLS cell's own uncertainty: a cell whose writers span the query scores
    about one, an unwritten cell scores ``‖q‖² / λ``, and a cell crowded with
    unrelated keys scores ``‖q‖² / (c + λ)``.  It therefore prefers crowded
    cells, which is exactly the regime where their map has the information to
    answer; only when the crowding exceeds the key dimension does the
    distinction disappear.

    Because a write places a copy within ``write_radius`` flips and a read
    searches ``read_radius`` flips, the two radii add: a query survives up to
    ``write_radius + read_radius`` corrupted bits.  Read redundancy costs
    probes, not cells, so it does not inflate the collision rate the way
    write-side replication does.

    Args:
        read_radius: Hamming radius searched at read time.  With
            ``write_radius=1`` the default ``read_radius=1`` corrects two
            corrupted bits.
        protect_bits: if set, a write replicates only the ``protect_bits``
            lowest-margin bits instead of every bit within ``write_radius``.
            Margins are standard normal for these projections, so the lowest
            ones are the bits key noise is most likely to flip.
        seed: seed of the projection generator.
        dtype: state dtype, promoted to at least float32.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        n_blocks: int = 3,
        n_bits: int = 12,
        write_radius: int = 1,
        read_radius: int = 1,
        protect_bits: int | None = None,
        rr_lambda: float = 1.0,
        seed: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            key_dim,
            value_dim,
            n_blocks=n_blocks,
            n_bits=n_bits,
            write_radius=write_radius,
            rr_lambda=rr_lambda,
            seed=seed,
            dtype=dtype,
        )
        if not 0 <= read_radius <= n_bits:
            raise ValueError(
                f"read_radius must lie in [0, {n_bits}], got {read_radius}"
            )
        if protect_bits is not None and not 0 <= protect_bits <= n_bits:
            raise ValueError(
                f"protect_bits must lie in [0, {n_bits}] or be None, "
                f"got {protect_bits}"
            )
        self.read_radius = read_radius
        self.protect_bits = protect_bits

    def _risky_bits(self, key: torch.Tensor) -> torch.Tensor:
        """``True`` for the ``protect_bits`` lowest-margin bits of each key."""
        margins = self.projections(key).abs().transpose(0, 1)
        rank = margins.argsort(dim=-1).argsort(dim=-1)
        return (rank < self.protect_bits).transpose(0, 1)

    def _write_addresses(
        self, key: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        base = self.addresses(key)
        masks = _flip_masks(self.n_bits, self.write_radius)
        if self.protect_bits is None:
            return [(base ^ _mask_value(mask), None) for mask in masks]
        risky = self._risky_bits(key)
        plan = []
        for mask in masks:
            valid = None
            if mask:
                valid = torch.stack([risky[..., bit] for bit in mask]).all(dim=0)
            plan.append((base ^ _mask_value(mask), valid))
        return plan

    def _select(self, key: torch.Tensor, probes: list[torch.Tensor]) -> CellRead:
        """Score every probe cell, keep the least uncertain one per block."""
        batch = key.shape[0]
        best_score = torch.full(
            (batch, self.n_blocks),
            float("inf"),
            dtype=self.cells.covariance.dtype,
            device=key.device,
        )
        best_index = torch.zeros(
            (batch, self.n_blocks), dtype=torch.int64, device=key.device
        )
        keys = key[:, None, :].expand(-1, self.n_blocks, -1)
        for address in probes:
            index = self._flat_indices(address)
            score = self.cells.variance(index, keys)
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best_index = torch.where(better, index, best_index)
        return CellRead(
            values=self.cells.predict(best_index, keys).transpose(0, 1),
            scores=best_score.transpose(0, 1),
            indices=best_index.transpose(0, 1),
        )

    def read_ball(self, key: torch.Tensor, radius: int | None = None) -> CellRead:
        """Verify every cell within ``radius`` flips of the query's address."""
        radius = self.read_radius if radius is None else radius
        if not 0 <= radius <= self.n_bits:
            raise ValueError(
                f"radius must lie in [0, {self.n_bits}], got {radius}"
            )
        base = self.addresses(key)
        probes = [
            base ^ _mask_value(mask)
            for mask in _flip_masks(self.n_bits, radius)
        ]
        return self._select(key, probes)

    def read_guided(
        self, key: torch.Tensor, search_bits: int = 6
    ) -> CellRead:
        """Verify every flip pattern of the least-confident ``search_bits`` bits.

        Cost is ``2 ** search_bits`` probes per block instead of
        ``sum(C(n_bits, j) for j <= radius)``, and it does not assume an upper
        bound on the number of flipped bits: a flip in a bit with a large
        margin is simply not searched.
        """
        if not 1 <= search_bits <= min(self.n_bits, 12):
            raise ValueError(
                f"search_bits must lie in [1, {min(self.n_bits, 12)}], "
                f"got {search_bits}"
            )
        base = self.addresses(key)
        order = self.projections(key).abs().argsort(dim=-1)[..., :search_bits]
        flip = (1 << order).unsqueeze(2)  # (batch, n_blocks, 1, search_bits)
        patterns = torch.tensor(
            list(product((0, 1), repeat=search_bits)), device=base.device
        )
        flips = (flip * patterns).sum(dim=-1)  # (batch, n_blocks, 2**search_bits)
        probes = [base ^ flips[..., i] for i in range(patterns.shape[0])]
        return self._select(key, probes)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        """Read the best-verified cell, shape ``(batch, value_dim)``."""
        return self.read_ball(key).best()
