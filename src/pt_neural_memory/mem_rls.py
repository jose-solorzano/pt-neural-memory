"""Recursive-least-squares linear associative memory.

The cell remembers how much each direction has already been constrained: it
carries the posterior covariance of the map and applies the exact Gaussian
update for every batch.  Consequently one pass over the data reaches the
least-squares fit, and later writes stop moving directions that earlier writes
already fixed.

``RLSMemCell`` is one cell updated by a batch.  ``RLSCellBank`` is a dense table
of cells with the same two buffers, so a whole write neighbourhood can be
updated in one indexed operation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RLSMemCell(nn.Module):
    """Linear memory updated by recursive least squares (a Kalman filter).

    The model is ``value = weights @ key + noise`` with prior
    ``weights ~ N(0, (1 / rr_lambda) I)``.  ``covariance`` is the posterior
    covariance of ``weights`` after all writes, ``P = (λI + Σ k kᵀ)⁻¹``, and
    ``store`` applies the exact posterior update for a batch.  ``weights``
    starts at zero, so an empty memory answers zero.

    State is ``in_features * out_features + in_features ** 2`` numbers: the
    map plus one symmetric ``in_features × in_features`` covariance.

    Args:
        decay: per-observation forgetting factor in (0, 1].  ``1.0`` keeps all
            history; smaller values discount old pairs and let the cell track a
            non-stationary target.
    """

    weights: torch.Tensor
    covariance: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype | None = None,
        rr_lambda: float = 1.0,
        decay: float = 1.0,
    ):
        super().__init__()
        if rr_lambda <= 0:
            raise ValueError(f"rr_lambda must be positive, got {rr_lambda}")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must lie in (0, 1], got {decay}")
        self.rr_lambda = rr_lambda
        self.decay = decay
        # float16 updates are numerically hopeless; keep the state in fp32+.
        state_dtype = torch.promote_types(dtype or torch.float32, torch.float32)
        weights = torch.zeros((out_features, in_features), dtype=state_dtype)
        covariance = torch.eye(in_features, dtype=state_dtype) / rr_lambda
        self.register_buffer("weights", weights)
        self.register_buffer("covariance", covariance)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        """Read the value associated with ``key`` of shape (..., in_features)."""
        return F.linear(key, self.weights, bias=None)

    def store(
        self, key: torch.Tensor, value: torch.Tensor, grad_enabled: bool = False
    ) -> None:
        """Fold a batch of associations into the weights and covariance.

        Args:
            key: (batch_size, in_features) keys to write.
            value: (batch_size, out_features) values to associate with them.
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
        if key.shape[0] == 0:
            return
        with torch.set_grad_enabled(mode=grad_enabled):
            if self.decay < 1.0:
                self.covariance = self.covariance / self.decay**key.shape[0]
            # Kalman gain in factored form: solve (I + K P Kᵀ) X = K P.
            key_cov = key @ self.covariance
            gram = key_cov @ key.transpose(0, 1)
            gram = gram + torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            gain = torch.cholesky_solve(key_cov, torch.linalg.cholesky(gram))
            residual = value - F.linear(key, self.weights, bias=None)
            self.weights = self.weights + residual.transpose(0, 1) @ gain
            self.covariance = self.covariance - gain.transpose(0, 1) @ key_cov
            # Keep P symmetric against round-off before the next Cholesky.
            self.covariance = 0.5 * (
                self.covariance + self.covariance.transpose(0, 1)
            )


# Rows gathered per update, so a large store does not build one (rows, d, d)
# temporary the size of the whole table.
_UPDATE_CHUNK = 512


class RLSCellBank(nn.Module):
    """A dense table of ``RLSMemCell`` states, indexed by a flat cell id.

    Cell ``i`` carries exactly the two buffers of an ``RLSMemCell`` — the map
    ``weights`` and the posterior ``covariance`` — and nothing else.  A write is
    the same rank-1 Kalman update, applied to every named cell:

    $$ P_i \\leftarrow P_i - \\frac{(P_i k)(P_i k)^\\top}{1 + k^\\top P_i k},
       \\qquad W_i \\leftarrow W_i +
       \\frac{(v - W_i k)(P_i k)^\\top}{1 + k^\\top P_i k}. $$

    Rows that name the same cell are applied in sequence, one "round" each, so
    the result matches storing them one at a time.  Rows that name different
    cells are independent, which is what makes a write neighbourhood one
    indexed batch instead of a Python loop.

    Keeping the posterior rather than the sufficient statistics means a read
    needs no solve: the prediction is ``W_i k`` and the cell's own confidence is
    the posterior variance ``k^\\top P_i k``.  Two consequences are worth
    stating because they are properties of the cell, not of this table:

    * an unwritten cell is the prior, ``W = 0`` and ``P = I / λ``, so it answers
      zero and its variance ``‖k‖² / λ`` is far above any written cell's;
    * the variance falls as a cell accumulates writers, so it measures how well
      a cell can explain a query, not whether the query was written to it.

    Args:
        in_features: key dimension of every cell.
        out_features: value dimension of every cell.
        n_cells: number of cells in the table.
        rr_lambda: ridge prior of every cell, the same ``lambda`` as
            ``RLSMemCell``'s.
        dtype: state dtype, promoted to at least float32.

    The bank has no forgetting factor: ``RLSMemCell.decay`` would have to be
    applied to every cell of the table on every write, which is a whole-table
    operation rather than an indexed one.
    """

    weights: torch.Tensor
    covariance: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_cells: int,
        rr_lambda: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError(
                f"in_features and out_features must be positive, "
                f"got {in_features} and {out_features}"
            )
        if n_cells < 1:
            raise ValueError(f"n_cells must be positive, got {n_cells}")
        if rr_lambda <= 0:
            raise ValueError(f"rr_lambda must be positive, got {rr_lambda}")
        self.in_features = in_features
        self.out_features = out_features
        self.n_cells = n_cells
        self.rr_lambda = rr_lambda
        # float16 posteriors would lose the ridge problem entirely.
        state_dtype = torch.promote_types(dtype or torch.float32, torch.float32)
        self.register_buffer(
            "weights", torch.zeros(n_cells, out_features, in_features, dtype=state_dtype)
        )
        self.register_buffer(
            "covariance",
            torch.eye(in_features, dtype=state_dtype).repeat(n_cells, 1, 1)
            / rr_lambda,
        )

    def _check_index(self, index: torch.Tensor) -> None:
        if index.dtype != torch.int64:
            raise ValueError(f"index must be int64, got {index.dtype}")
        if index.numel() and (index.min() < 0 or index.max() >= self.n_cells):
            raise ValueError(f"index must lie in [0, {self.n_cells})")

    def _rounds(self, index: torch.Tensor) -> list[torch.Tensor]:
        """Row positions grouped by occurrence number of their cell id.

        Rows in one round name pairwise distinct cells, so they can be updated
        together; successive rounds apply the repeats in order.
        """
        order = torch.argsort(index, stable=True)
        ordered = index[order]
        position = torch.arange(index.shape[0], device=index.device)
        opens = torch.ones_like(index, dtype=torch.bool)
        opens[1:] = ordered[1:] != ordered[:-1]
        group = torch.cumsum(opens.to(torch.int64), dim=0) - 1
        starts = position[opens]
        occurrence = position - starts[group]
        return [
            order[occurrence == round_]
            for round_ in range(int(occurrence.max().item()) + 1)
        ]

    def _update(self, index: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        """One Kalman update per row; the rows must name distinct cells."""
        covariance = self.covariance[index]
        weights = self.weights[index]
        projected = torch.einsum("ki,kij->kj", key, covariance)
        denominator = 1.0 + (projected * key).sum(dim=-1)
        residual = value - torch.einsum("koi,ki->ko", weights, key)
        scale = denominator.unsqueeze(-1)
        self.weights.index_copy_(
            0,
            index,
            weights + residual.unsqueeze(-1) * projected.unsqueeze(-2) / scale.unsqueeze(-1),
        )
        updated = covariance - (
            projected.unsqueeze(-1) * projected.unsqueeze(-2) / scale.unsqueeze(-1)
        )
        # Keep P symmetric against round-off before the next update.
        updated = 0.5 * (updated + updated.transpose(-1, -2))
        self.covariance.index_copy_(0, index, updated)

    def store(
        self,
        index: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Fold associations into the cells named by ``index``.

        Repeated indices are legal and exact: the cell receives the same
        sequence of Kalman updates as if the rows had been stored one at a
        time.

        Args:
            index: ``(batch,)`` flat cell ids.
            key: ``(batch, in_features)`` keys.
            value: ``(batch, out_features)`` values.
        """
        if index.dim() != 1:
            raise ValueError(f"index must be 1D, got shape {tuple(index.shape)}")
        if key.dim() != 2 or value.dim() != 2:
            raise ValueError(
                "key and value must be 2D (batch, features), got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        if not (index.shape[0] == key.shape[0] == value.shape[0]):
            raise ValueError(
                f"batch sizes differ: {index.shape[0]} indices, "
                f"{key.shape[0]} keys, {value.shape[0]} values"
            )
        if key.shape[1] != self.in_features:
            raise ValueError(
                f"key must have shape (batch, {self.in_features}), "
                f"got {tuple(key.shape)}"
            )
        if value.shape[1] != self.out_features:
            raise ValueError(
                f"value must have shape (batch, {self.out_features}), "
                f"got {tuple(value.shape)}"
            )
        self._check_index(index)
        if index.shape[0] == 0:
            return
        for rows in self._rounds(index):
            for start in range(0, rows.shape[0], _UPDATE_CHUNK):
                batch = rows[start : start + _UPDATE_CHUNK]
                self._update(index[batch], key[batch], value[batch])

    def predict(self, index: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Posterior mean of the cells named by ``index``, ``W_i k``.

        An unwritten cell has ``weights = 0`` and therefore answers zero.

        Args:
            index: ``(...)`` flat cell ids.
            key: ``(..., in_features)`` queries, one per id.
        """
        if key.shape[:-1] != index.shape:
            raise ValueError(
                f"key and index must agree on the leading shape, got "
                f"{tuple(key.shape)} and {tuple(index.shape)}"
            )
        return torch.einsum("...oi,...i->...o", self.weights[index], key)

    def variance(self, index: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Posterior variance ``kᵀ P_i k`` of the cells named by ``index``.

        The cell's own answer to "how well do I explain this query": small for
        a cell whose writes span the query, `‖k‖² / λ` for an unwritten cell.
        """
        if key.shape[:-1] != index.shape:
            raise ValueError(
                f"key and index must agree on the leading shape, got "
                f"{tuple(key.shape)} and {tuple(index.shape)}"
            )
        return torch.einsum("...i,...ij,...j->...", key, self.covariance[index], key)

    def written(self, index: torch.Tensor) -> torch.Tensor:
        """``True`` where the cell has been written at least once.

        A cell is written when its map is not exactly zero; the covariance
        alone cannot tell, because a write of a zero value leaves it zero.
        """
        return self.weights[index].abs().sum(dim=(-1, -2)) > 0
