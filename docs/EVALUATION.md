# Evaluating and improving the memory cell

> **Status.** This report records the comparison that led to adopting the RLS
> cell. `NeuralMemCell` and its benchmark series have since been removed: the
> checkout contains `RLSMemCell` and its dense table form `RLSCellBank`, and
> the cell benchmark reports RLS alone. The "baseline" columns below are
> historical numbers that the current code cannot regenerate.

## Summary

The original cell (since removed) was a linear key/value memory updated by a
**block Kaczmarz / affine-projection** step. It solved the current batch's
residual exactly but had no state describing what it had already learned, so one
online pass left most of the stream unexplained and later writes re-opened
earlier directions.

I found and fixed a transpose bug in the update, added the missing read path,
and built a synthetic benchmark suite covering capacity, retention, correlated
keys, noisy cues, drift, and write cost. I then implemented
`RLSMemCell` in [src/pt_neural_memory/mem_rls.py](../src/pt_neural_memory/mem_rls.py): the same interface,
plus one `d × d` posterior-covariance buffer, which makes every write the exact
Gaussian posterior update over all history.

At `d = m = 64`, `λ = 1`, one pair per write, the improvement is roughly 9–250×
lower recall error for loads `α = n/d ≤ 1`, at 2× the state and (as the code is
currently written) ~10× faster writes. The cost is that exact one-pass fitting
amplifies noisy cues at high noise, and freezes under distribution drift unless
a forgetting factor is enabled.

## 1. What the module does

The map is `W ∈ R^{m×d}` and a read is `y = W k`. A write of a batch
`K ∈ R^{n×d}`, `V ∈ R^{n×m}` computes the residual `E = V − K Wᵀ` and applies

$$ \Delta W = E^{\top} K \,(K^{\top} K + \lambda I)^{-1}. $$

For a single pair this is `ΔW = (v − W k) kᵀ / (‖k‖² + λ)`: a projection that
makes the new association true, moving the map only along `k`. It is the
affine-projection update from adaptive filtering, equivalent to one block
Kaczmarz step on `K Wᵀ = V`.

Defects found in the removed cell:

| issue | effect | status |
|---|---|---|
| `_ridge_regression` returns a `d × m` matrix that is added to the `m × d` weights | for square maps the **transpose** of the intended update is applied; rectangular maps raise a shape error | fixed with one `.transpose(0, 1)` |
| no `forward` / read method | a stored memory cannot be queried | added |
| weights initialised with `randn` | an unwritten cell already predicts a large random map, and the prior mean dominates the error | RLS now starts at zero |
| `torch.linalg.inv` on every write | `O(d³)` per write | left as-is in the baseline; RLS uses Cholesky solves |
| **no information state beyond `W`** | step size does not shrink as evidence accumulates; old associations are overwritten | the architectural problem addressed here |

The transpose bug is the reason the original cell never learned a single pair:
the update moved `W` along the key instead of along the residual.

## 2. How the cell is evaluated

The metric is relative squared error (RSE),

$$ \mathrm{RSE} = \frac{\lVert \hat{V} - V \rVert_F^2}{\lVert V \rVert_F^2}, $$

so `0` is perfect recall, `1` is what a cell that always answers zero scores,
and values above `1` mean actively wrong recall. Each task sweeps one factor of
the online-memory problem while holding the stream fixed:

* **capacity** — one pass over `α·d` random pairs, then recall all of them. This
  asks whether a single pass is enough.
* **retention** — write set A, measure it, write set B, measure A again. This
  isolates interference from capacity by keeping `|A| + |B| = d/2`.
* **correlated keys** — keys share a direction (`ρ`), so different associations
  compete for the same coordinates.
* **noisy cues** — query with `k + σ·ε`. A linear memory must trade exactness
  against noise amplification.
* **drift** — the values are rotated by an orthogonal map every epoch, so the
  target is non-stationary.
* **cost** — median microseconds per write and resident state.

Protocol: `d = m = 64`, `λ = 1`, five seeds, one pair per `store` call, and an
empty map (`W = 0`). Starting from zero matters: with `randn` weights the prior
mean is a large random map, and even a perfect ridge fit leaves ~17% error
relative to the values. The comparison below was run with both cells zeroed;
`RLSMemCell` now starts at zero by construction. The random-init ranking was
unchanged (see §6).

Reproduce the current RLS tables with:

```bash
python -m pytest -q tests            # correctness and invariants
python -m pt_neural_memory.benchmarks.memory_bench    # RLS tables + results/memory_bench.json
```

The suite is implemented in [src/pt_neural_memory/benchmarks/memory_bench.py](../src/pt_neural_memory/benchmarks/memory_bench.py)
and the invariants it relies on are in [tests/test_mem_rls.py](../tests/test_mem_rls.py).

## 3. Results

### 3.1 Capacity: one pass, then recall

| load `α = n/d` | baseline | RLS | ratio |
|---:|---:|---:|---:|
| 0.125 | 6.27e-02 | 3.71e-04 | 169× |
| 0.25 | 1.32e-01 | 5.27e-04 | 251× |
| 0.5 | 2.89e-01 | 1.83e-03 | 158× |
| 0.75 | 4.54e-01 | 9.00e-03 | 50× |
| 1.0 | 6.01e-01 | 6.78e-02 | 9× |
| 1.5 | 8.63e-01 | 3.37e-01 | 3× |
| 2.0 | 1.06e+00 | 5.06e-01 | 2× |
| 3.0 | 1.32e+00 | 6.67e-01 | 2× |

The baseline needs many passes to converge, so a single pass leaves error that
grows with load: more associations means more directions for later writes to
disturb. RLS is at the least-squares fit after the first pass. Above `α ≈ 1`
the system is overdetermined and inconsistent (values are independent), so both
approach the irreducible floor `1 − d/n`; the baseline is worse than a
zero-answer cell even at `α = 2`.

RLS's rise from `3.7e-04` to `6.8e-02` near `α = 1` is not drift in the update:
it is ridge bias. As `n` approaches `d`, the smallest singular values of `K`
become comparable to `λ`, so the prior deliberately refuses to fit those
directions. The error is still 9× below the baseline at that load, and the
closed-form posterior reproduces it to `1e-6` (see `tests` and §3.2).

### 3.2 Retention across a task boundary

| phase | baseline | RLS |
|---|---:|---:|
| A before B | 1.32e-01 | 5.27e-04 |
| A after B | 4.57e-01 | 1.90e-03 |
| B after B | 1.63e-01 | 2.25e-03 |

After B, the baseline's error on A is 3.5× worse and is 240× the RLS error.
RLS's own A error rises from `5.3e-04` to `1.9e-03`, but that is not
overwriting: writing A then B sequentially gives exactly the same weights as one
batch fit of `A ∪ B` (max element difference `3.2e-07`), and the joint fit is
simply slightly more biased at the higher load. The cell has no order-dependent
forgetting; the baseline does.

### 3.3 Correlated keys

| key correlation `ρ` | baseline | RLS |
|---:|---:|---:|
| 0.00 | 2.84e-01 | 1.80e-03 |
| 0.50 | 2.88e-01 | 1.81e-03 |
| 0.80 | 3.20e-01 | 1.93e-03 |
| 0.95 | 4.93e-01 | 2.73e-03 |
| 0.99 | 9.52e-01 | 8.69e-03 |

Shared directions are where the baseline is worst: every write along a shared
direction disturbs all associations that use it, and the projection step has no
way to notice that the direction is already spoken for. RLS keeps the error
near its clean-fit floor because the gain along an already-observed direction
is proportional to the remaining covariance in that direction.

### 3.4 Noisy cues

| cue noise `σ` | baseline | RLS |
|---:|---:|---:|
| 0.00 | 2.89e-01 | 1.83e-03 |
| 0.10 | 2.92e-01 | 1.04e-02 |
| 0.25 | 3.11e-01 | 5.58e-02 |
| 0.50 | 3.81e-01 | 2.19e-01 |
| 0.75 | 5.00e-01 | 4.90e-01 |
| 1.00 | 6.68e-01 | 8.71e-01 |

This is the honest failure mode of exact fitting. RLS learns a larger-norm map
because it actually fits the associations; a noisy cue then gets multiplied by
that larger map. The underfit baseline is implicitly regularized and wins at
`σ = 1`. The crossover is near `σ = 0.75`. If noisy cues matter, `λ` is the
control: it is now a real prior precision, so increasing it trades clean
capacity for robustness.

### 3.5 Drift

| epoch | baseline | RLS `decay=1` | RLS `decay=0.98` | RLS `decay=0.95` |
|---:|---:|---:|---:|---:|
| 1 | 2.89e-01 | 1.83e-03 | 1.08e-03 | 5.87e-04 |
| 2 | 4.06e-01 | 5.44e-01 | 2.59e-01 | 5.78e-02 |
| 3 | 4.22e-01 | 7.10e-01 | 3.28e-01 | 6.60e-02 |
| 4 | 4.26e-01 | 7.89e-01 | 3.56e-01 | 6.75e-02 |

One-pass least squares is optimal for a fixed target. When the target rotates,
`decay=1` keeps answering the time-average and becomes worse than the baseline
by epoch 2. A per-observation discount fixes it: `decay=0.95` tracks at
`6.8e-02`, 6× better than the baseline. The cost is a shorter memory, so this
must be a deliberate choice for non-stationary streams, not a default.

### 3.6 Write cost

Measured at `d = m = 256`, one CPU thread, median of 10 writes over 3 repeats:

| batch | baseline µs/write | RLS µs/write |
|---:|---:|---:|
| 1 | 2974 | 291 |
| 16 | 2872 | 611 |

State: 256 KiB → 512 KiB (the extra `d²` covariance).

The baseline inverts a `d × d` matrix on every write; RLS solves an `n × n`
system and does `O(n d²)` work. At batch 1 that is a 10× speedup, at batch 16
about 5×. This part is implementation-dependent: replacing the baseline's
`inv` with a solve, or solving in the `n`-space, would close much of the gap.
The retention gains do not depend on it.

## 4. Why the first cut behaves this way

The update uses only the current batch. The step applied to a key is
`1/(‖k‖² + λ)`, independent of how many times that direction has been written
before. Three consequences follow directly:

1. Convergence is slow: Kaczmarz needs repeated passes to approach the solution,
   while an online memory gets one pass.
2. A write to a key that overlaps earlier keys moves their predictions too, so
   old associations decay.
3. Correlated keys share directions, so they fight for the same coordinates.

The cell therefore behaves like an iterative linear solver that is used as a
memory. It is a good solver — with replay it converges — but a memory needs a
notion of "how well do I already know this direction", and that state is
missing.

## 5. The architectural change: recursive least squares

Add one state variable: the posterior covariance `P` of the map. With the prior
`W ~ N(0, (1/λ) I)` and unit-variance observation noise, after seeing keys
`K` the posterior is Gaussian with

$$ P = (\lambda I + K^{\top} K)^{-1}, \qquad W = V^{\top} K P. $$

For a new batch, Woodbury gives the exact rank-`n` update:

$$ S = I + K P K^{\top}, \qquad G = S^{-1} K P, $$

$$ W \leftarrow W + E^{\top} G, \qquad P \leftarrow P - G^{\top} K P. $$

This is what [src/pt_neural_memory/mem_rls.py](../src/pt_neural_memory/mem_rls.py) implements, with a
Cholesky solve for `S` and symmetrisation of `P`. The update is exact: processing
a batch one pair at a time produces the same weights and covariance as one batch
call (tested to `1e-4`), and the order of the pairs does not matter. It is also
the Kalman filter for this model, so the "confidence" interpretation is literal.

Why this fixes the three symptoms:

* the gain along a direction is proportional to the remaining covariance in that
  direction, so after a key has been written, the same key produces a smaller
  step rather than a full projection;
* as `λ → 0`, `P` becomes the projector onto the orthogonal complement of the
  keys seen so far, so writes become orthogonal to old associations;
* correlated keys are handled by the off-diagonal structure of `P`, which the
  baseline does not represent at all.

State grows from `m·d` to `m·d + d²`: 2× for a square map, the same `O(d²)`
class. The extra `d²` is independent of `m`, so it is proportionally small for
tall maps (`m ≫ d`) and dominant for narrow ones (`m ≪ d`). A packed symmetric
`P` makes the increase 1.5×; a diagonal `P` adds only `d` numbers but can only
rescale coordinates, which would not fix the correlated-key result. The
covariance is what makes one pass sufficient; it is not a larger model in the
usual sense.

For non-stationary streams, `decay < 1` discounts old evidence by dividing `P`
by `decay` before each observation (per observation, so `decay**n` for a batch).
This is the standard forgetting factor; the effective window is about
`1/(1 − decay)` observations.

Implementation notes:

* Solving in the `n × n` space avoids forming `KᵀK + λI` in single precision. I
  hit that ill-conditioning while writing the baseline test: at `λ = 1e-6` the
  `d × d` inverse is numerically unusable in fp32 even though the system is
  well-posed. The factored update stays well-conditioned.
* Covariance is out-of-place and symmetrised each step; a 320-write stream
  keeps `P` positive definite (`λ_min = 1.6e-3`) and exactly symmetric.
* `grad_enabled=True` is still supported: the buffers are reassigned rather
  than mutated in place, and the graph reaches the written keys
  (`test_grad_enabled_keeps_the_graph_alive`).

## 6. Sensitivity, limitations, and what was not tested

* **Initialisation.** The removed cell used `randn` weights; with that prior,
  capacity at `α = 1` was `8.72` for the baseline and `8.07e-02` for RLS, so the
  ranking was unchanged. `RLSMemCell` now starts at zero.
* **Spread.** Across the five seeds, capacity at `α = 0.5` is
  `0.289 ± 0.017` (baseline) and `1.83e-03 ± 3.4e-04` (RLS). The gaps are far
  larger than the seed noise, but the harness reports means only; adding error
  bars would be the next step.
* **One scale.** Everything is `d = m = 64` and synthetic Gaussian data. The
  mechanism is dimension-free and the x-axis is normalized load, but I did not
  verify `d = 256` capacity or real embeddings.
* **No real tasks.** No language-model or retrieval workload was involved, so
  the absolute RSE values are not a claim about downstream quality.
* **Noise.** The `σ = 1` loss is real; the fix is a prior/`λ` choice or a
  denoising read, neither of which is part of this change.
* **Decay.** `0.95` was chosen for the drift experiment, not tuned. It is a
  hyperparameter with a clear meaning but needs validation data in practice.
* **State.** `P` is stored densely and symmetrically redundant. Packing it, or
  maintaining a low-rank factor (exact while the number of stored keys is below
  `d`), are straightforward follow-ups. They reduce memory, not error.

## 7. Recommendation

1. `RLSMemCell` is the only memory cell; `NeuralMemCell` has been removed.
2. Keep the map initialised to zero. An empty memory should answer zero.
3. Keep `decay = 1` unless the stream is known to drift; then expose `decay`
   and set it from a validation window.
4. If a projection-based baseline is ever reintroduced for comparison, replace
   `inv` with `solve` and solve in the `n`-space; that is a free correction, not
   an architectural change.
5. If state matters more than exactness, the packed or low-rank `P` variants are
   the next step; they preserve the update rule.
