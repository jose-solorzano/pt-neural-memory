# pt-neural-memory

[![PyPI](https://img.shields.io/pypi/v/pt-neural-memory.svg)](https://pypi.org/project/pt-neural-memory/)
[![Python](https://img.shields.io/pypi/pyversions/pt-neural-memory.svg)](https://pypi.org/project/pt-neural-memory/)
[![tests](https://github.com/jose-solorzano/pt-neural-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jose-solorzano/pt-neural-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jose-solorzano/pt-neural-memory/blob/main/LICENSE)

A fast associative memory built from **addressed recursive-least-squares (RLS)
cells**, and a nonlinear regressor when keyed by random features.

The memory stores associations in one pass with an exact closed-form update. Its
capacity is a configuration choice: the table can be sized from a few hundred
cells to millions, and capacity grows with it. Because each address owns its own
local map, later writes do not overwrite earlier ones. On a ten-part continual
regression stream the memory ends at **ACC 0.954** and **BWT −0.006** — within
0.02 of joint training and 0.11 above replay — with no gradient step.

## What this is

Three composable layers, all in PyTorch:

| Layer | What it does |
|---|---|
| `RLSMemCell` | A linear map `value = W @ key` fitted by the exact RLS/Kalman posterior update, including its `d × d` covariance. One pass reaches the least-squares fit. |
| `AddressedMemory` | `n_blocks` tables of `2**n_bits` cells, selected by the sign bits of a random projection. A store trains the key's cell and its one-bit neighbours; a read is a lookup. `VerifiedMemory` adds a read-side probe. |
| `AddressedRegressor` | A random Fourier feature map (optionally ReLU features or raw input) composed with the table. The result is a nonlinear regressor fitted in closed form. |

It is a **lookup and regression** structure, not a trainable layer: stores are
indexed scatter updates, reads are gathers, and nothing is differentiated. That
is the property that makes it useful as memory (no shared weight to overwrite)
and the property that limits it (no end-to-end feature learning).

## Results at a glance

Measured on CPU, float32, against synthetic Gaussian data unless stated. Full
protocols, tables and caveats are in [docs](https://github.com/jose-solorzano/pt-neural-memory/tree/main/docs).

| Measurement | Result |
|---|---|
| Clean recall, 1,024 pairs (`d=32`, `m=12`, 3 blocks) | RSE `2.0e-7` own address, `3.8e-7` verified |
| Clean recall, 4,096 pairs | RSE `1.1e-6` own, `1.8e-5` verified |
| Noisy cues, `σ=0.3`, 1,024 pairs | RSE `1.8e-1` own, **`4.3e-2` verified** |
| Capacity, one cell vs. a 2¹²-cell table (`d=32`) | 24 vs. **5,041** pairs at 99% exact (**210×**) |
| Ten-part regression, 5 seeds | **ACC 0.954 ± 0.008, BWT −0.006 ± 0.005** |
| Joint training / replay reference | ACC 0.972 / 0.848, BWT +0.020 / −0.025 |
| Cost at `d=32, d_v=16, m=12, n=3` | ~300 µs per stored pair, 5–7 µs per own-address read, 72 MiB state |

The regressor's 60 MiB state and 5,120 stored pairs are the price of that ACC.
The memory is a ceiling on what keeping the data can do, not a compression
method. See [Limitations](#limitations).

## Install

```bash
pip install pt-neural-memory
```

From a checkout, with the test and benchmark dependencies:

```bash
git clone https://github.com/jose-solorzano/pt-neural-memory.git
cd pt-neural-memory
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+. The package is CPU-friendly; the code
works on CUDA tensors where PyTorch supports the operations, but it has only
been benchmarked on CPU.

## Quickstart

### 1. One RLS cell

```python
import torch
from pt_neural_memory import RLSMemCell

cell = RLSMemCell(in_features=32, out_features=16, rr_lambda=0.01)
keys = torch.randn(16, 32)
values = torch.randn(16, 16)

cell.store(keys, values)          # exact posterior update, one pass
prediction = cell(keys)           # W @ key

rse = (prediction - values).square().sum() / values.square().sum()
print(float(rse))                 # ~1e-6: 16 associations fit exactly
```

An empty cell is the prior: `W = 0`, so it answers zero. `rr_lambda` is the
prior precision; `decay < 1` turns the cell into a forgetting factor for
non-stationary streams. The cell is bounded by its key dimension — at most `d`
independent associations — which is why the table exists.

### 2. Addressed memory

```python
import torch
from pt_neural_memory import AddressedMemory

memory = AddressedMemory(
    key_dim=32,
    value_dim=16,
    n_blocks=3,
    n_bits=12,          # 2**12 = 4,096 cells per block
    write_radius=1,     # also train every one-bit neighbour
    rr_lambda=0.01,
    seed=0,
)

keys = torch.randn(4096, 32)
values = torch.randn(4096, 16)
memory.store(keys, values)

prediction = memory(keys)   # own address in each block, averaged
rse = (prediction - values).square().sum() / values.square().sum()
print(float(rse))           # ~1e-6
```

`memory.read_blocks(keys)` returns the per-block predictions and
`memory.read_variances(keys)` returns each cell's posterior variance.

### 3. Verified reads for noisy cues

```python
import torch
from pt_neural_memory import VerifiedMemory

memory = VerifiedMemory(
    key_dim=32,
    value_dim=16,
    n_blocks=3,
    n_bits=12,
    write_radius=1,
    read_radius=1,      # probe every address within one bit flip
    rr_lambda=0.01,
    seed=0,
)

keys = torch.randn(1024, 32)
values = torch.randn(1024, 16)
memory.store(keys, values)

noisy = keys + 0.3 * torch.randn_like(keys)
read = memory.read_ball(noisy)       # scores probes by q' P q
prediction = read.best()             # least-uncertain block
```

Write and read radii add: `write_radius=1` plus `read_radius=1` survives two
corrupted address bits. Verification is worth roughly 4–20× over the query's
own address at moderate noise, but it cannot remove noise amplification once a
cell is crowded (see [Limitations](#limitations)).

### 4. Nonlinear regression

```python
from pt_neural_memory import AddressedRegressor, regression_stream

model = AddressedRegressor(
    input_dim=6,
    value_dim=1,
    kind="rff",       # random Fourier features; "relu" or "raw" also work
    width=64,
    scale=2.0,
    n_blocks=3,
    n_bits=10,
    write_radius=1,
    rr_lambda=0.01,
    seed=0,
)

stream = regression_stream(seed=0)
for task in stream:
    model.store(task.train_inputs, task.train_targets)

part = stream[0]
prediction = model(part.test_inputs)
```

The key is
`[sqrt(2/64) cos(Wx + b), 1, x]`: 64 random Fourier features, an intercept, and
the raw input as a linear base. `W` and `b` are drawn once from `seed` and are
never trained. Every fit happens in the cells' closed-form posterior updates.

## How it works

### Addressing

Block `b` owns a random `m × d` projection `R_b` with orthonormal rows. The
address of key `k` in that block is the sign pattern of `R_b k`, read as a
binary integer:

```
a_b(k) = Σ_i 2^i [ (R_b k)_i ≥ 0 ]
```

For a standard-normal key each bit is an independent fair coin, so addresses
are uniform and two keys collide with probability `2**-m`. A block has `2**m`
cells; `n_blocks` independent blocks each route the same key.

### The cell

A cell is a ridge linear map with prior `W ~ N(0, I/λ)`. After keys `K` and
values `V`, its state is the exact posterior

```
P = (λI + KᵀK)⁻¹,        W = Vᵀ K P.
```

A write applies the rank-one Kalman update to every cell it names:

```
P_i ← P_i − (P_i k)(P_i k)ᵀ / (1 + kᵀ P_i k)
W_i ← W_i + (v − W_i k)(P_i k)ᵀ / (1 + kᵀ P_i k)
```

Rows naming the same cell are applied in sequence, so repeated writes reproduce
one-at-a-time storage exactly. Rows naming different cells are one indexed
batch. A read is `W_i k`; the cell's confidence is the posterior variance
`kᵀ P_i k`.

Keeping the posterior instead of the sufficient statistics means a read needs
no solve and the confidence is a genuine uncertainty. It also means a cell
cannot report how many writers it has: an unwritten cell is just `W = 0`.

### Writing and reading

A store trains the cell at `a_b(k)` and every address within `write_radius` bit
flips of it, in every block. Radius 1 is 13 cells per block, **39 per pair** in
the default three-block configuration. These are not redundant copies: each
neighbour fits its own local map from every pair that touches it, so a query
that lands on a neighbour still gets a map trained on its key.

A plain read evaluates the map at the query's own address in each block and
averages. `VerifiedMemory` instead probes the addresses within `read_radius`
flips and keeps the cell with the smallest posterior variance:

```
s(c) = qᵀ P_c q.
```

### Why there is no catastrophic forgetting

There is no shared parameter vector to overwrite. A later write either trains a
cell that already holds constraints (making it a better joint least-squares fit
of all its writers) or trains a cell that was empty. The state is the set of
stored associations, encoded as local maps; recall is a lookup, not a forward
pass through weights that have since changed.

That is also the honest caveat: the memory keeps the data in compressed form.
It is a **ceiling check** for continual learning — what does a model that
retains every pair score? — not a gradient-trained defence. In the ten-part
benchmark it stores all 5,120 pairs and 60 MiB, against replay's 0.848 ACC at
320 pairs and 140 KiB.

### State and sizing

Per cell: `d_v · d` weights plus `d · d` covariance. The full table is

```
n_blocks · (2**n_bits · (d_v · d + d²) + n_bits · d)   numbers.
```

The covariance dominates for every value width below `d`. Capacity is set by
how many writers one cell can interpolate, which is a fraction `f` of the key
dimension:

```
N ≈ f · d · 2**n_bits / (n_bits + 1).
```

`f = 0.8` is the default for clean keys at 1% error. Noise lowers it sharply;
measure it for the deployment's noise level. Blocks are replicas for capacity
purposes: they improve the read under noise, not the number of associations a
table can hold. Address width is the capacity knob, capped at `n_bits = 20` and
at `key_dim`; beyond that the state, not the design, is the limit.
`balanced_layout` splits a byte budget between address bits and blocks using
measured admissible occupancy:

```python
from pt_neural_memory import balanced_layout

layout = balanced_layout(1 << 30, value_dim=16, key_dim=64)
print(layout.n_blocks, layout.n_bits, layout.capacity)
# 3 14 6171
print(layout.summary())
```

The rule it implements, measured at `σ=0.25`: use two or three blocks, spend the
rest on address bits, and never buy a fourth block. Two are better when dropping
one block buys an extra address bit. See
[docs/ADDRESSED_MEMORY.md](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/ADDRESSED_MEMORY.md) §5.

## Benchmarks: recovery and forgetting

Both suites print Markdown and write JSON under `results/` relative to the
working directory. Run them from a checkout (or any writable directory when the
package is installed).

### Recovery / recall

"Recovery" here means recalling a stored value from its key, including when the
cue is noisy or address bits are flipped. The metric is relative squared error
(RSE):

```
RSE = ||prediction − value||² / ||value||²,
```

so `0` is perfect recall and `1` is what a memory that always answers zero
scores.

```bash
# One RLS cell: capacity, retention, correlated keys, noise, drift, cost.
python -m pt_neural_memory.benchmarks.memory_bench

# The addressed table: load, noise, read strategy, ridge, cost.
python -m pt_neural_memory.benchmarks.addressed_bench

# Cell versus table capacity, and the balance of bits and blocks.
python -m pt_neural_memory.benchmarks.capacity_bench
python -m pt_neural_memory.benchmarks.balance_bench
```

A single recovery number, without the suite (~15 s):

```python
from pt_neural_memory.benchmarks.addressed_bench import error, read_ball, read_own

print("clean, own address:     ", f"{error(read_own, 1024, 0.0):.1e}")
print("sigma=0.3, verified r1: ", f"{error(read_ball, 1024, 0.3):.1e}")
# clean, own address:      2.0e-07
# sigma=0.3, verified r1:  4.3e-02
```

Representative results (`d=32`, `d_v=16`, `m=12`, `n=3`, `λ=0.01`, write
radius 1, 3 seeds, 512 queries):

| associations | `σ=0` own | `σ=0` verified | `σ=0.3` own | `σ=0.3` verified |
|---:|---:|---:|---:|---:|
| 1,024 | 2.0e-7 | 3.8e-7 | 1.8e-1 | **4.3e-2** |
| 4,096 | 1.1e-6 | 1.8e-5 | 3.1e-1 | 2.3e-1 |
| 8,192 | 5.2e-3 | 1.9e-1 | 1.1 | 1.6 |

What to look for:

* **Clean recall is exact until the cell is crowded.** The knee is near
  `c ≈ d` writers per cell, not near `c = 1`. At `d=32`, `N=4096` is 13 writers
  per cell and still exact; `N=8192` is 26 and starts to break.
* **Noise is the real failure mode.** The map amplifies the noise component in
  the span of the cell's writers. Verified reads help by a factor of 4–20 at
  moderate noise, and stop helping beyond `c ≈ 13`.
* **`λ` trades clean shrinkage for nothing on noise.** It sets the clean-recall
  bias `λ/(||k||²+λ)`; it cannot damp the key-Gram inverse that causes the
  amplification. `0.01` is the default.

### Catastrophic forgetting

The stream has ten parts with disjoint input supports and one shared nonlinear
target, evaluated with no task identifier (domain-incremental). Every part is
tested after every stage, producing a 10 × 10 `R²` matrix. From it:

* **ACC** — mean final `R²` over all parts (0 is the per-part mean predictor).
* **BWT** — backward transfer: final minus when-current, averaged. Negative
  means forgetting.
* **LA** — learning accuracy: how well each part was fit when current.
* **FWT** — forward transfer; uninformative here because the supports are
  disjoint.

```bash
# Full suite: main table + ablations, ~30 min on four CPU threads.
python -m pt_neural_memory.benchmarks.forgetting_bench
```

One method on one seed, in a few seconds:

```python
from pt_neural_memory.benchmarks.forgetting_bench import benchmark, summarize

run = benchmark("addressed", seed=0)
print(summarize(run.matrix, run.baseline))
# {'acc': 0.948, 'bwt': -0.010, 'la': 0.957, 'fwt': -0.088}
```

Five-seed means, mean ± standard deviation:

| method | ACC | BWT |
|---|---:|---:|
| naive fine-tuning | −1.032 ± 1.221 | −2.110 ± 1.308 |
| EWC | −0.630 ± 0.802 | −1.656 ± 0.900 |
| SI | −0.498 ± 0.707 | −1.505 ± 0.798 |
| replay, 32 samples/part | 0.848 ± 0.033 | −0.025 ± 0.030 |
| joint training (ceiling) | 0.972 ± 0.009 | +0.020 ± 0.003 |
| **addressed memory** | **0.954 ± 0.008** | **−0.006 ± 0.005** |

The addressed memory stores every training pair and takes no gradient step. Its
lower triangle is flat: every part is between 0.92 and 0.98 after all ten
stages. Replay is the only MLP method that also retains; EWC and SI beat naive
on only 3 of 5 seeds. That contrast is the point of the benchmark: under a
change of input distribution with a shared target, remembering is a data
problem, and a few stored points solve it better than a parameter penalty.
Full matrices and sweeps: [docs/FORGETTING.md](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/FORGETTING.md).

## Limitations

This is a research prototype, not a drop-in continuous-learning solution.

* **Dense covariance is the state bill.** `d²` numbers per cell, independent of
  the value width. At the default `d=32, m=12, n=3` the table is 72 MiB; the
  regression configuration is 60 MiB for a scalar output. Sparse allocation,
  low-rank/diagonal covariance, or fp16 would change the size class.
* **Noise amplification.** The prediction error scales with `σ·||K⁺||`, which
  grows with occupancy. At `σ=0.3` the verified read is already at `4e-2` by
  three writers per cell and `0.23` by thirteen. No clean-key sizing rule
  survives a noisy deployment.
* **The variance score prefers crowded cells.** Beyond `c ≈ d/2` the verified
  read can lose to the plain own-address read. A crowding-aware score would need
  a statistic the cell does not keep.
* **No occupancy count and no rejection.** A cell is `W` and `P`; "have I seen
  this key?" is not recoverable. A rejection threshold has to use the variance,
  which is calibrated only within one occupancy regime.
* **No forgetting factor in the table.** `RLSMemCell.decay` exists, but
  `RLSCellBank` does not apply it: decaying every cell on every write is a
  whole-table operation. Non-stationary streams need a different mechanism.
* **Not differentiable.** Stores are indexed scatters and verified reads take an
  `argmin`. The feature map is random and untrained; training it would turn this
  into a different (gradient) model.
* **Synthetic evidence only.** Gaussian keys/values for recall, and a synthetic
  ten-part stream for regression. No trained embeddings, no real dataset, no
  downstream task.
* **The table is not data-efficient.** 0.954 ACC costs 5,120 stored pairs and
  60 MiB, against replay's 0.848 for 320 pairs. It is a ceiling, not a
  replacement for replay.

## Repository layout

```
src/pt_neural_memory/     package code and benchmark modules
tests/                    correctness, invariants, benchmark protocol
examples/                 two runnable end-to-end examples
docs/                     design reports and full measurement protocols
```

## Documentation

* [An addressed memory of RLS cells](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/ADDRESSED_MEMORY.md) — design,
  capacity law, noise measurements, sizing and the bits/blocks balance.
* [Evaluating and improving the memory cell](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/EVALUATION.md) — the RLS
  derivation, the failed projection baseline, retention and drift.
* [The ten-part regression forgetting benchmark](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/FORGETTING.md) —
  dataset card, protocol, ablations, literature and limitations.

## Citation

```bibtex
@software{solorzano2026ptneuralmemory,
  author  = {Jose Solorzano},
  title   = {pt-neural-memory: addressed RLS-cell associative memory and nonlinear regressor},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/jose-solorzano/pt-neural-memory},
  license = {MIT}
}
```

## License

MIT. See [LICENSE](https://github.com/jose-solorzano/pt-neural-memory/blob/main/LICENSE).
