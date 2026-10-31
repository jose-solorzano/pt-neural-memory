# pt-neural-memory

[![PyPI](https://img.shields.io/pypi/v/pt-neural-memory.svg)](https://pypi.org/project/pt-neural-memory/)
[![Python](https://img.shields.io/pypi/pyversions/pt-neural-memory.svg)](https://pypi.org/project/pt-neural-memory/)
[![tests](https://github.com/jose-solorzano/pt-neural-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jose-solorzano/pt-neural-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jose-solorzano/pt-neural-memory/blob/main/LICENSE)

A fast associative memory built from **addressed recursive-least-squares (RLS)
cells** — each an exact Kalman filter — and a nonlinear regressor when keyed by
random features.

The memory stores associations in one pass with an exact closed-form update. Its
capacity is a configuration choice: the table can be sized from a few hundred
cells to millions, and capacity grows with it. Because each address owns its own
local map, later writes do not overwrite earlier ones. On a ten-part continual
regression stream the memory ends at **ACC 0.954** and **BWT −0.006** — within
0.02 of joint training and 0.11 above replay — with no gradient step.

Each cell is a Kalman filter for the linear-Gaussian model
`value = W · key + noise`: `store` is a measurement update, `covariance` is the
filter's error covariance, and a read returns the posterior mean. The addressed
table is therefore a Kalman-filter memory: a bank of independent filters routed
by address. The same filter's `qᵀ P q` is the uncertainty signal used by the
verified read.

## What this is

Three composable layers, all in PyTorch:

| Layer | What it does |
|---|---|
| `RLSMemCell` | A linear map `value = W @ key` fitted by the exact RLS/Kalman posterior update, including its `d × d` covariance. This is the Kalman filter for `value = W key + noise`; one pass reaches the least-squares fit. |
| `AddressedMemory` | A bank of Kalman filters: `n_blocks` tables of `2**n_bits` cells, selected by the sign bits of a random projection. A store trains the key's cell and its one-bit neighbours; a read is a lookup. `VerifiedMemory` adds a read-side probe. |
| `AddressedRegressor` | A random Fourier feature map (optionally ReLU features or raw input) composed with the table. The result is a nonlinear regressor fitted in closed form; `kind="raw", use_bias=False` is the direct memory. |

It is a **lookup and regression** structure, not a trained layer: stores are
indexed updates, reads are gathers, and the feature map is random. Stores run
under `torch.no_grad` by default; `store(..., grad_enabled=True)` opts into
backpropagation through a write. That default is what makes it useful as memory
(no shared weight to overwrite), and the discrete address choice is what limits
end-to-end feature learning.

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

## Memory or regressor?

`AddressedMemory` is already a regression system. A cell fits a linear map from
its key to its value, so giving it `(x, y)` fits a piecewise-linear model of `y`
in `x`, with the address bits deciding which piece applies:

```python
import torch
from pt_neural_memory import AddressedMemory

x = torch.randn(4096, 6)
y = x @ torch.randn(6, 1)        # a linear target

memory = AddressedMemory(
    key_dim=x.shape[1],
    value_dim=1,
    n_blocks=3,
    n_bits=min(12, x.shape[1]),  # n_bits <= key_dim
    write_radius=1,
    rr_lambda=0.01,
    seed=0,
)
memory.store(x, y)
prediction = memory(x)
```

`AddressedRegressor` is the same memory preceded by a key map `φ(x)`.
`AddressedRegressor(..., kind="raw", use_bias=False)` is behaviorally equivalent
to direct memory; `kind="raw", use_bias=True` appends an intercept. The default
`kind="rff"` maps a low-dimensional input into a wider feature key. More
generally, `AddressedMemory` is the object to use with a custom key map: compute
`φ(x)` yourself and store/read it directly.

Use `AddressedMemory` directly when:

* `x` is already a good key — embeddings, one-hot vectors, engineered features,
  high-dimensional descriptors;
* the target is linear, or approximately linear within one address cell;
* you want the smallest, fastest state: no feature matrix, no feature compute,
  and a `key_dim` equal to the input dimension;
* you want each cell's local map to be directly inspectable.

Use `AddressedRegressor` when:

* the target is nonlinear in a low-dimensional `x`, and a piecewise-linear fit
  on raw `x` would need more pieces than the address space allows — `n_bits` is
  capped at `key_dim` (and at 20), so a 6-D input has at most 64 addresses per
  block;
* you need more address bits (hence more capacity) than the raw input dimension
  permits; with the defaults, the feature map expands the key to
  `width + 1 + input_dim`;
* you want the kernel-like random Fourier or ReLU features, with the intercept
  and linear base already included;
* accuracy matters more than the state difference.

The tradeoff is stark on the ten-part stream (seed 0, 10 × 512 stored pairs):

| model | key dim | address bits | ACC | BWT | state |
|---|---:|---:|---:|---:|---:|
| direct memory, raw 6-D input | 6 | 6 | 0.431 | −0.096 | 0.03 MiB |
| direct + intercept | 7 | 7 | 0.592 | −0.059 | 0.08 MiB |
| `AddressedRegressor` (RFF) | 71 | 10 | **0.948** | **−0.010** | 60 MiB |

This comparison is runnable as
[`examples/direct_vs_regressor.py`](https://github.com/jose-solorzano/pt-neural-memory/blob/main/examples/direct_vs_regressor.py).

The direct-memory BWT is negative because six address bits put hundreds of
writers in every cell: later parts keep changing the local fits. That is
capacity exhaustion, not weight overwriting. On a purely linear 6-D target the
same direct memory reaches RSE `2e-10` at 5,120 pairs — every writer imposes the
same constraint, so crowding does not matter. The feature map buys the ability
to represent a nonlinear target, at the cost of a 71-dimensional covariance and
a much larger table.

`AddressedMemory` also stores vector targets (`value_dim > 1`), so the same
interface covers multi-output regression.

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

### The cell is a Kalman filter

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

For one pair the gain is the Kalman gain `g = P k / (1 + kᵀ P k)`; `P` is the
filter's error covariance and `qᵀ P q` is the posterior variance of the answer
at `q`. `RLSMemCell.decay < 1` inflates `P` before each update, which is
exponential forgetting for a non-stationary stream (the bank does not expose
it). `VerifiedMemory` uses this covariance to choose the least-uncertain filter
among neighbouring addresses, so the read is driven by the filter's own
uncertainty rather than a separate score.

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

Stores are no-grad by default. Pass `grad_enabled=True` to keep the autograd
graph through the written keys and values; reads follow the ambient autograd
mode. The address choice itself is discrete (`≥ 0` sign bits), so gradients do
not flow through the routing decision.

A plain read evaluates the map at the query's own address in each block and
averages. `VerifiedMemory` instead probes the addresses within `read_radius`
flips and keeps the cell with the smallest posterior variance:

```
s(c) = qᵀ P_c q.
```

### Why there is no catastrophic forgetting

The mechanism is two things, not one: **local addressing** and **non-destructive
local updates**.

A store only touches the key's cell and its one-bit neighbours — 39 cells per
pair at the default settings — not a shared global parameter. With enough address
bits, different parts mostly write different cells, and the random feature map is
frozen, so there is no shared representation drifting underneath. The
[address-width sweep](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/FORGETTING.md#44-the-addressed-memory)
in the forgetting report makes this half visible: BWT is −0.217 with 2 cells per
block (5,120 writers per cell), −0.006 with 1,024 cells (55 writers), and −0.001
with 4,096 cells (16 writers).

But address locality alone would not be enough. Once a cell is touched, its whole
local map changes; what keeps the old association is that the update is the exact
Kalman/RLS posterior over *all* pairs written to the cell. Old pairs remain in the
normal equations, the new pair is added as another constraint, and the map is
refit jointly rather than overwritten. Even at 2 cells per block, where every
pair collides, BWT is −0.217 rather than the −2.1 of naive fine-tuning: the joint
fit retains as much as the occupancy allows.

That is why this is not EWC-style parameter isolation: there are no importance
weights, gradient masks, or per-task subnets. It is local in address space, and
non-destructive inside each cell.

The qualification is capacity. Once writers per cell exceed what the key
dimension can interpolate, even the joint fit cannot satisfy every constraint,
and old recall degrades. The direct raw 6-D memory in the table above shows it:
a 6-bit address space is too small for the ten-part stream, and its BWT is −0.096
even though no weight vector is shared. That is capacity exhaustion, not
overwriting.

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
| joint training (continual ceiling) | 0.972 ± 0.009 | +0.020 ± 0.003 |
| **addressed memory** | **0.954 ± 0.008** | **−0.006 ± 0.005** |

The addressed memory stores every training pair and takes no gradient step. Its
lower triangle is flat: every part is between 0.92 and 0.98 after all ten
stages. Replay is the only MLP method that also retains; EWC and SI beat naive
on only 3 of 5 seeds. That contrast is the point of the benchmark: under a
change of input distribution with a shared target, remembering is a data
problem, and a few stored points solve it better than a parameter penalty.
Full matrices and sweeps: [docs/FORGETTING.md](https://github.com/jose-solorzano/pt-neural-memory/blob/main/docs/FORGETTING.md).

### Training speed

The memory has no optimizer and, by default, no backward pass: fitting is a
sequence of closed-form stores. On the ten-part stream (5,120 training pairs),
one fit takes about **3.3 s** on four CPU threads — about as long as 300-step
fine-tuning or replay, and roughly 2.5–2.9× faster than EWC/SI and the 800-step
joint reference. The price is state: 60 MiB versus 18 KiB.

```bash
python -m pt_neural_memory.benchmarks.training_cost
```

| method | fit (s) | per part (ms) | training work | model state |
|---|---:|---:|---|---:|
| naive fine-tuning | 3.4 | 340 | 3,000 Adam steps | 18 KiB |
| EWC | 8.7 | 870 | 3,000 steps + Fisher anchors | 18 KiB + anchors |
| SI | 9.3 | 930 | 3,000 steps + path integral | 18 KiB + anchors |
| replay | 3.6 | 360 | 3,000 steps, 32 samples/part | 18 KiB + 9 KiB |
| joint | 9.5 | 950 | 8,000 steps (larger budget) | 18 KiB + 140 KiB |
| **addressed** | **3.3** | **330** | **5,120 stores × 33 cells** | **60 MiB** |

Conditions: CPU, float32, four threads, seed 0, median of three fresh fits;
data generation, model construction and the 10 × 10 evaluation are excluded.
The timing varies by 10–20% on a shared machine, so treat the table as
representative rather than a benchmark record. Model state excludes the MLP's
Adam moments (two per parameter, 2 × 18 KiB during training); the addressed
memory has no optimizer state.

This is total fit time on a small stream, not throughput. At the default 71-D
feature key the memory costs about **0.6 ms per stored pair**; its time is linear
in the number of pairs, while the MLP methods take a fixed step budget. On a
much larger stream, storing every pair will dominate. Threading also matters:
on one thread the addressed fit is ~4.5 s versus ~3.1 s for naive, because the
per-block work parallelizes; the table uses the benchmark's four-thread setting.
EWC and SI pay for Fisher/path-integral bookkeeping, and joint gets a larger
budget, so its time is a reference rather than an equal-budget comparison.

#### From-scratch MLP

The table above is the continual-learning protocol, not a from-scratch fit. Pool
all 5,120 training pairs, shuffle them, and train the same 2 × 64 MLP once from
random initialisation (Adam, lr 1e-3, batch 128, no task boundaries):

| steps | fit (s) | ACC |
|---:|---:|---:|
| 300 | 0.4 | 0.601 |
| 1,000 | 1.2 | 0.859 |
| 3,000 | 3.5 | **0.949** |
| 8,000 | 9.3 | 0.975 |
| 16,000 | 18.6 | 0.984 |
| **addressed memory** | **3.3** | **0.948** |

For about the same fit time, the 3,000-step MLP reaches the same seed-0 accuracy
as the memory. A longer-trained MLP beats it — 0.975 at 8,000 steps, 0.984 at
16,000 — with 18 KiB of weights instead of 60 MiB. The memory is not an
accuracy-per-byte win against offline training.

What it buys is a different mode of operation: the from-scratch MLP sees all the
pairs at once and needs a fresh optimizer run; in a continual stream, retaining
and retraining on the growing pool is the `joint` baseline. The memory stores
each pair once, takes no gradient step, and is immediately queryable. The
`joint` row is the ceiling within the continual protocol, not the offline
ceiling; the from-scratch MLP is the offline oracle.

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
* **Differentiability is opt-in and stops at the address choice.** Stores
  default to no-grad; `grad_enabled=True` keeps the graph for the cell, bank,
  memory, and regressor. The feature map is still random and untrained, and
  `VerifiedMemory`'s `argmin` is a discrete selection, so you can backprop
  through a fixed address's map but not through the choice of cell.
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
examples/                 three runnable end-to-end examples
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
  title   = {pt-neural-memory: addressed RLS/Kalman-cell associative memory and nonlinear regressor},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/jose-solorzano/pt-neural-memory},
  license = {MIT}
}
```

## License

MIT. See [LICENSE](https://github.com/jose-solorzano/pt-neural-memory/blob/main/LICENSE).
