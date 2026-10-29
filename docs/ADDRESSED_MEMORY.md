# An addressed memory of RLS cells

> **Status.** The memory is rebuilt on the cell module: `src/pt_neural_memory/mem_rls.py`
> gains `RLSCellBank`, a dense table whose cells carry exactly `RLSMemCell`'s
> two buffers, and `src/pt_neural_memory/addressed_memory.py` is now addressing plus routing
> on top of that bank. Sizing rules, both benchmarks, their results and the
> tests are updated. The previous mean-cell implementation is gone from the
> code; its last run is archived in the development checkout as
> `results/mean_cell_bench.json` (generated artifacts are not tracked) and is
> used below only as a baseline.

## Summary

Each address owns an `RLSMemCell`: a ridge linear map `value = weights @ key`
with its exact posterior covariance. A store trains the cell at the key's
address and every one-bit neighbour, so one pair now *trains* 39 cells (3
blocks × 13) instead of *writing a value* into 39 cells. A read is the cell's
map evaluated at the query, and the covariance is the read's confidence signal.

The consequences, measured at `d=32, d_v=16, m=12, n=3, λ=0.01`:

* **Capacity is set by the key dimension, not by the cell.** Clean recall is
  exact (`1e-7`) through `N=4096`, i.e. `c=13` writers per cell, and only
  degrades near `c ≈ d` (`5e-3` at `c=26`, `0.25` at `c=52`). A cell that
  stored a mean could hold one pair.
* **Collisions stop being the failure mode.** Three keys sharing an address are
  all recalled exactly; the old `(c-1)/c` law no longer applies, because the
  cell fits a map instead of averaging values.
* **Noisy cues are the failure mode.** Under `σ=0.3` the map amplifies the
  noise component that lies in the span of the cell's writers: the verified
  read gives `8.7e-3` at `N=64`, `4.2e-2` at `N=1024`, and collapses past
  `c≈13`. Verification by posterior variance is worth 4–20× over the query's
  own address, but it does not remove the amplification.
* **The cell is expensive.** `d² + d·d_v` numbers per cell (72 MiB at `m=12`,
  `d=32`) and ~300 µs per stored pair on one thread, against ~1.4 MiB and
  ~21 µs for the mean-cell design at the same `m` and `n`. At the same 4 GiB
  and 1% clean target the sizing rule now fits ~197k pairs instead of ~1.1M, so
  the table costs about 5.6× more bytes per stored pair.
* **Two or three blocks, the rest on address bits.** At `σ=0.25` the admissible
  occupancy saturates at `c≈5.6` by the third block and then falls, while the
  address width is neutral once `c` is fixed. A budget should therefore buy the
  largest `m` it can with two or three blocks (§5): 1 GiB at `d=64, d_v=16`
  holds 6.2k pairs at 5% error as 3 × 2^14 cells, 45 probes per query.
* **Capacity scales with the table, not with the cell.** One cell interpolates
  at most `key_dim` pairs (measured: 24 at `d=32`; the 99%-exact point is
  `0.75 d`). A table holds `c* · 2^m / (m+1)` with `c*≈16` at 99% exact, so
  `m=12, n=1` holds 5,041 pairs — **210× a single cell of the same key
  dimension** — and doubles per address bit. Blocks add none of it: they are
  replicas (§6).

That is the trade the architecture makes: far more capacity per cell and exact
interpolation, paid for in covariance state, write work, and noise sensitivity.
The per-byte comparison favours the mean cell; the per-cell comparison does not
even come close.

**As a regressor it needs a key map.** A cell is linear in its key, so a
nonlinear model is a feature map plus the table. Wrapped in 64 random Fourier
features, the memory is the `addressed` learner on the ten-part regression
stream of [FORGETTING.md](FORGETTING.md): R² = 0.95 with BWT ≈ 0, within 0.02
of joint training and 0.11 above replay, for 60 MiB of state and 5,120 stored
pairs. The same features and ridge in a two-cell table score 0.53, so the
table's capacity — not the feature map alone — is what the accuracy is bought
with.

## 1. The design

### 1.1 Addressing

Block `b` owns a fixed `m × d` projection `R_b` with orthonormal rows. The
address of key `k` in that block is the sign pattern of `R_b k` read as a
binary integer:

    a_b(k) = Σ_i 2^i [ (R_b k)_i ≥ 0 ]

For a key with i.i.d. standard-normal coordinates each bit is an independent
fair coin, so addresses are uniform and two keys share a block address with
probability `2^-m`. This part is unchanged from the earlier design and remains
the cheapest way to route a key to a table row: one projection, `m` sign tests.

### 1.2 The cell

A cell is an `RLSMemCell`, unchanged: model `v = W k + noise`, prior
`W ~ N(0, I/λ)`, posterior `P = (λI + Σ k kᵀ)⁻¹` and `W = P Σ k vᵀ`. `store`
applies the exact Gaussian update, `forward` returns `W k`. This is the Kalman
filter for that linear-Gaussian model: `P` is the error covariance and each
`store` is a measurement update. The table is therefore a bank of independent
Kalman filters, one per address.

`RLSCellBank` is the table. It holds the two buffers `weights` `(cells, d_v, d)`
and `covariance` `(cells, d, d)` and nothing else — no counts, no key or value
sums. A write is the same rank-1 update applied to every named row:

    P_i ← P_i − (P_i k)(P_i k)ᵀ / (1 + kᵀP_i k)
    W_i ← W_i + (v − W_i k)(P_i k)ᵀ / (1 + kᵀP_i k)

Rows naming the same cell are applied in successive rounds, exactly as if they
had been stored one at a time; rows naming different cells go in one batch.
`tests/test_rls_cell_bank.py` checks the bank against `RLSMemCell` for one batch
and for a sequence of single writes, and checks that the rounds reproduce
sequential storage.

Keeping the posterior rather than the sufficient statistics means a read needs
no solve: the prediction is `W_i k`, and the cell's confidence is the posterior
variance `kᵀP_i k`. It also means the cell cannot report how many writers it
has: a fully written cell and an unwritten one are distinguished only by
whether `W` is zero.

### 1.3 Writing

`store(key, value)` trains the cell at `a_b(k)` and every address within
`write_radius` flips of it, in every block:

    { a_b(k) XOR μ : popcount(μ) ≤ r_w }

Radius 1 is 13 cells per block, 39 per pair. Unlike the mean design, the copies
are not redundant: each neighbour cell sees the pair as one more constraint on
its own local map, and the query that lands on that neighbour gets a map fitted
partly on its own key.

The whole neighbourhood goes to the bank in one call. Within a mask two pairs
can collide; the bank sequences them.

**What one call touches.** A call with `B` pairs performs

    B · n · Σ_{j≤r_w} C(m, j)

cell updates — `39 B` in the default `n=3, m=12, r_w=1` configuration — and
touches fewer distinct cells, because colliding addresses merge updates. For
`B` pairs the expected number of distinct cells is

    n · 2^m · (1 − (1 − (m+1)/2^m)^B),

which matches the measurement to a couple of percent:

| batch B | cell updates | distinct cells | expectation |
|---|---|---|---|
| 1 | 39 | 39 | 39 |
| 8 | 312 | 312 | 309 |
| 64 | 2,496 | 2,278 | 2,262 |
| 256 | 9,984 | 6,812 | 6,842 |
| 1024 | 39,936 | 11,803 | 11,814 |

At `B=1024` a single call writes 96% of the 12,288-cell table, so batching a
store is also a whole-table operation. The formula needs uniform addresses,
which needs orthonormal projection rows, which is why `n_bits > key_dim` is now
rejected: with `d=8, m=12` the measured distinct count falls 15–20% below the
formula because the bits stop being independent.

### 1.4 Reading

`read_blocks` evaluates the map at the query's own address in each block;
`forward` averages the blocks.

`VerifiedMemory` probes every address within `read_radius` flips and keeps the
cell with the smallest posterior variance:

    s(c) = qᵀ P_c q

with the prediction `W_c q`. The score is the cell's own uncertainty about the
query: about one for a cell whose writers span the query, `‖q‖²/λ` for an
unwritten cell, and `‖q‖²/(c+λ)` for a cell crowded with unrelated keys. It
therefore prefers crowded cells — correct while a crowded cell still has the
information to answer, uninformative once crowding exceeds `d`. There is no
count penalty any more because there is no count.

Write and read radii add, so `w1 r1` survives two corrupted bits and `w1 r2`
three, at the cost of probes rather than cells.

### 1.5 State and cost

Per cell: `d_v·d` weights plus `d²` covariance, so `n·2^m·(d_v·d + d²)` numbers,
plus `n·m·d` for the projections. For `d=32, d_v=16, m=12, n=3` that is 72 MiB;
at `d=64` it is 240 MiB. The covariance dominates for every value width below
`d`, which is the single most important sizing fact about this design.

## 2. What changes when the cell is an RLS cell

| | mean cell | RLS cell |
|---|---|---|
| state per cell | `d_v + 1` | `d_v·d + d²` |
| pairs per cell at 1% clean | 0.04 | ~1 |
| error at `c` writers, clean | `(c-1)/c` | exact until `c ≈ d` |
| error under key noise | none beyond collisions | amplified by the map's conditioning |
| confidence | count (unused by the mean read) | posterior variance |
| write per pair (batch 64) | ~21 µs | ~300 µs |

The collision law that organised the earlier report, `(c-1)/c`, is gone: with
`c ≤ d` the map interpolates every stored pair, so the measured clean error at
`c=13` is `1e-6` where the law predicts `0.92`. The replacement law is
conditioning: the prediction error scales with `σ·‖K⁺‖`, and `‖K⁺‖` grows with
the number of writers, so noise robustness and capacity pull in opposite
directions.

## 3. Measurements

Protocol unless stated: `d=32`, `d_v=16`, `m=12`, `n=3`, `λ=0.01`, write radius
1, 3 seeds, error on a 512-query sample, relative squared error against the
stored values. `c` is the expected writers per cell, `N(m+1)/2^m`.

### 3.1 Load

| N | c | clean, own | clean, verified r1 | σ=0.3, own | σ=0.3, verified r1 |
|---|---|---|---|---|---|
| 64 | 0.2 | 1.24e-07 | 1.52e-07 | 1.46e-01 | 8.71e-03 |
| 256 | 0.8 | 1.38e-07 | 1.91e-07 | 1.78e-01 | 1.93e-02 |
| 1024 | 3.2 | 2.02e-07 | 3.80e-07 | 1.83e-01 | 4.25e-02 |
| 4096 | 13 | 1.08e-06 | 1.80e-05 | 3.07e-01 | 2.32e-01 |
| 8192 | 26 | 5.16e-03 | 1.88e-01 | 1.10e+00 | 1.64e+00 |

### 3.2 Occupancy knee

| writers per cell `c` | clean, own | clean, verified | σ=0.3, own | σ=0.3, verified |
|---|---|---|---|---|
| 0.2 | 1.21e-07 | 1.52e-07 | 1.92e-01 | 6.76e-03 |
| 0.8 | 1.38e-07 | 1.90e-07 | 1.91e-01 | 1.90e-02 |
| 3.2 | 2.00e-07 | 3.81e-07 | 1.95e-01 | 4.14e-02 |
| 6.5 | 3.22e-07 | 1.10e-06 | 2.24e-01 | 7.35e-02 |
| 13 | 1.08e-06 | 1.80e-05 | 3.07e-01 | 2.32e-01 |
| 26 | 5.16e-03 | 1.88e-01 | 1.10e+00 | 1.64e+00 |
| 52 | 2.50e-01 | 5.92e-01 | 6.96e-01 | 6.82e-01 |

Clean capacity is flat to `c≈13` and breaks between 26 and 52, i.e. at the key
dimension. At `d=64` the same protocol still gives `1.4e-07` at `c=26`, which is
the evidence that the knee tracks `d`, not `m`.

### 3.3 Noise

At `N=64`:

| σ | own | verified r1 | verified r2 | guided6 |
|---|---|---|---|---|
| 0 | 1.24e-07 | 1.52e-07 | 1.52e-07 | 1.49e-07 |
| 0.1 | 1.88e-02 | 1.17e-03 | 1.27e-03 | 1.18e-03 |
| 0.2 | 7.20e-02 | 4.09e-03 | 4.81e-03 | 4.69e-03 |
| 0.3 | 1.46e-01 | 8.71e-03 | 1.02e-02 | 1.05e-02 |
| 0.5 | 3.68e-01 | 2.98e-02 | 2.56e-02 | 2.69e-02 |
| 0.75 | 5.77e-01 | 1.79e-01 | 6.23e-02 | 5.33e-02 |
| 1.0 | 7.23e-01 | 3.38e-01 | 1.54e-01 | 1.23e-01 |

Verification is worth 4–20× over the query's own address. The wide-radius and
guided reads overtake radius 1 only at high noise, where most keys need more
than two bits of correction.

### 3.4 Ridge

At `N=1024`:

| λ | clean, own | clean, verified | σ=0.3, own | σ=0.3, verified |
|---|---|---|---|---|
| 1 | 1.79e-03 | 4.50e-03 | 1.93e-01 | 4.22e-02 |
| 0.1 | 2.00e-05 | 5.52e-05 | 1.84e-01 | 4.19e-02 |
| 0.01 | 2.02e-07 | 3.80e-07 | 1.83e-01 | 4.25e-02 |
| 0.001 | 2.02e-09 | 2.10e-09 | 1.83e-01 | 4.25e-02 |

`λ` sets the clean-recall shrinkage, `λ/(‖k‖²+λ)`, and nothing else: it cannot
damp the noise amplification, because that comes from the inverse of the key
Gram, not from the ridge. `0.01` is the default: exact recall at `d=32` and
essentially no effect on noisy reads.

### 3.5 Blocks and read strategy

At `N=1024`, `σ=0.3`, verified radius 1: `n=1` gives `1.51e-01`, `n=2` `4.70e-02`,
`n=3` `4.25e-02`, `n=5` `4.61e-02`, `n=9` `5.39e-02`. Blocks pay for themselves
once, then the wider argmin-min over blocks starts to pick crowded cells;
averaging the selected blocks (`CellRead.mean`) is better at `n=9`
(`3.12e-02`). Clean keys are unaffected at `3e-07`.

Read strategy at `N=1024`: clean, every strategy is exact (`2e-7` to `4e-7`);
at `σ=0.3`, own address `1.83e-01`, `r1` `4.25e-02`, `r2` `4.85e-02`, guided6
`4.47e-02`. At this noise the radius-1 ball already covers the failures; going
wider only adds crowded candidates.

### 3.6 Cost

One CPU thread, `d=32`, `d_v=16`, `m=12`, `n=3`, 1024 pairs, batch 64:

| write µs/pair | read µs/query (own) | read µs/query (verified r1) | state |
|---|---|---|---|
| 303 | 5.5 | 64.8 | 72 MiB |

39 cells written per pair and 39 probes per verified query. The write is
`O(d²)` per written cell and is the dominant cost; the verified read is
dominated by gathering 39 covariance matrices per query.

## 4. Sizing

`src/pt_neural_memory/memory_sizing.py` now uses the cell's real state,

    n · (2^m · (d_v·d + d²) + m·d) numbers,

and a capacity law taken from the occupancy knee,

    N = f · d · 2^m / (m + 1),   f = 0.8 by default,

where `f` is the measured writers per cell at the target error as a fraction of
`d`. `f` is not a design constant: §3.2 puts it near `0.8` at 1% clean, and the
noise columns put it near `0.05` at `σ=0.3`. Pass the value measured for the
deployment's noise level.

For 4 GiB, `d=64`, 1% clean, 2–8 blocks:

| value dim | blocks | bits | cells/block | capacity | state GiB | KiB/pair |
|---|---|---|---|---|---|---|
| 16 | 3 | 16 | 65,536 | 197,379 | 3.75 | 19.9 |
| 64 | 3 | 15 | 32,768 | 104,857 | 3.00 | 30.0 |
| 256 | 3 | 14 | 16,384 | 55,924 | 3.75 | 70.3 |
| 768 | 2 | 13 | 8,192 | 29,959 | 3.25 | 113.8 |

Capacity *falls* as the value widens, which is the opposite of the mean-cell
design: every cell pays `d²` for its covariance whatever its value width, so a
wider value forces a narrower address to fit the budget, and the address width
is what buys capacity. The 768-number value keeps only 30k pairs where the
16-number one keeps 197k.

For comparison, the archived mean-cell sizing rule put ~1.1M pairs in the same
4 GiB at the same nominal 1% target, i.e. about 5.6× more pairs per byte. The
two capacity laws are fitted to different protocols, so treat the factor as an
order of magnitude, not a precise constant.

This section sizes a configuration; it does not choose between the two ways to
spend a fixed budget, address bits or blocks. §5 does that from the measured
error law, and finds that two or three blocks are all that pay.

## 5. Balance of address bits and blocks

Protocol for this section: `d=32`, `d_v=16`, `λ=0.01`, `σ=0.25`, write and read
radius 1, 3 seeds, 512-query sample, verified read. (`src/pt_neural_memory/benchmarks/balance_bench.py`.)

### 5.1 The error law

Error against writers per cell, one series per block count:

| c | n=1 | n=2 | n=3 | n=4 | n=5 | n=8 |
|---|---|---|---|---|---|---|
| 0.25 | 5.63e-02 | 1.18e-02 | 7.91e-03 | 8.48e-03 | 7.19e-03 | 8.60e-03 |
| 0.5 | 8.98e-02 | 1.18e-02 | 9.63e-03 | 1.10e-02 | 1.19e-02 | 1.40e-02 |
| 1 | 9.25e-02 | 1.63e-02 | 1.27e-02 | 1.46e-02 | 1.62e-02 | 1.83e-02 |
| 2 | 1.03e-01 | 2.59e-02 | 2.02e-02 | 2.24e-02 | 2.40e-02 | 2.68e-02 |
| 4 | 1.06e-01 | 3.63e-02 | 3.44e-02 | 3.78e-02 | 3.92e-02 | 4.34e-02 |
| 8 | 1.65e-01 | 7.24e-02 | 7.28e-02 | 7.86e-02 | 8.10e-02 | 8.76e-02 |

Two structural facts fall out, and they are what makes a sizing rule possible:

* **Address width is nearly neutral at fixed occupancy.** At `c=2, n=3`, the
  error is `1.82e-2, 2.09e-2, 2.02e-2, 2.21e-2` for `m = 8, 10, 12, 13` — a 20%
  spread, within seed noise. Once `c` is fixed, the address width only sets how
  many cells are needed to reach that `c`. This is why capacity can be written
  as `c · 2^m / (m + 1)` without a correction for `m`.
* **Blocks saturate at three.** At `c=2` the error is `1.03e-1, 2.59e-2,
  2.02e-2, 2.40e-2, 2.77e-2` for `n = 1, 2, 3, 5, 9`; past three, another block
  makes it slightly worse. The variance score prefers a crowded cell, and more
  blocks give it more chances to find one. (Averaging the per-block reads
  instead of taking the best block keeps improving: `1.77e-2` at `n=9`. The
  default `forward` takes the best block.)

### 5.2 The approximation

Splitting a budget `B` with `n` blocks of `2^m` cells fixes the width:
`n · 2^m · cell_bytes ≤ B`, where `cell_bytes = 4 (d_v d + d^2)`. The error law
then gives the largest admissible occupancy `c_max(n)`, and capacity follows:

    capacity(n) = c_max(n) · 2^m / (m + 1),   m = floor(log2(B / (n · cell_bytes)))

so the block count is the only free variable. Capacity per byte is proportional
to `c_max(n) / (n (m + 1))`, and `c_max` saturates at three blocks, so:

> **Use two or three blocks. Use the largest address width the budget allows,
> capped at `key_dim`. Never spend bytes on a fourth block.**

Two, rather than three, when dropping one block buys an extra address bit: that
doubles `2^m` for a `c_max(2)/c_max(3) ≈ 0.98` occupancy penalty. For example
at `d=64, d_v=768`, `n=2` fits `m=13` where `n=3` only fits `m=12`, and the
sizing table below picks two there. Otherwise three wins.

Measured admissible occupancy (`σ=0.25`), interpolated from the grid above:

| blocks n | 5% target | 2% target |
|---|---|---|
| 1 | 0.22 | 0.09 |
| 2 | 5.51 | 1.36 |
| 3 | 5.65 | 1.97 |
| 4 | 5.21 | 1.66 |
| 5 | 5.04 | 1.45 |
| 8 | 4.60 | 1.17 |

### 5.3 Resulting splits

`d=64`, `d_v=16`, target 5%:

| budget | blocks | bits | capacity | probes/query |
|---|---|---|---|---|
| 16 MiB | 3 | 8 | 160 | 27 |
| 64 MiB | 3 | 10 | 525 | 33 |
| 256 MiB | 3 | 12 | 1,780 | 39 |
| 1 GiB | 3 | 14 | 6,171 | 45 |

At a 2% target the same budgets hold `56, 183, 620, 2,151` pairs. Capacity
grows almost exactly linearly in budget — one extra address bit per doubling,
and the extra pairs per bit are eaten by the `(m+1)` write factor.

Measuring the recommendation instead of trusting it:

| budget | m | n | c | capacity | target | measured error |
|---|---|---|---|---|---|---|
| 16 MiB | 8 | 3 | 5.65 | 160 | 5% | 4.3% |
| 64 MiB | 10 | 3 | 5.65 | 525 | 5% | 4.5% |

`balanced_layout` in [src/pt_neural_memory/memory_sizing.py](../src/pt_neural_memory/memory_sizing.py)
implements this rule, with the measured tables as
`ADMISSIBLE_WRITERS_5PCT` and `ADMISSIBLE_WRITERS_2PCT`.

### 5.4 Caveats

* The occupancy tables are for `σ=0.25`. There is no noise scaling law: `c_max`
  falls steeply with `σ` (at `σ=0.3`, `d=32`, the 1% point is below `c=1`, see
  §3.3), so a deployment must re-measure the grid for its own noise level
  rather than reuse these numbers.
* The saturation at three blocks is a property of the variance score, not of
  the ensemble. Switching `forward` to `CellRead.mean` moves the optimum to
  more blocks, and the whole table would need redoing.
* Capacity per byte falls as `d_v` grows (the covariance is fixed per cell), so
  wide values prefer fewer address bits and, at the extreme, two blocks.

## 6. Information capacity: one cell versus the table

**Yes, by about two orders of magnitude at `m=12` — but that is capacity, not
efficiency.** The two claims are worth separating.

### 6.1 One cell is bounded by its rank

The cell's map is `d_v × d`, so it has at most `d` independent key directions.
With `c > d` stored keys the key matrix is linearly dependent and arbitrary
values are inconsistent, so the cell can interpolate at most `d` associations —
`d · d_v` numbers, which is exactly the size of its map. Measured
(`d=32`, `d_v=16`, `λ=0.01`, clean, 99% of pairs within `1e-4` relative error):

| pairs stored in one cell | 8 | 16 | 24 | 32 | 40 | 64 |
|---|---|---|---|---|---|---|
| exact fraction | 1.00 | 1.00 | 0.99 | 0.10 | 0.00 | 0.00 |

The cap is at `d`, and the 99% point at `0.75 d`.

### 6.2 The table multiplies it by the number of cells

The exact fraction is a function of the occupancy `c` alone, at every address
width:

| m | cells | c=8 | c=16 | c=24 | c=32 | pairs at 99% exact |
|---|---|---|---|---|---|---|
| 8 | 256 | 1.00 | 0.99 | 0.79 | 0.27 | 455 |
| 10 | 1,024 | 1.00 | 0.99 | 0.77 | 0.24 | 1,489 |
| 12 | 4,096 | 1.00 | 0.99 | 0.76 | 0.24 | 5,041 |

Capacity is `c* · 2^m / (m + 1)` with `c* = 16` at 99% exactness, so it doubles
per address bit up to the `(m+1)` factor: the measured ratios for `m: 8→10→12` are
`3.27` and `3.39`, against `4 · 9/11 = 3.27` and `4 · 11/13 = 3.38`. Against one cell of the same `d=32`,
`m=12` holds **210×** more (5,041 against 24). The 1% *mean*-error capacity used
in §3–§5 is larger (`c≈26` at `d=32`) because it averages over pairs instead of
requiring each pair to be exact; the two numbers are different criteria, not a
contradiction.

**Blocks add no capacity.** The exact fraction is identical for one and three
blocks at every occupancy (1.00/0.99/0.77/0.24/0.03 at both), which is the same
statement as `c` not depending on `n`: blocks are replicas that buy noise
robustness, not storage.

### 6.3 But not per byte

Each pair is trained into `m+1` cells (one block in this comparison), and
the covariance is `d²` of the `d_v · d + d²` state. Pairs per stored number:

| | pairs / number |
|---|---|
| one cell, `d=32` | 0.016 |
| table, `m=12`, `c=16` | 0.0008 |

The table is 20× less information-dense. It wins only because you can buy
cells: at a fixed float budget, give one cell the same state as a one-block
table and its width is `D = sqrt(state)`:

| table m | cell width D | one cell, 99% exact | table, 99% exact |
|---|---|---|---|
| 8 | 619 | 588 | 455 |
| 10 | 1,246 | 1,180 | 1,489 |

The single cell wins at `m=8` and the table wins at `m=10`, and from there the
gap grows without bound, because the table's capacity is **linear** in its state
(`c* · 2^m/(m+1)`) while one cell's is **square-root** (`0.95 sqrt(state)`).
The crossover for these dimensions is around `2^10` cells.

That equal-state comparison assumes the key dimension can be chosen freely. For
a fixed application `d` — 32-dimensional keys, say — the single-cell alternative
does not exist at all: `d` is not a budget you can spend, and the only way to
hold more than `0.75 d` associations is more addressed cells.

## 7. Limitations

* **Noise amplification.** The map's prediction error is `σ·‖K⁺‖`, which grows
  with occupancy. At `σ=0.3` the verified read is already at `4e-2` by `c≈3`
  and `0.23` by `c=13`, so capacity per byte is far below the clean-key figure.
* **The variance score prefers crowded cells.** Beyond `c≈d/2` the verified
  read can lose to the plain own-address read (`c=26`: `1.9e-01` against
  `5.2e-03` clean). A crowding-aware score would need a statistic the cell does
  not carry.
* **No occupancy, no reason to reject.** The cell has weights and covariance
  only: an unwritten cell is `W=0`, and "have I seen this key" is not
  recoverable. A rejection threshold has to use the variance, which is
  calibrated only within one occupancy regime.
* **No forgetting.** `RLSMemCell.decay` is not in the bank; applying it would
  touch every cell on every write.
* **Dense covariance.** `d²` per cell makes the table only viable for small
  `m` or wide values; the 4 GiB sizing is `2^16` cells per block where the
  mean-cell design reached `2^23`.
* **Not differentiable.** Writes are indexed scatters and the read takes an
  `argmin`.
* **Synthetic data only.** Gaussian keys and values, no trained embeddings. The
  regression use in [FORGETTING.md](FORGETTING.md) is synthetic too, and its
  key map is random and untrained; the cell itself only ever fits a linear map.

## 8. What I would do next

1. **Sparse cells.** Allocate a cell on first write instead of a dense
   `2^m`-row table. The state would then grow with occupancy rather than with
   the address space, and `m` could be raised without paying for `d²` per
   unoccupied row. This is the single change with the largest effect on bytes
   per pair.
2. **A noise-aware read.** `W q` is the right answer only if `q` is the stored
   key. If the cell also kept `Σk` (one `d`-vector), it could read at the
   prototype or shrink `q` toward it by the measured noise level, which would
   remove most of the amplification at noisy cues — at the cost of the
   within-cell discrimination that makes clean reads exact.
3. **Crowding-aware selection.** The posterior variance is the cell's
   uncertainty, not its specificity. A score that also penalises `tr(P)` low
   (many writers) would fix the high-`c` regression of the verified read.
4. **Low-rank or quantised covariance.** `P` is the whole state bill; a
   diagonal plus low-rank approximation, or fp16, would make `m=18` reachable
   again.
5. **Hybrid per block.** A mean cell for the value and an RLS cell for the
   local map, with the mean used when the variance says the map is
   ill-conditioned. The two failure modes are disjoint, so an ensemble of the
   two readouts is the obvious next experiment.
6. **A trained key map.** The regression arm's feature map is random. Training
   it, or distilling it from an MLP's hidden layer, would separate "the cell is
   a good closed-form readout" from "random features of this input are enough".

## Reproduce

```bash
python -m pytest -q tests            # 116 tests: cell, bank, memory, regressor, sizing
python -m pt_neural_memory.benchmarks.addressed_bench # §3.1-3.4, §3.6 + results/addressed_bench.json
python -m pt_neural_memory.benchmarks.scaling_bench   # §3.2, §3.5 + results/scaling_bench.json
python -m pt_neural_memory.benchmarks.balance_bench   # §5 error law and budget split + results/balance_bench.json
python -m pt_neural_memory.benchmarks.capacity_bench  # §6 cell versus table capacity + results/capacity_bench.json
python -m pt_neural_memory.memory_sizing      # §4 and §5.3 tables
```
