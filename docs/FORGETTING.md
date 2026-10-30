# The ten-part regression forgetting benchmark

> **Status.** Benchmark documentation, v2. The dataset generator is
> [src/pt_neural_memory/forgetting_data.py](../src/pt_neural_memory/forgetting_data.py), the protocol and
> baselines are [src/pt_neural_memory/benchmarks/forgetting_bench.py](../src/pt_neural_memory/benchmarks/forgetting_bench.py),
> and every result table below is reproduced by
> `python -m pt_neural_memory.benchmarks.forgetting_bench` into
> `results/forgetting_bench.json`; the timing line
> in §4.4 is machine-dependent wall-clock. The stream is synthetic; no real
> dataset is used.
>
> v2 adds the addressed-memory arm: the `addressed` method and
> [src/pt_neural_memory/addressed_regressor.py](../src/pt_neural_memory/addressed_regressor.py). It is the
> only method that is not an MLP, and the comparison is the point: the memory is
> the same data-side answer as replay, taken to the limit of storing every pair.
>
> The data is published as a Hugging Face dataset,
> **[ccde/forgetting-regression](https://huggingface.co/datasets/ccde/forgetting-regression)**
> (five seeds × train/test; rebuilt by `src/pt_neural_memory/benchmarks/export_forgetting.py`). The
> published `seed_0` CSVs reproduce `regression_stream(seed=0)` exactly: every
> input, noisy training target and clean target matches to the last float32.
>
> Sections 1–3 are the reference (what the data is, how a run is scored, how to
> add a method); §4–5 are the measured results; §6–8 are the literature, limits
> and recommendation; §9 is the test inventory.

## Summary

Ten-part regression stream: one shared nonlinear target $f:\mathbb{R}^6\to\mathbb{R}$,
ten input distributions with disjoint support, one MLP with one scalar output and
no task identifier. Parts are trained in order and every part is evaluated after
every stage. Use it to compare continual-learning methods when the target is a
scalar regression and the thing that shifts is the input distribution.

The benchmark is not subtle. A plain MLP reaches $R^2 \approx 0.85\text{–}0.93$
on the part it is currently trained on and then loses essentially all of it:
after the tenth part, the first nine parts sit at $R^2 \in [-5.6, 0.25]$, worse
than predicting the per-part mean. Replay of 32 samples per part (6% of the
data, 320 points total) removes the forgetting ($\text{BWT} = -0.025$), while the
two parameter-space defences do not reliably help: EWC and SI beat plain
fine-tuning on only 3 of 5 seeds, and their mean gain comes from the one seed
where fine-tuning diverged worst. Joint training on all data seen so far is the
continual-protocol ceiling at $R^2 = 0.972$; a from-scratch MLP on the pooled
data is higher still (§4.4).

That is the useful result for this repository: **under a shift of the input
distribution with a shared target, remembering is a data problem, not a
parameter-regularisation problem.** A few hundred stored points solve it; a
Fisher-weighted quadratic penalty does not.

The addressed memory takes that conclusion to its limit. Keyed by 64 random
Fourier features and read through three blocks of $2^{10}$ RLS cells, it stores
every training pair and fits each cell's local map in closed form — no gradient
step, no shared weight to overwrite. It ends at ACC 0.954 and BWT −0.006, within
0.02 of joint training and 0.11 above replay, and its learning accuracy (0.960)
is the best in the table. It pays with state and data: 60 MiB and all 5,120
pairs, against the MLP's 18 KiB and replay's 320 pairs. A table of two cells per
block with the same features scores 0.53, so the accuracy comes from the table's
capacity to keep many local maps, not from the random features alone.

## Quickstart

```bash
python -m pytest -q tests/test_forgetting_data.py tests/test_forgetting_bench.py tests/test_addressed_regressor.py   # 45 tests, ~10 s
python -m pt_neural_memory.benchmarks.forgetting_bench          # tables below + results/forgetting_bench.json, ~30 min
```

A single run from Python, without the suite:

```python
from pt_neural_memory.benchmarks.forgetting_bench import benchmark, summarize

run = benchmark("replay", seed=0, buffer_per_task=32)
summarize(run.matrix, run.baseline)
# {'acc': 0.861, 'bwt': -0.036, 'la': 0.893, 'fwt': 1.803}   (this checkout, seed 0)

run = benchmark("addressed", seed=0)
summarize(run.matrix, run.baseline)
# {'acc': 0.948, 'bwt': -0.010, 'la': 0.957, 'fwt': -0.088}   (this checkout, seed 0)
```

`benchmark()` returns the full $10\times10$ matrix, so one configuration can be
inspected in a few seconds.

## 1. What the stream is

`regression_stream()` builds ten parts. Part $t$ draws

$$
x = c_t + s\, z_t\, L_t^{\top}, \qquad y = f(x) + \varepsilon,
$$

with:

* $c_t$ — a part-specific mean on the sphere of radius 3 in $\mathbb{R}^6$,
  placed by greedy farthest-point sampling so that the means are as far apart as
  the radius allows (minimum pairwise distance 3.80 for seed 0);
* $L_t$ — a random rotation times log-spaced scales spanning a factor
  `anisotropy` = 2, geometric-mean scale 1, so each part is an ellipsoid with a
  different orientation;
* $z_t$ — a unit-covariance sample whose marginal cycles through Gaussian,
  uniform and Laplace from part to part;
* $s$ = `spread` = 0.35 — the within-part scale, small enough that the parts are
  disjoint (a nearest-centre classifier recovers the part from its input with
  accuracy 0.9996);
* $\varepsilon$ — training-target noise with standard deviation
  $0.1\cdot\operatorname{std}(y)$, applied to the training set only. Test targets
  are the clean function, so a score measures retained function, not the noise
  draw.

The shared target is nonlinear in four of the six inputs plus a radial term:

$$
f(x) = \sin(1.5x_1)\cos(1.5x_2) + 0.6\tanh(x_3x_4) + 0.3\left(\frac{1}{6}\sum_i x_i^2 - 9\right)/4 .
$$

Because the supports are disjoint, each part constrains $f$ on its own patch and
leaves the rest of the input space free. The model must hold ten patches in one
set of weights — the regression analogue of permuted MNIST, with a *distribution*
shift instead of a pixel permutation.

Per-part target scale differs (standard deviation 0.35–0.60 across parts), which
is why scores are $R^2$ rather than MSE: each part is normalised by its own
variance.

### Why these choices

* **One shared target, ten supports.** Forgetting is then about *where* $f$ was
  learned, not about which of ten functions was learned, and a single-head model
  has no legitimate shortcut.
* **Disjoint supports.** Nothing the model sees on part $t$ constrains $f$ on
  part $t-1$'s region unless the method stores data. This is what makes the
  benchmark sensitive; §4.3 shows the effect disappearing once the supports
  overlap.
* **No task identifier.** Domain-incremental is the harder of the two scenarios
  (van de Ven & Tolias report 63.95 vs 98.64 for EWC on split MNIST), and it is
  the one where replay and regularisation separate.
* **10 parts × 512 samples.** One method on one seed takes 3–9 s, so paired
  five-seed comparisons are cheap and the seed spread can be reported rather
  than hidden.
* **Noiseless test targets.** The score reflects the learned function; the 10%
  training noise only stops the fit from being trivially exact.

### Data card

| field | shape | dtype | contents |
|---|---|---|---|
| `Task.index` | — | `int` | position in the stream, 0–9 |
| `Task.name` | — | `str` | `t00` … `t09` |
| `train_inputs` | (512, 6) | float32 | draws from the part's distribution |
| `train_targets` | (512, 1) | float32 | $f(x) + 0.1\operatorname{std}(f)\cdot\xi$, $\xi\sim N(0,1)$ |
| `test_inputs` | (256, 6) | float32 | independent draw, same distribution |
| `test_targets` | (256, 1) | float32 | $f(x)$, noiseless |

### Guarantees

* **Deterministic.** The same `seed` gives bit-identical tensors. Every draw
  uses its own generator derived from `(seed, part, purpose)`, and the global
  torch RNG is never touched, so building a stream cannot perturb a model's
  initialisation.
* **Disjoint at the defaults.** A nearest-centre classifier recovers the part
  from its input with accuracy ≥ 0.99 (0.9996 for seed 0): minimum distance
  between part means 3.80 against a within-part scale of 0.35.
* **Different marginals, not just different means.** The marginal cycles
  Gaussian → uniform → Laplace, and each part's covariance is a random rotation
  times log-spaced axes spanning a factor of 2 with geometric-mean scale 1.
* **Evaluation is noise-free.** Only training targets carry noise, so a score
  reflects the learned function rather than a particular noise draw.
* **Not included.** No task labels, no validation split, no real-world data.
  Test inputs come from the same distribution as their training part, so a score
  measures retention of that part, not generalisation within it.

### Knobs

| argument | default | controls |
|---|---|---|
| `n_tasks` | 10 | number of parts |
| `dim` | 6 | input variables |
| `n_train`, `n_test` | 512, 256 | samples per part |
| `center_radius` | 3.0 | distance of the part means from the origin / overlap |
| `spread` | 0.35 | within-part scale |
| `anisotropy` | 2.0 | ratio of the largest to the smallest covariance axis |
| `noise` | 0.1 | training-target noise, relative to the part's target scale |
| `rotated_targets` | False | give each part its own rotated copy of $f$ |
| `seed` | 0 | centres, covariances, samples, rotations, noise |

### Use

```python
from pt_neural_memory.forgetting_data import regression_stream

stream = regression_stream(seed=0)      # 10 tasks in training order
part = stream[0]
part.train_inputs   # (512, 6)
part.train_targets  # (512, 1), noisy
part.test_inputs    # (256, 6)
part.test_targets   # (256, 1), clean
```

Or load the published copies (`seed_0`–`seed_4`, one config per stream):

```python
from datasets import load_dataset

train = load_dataset("ccde/forgetting-regression", "seed_0", split="train")
part3 = train.filter(lambda row: row["task_id"] == 3)
```

## 2. Protocol

Single MLP on a shared scalar head, no task identifier: this is the
domain-incremental scenario of van de Ven & Tolias (2020) — the input
distribution changes, the task identity is not available at test time. Train
part 0, then part 1, ..., and after each part evaluate **every** part.

Scores are the accuracy matrix

$$
a_{tj} = 1 - \frac{\operatorname{MSE}(\hat y_j \text{ after stage } t)}{\operatorname{Var}(y_j)},
$$

so 0 is the per-part mean predictor and 1 is perfect. With $n$ parts and
random-init scores $b_j$, following Lopez-Paz & Ranzato (2017) with accuracy
replaced by $R^2$:

$$
\mathrm{ACC}=\frac{1}{n}\sum_j a_{n-1,j},\quad
\mathrm{BWT}=\frac{1}{n-1}\sum_{j<n-1}\left(a_{n-1,j}-a_{j,j}\right),\quad
\mathrm{LA}=\frac{1}{n}\sum_j a_{j,j},
$$

$$
\mathrm{FWT}=\frac{1}{n-1}\sum_{j>0}\left(a_{j-1,j}-b_j\right).
$$

BWT is negative when a part is worse at the end than when it was current; LA is
how well each part was learned while current. All MLP methods share one
architecture (2 × 64 ReLU), one optimiser (Adam, lr $10^{-3}$), one budget (300
steps per part, batch 128) and one initialisation per seed. The joint reference
is the exception: it is an upper bound and gets 800 steps per part at lr
$3\cdot10^{-3}$. The addressed memory is outside this contract — it has no
optimiser — and §3.2 describes its architecture.

### Reading the matrix

Rows are the stage (which part was trained last), columns are the part being
evaluated:

| region | meaning | what to look for |
|---|---|---|
| diagonal, $a_{jj}$ | how well the part was learned while current | high and flat across parts (LA) |
| below diagonal, $a_{tj}$ with $t>j$ | retention of a part after later training | the forgetting signal |
| above diagonal, $a_{tj}$ with $t<j$ | predictions on a part never trained on | usually very negative (extrapolation) |
| final row, $a_{n-1,\cdot}$ | what the finished model knows | the user-visible answer |

Scores are per-part $R^2$: 0 is the per-part mean predictor, 1 is perfect, and
negative is worse than the mean. Values reach $-20$ in the upper triangle; clamp
colours when plotting, never the data.

A method that retains leaves a flat lower triangle (replay in §4.2: every entry
≥ 0.75). A method that forgets leaves a bright diagonal and a cold lower
triangle (naive: diagonal 0.78–0.93, final row mostly negative).

### Comparing methods

* **Pair on seeds.** Every method sees the same stream and the same initial
  weights for a given seed, so compare per-seed deltas, not marginal means.
  With five seeds, a mean BWT difference below the seed standard deviation
  (about 0.8 here) is not a result. §4.1 reports both views.
* **Check LA before blaming retention.** A method can look forgetful only
  because it never fitted the current part. The joint reference has a larger
  budget, so its LA is not comparable with the others.
* **Treat FWT as noise here.** With disjoint supports it measures extrapolation
  to an unseen region, and its seed spread exceeds its mean.
* **Sweeps are for shape, not significance.** The ablations use three seeds and
  exist to show which axis moves the result, not to rank methods.

## 3. Methods

| name | mechanism | hyperparameter |
|---|---|---|
| `naive` | fine-tune on each part, no defence | — |
| `ewc` | diagonal-Fisher quadratic anchor per past part | strength λ = 100 |
| `si` | synaptic intelligence, online path-integral importance | strength c = 1 |
| `replay` | half of each batch drawn from a fixed store | 32 samples per part |
| `joint` | train on all parts seen so far (unbounded store) | — |
| `addressed` | random-feature key, one RLS cell per address, closed-form fit | 64 features, 3 × 2^10 cells, λ = 0.01 |

The two defences keep one anchor per past part and sum the penalties, the
standard multi-anchor form. `joint` is replay with an infinite store and is not
a continual-learning method; it is what the model could do if it were allowed to
re-read everything.

### The addressed memory as a regressor

A cell is an RLS linear map, so the memory only regresses nonlinearly once its
key is a nonlinear function of the input. The `addressed` method uses

$$
\varphi(x) = \left[\sqrt{\tfrac{2}{64}}\cos(Wx + b),\; 1,\; x\right],
$$

64 random Fourier features of a Gaussian kernel with length scale 2, a constant
for the intercept, and the raw input for a linear base, with $W$ and $b$ drawn
once from the seed and never trained. A store writes $(\varphi(x), y)$ into the
cell at the key's address and every one-bit neighbour, in each of three blocks
of $2^{10}$ cells; a read averages the addressed cell's map over the blocks.
All fitting is the cell's closed-form ridge update with $\lambda = 0.01$, so the
memory takes no gradient step.

Two properties make the comparison interesting. First, nothing is overwritten: a
later part writes new cells, or adds constraints to a cell, which is a joint
least-squares fit of all its writers rather than a move of a shared parameter.
Second, it is not a continual-learning defence — it keeps the data. The question
it answers is therefore the ceiling question ("what does a model that stores
everything and trains nothing score?"), not the usual "which defence is best".

Implementation: [src/pt_neural_memory/addressed_regressor.py](../src/pt_neural_memory/addressed_regressor.py)
holds the key map and the composition; `MemoryLearner` in the benchmark adapts
it to the protocol. A cell is `RLSMemCell`'s state inside `RLSCellBank`; see
[ADDRESSED_MEMORY.md](ADDRESSED_MEMORY.md).

### Adding a method

Subclass `SequentialLearner`, override the hooks you need, register the class in
`LEARNERS`. `benchmark()` passes any extra keyword argument straight to the
constructor.

A method that is not gradient-trained does not need `SequentialLearner`:
implement the two `Learner` methods, `train_task(task)` and
`predict(inputs)`, and register the class, as `MemoryLearner` does.

```python
import torch
from pt_neural_memory.benchmarks.forgetting_bench import (
    DEFAULT_PROTOCOL,
    LEARNERS,
    SequentialLearner,
)


class AnchoredL2(SequentialLearner):
    """Pull every weight toward its initial value: a negative control."""

    name = "anchored-l2"

    def __init__(self, dim, seed, protocol=DEFAULT_PROTOCOL, strength=1e-2):
        super().__init__(dim, seed, protocol)
        self.strength = strength
        self._initial = self._snapshot()

    def _penalty(self):
        total = torch.zeros(())
        for parameter, initial in zip(self.params, self._initial):
            total = total + (parameter - initial).square().sum()
        return 0.5 * self.strength * total


LEARNERS["anchored-l2"] = AnchoredL2
```

| hook | default | override it to |
|---|---|---|
| `_batch(task)` | draw `batch_size` samples from the current part | mix in stored data (`ReplayLearner`) or a growing pool (`JointLearner`) |
| `_penalty()` | zero | add a regulariser (`EWCLearner`, `SILearner`) |
| `_begin_task(task)` | nothing | reset per-task state, grow the pool |
| `_end_task(task)` | nothing | store samples, consolidate importance |
| `_after_step(before, grads)` | nothing | accumulate a path integral (set `tracks_path = True`) |

That control is worth running once: at `strength=1e-2` on seed 0 it scores
ACC −3.07 and BWT −4.39, worse than fine-tuning on both counts. Limiting drift
toward the initial function does not recover what later parts overwrite — the
same lesson as EWC and SI, from a one-hook example.

A new *ablation axis* goes through `sweep_experiment(title, x_label, values,
build_method_options, methods)`: it maps each value to per-method keyword
arguments (plus an optional `{"stream": {...}}` override for the generator) and
returns a `Sweep` of summary metrics. `separation_sweep()` is the pattern.

## 4. Results

### 4.1 Main comparison

Ten parts, five seeds, mean ± standard deviation over seeds:

| method | ACC | BWT | LA | FWT |
|---|---:|---:|---:|---:|
| naive | −1.032 ± 1.221 | −2.110 ± 1.308 | 0.867 ± 0.051 | −1.95 ± 3.19 |
| ewc | −0.630 ± 0.802 | −1.656 ± 0.900 | 0.860 ± 0.051 | −1.55 ± 2.64 |
| si | −0.498 ± 0.707 | −1.505 ± 0.798 | 0.857 ± 0.049 | −1.52 ± 2.72 |
| replay | **0.848 ± 0.033** | **−0.025 ± 0.030** | 0.870 ± 0.045 | −1.55 ± 3.82 |
| joint | **0.972 ± 0.009** | **+0.020 ± 0.003** | 0.954 ± 0.011 | −1.37 ± 2.41 |
| addressed | 0.954 ± 0.008 | −0.006 ± 0.005 | **0.960 ± 0.005** | −0.19 ± 0.36 |

Read the ACC and BWT columns together: an ACC below 0 means the final model is
*worse than predicting each part's mean*, and a BWT near −2 means the average
part lost about twice its own explained variance. EWC and SI improve the mean by
0.45 and 0.61 BWT units, but the per-seed paired deltas tell a weaker story:

| comparison | BWT delta per seed | mean | seeds won |
|---|---|---:|---:|
| ewc − naive | −0.60, −0.41, +0.28, +0.09, +2.91 | +0.45 | 3/5 |
| si − naive | −0.50, +0.06, +0.62, −0.09, +2.93 | +0.61 | 3/5 |
| replay − naive | +2.00, +2.19, +1.26, +0.75, +4.23 | **+2.09** | **5/5** |

On two seeds EWC is *worse* than doing nothing, and both defences owe their mean
gain to seed 4, where fine-tuning diverged furthest. Replay wins on every seed
by at least 0.75 BWT units. Among the five MLP methods that is the one
method-versus-method difference which is not seed noise.

The addressed memory is the second row that does not forget, and the only one
that does not do so by defending a parameter. Its BWT is −0.006 ± 0.005, i.e.
within 0.011 of zero on every seed, so against naive's per-seed −0.81 to −4.23
it wins every seed by at least 0.8 units. Its final ACC (0.954 ± 0.008, range
0.948–0.966) sits within 0.02 of joint training and 0.11 above replay, and its
learning accuracy (0.960) is the highest in the table: it fits each part better
than fine-tuning does, without a gradient step. Its FWT (−0.19) is also the
mildest, for a different reason than a better model — on an unseen part its
cells answer from a local map fitted nearby or from the prior, which is a much
smaller extrapolation than the MLP's (−1.4 to −1.9).

FWT is negative for every method and comes with a standard deviation larger than
its mean. It is not informative here, and the reason is the design: a part's
support is disjoint from the next part's, so "zero-shot transfer to the next
part" is pure extrapolation. The matrix shows how bad that can get — after
training on part 0 alone, the model scores $R^2 = -23.6$ on part 1.

### 4.2 What the matrix shows

Mean $R^2$ matrix of naive fine-tuning (row = stage, column = part):

| after / part | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t0 | +0.85 | −3.52 | −7.35 | −7.27 | −5.00 | −7.17 | −4.52 | −9.06 | −8.79 | −1.02 |
| t1 | −0.69 | +0.87 | −9.66 | −13.65 | −2.62 | −20.67 | −1.22 | −5.44 | −5.74 | −3.62 |
| t2 | −4.23 | −2.04 | +0.78 | −3.40 | −3.82 | −4.79 | −0.92 | −3.22 | −7.48 | −3.37 |
| t3 | −10.69 | −0.30 | +0.11 | +0.87 | −1.68 | −4.60 | −0.75 | −1.83 | −4.66 | −2.96 |
| t4 | −15.69 | −1.89 | +0.10 | −0.45 | +0.88 | −8.99 | −0.01 | −2.45 | −3.76 | −2.43 |
| t5 | −2.79 | −2.94 | −4.71 | −1.40 | −0.87 | +0.85 | −1.19 | −4.67 | −3.55 | −4.40 |
| t6 | −6.69 | −3.30 | −3.55 | −0.26 | +0.07 | −0.32 | +0.90 | −10.45 | −0.21 | −1.70 |
| t7 | −1.66 | −0.96 | −8.85 | −0.19 | −1.71 | −0.82 | −0.46 | +0.89 | −0.79 | −2.26 |
| t8 | −1.93 | −0.29 | −5.67 | −0.47 | −0.98 | −1.75 | −0.54 | +0.37 | +0.86 | −1.26 |
| t9 | −0.71 | −2.65 | −5.56 | −0.21 | −1.06 | −1.21 | +0.25 | −0.03 | −0.06 | +0.93 |

The diagonal is the only place with high scores. Part 0 falls from 0.85 to
−0.71 (11× worse in MSE terms: 0.15 → 1.71) by the end and passes through
−15.7 on the way. Off-diagonal entries in the upper triangle are negative
because the model has not seen those regions yet and extrapolates — this is the
negative FWT.

The same matrix for the addressed memory:

| after / part | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t0 | +0.97 | −2.43 | −1.00 | −2.99 | −1.22 | −3.42 | −3.59 | −3.64 | −3.08 | −1.91 |
| t1 | +0.97 | +0.98 | −3.63 | −3.21 | −1.67 | −6.51 | −2.64 | −3.98 | −3.35 | −2.63 |
| t2 | +0.97 | +0.98 | +0.93 | −3.92 | −2.19 | −5.29 | −2.61 | −3.98 | −3.41 | −2.74 |
| t3 | +0.97 | +0.98 | +0.92 | +0.95 | −1.86 | −5.36 | −2.45 | −2.37 | −2.76 | −2.52 |
| t4 | +0.97 | +0.98 | +0.92 | +0.95 | +0.98 | −5.34 | −2.48 | −2.76 | −2.37 | −1.72 |
| t5 | +0.97 | +0.98 | +0.93 | +0.95 | +0.98 | +0.94 | −2.20 | −3.09 | −2.95 | −1.50 |
| t6 | +0.97 | +0.98 | +0.93 | +0.95 | +0.98 | +0.94 | +0.97 | −2.92 | −2.61 | −1.46 |
| t7 | +0.96 | +0.98 | +0.93 | +0.95 | +0.98 | +0.94 | +0.97 | +0.97 | −1.59 | −1.46 |
| t8 | +0.96 | +0.98 | +0.93 | +0.94 | +0.98 | +0.93 | +0.97 | +0.97 | +0.95 | −0.99 |
| t9 | +0.96 | +0.97 | +0.92 | +0.94 | +0.98 | +0.93 | +0.97 | +0.97 | +0.94 | +0.96 |

Every entry below the diagonal lies between +0.92 and +0.98, and the diagonal
and the final row differ by at most 0.02 on every part: there is no darkening to
look for. The upper triangle is still negative, but at −1 to −6 rather than −20.
An unseen part's query lands either on a cell fitted to a nearby region — a
small extrapolation — or on a cell fitted to another part — a wrong one — and
neither produces the MLP's global blow-up.

Final row, all methods (the user-visible question: how does the finished model
do on each part?):

| method | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | −0.71 | −2.65 | −5.56 | −0.21 | −1.06 | −1.21 | +0.25 | −0.03 | −0.06 | +0.93 |
| ewc | −0.34 | −4.01 | −1.67 | −1.28 | +0.40 | −0.47 | +0.31 | +0.03 | −0.20 | +0.91 |
| si | −1.66 | −2.97 | −0.77 | +0.16 | +0.40 | −0.68 | +0.46 | −0.24 | −0.56 | +0.90 |
| replay | +0.75 | +0.84 | +0.81 | +0.85 | +0.87 | +0.84 | +0.88 | +0.87 | +0.84 | +0.93 |
| joint | +0.98 | +0.99 | +0.96 | +0.97 | +0.99 | +0.96 | +0.98 | +0.98 | +0.94 | +0.98 |
| addressed | +0.96 | +0.97 | +0.92 | +0.94 | +0.98 | +0.93 | +0.97 | +0.97 | +0.94 | +0.96 |

Replay and the addressed memory are the only rows where every part is positive.
Replay's gap to joint (0.848 vs 0.972) is mostly optimisation budget rather than
a failure of retention: replay's learning accuracy (0.870) sits with
fine-tuning's (0.867), both below joint's (0.954), because joint gets 800 steps
per part at a higher learning rate. The memory's row is flatter still — its
worst part is 0.92 and its best 0.98 — and it reaches that with no optimisation
at all.

### 4.3 Ablations

**Part separation** (3 seeds). Shrinking the radius the means sit on moves the
parts toward a common region. Forgetting collapses once the supports overlap
enough to constrain each other:

| centre radius | naive ACC | naive BWT | ewc BWT | replay BWT |
|---:|---:|---:|---:|---:|
| 3.0 | −0.77 | −1.84 | −2.08 | −0.03 |
| 1.6 | +0.86 | −0.11 | −0.17 | −0.00 |
| 0.9 | +0.96 | −0.01 | −0.03 | +0.01 |
| 0.6 | +0.98 | +0.01 | +0.00 | +0.02 |

At radius 1.6 the nearest-centre separability is 0.96 rather than 1.00: the parts
still have different means, but each part's samples are close enough to the
others that training on one transfers. The transition is sharp between radius 3.0
and 1.6, so this axis is closer to a switch than to the smooth, non-monotonic
similarity dependence Ramasesh et al. (2020) report for classification. Four
points are too few to settle whether forgetting peaks at intermediate similarity.

**Replay store size** (3 seeds). Forgetting decreases smoothly with the fraction
of each part retained:

| samples per part | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| ACC | +0.57 | +0.73 | +0.81 | +0.87 | +0.89 | +0.90 |
| BWT | −0.36 | −0.18 | −0.08 | −0.03 | −0.01 | +0.01 |

32 per part is 6.3% of a part and 320 points in total; 4 per part already
recovers most of part 0's score, which says the interference is not a capacity
problem for this model.

**Regularisation strength** (3 seeds). Neither defence has a stable optimum:

| strength | 0.1 | 1 | 10 | 100 | 1 000 | 10 000 |
|---|---:|---:|---:|---:|---:|---:|
| ewc BWT | −1.81 | −1.83 | −1.59 | −2.08 | −1.79 | −2.98 |
| si BWT | −1.65 | −1.78 | −1.54 | −3.45 | −0.80 | −1.75 |

The sweep is non-monotone for both, and the same λ = 100 point measures −2.08
here (seeds 0–2) and −1.66 in the main table (seeds 0–4). With a seed standard
deviation around 0.8 BWT units, only the extremes of this table are
distinguishable — which is itself the finding: the method is dominated by seed
noise, not by its hyperparameter.

**Target shift** (3 seeds). Replacing the shared target with a per-part rotated
copy makes fine-tuning worse and leaves replay intact:

| target | naive ACC | naive BWT | replay ACC | replay BWT |
|---|---:|---:|---:|---:|
| shared | −0.77 | −1.84 | +0.87 | −0.03 |
| rotated | −1.76 | −2.90 | +0.77 | −0.07 |

**Part order** (3 seeds). Fine-tuning is order-sensitive; replay is not:

| order | naive ACC | naive BWT | replay ACC | replay BWT |
|---|---:|---:|---:|---:|
| forward | −0.77 | −1.84 | +0.87 | −0.03 |
| reverse | −2.03 | −3.22 | +0.86 | −0.03 |
| shuffled | −1.77 | −2.94 | +0.82 | −0.05 |

### 4.4 The addressed memory

The memory is not a continual-learning defence and should not be read as one. It
answers a different question: if the model keeps every pair, how much is left
for a defence to recover? The answer is almost all of it, and the price is paid
in state and data rather than in a penalty term.

**Cost.** The final model is a different size class from the MLP:

| | MLP (naive) | addressed |
|---|---:|---:|
| learned parameters / resident state | 4,673 (18.3 KiB) | 60 MiB |
| stored pairs | 0 | 5,120 |
| per-part fitting | 300 Adam steps | closed form, 33 cell updates per pair |
| probes per query | one forward pass | 3 cells |
| fit seconds (seed 0, 4 threads) | 3.4 | 3.3 |

The memory is about 3,400× the MLP's state, and about 440× the 140 KiB the raw
pairs would occupy. The bill is the dense covariance: $71^2$ numbers for each of
3,072 cells, the same fact as [ADDRESSED_MEMORY.md](ADDRESSED_MEMORY.md) §1.5.
The state and probe counts follow from the configuration; the fit seconds are
medians of three fresh fits, and both they and the microsecond figures are
wall-clock on this checkout (four threads) and will vary.

**From-scratch reference.** Pooling all 5,120 pairs and training the same MLP
from scratch (Adam, lr $10^{-3}$, batch 128) gives ACC 0.601 after 300 steps
(0.35 s), 0.859 after 1,000 (1.15 s), 0.949 after 3,000 (3.5 s), 0.975 after
8,000 (9.3 s), and 0.984 after 16,000 (18.6 s). The addressed memory reaches
0.948 in 3.3 s: at equal time the two are level, while a longer offline fit is
more accurate and about 3,400× smaller. The offline MLP is not a continual
learner — it needs all the data and a fresh optimizer run — so this prices the
memory's incremental, gradient-free mode; it is not a claim that the memory
beats offline training on accuracy. The `joint` row is the ceiling of the
continual protocol, not the offline ceiling.

**Size.** One axis moves capacity, locality and interference together:

| n_bits | cells/block | writers per cell | state MiB | ACC | BWT | LA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 5,120 | 0.1 | 0.538 | −0.217 | 0.734 |
| 4 | 16 | 1,600 | 0.9 | 0.815 | −0.085 | 0.891 |
| 6 | 64 | 560 | 3.8 | 0.904 | −0.032 | 0.933 |
| 8 | 256 | 180 | 15 | 0.940 | −0.012 | 0.951 |
| 10 | 1,024 | 55 | 60 | 0.952 | −0.006 | 0.957 |
| 12 | 4,096 | 16 | 240 | 0.958 | −0.001 | 0.958 |

With one bit per block the memory is a near-global ridge map in six cells: ACC
0.538 and BWT −0.217, because every part is fitted in the same two cells per
block. Each added bit buys accuracy and isolation together, and the curve bends
at the default: going from 10 to 12 bits adds 0.006 ACC for 4× the state.
Writers per cell fall from 5,120 to 16, and the default's 55 is below the 71
associations an RLS cell of this key dimension can interpolate. This is the same
occupancy law as the memory's own report, now with retention on the vertical
axis: a cell that holds every part's data also contaminates every part's
predictions.

**Store size.** The memory needs more points than replay for the same score:

| samples per part | 4 | 8 | 16 | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| replay ACC | +0.57 | +0.73 | +0.81 | +0.87 | +0.89 | +0.90 | — |
| memory ACC | −0.26 | +0.21 | +0.60 | +0.76 | +0.86 | +0.90 | +0.95 |
| memory BWT | +0.00 | +0.01 | +0.00 | −0.01 | −0.01 | −0.00 | −0.01 |

Replay reaches 0.867 with 32 points per part; the memory needs 64 to reach
0.857, and 128 to reach 0.896, which is replay's score at 128. The reason is
structural: a cell fits 71-dimensional keys independently, so the data is not
shared across cells the way gradient training shares it across the MLP.
Retention is never the problem — BWT stays within ±0.01 at *every* store size,
and is slightly *positive* below 32 because more writers improve the earlier
parts' cells. The conclusion is the replay sweep's, with the efficiency margin
measured: storing data buys retention for free; converting it into accuracy is
the modelling problem. (The sweep holds the table at $2^{10}$ cells; a 320-pair
store has 3.4 writers per cell there, so sizing the table to the store — $m=6$
at 320 pairs, by the capacity law — should recover some of the gap. The two axes
were not swept jointly.)

**Design.** One change at a time, with the rest at the default:

| variant | ACC | BWT | LA |
|---|---:|---:|---:|
| default | 0.952 | −0.006 | 0.957 |
| width 32 | 0.927 | −0.008 | 0.934 |
| no raw input | 0.893 | −0.030 | 0.921 |
| λ = 0.1 | 0.925 | −0.005 | 0.929 |
| λ = 0.001 | 0.959 | −0.008 | 0.966 |
| verified read | 0.924 | −0.018 | 0.940 |
| write radius 0 | 0.932 | 0.000 | 0.932 |

Feature width matters (32 → 0.927, 64 → 0.952) and the raw input matters
(0.893 without it): a local map needs a linear base as well as Fourier features.
The ridge is mild — λ = 0.001 is +0.007, inside seed noise and a worse-
conditioned choice — so the default 0.01 stands. The verified read is *worse*
than plain block averaging (0.924 against 0.952), the opposite of the memory's
noise benchmark: here the query is a fresh point rather than a corrupted stored
key, and the variance score prefers crowded cells, which with structured keys
are the cells shared across parts (BWT falls to −0.018). Write radius 0 loses
0.02, which is the neighbours' job — giving a fresh query a cell trained near
it.

## 5. Why the parameter-space defences fail here

After a part is fitted, its loss is small and its gradients are small. The
diagonal Fisher of the 4,673-parameter MLP has mean $7.7\cdot10^{-6}$ and maximum
$1.5\cdot10^{-3}$. A quadratic penalty therefore needs $\lambda \approx 10^4$
before it competes with the current part's loss — the same "easy tasks give tiny
Fisher, so EWC needs a much larger λ than usual" effect that van de Ven & Tolias
report for split MNIST.

Two checks on seed 0 show what happens at that strength:

* Two parts, $\lambda=10^4$: the anchor does its job — part 0 stays at
  $R^2=0.58$ where fine-tuning leaves it at −0.87.
* Ten parts, $\lambda=10^4$: BWT degrades to −5.80, worse than fine-tuning's
  −2.03 for that seed. At $\lambda=10^6$ the model freezes (LA 0.52) and old
  parts still end up negative.

The plausible reading is that the penalty pins parameters to a weighted average
of ten anchors while the current part pulls away, and the compromise is far from
every anchor in exactly the directions the Fisher marks as important. Outside a
part's support nothing constrains $f$, so a parameter move that looks small in
weight space can change predictions by a full target standard deviation across a
disjoint region. Replay does not have this problem because it supplies the
missing constraint directly: a few dozen points per part are enough to pin $f$
down where it was learned.

This is a claim about *this* benchmark, not a general verdict on EWC. On
classification benchmarks with shared input support, the old task's data manifold
is revisited by the new task's inputs, and parameter regularisation has
something to work with.

The addressed memory makes the same point from the opposite direction. It has
no parameter budget to protect: its state is the pairs themselves, and a later
part adds cells or adds constraints to cells that already hold writers. Its BWT
is near zero because those old constraints remain in each cell's joint fit, not
because a shared weight is protected, and its ACC lands within 0.02 of joint
training because every pair is still available at read time. That is replay's
argument taken to its limit — the missing constraint is supplied directly — and
it prices the alternative: to reach 0.95 without gradient training you spend
60 MiB and keep every pair.

## 6. What was already out there

A short survey, because the design choices above came from it. The honest
headline: **I could not find a standard synthetic regression benchmark with a
different input distribution per task**; continual-learning benchmarks are
overwhelmingly classification.

* **Metrics and protocol.** The accuracy matrix, ACC, BWT and FWT are from
  Lopez-Paz & Ranzato, *Gradient Episodic Memory* ([arXiv:1706.08840](https://arxiv.org/abs/1706.08840)).
  Their protocol assumes a task descriptor is supplied at test time; the present
  benchmark drops it and is therefore the harder domain-incremental case.
  Van de Ven & Tolias, *Three scenarios for continual learning*
  ([arXiv:1904.07734](https://arxiv.org/abs/1904.07734)) define that taxonomy and
  show the scenario dominates the method: EWC scores 98.64 on split MNIST
  task-incrementally and 63.95 domain-incrementally. Any regression number
  should therefore state its scenario, as §2 does.
* **The closest published stream.** Von Oswald et al., *Continual learning with
  hypernetworks* ([arXiv:1906.00695](https://arxiv.org/abs/1906.00695)), use a
  1-D regression toy in which each task is a polynomial of increasing degree on
  its own input domain $X^{(t)}$, 100 noisy samples per task, $\sigma=0.05$.
  That is the same idea as this stream — same target family, different input
  domain per task — generalised to six inputs, ten parts and a shared nonlinear
  target.
* **Regression benchmarks exist, but recently and in engineering.** Samuel &
  Ahmed, *Continual Learning Strategies for 3D Engineering Regression Problems:
  A Benchmarking Study* ([arXiv:2504.12503](https://arxiv.org/abs/2504.12503)),
  build nine regression benchmarks over five engineering datasets, with
  experience splits by target bin or by input domain. They report replay
  matching full retraining at about half the training cost — consistent with §4
  here — and note that regression has no benchmark suite comparable to the
  classification ones.
* **Libraries.** Avalanche's packaged benchmarks are classification-only, but it
  ships regression *metrics*: `TaskAwareRMSE`, `TaskAwareR2`, and
  `ExperienceRMSEForgetting`, defined as the first-minus-final RMSE so that
  "forgetting is observed when the result is negative"
  ([API docs](https://avalanche-api.continualai.org/en/latest/_modules/avalanche/evaluation/metrics/regression_forgetting.html)).
  That is the same sign convention as BWT here. There is no ready-made
  multi-distribution regression suite to import.
* **Baselines.** EWC (Kirkpatrick et al.,
  [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)), synaptic intelligence
  (Zenke et al., [arXiv:1703.04200](https://arxiv.org/abs/1703.04200)), experience
  replay (Rolnick et al., [arXiv:1811.11682](https://arxiv.org/abs/1811.11682))
  and joint training as the upper bound. Li & Hoiem's LwF
  ([arXiv:1606.09282](https://arxiv.org/abs/1606.09282)) is the obvious omission
  here: it distils on the *current* task's inputs, and with disjoint supports
  those inputs say nothing about the old parts, so it is expected to fail for the
  same reason EWC does. Masks/importance from output sensitivity (MAS,
  [arXiv:1711.09601](https://arxiv.org/abs/1711.09601)) would be a cheap fourth
  regulariser.
* **The regressor's ingredients.** Random Fourier features (Rahimi & Recht,
  *Random features for large-scale kernel machines*, NIPS 2007) turn a linear
  readout into a kernel regressor; addressing makes that readout local. The CL
  relatives are memory-based and replay methods, which also keep data, not the
  parameter-space defences.
* **What to watch for.** Ramasesh et al.
  ([arXiv:2007.07400](https://arxiv.org/abs/2007.07400)) find maximal forgetting
  at *intermediate* task similarity, which argues against testing only the fully
  disjoint setting; §4.3's separation sweep is a first attempt at that axis.
  And the `joint` reference must be given enough budget to converge, or it
  understates the ceiling.

## 7. Limitations

* **Five seeds, large variance.** Naive BWT ranges from −0.81 to −4.23 across
  seeds, so the main table's means are weak summaries. The paired table in §4.1
  is the load-bearing one. Replay's variance is tiny by comparison.
* **One architecture, one budget.** 2 × 64 ReLU, 300 steps, lr $10^{-3}$, no
  per-method tuning beyond the strength sweep. Joint gets a larger budget, so
  the replay-to-joint gap confounds retention with optimisation.
* **Fixed order in the main run.** The order ablation covers three orders on
  three seeds; it shows fine-tuning moves by more than a full BWT unit between
  orders.
* **Domain-incremental only.** No task-ID (task-incremental) control. For a
  single scalar output the task-ID variant would need a per-task head, which is
  a different model and a different question.
* **Synthetic target, one family.** The rotated-target variant is a crude
  function shift. Real tabular streams — bike sharing by season, California
  housing by region — would test whether the conclusion survives real
  distributions, but they bring their own confounds.
* **The separation sweep is coarse.** Four radii, three seeds; the transition is
  sharp enough that the interesting region is between 1.6 and 3.0, where it has
  no points.
* **No LwF, MAS, or other regulariser beyond EWC and SI**, and no per-method
  learning-rate tuning.
* **The memory is not data-efficient.** At the default 2^10-cell table the
  memory needs the whole part; a small store leaves its cells nearly empty and
  its accuracy collapses (the store-size table in §4.4). Its 0.954 ACC costs
  60 MiB and 5,120 stored pairs, against replay's 0.848 for 320 pairs. It is a
  ceiling, not a replacement for replay.
* **Random, untrained features.** The key map is drawn from the seed and never
  trained, so the memory's seed spread includes the feature draw; a trained
  feature map would be a different (gradient) model. The result is also one
  feature family and one table configuration — the two were not swept jointly.
* **Hyperparameters were chosen on this stream.** The memory's width, length
  scale, address width and ridge were selected in a pilot on seed 0 and
  confirmed on seeds 0–4, while the MLP baselines kept the fixed values of v1.
  The memory's numbers are therefore slightly optimistic relative to a fully
  held-out protocol.
* **Verified reads are not used.** The memory's own default verified read scores
  below plain block averaging here (§4.4); the variance score prefers crowded
  cells, and with structured keys a crowded cell is often one shared across
  parts.

## 8. Recommendation

1. Use this stream as the regression arm of the forgetting tests: it is small
   (10 × 512 points), fast (3–9 s per method per seed), deterministic, and the
   joint ceiling is known (0.972).
2. Report the full $R^2$ matrix, not just ACC/BWT. The matrix is where the
   failure is visible; the summary statistics are noisy.
3. Keep replay in the comparison as the reference defence. Any method that does
   not beat 32 samples per part is not solving the problem the benchmark poses.
4. If a parameter-space method is the object of study, add the separation sweep
   as a second axis and expect the effect to vanish at radius ≈1.6.
5. Add a task-ID control before drawing conclusions about classification
   results: the scenario gap is larger than most method gaps.
6. Report the addressed memory as the ceiling check, with its state and stored
   pairs. It shows what "keep the data" scores without bound; a defence that
   cannot approach that ACC within a stated budget is losing to data, not to
   architecture.

## 9. Tests

`tests/test_forgetting_data.py` pins the data contract: shapes and dtypes,
determinism under a seed, support disjointness above 0.99 nearest-centre,
noiseless test targets with correctly scaled training noise, distinct per-part
covariances, the rotated-target mode, the single-part edge case, and every
argument-validation error.

`tests/test_forgetting_bench.py` pins the protocol: `summarize()` against a
hand-computed matrix, `aggregate()`'s cell means and standard deviation,
identical initialisation across methods, EWC's penalty being zero at its anchor
and positive after another part, non-negative SI importance, a replay store
holding past parts only, a complete and finite matrix, and replay forgetting
less than fine-tuning on a three-part stream. It also covers the memory arm:
registration, one stored pair per training row, the `samples_per_task` subset,
deterministic reads, better BWT than fine-tuning, and a complete finite matrix.

`tests/test_addressed_regressor.py` pins the key map (shape, constant,
determinism, argument errors), exact recall of a linear function, a nonlinear
fit from stored pairs, equivalence of one batch and a sequence of stores, the
verified read, and the state arithmetic.

Files: [tests/test_forgetting_data.py](../tests/test_forgetting_data.py),
[tests/test_forgetting_bench.py](../tests/test_forgetting_bench.py),
[tests/test_addressed_regressor.py](../tests/test_addressed_regressor.py). The
forgetting tests take about 10 s; the full suite is 119 tests and passes on this
checkout.

## Reproduce

```bash
python -m pytest -q tests/test_forgetting_data.py tests/test_forgetting_bench.py tests/test_addressed_regressor.py
python -m pt_neural_memory.benchmarks.forgetting_bench            # all tables above + results/forgetting_bench.json
python -m pt_neural_memory.benchmarks.training_cost             # fit-time table + results/training_cost.json
python -m pt_neural_memory.benchmarks.export_forgetting --upload --token-file <token-file>   # rebuild and publish the Hub dataset
```

The Hub dataset is CSV (five configs × train/test). The commit endpoint accepts
text inline but rejects binaries, so Parquet would need LFS or Xet; the Hub
converts the CSV to Parquet for its viewer, and the conversion is byte-identical
to these files.

The benchmark takes about half an hour on four CPU threads (five seeds for the
main table, three for the sweeps) plus a few minutes for the three memory
sweeps. `results/forgetting_bench.json` is laid out as:

| key | contents |
|---|---|
| `protocol`, `joint_protocol` | the `Protocol` fields actually used |
| `addressed` | the memory's configuration, resident state, writes per pair and stored pairs |
| `main[method]` | `matrix` (10 × 10), `baseline`, `summary` (ACC/BWT/LA/FWT), `spread` (seed std) |
| `sweeps[name]` | `title`, `x_label`, `x`, and `series[method][metric]` |

`results/` is git-ignored, so the JSON is an artifact to regenerate rather than
a tracked file — the same convention as `memory_bench.json`.
