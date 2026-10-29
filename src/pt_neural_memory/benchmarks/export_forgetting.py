"""Export the ten-part forgetting stream as a Hugging Face dataset.

Run from the repository root::

    python -m pt_neural_memory.benchmarks.export_forgetting                        # build results/hf_dataset/
    python -m pt_neural_memory.benchmarks.export_forgetting --upload --token-file .scratch/creds/huggingface-key.txt

The dataset is five configs (``seed_0`` … ``seed_4``), each with a ``train``
and a ``test`` split.  Rows keep the generator's order, so filtering
``task_id == k`` reproduces part ``k`` exactly as the benchmark trains on it.

Files are CSV: the Hub's commit endpoint accepts text inline but rejects binary
files (Parquet needs LFS or Xet), and the dataset is small enough that the
extra bytes do not matter.  The Hub converts CSV to Parquet for its viewer.
"""

import argparse
import base64
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import torch

from pt_neural_memory.forgetting_data import Task, regression_stream, target_function

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
REPO_ID = "ccde/forgetting-regression"
HUB = "https://huggingface.co"
OUTPUT = Path("results/hf_dataset")


def to_frame(task: Task, split: str) -> pd.DataFrame:
    """One task as a flat table; ``y_clean`` is the noiseless target."""
    inputs = task.train_inputs if split == "train" else task.test_inputs
    targets = task.train_targets if split == "train" else task.test_targets
    columns = {f"x{i}": inputs[:, i].numpy() for i in range(inputs.shape[1])}
    frame = pd.DataFrame(columns)
    frame.insert(0, "task_id", task.index)
    frame["task_id"] = frame["task_id"].astype("int8")
    frame["y"] = targets[:, 0].numpy()
    frame["y_clean"] = (
        target_function(inputs)[:, 0] if split == "train" else targets[:, 0]
    ).numpy()
    return frame


def build(seeds: tuple[int, ...] = SEEDS, output: Path = OUTPUT) -> None:
    """Write ``data/seed_<n>/{train,test}.csv`` and the card under ``output``."""
    if output.exists():
        shutil.rmtree(output)  # generated staging directory
    for seed in seeds:
        parts = regression_stream(seed=seed)
        directory = output / "data" / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            frame = pd.concat(
                [to_frame(task, split) for task in parts], ignore_index=True
            )
            path = directory / f"{split}.csv"
            frame.to_csv(path, index=False, float_format="%.9g")
            print(f"seed {seed} {split}: {len(frame)} rows -> {path}")
    (output / "README.md").write_text(dataset_card(seeds), encoding="utf-8")
    print(f"wrote {output / 'README.md'}")


def dataset_card(seeds: tuple[int, ...]) -> str:
    """The dataset card, including the ``configs`` block the Hub needs."""
    configs = []
    for seed in seeds:
        configs.append(
            f"- config_name: seed_{seed}\n"
            "  data_files:\n"
            f"  - split: train\n    path: data/seed_{seed}/train.csv\n"
            f"  - split: test\n    path: data/seed_{seed}/test.csv"
        )
    return f"""---
license: cc-by-4.0
task_categories:
- tabular-regression
tags:
- continual-learning
- catastrophic-forgetting
- synthetic
- tabular
- regression
pretty_name: Ten-part regression forgetting benchmark
size_categories:
- 10K<n<100K
configs:
{chr(10).join(configs)}
---

# Ten-part regression forgetting benchmark

A synthetic multi-variable regression stream for measuring **catastrophic
forgetting**. One shared nonlinear target `f: R^6 -> R`; ten input
distributions with disjoint support; one scalar output. A model is trained on
part 0, then part 1, …, evaluated on every part after every stage. Task identity
is never given, so this is a *domain-incremental* regression benchmark.

It is the data behind the forgetting experiments in
`neural_memory/FORGETTING.md`: a plain MLP fits each part to R² ≈ 0.85–0.93 and
then loses it all (mean backward transfer −2.1), replay of 32 points per part
removes the forgetting (−0.03), and EWC/SI barely help.

## Structure

| config | splits | rows |
|---|---|---|
| `seed_0` … `seed_{len(seeds) - 1}` | `train` (5,120), `test` (2,560) | 7,680 per config |

Each config is an independent draw of the whole stream (same generator, different
seed). Rows are grouped by `task_id` in training order and keep the generator's
row order inside a part. Files are CSV; the Hub converts them to Parquet for its
viewer, and numeric columns load as `float64` holding the original float32
values.

| column | type | meaning |
|---|---|---|
| `task_id` | int | part index, 0–9 (training order) |
| `x0` … `x5` | float | input variables |
| `y` | float | target; noisy on `train`, clean on `test` |
| `y_clean` | float | the noiseless target `f(x)` |

Training targets carry noise with standard deviation `0.1 · std(f)` on that
part; test targets are noiseless, so a score measures the learned function
rather than a noise draw.

## Loading

```python
from datasets import load_dataset

train = load_dataset("{REPO_ID}", "seed_0", split="train")
part3 = train.filter(lambda row: row["task_id"] == 3)

# or the whole stream, part by part, in benchmark order
stream = [train.filter(lambda row, k=k: row["task_id"] == k) for k in range(10)]
```

```python
import pandas as pd

frame = pd.read_csv("hf://datasets/{REPO_ID}/data/seed_0/train.csv")
```

## Generation

Part `t` draws `x = c_t + s · z_t · L_t^T` and `y = f(x) + ε`, with

- `c_t`: part mean on the sphere of radius 3 in R^6, placed by greedy
  farthest-point sampling (minimum pairwise distance 3.80 for seed 0);
- `L_t`: random rotation × log-spaced axes spanning a factor of 2 (geometric
  mean 1), so each part is a differently oriented ellipsoid;
- `z_t`: unit-covariance sample whose marginal cycles Gaussian → uniform →
  Laplace from part to part;
- `s = 0.35`: within-part scale; parts are disjoint (a nearest-centre classifier
  recovers the part with accuracy ≥ 0.99);
- `ε`: training-only noise, `0.1 · std(f)`.

The shared target is

```
f(x) = sin(1.5 x1) cos(1.5 x2) + 0.6 tanh(x3 x4) + 0.3 ((mean_i x_i^2) - 9)/4
```

## Evaluation protocol

Score each part by `R² = 1 − MSE / Var(y_clean)`, giving a 10 × 10 matrix
`a[t][j]`: R² on part `j` after training part `t`. Report `ACC` (mean of the
final row), `BWT` (mean over `j < 9` of `a[9][j] − a[j][j]`; negative means
forgetting), `LA` (mean diagonal) and `FWT`. Per-part normalisation matters:
part target standard deviations range from 0.35 to 0.60.

## Limitations

- Synthetic and deliberately disjoint: it stresses retention, not
  generalisation within a part.
- One target family, six inputs, ten parts. Vary `f` for a harder stream; the
  generator supports per-part rotations of `f`.
- No task identifiers, no validation split, no real-world data.
- The five configs are independent streams, not five views of one stream.

## Provenance

Generated by `src/pt_neural_memory/forgetting_data.py` (seed = config suffix) from the
`pt-neural-memory` repository; `src/pt_neural_memory/benchmarks/export_forgetting.py` rebuilds these
files. Version 1.0.
"""


def read_token(token_file: str | None) -> str:
    """Token from ``HF_TOKEN`` or from a file holding ``hf_…``/``HF_TOKEN=…``."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    if token_file:
        match = re.search(
            r"hf_[A-Za-z0-9]+", Path(token_file).read_text(encoding="utf-8")
        )
        if match:
            return match.group(0)
    raise SystemExit("no token: set HF_TOKEN or pass --token-file")


def _post(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    """POST with an explicit method/body; HTTP errors are returned, not raised."""
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _commit(
    repo_id: str, token: str, files: list[Path], root: Path, message: str
) -> None:
    """Commit text files through the Hub's NDJSON commit endpoint."""
    lines = [
        json.dumps(
            {"key": "header", "value": {"summary": message, "description": ""}}
        )
    ]
    for path in files:
        lines.append(
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "path": path.relative_to(root).as_posix(),
                        "encoding": "base64",
                        "content": base64.b64encode(path.read_bytes()).decode(),
                    },
                }
            )
        )
    status, body = _post(
        f"{HUB}/api/datasets/{repo_id}/commit/main",
        ("\n".join(lines) + "\n").encode(),
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-ndjson",
        },
    )
    if status not in (200, 201):
        raise RuntimeError(f"commit {message!r} failed: {status} {body[:600]!r}")
    print(f"committed {len(files)} files: {message}")


def upload(
    output: Path = OUTPUT,
    repo_id: str = REPO_ID,
    token_file: str | None = None,
) -> None:
    """Create the dataset repo if needed, then commit the card and each config."""
    token = read_token(token_file)
    organization, name = repo_id.split("/")
    status, body = _post(
        f"{HUB}/api/repos/create",
        json.dumps(
            {
                "name": name,
                "organization": organization,
                "type": "dataset",
                "private": False,
            }
        ).encode(),
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    if status not in (200, 201, 409):
        raise RuntimeError(f"create repo failed: {status} {body[:300]!r}")

    _commit(repo_id, token, [output / "README.md"], output, "Add dataset card")
    for directory in sorted((output / "data").iterdir()):
        _commit(
            repo_id,
            token,
            sorted(directory.glob("*.csv")),
            output,
            f"Add {directory.name}",
        )
    print(f"uploaded -> {HUB}/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload", action="store_true", help="push the built files to the Hub"
    )
    parser.add_argument("--repo-id", default=REPO_ID, help="target dataset repository")
    parser.add_argument("--token-file", help="file holding the HF token")
    args = parser.parse_args()
    build()
    if args.upload:
        upload(repo_id=args.repo_id, token_file=args.token_file)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
