import numpy as np
import pandas as pd
import pytest
import torch

from pt_neural_memory.benchmarks.export_forgetting import to_frame
from pt_neural_memory.forgetting_data import regression_stream

COLUMNS = ["task_id"] + [f"x{i}" for i in range(6)] + ["y", "y_clean"]


@pytest.fixture(scope="module")
def task():
    return regression_stream(n_tasks=10, seed=0)[3]


def test_frame_has_the_published_schema(task):
    frame = to_frame(task, "train")
    assert list(frame.columns) == COLUMNS
    assert (frame["task_id"] == 3).all()
    assert frame["task_id"].dtype == "int8"


def test_train_targets_are_noisy_and_test_targets_are_clean(task):
    train = to_frame(task, "train")
    test = to_frame(task, "test")
    assert not torch.allclose(
        torch.from_numpy(train.y.to_numpy()), torch.from_numpy(train.y_clean.to_numpy())
    )
    assert torch.allclose(
        torch.from_numpy(test.y.to_numpy()), torch.from_numpy(test.y_clean.to_numpy())
    )


def test_frame_matches_the_stream_tensors(task):
    frame = to_frame(task, "train")
    assert np.array_equal(
        frame[[f"x{i}" for i in range(6)]].to_numpy(np.float32),
        task.train_inputs.numpy(),
    )
    assert np.array_equal(frame.y.to_numpy(np.float32), task.train_targets.numpy()[:, 0])


def test_csv_round_trip_is_exact(task, tmp_path):
    frame = to_frame(task, "train")
    path = tmp_path / "part.csv"
    frame.to_csv(path, index=False, float_format="%.9g")
    back = pd.read_csv(path)
    assert np.array_equal(
        back[[f"x{i}" for i in range(6)]].to_numpy(np.float32),
        task.train_inputs.numpy(),
    )
    assert np.array_equal(back.y.to_numpy(np.float32), task.train_targets.numpy()[:, 0])
    assert (back.task_id == 3).all()
