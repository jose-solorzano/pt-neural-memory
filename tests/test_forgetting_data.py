import pytest
import torch

from pt_neural_memory.forgetting_data import regression_stream, target_function


def _stream(seed: int = 0, **options):
    return regression_stream(n_tasks=10, seed=seed, **options)


def test_stream_shapes_and_types():
    tasks = _stream()
    assert len(tasks) == 10
    for task in tasks:
        assert task.train_inputs.shape == (512, 6)
        assert task.train_targets.shape == (512, 1)
        assert task.test_inputs.shape == (256, 6)
        assert task.test_targets.shape == (256, 1)
        for tensor in (task.train_inputs, task.train_targets, task.test_targets):
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()


def test_parts_have_disjoint_support():
    """A nearest-centre classifier should recover the part from the input."""
    tasks = _stream()
    centres = torch.stack([task.train_inputs.mean(dim=0) for task in tasks])
    for split in ("train_inputs", "test_inputs"):
        inputs = [getattr(task, split) for task in tasks]
        correct = sum(
            (
                ((part[:, None, :] - centres[None]) ** 2).sum(dim=-1).argmin(dim=1)
                == index
            ).sum().item()
            for index, part in enumerate(inputs)
        )
        total = sum(len(part) for part in inputs)
        assert correct / total > 0.99


def test_test_targets_are_the_clean_function_and_train_targets_are_noisy():
    task = _stream()[0]
    assert torch.allclose(task.test_targets, target_function(task.test_inputs))
    residual = task.train_targets - target_function(task.train_inputs)
    assert residual.std().item() == pytest.approx(0.1 * (task.train_targets - residual).std().item(), rel=0.05)


def test_stream_is_deterministic_given_the_seed():
    first, second, other = _stream(seed=1), _stream(seed=1), _stream(seed=2)
    assert torch.equal(first[0].train_inputs, second[0].train_inputs)
    assert torch.equal(first[0].train_targets, second[0].train_targets)
    assert not torch.equal(first[0].train_inputs, other[0].train_inputs)


def test_every_part_uses_a_different_input_distribution():
    tasks = _stream()
    means = torch.stack([task.test_inputs.mean(dim=0) for task in tasks])
    distances = torch.cdist(means, means).fill_diagonal_(float("inf"))
    assert distances.min().item() > 2.0
    covariances = [torch.cov(task.test_inputs.T) for task in tasks]
    for first in range(len(tasks)):
        for second in range(first):
            assert not torch.allclose(
                covariances[first], covariances[second], atol=1e-3
            )


def test_rotated_targets_differ_from_the_shared_function():
    rotated = regression_stream(rotated_targets=True)
    for task in rotated:
        assert not torch.allclose(
            task.test_targets, target_function(task.test_inputs), atol=1e-3
        )


def test_single_task_stream_works():
    tasks = regression_stream(n_tasks=1)
    assert len(tasks) == 1
    assert torch.isfinite(tasks[0].test_targets).all()


@pytest.mark.parametrize(
    "options",
    [
        {"n_tasks": 0},
        {"dim": 3},
        {"n_train": 0},
        {"n_test": 0},
        {"center_radius": 0.0},
        {"spread": 0.0},
        {"anisotropy": 0.5},
        {"noise": 1.0},
        {"noise": -0.1},
    ],
)
def test_invalid_arguments_raise_value_error(options):
    with pytest.raises(ValueError):
        regression_stream(**options)


def test_anisotropy_controls_the_covariance_spread():
    isotropic = regression_stream(n_tasks=2, anisotropy=1.0, seed=0)
    stretched = regression_stream(n_tasks=2, anisotropy=4.0, seed=0)

    def axis_ratio(task) -> float:
        eigenvalues = torch.linalg.eigvalsh(torch.cov(task.test_inputs.T))
        return (eigenvalues[-1] / eigenvalues[0]).item()

    assert axis_ratio(isotropic[0]) < 2.0
    assert axis_ratio(stretched[0]) > 5.0
