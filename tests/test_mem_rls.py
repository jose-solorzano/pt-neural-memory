import pytest
import torch

from pt_neural_memory.mem_rls import RLSMemCell


def _pairs(n: int, d: int, m: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randn(n, d, generator=gen)
    values = torch.randn(n, m, generator=gen)
    return keys, values


def _error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).square().sum().item() / target.square().sum().item()


def test_read_matches_the_stored_linear_map():
    cell = RLSMemCell(5, 3)
    key = torch.randn(7, 5)
    assert torch.allclose(cell(key), key @ cell.weights.T)


def test_rectangular_map_learns_a_single_pair():
    cell = RLSMemCell(5, 3, rr_lambda=1e-2)
    generator = torch.Generator().manual_seed(0)
    key = torch.randn(1, 5, generator=generator)
    value = torch.randn(1, 3, generator=generator)
    cell.store(key, value)
    assert torch.allclose(cell(key), value, atol=1e-2)


def test_single_batch_matches_the_ridge_solution():
    d, m, n, lam = 6, 4, 5, 0.7
    keys, values = _pairs(n, d, m)
    cell = RLSMemCell(d, m, rr_lambda=lam)
    w0 = cell.weights.clone()
    cell.store(keys, values)
    residual = values - keys @ w0.T
    expected = w0 + residual.T @ keys @ torch.linalg.inv(
        keys.T @ keys + lam * torch.eye(d)
    )
    assert torch.allclose(cell.weights, expected, atol=1e-5)


def test_sequential_updates_equal_one_batch_update():
    d, m, n = 6, 4, 5
    keys, values = _pairs(n, d, m, seed=3)
    batch_cell = RLSMemCell(d, m)
    seq_cell = RLSMemCell(d, m)
    seq_cell.load_state_dict(batch_cell.state_dict())
    batch_cell.store(keys, values)
    for i in range(n):
        seq_cell.store(keys[i : i + 1], values[i : i + 1])
    assert torch.allclose(batch_cell.weights, seq_cell.weights, atol=1e-4)
    assert torch.allclose(batch_cell.covariance, seq_cell.covariance, atol=1e-4)


def test_is_invariant_to_write_order():
    d, m, n = 6, 4, 9
    keys, values = _pairs(n, d, m, seed=5)
    cell_a = RLSMemCell(d, m)
    cell_b = RLSMemCell(d, m)
    cell_b.load_state_dict(cell_a.state_dict())
    cell_a.store(keys, values)
    for i in torch.randperm(n, generator=torch.Generator().manual_seed(0)):
        i = int(i)
        cell_b.store(keys[i : i + 1], values[i : i + 1])
    assert torch.allclose(cell_a.weights, cell_b.weights, atol=1e-4)


def test_empty_batch_is_a_no_op():
    cell = RLSMemCell(4, 3)
    before = {name: buf.clone() for name, buf in cell.state_dict().items()}
    cell.store(torch.empty(0, 4), torch.empty(0, 3))
    for name, buf in cell.state_dict().items():
        assert torch.equal(buf, before[name])


def test_zero_key_is_a_no_op():
    cell = RLSMemCell(4, 3)
    before = {name: buf.clone() for name, buf in cell.state_dict().items()}
    cell.store(torch.zeros(1, 4), torch.randn(1, 3))
    for name, buf in cell.state_dict().items():
        assert torch.allclose(buf, before[name])


def test_covariance_stays_symmetric_and_positive_definite():
    cell = RLSMemCell(12, 8)
    keys, values = _pairs(20, 12, 8)
    for i in range(20):
        cell.store(keys[i : i + 1], values[i : i + 1])
    covariance = cell.covariance
    assert torch.allclose(covariance, covariance.T, atol=1e-6)
    assert torch.linalg.eigvalsh(covariance).min() > 0


def test_state_is_the_map_plus_one_covariance():
    d, m = 16, 8
    cell = RLSMemCell(d, m)
    assert cell.weights.shape == (m, d)
    assert cell.covariance.shape == (d, d)
    assert sum(buf.numel() for buf in cell.buffers()) == m * d + d * d


def test_float64_state_is_preserved():
    cell = RLSMemCell(5, 3, dtype=torch.float64)
    assert cell.weights.dtype == torch.float64
    assert cell.covariance.dtype == torch.float64
    keys, values = _pairs(4, 5, 3)
    keys, values = keys.double(), values.double()
    cell.store(keys, values)
    assert cell(keys).dtype == torch.float64


def test_grad_enabled_keeps_the_graph_alive():
    cell = RLSMemCell(4, 3)
    key = torch.randn(2, 4, requires_grad=True)
    value = torch.randn(2, 3)
    cell.store(key, value, grad_enabled=True)
    cell(key).sum().backward()
    assert key.grad is not None
    assert torch.isfinite(key.grad).all()


def test_rewriting_a_batch_does_not_degrade_recall():
    cell = RLSMemCell(16, 8, rr_lambda=1e-4)
    keys, values = _pairs(8, 16, 8)
    cell.store(keys, values)
    first = _error(cell(keys), values)
    for _ in range(5):
        cell.store(keys, values)
    assert _error(cell(keys), values) <= first + 1e-6


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        RLSMemCell(4, 3, rr_lambda=0.0)
    with pytest.raises(ValueError):
        RLSMemCell(4, 3, decay=0.0)
    with pytest.raises(ValueError):
        RLSMemCell(4, 3, decay=1.5)
    cell = RLSMemCell(4, 3)
    with pytest.raises(ValueError):
        cell.store(torch.randn(4), torch.randn(3))
    with pytest.raises(ValueError):
        cell.store(torch.randn(2, 4), torch.randn(3, 3))
