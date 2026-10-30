import pytest
import torch

from pt_neural_memory.mem_rls import RLSCellBank, RLSMemCell

D = 6
DV = 4


def pairs(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, D, generator=generator),
        torch.randn(n, DV, generator=generator),
    )


def one(index: int) -> torch.Tensor:
    return torch.tensor([index], dtype=torch.int64)


def test_a_bank_cell_equals_the_rls_cell_after_the_same_batch():
    keys, values = pairs(7)
    lam = 0.7
    cell = RLSMemCell(D, DV, rr_lambda=lam)
    cell.store(keys, values)
    bank = RLSCellBank(D, DV, 3, rr_lambda=lam)
    bank.store(torch.full((7,), 1, dtype=torch.int64), keys, values)
    assert torch.allclose(bank.weights[one(1)][0], cell.weights, atol=1e-4)
    assert torch.allclose(bank.covariance[one(1)][0], cell.covariance, atol=1e-4)


def test_a_bank_cell_equals_the_rls_cell_after_sequential_writes():
    keys, values = pairs(5, seed=2)
    cell = RLSMemCell(D, DV)
    bank = RLSCellBank(D, DV, 1)
    for i in range(5):
        cell.store(keys[i : i + 1], values[i : i + 1])
        bank.store(torch.zeros(1, dtype=torch.int64), keys[i : i + 1], values[i : i + 1])
    assert torch.allclose(bank.weights[one(0)][0], cell.weights, atol=1e-6)
    assert torch.allclose(bank.covariance[one(0)][0], cell.covariance, atol=1e-6)


def test_a_bank_cell_predicts_what_the_rls_cell_predicts():
    keys, values = pairs(8, seed=3)
    cell = RLSMemCell(D, DV)
    cell.store(keys, values)
    bank = RLSCellBank(D, DV, 1)
    bank.store(torch.zeros(8, dtype=torch.int64), keys, values)
    assert torch.allclose(
        bank.predict(torch.zeros(8, dtype=torch.int64), keys), cell(keys), atol=1e-6
    )


def test_repeated_indices_are_applied_in_order():
    keys, values = pairs(5, seed=4)
    batched = RLSCellBank(D, DV, 2)
    separate = RLSCellBank(D, DV, 2)
    batched.store(torch.zeros(5, dtype=torch.int64), keys, values)
    for i in range(5):
        separate.store(torch.zeros(1, dtype=torch.int64), keys[i : i + 1], values[i : i + 1])
    assert torch.allclose(batched.weights, separate.weights, atol=1e-6)
    assert torch.allclose(batched.covariance, separate.covariance, atol=1e-6)


def test_a_store_larger_than_the_chunk_boundary_is_exact():
    keys, values = pairs(2 * 512 + 3, seed=5)
    index = torch.randint(
        0, 4, (keys.shape[0],), generator=torch.Generator().manual_seed(0)
    )
    chunked = RLSCellBank(D, DV, 4)
    chunked.store(index, keys, values)
    halves = RLSCellBank(D, DV, 4)
    halves.store(index[:512], keys[:512], values[:512])
    halves.store(index[512:], keys[512:], values[512:])
    assert torch.allclose(chunked.weights, halves.weights, atol=1e-6)
    assert torch.allclose(chunked.covariance, halves.covariance, atol=1e-6)


def test_an_unwritten_cell_is_the_prior_and_answers_zero():
    bank = RLSCellBank(D, DV, 2, rr_lambda=0.5)
    assert torch.equal(bank.weights[one(1)], torch.zeros(1, DV, D))
    assert torch.allclose(bank.covariance[one(1)][0], 2.0 * torch.eye(D))
    query = torch.randn(1, D)
    assert torch.equal(bank.predict(one(1), query), torch.zeros(1, DV))
    assert torch.allclose(
        bank.variance(one(1), query), query.square().sum(-1) / 0.5, atol=1e-5
    )
    assert not bank.written(one(1)).item()


def test_predict_is_the_posterior_map_applied_to_the_key():
    keys, values = pairs(4, seed=6)
    bank = RLSCellBank(D, DV, 1)
    bank.store(torch.zeros(4, dtype=torch.int64), keys, values)
    query = torch.randn(3, D)
    expected = torch.einsum("oi,ni->no", bank.weights[one(0)][0], query)
    assert torch.allclose(
        bank.predict(torch.zeros(3, dtype=torch.int64), query), expected, atol=1e-5
    )


def test_variance_falls_after_a_write_and_is_not_the_prediction():
    bank = RLSCellBank(D, DV, 1)
    key = torch.randn(1, D)
    before = bank.variance(one(0), key).item()
    bank.store(torch.zeros(1, dtype=torch.int64), key, torch.randn(1, DV))
    after = bank.variance(one(0), key).item()
    assert after < before
    assert after < 2.0


def test_float64_state_is_preserved():
    bank = RLSCellBank(D, DV, 2, dtype=torch.float64)
    assert bank.weights.dtype == torch.float64
    assert bank.covariance.dtype == torch.float64
    keys, values = pairs(3)
    bank.store(torch.zeros(3, dtype=torch.int64), keys.double(), values.double())
    assert bank.predict(torch.zeros(1, dtype=torch.int64), keys.double()[:1]).dtype == torch.float64
    assert bank.variance(torch.zeros(1, dtype=torch.int64), keys.double()[:1]).dtype == torch.float64


def test_state_is_the_map_and_the_covariance():
    bank = RLSCellBank(D, DV, 5)
    assert bank.weights.shape == (5, DV, D)
    assert bank.covariance.shape == (5, D, D)
    assert sum(buffer.numel() for buffer in bank.buffers()) == 5 * (DV * D + D * D)


def test_store_is_no_grad_by_default_and_can_keep_the_graph():
    index = torch.tensor([0, 1], dtype=torch.int64)
    plain = RLSCellBank(D, DV, 3)
    plain_key = torch.randn(2, D, requires_grad=True)
    plain.store(index, plain_key, torch.randn(2, DV))
    assert plain.weights.grad_fn is None
    assert plain.covariance.grad_fn is None
    assert not plain.weights.requires_grad

    grad_bank = RLSCellBank(D, DV, 3)
    grad_key = torch.randn(2, D, requires_grad=True)
    grad_bank.store(index, grad_key, torch.randn(2, DV), grad_enabled=True)
    assert grad_bank.weights.grad_fn is not None
    assert grad_bank.covariance.grad_fn is not None
    (grad_bank.weights.sum() + grad_bank.covariance.sum()).backward()
    assert grad_key.grad is not None
    assert torch.isfinite(grad_key.grad).all()


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        RLSCellBank(0, DV, 1)
    with pytest.raises(ValueError):
        RLSCellBank(D, 0, 1)
    with pytest.raises(ValueError):
        RLSCellBank(D, DV, 0)
    with pytest.raises(ValueError):
        RLSCellBank(D, DV, 1, rr_lambda=0.0)
    bank = RLSCellBank(D, DV, 2)
    keys, values = pairs(3)
    with pytest.raises(ValueError):
        bank.store(torch.zeros(3, 1, dtype=torch.int64), keys, values)
    with pytest.raises(ValueError):
        bank.store(torch.zeros(2, dtype=torch.int64), keys, values)
    with pytest.raises(ValueError):
        bank.store(torch.zeros(2, dtype=torch.int64), torch.randn(2, D + 1), torch.randn(2, DV))
    with pytest.raises(ValueError):
        bank.store(torch.zeros(2, dtype=torch.int64), torch.randn(2, D), torch.randn(2, DV + 1))
    with pytest.raises(ValueError):
        bank.store(torch.tensor([2], dtype=torch.int64), torch.randn(1, D), torch.randn(1, DV))
    with pytest.raises(ValueError):
        bank.store(torch.zeros(1, dtype=torch.int32), torch.randn(1, D), torch.randn(1, DV))
    with pytest.raises(ValueError):
        bank.predict(torch.zeros(2, dtype=torch.int64), torch.randn(3, D))
    with pytest.raises(ValueError):
        bank.variance(torch.zeros(2, dtype=torch.int64), torch.randn(3, D))
