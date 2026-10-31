import pytest
import torch

from pt_neural_memory.addressed_memory import CellRead, VerifiedMemory

N_BITS = 6
KEY_DIM = 16
VALUE_DIM = 4
LAMBDA = 1e-9


def make_memory(**kwargs: object) -> VerifiedMemory:
    """One block, so Hamming distances are the block's own address distances."""
    kwargs.setdefault("rr_lambda", LAMBDA)
    kwargs.setdefault("dtype", torch.float64)
    return VerifiedMemory(
        KEY_DIM, VALUE_DIM, n_blocks=1, n_bits=N_BITS, **kwargs
    )


def random_pairs(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, KEY_DIM, generator=generator, dtype=torch.float64),
        torch.randn(n, VALUE_DIM, generator=generator, dtype=torch.float64),
    )


def written_count(memory: VerifiedMemory) -> int:
    return int((memory.cells.weights.abs().sum(dim=(-1, -2)) > 0).sum())


def flip_lowest_margin_bits(
    memory: VerifiedMemory, key: torch.Tensor, count: int
) -> tuple[torch.Tensor, list[int]]:
    """A key at address distance ``count``, flipping the weakest bits.

    The projection rows are orthonormal, so negating the coordinate along row
    ``i`` flips bit ``i`` and leaves every other bit untouched.  Flipping the
    weakest bits is what key noise does, and it keeps the query close to the
    stored key in key space, which is the regime a linear cell extrapolates in.
    """
    query = key.clone()
    bits = []
    for bit in memory.projections(key)[0, 0].abs().argsort()[:count].tolist():
        direction = memory.projection[0, bit]
        query = query - 2.0 * (query @ direction).unsqueeze(-1) * direction
        bits.append(bit)
    return query, bits


def test_a_pure_cell_verifies_with_unit_variance_and_the_stored_value():
    memory = make_memory(write_radius=1)
    key, value = random_pairs(1)
    memory.store(key, value)
    read = memory.read_ball(key)
    assert torch.allclose(read.values, value.unsqueeze(0), atol=1e-6)
    assert torch.allclose(read.scores, torch.ones_like(read.scores), atol=1e-6)


def test_empty_memory_returns_zero_at_prior_variance():
    memory = make_memory()
    keys, _ = random_pairs(3)
    read = memory.read_ball(keys)
    assert torch.equal(read.values, torch.zeros(1, 3, VALUE_DIM, dtype=torch.float64))
    expected = keys.square().sum(dim=-1) / LAMBDA
    assert torch.allclose(read.scores, expected.unsqueeze(0))


def test_read_radius_corrects_an_address_error_that_the_center_cannot():
    memory = make_memory(write_radius=0, read_radius=1)
    key, value = random_pairs(1)
    memory.store(key, value)
    query, _ = flip_lowest_margin_bits(memory, key, count=1)
    assert (memory.bit_patterns(query) != memory.bit_patterns(key)).sum().item() == 1
    empty = memory.read_ball(query, radius=0).best()
    assert torch.equal(empty, torch.zeros(1, VALUE_DIM, dtype=torch.float64))
    assert torch.allclose(memory.read_ball(query, radius=1).best(), value, atol=2e-2)


def test_two_bit_error_needs_write_radius_plus_read_radius():
    memory = make_memory(write_radius=1, read_radius=1)
    key, value = random_pairs(1)
    memory.store(key, value)
    query, _ = flip_lowest_margin_bits(memory, key, count=2)
    assert (memory.bit_patterns(query) != memory.bit_patterns(key)).sum().item() == 2
    assert torch.allclose(memory.read_ball(query, radius=1).best(), value, atol=2e-2)


def test_an_unrelated_key_does_not_find_a_cell():
    memory = make_memory(write_radius=0, read_radius=1)
    key, value = random_pairs(1)
    memory.store(key, value)
    generator = torch.Generator().manual_seed(0)
    while True:
        query = torch.randn(1, KEY_DIM, generator=generator, dtype=torch.float64)
        distance = int(
            (memory.bit_patterns(query) != memory.bit_patterns(key)).sum()
        )
        if distance >= 3:
            break
    read = memory.read_ball(query, radius=1)
    assert torch.equal(read.best(), torch.zeros(1, VALUE_DIM, dtype=torch.float64))
    assert torch.allclose(
        read.scores, query.square().sum(dim=-1, keepdim=True) / LAMBDA
    )


def test_guided_read_returns_the_stored_pair():
    memory = make_memory(write_radius=1, read_radius=0)
    key, value = random_pairs(1)
    memory.store(key, value)
    read = memory.read_guided(key, search_bits=3)
    assert torch.allclose(read.best(), value, atol=1e-6)


def test_guided_search_follows_the_least_confident_bit():
    memory = make_memory(write_radius=0, read_radius=0)
    key, value = random_pairs(1)
    memory.store(key, value)
    query, bits = flip_lowest_margin_bits(memory, key, count=1)
    assert int(memory.projections(query)[0, 0].abs().argmin()) == bits[0]
    assert torch.allclose(
        memory.read_guided(query, search_bits=1).best(), value, atol=2e-2
    )


def test_batch_store_and_read_match_per_item_operations():
    memory = make_memory(write_radius=1)
    keys, values = random_pairs(5)
    memory.store(keys, values)
    batched = memory.read_ball(keys)
    for i in range(5):
        single = memory.read_ball(keys[i : i + 1])
        assert torch.allclose(batched.values[:, i], single.values[:, 0])
        assert torch.allclose(batched.scores[:, i], single.scores[:, 0])
        assert torch.equal(batched.indices[:, i], single.indices[:, 0])


def test_protect_bits_replicates_only_the_lowest_margin_bits():
    memory = make_memory(write_radius=1, protect_bits=2)
    key, value = random_pairs(1)
    memory.store(key, value)
    assert written_count(memory) == 3
    base = int(memory.addresses(key)[0, 0])
    weakest = memory.projections(key)[0, 0].abs().argsort()[:2].tolist()
    expected = {base} | {base ^ (1 << bit) for bit in weakest}
    touched = set(
        (memory.cells.weights.abs().sum(dim=(-1, -2)) > 0).nonzero().flatten().tolist()
    )
    assert touched == expected


def test_empty_batch_is_a_no_op():
    memory = make_memory()
    before = {name: buffer.clone() for name, buffer in memory.state_dict().items()}
    memory.store(torch.empty(0, KEY_DIM), torch.empty(0, VALUE_DIM))
    for name, buffer in memory.state_dict().items():
        assert torch.equal(buffer, before[name])


def test_float64_state_is_preserved():
    memory = make_memory(dtype=torch.float64)
    keys, values = random_pairs(3)
    memory.store(keys, values)
    assert memory.cells.covariance.dtype == torch.float64
    assert memory.read_ball(keys).values.dtype == torch.float64


def test_cell_read_aggregation():
    read = CellRead(
        values=torch.tensor([[[1.0]], [[2.0]]]),
        scores=torch.tensor([[3.0], [1.0]]),
        indices=torch.tensor([[[0]], [[2]]]),
    )
    assert torch.equal(read.best(), torch.tensor([[2.0]]))
    assert torch.equal(read.mean(), torch.tensor([[1.5]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_verified_read_runs_on_cuda():
    """The score buffer must live on the query's device."""
    device = torch.device("cuda")
    memory = make_memory(write_radius=1).to(device)
    keys = torch.randn(8, KEY_DIM, device=device, dtype=torch.float64)
    values = torch.randn(8, VALUE_DIM, device=device, dtype=torch.float64)
    memory.store(keys, values)
    assert memory(keys).device.type == "cuda"
    assert memory.read_ball(keys).best().device.type == "cuda"


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        make_memory(read_radius=N_BITS + 1)
    with pytest.raises(ValueError):
        make_memory(protect_bits=N_BITS + 1)
    memory = make_memory()
    with pytest.raises(ValueError):
        memory.read_ball(torch.randn(1, KEY_DIM, dtype=torch.float64), radius=-1)
    with pytest.raises(ValueError):
        memory.read_guided(torch.randn(1, KEY_DIM, dtype=torch.float64), search_bits=0)
    with pytest.raises(ValueError):
        memory.read_guided(torch.randn(1, KEY_DIM, dtype=torch.float64), search_bits=13)
