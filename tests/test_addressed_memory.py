import pytest
import torch

from pt_neural_memory.addressed_memory import AddressedMemory

N_BITS = 6
N_BLOCKS = 3
KEY_DIM = 8
VALUE_DIM = 4
# Small enough that the ridge shrink is far below the test tolerance; with
# float64 state the update stays well conditioned at this lambda.
LAMBDA = 1e-9


def make_memory(**kwargs: object) -> AddressedMemory:
    """Small memory with several blocks, cheap enough to inspect cell by cell."""
    kwargs.setdefault("rr_lambda", LAMBDA)
    kwargs.setdefault("dtype", torch.float64)
    return AddressedMemory(
        KEY_DIM, VALUE_DIM, n_blocks=N_BLOCKS, n_bits=N_BITS, **kwargs
    )


def random_pairs(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, KEY_DIM, generator=generator, dtype=torch.float64),
        torch.randn(n, VALUE_DIM, generator=generator, dtype=torch.float64),
    )


def written_cells(memory: AddressedMemory) -> torch.Tensor:
    """Boolean ``(n_blocks, 2**n_bits)`` mask of the cells that were written."""
    norms = memory.cells.weights.abs().sum(dim=(-1, -2))
    return (norms > 0).view(N_BLOCKS, 2**N_BITS)


def test_block_projections_are_orthonormal():
    memory = make_memory()
    gram = memory.projection @ memory.projection.transpose(1, 2)
    identity = torch.eye(N_BITS, dtype=memory.projection.dtype).expand(
        N_BLOCKS, N_BITS, N_BITS
    )
    assert torch.allclose(gram, identity, atol=1e-5)


def test_addresses_are_in_range_deterministic_and_block_specific():
    memory = make_memory()
    keys, _ = random_pairs(9)
    addresses = memory.addresses(keys)
    assert addresses.shape == (9, N_BLOCKS)
    assert addresses.dtype == torch.int64
    assert torch.equal(addresses, memory.addresses(keys))
    assert ((addresses >= 0) & (addresses < 2**N_BITS)).all()
    # Independent projections: the blocks must not collapse to one address.
    assert not torch.equal(addresses[:, 0], addresses[:, 1])


def test_bit_patterns_match_the_projection_signs():
    memory = make_memory()
    keys, _ = random_pairs(5)
    assert torch.equal(
        memory.bit_patterns(keys),
        memory.projections(keys) >= 0,
    )
    powers = 1 << torch.arange(N_BITS)
    expected = (memory.bit_patterns(keys).long() * powers).sum(dim=-1)
    assert torch.equal(memory.addresses(keys), expected)


def test_empty_memory_answers_zero_at_prior_variance():
    memory = make_memory()
    keys, _ = random_pairs(4)
    assert torch.equal(memory(keys), torch.zeros(4, VALUE_DIM, dtype=torch.float64))
    expected = keys.square().sum(dim=-1) / LAMBDA
    assert torch.allclose(memory.read_variances(keys), expected.expand(N_BLOCKS, 4))


def test_single_pair_is_recalled_exactly_at_every_radius():
    for radius in (0, 1, 2):
        memory = make_memory(write_radius=radius)
        key, value = random_pairs(1)
        memory.store(key, value)
        assert torch.allclose(memory(key), value, atol=1e-6)


def test_radius_zero_touches_only_the_own_address():
    memory = make_memory(write_radius=0)
    key, value = random_pairs(1)
    memory.store(key, value)
    base = memory.addresses(key)[0]
    touched = written_cells(memory)
    for block in range(N_BLOCKS):
        cells = touched[block].nonzero().flatten()
        assert cells.tolist() == [int(base[block])]


def test_radius_one_touches_the_own_address_and_every_single_bit_neighbour():
    memory = make_memory(write_radius=1)
    key, value = random_pairs(1)
    memory.store(key, value)
    base = memory.addresses(key)[0]
    touched = written_cells(memory)
    for block in range(N_BLOCKS):
        expected = {int(base[block]) ^ (1 << bit) for bit in range(N_BITS)}
        expected.add(int(base[block]))
        assert set(touched[block].nonzero().flatten().tolist()) == expected


def test_every_written_cell_learns_the_pair():
    memory = make_memory(write_radius=1)
    key, value = random_pairs(1)
    memory.store(key, value)
    touched = written_cells(memory)
    assert touched.sum().item() == N_BLOCKS * (N_BITS + 1)
    blocks, cells = touched.nonzero().transpose(0, 1)
    flat = blocks * memory.n_cells + cells
    assert torch.allclose(
        memory.cells.predict(flat, key.expand(flat.shape[0], -1)),
        value.expand(flat.shape[0], -1),
        atol=1e-6,
    )


def test_a_cell_recalls_several_distinct_keys_that_share_an_address():
    """The whole point of an RLS cell: saturation is not one pair per cell."""
    memory = AddressedMemory(
        KEY_DIM,
        VALUE_DIM,
        n_blocks=1,
        n_bits=3,
        write_radius=0,
        rr_lambda=LAMBDA,
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(0)
    by_address: dict[int, list[torch.Tensor]] = {}
    while True:
        candidate = torch.randn(KEY_DIM, generator=generator, dtype=torch.float64)
        address = int(memory.addresses(candidate.unsqueeze(0))[0, 0])
        bucket = by_address.setdefault(address, [])
        bucket.append(candidate)
        if len(bucket) == 3:
            break
    keys = torch.stack(by_address[address])
    values = torch.randn(3, VALUE_DIM, generator=generator, dtype=torch.float64)
    memory.store(keys, values)
    assert torch.allclose(memory(keys), values, atol=1e-6)


def test_batch_store_equals_sequential_store():
    keys, values = random_pairs(6)
    batched = make_memory(write_radius=1)
    sequential = make_memory(write_radius=1)
    batched.store(keys, values)
    for key, value in zip(keys, values):
        sequential.store(key.unsqueeze(0), value.unsqueeze(0))
    assert torch.allclose(batched.cells.weights, sequential.cells.weights, atol=1e-9)
    assert torch.allclose(
        batched.cells.covariance, sequential.cells.covariance, atol=1e-9
    )


def test_store_is_invariant_to_pair_order():
    keys, values = random_pairs(7)
    forward = make_memory()
    shuffled = make_memory()
    order = torch.randperm(7, generator=torch.Generator().manual_seed(0))
    forward.store(keys, values)
    shuffled.store(keys[order], values[order])
    assert torch.allclose(forward.cells.weights, shuffled.cells.weights, atol=1e-9)
    assert torch.allclose(
        forward.cells.covariance, shuffled.cells.covariance, atol=1e-9
    )


def test_repeated_store_keeps_the_prediction():
    memory = make_memory()
    key, value = random_pairs(1)
    memory.store(key, value)
    before = memory(key)
    memory.store(key, value)
    assert torch.allclose(memory(key), before, atol=1e-6)


def test_empty_batch_is_a_no_op():
    memory = make_memory()
    before = {name: buffer.clone() for name, buffer in memory.state_dict().items()}
    memory.store(torch.empty(0, KEY_DIM), torch.empty(0, VALUE_DIM))
    for name, buffer in memory.state_dict().items():
        assert torch.equal(buffer, before[name])


def test_float32_state_is_promoted_to_at_least_float32():
    memory = make_memory(dtype=torch.float32, rr_lambda=1.0)
    assert memory.cells.weights.dtype == torch.float32
    keys = random_pairs(3)[0].float()
    values = random_pairs(3)[1].float()
    memory.store(keys, values)
    assert memory(keys).dtype == torch.float32


def test_state_is_the_map_and_the_covariance():
    memory = make_memory()
    cells = N_BLOCKS * 2**N_BITS
    assert memory.projection.shape == (N_BLOCKS, N_BITS, KEY_DIM)
    assert memory.cells.weights.shape == (cells, VALUE_DIM, KEY_DIM)
    assert memory.cells.covariance.shape == (cells, KEY_DIM, KEY_DIM)
    assert sum(buffer.numel() for buffer in memory.buffers()) == (
        N_BLOCKS * N_BITS * KEY_DIM
        + cells * (VALUE_DIM * KEY_DIM + KEY_DIM**2)
    )


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        AddressedMemory(0, 4)
    with pytest.raises(ValueError):
        AddressedMemory(4, 0)
    with pytest.raises(ValueError):
        AddressedMemory(4, 4, n_blocks=0)
    with pytest.raises(ValueError):
        AddressedMemory(4, 4, n_bits=0)
    with pytest.raises(ValueError):
        AddressedMemory(4, 4, n_bits=21)
    with pytest.raises(ValueError):
        AddressedMemory(4, 4, n_bits=5)
    with pytest.raises(ValueError):
        AddressedMemory(4, 4, n_bits=4, write_radius=5)
    with pytest.raises(ValueError):
        AddressedMemory(4, 4, rr_lambda=0.0)
    memory = AddressedMemory(4, 4, n_bits=3)
    with pytest.raises(ValueError):
        memory.store(torch.randn(4), torch.randn(4))
    with pytest.raises(ValueError):
        memory.store(torch.randn(2, 4), torch.randn(3, 4))
    with pytest.raises(ValueError):
        memory.store(torch.randn(2, 4), torch.randn(2, 5))
    with pytest.raises(ValueError):
        memory(torch.randn(2, 5))
