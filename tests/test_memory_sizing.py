import pytest

from pt_neural_memory.addressed_memory import VerifiedMemory
from pt_neural_memory.memory_sizing import (
    balanced_layout,
    bits_for_capacity,
    capacity,
    pairs_per_cell,
    size_for_budget,
    state_bytes,
    state_parts,
    writers_per_cell,
)


def test_state_bytes_matches_the_actual_memory():
    memory = VerifiedMemory(8, 4, n_blocks=2, n_bits=5, seed=0)
    actual = sum(
        buffer.numel() * buffer.element_size() for buffer in memory.buffers()
    )
    assert state_bytes(5, 2, 4, 8) == actual


def test_state_parts_sum_and_component_sizes():
    parts = state_parts(10, 2, 16, 64)
    assert set(parts) == {"maps", "covariances", "projection"}
    assert sum(parts.values()) == state_bytes(10, 2, 16, 64)
    cells = 1 << 10
    assert parts["maps"] == 4 * 2 * cells * 16 * 64
    assert parts["covariances"] == 4 * 2 * cells * 64 * 64
    assert parts["projection"] == 4 * 2 * 10 * 64


def test_capacity_grows_with_bits_and_with_the_key_dimension():
    assert capacity(12, 3, 32) < capacity(13, 3, 32)
    assert capacity(12, 3, 64) > capacity(12, 3, 32)
    assert capacity(12, 3, 32) == int(pairs_per_cell(12, 32) * 2**12)
    assert writers_per_cell(32) == pytest.approx(25.6)


def test_widening_the_address_reduces_pairs_per_cell():
    """Every extra address bit doubles the cells but also doubles the writes."""
    assert pairs_per_cell(13, 64) < pairs_per_cell(12, 64)


def test_4gib_with_16_number_values_picks_three_blocks_of_2_pow_16():
    sizing = size_for_budget(4 * 2**30, 16, key_dim=64)
    assert (sizing.n_bits, sizing.n_blocks) == (16, 3)
    assert 180_000 < sizing.capacity < 210_000
    assert sizing.total_bytes <= 4 * 2**30
    assert "3 blocks" in sizing.summary()


def test_size_for_budget_respects_the_budget_and_block_bounds():
    budget = 4 * 2**30
    for value_dim in (16, 64, 256, 768):
        sizing = size_for_budget(budget, value_dim, key_dim=64)
        assert 2 <= sizing.n_blocks <= 8
        assert sizing.total_bytes <= budget
        assert sizing.capacity == capacity(
            sizing.n_bits, sizing.n_blocks, 64, sizing.writers_per_cell / 64
        )
        assert sizing.probes_per_query == sizing.n_blocks * (sizing.n_bits + 1)
        assert sizing.bytes_per_pair > 0


def test_bits_for_capacity_round_trip():
    for target in (1_000, 10_000, 100_000):
        n_bits = bits_for_capacity(target, key_dim=64)
        assert capacity(n_bits, 1, 64) >= target
        if n_bits > 1:
            assert capacity(n_bits - 1, 1, 64) < target


def test_balanced_layout_uses_two_or_three_blocks():
    for value_dim in (16, 64, 256, 768):
        sizing = balanced_layout(4 * 2**30, value_dim, key_dim=64)
        assert sizing.n_blocks in (2, 3)
        assert sizing.total_bytes <= 4 * 2**30
        assert sizing.n_bits <= 64
        assert sizing.capacity == int(
            sizing.writers_per_cell * 2**sizing.n_bits / (sizing.n_bits + 1)
        )
        assert sizing.probes_per_query == sizing.n_blocks * (sizing.n_bits + 1)


def test_balanced_layout_capacity_grows_with_the_budget():
    capacities = [
        balanced_layout(budget, 16, key_dim=64).capacity
        for budget in (2**24, 2**26, 2**28, 2**30)
    ]
    assert capacities == sorted(capacities)


def test_balanced_layout_follows_the_table_it_is_given():
    one = balanced_layout(2**30, 16, key_dim=64, admissible={1: 0.1, 6: 10.0})
    assert one.n_blocks == 6
    cheap = balanced_layout(2**30, 16, key_dim=64, admissible={1: 10.0, 6: 1.0})
    assert cheap.n_blocks == 1


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        capacity(12, 0, 32)
    with pytest.raises(ValueError):
        writers_per_cell(0)
    with pytest.raises(ValueError):
        writers_per_cell(32, fraction=0.0)
    with pytest.raises(ValueError):
        pairs_per_cell(0, 32)
    with pytest.raises(ValueError):
        state_bytes(0, 1, 4)
    with pytest.raises(ValueError):
        state_bytes(4, 1, 0)
    with pytest.raises(ValueError):
        size_for_budget(0, 16)
    with pytest.raises(ValueError):
        size_for_budget(100, 16, min_blocks=4, max_blocks=2)
    with pytest.raises(ValueError):
        size_for_budget(1_000, 16, key_dim=64, max_blocks=8)
    with pytest.raises(ValueError):
        bits_for_capacity(0, key_dim=64)
    with pytest.raises(ValueError):
        balanced_layout(0, 16)
    with pytest.raises(ValueError):
        balanced_layout(1_000, 0, key_dim=64)
    with pytest.raises(ValueError):
        balanced_layout(1_000, 16, key_dim=64)
