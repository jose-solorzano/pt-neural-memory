import pytest
import torch

from pt_neural_memory.benchmarks.forgetting_bench import (
    DEFAULT_PROTOCOL,
    EWCLearner,
    LEARNERS,
    METHODS,
    MLP,
    MemoryLearner,
    Protocol,
    ReplayLearner,
    Run,
    SILearner,
    SequentialLearner,
    aggregate,
    benchmark,
    r2_score,
    summarize,
)
from pt_neural_memory.forgetting_data import regression_stream

FAST = Protocol(steps_per_task=200, hidden=32)
SMALL = {"n_train": 128, "n_test": 64, "dim": 4}
BENCH = {**SMALL, "n_tasks": 3}
TINY_MEMORY = {"kind": "rff", "width": 8, "scale": 1.0, "n_bits": 4, "n_blocks": 1}


def test_r2_score_is_one_for_a_perfect_prediction_and_zero_for_the_mean():
    generator = torch.Generator().manual_seed(0)
    targets = torch.randn(64, 1, generator=generator)
    assert r2_score(targets, targets) == pytest.approx(1.0)
    assert r2_score(targets.mean().expand_as(targets), targets) == pytest.approx(
        0.0, abs=1e-6
    )


def test_summary_metrics_match_a_hand_computed_matrix():
    matrix = [
        [0.9, 0.1, -0.2],
        [0.7, 0.8, 0.0],
        [0.3, 0.5, 0.6],
    ]
    summary = summarize(matrix, baseline=[0.0, 0.0, 0.0])
    assert summary["acc"] == pytest.approx((0.3 + 0.5 + 0.6) / 3)
    assert summary["bwt"] == pytest.approx(((0.3 - 0.9) + (0.5 - 0.8)) / 2)
    assert summary["la"] == pytest.approx((0.9 + 0.8 + 0.6) / 3)
    assert summary["fwt"] == pytest.approx((0.1 - 0.0 + 0.0 - 0.0) / 2)


def test_aggregate_averages_cells_and_summaries():
    runs = [
        Run("x", 0, [[1.0, 0.0], [0.5, 0.5]], [0.0, 0.0]),
        Run("x", 1, [[0.0, 1.0], [0.0, 0.0]], [0.0, 0.0]),
    ]
    result = aggregate(runs)
    assert result.matrix[0][0] == pytest.approx(0.5)
    assert result.matrix[1][0] == pytest.approx(0.25)
    assert result.summary["acc"] == pytest.approx(0.25)
    assert result.spread["acc"] == pytest.approx(0.353553, rel=1e-5)


def test_all_methods_start_from_the_same_weights():
    inputs = torch.randn(32, 4)
    naive = SequentialLearner(4, seed=7)
    ewc = EWCLearner(4, seed=7)
    si = SILearner(4, seed=7)
    replay = ReplayLearner(4, seed=7)
    reference = MLP(4, DEFAULT_PROTOCOL.hidden, DEFAULT_PROTOCOL.depth,
                    torch.Generator().manual_seed(7))(inputs)
    for learner in (naive, ewc, si, replay):
        assert torch.equal(learner.predict(inputs), reference)


def test_naive_learns_the_current_part():
    run = benchmark("naive", 0, BENCH, protocol=FAST)
    assert run.matrix[0][0] > 0.5


def test_accuracy_matrix_is_complete_and_finite():
    run = benchmark("replay", 0, BENCH, protocol=FAST)
    assert len(run.matrix) == 3
    assert all(len(row) == 3 for row in run.matrix)
    assert all(torch.isfinite(torch.tensor(row)).all() for row in run.matrix)


def test_ewc_penalty_is_zero_right_after_consolidation_and_positive_later():
    tasks = regression_stream(n_tasks=2, **SMALL)
    learner = EWCLearner(4, seed=0, protocol=FAST)
    learner.train_task(tasks[0])
    assert learner._penalty().item() == pytest.approx(0.0, abs=1e-12)
    learner.train_task(tasks[1])
    assert learner._penalty().item() > 0.0


def test_synaptic_intelligence_importance_is_non_negative():
    tasks = regression_stream(n_tasks=1, **SMALL)
    learner = SILearner(4, seed=0, protocol=FAST)
    learner.train_task(tasks[0])
    importance = learner._anchors[0][1]
    assert all((weight >= 0).all() for weight in importance)


def test_replay_store_holds_past_parts_only():
    tasks = regression_stream(n_tasks=3, **SMALL)
    learner = ReplayLearner(4, seed=0, protocol=FAST, buffer_per_task=8)
    learner.train_task(tasks[0])
    assert len(learner._buffer_inputs) == 8
    learner.train_task(tasks[1])
    assert len(learner._buffer_inputs) == 16
    assert len(learner._buffer_targets) == 16
    # Every stored row is a training row of one of the two parts seen.
    stored = learner._buffer_inputs
    sources = torch.cat([tasks[0].train_inputs, tasks[1].train_inputs])
    matches = (stored[:, None, :] - sources[None]).abs().sum(dim=-1) < 1e-5
    assert matches.any(dim=1).all()
    # And the store does not yet contain part 3's data.
    later = tasks[2].train_inputs
    assert not ((stored[:, None, :] - later[None]).abs().sum(dim=-1) < 1e-5).any()


def test_replay_forgets_less_than_naive():
    naive = benchmark("naive", 0, BENCH, protocol=FAST)
    replay = benchmark("replay", 0, BENCH, protocol=FAST, buffer_per_task=16)
    assert summarize(replay.matrix, replay.baseline)["bwt"] > summarize(
        naive.matrix, naive.baseline
    )["bwt"]


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        benchmark("does-not-exist", 0, SMALL)


def test_addressed_memory_is_a_registered_method():
    assert "addressed" in LEARNERS
    assert "addressed" in METHODS


def test_memory_learner_stores_every_training_pair():
    tasks = regression_stream(n_tasks=2, **SMALL)
    learner = MemoryLearner(4, 0, protocol=FAST, **TINY_MEMORY)
    learner.train_task(tasks[0])
    assert learner.stored_pairs == 128
    learner.train_task(tasks[1])
    assert learner.stored_pairs == 256


def test_memory_learner_can_keep_only_a_subset_of_each_part():
    tasks = regression_stream(n_tasks=1, **SMALL)
    learner = MemoryLearner(
        4, 0, protocol=FAST, samples_per_task=16, **TINY_MEMORY
    )
    learner.train_task(tasks[0])
    assert learner.stored_pairs == 16
    with pytest.raises(ValueError, match="samples_per_task"):
        MemoryLearner(4, 0, samples_per_task=0)


def test_memory_predictions_are_deterministic():
    tasks = regression_stream(n_tasks=1, **SMALL)
    learner = MemoryLearner(4, 0, protocol=FAST, **TINY_MEMORY)
    learner.train_task(tasks[0])
    first = learner.predict(tasks[0].test_inputs)
    second = learner.predict(tasks[0].test_inputs)
    assert torch.equal(first, second)


def test_addressed_memory_retains_earlier_parts_better_than_naive():
    naive = benchmark("naive", 0, BENCH, protocol=FAST)
    addressed = benchmark("addressed", 0, BENCH, protocol=FAST, **TINY_MEMORY)
    naive_summary = summarize(naive.matrix, naive.baseline)
    addressed_summary = summarize(addressed.matrix, addressed.baseline)
    assert addressed_summary["bwt"] > naive_summary["bwt"]
    assert addressed_summary["la"] > 0.5


def test_addressed_matrix_is_complete_and_finite():
    run = benchmark("addressed", 0, BENCH, protocol=FAST, **TINY_MEMORY)
    assert len(run.matrix) == 3
    assert all(len(row) == 3 for row in run.matrix)
    assert all(torch.isfinite(torch.tensor(row)).all() for row in run.matrix)
