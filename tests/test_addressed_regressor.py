import pytest
import torch

from pt_neural_memory.addressed_regressor import AddressedRegressor, RandomFeatureMap


def r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        1.0 - (prediction - target).square().mean() / target.var(unbiased=False)
    )


def test_raw_key_is_the_input_plus_a_constant():
    inputs = torch.randn(5, 4)
    features = RandomFeatureMap(4, kind="raw", seed=0)
    keys = features(inputs)
    assert keys.shape == (5, 5)
    assert torch.equal(keys[:, 0], torch.ones(5))
    assert torch.equal(keys[:, 1:], inputs)


def test_feature_maps_have_the_documented_key_width():
    inputs = torch.randn(3, 4)
    rff = RandomFeatureMap(4, kind="rff", width=8, seed=0)
    assert rff.key_dim == 8 + 1 + 4
    assert rff(inputs).shape == (3, 13)
    relu = RandomFeatureMap(
        4, kind="relu", width=8, use_raw=False, use_bias=False, seed=0
    )
    assert relu.key_dim == 8
    assert relu(inputs).shape == (3, 8)


def test_features_are_deterministic_in_the_seed():
    inputs = torch.randn(6, 4)
    same = RandomFeatureMap(4, kind="rff", width=8, seed=3)(inputs)
    again = RandomFeatureMap(4, kind="rff", width=8, seed=3)(inputs)
    other = RandomFeatureMap(4, kind="rff", width=8, seed=4)(inputs)
    assert torch.equal(same, again)
    assert not torch.equal(same, other)


def test_features_reject_a_wrong_input_shape():
    features = RandomFeatureMap(4, seed=0)
    with pytest.raises(ValueError, match="inputs must have shape"):
        features(torch.randn(3, 5))


def test_invalid_feature_map_arguments_raise():
    with pytest.raises(ValueError, match="kind"):
        RandomFeatureMap(4, kind="fourier")
    with pytest.raises(ValueError, match="width"):
        RandomFeatureMap(4, width=0)
    with pytest.raises(ValueError, match="scale"):
        RandomFeatureMap(4, scale=0.0)
    with pytest.raises(ValueError, match="empty key"):
        RandomFeatureMap(4, kind="raw", use_raw=False)


def test_a_linear_function_is_interpolated_exactly():
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(64, 4, generator=generator)
    targets = inputs @ torch.randn(4, 1, generator=generator) + 0.3
    model = AddressedRegressor(
        4, kind="raw", n_blocks=1, n_bits=3, write_radius=1, rr_lambda=1e-3
    )
    model.store(inputs, targets)
    assert r2(model(inputs), targets) == pytest.approx(1.0, abs=1e-5)


def test_a_nonlinear_function_is_fitted_from_stored_pairs():
    generator = torch.Generator().manual_seed(0)
    inputs = torch.rand(512, 1, generator=generator) * 6.0 - 3.0
    targets = torch.sin(2.0 * inputs)
    train, test = torch.randperm(512, generator=generator).split(400)
    model = AddressedRegressor(
        1, kind="rff", width=32, scale=1.0, n_bits=4, n_blocks=1, rr_lambda=1e-3
    )
    model.store(inputs[train], targets[train])
    assert r2(model(inputs[test]), targets[test]) > 0.99


def test_sequential_store_matches_one_batch():
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(64, 4, generator=generator)
    targets = torch.sin(inputs.sum(dim=1, keepdim=True))
    options = dict(kind="rff", width=8, n_bits=3, n_blocks=1, seed=5)
    one_batch = AddressedRegressor(4, **options)
    two_batches = AddressedRegressor(4, **options)
    one_batch.store(inputs, targets)
    two_batches.store(inputs[:32], targets[:32])
    two_batches.store(inputs[32:], targets[32:])
    queries = torch.randn(16, 4, generator=generator)
    assert torch.allclose(
        one_batch(queries), two_batches(queries), atol=1e-4
    )


def test_verified_read_runs_and_agrees_with_the_own_address():
    generator = torch.Generator().manual_seed(2)
    inputs = torch.randn(64, 4, generator=generator)
    targets = inputs[:, :1] * 2.0
    model = AddressedRegressor(
        4,
        kind="rff",
        width=8,
        n_bits=4,
        n_blocks=1,
        read="verified",
        seed=0,
    )
    model.store(inputs, targets)
    predictions = model(inputs)
    assert predictions.shape == (64, 1)
    assert torch.isfinite(predictions).all()


def test_state_bytes_matches_the_table_arithmetic():
    width, input_dim, n_blocks, n_bits = 8, 4, 2, 4
    model = AddressedRegressor(
        input_dim, kind="rff", width=width, n_blocks=n_blocks, n_bits=n_bits
    )
    key_dim = width + 1 + input_dim
    cells = n_blocks * 2**n_bits
    expected = (
        width * input_dim  # feature weights
        + width  # feature offsets
        + n_blocks * n_bits * key_dim  # address projections
        + cells * 1 * key_dim  # cell maps (one output)
        + cells * key_dim**2  # cell covariances
    ) * 4
    assert model.state_bytes() == expected


def test_store_and_read_shape_errors_raise():
    model = AddressedRegressor(4, kind="rff", width=8, n_bits=3, n_blocks=1)
    with pytest.raises(ValueError, match="targets must have shape"):
        model.store(torch.randn(5, 4), torch.randn(5, 2))
    with pytest.raises(ValueError, match="batch sizes differ"):
        model.store(torch.randn(5, 4), torch.randn(4, 1))
    with pytest.raises(ValueError, match="read must be"):
        AddressedRegressor(4, read="best")
