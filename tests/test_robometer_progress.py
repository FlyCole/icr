"""Exercise progress conversion without loading Robometer's model dependencies."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party/robometer/robometer/models/utils.py"
)
SPEC = importlib.util.spec_from_file_location("robometer_progress_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = utils
SPEC.loader.exec_module(utils)


@pytest.fixture(params=["torch", "numpy"])
def backend(request):
    return torch.as_tensor if request.param == "torch" else np.asarray


def as_numpy(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else value


def test_prediction_logits_summing_to_one_are_always_softmaxed(backend):
    # Both negative raw logits and nonnegative raw logits can sum to one.
    logits = backend([[-1.0, 2.0], [0.2, 0.8]])
    result = utils.convert_bins_to_continuous(logits)
    expected = np.array([1 / (1 + np.exp(-3)), 1 / (1 + np.exp(-0.6))])
    np.testing.assert_allclose(as_numpy(result), expected, rtol=1e-6)
    assert np.all((as_numpy(result) >= 0) & (as_numpy(result) <= 1))


def test_logits_and_probabilities_have_explicit_distinct_semantics(backend):
    values = backend([0.2, 0.8])
    prediction = utils.convert_bins_to_continuous(values)
    target = utils.convert_bins_to_continuous(values, from_logits=False)
    assert float(prediction) == pytest.approx(1 / (1 + np.exp(-0.6)))
    assert float(target) == pytest.approx(0.8)


def test_stable_softmax_handles_extreme_finite_logits(backend):
    logits = backend(np.array([[-1e308, 1e308], [1e308, -1e308], [1e308, 1e308]]))
    with np.errstate(all="raise"):
        result = utils.convert_bins_to_continuous(logits)
    np.testing.assert_allclose(as_numpy(result), [1.0, 0.0, 0.5])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_logits_accumulate_in_float32(dtype):
    result = utils.convert_bins_to_continuous(
        torch.tensor([[-10000.0, 10000.0], [0.0, 0.0]], dtype=dtype)
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([1.0, 0.5]))


def test_soft_and_onehot_targets_use_weighted_centers(backend):
    target = backend([[[0.25, 0.75, 0.0], [0.0, 0.0, 1.0]]])
    result = utils.convert_discrete_target_to_continuous(target, num_bins=3)
    np.testing.assert_allclose(as_numpy(result), [[0.375, 1.0]])


def test_integer_target_indices_return_float_progress(backend):
    indices = backend(np.array([[0, 1, 2, 3, 4]], dtype=np.int64))
    result = utils.convert_discrete_target_to_continuous(indices, num_bins=5)
    np.testing.assert_allclose(as_numpy(result), [[0.0, 0.25, 0.5, 0.75, 1.0]])
    assert np.issubdtype(as_numpy(result).dtype, np.floating)


def test_zero_padded_targets_map_to_zero_without_mutation(backend):
    target = backend([[[0.0, 0.25, 0.75], [0.0, 0.0, 0.0]]])
    original = as_numpy(target).copy()
    result = utils.convert_discrete_target_to_continuous(target, num_bins=3)
    np.testing.assert_allclose(as_numpy(result), [[0.875, 0.0]])
    np.testing.assert_array_equal(as_numpy(target), original)


@pytest.mark.parametrize(
    "values",
    [
        [-0.1, 1.1],
        [0.0, 0.0],
        [0.2, 0.2],
        [np.nan, 1.0],
        [np.inf, 0.0],
        [[0.2, 0.8], [0.3, 0.4]],
    ],
)
def test_invalid_explicit_probabilities_are_rejected(backend, values):
    with pytest.raises(ValueError, match="finite|nonnegative and sum to one"):
        utils.convert_bins_to_continuous(backend(values), from_logits=False)


@pytest.mark.parametrize("values", [[np.nan, 0.0], [0.0, np.inf], [-np.inf, 0.0]])
def test_nonfinite_prediction_logits_are_rejected(backend, values):
    with pytest.raises(ValueError, match="finite"):
        utils.convert_bins_to_continuous(backend(values))


@pytest.mark.parametrize("values", [[[[-0.1, 1.1]]], [[[0.2, 0.2]]]])
def test_invalid_nonpadding_targets_are_rejected(backend, values):
    with pytest.raises(ValueError, match="nonnegative and sum to one"):
        utils.convert_discrete_target_to_continuous(backend(values), num_bins=2)


@pytest.mark.parametrize("values", [[[-1, 0]], [[0.5, 1.0]], [[0, 5]]])
def test_invalid_target_indices_are_rejected(backend, values):
    with pytest.raises(ValueError, match="integers in"):
        utils.convert_discrete_target_to_continuous(backend(values), num_bins=5)


def test_empty_bin_axis_is_rejected(backend):
    with pytest.raises(ValueError, match="nonempty"):
        utils.convert_bins_to_continuous(backend([]))
