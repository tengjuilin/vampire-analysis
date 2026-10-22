import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy as scipy_entropy

import vampire as vp


@pytest.mark.parametrize("wrap", [list, np.array, pd.Series])
def test_distribution(wrap):
    distribution = vp.shape_mode_distribution(wrap([2, 0, 2, 1, 2, 0]))
    np.testing.assert_allclose(distribution, [2 / 6, 1 / 6, 3 / 6])
    assert distribution.dtype == np.float64


def test_distribution_pads_empty_modes():
    distribution = vp.shape_mode_distribution([1, 1, 3], n_clusters=5)
    np.testing.assert_allclose(distribution, [0, 2 / 3, 0, 1 / 3, 0])
    assert distribution.sum() == pytest.approx(1.0)


def test_distribution_accepts_integral_floats():
    np.testing.assert_allclose(vp.shape_mode_distribution([0.0, 1.0]), [0.5, 0.5])


def test_distribution_of_model_labels(shapes):
    X, _ = shapes
    model = vp.Vampire(n_clusters=3, n_components=5, random_state=0).fit(X)
    distribution = vp.shape_mode_distribution(model.labels_, n_clusters=3)
    assert distribution.shape == (3,)
    assert distribution.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "kwargs", "match"),
    [
        ([], {}, "non-empty"),
        ([[0, 1]], {}, "1-D"),
        ([0, -1], {}, "non-negative"),
        ([0, 1.5], {}, "integers"),
        (["a"], {}, "integers"),
        ([0, 3], {"n_clusters": 3}, "below n_clusters"),
    ],
)
def test_distribution_rejects_bad_labels(labels, kwargs, match):
    with pytest.raises(ValueError, match=match):
        vp.shape_mode_distribution(labels, **kwargs)


def test_entropy_values():
    assert vp.shannon_entropy([0.25] * 4) == pytest.approx(np.log(4))
    assert vp.shannon_entropy([0.25] * 4, base=2) == pytest.approx(2.0)
    assert vp.shannon_entropy([1.0, 0.0, 0.0]) == 0.0
    p = [0.1, 0.0, 0.6, 0.3]
    assert vp.shannon_entropy(p) == pytest.approx(scipy_entropy(p))
    assert vp.shannon_entropy(p, base=10) == pytest.approx(scipy_entropy(p, base=10))
    assert isinstance(vp.shannon_entropy(p), float)


def test_entropy_ignores_zeros():
    assert vp.shannon_entropy([0.5, 0.0, 0.5]) == pytest.approx(np.log(2))


@pytest.mark.parametrize(
    ("distribution", "match"),
    [
        ([], "non-empty"),
        ([[0.5, 0.5]], "1-D"),
        ([1.5, -0.5], "non-negative"),
        ([0.5, np.nan], "finite"),
        ([2, 3], "sum to 1"),
    ],
)
def test_entropy_rejects_bad_distribution(distribution, match):
    with pytest.raises(ValueError, match=match):
        vp.shannon_entropy(distribution)
