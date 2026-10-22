import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy as scipy_entropy

import vampire as vp


@pytest.mark.parametrize("wrap", [list, np.array, pd.Series])
def test_distribution(wrap):
    distribution = vp.shape_mode_distribution(wrap([3, 1, 3, 2, 3, 1]))
    np.testing.assert_allclose(distribution, [2 / 6, 1 / 6, 3 / 6])
    assert distribution.dtype == np.float64


def test_distribution_pads_empty_modes():
    distribution = vp.shape_mode_distribution([2, 2, 4], n_clusters=5)
    np.testing.assert_allclose(distribution, [0, 2 / 3, 0, 1 / 3, 0])
    assert distribution.sum() == pytest.approx(1.0)


def test_distribution_accepts_integral_floats():
    np.testing.assert_allclose(vp.shape_mode_distribution([1.0, 2.0]), [0.5, 0.5])


def test_distribution_of_model_labels(shapes):
    X, _ = shapes
    model = vp.Vampire(n_clusters=3, n_components=5, random_state=0).fit(X)
    distribution = vp.shape_mode_distribution(model.labels_, n_clusters=3)
    assert distribution.shape == (3,)
    assert distribution.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        distribution, [np.mean(model.labels_ == m) for m in (1, 2, 3)]
    )


@pytest.mark.parametrize(
    ("labels", "kwargs", "match"),
    [
        ([], {}, "non-empty"),
        ([[1, 2]], {}, "1-D"),
        ([1, -1], {}, "non-negative"),
        ([1, 1.5], {}, "integers"),
        (["a"], {}, "integers"),
        ([1, 4], {"n_clusters": 3}, "at most n_clusters"),
        ([0, 1], {}, "1-based"),
        ([0, 2], {"n_clusters": 3}, "1-based"),
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


@pytest.mark.parametrize("n_clusters", [0, -2, 2.5, 3.0, True, "3"])
def test_distribution_rejects_bad_n_clusters(n_clusters):
    with pytest.raises(ValueError, match="n_clusters"):
        vp.shape_mode_distribution([1], n_clusters=n_clusters)


@pytest.mark.parametrize("base", [1, 1.0, 0, -2, np.inf, np.nan, True, "2"])
def test_entropy_rejects_bad_base(base):
    with pytest.raises(ValueError, match="base"):
        vp.shannon_entropy([0.5, 0.5], base=base)


def test_entropy_accepts_fractional_base():
    assert vp.shannon_entropy([0.5, 0.5], base=0.5) == pytest.approx(-1.0)
