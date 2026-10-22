import numpy as np
import pandas as pd
import pytest

from vampire._validation import (
    check_contours,
    check_dense_contours,
    check_labels,
    check_n_clusters,
)


@pytest.fixture
def ragged():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((2, n)) for n in (5, 8, 6)]


@pytest.fixture
def dense():
    return np.random.default_rng(0).standard_normal((3, 2, 7))


def as_series(contours):
    return pd.Series(list(contours))


def as_frame(contours):
    return pd.DataFrame({"contour": list(contours), "area": np.arange(len(contours))})


def as_object_array(contours):
    out = np.empty(len(contours), dtype=object)
    out[:] = list(contours)
    return out


@pytest.mark.parametrize("wrap", [list, tuple, as_series, as_frame, as_object_array])
def test_check_contours_accepts_containers(ragged, wrap):
    result = check_contours(wrap(ragged))
    assert isinstance(result, list)
    assert len(result) == 3
    for got, expected in zip(result, ragged, strict=True):
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, expected)


def test_check_contours_accepts_dense_array(dense):
    result = check_contours(dense)
    assert len(result) == 3
    np.testing.assert_array_equal(result[1], dense[1])


def test_check_contours_converts_integers():
    result = check_contours([np.arange(10).reshape(2, 5)])
    assert result[0].dtype == np.float64


@pytest.mark.parametrize(
    "empty",
    [
        [],
        np.empty((0, 2, 10)),
        pd.Series([], dtype=object),
        pd.DataFrame({"contour": []}),
    ],
)
def test_empty_input_raises(empty):
    with pytest.raises(ValueError, match="empty"):
        check_contours(empty)


def test_missing_contour_column_raises():
    with pytest.raises(ValueError, match="'contour' column"):
        check_contours(pd.DataFrame({"area": [1, 2]}))


@pytest.mark.parametrize(
    ("contour", "match"),
    [
        (np.zeros((3, 10)), "shape"),
        (np.zeros(10), "shape"),
        (np.zeros((2, 2)), "at least 3"),
        (np.array([[0.0, 1, np.nan, 0], [0, 0, 1, 1]]), "NaN"),
        (np.array([[0.0, 1, np.inf, 0], [0, 0, 1, 1]]), "NaN"),
        ([["a", "b", "c", "d"], ["e", "f", "g", "h"]], "floats"),
    ],
)
def test_bad_contour_raises(contour, match):
    with pytest.raises(ValueError, match=match):
        check_contours([np.zeros((2, 5)), contour])


@pytest.mark.parametrize("bad", [np.zeros((5, 10)), np.zeros((4, 3, 10)), 5, "abc"])
def test_bad_container_raises(bad):
    with pytest.raises(ValueError):
        check_contours(bad)


def test_check_dense_contours_stacks(ragged, dense):
    result = check_dense_contours(list(dense))
    assert result.shape == (3, 2, 7)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(check_dense_contours(as_frame(dense)), dense)


def test_check_dense_contours_unflattens_x_then_y(dense):
    flat = dense.reshape(3, -1)
    np.testing.assert_array_equal(flat[0, :7], dense[0, 0])  # x first
    np.testing.assert_array_equal(check_dense_contours(flat), dense)


def test_check_dense_contours_rejects_odd_columns():
    with pytest.raises(ValueError, match="2 \\* n_points"):
        check_dense_contours(np.zeros((3, 11)))


def test_check_dense_contours_rejects_ragged(ragged):
    with pytest.raises(ValueError, match="ContourResampler"):
        check_dense_contours(ragged)


def test_check_dense_contours_rejects_single_2d_contour():
    contour = np.random.default_rng(0).standard_normal((2, 50))
    with pytest.raises(ValueError, match="wrap a single contour in a list"):
        check_dense_contours(contour)
    with pytest.raises(ValueError, match="Wrap a single contour in a list"):
        check_contours(contour)
    assert check_dense_contours([contour]).shape == (1, 2, 50)
    assert check_dense_contours(contour[None]).shape == (1, 2, 50)
    # one or three flattened contours are unambiguous
    assert check_dense_contours(contour.reshape(1, -1)).shape == (1, 2, 50)
    assert check_dense_contours(np.zeros((3, 100))).shape == (3, 2, 50)


def test_check_labels():
    values, k = check_labels([0.0, 2.0, 1.0])
    assert values.dtype == np.intp and values.tolist() == [0, 2, 1] and k == 3
    assert check_labels([0], n_clusters=4)[1] == 4
    values, k = check_labels([], allow_empty=True)
    assert values.size == 0 and k == 0
    for bad, match in [
        ([], "non-empty"),
        ([True, False], "integers"),
        ([0.5], "integers"),
        ([np.nan], "integers"),
        ([-1], "non-negative"),
    ]:
        with pytest.raises(ValueError, match=match):
            check_labels(bad)
    with pytest.raises(ValueError, match="below n_clusters"):
        check_labels([3], n_clusters=3)


@pytest.mark.parametrize("bad", [0, -1, 2.5, 2.0, True, "3", None])
def test_check_n_clusters_rejects(bad):
    with pytest.raises(ValueError, match="n_clusters"):
        check_n_clusters(bad)


def test_check_n_clusters_accepts_numpy_int():
    assert check_n_clusters(np.int64(3)) == 3
