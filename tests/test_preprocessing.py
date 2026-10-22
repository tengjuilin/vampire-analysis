import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline

import vampire as vp
from vampire.contours import align_contour, register_contour, resample_contour


@pytest.fixture
def contours():
    """Ragged raw contours."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(6):
        out += vp.datasets.make_shapes(
            n_samples=1, n_points=int(rng.integers(60, 120)), random_state=i
        )
    return out


@pytest.fixture
def registered(contours):
    return vp.ContourRegistrar().transform(vp.ContourResampler(30).transform(contours))


def as_frame(c):
    return pd.DataFrame({"contour": list(c)})


def as_series(c):
    return pd.Series(list(c))


ALL = [vp.ContourResampler(), vp.ContourRegistrar(), vp.ContourAligner()]


# --- ContourResampler ------------------------------------------------------


@pytest.mark.parametrize("wrap", [list, as_frame, as_series])
def test_resampler_accepts_contour_sets(contours, wrap):
    out = vp.ContourResampler(n_points=30).fit_transform(wrap(contours))
    assert out.shape == (6, 2, 30)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out[2], resample_contour(contours[2], 30))


def test_resampler_accepts_dense_array():
    X = np.stack(vp.datasets.make_shapes(n_samples=3, random_state=0))
    assert vp.ContourResampler(20).fit_transform(X).shape == (3, 2, 20)


def test_resampler_is_stateless(contours):
    resampler = vp.ContourResampler(n_points=25)
    fitted = clone(resampler).fit(contours)
    np.testing.assert_array_equal(
        resampler.transform(contours), fitted.transform(contours)
    )


@pytest.mark.parametrize("n_points", [0, 2, 10.0, True])
def test_resampler_rejects_bad_n_points(contours, n_points):
    with pytest.raises(ValueError, match="n_points"):
        vp.ContourResampler(n_points=n_points).fit(contours)
    with pytest.raises(ValueError, match="n_points"):
        vp.ContourResampler(n_points=n_points).transform(contours)


def test_resampler_rejects_bad_input(contours):
    with pytest.raises(ValueError, match="empty"):
        vp.ContourResampler().fit_transform([])
    with pytest.raises(ValueError, match="at least 3"):
        vp.ContourResampler().fit_transform([*contours, np.zeros((2, 2))])


# --- ContourRegistrar ------------------------------------------------------


def test_registrar_registers_each_contour(contours):
    resampled = vp.ContourResampler(30).transform(contours)
    out = vp.ContourRegistrar().fit_transform(resampled)
    assert out.shape == (6, 2, 30)
    np.testing.assert_array_equal(out[4], register_contour(resampled[4]))


def flattened(c):
    return c.reshape(len(c), -1)


@pytest.mark.parametrize("wrap", [list, as_frame, as_series, flattened])
def test_registrar_accepts_contour_sets(contours, wrap):
    resampled = vp.ContourResampler(30).transform(contours)
    expected = vp.ContourRegistrar().transform(resampled)
    np.testing.assert_array_equal(
        vp.ContourRegistrar().transform(wrap(resampled)), expected
    )


def test_registrar_rejects_ragged(contours):
    with pytest.raises(ValueError, match="different numbers of points"):
        vp.ContourRegistrar().fit(contours)


# --- ContourAligner --------------------------------------------------------


def test_aligner_single_pass_uses_registered_mean(registered):
    aligner = vp.ContourAligner(flatten=False, max_iter=1).fit(registered)
    assert aligner.n_iter_ == 1
    np.testing.assert_allclose(aligner.reference_contour_, registered.mean(axis=0))


def _mean_ssd(aligned):
    return np.mean(np.sum((aligned - aligned.mean(axis=0)) ** 2, axis=(1, 2)))


@pytest.fixture
def triangles():
    X = vp.datasets.make_shapes(80, kinds=("triangle",), noise=0.02, random_state=0)
    return vp.ContourRegistrar().transform(vp.ContourResampler(40).transform(X))


def test_aligner_iterates_to_a_fixed_point(triangles):
    aligner = vp.ContourAligner(flatten=False)
    aligned = aligner.fit_transform(triangles)
    assert 1 < aligner.n_iter_ < aligner.max_iter
    reference = aligner.reference_contour_
    # Fixed point: the mean of the aligned contours is the reference.
    np.testing.assert_allclose(aligned.mean(axis=0), reference, atol=1e-5)
    np.testing.assert_allclose(aligner.mean_contour_, aligned.mean(axis=0))
    # Training contours are aligned to the final reference, like new ones.
    np.testing.assert_array_equal(aligned, aligner.transform(triangles))


def test_aligner_iterations_reduce_spread(triangles):
    once = vp.ContourAligner(flatten=False, max_iter=1).fit_transform(triangles)
    iterated = vp.ContourAligner(flatten=False).fit_transform(triangles)
    assert _mean_ssd(iterated) < 0.9 * _mean_ssd(once)


def test_aligner_reference_does_not_drift(triangles):
    aligner = vp.ContourAligner(flatten=False).fit(triangles)
    reference = aligner.reference_contour_
    # The reference stays in the frame of the registered mean: aligning it
    # to that mean (rotation and shift) barely moves it (< 0.1 degree here).
    np.testing.assert_allclose(
        align_contour(reference, triangles.mean(axis=0)), reference, atol=1e-2
    )


@pytest.mark.parametrize(
    "params", [{"max_iter": 0}, {"max_iter": 2.0}, {"max_iter": True}, {"tol": -1}]
)
def test_aligner_rejects_bad_params(registered, params):
    with pytest.raises(ValueError, match=r"max_iter|tol"):
        vp.ContourAligner(**params).fit(registered)


def test_aligner_learns_reference_and_mean(registered):
    aligner = vp.ContourAligner(flatten=False).fit(registered)
    assert aligner.n_points_ == 30
    aligned = aligner.transform(registered)
    assert aligned.shape == (6, 2, 30)
    np.testing.assert_allclose(aligner.mean_contour_, aligned.mean(axis=0))
    np.testing.assert_array_equal(
        aligned[1], align_contour(registered[1], aligner.reference_contour_)
    )


def test_aligner_flatten_is_x_then_y(registered):
    flat = vp.ContourAligner(flatten=True).fit_transform(registered)
    dense = vp.ContourAligner(flatten=False).fit_transform(registered)
    assert flat.shape == (6, 60)
    np.testing.assert_array_equal(flat[:, :30], dense[:, 0])
    np.testing.assert_array_equal(flat, dense.reshape(6, -1))


@pytest.mark.parametrize("flatten", [True, False])
def test_aligner_fit_transform_equals_fit_then_transform(registered, flatten):
    aligner = vp.ContourAligner(flatten=flatten)
    np.testing.assert_array_equal(
        aligner.fit_transform(registered),
        clone(aligner).fit(registered).transform(registered),
    )


def test_aligner_accepts_flattened_and_frame(registered):
    aligner = vp.ContourAligner().fit(registered)
    expected = aligner.transform(registered)
    np.testing.assert_array_equal(aligner.transform(flattened(registered)), expected)
    np.testing.assert_array_equal(
        aligner.transform(as_frame(list(registered))), expected
    )


def test_aligner_requires_fit(registered):
    with pytest.raises(NotFittedError):
        vp.ContourAligner().transform(registered)


def test_aligner_rejects_wrong_n_points(registered, contours):
    aligner = vp.ContourAligner().fit(registered)
    other = vp.ContourResampler(20).transform(contours)
    with pytest.raises(ValueError, match="fitted with 30"):
        aligner.transform(other)


# --- sklearn API -----------------------------------------------------------


@pytest.mark.parametrize("estimator", ALL, ids=lambda e: type(e).__name__)
def test_clone_and_params(estimator):
    cloned = clone(estimator)
    assert type(cloned) is type(estimator)
    assert cloned.get_params() == estimator.get_params()


def test_set_params():
    assert vp.ContourResampler().set_params(n_points=7).n_points == 7
    assert vp.ContourAligner().set_params(flatten=False).flatten is False


def test_chain_feeds_sklearn_estimators(contours):
    pipeline = make_pipeline(
        vp.ContourResampler(30), vp.ContourRegistrar(), vp.ContourAligner(), PCA(2)
    )
    assert pipeline.fit_transform(contours).shape == (6, 2)
    assert pipeline.transform(contours[:2]).shape == (2, 2)


def test_registrar_rejects_single_2d_contour(contours):
    resampled = vp.ContourResampler(20).transform(contours)
    with pytest.raises(ValueError, match="wrap a single contour in a list"):
        vp.ContourRegistrar().fit_transform(resampled[0])
    with pytest.raises(ValueError, match="wrap a single contour in a list"):
        vp.ContourAligner().fit(resampled[0])
    assert vp.ContourRegistrar().fit_transform([resampled[0]]).shape == (1, 2, 20)


def test_aligner_feature_names(registered):
    aligner = vp.ContourAligner().fit(registered)
    n = aligner.n_points_
    names = aligner.get_feature_names_out().tolist()
    assert names == [f"x{i}" for i in range(n)] + [f"y{i}" for i in range(n)]
    with pytest.raises(ValueError, match="flatten=True"):
        vp.ContourAligner(flatten=False).fit(registered).get_feature_names_out()


def test_pipeline_set_output_pandas(contours):
    pipeline = make_pipeline(
        vp.ContourResampler(30), vp.ContourRegistrar(), vp.ContourAligner(), PCA(3)
    ).set_output(transform="pandas")
    out = pipeline.fit_transform(contours)
    assert isinstance(out, pd.DataFrame)
    assert out.columns.tolist() == ["pca0", "pca1", "pca2"]
    aligned = pipeline[:3].transform(contours)
    assert isinstance(aligned, pd.DataFrame)
    assert aligned.columns[0] == "x0" and aligned.columns[-1] == "y29"


@pytest.mark.parametrize("cls", [vp.ContourResampler, vp.ContourRegistrar])
def test_contour_steps_set_output(cls, contours):
    estimator = cls()
    for transform in (None, "default", "pandas"):
        assert estimator.set_output(transform=transform) is estimator
    resampled = vp.ContourResampler(20).transform(contours)
    assert isinstance(estimator.fit_transform(resampled), np.ndarray)
    with pytest.raises(ValueError, match="3-D ndarray"):
        estimator.set_output(transform="csv")
