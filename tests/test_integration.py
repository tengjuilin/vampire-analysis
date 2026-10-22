"""End-to-end regression tests: images -> properties -> Vampire -> predictions.

Golden values live in ``tests/data/integration_golden.npz`` and are produced
by ``tests/data/make_integration_golden.py`` (see its docstring for when to
regenerate) from ``vampire.datasets.load_example_images()``. The pipeline is
imported from that script so the test and the generator cannot drift apart.

Tolerances
----------
- Object counts, ids, region labels and areas (pixel counts) are compared
  exactly; perimeter and circularity with ``rtol=1e-12`` (skimage's perimeter
  is a weighted sum, so only summation order could change it).
- Model floats (explained variance, mean contours, inertia) use ``rtol=1e-6``
  plus a small ``atol`` for entries near zero: they go through SVD and
  K-means, whose last bits depend on the BLAS/LAPACK build (CI runs Linux,
  macOS and Windows).
- ``labels_`` are compared exactly. This is safe because every object's
  nearest K-means center beats the runner-up by a clear margin and the mean
  shape modes are well separated (``test_fit_is_far_from_ties``), so
  BLAS-level noise cannot flip an assignment or the dendrogram order.
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

import vampire as vp

_spec = importlib.util.spec_from_file_location(
    "make_integration_golden",
    Path(__file__).parent / "data" / "make_integration_golden.py",
)
golden_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden_script)

RTOL = 1e-6
PROP_RTOL = 1e-12


@pytest.fixture(scope="module")
def golden():
    with np.load(golden_script.GOLDEN_PATH) as data:
        return dict(data)


@pytest.fixture(scope="module")
def fitted():
    return golden_script.run_pipeline()


# --- golden regression ------------------------------------------------------


def test_golden_file_is_small():
    assert golden_script.GOLDEN_PATH.stat().st_size < 200 * 1024


def test_properties_match_golden(fitted, golden):
    props, _ = fitted
    assert len(props) == int(golden["n_objects"])
    np.testing.assert_array_equal(props["image_id"], golden["image_id"])
    np.testing.assert_array_equal(props["label"], golden["label"])
    np.testing.assert_array_equal(props["area"], golden["area"])
    for col in ("perimeter", "circularity"):
        np.testing.assert_allclose(props[col], golden[col], rtol=PROP_RTOL, err_msg=col)


def test_model_matches_golden(fitted, golden):
    _, model = fitted
    assert model.n_components_ == int(golden["n_components"])
    np.testing.assert_allclose(
        model.explained_variance_ratio_,
        golden["explained_variance_ratio"],
        rtol=RTOL,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.mean_contour_, golden["mean_contour"], rtol=RTOL, atol=1e-8
    )
    np.testing.assert_allclose(
        model.mean_cluster_contours_,
        golden["mean_cluster_contours"],
        rtol=RTOL,
        atol=1e-8,
    )
    np.testing.assert_allclose(model.inertia_, golden["inertia"], rtol=RTOL)
    np.testing.assert_array_equal(model.labels_, golden["labels"])


def test_fit_is_far_from_ties(fitted):
    """Guard the exact-label comparison: no near-ties to flip across platforms."""
    props, model = fitted
    distances = model.transform(props)
    nearest = np.sort(distances, axis=1)
    assert np.min(nearest[:, 1] - nearest[:, 0]) > 1e-3
    mode_distances = pdist(model.mean_cluster_contours_.reshape(model.n_clusters, -1))
    assert np.min(mode_distances) > 0.1
    assert np.min(np.diff(np.sort(mode_distances))) > 1e-3


# --- end-to-end workflow (no golden values) --------------------------------


def test_fit_on_one_condition_predict_on_another(fitted):
    props, _ = fitted
    train = props[props["filename"] == "img1"]
    new = props[props["filename"] == "img2"]
    assert len(train) > 0
    assert len(new) > 0

    model = vp.Vampire(n_clusters=5, random_state=0).fit(train)
    labels = model.predict(new)

    assert labels.shape == (len(new),)
    assert labels.min() >= 0
    assert labels.max() < model.n_clusters
    np.testing.assert_array_equal(labels, np.argmin(model.transform(new), axis=1))
    np.testing.assert_array_equal(model.predict(train), model.labels_)
    assert np.isfinite(model.score(new))
    distribution = vp.shape_mode_distribution(labels, n_clusters=model.n_clusters)
    assert distribution.shape == (model.n_clusters,)
    assert distribution.sum() == pytest.approx(1.0)
    normalized = model.normalize(new)
    assert normalized.shape == (len(new), 2, model.n_points)


def test_label_by_regex_groupby_distribution(fitted):
    props, model = fitted
    props = props.assign(shape_mode=model.predict(props))
    props = vp.label_by_regex(
        props, {r"img1": {"condition": "A"}, r"img2": {"condition": "B"}}
    )
    assert props["condition"].notna().all()

    distributions = props.groupby("condition")["shape_mode"].apply(
        lambda s: vp.shape_mode_distribution(s, n_clusters=model.n_clusters)
    )
    assert list(distributions.index) == ["A", "B"]
    for dist in distributions:
        assert dist.shape == (model.n_clusters,)
        assert dist.sum() == pytest.approx(1.0)
        assert 0.0 <= vp.shannon_entropy(dist) <= np.log(model.n_clusters)

    # Group fractions, weighted by group size, recover the overall distribution.
    sizes = props["condition"].value_counts().reindex(distributions.index)
    pooled = sum(n * d for n, d in zip(sizes, distributions, strict=True)) / len(props)
    np.testing.assert_allclose(
        pooled, vp.shape_mode_distribution(model.labels_, n_clusters=model.n_clusters)
    )


def test_pickle_round_trip_predicts_identically(fitted):
    props, model = fitted
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(props), model.predict(props))
    np.testing.assert_array_equal(restored.transform(props), model.transform(props))
    np.testing.assert_array_equal(
        restored.mean_cluster_contours_, model.mean_cluster_contours_
    )


def test_refit_is_deterministic(fitted):
    props, model = fitted
    again = vp.Vampire(n_clusters=5, random_state=0).fit(props)
    np.testing.assert_array_equal(again.labels_, model.labels_)
    np.testing.assert_array_equal(
        again.mean_cluster_contours_, model.mean_cluster_contours_
    )
