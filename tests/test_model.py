import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import vampire as vp


@pytest.fixture(scope="module")
def four_kinds():
    return vp.datasets.make_shapes(
        n_samples=120,
        kinds=("circle", "ellipse", "triangle", "star"),
        random_state=1,
        return_labels=True,
    )


@pytest.fixture(scope="module")
def fitted(four_kinds):
    X, _ = four_kinds
    return vp.Vampire(n_clusters=6, random_state=0).fit(X)


def test_separates_circles_and_ellipses(shapes):
    X, y = shapes
    model = vp.Vampire(n_clusters=2, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) == 1.0


def test_separates_four_kinds(four_kinds):
    X, y = four_kinds
    model = vp.Vampire(n_clusters=4, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) == 1.0


def test_predict_on_training_data_equals_labels(fitted, four_kinds):
    X, _ = four_kinds
    np.testing.assert_array_equal(fitted.predict(X), fitted.labels_)
    np.testing.assert_array_equal(fitted.pipeline_.predict(X), fitted.labels_)


def test_fitted_attributes(fitted, four_kinds):
    X, _ = four_kinds
    k, p = 6, 50
    assert isinstance(fitted.pipeline_, Pipeline)
    assert list(fitted.pipeline_.named_steps) == [
        "resample",
        "register",
        "align",
        "pca",
        "normalize",
        "kmeans",
    ]
    assert fitted.mean_contour_.shape == (2, p)
    assert fitted.mean_cluster_contours_.shape == (k, 2, p)
    assert fitted.cluster_centers_.shape == (k, fitted.n_components_)
    assert fitted.explained_variance_ratio_.shape == (fitted.n_components_,)
    assert fitted.labels_.shape == (len(X),)
    assert set(fitted.labels_) == set(range(k))
    assert fitted.linkage_matrix_.shape == (k - 1, 4)
    assert fitted.n_features_in_ == 2 * p
    assert fitted.inertia_ == pytest.approx(-fitted.score(X))


def test_dendrogram_order(fitted, four_kinds):
    X, _ = four_kinds
    k = fitted.n_clusters
    assert fitted.dendrogram_["leaves"] == list(range(k))
    assert dendrogram(fitted.linkage_matrix_, no_plot=True)["leaves"] == list(range(k))
    # The stored linkage is the one computed from the renumbered mean contours.
    flat = fitted.mean_cluster_contours_.reshape(k, -1)
    recomputed = linkage(pdist(flat), "complete")
    np.testing.assert_allclose(fitted.linkage_matrix_, recomputed)
    # Mean cluster contours follow the new numbering.
    normalized = fitted.normalize(X)
    for label in range(k):
        np.testing.assert_allclose(
            fitted.mean_cluster_contours_[label],
            normalized[fitted.labels_ == label].mean(axis=0),
        )


def test_kmeans_centers_and_labels_permuted_consistently(fitted, four_kinds):
    X, _ = four_kinds
    features = fitted.pipeline_[:-1].transform(X)
    centers = fitted.cluster_centers_
    for k in range(fitted.n_clusters):
        member_mean = features[fitted.labels_ == k].mean(axis=0)
        nearest = np.argmin(np.linalg.norm(centers - member_mean, axis=1))
        assert nearest == k  # center k sits among the points labelled k
    np.testing.assert_array_equal(
        fitted.pipeline_.named_steps["kmeans"].predict(features), fitted.labels_
    )


def test_transform_is_distance_to_centers(fitted, four_kinds):
    X, _ = four_kinds
    distances = fitted.transform(X)
    assert distances.shape == (len(X), 6)
    np.testing.assert_array_equal(distances.argmin(axis=1), fitted.labels_)


def test_fit_transform_and_fit_predict(four_kinds):
    X, _ = four_kinds
    model = vp.Vampire(n_clusters=3, random_state=0)
    distances = model.fit_transform(X)
    np.testing.assert_allclose(distances, model.transform(X))
    np.testing.assert_array_equal(clone(model).fit_predict(X), model.labels_)


def test_normalize(fitted, four_kinds):
    X, _ = four_kinds
    normalized = fitted.normalize(X[:5])
    assert normalized.shape == (5, 2, 50)
    np.testing.assert_allclose(
        normalized.reshape(5, -1), fitted.pipeline_[:3].transform(X[:5])
    )


def test_deterministic_with_random_state(four_kinds):
    X, _ = four_kinds
    a = vp.Vampire(n_clusters=3, random_state=42).fit(X)
    b = vp.Vampire(n_clusters=3, random_state=42).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    np.testing.assert_array_equal(a.linkage_matrix_, b.linkage_matrix_)


def test_random_state_none(shapes):
    X, _ = shapes
    model = vp.Vampire(n_clusters=2).fit(X)
    assert model.labels_.shape == (len(X),)
    assert set(model.labels_) <= {0, 1}
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    assert model.dendrogram_["leaves"] == [0, 1]


@pytest.mark.parametrize(
    "wrap",
    [
        lambda X: pd.DataFrame({"contour": X, "area": 1.0}),
        lambda X: pd.Series(X),
        np.stack,
        tuple,
    ],
    ids=["frame", "series", "array", "tuple"],
)
def test_accepts_contour_sets(shapes, wrap):
    X, _ = shapes
    expected = vp.Vampire(n_clusters=2, random_state=0).fit(X).labels_
    model = vp.Vampire(n_clusters=2, random_state=0).fit(wrap(X))
    np.testing.assert_array_equal(model.labels_, expected)
    np.testing.assert_array_equal(model.predict(wrap(X)), expected)


def test_accepts_ragged_contours(shapes):
    X, _ = shapes
    ragged = [c[:, : 60 + i] for i, c in enumerate(X)]
    model = vp.Vampire(n_clusters=2, random_state=0).fit(ragged)
    assert model.predict(ragged).shape == (len(X),)


def test_pickle_round_trip(fitted, four_kinds):
    X, _ = four_kinds
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict(X), fitted.predict(X))
    np.testing.assert_array_equal(restored.mean_contour_, fitted.mean_contour_)


def test_grid_search(shapes):
    X, _ = shapes
    search = GridSearchCV(
        vp.Vampire(n_components=5, random_state=0), {"n_clusters": [2, 3]}, cv=2
    ).fit(X)
    assert search.best_params_["n_clusters"] in (2, 3)
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert search.predict(X).shape == (len(X),)


def test_n_components_int(shapes):
    X, _ = shapes
    model = vp.Vampire(n_clusters=2, n_components=4, random_state=0).fit(X)
    assert model.n_components_ == 4
    assert model.explained_variance_ratio_.shape == (4,)
    assert model.cluster_centers_.shape == (2, 4)


def test_n_components_float(shapes):
    X, _ = shapes
    model = vp.Vampire(n_clusters=2, n_components=0.99, random_state=0).fit(X)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    assert cumulative[-1] >= 0.99
    assert model.n_components_ == 1 or cumulative[-2] < 0.99


def test_n_points(shapes):
    X, _ = shapes
    model = vp.Vampire(n_points=30, n_clusters=2, random_state=0).fit(X)
    assert model.mean_contour_.shape == (2, 30)
    assert model.n_features_in_ == 60


def test_clone_and_params():
    model = vp.Vampire(n_clusters=3, n_components=5, random_state=1)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert cloned.set_params(n_clusters=4).n_clusters == 4
    assert model.n_clusters == 3


@pytest.mark.parametrize("method", ["predict", "transform", "normalize", "score"])
def test_not_fitted(shapes, method):
    X, _ = shapes
    with pytest.raises(NotFittedError):
        getattr(vp.Vampire(), method)(X)


@pytest.mark.parametrize("n_clusters", [0, 1, 2.0])
def test_rejects_bad_n_clusters(shapes, n_clusters):
    X, _ = shapes
    with pytest.raises(ValueError, match="n_clusters"):
        vp.Vampire(n_clusters=n_clusters).fit(X)


def test_rejects_bad_input(shapes):
    X, _ = shapes
    with pytest.raises(ValueError, match="empty"):
        vp.Vampire(n_clusters=2).fit([])
    with pytest.raises(ValueError, match="at least 3"):
        vp.Vampire(n_clusters=2).fit([*X, np.zeros((2, 2))])
    model = vp.Vampire(n_clusters=2, random_state=0).fit(X)
    with pytest.raises(ValueError, match="at least 3"):
        model.predict([np.zeros((2, 2))])


def test_empty_cluster_falls_back_to_mean_contour(shapes):
    # One PC + Normalizer maps every contour to +1 or -1: only 2 distinct points.
    X, _ = shapes
    with pytest.warns(ConvergenceWarning):
        model = vp.Vampire(n_clusters=3, n_components=1, random_state=0).fit(X)
    assert len(set(model.labels_)) == 2
    assert np.all(np.isfinite(model.mean_cluster_contours_))
    empty = (set(range(3)) - set(model.labels_)).pop()
    np.testing.assert_array_equal(
        model.mean_cluster_contours_[empty], model.mean_contour_
    )
    assert model.dendrogram_["leaves"] == [0, 1, 2]


def test_get_feature_names_out(fitted):
    names = fitted.get_feature_names_out()
    assert names.tolist() == [f"vampire{i}" for i in range(6)]
    with pytest.raises(NotFittedError):
        vp.Vampire().get_feature_names_out()


def test_set_output_pandas(four_kinds):
    X, _ = four_kinds
    model = vp.Vampire(n_clusters=3, random_state=0).set_output(transform="pandas")
    distances = model.fit_transform(X)
    assert isinstance(distances, pd.DataFrame)
    assert distances.columns.tolist() == ["vampire0", "vampire1", "vampire2"]
    out = model.transform(X)
    assert isinstance(out, pd.DataFrame)
    np.testing.assert_allclose(out.to_numpy(), distances.to_numpy())
    np.testing.assert_array_equal(out.to_numpy().argmin(axis=1), model.labels_)
    # A property table's index is carried over.
    frame = pd.DataFrame({"contour": list(X)}, index=np.arange(len(X)) + 100)
    assert model.transform(frame).index.tolist() == frame.index.tolist()
    assert isinstance(model.predict(X), np.ndarray)
