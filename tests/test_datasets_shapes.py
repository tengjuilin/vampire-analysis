import numpy as np
import pytest

import vampire as vp
from vampire.contours import register_contour, signed_area

ALL_KINDS = ("circle", "ellipse", "triangle", "star")


def test_shapes_and_labels():
    contours, labels = vp.datasets.make_shapes(
        n_samples=11, kinds=ALL_KINDS, n_points=64, random_state=0, return_labels=True
    )
    assert len(contours) == 11
    assert all(c.shape == (2, 64) and c.dtype == np.float64 for c in contours)
    assert labels.shape == (11,)
    assert np.issubdtype(labels.dtype, np.integer)
    np.testing.assert_array_equal(np.bincount(labels), [3, 3, 3, 2])  # balanced


def test_default_returns_list_only():
    contours = vp.datasets.make_shapes(n_samples=3, random_state=0)
    assert isinstance(contours, list)
    assert len(contours) == 3


def test_deterministic_and_accepts_generator():
    a = vp.datasets.make_shapes(n_samples=5, random_state=7)
    b = vp.datasets.make_shapes(n_samples=5, random_state=7)
    c = vp.datasets.make_shapes(n_samples=5, random_state=np.random.default_rng(7))
    for x, y, z in zip(a, b, c, strict=True):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, z)


def test_kinds_string():
    _, labels = vp.datasets.make_shapes(
        n_samples=4, kinds="star", random_state=0, return_labels=True
    )
    np.testing.assert_array_equal(labels, 0)


def test_random_pose_and_orientation():
    contours = vp.datasets.make_shapes(n_samples=40, random_state=0)
    centers = np.array([c.mean(axis=1) for c in contours])
    sizes = np.array([np.ptp(c[0]) for c in contours])
    orientations = np.sign([signed_area(c) for c in contours])
    assert np.ptp(centers, axis=0).min() > 100  # random translation
    assert sizes.max() / sizes.min() > 1.5  # random scale
    assert set(orientations) == {-1.0, 1.0}  # both orientations occur


def test_ellipses_are_elongated_and_circles_round():
    contours, labels = vp.datasets.make_shapes(
        n_samples=40, noise=0.0, random_state=0, return_labels=True
    )
    for contour, label in zip(contours, labels, strict=True):
        registered = register_contour(contour)
        aspect = np.ptp(registered[0]) / np.ptp(registered[1])
        if label == 0:
            np.testing.assert_allclose(np.hypot(*registered), 1.0, atol=1e-12)
        else:
            assert 2.5 - 1e-9 <= aspect <= 3.0 + 1e-9


def test_noise_perturbs_radius():
    contours = vp.datasets.make_shapes(
        n_samples=1, kinds=("circle",), noise=0.05, random_state=0
    )
    radius = np.hypot(*register_contour(contours[0]))
    assert 0.02 < radius.std() < 0.08


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kinds": ()},
        {"kinds": ("square",)},
        {"n_samples": 0},
        {"n_points": 3},
        {"noise": -0.1},
    ],
)
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        vp.datasets.make_shapes(**kwargs)
