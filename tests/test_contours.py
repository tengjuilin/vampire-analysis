import numpy as np
import pytest

from vampire._linalg import kabsch_rotation
from vampire.contours import (
    align_contour,
    register_contour,
    resample_contour,
    signed_area,
)


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def chiral_shape(n=400):
    """Asymmetric, non-mirror-symmetric blob (counterclockwise)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.3 * np.cos(t) + 0.2 * np.sin(2 * t)
    return np.vstack([r * np.cos(t), r * np.sin(t)])


def steps(contour):
    """Lengths of all segments of the closed contour, including last -> first."""
    return np.hypot(*(np.roll(contour, -1, axis=1) - contour))


# --- signed_area -----------------------------------------------------------


def test_signed_area_orientation():
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    circle = np.vstack([np.cos(t), np.sin(t)])
    assert signed_area(circle) == pytest.approx(np.pi, rel=1e-4)
    assert signed_area(circle[:, ::-1]) == pytest.approx(-np.pi, rel=1e-4)


# --- resample_contour ------------------------------------------------------


@pytest.mark.parametrize("n_points", [10, 50, 123])
def test_resample_unevenly_sampled_circle(n_points):
    t = np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, 80))
    circle = np.vstack([np.cos(t), np.sin(t)])
    resampled = resample_contour(circle, n_points)
    assert resampled.shape == (2, n_points)
    assert resampled.dtype == np.float64
    np.testing.assert_allclose(np.hypot(*resampled), 1.0, atol=1e-3)
    gaps = steps(resampled)
    np.testing.assert_allclose(gaps, gaps.mean(), rtol=1e-3)


def test_resample_is_equidistant_on_polygon_and_does_not_repeat_start():
    square = np.array([[0.0, 2, 2, 0], [0, 0, 1, 1]])
    resampled = resample_contour(square, 40)
    gaps = steps(resampled)  # includes the closing segment last -> first
    np.testing.assert_allclose(gaps, gaps.mean(), rtol=1e-3)
    assert gaps.min() > 0  # start point not duplicated at the end


def test_resample_keeps_start_point_and_orientation():
    contour = chiral_shape(100)
    for c in (contour, contour[:, ::-1], np.roll(contour, 13, axis=1)):
        resampled = resample_contour(c, 50)
        np.testing.assert_allclose(resampled[:, 0], c[:, 0], atol=1e-12)
        assert np.sign(signed_area(resampled)) == np.sign(signed_area(c))


def test_resample_ignores_duplicate_points():
    contour = chiral_shape(60)
    duplicated = np.repeat(contour, [3 if i % 7 == 0 else 1 for i in range(60)], axis=1)
    closed = np.hstack([contour, contour[:, :1]])  # repeated closing point
    expected = resample_contour(contour, 50)
    np.testing.assert_allclose(resample_contour(duplicated, 50), expected, atol=1e-10)
    np.testing.assert_allclose(resample_contour(closed, 50), expected, atol=1e-10)


def test_resample_upsamples_few_points():
    triangle = np.array([[0.0, 1, 0], [0, 0, 1]])
    assert resample_contour(triangle, 30).shape == (2, 30)


@pytest.mark.parametrize(
    "contour",
    [
        np.array([[0.0, 1], [0, 0]]),  # too few points
        np.array([[0.0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),  # only 2 distinct points
        np.zeros((2, 10)),  # all points coincide
        np.zeros((3, 10)),
    ],
)
def test_resample_rejects_degenerate_contours(contour):
    with pytest.raises(ValueError):
        resample_contour(contour)


@pytest.mark.parametrize("n_points", [0, 2, 3.5, "50"])
def test_resample_rejects_bad_n_points(n_points):
    with pytest.raises(ValueError, match="n_points"):
        resample_contour(chiral_shape(), n_points)


# --- register_contour ------------------------------------------------------


def test_register_normalizes_position_scale_rotation_and_orientation():
    contour = 30 * rotation(1.1) @ chiral_shape(60) + np.array([[100.0], [-40.0]])
    registered = register_contour(contour[:, ::-1])
    np.testing.assert_allclose(registered.mean(axis=1), 0, atol=1e-12)
    assert np.sqrt(np.mean(np.sum(registered**2, axis=0))) == pytest.approx(1.0)
    covariance = registered @ registered.T
    assert abs(covariance[0, 1]) < 1e-10  # principal axes
    assert covariance[0, 0] >= covariance[1, 1]  # major axis along x
    assert signed_area(registered) > 0  # counterclockwise
    angles = np.abs(np.arctan2(registered[1], registered[0]))
    assert angles[0] == angles.min()  # starts closest to the +x axis
    assert np.sum(registered[0] ** 3) >= 0  # sign convention: skew toward +x


@pytest.mark.parametrize(
    "transform",
    [
        lambda c: c + np.array([[12.0], [-7.0]]),
        lambda c: 4.5 * c,
        lambda c: rotation(2.0) @ c,
        lambda c: rotation(np.pi) @ c,
        lambda c: np.roll(c, 17, axis=1),
        lambda c: c[:, ::-1],
        lambda c: np.roll(3 * rotation(-0.8) @ c[:, ::-1] + 5, 4, axis=1),
    ],
    ids=["translate", "scale", "rotate", "rotate180", "shift_start", "reverse", "all"],
)
def test_register_is_invariant_to_similarity_and_reparameterization(transform):
    contour = chiral_shape(60)
    np.testing.assert_allclose(
        register_contour(transform(contour)), register_contour(contour), atol=1e-10
    )


def test_register_does_not_conflate_mirror_images():
    contour = chiral_shape(60)
    mirrored = np.diag([1.0, -1.0]) @ contour
    reference = register_contour(contour)
    aligned = align_contour(register_contour(mirrored), reference)
    assert np.abs(aligned - reference).max() > 0.05


def test_register_rejects_degenerate_contour():
    with pytest.raises(ValueError, match="degenerate"):
        register_contour(np.ones((2, 10)))


# --- align_contour ---------------------------------------------------------


def test_align_recovers_shift_and_rotation():
    reference = register_contour(resample_contour(chiral_shape(), 50))
    moved = np.roll(rotation(2.2) @ reference, 21, axis=1)
    np.testing.assert_allclose(align_contour(moved, reference), reference, atol=1e-12)


def test_align_matches_brute_force_search():
    rng = np.random.default_rng(0)
    reference = register_contour(resample_contour(chiral_shape(), 30))
    contour = register_contour(reference + 0.1 * rng.standard_normal(reference.shape))
    aligned = align_contour(contour, reference)

    def best_for_shift(k):
        shifted = np.roll(contour, -k, axis=1)
        return kabsch_rotation(shifted, reference) @ shifted

    best = min(
        (best_for_shift(k) for k in range(30)),
        key=lambda a: np.linalg.norm(a - reference),
    )
    np.testing.assert_allclose(aligned, best, atol=1e-12)


def test_align_uses_proper_rotation_of_cyclic_shift():
    reference = register_contour(resample_contour(chiral_shape(), 40))
    contour = register_contour(np.diag([1.0, -1.0]) @ reference)
    aligned = align_contour(contour, reference)
    assert signed_area(aligned) == pytest.approx(signed_area(contour))
    # Same point set, same cyclic order: some shift maps it to `contour` by a rotation.
    gram = lambda c: c.T @ c  # noqa: E731
    assert any(
        np.allclose(gram(np.roll(aligned, k, axis=1)), gram(contour)) for k in range(40)
    )


def test_align_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same shape"):
        align_contour(np.ones((2, 10)), np.ones((2, 12)))


# --- full chain ------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        lambda c: np.roll(c, 37, axis=1),
        lambda c: c[:, ::-1],
        lambda c: 0.2 * rotation(0.9) @ np.roll(c[:, ::-1], 101, axis=1) - 3,
    ],
    ids=["shift_start", "reverse", "all"],
)
def test_chain_is_invariant_up_to_sampling(transform):
    contour = chiral_shape(400)

    def normalize(c, reference=None):
        registered = register_contour(resample_contour(c, 50))
        return registered if reference is None else align_contour(registered, reference)

    reference = normalize(contour)
    # Re-sampling a shifted start point moves samples by < 1/2 spacing (~0.06).
    np.testing.assert_allclose(
        normalize(transform(contour), reference), reference, atol=0.05
    )
