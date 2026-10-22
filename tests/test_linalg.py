import numpy as np
import pytest

from vampire._linalg import kabsch_rotation


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 30))
    return A - A.mean(axis=1, keepdims=True)


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, 2.5, np.pi, -1.2])
def test_recovers_known_rotation(points, angle):
    R = kabsch_rotation(points, rotation(angle) @ points)
    np.testing.assert_allclose(R, rotation(angle), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_noisy_target_gives_nearby_rotation(points):
    rng = np.random.default_rng(1)
    B = rotation(0.7) @ points + 0.01 * rng.standard_normal(points.shape)
    np.testing.assert_allclose(kabsch_rotation(points, B), rotation(0.7), atol=0.01)


def test_reflected_target_still_gives_proper_rotation(points):
    reflection = np.diag([1.0, -1.0])
    B = rotation(0.4) @ reflection @ points
    R = kabsch_rotation(points, B)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)
    # R is the best proper rotation: no rotation on a fine grid does better.
    loss = np.linalg.norm(R @ points - B) ** 2
    grid = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    best_grid = min(np.linalg.norm(rotation(a) @ points - B) ** 2 for a in grid)
    assert loss <= best_grid + 1e-9


def test_returns_float_array_for_integer_input():
    A = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
    R = kabsch_rotation(A, A)
    assert R.dtype == np.float64
    np.testing.assert_allclose(R, np.eye(2), atol=1e-12)


@pytest.mark.parametrize(
    ("shape_a", "shape_b"), [((2, 5), (2, 6)), ((3, 5), (3, 5)), ((5,), (5,))]
)
def test_bad_shapes_raise(shape_a, shape_b):
    with pytest.raises(ValueError, match="shape"):
        kabsch_rotation(np.zeros(shape_a), np.zeros(shape_b))
