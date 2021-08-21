import numpy as np
import pytest
from numpy import testing

from vampire import amath


@pytest.mark.parametrize('A, expected', [
    (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])),
    (np.array([[1, 0], [0, 1]]), np.array([[0.5, -0.5], [-0.5, 0.5]])),
    (np.array([[1.5, -1, 3], [0, 1, 9]]), np.array([[0.75, -1, -3], [-0.75, 1, 3]])),
    (np.array([[7, 4, 3.2, 5], [5, 1.1, 1, 3]]), np.array([[1, 1.45, 1.1, 1], [-1, -1.45, -1.1, -1]])),
    (np.array([[2, 3.4], [-9.5, 142], [10, -2], [0, 3.2]]),
     np.array([[1.375, -33.25], [-10.125, 105.35], [9.375, -38.65], [-0.625, -33.45]])),
])
def test_mean_center(A, expected):
    testing.assert_allclose(amath.mean_center(A), expected)


@pytest.mark.parametrize('A, expected_V, expected_T, expected_d', [
    (np.array([[0, 0], [0, 0]]), np.array([[1, 0], [0, 1]]), np.array([[0, 0], [0, 0]]), np.array([0, 0])),  # 2 x 2 zero
    (np.array([[1, 0], [0, 1]]),  # 2 x 2 identity
     np.array([[-0.707106781187, 0.707106781187], [0.707106781187, 0.707106781187]]),
     np.array([[-0.707106781187, 0], [0.707106781187, 0]]),
     np.array([1, 0])),
    (np.array([[3, -2], [5, 1.2]]),  # 2 x 2
     np.array([[0.52999894, -0.8479983], [0.8479983, 0.52999894]]),
     np.array([[-1.88679623e+00, 0], [1.88679623e+00, 0]]),
     np.array([7.12000000e+00, 0])),
    (np.array([[7.2, -12.5, 11], [32, -2.1, 90]]),  # 2 x 3
     np.array([[0.29717759479,  0.124622862331], [0.124622862331, -0.988027192361], [0.946654435018,  0.090947288794]]),
     np.array([[-4.172589124273e+01,  0], [4.172589124273e+01,  0]]),
     np.array([3.482100000000e+03, 0])),
    (np.array([[2.3, -5.1], [32, -3.1], [6.9, 90]]),  # 3 x 2
     np.array([[-0.11156697501, -0.993756917001], [0.993756917001, -0.11156697501]]),
     np.array([[-30.889016465975, 14.973005175547], [-32.215041789783, -14.764709209399], [63.104058255758,  -0.208295966147]]),
     np.array([2987.031212049375, 221.11545461729])),
])
def test__pca_svd(A, expected_V, expected_T, expected_d):
    V, T, d = amath._pca_svd(A)
    testing.assert_allclose(V, expected_V, atol=1e-7)
    testing.assert_allclose(T, expected_T, atol=1e-7)
    testing.assert_allclose(d, expected_d, atol=1e-7)


@pytest.mark.parametrize('A, expected_V, expected_T, expected_d', [
    (np.array([[0, 0], [0, 0]]), np.array([[0, 1], [1, 0]]), np.array([[0, 0], [0, 0]]), np.array([0, 0])),  # 2 x 2 zero
    (np.array([[1, 0], [0, 1]]),  # 2 x 2 identity
     np.array([[-0.707106781187, -0.707106781187], [0.707106781187, -0.707106781187]]),
     np.array([[-0.707106781187, 0], [0.707106781187, 0]]),
     np.array([1, 0])),
    (np.array([[3, -2], [5, 1.2]]),  # 2 x 2
     np.array([[0.52999894, -0.8479983], [0.8479983, 0.52999894]]),
     np.array([[-1.88679623e+00, 0], [1.88679623e+00, 0]]),
     np.array([7.12000000e+00, 0])),
    (np.array([[7.2, -12.5, 11], [32, -2.1, 90]]),  # 2 x 3
     np.array([[0.29717759479,  0.124622862331], [0.124622862331, -0.988027192361], [0.946654435018,  0.090947288794]]),
     np.array([[-4.172589124273e+01, 0], [4.172589124273e+01, 0]]),
     np.array([3.482100000000e+03, 0])),
    (np.array([[2.3, -5.1], [32, -3.1], [6.9, 90]]),  # 3 x 2
     np.array([[-0.11156697501, -0.993756917001], [0.993756917001, -0.11156697501]]),
     np.array([[-30.889016465975, 14.973005175547], [-32.215041789783, -14.764709209399], [63.104058255758,  -0.208295966147]]),
     np.array([2987.031212049375, 221.11545461729])),
])
def test__pca_eig(A, expected_V, expected_T, expected_d):
    V, T, d = amath._pca_eig(A)
    testing.assert_allclose(V, expected_V, atol=1e-7)
    testing.assert_allclose(T, expected_T, atol=1e-7)
    testing.assert_allclose(d, expected_d, atol=1e-7)


@pytest.mark.parametrize('A, B, expected', [
    (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[1, 0], [0, 1]])),
    (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])),
    (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
    (np.array([[1, 2, 3, 4, 5], [-4, 3, -2, -3, 1]]),
     np.array([[-2, -1, -5, 0, 9], [8, 5, -6, -3, 2]]),
     np.array([[0.776114, 0.63059263], [-0.63059263, 0.776114]])),
])
def test_get_rotation_matrix(A, B, expected):
    testing.assert_allclose(amath.get_rotation_matrix(A, B), expected)
