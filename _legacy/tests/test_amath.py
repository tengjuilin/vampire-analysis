import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from vampire import amath


@pytest.mark.parametrize(
    'A',
    [
        np.random.rand(3, 2),
        np.random.rand(10, 3),
        np.random.rand(20, 5),
        np.random.rand(200, 50),
        np.random.rand(2, 3),
        np.random.rand(5, 10),
        np.random.rand(5, 20),
        np.random.rand(50, 200)
    ]
)
def test_mean_center(A):
    expected = A - A.mean(axis=0)
    actual = amath.mean_center(A)
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    'A',
    [
        np.random.rand(3, 2),
        np.random.rand(10, 3),
        np.random.rand(20, 5),
        np.random.rand(200, 50)
    ]
)
def test__pca_eig(A):
    pca = PCA(svd_solver='full')
    pca.fit(A)
    expected = (pca.components_.T,
                pca.transform(A),
                pca.explained_variance_)

    actual = amath._pca_eig(A)

    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_allclose(np.abs(actual[i]),
                        np.abs(expected[i]),
                        atol=1e-7)


@pytest.mark.parametrize(
    'A',
    [
        np.random.rand(2, 3),
        np.random.rand(5, 10),
        np.random.rand(5, 20),
        np.random.rand(50, 200)
    ]
)
def test__pca_svd(A):
    pca = PCA(svd_solver='full')
    pca.fit(A)
    expected = (pca.components_.T,
                pca.transform(A),
                pca.explained_variance_)

    actual = amath._pca_svd(A)

    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_allclose(np.abs(actual[i]),
                        np.abs(expected[i]),
                        atol=1e-7)


@pytest.mark.parametrize(
    'A',
    [
        np.random.rand(3, 2),
        np.random.rand(10, 3),
        np.random.rand(20, 5),
        np.random.rand(200, 50),
        np.random.rand(2, 3),
        np.random.rand(5, 10),
        np.random.rand(5, 20),
        np.random.rand(50, 200)
    ]
)
def test__pca(A):
    pca = PCA(svd_solver='full')
    pca.fit(A)
    expected = (pca.components_.T,
                pca.transform(A),
                pca.explained_variance_)

    actual = amath.pca(A)

    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_allclose(np.abs(actual[i]),
                        np.abs(expected[i]),
                        atol=1e-7)


@pytest.mark.parametrize(
    'A, B',
    [
        (np.random.rand(2, 30),
         np.random.rand(2, 30),),
        (np.random.rand(2, 49),
         np.random.rand(2, 49),),
        (np.random.rand(2, 100),
         np.random.rand(2, 100),),
        (np.random.rand(2, 151),
         np.random.rand(2, 151),)
    ]
)
def test_get_rotation_matrix(A, B):
    A_3d_T = np.c_[A.T, np.zeros((A.shape[1], 1))]
    B_3d_T = np.c_[B.T, np.zeros((B.shape[1], 1))]
    expected_3d, _ = Rotation.align_vectors(A_3d_T, B_3d_T)
    expected = expected_3d.as_matrix()[:2, :2]

    actual = amath.get_rotation_matrix(A, B)

    assert_allclose(np.abs(actual),
                    np.abs(expected),
                    atol=1e-7)
