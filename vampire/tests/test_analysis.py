import numpy as np
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from vampire import analysis
from vampire.tests.testing import assert_list_allclose, read_abs_pickle


@pytest.fixture
def aligned_contours_flat():
    return read_abs_pickle('data/processing/align_contours.pickle')


@pytest.fixture
def mean_aligned_contour_flat():
    return read_abs_pickle('data/processing/get_mean_aligned_contour.pickle')


@pytest.fixture
def pca_contour():
    return read_abs_pickle('data/analysis/pca_contours.pickle')


@pytest.fixture
def principal_components():
    return read_abs_pickle('data/analysis/pca_transform_contours.pickle')


@pytest.fixture
def cluster_contours_info():
    return read_abs_pickle('data/analysis/cluster_contours.pickle')


@pytest.fixture
def cluster_id_df(cluster_contours_info):
    return cluster_contours_info[0]


@pytest.fixture
def centroids(cluster_contours_info):
    return cluster_contours_info[1]


@pytest.fixture
def labeled_contours_df(cluster_contours_info):
    return read_abs_pickle('data/analysis/labeled_contours_df.pickle')


@pytest.fixture
def hierarchical_cluster_contour(cluster_contours_info):
    return read_abs_pickle('data/analysis/hierarchical_cluster_contour.pickle')


@pytest.fixture
def branches(hierarchical_cluster_contour):
    return hierarchical_cluster_contour[2]


@pytest.fixture
def object_index():
    return np.array([0, 1, 2, 3, 4])


@pytest.fixture
def applied_contours_df():
    return read_abs_pickle('data/analysis/assign_clusters_id.pickle')


def test_pca_contours(aligned_contours_flat, pca_contour):
    actual = analysis.pca_contours(aligned_contours_flat)
    expected = pca_contour
    assert_list_allclose(actual, expected)


def test_pca_transform_contours(aligned_contours_flat,
                                mean_aligned_contour_flat,
                                pca_contour,
                                principal_components):
    principal_directions = pca_contour[0]
    actual = analysis.pca_transform_contours(aligned_contours_flat,
                                             mean_aligned_contour_flat,
                                             principal_directions)
    expected = principal_components
    assert_allclose(actual, expected)


def test_cluster_contours(principal_components,
                          cluster_contours_info):
    actual = analysis.cluster_contours(principal_components,
                                       random_state=1)
    expected = cluster_contours_info
    assert_list_allclose(actual, expected)


def test_assign_clusters_id(principal_components,
                            aligned_contours_flat,
                            centroids,
                            applied_contours_df):
    actual = analysis.assign_clusters_id(principal_components,
                                         aligned_contours_flat,
                                         centroids)
    expected = applied_contours_df
    assert_frame_equal(actual, expected)


def test_get_labeled_contours_df(aligned_contours_flat,
                                 cluster_id_df,
                                 labeled_contours_df):
    actual = analysis.get_labeled_contours_df(aligned_contours_flat,
                                              cluster_id_df)
    expected = labeled_contours_df
    assert_frame_equal(actual, expected)


def test_get_mean_cluster_contours(labeled_contours_df):
    actual = analysis.get_mean_cluster_contours(labeled_contours_df)
    expected = read_abs_pickle('data/analysis/mean_cluster_contours.pickle')
    assert_allclose(actual, expected)


def test_hierarchical_cluster_contour(labeled_contours_df,
                                      hierarchical_cluster_contour):
    actual = analysis.hierarchical_cluster_contour(labeled_contours_df)
    expected = hierarchical_cluster_contour
    assert_list_allclose(actual[:2], expected[:2])


def test_get_cluster_order(branches, object_index):
    actual = analysis.get_cluster_order(branches)
    expected = object_index
    assert_allclose(actual, expected)


def test_get_distribution(applied_contours_df):
    actual = analysis.get_distribution(applied_contours_df)
    expected = np.array([0.08823529, 0.21008403, 0.17647059,
                         0.25630252, 0.26890756])
    assert_allclose(actual, expected)


def test_reorder_clusters(cluster_id_df, object_index):
    actual = analysis.reorder_clusters(cluster_id_df['cluster_id'],
                                       object_index)
    expected = read_abs_pickle('data/analysis/reordered_cluster.pickle')
    assert_allclose(actual, expected)


def test_reorder_centroids(centroids, object_index):
    actual = analysis.reorder_centroids(centroids,
                                        object_index)
    expected = read_abs_pickle('data/analysis/reordered_centroids.pickle')
    assert_allclose(actual, expected)
