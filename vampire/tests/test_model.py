import pytest
from pandas.testing import assert_frame_equal

from vampire import model
from vampire.tests.testing import read_abs_pickle


@pytest.fixture
def properties_df():
    return read_abs_pickle('data/extraction/extract_properties_img_set.pickle')


@pytest.fixture
def built_model():
    return read_abs_pickle('data/model/Vampire_build.pickle')


@pytest.fixture
def apply_properties_df():
    return read_abs_pickle('data/model/Vampire_apply.pickle')


def test_Vampire__init__():
    model_name = 'test'
    num_points = 76
    num_coord = num_points * 2
    num_clusters = 7
    num_pc = 38
    random_state = 2
    vampire_model = model.Vampire(model_name,
                                  num_points,
                                  num_clusters,
                                  num_pc,
                                  random_state)
    # model hyperparameters
    assert vampire_model.model_name == model_name
    assert vampire_model.num_points == num_points
    assert vampire_model.num_coord == num_coord
    assert vampire_model.num_clusters == num_clusters
    assert vampire_model.num_pc == num_pc
    assert vampire_model.random_state == random_state
    # contour info
    assert vampire_model.mean_registered_contour is None
    assert vampire_model.mean_aligned_contour is None
    assert vampire_model.contours is None
    # pca analysis info
    assert vampire_model.principal_directions is None
    # k-means clustering info
    assert vampire_model.cluster_id_df is None
    assert vampire_model.labeled_contours_df is None
    assert vampire_model.centroids is None
    assert vampire_model.inertia is None
    assert vampire_model.mean_cluster_contours is None
    # hierarchical clustering info
    assert vampire_model.pair_distance is None
    assert vampire_model.linkage_matrix is None
    assert vampire_model.branches is None


def test_Vampire_build(properties_df, built_model):
    model_name = 'test'
    num_points = 50
    num_clusters = 5
    num_pc = 20
    random_state = 1
    vampire_model = model.Vampire(model_name,
                                  num_points,
                                  num_clusters,
                                  num_pc,
                                  random_state)
    vampire_model.build(properties_df)
    actual = vampire_model
    expected = built_model
    assert actual == expected


def test_Vampire_apply(properties_df, built_model, apply_properties_df):
    vampire_model = built_model
    actual = vampire_model.apply(properties_df)
    expected = apply_properties_df
    assert_frame_equal(actual, expected)


def test_Vampire___eq__(properties_df):
    vampire_model_1_1 = model.Vampire('model_name1',
                                      50, 5, 20, 1)
    vampire_model_1_1.build(properties_df)
    vampire_model_1_2 = model.Vampire('model_name1',
                                      50, 5, 20, 1)
    vampire_model_1_2.build(properties_df)
    vampire_model_2 = model.Vampire('model_name2',
                                    50, 5, 20, 2)
    vampire_model_2.build(properties_df)
    assert vampire_model_1_1 == vampire_model_1_2
    assert vampire_model_1_1 != vampire_model_2
    assert vampire_model_1_2 != vampire_model_2
