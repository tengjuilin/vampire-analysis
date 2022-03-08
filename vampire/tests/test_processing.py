import pytest
from numpy.testing import assert_allclose

from vampire import processing
from vampire import util
from vampire.tests.testing import assert_list_allclose, read_abs_pickle


@pytest.fixture
def properties_df():
    return read_abs_pickle('data/extraction/extract_properties_img_set.pickle')


@pytest.fixture
def raw_contours(properties_df):
    return list(properties_df['raw_contour'])


@pytest.fixture
def raw_contour1():
    return read_abs_pickle('data/extraction/extract_contour_from_object_1.pickle')


@pytest.fixture
def raw_contour2():
    return read_abs_pickle('data/extraction/extract_contour_from_object_2.pickle')


@pytest.fixture
def sampled_contours():
    return read_abs_pickle('data/processing/sample_contours.pickle')


@pytest.fixture
def sampled_contour1():
    return read_abs_pickle('data/processing/sample_contour_1.pickle')


@pytest.fixture
def sampled_contour2():
    return read_abs_pickle('data/processing/sample_contour_2.pickle')


@pytest.fixture
def registered_contours():
    return read_abs_pickle('data/processing/register_contours.pickle')


@pytest.fixture
def registered_contour1():
    return read_abs_pickle('data/processing/register_contour_1.pickle')


@pytest.fixture
def registered_contour2():
    return read_abs_pickle('data/processing/register_contour_2.pickle')


@pytest.fixture
def mean_registered_contour():
    return read_abs_pickle('data/processing/get_mean_registered_contour.pickle')


@pytest.fixture
def aligned_contours_flat():
    return read_abs_pickle('data/processing/align_contours.pickle')


@pytest.fixture
def aligned_contour1():
    return read_abs_pickle('data/processing/align_contour_1.pickle')


@pytest.fixture
def aligned_contour2():
    return read_abs_pickle('data/processing/align_contour_2.pickle')


@pytest.fixture
def mean_aligned_contour_flat():
    return read_abs_pickle('data/processing/get_mean_aligned_contour.pickle')


@pytest.fixture
def num_points():
    return 50


def test_sample_contour(raw_contour1,
                        raw_contour2,
                        sampled_contour1,
                        sampled_contour2,
                        num_points):
    actual = processing.sample_contour(raw_contour1, num_points)
    expected = sampled_contour1
    assert_allclose(actual, expected)

    actual = processing.sample_contour(raw_contour2, num_points)
    expected = sampled_contour2
    assert_allclose(actual, expected)


def test_sample_contours(raw_contours, sampled_contours, num_points):
    actual = processing.sample_contours(raw_contours, num_points)
    expected = sampled_contours
    assert_list_allclose(actual, expected)


def test_register_contour(sampled_contour1,
                          sampled_contour2,
                          registered_contour1,
                          registered_contour2):
    actual = processing.register_contour(sampled_contour1)
    expected = registered_contour1
    assert_allclose(actual, expected)

    actual = processing.register_contour(sampled_contour2)
    expected = registered_contour2
    assert_allclose(actual, expected)


def test_register_contours(sampled_contours, registered_contours):
    actual = processing.register_contours(sampled_contours)
    expected = registered_contours
    assert_list_allclose(actual, expected)


def test_get_mean_registered_contour(registered_contours,
                                     mean_registered_contour):
    actual = processing.get_mean_registered_contour(registered_contours)
    expected = mean_registered_contour
    assert_list_allclose(actual, expected)


def test_align_contour(registered_contour1,
                       registered_contour2,
                       mean_registered_contour,
                       aligned_contour1,
                       aligned_contour2):
    actual = processing.align_contour(registered_contour1,
                                      mean_registered_contour)
    expected = aligned_contour1
    assert_allclose(actual, expected)

    actual = processing.align_contour(registered_contour2,
                                      mean_registered_contour)
    expected = aligned_contour2
    assert_allclose(actual, expected)


def test_align_contours(registered_contours,
                        mean_registered_contour,
                        aligned_contours_flat):
    actual = processing.align_contours(registered_contours,
                                       mean_registered_contour)
    expected = aligned_contours_flat
    assert_list_allclose(actual, expected)


def test_get_get_mean_aligned_contours(aligned_contours_flat,
                                       mean_aligned_contour_flat):
    actual = processing.get_mean_aligned_contour(aligned_contours_flat)
    expected = mean_aligned_contour_flat
    assert_list_allclose(actual, expected)
