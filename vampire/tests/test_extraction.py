import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from pandas.testing import assert_frame_equal
from skimage import measure

from vampire import extraction
from vampire.tests.testing import assert_list_allclose, read_abs_pickle, get_abs_path


@pytest.fixture
def img1():
    return measure.label((np.load(get_abs_path(r'data/real_img/img1.npy'))))


@pytest.fixture
def img2():
    return measure.label((np.load(get_abs_path(r'data/real_img/img2.npy'))))


@pytest.fixture
def img_set(img1, img2):
    return [img1, img2]


@pytest.fixture
def img_set_path():
    return r'data/img/'


@pytest.fixture
def real_img_set_path():
    return r'data/real_img/'


@pytest.fixture
def empty_filter():
    return np.array([], dtype=str)


@pytest.fixture
def cortex_40x_filter():
    return np.array(['cortex', '40x'])


@pytest.fixture
def midbrain_40x_filter():
    return np.array(['midbrain', '40x'])


@pytest.fixture
def cortex_40x_2_filter():
    return np.array(['cortex', '40x', '2'])


@pytest.fixture
def tif_40x_4_filter():
    return np.array(['tif', '40x', '4'])


@pytest.fixture
def ogd_40x_filter():
    return np.array(['ogd', '40x'])


@pytest.fixture
def cortex_yen_or_otsu_filter():
    return np.array(['cortex', 'yen|otsu'])


@pytest.fixture
def cortex_or_hypothalamus_otsu_filter():
    return np.array(['cortex|hypothalamus', 'otsu'])


@pytest.fixture
def img1_filter():
    return np.array(['img1'])


@pytest.fixture
def img2_filter():
    return np.array(['img2'])


@pytest.fixture
def img_tif_filter():
    return np.array(['img', 'tif'])


@pytest.fixture
def tif_40x_filter():
    return np.array(['tif', '40x'])


def test_check_property_csv_existence(img_set_path,
                                      empty_filter,
                                      cortex_40x_filter,
                                      midbrain_40x_filter,
                                      cortex_40x_2_filter,
                                      cortex_yen_or_otsu_filter,
                                      cortex_or_hypothalamus_otsu_filter,
                                      tif_40x_4_filter,
                                      ogd_40x_filter):
    actual = extraction.check_property_csv_existence(img_set_path, empty_filter)
    expected = True
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, cortex_40x_filter)
    expected = True
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, midbrain_40x_filter)
    expected = False
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, cortex_40x_2_filter)
    expected = False
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, cortex_yen_or_otsu_filter)
    expected = False
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, cortex_or_hypothalamus_otsu_filter)
    expected = False
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, tif_40x_4_filter)
    expected = False
    assert actual == expected

    actual = extraction.check_property_csv_existence(img_set_path, ogd_40x_filter)
    expected = False
    assert actual == expected


def test_get_filtered_filenames(img_set_path,
                                real_img_set_path,
                                empty_filter,
                                cortex_40x_filter,
                                cortex_40x_2_filter,
                                midbrain_40x_filter,
                                cortex_yen_or_otsu_filter,
                                cortex_or_hypothalamus_otsu_filter):
    actual = extraction.get_filtered_filenames(img_set_path, empty_filter)
    expected = np.array(['4-50-1_40x_cortex_1_otsu_thresh_ogd.tif',
                         '4-50-1_40x_hypothalamus_1_otsu_thresh_ogd.tif',
                         '4-50-2_40x_cortex_2_yen_thresh_ogd.tif',
                         '4-50-2_40x_hypothalamus_2_yen_thresh_ogd.tif'])
    assert_equal(actual, expected)

    actual = extraction.get_filtered_filenames(real_img_set_path, empty_filter)
    expected = np.array(['img1.tif',
                         'img2.tif'])
    assert_equal(actual, expected)

    actual = extraction.get_filtered_filenames(img_set_path, cortex_40x_filter)
    expected = np.array(['4-50-1_40x_cortex_1_otsu_thresh_ogd.tif',
                         '4-50-2_40x_cortex_2_yen_thresh_ogd.tif'])
    assert_equal(actual, expected)

    actual = extraction.get_filtered_filenames(img_set_path, cortex_40x_2_filter)
    expected = np.array(['4-50-2_40x_cortex_2_yen_thresh_ogd.tif'])
    assert_equal(actual, expected)

    actual = extraction.get_filtered_filenames(img_set_path, midbrain_40x_filter)
    expected = np.array([])
    assert_equal(actual, expected)

    actual = extraction.get_filtered_filenames(img_set_path, cortex_yen_or_otsu_filter)
    expected = np.array(['4-50-1_40x_cortex_1_otsu_thresh_ogd.tif',
                         '4-50-2_40x_cortex_2_yen_thresh_ogd.tif'])
    assert_equal(actual, expected)

    actual = extraction.get_filtered_filenames(img_set_path, cortex_or_hypothalamus_otsu_filter)
    expected = np.array(['4-50-1_40x_cortex_1_otsu_thresh_ogd.tif',
                         '4-50-1_40x_hypothalamus_1_otsu_thresh_ogd.tif'])
    assert_equal(actual, expected)


def test_get_img_set(img_set_path, empty_filter):
    filenames = extraction.get_filtered_filenames(img_set_path, empty_filter)
    actual = extraction.get_img_set(img_set_path, filenames)

    np.random.seed(1)
    img1 = np.random.randint(0, 255, (100, 100))
    np.random.seed(2)
    img2 = np.random.randint(0, 255, (20, 50))
    np.random.seed(3)
    img3 = np.random.randint(0, 255, (50, 20))
    np.random.seed(4)
    img4 = np.random.randint(0, 255, (50, 70))
    expected = [img1, img2, img3, img4]

    assert_list_allclose(actual, expected)


def test_extract_contour_from_object(img1, img2):
    actual = extraction.extract_contour_from_object(img1)
    expected = read_abs_pickle('data/extraction/extract_contour_from_object_1.pickle')
    assert_allclose(actual, expected)

    actual = extraction.extract_contour_from_object(img2)
    expected = read_abs_pickle('data/extraction/extract_contour_from_object_2.pickle')
    assert_allclose(actual, expected)


def test_extract_properties_from_img(img1, img2):
    actual = extraction.extract_properties_from_img(img1)
    expected = read_abs_pickle('data/extraction/extract_properties_from_img_1.pickle')
    assert_frame_equal(actual, expected)

    actual = extraction.extract_properties_from_img(img2)
    expected = read_abs_pickle('data/extraction/extract_properties_from_img_2.pickle')
    assert_frame_equal(actual, expected)


def test_extract_properties_from_img_set(img_set):
    actual = extraction.extract_properties_from_img_set(img_set)
    expected = read_abs_pickle('data/extraction/extract_properties_from_img_set.pickle')
    assert_frame_equal(actual, expected)

    # argument len inconsistent
    with pytest.raises(ValueError):
        extraction.extract_properties_from_img_set([np.array([1, 2]),
                                                    np.array([3, 4])],
                                                   np.array(['1']))


def test_read_properties(real_img_set_path, img1_filter):
    actual = extraction.read_properties(real_img_set_path, img1_filter)
    actual = actual.drop(['filename', 'image_id'], axis=1)
    expected = read_abs_pickle('data/extraction/extract_properties_from_img_1.pickle')
    assert_frame_equal(actual, expected)


def test_extract_properties(real_img_set_path,
                            img1_filter,
                            img2_filter,
                            img_tif_filter,
                            empty_filter):
    # full property already exist
    actual = extraction.extract_properties(real_img_set_path, empty_filter, write=False)
    expected = read_abs_pickle('data/extraction/extract_properties_img_set.pickle')
    assert_frame_equal(actual, expected)

    # specific property does not already exist
    # but the result is the same as full property
    actual = extraction.extract_properties(real_img_set_path, img_tif_filter, write=False)
    expected = read_abs_pickle('data/extraction/extract_properties_img_set.pickle')
    assert_frame_equal(actual, expected)

    # specific property already exist
    actual = extraction.extract_properties(real_img_set_path, img1_filter, write=False)
    actual = actual.drop(['filename', 'image_id'], axis=1)
    expected = read_abs_pickle('data/extraction/extract_properties_from_img_1.pickle')
    assert_frame_equal(actual, expected)

    # specific property does not exist
    actual = extraction.extract_properties(real_img_set_path, img2_filter, write=False)
    actual = actual.drop(['filename', 'image_id'], axis=1)
    expected = read_abs_pickle('data/extraction/extract_properties_from_img_2.pickle')
    assert_frame_equal(actual, expected)
