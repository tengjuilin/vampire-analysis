import numpy as np
import pytest

from vampire import util, model


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


def test_generate_file_paths(img_set_path,
                             real_img_set_path,
                             empty_filter,
                             cortex_40x_filter,
                             midbrain_40x_filter,
                             cortex_40x_2_filter,
                             cortex_yen_or_otsu_filter,
                             cortex_or_hypothalamus_otsu_filter):
    actual = util.generate_file_paths(img_set_path, 'filename', empty_filter, '.png')
    expected = r'data/img/filename__.png'
    assert actual == expected

    actual = util.generate_file_paths(img_set_path, 'file', cortex_40x_filter, '.tif')
    expected = r'data/img/file__cortex_40x.tif'
    assert actual == expected

    actual = util.generate_file_paths(img_set_path, 'file_name', midbrain_40x_filter, '.tiff')
    expected = r'data/img/file_name__midbrain_40x.tiff'
    assert actual == expected

    actual = util.generate_file_paths(real_img_set_path, 'img_name', cortex_40x_2_filter, '.jpg')
    expected = r'data/real_img/img_name__cortex_40x_2.jpg'
    assert actual == expected

    actual = util.generate_file_paths(real_img_set_path, 'img', cortex_yen_or_otsu_filter, '.jpeg')
    expected = r'data/real_img/img__cortex_yen-otsu.jpeg'
    assert actual == expected

    actual = util.generate_file_paths(real_img_set_path, 'pick-', cortex_or_hypothalamus_otsu_filter, '.pickle')
    expected = r'data/real_img/pick-__cortex-hypothalamus_otsu.pickle'
    assert actual == expected


def test_get_properties_pickle_path(img_set_path,
                                    real_img_set_path,
                                    empty_filter,
                                    cortex_40x_filter,
                                    midbrain_40x_filter,
                                    cortex_40x_2_filter,
                                    cortex_yen_or_otsu_filter,
                                    cortex_or_hypothalamus_otsu_filter):
    actual = util.get_properties_pickle_path(img_set_path, empty_filter)
    expected = r'data/img/raw-properties__.pickle'
    assert actual == expected

    actual = util.get_properties_pickle_path(img_set_path, cortex_40x_filter)
    expected = r'data/img/raw-properties__cortex_40x.pickle'
    assert actual == expected

    actual = util.get_properties_pickle_path(img_set_path, midbrain_40x_filter)
    expected = r'data/img/raw-properties__midbrain_40x.pickle'
    assert actual == expected

    actual = util.get_properties_pickle_path(real_img_set_path, cortex_40x_2_filter)
    expected = r'data/real_img/raw-properties__cortex_40x_2.pickle'
    assert actual == expected

    actual = util.get_properties_pickle_path(real_img_set_path, cortex_yen_or_otsu_filter)
    expected = r'data/real_img/raw-properties__cortex_yen-otsu.pickle'
    assert actual == expected

    actual = util.get_properties_pickle_path(real_img_set_path, cortex_or_hypothalamus_otsu_filter)
    expected = r'data/real_img/raw-properties__cortex-hypothalamus_otsu.pickle'
    assert actual == expected


def test_get_properties_csv_path(img_set_path,
                                 real_img_set_path,
                                 empty_filter,
                                 cortex_40x_filter,
                                 midbrain_40x_filter,
                                 cortex_40x_2_filter,
                                 cortex_yen_or_otsu_filter,
                                 cortex_or_hypothalamus_otsu_filter):
    actual = util.get_properties_csv_path(img_set_path, empty_filter)
    expected = r'data/img/raw-properties__.csv'
    assert actual == expected

    actual = util.get_properties_csv_path(img_set_path, cortex_40x_filter)
    expected = r'data/img/raw-properties__cortex_40x.csv'
    assert actual == expected

    actual = util.get_properties_csv_path(img_set_path, midbrain_40x_filter)
    expected = r'data/img/raw-properties__midbrain_40x.csv'
    assert actual == expected

    actual = util.get_properties_csv_path(real_img_set_path, cortex_40x_2_filter)
    expected = r'data/real_img/raw-properties__cortex_40x_2.csv'
    assert actual == expected

    actual = util.get_properties_csv_path(real_img_set_path, cortex_yen_or_otsu_filter)
    expected = r'data/real_img/raw-properties__cortex_yen-otsu.csv'
    assert actual == expected

    actual = util.get_properties_csv_path(real_img_set_path, cortex_or_hypothalamus_otsu_filter)
    expected = r'data/real_img/raw-properties__cortex-hypothalamus_otsu.csv'
    assert actual == expected


def test_get_model_pickle_path(img_set_path,
                               real_img_set_path,
                               empty_filter,
                               cortex_40x_filter,
                               midbrain_40x_filter,
                               cortex_40x_2_filter,
                               cortex_yen_or_otsu_filter,
                               cortex_or_hypothalamus_otsu_filter):
    vampire_model = model.Vampire('ogd')
    actual = util.get_model_pickle_path(img_set_path, empty_filter, vampire_model)
    expected = r'data/img/model_ogd_(50_5_20)__.pickle'
    assert actual == expected

    vampire_model = model.Vampire('MEF')
    actual = util.get_model_pickle_path(img_set_path, cortex_40x_filter, vampire_model)
    expected = r'data/img/model_MEF_(50_5_20)__cortex_40x.pickle'
    assert actual == expected

    vampire_model = model.Vampire('hypoxic_ischemic')
    actual = util.get_model_pickle_path(img_set_path, midbrain_40x_filter, vampire_model)
    expected = r'data/img/model_hypoxic_ischemic_(50_5_20)__midbrain_40x.pickle'
    assert actual == expected

    vampire_model = model.Vampire('lipo-mouse')
    actual = util.get_model_pickle_path(real_img_set_path, cortex_40x_2_filter, vampire_model)
    expected = r'data/real_img/model_lipo-mouse_(50_5_20)__cortex_40x_2.pickle'
    assert actual == expected

    vampire_model = model.Vampire('rat-eye')
    actual = util.get_model_pickle_path(real_img_set_path, cortex_yen_or_otsu_filter, vampire_model)
    expected = r'data/real_img/model_rat-eye_(50_5_20)__cortex_yen-otsu.pickle'
    assert actual == expected

    vampire_model = model.Vampire('regional')
    actual = util.get_model_pickle_path(real_img_set_path, cortex_or_hypothalamus_otsu_filter, vampire_model)
    expected = r'data/real_img/model_regional_(50_5_20)__cortex-hypothalamus_otsu.pickle'
    assert actual == expected


def test_get_apply_properties_pickle_path(img_set_path,
                                          real_img_set_path,
                                          empty_filter,
                                          cortex_40x_filter,
                                          midbrain_40x_filter,
                                          cortex_40x_2_filter,
                                          cortex_yen_or_otsu_filter,
                                          cortex_or_hypothalamus_otsu_filter):
    vampire_model = model.Vampire('ogd')
    actual = util.get_apply_properties_pickle_path(img_set_path, empty_filter, vampire_model, 'cortex')
    expected = r'data/img/apply-properties_ogd_on_cortex_(50_5_20)__.pickle'
    assert actual == expected

    vampire_model = model.Vampire('MEF')
    actual = util.get_apply_properties_pickle_path(img_set_path, cortex_40x_filter, vampire_model, 'neg')
    expected = r'data/img/apply-properties_MEF_on_neg_(50_5_20)__cortex_40x.pickle'
    assert actual == expected

    vampire_model = model.Vampire('hypoxic_ischemic')
    actual = util.get_apply_properties_pickle_path(img_set_path, midbrain_40x_filter, vampire_model, 'treated')
    expected = r'data/img/apply-properties_hypoxic_ischemic_on_treated_(50_5_20)__midbrain_40x.pickle'
    assert actual == expected

    vampire_model = model.Vampire('lipo-mouse')
    actual = util.get_apply_properties_pickle_path(real_img_set_path, cortex_40x_2_filter, vampire_model, 'lipo-rat')
    expected = r'data/real_img/apply-properties_lipo-mouse_on_lipo-rat_(50_5_20)__cortex_40x_2.pickle'
    assert actual == expected

    vampire_model = model.Vampire('rat-eye')
    actual = util.get_apply_properties_pickle_path(real_img_set_path, cortex_yen_or_otsu_filter, vampire_model, 'mouse-eye')
    expected = r'data/real_img/apply-properties_rat-eye_on_mouse-eye_(50_5_20)__cortex_yen-otsu.pickle'
    assert actual == expected

    vampire_model = model.Vampire('regional')
    actual = util.get_apply_properties_pickle_path(real_img_set_path, cortex_or_hypothalamus_otsu_filter, vampire_model, 'temporal')
    expected = r'data/real_img/apply-properties_regional_on_temporal_(50_5_20)__cortex-hypothalamus_otsu.pickle'
    assert actual == expected


def test_get_apply_properties_csv_path(img_set_path,
                                       real_img_set_path,
                                       empty_filter,
                                       cortex_40x_filter,
                                       midbrain_40x_filter,
                                       cortex_40x_2_filter,
                                       cortex_yen_or_otsu_filter,
                                       cortex_or_hypothalamus_otsu_filter):
    vampire_model = model.Vampire('ogd')
    actual = util.get_apply_properties_csv_path(img_set_path, empty_filter, vampire_model, 'cortex')
    expected = r'data/img/apply-properties_ogd_on_cortex_(50_5_20)__.csv'
    assert actual == expected

    vampire_model = model.Vampire('MEF')
    actual = util.get_apply_properties_csv_path(img_set_path, cortex_40x_filter, vampire_model, 'neg')
    expected = r'data/img/apply-properties_MEF_on_neg_(50_5_20)__cortex_40x.csv'
    assert actual == expected

    vampire_model = model.Vampire('hypoxic_ischemic')
    actual = util.get_apply_properties_csv_path(img_set_path, midbrain_40x_filter, vampire_model, 'treated')
    expected = r'data/img/apply-properties_hypoxic_ischemic_on_treated_(50_5_20)__midbrain_40x.csv'
    assert actual == expected

    vampire_model = model.Vampire('lipo-mouse')
    actual = util.get_apply_properties_csv_path(real_img_set_path, cortex_40x_2_filter, vampire_model, 'lipo-rat')
    expected = r'data/real_img/apply-properties_lipo-mouse_on_lipo-rat_(50_5_20)__cortex_40x_2.csv'
    assert actual == expected

    vampire_model = model.Vampire('rat-eye')
    actual = util.get_apply_properties_csv_path(real_img_set_path, cortex_yen_or_otsu_filter, vampire_model, 'mouse-eye')
    expected = r'data/real_img/apply-properties_rat-eye_on_mouse-eye_(50_5_20)__cortex_yen-otsu.csv'
    assert actual == expected

    vampire_model = model.Vampire('regional')
    actual = util.get_apply_properties_csv_path(real_img_set_path, cortex_or_hypothalamus_otsu_filter, vampire_model, 'temporal')
    expected = r'data/real_img/apply-properties_regional_on_temporal_(50_5_20)__cortex-hypothalamus_otsu.csv'
    assert actual == expected
