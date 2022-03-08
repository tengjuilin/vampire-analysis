import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage import measure

from vampire import coloring
from vampire import util
from vampire.tests.testing import assert_list_allclose


@pytest.fixture
def img1():
    return measure.label((np.load(r'data/real_img/img1.npy')))


@pytest.fixture
def img2():
    return measure.label((np.load(r'data/real_img/img2.npy')))


@pytest.fixture
def img_set(img1, img2):
    return [img1, img2]


@pytest.fixture
def apply_properties_df():
    return util.read_pickle(r'data/model/Vampire_apply.pickle')


@pytest.fixture
def img1_df(apply_properties_df):
    return apply_properties_df[apply_properties_df['image_id'] == 0]


@pytest.fixture
def img2_df(apply_properties_df):
    return apply_properties_df[apply_properties_df['image_id'] == 1].reset_index(drop=True)


@pytest.fixture
def labeled_img1():
    return util.read_pickle(r'data/coloring/label_img_1.pickle')


@pytest.fixture
def labeled_img2():
    return util.read_pickle(r'data/coloring/label_img_2.pickle')


@pytest.fixture
def labeled_imgs():
    return util.read_pickle(r'data/coloring/label_imgs.pickle')


def test_label_img(img1, img2, img1_df, img2_df, labeled_img1, labeled_img2):
    actual = coloring.label_img(img1, img1_df)
    expected = labeled_img1
    assert_allclose(actual, expected)

    actual = coloring.label_img(img2, img2_df)
    expected = labeled_img2
    assert_allclose(actual, expected)


def test_label_imgs(img_set, apply_properties_df, labeled_imgs):
    actual = coloring.label_imgs(img_set, apply_properties_df)
    expected = labeled_imgs
    assert_list_allclose(actual, expected)


def test_color_img(labeled_img1, labeled_img2):
    # only test output colors, fig is hard to test
    _, _, actual = coloring.color_img(labeled_img1)
    expected = util.read_pickle('data/coloring/color_img_1.pickle')
    assert_allclose(actual, expected)

    _, _, actual = coloring.color_img(labeled_img2)
    expected = util.read_pickle('data/coloring/color_img_2.pickle')
    assert_allclose(actual, expected)
