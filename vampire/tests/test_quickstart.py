from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from vampire import quickstart
from vampire.tests.testing import assert_list_equal, read_abs_pickle


@pytest.fixture
def empty_filter():
    return np.array([], dtype=str)


@pytest.fixture
def img1_filter():
    return np.array(['img1'])


@pytest.fixture
def img2_filter():
    return np.array(['img2'])


@pytest.fixture
def img1_nan_filter():
    return np.array(['img1', 'nan'])


@pytest.fixture
def img2_nan_filter():
    return np.array(['img2', 'nan'])


@pytest.fixture
def img_set_path():
    return 'data\\real_img'


@pytest.fixture
def output_path():
    return 'data\\quickstart\\output'


@pytest.fixture
def model_name():
    return 'quickstart-test'


@pytest.fixture
def num_points():
    return 70


@pytest.fixture
def num_clusters():
    return 10


@pytest.fixture
def random_state():
    return 1


@pytest.fixture
def build_img_info_df(img_set_path, output_path,
                      model_name, num_points, num_clusters):
    img_info = {'img_set_path': img_set_path,
                'output_path': output_path,
                'model_name': model_name,
                'num_points': num_points,
                'num_clusters': num_clusters,
                'filename': 'img'}
    img_info_df = pd.DataFrame(img_info, index=np.arange(1))
    return img_info_df


@pytest.fixture
def build_required_info(build_img_info_df):
    return build_img_info_df.iloc[0, :5]


@pytest.fixture
def model_path():
    return 'data\\quickstart\\build_model.pickle'


@pytest.fixture
def built_model(model_path):
    return read_abs_pickle(model_path)


@pytest.fixture
def apply_img_info_df(img_set_path, model_path,
                      output_path, model_name):
    img_info = {'img_set_path': img_set_path,
                'model_path': model_path,
                'output_path': output_path,
                'img_set_name': model_name,
                'filename': 'img'}
    img_info_df = pd.DataFrame(img_info, index=np.arange(1))
    return img_info_df


@pytest.fixture
def apply_required_info(apply_img_info_df):
    return apply_img_info_df.iloc[0, :4]


@pytest.fixture
def apply_model_df():
    return read_abs_pickle('data\\quickstart\\apply_model.pickle')


def test__check_prohibited_char():
    # filepath
    quickstart._check_prohibited_char(r'normal path\very_normal')
    quickstart._check_prohibited_char(r'path-with/&intere^sting/(charac`ters)/but%/legal~!')
    quickstart._check_prohibited_char(r'totaly-legal=/+[why]not?!/#@.:;/')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'paths,not/good')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'bad*paths/BAD')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'not"allowed/path')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'strictly<not\allowed')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'forbiddened|\path')

    # filename
    quickstart._check_prohibited_char(r'legal-file_name=with(some)characters.txt', input_type='file')
    quickstart._check_prohibited_char(r'ToTaLLY!Ok;file`name[it]looks{like}.tiff', input_type='file')
    quickstart._check_prohibited_char(r'wierd~symb0ls?wel!comed&x^10x2.pdf', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'not\\good_filename', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'bad/filename', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'not_allowed,file', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'prohibited:filename', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'strictly*prohibited.txt', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'no"please', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'<worst-filename>', input_type='file')
    with pytest.raises(ValueError):
        quickstart._check_prohibited_char(r'don\'t|use', input_type='file')


def test__parse_filter_info(empty_filter,
                            img1_filter, img2_filter,
                            img1_nan_filter, img2_nan_filter):
    # empty filter
    actual = quickstart._parse_filter_info(np.array([]))
    expected = empty_filter
    assert_array_equal(actual, expected)

    # regular filter
    actual = quickstart._parse_filter_info(img1_filter)
    expected = img1_filter
    assert_array_equal(actual, expected)

    actual = quickstart._parse_filter_info(img2_filter)
    expected = img2_filter
    assert_array_equal(actual, expected)

    # filter with nan
    actual = quickstart._parse_filter_info(img1_nan_filter)
    expected = img1_filter
    assert_array_equal(actual, expected)

    actual = quickstart._parse_filter_info(img2_nan_filter)
    expected = img2_filter
    assert_array_equal(actual, expected)


def test__build_models_check_df(build_img_info_df):
    quickstart._build_models_check_df(build_img_info_df)
    quickstart._build_models_check_df(build_img_info_df.drop('filename', axis=1))
    with pytest.raises(ValueError):
        quickstart._build_models_check_df(build_img_info_df.drop(['filename', 'num_clusters'], axis=1))


def test__build_models_check_required_info(build_required_info, img_set_path,
                                           output_path, model_name,
                                           num_points, num_clusters):
    actual = quickstart._build_models_check_required_info(build_required_info)
    expected = (img_set_path, output_path,
                model_name, num_points, num_clusters)
    assert_list_equal(actual, expected)

    required_info = pd.Series([img_set_path, None, None, None, None])
    actual = quickstart._build_models_check_required_info(required_info)
    expected = (img_set_path, img_set_path,
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                50, 5)
    assert_list_equal(actual, expected)


def test_build_model(img_set_path, output_path, model_name,
                     num_points, num_clusters,
                     empty_filter, random_state,
                     built_model):
    actual = quickstart.build_model(img_set_path,
                                    output_path,
                                    model_name,
                                    num_points,
                                    num_clusters,
                                    empty_filter,
                                    random_state=random_state,
                                    savefig=False)
    actual_write = read_abs_pickle(r'data/quickstart/output/model_quickstart-test__.pickle')
    expected = built_model
    assert actual == actual_write
    assert actual == expected
    assert actual_write == expected


def test_build_models(build_img_info_df, random_state,
                      built_model):
    quickstart.build_models(build_img_info_df,
                            random_state=random_state,
                            savefig=False)
    actual_write = read_abs_pickle(r'data/quickstart/output/model_quickstart-test__img.pickle')
    expected = built_model
    assert actual_write == expected


def test__apply_models_check_df(apply_img_info_df):
    quickstart._apply_models_check_df(apply_img_info_df)
    quickstart._apply_models_check_df(apply_img_info_df.drop('filename', axis=1))
    with pytest.raises(ValueError):
        quickstart._apply_models_check_df(apply_img_info_df.drop(['filename', 'img_set_name'], axis=1))


def test__apply_models_check_required_info(apply_required_info, img_set_path,
                                           model_path, output_path, model_name):
    actual = quickstart._apply_models_check_required_info(apply_required_info)
    expected = (img_set_path, model_path,
                output_path, model_name)
    assert_list_equal(actual, expected)

    required_info = pd.Series([img_set_path, model_path, None, None])
    actual = quickstart._apply_models_check_required_info(required_info)
    expected = (img_set_path, model_path, img_set_path,
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    assert_list_equal(actual, expected)


def test_apply_model(img_set_path, model_path, output_path,
                     model_name, empty_filter, apply_model_df):
    actual = quickstart.apply_model(img_set_path,
                                    model_path,
                                    output_path,
                                    model_name,
                                    empty_filter,
                                    write_csv=False,
                                    savefig=False)
    actual_write = read_abs_pickle(r'data/quickstart/output/apply-properties_quickstart-test_on_quickstart-test__.pickle')
    expected = apply_model_df
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual, actual_write)
    assert_frame_equal(actual_write, expected)


def test_apply_models(apply_img_info_df, apply_model_df):
    quickstart.apply_models(apply_img_info_df,
                            write_csv=False,
                            savefig=False)
    actual_write = read_abs_pickle(r'data/quickstart/output/apply-properties_quickstart-test_on_quickstart-test__img.pickle')
    expected = apply_model_df
    assert_frame_equal(actual_write, expected)
