import os

from numpy.testing import assert_allclose, assert_equal
from vampire import util


def assert_list_allclose(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_allclose(actual[i], expected[i])


def assert_list_equal(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def get_abs_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)


def read_abs_pickle(rel_path):
    abs_path = get_abs_path(rel_path)
    return util.read_pickle(abs_path)
