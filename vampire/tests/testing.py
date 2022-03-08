from numpy.testing import assert_allclose, assert_equal


def assert_list_allclose(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_allclose(actual[i], expected[i])


def assert_list_equal(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])
