"""Tests for vampire.plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

import vampire as vp
from vampire import plotting

K = 4
KINDS = ("circle", "ellipse", "triangle", "star")


@pytest.fixture(scope="module")
def fitted():
    X = vp.datasets.make_shapes(80, kinds=KINDS, random_state=0)
    model = vp.Vampire(n_clusters=K, n_components=5, random_state=0).fit(X)
    return model, X


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _leaf_x(ax):
    return np.array(sorted(ax.get_xticks()))


def test_dendrogram(fitted):
    model, _ = fitted
    ax = plotting.plot_dendrogram(model)
    assert isinstance(ax, Axes)
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(i) for i in range(K)]
    np.testing.assert_allclose(_leaf_x(ax), 5 + 10 * np.arange(K))
    assert len(ax.get_yticks()) == 0


def test_dendrogram_given_ax_and_kwargs(fitted):
    model, _ = fitted
    _, ax = plt.subplots()
    out = plotting.plot_dendrogram(model, ax=ax, orientation="bottom")
    assert out is ax


@pytest.mark.parametrize("orientation", ["left", "right"])
def test_dendrogram_sideways_keeps_leaf_labels(fitted, orientation):
    model, _ = fitted
    ax = plotting.plot_dendrogram(model, orientation=orientation)
    assert [t.get_text() for t in ax.get_yticklabels()] == [str(i) for i in range(K)]
    np.testing.assert_allclose(sorted(ax.get_yticks()), 5 + 10 * np.arange(K))
    assert len(ax.get_xticks()) == 0


def test_dendrogram_rejects_bad_orientation(fitted):
    model, _ = fitted
    with pytest.raises(ValueError, match="orientation"):
        plotting.plot_dendrogram(model, orientation="diagonal")


def test_contour_axes_have_inverted_y(fitted):
    model, X = fitted
    assert plotting.plot_shape_modes(model).yaxis_inverted()
    _, ax = plt.subplots()
    ax.invert_yaxis()  # already inverted: stays inverted
    assert plotting.plot_shape_modes(model, ax=ax).yaxis_inverted()
    _, axes = plotting.plot_representatives(model, X, n_per_mode=2)
    assert all(ax.yaxis_inverted() for ax in axes)
    _, axes = plotting.plot_summary(model)
    assert axes[1].yaxis_inverted()
    assert not axes[0].yaxis_inverted()
    bottom, top = axes[1].get_ylim()
    assert bottom == pytest.approx(-top)


def test_shape_modes(fitted):
    model, _ = fitted
    ax = plotting.plot_shape_modes(model)
    assert len(ax.lines) == K
    for i, line in enumerate(ax.lines):
        x, y = line.get_data()
        assert len(x) == model.mean_cluster_contours_.shape[2] + 1
        assert (x[0], y[0]) == (x[-1], y[-1])  # closed
        assert abs(np.mean(x) - (5 + 10 * i)) < 3  # near its leaf position
    np.testing.assert_allclose(_leaf_x(ax), 5 + 10 * np.arange(K))
    assert ax.get_aspect() == 1.0


def test_shape_modes_colors_and_errors(fitted):
    model, _ = fitted
    ax = plotting.plot_shape_modes(model, colors="red")
    assert all(to_rgba(line.get_color()) == (1.0, 0.0, 0.0, 1.0) for line in ax.lines)
    with pytest.raises(ValueError, match="colors"):
        plotting.plot_shape_modes(model, colors=["red", "blue"])
    with pytest.raises(ValueError, match="spacing"):
        plotting.plot_shape_modes(model, spacing=0)


def test_distribution(fitted):
    model, _ = fitted
    ax = plotting.plot_distribution(model.labels_, n_clusters=K)
    assert len(ax.patches) == K
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(sum(heights), 100)
    np.testing.assert_allclose(
        heights, vp.metrics.shape_mode_distribution(model.labels_, K) * 100
    )
    centers = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    np.testing.assert_allclose(centers, 5 + 10 * np.arange(K))


def test_distribution_series_and_empty_modes():
    ax = plotting.plot_distribution(pd.Series([0, 0, 1]), n_clusters=4)
    assert [round(p.get_height()) for p in ax.patches] == [67, 33, 0, 0]


@pytest.mark.parametrize(
    "labels, n", [([], None), ([[0, 1]], None), ([0.5], None), ([-1], None), ([5], 3)]
)
def test_distribution_bad_input(labels, n):
    with pytest.raises(ValueError):
        plotting.plot_distribution(labels, n_clusters=n)


def test_alignment_across_axes(fitted):
    model, _ = fitted
    fig, axes = plotting.plot_summary(model)
    assert isinstance(fig, Figure)
    assert axes.shape == (3,)
    bars = [p.get_x() + p.get_width() / 2 for p in axes[0].patches]
    dend = _leaf_x(axes[2])
    modes = _leaf_x(axes[1])
    np.testing.assert_allclose(bars, dend)
    np.testing.assert_allclose(modes, dend)
    for ax in axes:
        assert ax.get_xlim() == axes[0].get_xlim()
    assert len(axes[1].lines) == K


def test_summary_custom_labels_and_errors(fitted):
    model, _ = fitted
    _, axes = plotting.plot_summary(model, labels=[0, 0, 0, 1], figsize=(5, 4))
    assert [round(p.get_height()) for p in axes[0].patches] == [75, 25, 0, 0]
    with pytest.raises(ValueError, match="height_ratios"):
        plotting.plot_summary(model, height_ratios=(1, 1))
    with pytest.raises(ValueError):
        plotting.plot_summary(model, labels=[0, K])


def test_unfitted_model_raises():
    with pytest.raises(ValueError, match="fitted"):
        plotting.plot_shape_modes(vp.Vampire())


def test_representatives(fitted):
    model, X = fitted
    fig, axes = plotting.plot_representatives(model, X, n_per_mode=3)
    assert isinstance(fig, Figure)
    assert axes.shape == (K,)
    counts = np.bincount(model.predict(X), minlength=K)
    for i, ax in enumerate(axes):
        assert len(ax.lines) == min(3, counts[i]) + 1  # + mean
    _, axes = plotting.plot_representatives(model, X, n_per_mode=2, show_mean=False)
    assert all(len(ax.lines) <= 2 for ax in axes)


def test_representatives_are_closest(fitted):
    model, X = fitted
    _, axes = plotting.plot_representatives(model, X, n_per_mode=1, show_mean=False)
    labels = model.predict(X)
    dist = model.transform(X)
    normalized = model.normalize(X)
    for i, ax in enumerate(axes):
        members = np.flatnonzero(labels == i)
        best = members[np.argmin(dist[members, i])]
        x, y = ax.lines[0].get_data()
        np.testing.assert_allclose(x[:-1], normalized[best, 0])
        np.testing.assert_allclose(y[:-1], normalized[best, 1])


def test_representatives_bad_input(fitted):
    model, X = fitted
    with pytest.raises(ValueError, match="n_per_mode"):
        plotting.plot_representatives(model, X, n_per_mode=0)
    with pytest.raises(ValueError, match="labels"):
        plotting.plot_representatives(model, X, labels=[0, 1])


@pytest.fixture(scope="module")
def image_and_props():
    images, _ = vp.datasets.load_example_images()
    props = vp.extract_properties(images[:1])
    return images[0], props


def test_labeled_image(image_and_props):
    image, props = image_and_props
    modes = np.arange(len(props)) % 3
    ax = plotting.plot_labeled_image(image, props, modes, background_color="white")
    rgba = np.asarray(ax.images[0].get_array())
    assert rgba.shape == (*image.shape, 4)
    labeled = vp.extraction._to_label_image(image)
    assert np.all(rgba[labeled == 0] == [1, 1, 1, 1])
    palette = plotting._mode_colors(3)
    for lab, mode in zip(props["label"], modes, strict=True):
        assert np.all(rgba[labeled == lab] == palette[mode])
    assert len({tuple(c) for c in rgba.reshape(-1, 4)}) == 4  # bg + 3 modes


def test_labeled_image_background_and_ax(image_and_props):
    image, props = image_and_props
    _, ax = plt.subplots()
    out = plotting.plot_labeled_image(
        image, props, np.zeros(len(props), int), ax=ax, background_color="black"
    )
    assert out is ax
    rgba = np.asarray(ax.images[0].get_array())
    labeled = vp.extraction._to_label_image(image)
    assert np.all(rgba[labeled == 0] == [0, 0, 0, 1])


def test_labeled_image_partial_objects(image_and_props):
    image, props = image_and_props
    ax = plotting.plot_labeled_image(image, props.iloc[:1], [1], n_clusters=3)
    rgba = np.asarray(ax.images[0].get_array())
    assert len({tuple(c) for c in rgba.reshape(-1, 4)}) == 2


def test_labeled_image_bad_input(image_and_props):
    image, props = image_and_props
    with pytest.raises(ValueError, match="2-D"):
        plotting.plot_labeled_image(image[None], props, [0] * len(props))
    with pytest.raises(ValueError, match="equal length"):
        plotting.plot_labeled_image(image, props, [0])
    with pytest.raises(ValueError, match="label"):
        plotting.plot_labeled_image(
            image, props.drop(columns="label"), [0] * len(props)
        )
    with pytest.raises(ValueError, match="non-negative"):
        plotting.plot_labeled_image(image, props, [-1] * len(props))
    with pytest.raises(ValueError, match="n_clusters"):
        plotting.plot_labeled_image(image, props, [3] * len(props), n_clusters=3)
    with pytest.raises(ValueError, match="integers"):
        plotting.plot_labeled_image(image, props, [0.5] * len(props))
    with pytest.raises(ValueError, match="Object labels"):
        plotting.plot_labeled_image(image, props["label"] - 1000, [0] * len(props))


def test_labeled_image_accepts_whole_number_float_modes(image_and_props):
    image, props = image_and_props
    modes = np.arange(len(props)) % 3
    as_int = plotting.plot_labeled_image(image, props, modes).images[0].get_array()
    as_float = plotting.plot_labeled_image(image, props, modes.astype(float))
    np.testing.assert_array_equal(as_float.images[0].get_array(), as_int)
    np.testing.assert_array_equal(
        vp.shape_mode_distribution(modes.astype(float)),
        vp.shape_mode_distribution(modes),
    )


def test_labeled_image_binary_keyword():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 7  # single object: auto treats it as a mask (label 1)
    ax = plotting.plot_labeled_image(labels, [7], [0], binary=False)
    rgba = np.asarray(ax.images[0].get_array())
    np.testing.assert_allclose(rgba[2, 2], plotting._mode_colors(1)[0])
    ax = plotting.plot_labeled_image(labels, [1], [0])
    rgba = np.asarray(ax.images[0].get_array())
    np.testing.assert_allclose(rgba[2, 2], plotting._mode_colors(1)[0])


def test_mode_colors():
    assert plotting._mode_colors(4).shape == (4, 4)
    assert plotting._mode_colors(12).shape == (12, 4)
    assert len({tuple(c) for c in plotting._mode_colors(12)}) == 12
    assert plotting._mode_colors(5, "viridis").shape == (5, 4)
    np.testing.assert_allclose(plotting._mode_colors(2, ["r", "b"])[1], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        plotting._mode_colors(0)
    with pytest.raises(ValueError):
        plotting._mode_colors(3, "notacolor")
