"""Matplotlib visualizations of fitted VAMPIRE models.

Every function accepts an optional ``ax`` (or creates figure and axes), returns
the matplotlib ``Axes`` or ``(Figure, ndarray of Axes)`` it drew on, never calls
``plt.show()``, never saves files (use ``fig.savefig``), and does not change
global style.

Shape modes are 1-based (``1..n_clusters``, dendrogram leaf order), matching
:meth:`vampire.Vampire.predict`; every function takes 1-based labels and shows
the mode numbers 1..k. They are drawn with one consistent categorical palette
(``tab10`` for up to 10 modes, ``tab20`` beyond) so mode ``m`` has the same
color (palette entry ``m - 1``) in every figure. Distribution bars, shape-mode
contours, and dendrogram leaves share x positions ``5, 15, 25, ...`` (SciPy's
leaf positions; mode ``m`` at ``10 * m - 5``), so they can be stacked with a
shared x axis, as :func:`plot_summary` does.

Orientation: contours keep the image convention of
:func:`vampire.extract_properties` (x = column, y = row, so y grows
downward). Every axes that draws contours (:func:`plot_shape_modes`,
:func:`plot_representatives`, and the shape-mode axes of
:func:`plot_summary`) therefore has an inverted y axis, so shapes appear as
in the image rather than mirrored top to bottom (which would flip chiral
shapes).
"""

from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import dendrogram
from sklearn.utils.validation import check_is_fitted

from ._validation import ContourSet, check_labels
from .extraction import _to_label_image
from .metrics import shape_mode_distribution

__all__ = [
    "plot_dendrogram",
    "plot_distribution",
    "plot_labeled_image",
    "plot_representatives",
    "plot_shape_modes",
    "plot_summary",
]

#: Distance between adjacent SciPy dendrogram leaves in data units.
_LEAF_SPACING = 10.0

ColorSpec = str | Colormap | Sequence[Any] | None


def _mode_colors(k: int, colors: ColorSpec = None) -> NDArray[np.float64]:
    """Return an array ``(k, 4)`` of RGBA colors for ``k`` shape modes.

    ``None`` gives ``tab10`` (``tab20`` if ``k > 10``, cycled beyond 20). A
    colormap or a registered colormap name is sampled evenly (qualitative
    colormaps are indexed directly). A sequence of ``k`` colors is used as is,
    and a single color is repeated.
    """
    if k < 1:
        raise ValueError(f"n_clusters must be at least 1; got {k}.")
    if colors is None:
        name = "tab10" if k <= 10 else "tab20"
        cmap = mpl.colormaps[name]
        return np.array([cmap(i % cmap.N) for i in range(k)])
    if isinstance(colors, str) and colors in mpl.colormaps:
        colors = mpl.colormaps[colors]
    if isinstance(colors, Colormap):
        if colors.N <= 20:  # qualitative: index directly
            return np.array([colors(i % colors.N) for i in range(k)])
        return np.array(colors(np.linspace(0.05, 0.95, k)))
    if isinstance(colors, str):
        return np.tile(to_rgba(colors), (k, 1))
    try:
        rgba = [to_rgba(c) for c in colors]
    except (ValueError, TypeError):
        try:
            return np.tile(to_rgba(colors), (k, 1))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            raise ValueError(f"Invalid colors specification: {colors!r}.") from None
    if len(rgba) == 1:
        return np.tile(rgba[0], (k, 1))
    if len(rgba) != k:
        raise ValueError(f"colors must contain 1 or {k} colors; got {len(rgba)}.")
    return np.array(rgba)


def _check_model(model: Any) -> None:
    if not hasattr(model, "mean_cluster_contours_"):
        raise ValueError(
            "model must be a fitted vampire.Vampire (missing 'mean_cluster_contours_')."
        )
    check_is_fitted(model)


def _get_ax(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _as_labels(labels: ArrayLike) -> NDArray:
    """Convert labels (array-like or Series) to a 1-D ndarray."""
    return np.asarray(labels)


def _invert_y(ax: Axes) -> None:
    """Make y grow downward (image convention), idempotently."""
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def plot_dendrogram(model: Any, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Plot the dendrogram linking shape modes.

    By default (``orientation="bottom"``) the root is at the bottom and the
    links branch upward to the leaves at the top, so the dendrogram points at
    shape modes drawn above it (as in :func:`plot_summary`).

    Parameters
    ----------
    model : vampire.Vampire
        Fitted model; uses ``model.linkage_matrix_`` of shape
        ``(n_clusters - 1, 4)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    **kwargs
        Passed to :func:`scipy.cluster.hierarchy.dendrogram`. Defaults:
        ``orientation="bottom"`` (root at the bottom, leaves at the top; also
        ``"top"``, ``"left"``, ``"right"``), black links, and leaves labeled
        with the shape-mode numbers ``"1"``, ..., ``"k"``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Shape mode ``m`` sits at ``10 * m - 5`` along the leaf axis (x for
        ``orientation`` ``"top"``/``"bottom"``, y for ``"left"``/``"right"``).
        The distance axis and all spines (axis lines) are hidden.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(
    ...     60, kinds=("circle", "ellipse", "triangle", "star"), random_state=0
    ... )
    >>> model = vp.Vampire(n_clusters=4, n_components=5, random_state=0).fit(X)
    >>> ax = vp.plotting.plot_dendrogram(model)
    >>> [t.get_text() for t in ax.get_xticklabels()]
    ['1', '2', '3', '4']
    >>> bool(ax.yaxis_inverted())  # orientation="bottom": leaves (y=0) on top
    True
    """
    _check_model(model)
    k = len(model.linkage_matrix_) + 1
    options: dict[str, Any] = {
        "labels": [str(m) for m in range(1, k + 1)],
        "link_color_func": lambda _: "black",
        "above_threshold_color": "black",
        "orientation": "bottom",
    }
    options.update(kwargs)
    orientation = options["orientation"]
    if orientation not in ("top", "bottom", "left", "right"):
        raise ValueError(
            "orientation must be 'top', 'bottom', 'left' or 'right'; "
            f"got {orientation!r}."
        )
    ax = _get_ax(ax, (6, 2))
    dendrogram(model.linkage_matrix_, ax=ax, **options)
    leaf_axis = "x" if orientation in ("top", "bottom") else "y"
    for spine in ax.spines.values():
        spine.set_visible(False)
    if leaf_axis == "x":
        ax.set_yticks([])
    else:
        ax.set_xticks([])
    ax.tick_params(axis=leaf_axis, length=0)
    return ax


def plot_shape_modes(
    model: Any,
    ax: Axes | None = None,
    colors: ColorSpec = None,
    spacing: float = _LEAF_SPACING,
    scale: float | None = None,
    linewidth: float = 2.0,
    equal_aspect: bool = True,
) -> Axes:
    """Draw the mean contour of each shape mode side by side.

    Parameters
    ----------
    model : vampire.Vampire
        Fitted model; uses ``model.mean_cluster_contours_`` of shape
        ``(n_clusters, 2, n_points)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    colors : str, Colormap, or sequence, optional
        Colors of the modes: a sequence of ``n_clusters`` colors, one color, or
        a colormap (name). Defaults to the shared categorical palette.
    spacing : float, default=10
        Distance between mode centers. Keep the default for the modes to line
        up with :func:`plot_dendrogram` and :func:`plot_distribution`.
    scale : float, optional
        Multiplier applied to contour coordinates. Defaults to the value that
        makes the widest contour fill 85% of ``spacing``.
    linewidth : float, default=2
        Line width of the contours.
    equal_aspect : bool, default=True
        Use equal data aspect (shapes undistorted). Set ``False`` when the axes
        shares x with other axes; :func:`plot_summary` then adjusts y limits.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Contains one closed line per mode (``ax.lines[m - 1]`` is mode ``m``,
        labeled ``"m"``); x ticks mark the mode centers and are labeled with
        the mode numbers ``1..k``. The
        y axis is inverted (image orientation, see the module docstring).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(
    ...     60, kinds=("circle", "ellipse", "triangle", "star"), random_state=0
    ... )
    >>> model = vp.Vampire(n_clusters=4, n_components=5, random_state=0).fit(X)
    >>> ax = vp.plotting.plot_shape_modes(model)
    >>> len(ax.lines)
    4
    """
    _check_model(model)
    contours = np.asarray(model.mean_cluster_contours_, dtype=float)
    k = len(contours)
    if spacing <= 0:
        raise ValueError(f"spacing must be positive; got {spacing}.")
    palette = _mode_colors(k, colors)
    ax = _get_ax(ax, (6, 1.5))
    if scale is None:
        width = np.ptp(contours[:, 0, :], axis=1).max()
        scale = 0.85 * spacing / width if width > 0 else 1.0
    centers = spacing * (np.argsort(model.dendrogram_["leaves"]) + 0.5)
    for i, contour in enumerate(contours):
        x = np.append(contour[0], contour[0, 0]) * scale + centers[i]
        y = np.append(contour[1], contour[1, 0]) * scale
        ax.plot(x, y, color=palette[i], lw=linewidth, label=str(i + 1))
    ax.set_xlim(0, spacing * k)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks(centers, [str(i + 1) for i in range(k)])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    _invert_y(ax)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return ax


def plot_distribution(
    labels: ArrayLike,
    n_clusters: int | None = None,
    ax: Axes | None = None,
    colors: ColorSpec = None,
) -> Axes:
    """Plot the fraction of objects in each shape mode as a bar chart.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Shape-mode labels in ``1..n_clusters`` (array, list, or pandas
        Series), e.g. from :meth:`vampire.Vampire.predict`.
    n_clusters : int, optional
        Number of shape modes. Defaults to ``max(labels)``; pass
        ``model.n_clusters`` so empty modes still get a bar.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    colors : str, Colormap, or sequence, optional
        As in :func:`plot_shape_modes`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Bar ``m`` (``ax.patches[m - 1]``) is centered at ``x = 10 * m - 5``
        (aligned with the dendrogram leaves), labeled ``"m"``, and has height
        equal to the percentage of objects in shape mode ``m``.

    Raises
    ------
    ValueError
        If ``labels`` is not a non-empty 1-D array of integers in
        ``1..n_clusters`` (0 is rejected: shape modes are 1-based).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import vampire as vp
    >>> ax = vp.plotting.plot_distribution([1, 1, 2, 4], n_clusters=4)
    >>> [round(p.get_height()) for p in ax.patches]
    [50, 25, 0, 25]
    """
    fractions = shape_mode_distribution(_as_labels(labels), n_clusters)
    k = len(fractions)
    palette = _mode_colors(k, colors)
    ax = _get_ax(ax, (6, 3))
    x = _LEAF_SPACING * (np.arange(k) + 0.5)
    ax.bar(x, fractions * 100, width=_LEAF_SPACING / 2, color=palette)
    ax.set_xlim(0, _LEAF_SPACING * k)
    ax.set_xticks(x, [str(i + 1) for i in range(k)])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Objects [%]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_summary(
    model: Any,
    labels: ArrayLike | None = None,
    figsize: tuple[float, float] | None = None,
    height_ratios: Sequence[float] = (4, 1, 1),
) -> tuple[Figure, NDArray[Any]]:
    """Plot shape-mode distribution, mean contours, and dendrogram stacked.

    From top to bottom: a bar chart of the fraction of objects per shape mode
    (x tick labels give the mode numbers ``1..k``, the only place they appear),
    the mean contour of each mode, and the dendrogram with its root at the
    bottom and its leaves pointing up at the shape modes. Bars, modes, and
    leaves are x-aligned.

    Parameters
    ----------
    model : vampire.Vampire
        Fitted model.
    labels : array-like of shape (n_samples,), optional
        Shape-mode labels in ``1..n_clusters`` whose distribution is shown,
        e.g. of one experimental group. Defaults to ``model.labels_`` (the
        training data).
    figsize : (float, float), optional
        Figure size in inches. Defaults to ``(max(4, 0.9 * n_clusters + 1.5), 5)``.
    height_ratios : sequence of 3 floats, default=(4, 1, 1)
        Relative heights of the distribution, shape-mode, and dendrogram axes.
        Use smaller shape-mode heights for many modes.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (3,)
        ``[distribution, shape modes, dendrogram]``, sharing the x axis. Only
        the distribution axes shows x tick labels (``"1"``, ..., ``"k"``). The
        shape-mode axes has an inverted y axis (image orientation).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(
    ...     60, kinds=("circle", "ellipse", "triangle", "star"), random_state=0
    ... )
    >>> model = vp.Vampire(n_clusters=4, n_components=5, random_state=0).fit(X)
    >>> fig, axes = vp.plotting.plot_summary(model)
    >>> axes.shape
    (3,)
    >>> [t.get_text() for t in axes[0].get_xticklabels()]
    ['1', '2', '3', '4']
    """
    _check_model(model)
    if len(height_ratios) != 3:
        raise ValueError("height_ratios must contain 3 values.")
    k = len(model.mean_cluster_contours_)
    labels = model.labels_ if labels is None else labels
    if figsize is None:
        figsize = (max(4.0, 0.9 * k + 1.5), 5.0)
    fig = plt.figure(figsize=figsize)
    top, bottom = 0.97, 0.03
    # A gap between bars and shape modes holds the mode-number tick labels;
    # the dendrogram leaves touch the shape-mode axes.
    gap_in = 2.0 * _font_size_pt("xtick.labelsize") / 72
    ratios = [float(r) for r in height_ratios]
    gap = gap_in / ((top - bottom) * figsize[1] - gap_in) * sum(ratios)
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[ratios[0], gap, ratios[1], ratios[2]],
        hspace=0.0,
        left=0.14,
        right=0.97,
        top=top,
        bottom=bottom,
    )
    ax_dist = fig.add_subplot(grid[0])
    ax_modes = fig.add_subplot(grid[2], sharex=ax_dist)
    ax_dend = fig.add_subplot(grid[3], sharex=ax_dist)
    axes = np.array([ax_dist, ax_modes, ax_dend], dtype=object)
    plot_distribution(labels, n_clusters=k, ax=ax_dist)
    # Largest scale at which every contour fits in both the x slot and the axes
    # height (data aspect is made equal by _equalize_y below).
    bbox = ax_modes.get_position()
    width_in = bbox.width * fig.get_figwidth()
    height_in = bbox.height * fig.get_figheight()
    half_height = 0.5 * _LEAF_SPACING * k * height_in / width_in
    contours = np.asarray(model.mean_cluster_contours_)
    widest = np.ptp(contours[:, 0, :], axis=1).max()
    tallest = np.abs(contours[:, 1, :]).max()
    scale = min(0.85 * _LEAF_SPACING / widest, 0.9 * half_height / tallest)
    plot_shape_modes(model, ax=ax_modes, scale=scale, equal_aspect=False)
    plot_dendrogram(model, ax=ax_dend, orientation="bottom")
    ax_dend.set_xlim(0, _LEAF_SPACING * k)
    ax_dist.tick_params(axis="x", length=0, labelbottom=True)
    for ax in (ax_modes, ax_dend):
        ax.tick_params(axis="x", length=0, labelbottom=False, labeltop=False, top=False)
    _equalize_y(ax_modes)
    return fig, axes


def _font_size_pt(rc_key: str) -> float:
    """Return a font-size rcParam in points (resolving names like ``"medium"``)."""
    return float(FontProperties(size=mpl.rcParams[rc_key]).get_size_in_points())


def _equalize_y(ax: Axes) -> None:
    """Set symmetric, inverted y limits so x and y data units have equal length."""
    fig = ax.get_figure()
    assert fig is not None
    bbox = ax.get_position()
    width_in = bbox.width * fig.get_figwidth()
    height_in = bbox.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    half = 0.5 * (x1 - x0) * height_in / width_in
    ax.set_ylim(half, -half)  # inverted: image orientation


def plot_representatives(
    model: Any,
    X: ContourSet,
    labels: ArrayLike | None = None,
    n_per_mode: int = 5,
    colors: ColorSpec = None,
    show_mean: bool = True,
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Overlay the contours closest to each shape-mode center.

    For each mode, the ``n_per_mode`` objects assigned to it with the smallest
    distance (``model.transform``) to its center are drawn, aligned
    (``model.normalize``), on top of each other.

    Parameters
    ----------
    model : vampire.Vampire
        Fitted model.
    X : contour set
        Contours to pick representatives from (see ``Vampire.fit``).
    labels : array-like of shape (n_samples,), optional
        Shape-mode labels of ``X`` in ``1..n_clusters``. Defaults to
        ``model.predict(X)``.
    n_per_mode : int, default=5
        Maximum number of contours per mode (fewer if the mode has fewer
        objects).
    colors : str, Colormap, or sequence, optional
        As in :func:`plot_shape_modes`.
    show_mean : bool, default=True
        Also draw the mean contour of the mode as a thick dashed black line.
    alpha : float, default=0.7
        Opacity of the representative contours.
    figsize : (float, float), optional
        Defaults to ``(1.6 * n_clusters, 1.8)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (n_clusters,)
        One axes per mode; ``axes[m - 1]`` is titled ``"m"`` and holds
        ``min(n_per_mode, count_m)`` contour lines of shape mode ``m`` (plus
        the mean line if ``show_mean``). Y axes are inverted (image
        orientation).

    Raises
    ------
    ValueError
        If ``n_per_mode < 1``, or ``labels`` does not match ``X`` or is not in
        ``1..n_clusters``.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(
    ...     60, kinds=("circle", "ellipse", "triangle", "star"), random_state=0
    ... )
    >>> model = vp.Vampire(n_clusters=4, n_components=5, random_state=0).fit(X)
    >>> fig, axes = vp.plotting.plot_representatives(model, X, n_per_mode=3)
    >>> axes.shape
    (4,)
    """
    _check_model(model)
    if n_per_mode < 1:
        raise ValueError(f"n_per_mode must be at least 1; got {n_per_mode}.")
    normalized = model.normalize(X)  # (n_samples, 2, n_points), computed once
    features = np.asarray(
        model.pipeline_[3:-1].transform(normalized.reshape(len(normalized), -1))
    )
    kmeans = model.pipeline_[-1]
    distances = np.asarray(kmeans.transform(features))
    k = len(model.mean_cluster_contours_)
    if labels is None:
        labels = kmeans.predict(features) + 1  # 1-based shape modes
    else:
        labels = _as_labels(labels)
        if labels.shape != (len(normalized),):
            raise ValueError(
                f"labels must have shape ({len(normalized)},); got {labels.shape}."
            )
        labels, _ = check_labels(labels, k)
    palette = _mode_colors(k, colors)
    if figsize is None:
        figsize = (1.6 * k, 1.8)
    fig, axes = plt.subplots(1, k, figsize=figsize, squeeze=False, sharex=True)
    axes = axes[0]
    for i, ax in enumerate(axes):
        members = np.flatnonzero(labels == i + 1)
        closest = members[np.argsort(distances[members, i])[:n_per_mode]]
        for idx in closest:
            c = normalized[idx]
            ax.plot(
                np.append(c[0], c[0, 0]),
                np.append(c[1], c[1, 0]),
                color=palette[i],
                alpha=alpha,
                lw=1.5,
            )
        if show_mean:
            m = model.mean_cluster_contours_[i]
            ax.plot(
                np.append(m[0], m[0, 0]),
                np.append(m[1], m[1, 0]),
                color="black",
                lw=2,
                ls="--",
            )
        ax.set_aspect("equal", adjustable="datalim")
        _invert_y(ax)
        ax.set_title(str(i + 1), color=palette[i], fontweight="bold")
        ax.axis("off")
    return fig, axes


def plot_labeled_image(
    image_label: ArrayLike,
    objects: Any,
    labels: ArrayLike,
    ax: Axes | None = None,
    cmap: ColorSpec = None,
    background_color: str = "white",
    n_clusters: int | None = None,
    binary: bool | None = None,
) -> Axes:
    """Color each object of a segmented image by its shape mode.

    Parameters
    ----------
    image_label : ndarray of shape (height, width)
        Label image in which object ``l`` has pixel value ``l`` (background
        0), or a binary mask. Interpreted exactly as by
        :func:`vampire.extract_properties` with the same ``binary``.
    objects : DataFrame or array-like of shape (n_objects,)
        Rows of the property table belonging to this image (needs a ``label``
        column), or directly the integer object labels.
    labels : array-like of shape (n_objects,)
        Shape-mode label (``1..n_clusters``) of each row of ``objects``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    cmap : str, Colormap, or sequence, optional
        Colors of the shape modes, as ``colors`` in :func:`plot_shape_modes`.
    background_color : color, default="white"
        Color of pixels that are background or belong to unlisted objects.
    n_clusters : int, optional
        Number of shape modes, so colors match other figures. Defaults to
        ``max(labels)``; pass ``model.n_clusters``.
    binary : bool or None, default=None
        How ``image_label`` is interpreted; see the ``binary`` parameter of
        :func:`vampire.extract_properties`. Pass the value used there so the
        object labels match.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Shows an RGBA image of shape ``(height, width, 4)`` in which objects
        of shape mode ``m`` have palette color ``m - 1``; axes decorations
        are hidden.

    Raises
    ------
    ValueError
        If ``image_label`` is not 2-D, or ``objects`` and ``labels`` differ in
        length, or labels are not integers in ``1..n_clusters``, or object
        labels are not non-negative integers (whole-number floats are
        accepted).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import vampire as vp
    >>> images, names = vp.datasets.load_example_images()
    >>> props = vp.extract_properties(images[:1])
    >>> modes = [i % 3 + 1 for i in range(len(props))]
    >>> ax = vp.plotting.plot_labeled_image(images[0], props, modes)
    >>> ax.images[0].get_array().shape[-1]
    4
    """
    image = np.asarray(image_label)
    if image.ndim != 2:
        raise ValueError(f"image_label must be 2-D; got shape {image.shape}.")
    if hasattr(objects, "columns"):
        if "label" not in objects.columns:
            raise ValueError("objects must have a 'label' column.")
        object_labels = np.asarray(objects["label"])
    else:
        object_labels = _as_labels(objects)
    modes = _as_labels(labels)
    if object_labels.ndim != 1 or modes.shape != object_labels.shape:
        raise ValueError(
            "objects and labels must be 1-D and of equal length; got "
            f"{object_labels.shape} and {modes.shape}."
        )
    modes, k = check_labels(modes, n_clusters, allow_empty=True)
    try:
        object_labels, _ = check_labels(
            object_labels, allow_empty=True, one_based=False
        )
    except ValueError as err:
        raise ValueError(f"Object labels are invalid: {err}") from None
    image = _to_label_image(image, binary=binary)
    # Lookup table: pixel label -> color index (k = background).
    size = max(int(image.max(initial=0)), int(object_labels.max(initial=0))) + 1
    lut = np.full(size, k)
    lut[object_labels] = modes - 1  # palette row of each 1-based mode
    table = np.vstack(
        [_mode_colors(k, cmap) if k else np.empty((0, 4)), to_rgba(background_color)]
    )
    rgba = table[lut[image]]
    ax = _get_ax(ax, (5, 5))
    ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    return ax
