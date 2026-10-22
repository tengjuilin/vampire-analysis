"""Example and synthetic data for tutorials and tests."""

from importlib import resources
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def load_example_images() -> tuple[list[NDArray[np.uint16]], list[str]]:
    """Load the bundled segmented microglia images.

    Two 512 x 512 label images of segmented microglia (about 117 and 122
    objects). Background is 0 and each object has a unique positive integer
    label, so they can be passed straight to
    :func:`vampire.extract_properties`. Stored compressed in
    ``vampire/_data/example_images.npz``.

    Returns
    -------
    images : list of ndarray of shape (512, 512), dtype uint16
        Label images.
    filenames : list of str
        Names ``["img1", "img2"]``, usable as the ``filenames`` argument of
        :func:`vampire.extract_properties`.

    Examples
    --------
    >>> images, filenames = load_example_images()
    >>> len(images), images[0].shape
    (2, (512, 512))
    >>> filenames
    ['img1', 'img2']
    """
    data = resources.files("vampire") / "_data" / "example_images.npz"
    with data.open("rb") as f, np.load(f) as archive:
        filenames = sorted(archive.files)
        images = [archive[name] for name in filenames]
    return images, filenames


def make_shapes(
    n_samples: int = 100,
    kinds: tuple[Literal["circle", "ellipse", "triangle", "star"], ...] = (
        "circle",
        "ellipse",
    ),
    n_points: int = 120,
    noise: float = 0.01,
    random_state: int | np.random.Generator | None = None,
    return_labels: bool = False,
) -> list[NDArray[np.float64]] | tuple[list[NDArray[np.float64]], NDArray[np.intp]]:
    """Generate random contours of simple geometric shapes.

    Each contour is a shape of the given kind with random scale, rotation,
    translation, starting point, and orientation, plus Gaussian radial noise —
    so a correct VAMPIRE pipeline should group contours by ``kind`` only.

    Parameters
    ----------
    n_samples : int, default=100
        Total contours, split as evenly as possible across ``kinds``.
    kinds : tuple of str, default=("circle", "ellipse")
        Shape kinds to draw.
    n_points : int, default=120
        Points per contour.
    noise : float, default=0.01
        Std of radial noise relative to shape size.
    random_state : int, Generator or None
        Seed or generator.
    return_labels : bool, default=False
        If True, also return the index into ``kinds`` of each contour.

    Returns
    -------
    contours : list of ndarray of shape (2, n_points)
        ``float64`` contours in random order (kinds are interleaved).
    labels : ndarray of shape (n_samples,)
        Integer index into ``kinds`` of each contour. Only if
        ``return_labels`` is True.

    Raises
    ------
    ValueError
        If ``kinds`` is empty or contains an unknown kind, ``n_samples < 1``,
        ``n_points < 4``, or ``noise < 0``.

    Notes
    -----
    Base shapes (before the random similarity transform):

    - ``"circle"``: unit circle;
    - ``"ellipse"``: ellipse with random aspect ratio in [2.5, 3];
    - ``"triangle"``: equilateral triangle, points evenly spaced on its sides;
    - ``"star"``: smooth five-pointed star, ``r = 1 + 0.4 cos(5 t)``.

    Each point is then moved radially by a factor ``1 + noise * N(0, 1)``, and
    the shape is scaled by a random factor in [15, 60], rotated by a random
    angle, translated by a random offset in [0, 500)², cyclically shifted to a
    random starting point, and reversed (clockwise) with probability 1/2.

    Examples
    --------
    >>> contours, labels = make_shapes(n_samples=5, random_state=0, return_labels=True)
    >>> len(contours), contours[0].shape
    (5, (2, 120))
    >>> sorted(labels.tolist())  # 3 circles (0) and 2 ellipses (1)
    [0, 0, 0, 1, 1]
    """
    if isinstance(kinds, str):
        kinds = (kinds,)
    unknown = [kind for kind in kinds if kind not in _SHAPE_MAKERS]
    if len(kinds) == 0 or unknown:
        raise ValueError(
            f"kinds must be a non-empty tuple of {sorted(_SHAPE_MAKERS)}; "
            f"got {kinds!r}."
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}.")
    if n_points < 4:
        raise ValueError(f"n_points must be >= 4; got {n_points}.")
    if noise < 0:
        raise ValueError(f"noise must be >= 0; got {noise}.")

    rng = np.random.default_rng(random_state)
    labels = rng.permutation(np.arange(n_samples) % len(kinds)).astype(np.intp)
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    contours = []
    for label in labels:
        shape = _SHAPE_MAKERS[kinds[label]](t, rng)
        shape = shape * (1.0 + noise * rng.standard_normal(n_points))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        scale = rng.uniform(15.0, 60.0)
        offset = rng.uniform(0.0, 500.0, size=(2, 1))
        contour = scale * rotation @ shape + offset
        contour = np.roll(contour, rng.integers(n_points), axis=1)
        if rng.random() < 0.5:
            contour = contour[:, ::-1]
        contours.append(np.ascontiguousarray(contour))

    if return_labels:
        return contours, labels
    return contours


def _circle(t: NDArray, rng: np.random.Generator) -> NDArray[np.float64]:
    return np.vstack([np.cos(t), np.sin(t)])


def _ellipse(t: NDArray, rng: np.random.Generator) -> NDArray[np.float64]:
    aspect = rng.uniform(2.5, 3.0)
    return np.vstack([aspect * np.cos(t), np.sin(t)]) / np.sqrt(aspect)


def _triangle(t: NDArray, rng: np.random.Generator) -> NDArray[np.float64]:
    angles = np.pi / 2 + 2 * np.pi / 3 * np.arange(4)  # closed: last == first
    vertices = np.vstack([np.cos(angles), np.sin(angles)])
    position = 3 * t / (2 * np.pi)  # side index + fraction along the side
    side = np.floor(position).astype(int)
    frac = position - side
    return vertices[:, side] * (1 - frac) + vertices[:, side + 1] * frac


def _star(t: NDArray, rng: np.random.Generator) -> NDArray[np.float64]:
    radius = 1.0 + 0.4 * np.cos(5 * t)
    return radius * np.vstack([np.cos(t), np.sin(t)])


_SHAPE_MAKERS = {
    "circle": _circle,
    "ellipse": _ellipse,
    "triangle": _triangle,
    "star": _star,
}
