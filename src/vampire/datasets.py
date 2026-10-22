"""Example and synthetic data for tutorials and tests."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def load_example_images() -> tuple[list[NDArray], list[str]]:
    """Load the bundled segmented microglia images.

    Four small binary images of microglia from rat brain slices (cortex and
    hypothalamus) shipped in ``vampire/_data``.

    Returns
    -------
    images : list of ndarray of shape (height, width)
    filenames : list of str
    """
    raise NotImplementedError


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
    labels : ndarray of shape (n_samples,)
        Only if ``return_labels`` is True.
    """
    raise NotImplementedError
