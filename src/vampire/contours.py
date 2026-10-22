"""Pure functions operating on a single contour.

A contour is a float array of shape ``(2, n_points)``: row 0 holds x
coordinates and row 1 holds y coordinates of points ordered along the object
boundary. The contour is treated as closed (the last point connects back to
the first); the first point is not repeated at the end.

These functions are the building blocks of the transformers in
:mod:`vampire.preprocessing`.
"""

import numpy as np
from numpy.typing import NDArray


def resample_contour(contour: NDArray, n_points: int = 50) -> NDArray[np.float64]:
    """Resample a closed contour to ``n_points`` points equidistant in arc length.

    Fits a periodic interpolating B-spline through the contour, parameterized
    by cumulative arc length, and evaluates it at ``n_points`` evenly spaced
    arc-length positions.

    Parameters
    ----------
    contour : ndarray of shape (2, n)
        Closed contour with ``n >= 4`` points.
    n_points : int, default=50
        Number of output points.

    Returns
    -------
    resampled : ndarray of shape (2, n_points)
    """
    raise NotImplementedError


def register_contour(contour: NDArray) -> NDArray[np.float64]:
    """Remove translation, scale, and rotation from a contour.

    Steps: (1) subtract the centroid; (2) divide by the characteristic length
    ``R = sqrt(mean(x**2 + y**2))``; (3) rotate onto principal axes (SVD) so the
    major axis lies along x; (4) cyclically shift so the first point is the one
    with the smallest absolute polar angle; (5) reverse point order if needed so
    the contour runs counterclockwise.

    Parameters
    ----------
    contour : ndarray of shape (2, n_points)

    Returns
    -------
    registered : ndarray of shape (2, n_points)
        Centered, unit-scale, principal-axis-aligned, counterclockwise contour.
    """
    raise NotImplementedError


def align_contour(contour: NDArray, reference: NDArray) -> NDArray[np.float64]:
    """Align a registered contour to a reference contour.

    Tries every cyclic shift of the contour's starting point, finds the optimal
    rotation to ``reference`` for each (Kabsch algorithm), and returns the
    shifted and rotated contour with the smallest sum of squared distances.

    Parameters
    ----------
    contour : ndarray of shape (2, n_points)
        Registered contour.
    reference : ndarray of shape (2, n_points)
        Reference (typically the mean registered contour).

    Returns
    -------
    aligned : ndarray of shape (2, n_points)
    """
    raise NotImplementedError
