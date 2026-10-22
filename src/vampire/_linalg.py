"""Private linear-algebra helpers."""

import numpy as np
from numpy.typing import NDArray


def kabsch_rotation(A: NDArray[np.floating], B: NDArray[np.floating]) -> NDArray:
    """Return the proper rotation matrix ``R`` minimizing ``||R @ A - B||_F``.

    Parameters
    ----------
    A, B : ndarray of shape (2, n_points)
        Centered point sets with corresponding columns.

    Returns
    -------
    R : ndarray of shape (2, 2)
        Rotation matrix with ``det(R) == +1`` (reflections are excluded).
    """
    raise NotImplementedError
