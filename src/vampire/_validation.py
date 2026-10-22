"""Private input-validation helpers shared by transformers and the model."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Column holding raw contours in property tables from ``extract_properties``.
CONTOUR_COLUMN = "contour"

ContourSet = Sequence[NDArray] | NDArray | pd.DataFrame | pd.Series


def check_contours(X: ContourSet) -> list[NDArray[np.float64]]:
    """Validate a contour set and return it as a list of float arrays.

    Parameters
    ----------
    X : sequence of ndarray of shape (2, n_i), ndarray of shape (n_samples, 2, n), \
DataFrame with a ``"contour"`` column, or Series of such arrays
        Contour set. Contours may have different lengths ``n_i``.

    Returns
    -------
    contours : list of ndarray of shape (2, n_i)
        Contours as ``float64`` arrays.

    Raises
    ------
    ValueError
        If ``X`` is empty, a DataFrame lacks the ``"contour"`` column, or any
        contour is not 2-D with 2 rows and at least 4 points.
    """
    raise NotImplementedError


def check_dense_contours(X: ContourSet) -> NDArray[np.float64]:
    """Validate equal-length contours and return a dense array.

    Accepts everything :func:`check_contours` accepts plus flattened arrays of
    shape ``(n_samples, 2 * n_points)`` (x coordinates first, then y).

    Returns
    -------
    contours : ndarray of shape (n_samples, 2, n_points)
    """
    raise NotImplementedError
