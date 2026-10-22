"""Private input-validation helpers shared by transformers and the model."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Column holding raw contours in property tables from ``extract_properties``.
CONTOUR_COLUMN = "contour"

#: Minimum number of points of a valid contour.
MIN_CONTOUR_POINTS = 4

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
        Contours as C-contiguous ``float64`` arrays.

    Raises
    ------
    ValueError
        If ``X`` is empty, a DataFrame lacks the ``"contour"`` column, or any
        contour is not 2-D with 2 rows and at least 4 finite points.

    Examples
    --------
    >>> import numpy as np
    >>> from vampire._validation import check_contours
    >>> contours = check_contours([np.zeros((2, 5)), np.ones((2, 7))])
    >>> [c.shape for c in contours]
    [(2, 5), (2, 7)]
    """
    if isinstance(X, pd.DataFrame):
        if CONTOUR_COLUMN not in X.columns:
            raise ValueError(
                f"DataFrame input must have a {CONTOUR_COLUMN!r} column "
                "(as returned by vampire.extract_properties); "
                f"got columns {list(X.columns)}."
            )
        items = list(X[CONTOUR_COLUMN])
    elif isinstance(X, pd.Series):
        items = list(X)
    elif isinstance(X, np.ndarray) and X.dtype != object:
        if X.ndim != 3 or X.shape[1] != 2:
            raise ValueError(
                "Array input must have shape (n_samples, 2, n_points); "
                f"got shape {X.shape}. Wrap a single contour in a list: [contour]."
            )
        items = list(X)
    elif isinstance(X, str) or not hasattr(X, "__len__"):
        raise ValueError(
            "Expected a contour set (list of (2, n) arrays, ndarray, DataFrame "
            f"or Series); got {type(X).__name__}."
        )
    else:
        items = list(X)

    if len(items) == 0:
        raise ValueError("Contour set is empty; at least one contour is required.")

    contours = []
    for i, item in enumerate(items):
        try:
            contour = np.asarray(item, dtype=np.float64, order="C")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Contour {i} cannot be converted to floats.") from err
        if contour.ndim != 2 or contour.shape[0] != 2:
            raise ValueError(
                f"Contour {i} must have shape (2, n_points) (row 0 = x, row 1 = y); "
                f"got shape {contour.shape}."
            )
        if contour.shape[1] < MIN_CONTOUR_POINTS:
            raise ValueError(
                f"Contour {i} has {contour.shape[1]} points; at least "
                f"{MIN_CONTOUR_POINTS} are required."
            )
        if not np.all(np.isfinite(contour)):
            raise ValueError(f"Contour {i} contains NaN or infinite coordinates.")
        contours.append(contour)
    return contours


def check_dense_contours(X: ContourSet) -> NDArray[np.float64]:
    """Validate equal-length contours and return a dense array.

    Accepts everything :func:`check_contours` accepts plus flattened arrays of
    shape ``(n_samples, 2 * n_points)`` (x coordinates first, then y).

    Parameters
    ----------
    X : contour set or ndarray of shape (n_samples, 2 * n_points)
        Contours that all have the same number of points.

    Returns
    -------
    contours : ndarray of shape (n_samples, 2, n_points)
        ``float64`` array.

    Raises
    ------
    ValueError
        If the input is invalid (see :func:`check_contours`), a flattened array
        has an odd number of columns, or contours differ in length.

    Examples
    --------
    >>> import numpy as np
    >>> from vampire._validation import check_dense_contours
    >>> flat = np.arange(12.0).reshape(1, 12)  # x0..x5, then y0..y5
    >>> check_dense_contours(flat).shape
    (1, 2, 6)
    """
    if isinstance(X, np.ndarray) and X.dtype != object and X.ndim == 2:
        if X.shape[1] % 2 != 0:
            raise ValueError(
                "Flattened contours must have shape (n_samples, 2 * n_points) "
                f"with x coordinates first, then y; got {X.shape[1]} columns."
            )
        X = X.reshape(X.shape[0], 2, -1)
    contours = check_contours(X)
    lengths = {contour.shape[1] for contour in contours}
    if len(lengths) > 1:
        raise ValueError(
            f"Contours have different numbers of points {sorted(lengths)}; "
            "resample them first, e.g. with vampire.ContourResampler."
        )
    return np.stack(contours)
