"""Private input-validation helpers shared by transformers and the model."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Column holding raw contours in property tables from ``extract_properties``.
CONTOUR_COLUMN = "contour"

#: Minimum number of points of a valid contour (a closed spline needs 3).
MIN_CONTOUR_POINTS = 3

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
        contour is not 2-D with 2 rows and at least 3 finite points.

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
    shape ``(n_samples, 2 * n_points)`` (x coordinates first, then y). A 2-D
    array with exactly 2 rows is rejected as ambiguous: it could be one
    ``(2, n_points)`` contour or two flattened contours.

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
        If the input is invalid (see :func:`check_contours`), a 2-D array has 2
        rows, a flattened array has an odd number of columns, or contours
        differ in length.

    Examples
    --------
    >>> import numpy as np
    >>> from vampire._validation import check_dense_contours
    >>> flat = np.arange(12.0).reshape(1, 12)  # x0..x5, then y0..y5
    >>> check_dense_contours(flat).shape
    (1, 2, 6)
    """
    if isinstance(X, np.ndarray) and X.dtype != object and X.ndim == 2:
        if X.shape[0] == 2:
            raise ValueError(
                f"Got a 2-D array of shape {X.shape}, which is ambiguous (one "
                "(2, n_points) contour or two flattened contours): wrap a single "
                "contour in a list, or pass shape (n_samples, 2, n_points)."
            )
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


def check_n_clusters(n_clusters: object, name: str = "n_clusters") -> int:
    """Return ``n_clusters`` as an int, raising ``ValueError`` unless it is >= 1.

    Examples
    --------
    >>> from vampire._validation import check_n_clusters
    >>> check_n_clusters(3)
    3
    """
    if (
        not isinstance(n_clusters, int | np.integer)
        or isinstance(n_clusters, bool | np.bool_)
        or n_clusters < 1
    ):
        raise ValueError(f"{name} must be an integer >= 1; got {n_clusters!r}.")
    return int(n_clusters)


def check_labels(
    labels: object,
    n_clusters: int | None = None,
    *,
    allow_empty: bool = False,
) -> tuple[NDArray[np.intp], int]:
    """Validate shape-mode labels and return them with the number of modes.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Non-negative integer labels. Whole-number floats (e.g. ``1.0``) are
        accepted; booleans are not.
    n_clusters : int, optional
        Number of shape modes (>= 1). Defaults to ``max(labels) + 1`` (0 for
        empty labels).
    allow_empty : bool, default=False
        Accept an empty array.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        ``intp`` labels.
    n_clusters : int

    Raises
    ------
    ValueError
        If ``labels`` is not 1-D, is empty (unless ``allow_empty``), is not
        integer-valued, is negative, or is not below ``n_clusters``; or if
        ``n_clusters`` is not an integer >= 1.

    Examples
    --------
    >>> from vampire._validation import check_labels
    >>> values, k = check_labels([0.0, 2.0])
    >>> values.tolist(), k
    ([0, 2], 3)
    """
    values = np.asarray(labels)
    if values.ndim != 1 or (values.size == 0 and not allow_empty):
        raise ValueError(
            f"labels must be a non-empty 1-D array; got shape {values.shape}."
        )
    if values.size:
        integral_floats = (
            np.issubdtype(values.dtype, np.floating)
            and np.all(np.isfinite(values))
            and np.all(values == np.round(values))
        )
        if not (np.issubdtype(values.dtype, np.integer) or integral_floats):
            raise ValueError("labels must be integers.")
    values = values.astype(np.intp)
    if values.size and values.min() < 0:
        raise ValueError(f"labels must be non-negative; got {values.min()}.")
    if n_clusters is None:
        n_clusters = int(values.max(initial=-1)) + 1
    else:
        n_clusters = check_n_clusters(n_clusters)
        if values.size and values.max() >= n_clusters:
            raise ValueError(
                f"labels must be below n_clusters={n_clusters}; got {values.max()}."
            )
    return values, n_clusters
