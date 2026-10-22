"""Scikit-learn transformers that normalize contours.

Chain them as ``ContourResampler -> ContourRegistrar -> ContourAligner`` to turn
raw, variable-length contours into a fixed-length feature matrix suitable for
any scikit-learn estimator (PCA, KMeans, ...). :class:`vampire.Vampire` uses
exactly this chain internally.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from ._validation import ContourSet


class ContourResampler(TransformerMixin, BaseEstimator):
    """Resample contours to a fixed number of equidistant points.

    Stateless: ``fit`` only records input metadata.

    Parameters
    ----------
    n_points : int, default=50
        Number of points per output contour.

    Notes
    -----
    Input: any contour set accepted by ``vampire._validation.check_contours``
    (list of ``(2, n_i)`` arrays, DataFrame with ``"contour"`` column, ...).
    Output of ``transform``: ndarray of shape ``(n_samples, 2, n_points)``.
    """

    def __init__(self, n_points: int = 50):
        self.n_points = n_points

    def fit(self, X: ContourSet, y=None) -> "ContourResampler":
        """Validate parameters and return self."""
        raise NotImplementedError

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Resample each contour; returns shape ``(n_samples, 2, n_points)``."""
        raise NotImplementedError


class ContourRegistrar(TransformerMixin, BaseEstimator):
    """Remove translation, scale, and rotation from each contour.

    Stateless. Applies :func:`vampire.contours.register_contour` to every
    contour. Input and output: ndarray of shape ``(n_samples, 2, n_points)``.
    """

    def fit(self, X: ContourSet, y=None) -> "ContourRegistrar":
        """Validate input and return self."""
        raise NotImplementedError

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Register each contour; returns shape ``(n_samples, 2, n_points)``."""
        raise NotImplementedError


class ContourAligner(TransformerMixin, BaseEstimator):
    """Align registered contours to their mean contour.

    Parameters
    ----------
    flatten : bool, default=True
        If True, ``transform`` returns shape ``(n_samples, 2 * n_points)``
        (x coordinates first, then y) so the output can feed scikit-learn
        estimators directly. If False, returns ``(n_samples, 2, n_points)``.

    Attributes
    ----------
    reference_contour_ : ndarray of shape (2, n_points)
        Mean of the registered training contours; the alignment target.
    mean_contour_ : ndarray of shape (2, n_points)
        Mean of the aligned training contours.
    n_points_ : int
        Number of points per contour seen during ``fit``.
    """

    def __init__(self, flatten: bool = True):
        self.flatten = flatten

    def fit(self, X: ContourSet, y=None) -> "ContourAligner":
        """Learn the reference contour from registered contours."""
        raise NotImplementedError

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Align each contour to ``reference_contour_``."""
        raise NotImplementedError
