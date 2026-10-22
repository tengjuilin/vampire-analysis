"""Scikit-learn transformers that normalize contours.

Chain them as ``ContourResampler -> ContourRegistrar -> ContourAligner`` to turn
raw, variable-length contours into a fixed-length feature matrix suitable for
any scikit-learn estimator (PCA, KMeans, ...). :class:`vampire.Vampire` uses
exactly this chain internally.

All transformers accept any contour set understood by
``vampire._validation.check_contours``: a list of ``(2, n_i)`` arrays, an
ndarray of shape ``(n_samples, 2, n)``, a pandas Series of arrays, or a
DataFrame with a ``"contour"`` column. :class:`ContourRegistrar` and
:class:`ContourAligner` additionally accept flattened arrays of shape
``(n_samples, 2 * n_points)`` and require all contours to have the same
length.

``set_output``: :class:`ContourAligner` (with ``flatten=True``) supports
``set_output(transform="pandas")`` with columns ``x0, ..., x{n-1}, y0, ...,
y{n-1}``. :class:`ContourResampler` and :class:`ContourRegistrar` return 3-D
arrays, which cannot be DataFrames; their ``set_output`` accepts the standard
values (so ``Pipeline.set_output`` works) but leaves the output an ndarray.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._validation import ContourSet, check_contours, check_dense_contours
from .contours import align_contour, register_contour, resample_contour

_OUTPUT_CONTAINERS = (None, "default", "pandas", "polars")


class _ContourOutputMixin:
    """``set_output`` for transformers whose output is a 3-D contour array."""

    def set_output(self, *, transform: str | None = None):
        """Accept an output container request; output stays a 3-D ndarray.

        Contour arrays of shape ``(n_samples, 2, n_points)`` cannot be stored
        in a DataFrame, so ``"pandas"`` and ``"polars"`` are accepted (letting
        :meth:`sklearn.pipeline.Pipeline.set_output` configure a whole
        pipeline) but have no effect on this step.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"} or None, default=None
            Requested output container.

        Returns
        -------
        self : estimator

        Raises
        ------
        ValueError
            If ``transform`` is not one of the values above.
        """
        if transform not in _OUTPUT_CONTAINERS:
            raise ValueError(
                f"transform must be one of {list(_OUTPUT_CONTAINERS)}; got "
                f"{transform!r}. {type(self).__name__} always returns a 3-D "
                "ndarray of shape (n_samples, 2, n_points)."
            )
        return self


class ContourResampler(_ContourOutputMixin, TransformerMixin, BaseEstimator):
    """Resample contours to a fixed number of equidistant points.

    Stateless: ``fit`` only validates the input and parameters, and
    ``transform`` may be called without fitting. Applies
    :func:`vampire.contours.resample_contour` to every contour.

    Parameters
    ----------
    n_points : int, default=50
        Number of points per output contour, at least 3.

    Notes
    -----
    Input: any contour set accepted by ``vampire._validation.check_contours``
    (list of ``(2, n_i)`` arrays, DataFrame with ``"contour"`` column, ...).
    Output of ``transform``: ndarray of shape ``(n_samples, 2, n_points)``.

    Examples
    --------
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(n_samples=4, n_points=80, random_state=0)
    >>> vp.ContourResampler(n_points=30).fit_transform(X).shape
    (4, 2, 30)
    """

    def __init__(self, n_points: int = 50):
        self.n_points = n_points

    def __sklearn_tags__(self):
        """Declare the input format (and statelessness, where applicable)."""
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        tags.input_tags.two_d_array = False
        return tags

    def _check_params(self) -> None:
        if (
            not isinstance(self.n_points, int | np.integer)
            or isinstance(self.n_points, bool)
            or self.n_points < 3
        ):
            raise ValueError(
                f"n_points must be an integer >= 3; got {self.n_points!r}."
            )

    def fit(self, X: ContourSet, y=None) -> "ContourResampler":
        """Validate parameters and input, and return self.

        Parameters
        ----------
        X : contour set
            Contours of any lengths.
        y : None
            Ignored.

        Returns
        -------
        self : ContourResampler
        """
        self._check_params()
        check_contours(X)
        return self

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Resample each contour.

        Parameters
        ----------
        X : contour set
            Contours of any lengths (at least 3 distinct points each).

        Returns
        -------
        resampled : ndarray of shape (n_samples, 2, n_points)
        """
        self._check_params()
        contours = check_contours(X)
        return np.stack([resample_contour(c, self.n_points) for c in contours])


class ContourRegistrar(_ContourOutputMixin, TransformerMixin, BaseEstimator):
    """Remove translation, scale, and rotation from each contour.

    Stateless: ``transform`` may be called without fitting. Applies
    :func:`vampire.contours.register_contour` to every contour. Input and
    output: ndarray of shape ``(n_samples, 2, n_points)`` (flattened input of
    shape ``(n_samples, 2 * n_points)`` is also accepted).

    Examples
    --------
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(n_samples=4, random_state=0)
    >>> resampled = vp.ContourResampler(n_points=30).fit_transform(X)
    >>> registered = vp.ContourRegistrar().fit_transform(resampled)
    >>> registered.shape
    (4, 2, 30)
    >>> bool(abs(registered.mean(axis=2)).max() < 1e-12)  # centered
    True
    """

    def __sklearn_tags__(self):
        """Declare the input format (and statelessness, where applicable)."""
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        tags.input_tags.two_d_array = False
        return tags

    def fit(self, X: ContourSet, y=None) -> "ContourRegistrar":
        """Validate input and return self.

        Parameters
        ----------
        X : contour set of equal-length contours
        y : None
            Ignored.

        Returns
        -------
        self : ContourRegistrar
        """
        check_dense_contours(X)
        return self

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Register each contour.

        Parameters
        ----------
        X : contour set of equal-length contours

        Returns
        -------
        registered : ndarray of shape (n_samples, 2, n_points)
        """
        contours = check_dense_contours(X)
        return np.stack([register_contour(c) for c in contours])


def _align_all(
    registered: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Align each contour of ``registered`` (n_samples, 2, n) to ``reference``."""
    return np.stack([align_contour(c, reference) for c in registered])


class ContourAligner(TransformerMixin, BaseEstimator):
    """Align registered contours to their mean contour.

    Applies :func:`vampire.contours.align_contour` with
    ``reference_contour_`` as the reference to every contour.

    ``fit`` finds the reference by generalized Procrustes analysis: it starts
    from the mean of the registered contours, aligns every contour to it,
    replaces the reference by the mean of the aligned contours, and repeats
    until the reference stops changing.

    Parameters
    ----------
    flatten : bool, default=True
        If True, ``transform`` returns shape ``(n_samples, 2 * n_points)``
        (x coordinates first, then y) so the output can feed scikit-learn
        estimators directly. If False, returns ``(n_samples, 2, n_points)``.
    max_iter : int, default=30
        Maximum number of alignment passes over the training contours
        (>= 1). ``max_iter=1`` aligns once to the mean of the registered
        contours, as before version 1.0.
    tol : float, default=1e-6
        Stop when the root-mean-square change of the reference between two
        passes, relative to its root-mean-square size, is at most ``tol``
        (>= 0). Alignment picks a discrete cyclic shift per contour; once
        those choices stop changing, the reference converges geometrically,
        typically within 10-20 passes.

    Attributes
    ----------
    reference_contour_ : ndarray of shape (2, n_points)
        The alignment target: the mean of the registered training contours,
        refined by the alignment passes.
    mean_contour_ : ndarray of shape (2, n_points)
        Mean of the aligned training contours.
    n_iter_ : int
        Number of alignment passes run during ``fit``.
    n_points_ : int
        Number of points per contour seen during ``fit``.

    Notes
    -----
    Each pass aligns every contour to the current reference ``M``. The mean
    of the aligned contours is itself aligned to ``M`` (rotation and cyclic
    shift), so the reference cannot drift, and becomes the next reference.
    The training contours returned by ``fit_transform`` are aligned to the
    final ``reference_contour_``, exactly as ``transform`` aligns new ones.

    With ``flatten=True``, :meth:`get_feature_names_out` returns
    ``x0, ..., x{n-1}, y0, ..., y{n-1}``, so ``set_output(transform="pandas")``
    yields a DataFrame with these columns.

    Examples
    --------
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(n_samples=6, random_state=0)
    >>> resampled = vp.ContourResampler(n_points=30).fit_transform(X)
    >>> registered = vp.ContourRegistrar().fit_transform(resampled)
    >>> aligner = vp.ContourAligner().fit(registered)
    >>> aligner.transform(registered).shape
    (6, 60)
    >>> aligner.mean_contour_.shape
    (2, 30)
    >>> aligner.n_iter_ <= aligner.max_iter
    True
    """

    def __init__(self, flatten: bool = True, max_iter: int = 30, tol: float = 1e-6):
        self.flatten = flatten
        self.max_iter = max_iter
        self.tol = tol

    def __sklearn_tags__(self):
        """Declare the input format (and statelessness, where applicable)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags

    def _fit(self, X: ContourSet) -> NDArray[np.float64]:
        """Fit and return the aligned training contours, (n_samples, 2, n_points)."""
        if (
            not isinstance(self.max_iter, int | np.integer)
            or isinstance(self.max_iter, bool)
            or self.max_iter < 1
        ):
            raise ValueError(
                f"max_iter must be an integer >= 1; got {self.max_iter!r}."
            )
        if not (isinstance(self.tol, int | float | np.number) and self.tol >= 0):
            raise ValueError(f"tol must be a number >= 0; got {self.tol!r}.")
        registered = check_dense_contours(X)
        self.n_points_ = registered.shape[2]
        reference = registered.mean(axis=0)
        for n_iter in range(1, self.max_iter + 1):
            aligned = _align_all(registered, reference)
            if n_iter == self.max_iter:
                break
            # Align the new mean to the old one so the reference cannot drift.
            new_reference = align_contour(aligned.mean(axis=0), reference)
            change = np.sqrt(np.mean((new_reference - reference) ** 2))
            size = np.sqrt(np.mean(reference**2))
            if change <= self.tol * size:
                break
            reference = new_reference
        self.n_iter_ = n_iter
        self.reference_contour_ = reference
        self.mean_contour_ = aligned.mean(axis=0)
        return aligned

    def _align(self, registered: NDArray[np.float64]) -> NDArray[np.float64]:
        return _align_all(registered, self.reference_contour_)

    def _format(self, aligned: NDArray[np.float64]) -> NDArray[np.float64]:
        return aligned.reshape(len(aligned), -1) if self.flatten else aligned

    def fit(self, X: ContourSet, y=None) -> "ContourAligner":
        """Learn the reference contour from registered contours.

        Parameters
        ----------
        X : contour set of equal-length registered contours
            Typically the output of :class:`ContourRegistrar`.
        y : None
            Ignored.

        Returns
        -------
        self : ContourAligner
        """
        self._fit(X)
        return self

    def fit_transform(self, X: ContourSet, y=None) -> NDArray[np.float64]:
        """Fit and return the aligned training contours (aligning only once).

        Parameters
        ----------
        X : contour set of equal-length registered contours
        y : None
            Ignored.

        Returns
        -------
        aligned : ndarray of shape (n_samples, 2 * n_points) or \
(n_samples, 2, n_points)
            Flattened if ``flatten`` is True.
        """
        return self._format(self._fit(X))

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Align each contour to ``reference_contour_``.

        Parameters
        ----------
        X : contour set of registered contours with ``n_points_`` points

        Returns
        -------
        aligned : ndarray of shape (n_samples, 2 * n_points) or \
(n_samples, 2, n_points)
            Flattened if ``flatten`` is True.
        """
        check_is_fitted(self)
        registered = check_dense_contours(X)
        if registered.shape[2] != self.n_points_:
            raise ValueError(
                f"Contours have {registered.shape[2]} points but ContourAligner "
                f"was fitted with {self.n_points_}; resample them to "
                f"{self.n_points_} points."
            )
        return self._format(self._align(registered))

    def get_feature_names_out(self, input_features=None) -> NDArray[np.object_]:
        """Return the names of the flattened output columns.

        Parameters
        ----------
        input_features : None
            Ignored (the input is a contour set, not named features).

        Returns
        -------
        feature_names_out : ndarray of str objects of shape ``(2 * n_points_,)``
            ``["x0", ..., "x{n-1}", "y0", ..., "y{n-1}"]``.

        Raises
        ------
        ValueError
            If ``flatten`` is False (the output is 3-D and has no columns).
        """
        check_is_fitted(self)
        if not self.flatten:
            raise ValueError(
                "get_feature_names_out requires flatten=True; with flatten=False "
                "the output has shape (n_samples, 2, n_points) and no columns."
            )
        n = self.n_points_
        return np.asarray(
            [f"x{i}" for i in range(n)] + [f"y{i}" for i in range(n)], dtype=object
        )
