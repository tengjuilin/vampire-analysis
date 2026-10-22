r"""Pure functions operating on a single contour.

A contour is a float array of shape ``(2, n_points)``: row 0 holds x
coordinates and row 1 holds y coordinates of points ordered along the object
boundary. The contour is treated as closed (the last point connects back to
the first); the first point is not repeated at the end.

These functions are the building blocks of the transformers in
:mod:`vampire.preprocessing`. Applied in order, they normalize away everything
except shape:

1. :func:`resample_contour` -- same number of points, evenly spaced;
2. :func:`register_contour` -- remove translation, scale, and rotation, and
   fix the starting point and direction;
3. :func:`align_contour` -- fine-tune starting point and rotation against a
   reference (the mean registered contour of a data set).

:func:`lowpass_contour` is not part of that pipeline: it removes
high-frequency detail (pixel staircase, segmentation noise) for display, as
used by the ``n_harmonics`` option of :mod:`vampire.plotting`.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

from ._linalg import kabsch_rotation
from ._validation import MIN_CONTOUR_POINTS

#: Dense spline evaluations per output (or input) point used to measure the
#: spline's arc length in :func:`resample_contour`.
_ARC_LENGTH_OVERSAMPLING = 16


def _check_contour(contour: NDArray, name: str = "contour") -> NDArray[np.float64]:
    """Return ``contour`` as a finite float64 array of shape (2, n >= 3)."""
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[0] != 2:
        raise ValueError(
            f"{name} must have shape (2, n_points) (row 0 = x, row 1 = y); "
            f"got shape {contour.shape}."
        )
    if contour.shape[1] < MIN_CONTOUR_POINTS:
        raise ValueError(
            f"{name} has {contour.shape[1]} points; at least "
            f"{MIN_CONTOUR_POINTS} are required."
        )
    if not np.all(np.isfinite(contour)):
        raise ValueError(f"{name} contains NaN or infinite coordinates.")
    return contour


def _segment_lengths(points: NDArray) -> NDArray[np.float64]:
    """Return lengths of the segments of an open polyline, shape (n - 1,)."""
    return np.hypot(*np.diff(points, axis=1))


def _drop_repeated_points(contour: NDArray) -> NDArray[np.float64]:
    """Remove points that coincide with their successor along the closed contour.

    This also removes a closing point that repeats the first point.
    """
    successor = np.roll(contour, -1, axis=1)
    gap = np.hypot(*(successor - contour))
    scale = np.max(np.abs(contour)) + 1.0
    keep = gap > np.finfo(np.float64).eps * scale
    return contour[:, keep]


def signed_area(contour: NDArray) -> float:
    r"""Return the signed area enclosed by a closed contour (shoelace formula).

    .. math::

        A = \frac{1}{2} \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i),
        \qquad (x_{n+1}, y_{n+1}) = (x_1, y_1).

    Parameters
    ----------
    contour : ndarray of shape (2, n_points)

    Returns
    -------
    area : float
        Positive if the points run counterclockwise, negative if clockwise.

    Examples
    --------
    >>> import numpy as np
    >>> from vampire.contours import signed_area
    >>> square = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    >>> signed_area(square)
    1.0
    >>> signed_area(square[:, ::-1])
    -1.0
    """
    x, y = _check_contour(contour)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_contour(contour: NDArray, n_points: int = 50) -> NDArray[np.float64]:
    """Resample a closed contour to ``n_points`` points equidistant in arc length.

    Fits a periodic interpolating cubic B-spline through the contour,
    parameterized by cumulative chord length, and evaluates it at ``n_points``
    positions evenly spaced in the arc length *of the spline* (measured
    numerically on a dense evaluation of the spline).

    Parameters
    ----------
    contour : ndarray of shape (2, n)
        Closed contour, first point not repeated at the end. Consecutive
        duplicate points (and a repeated closing point) are dropped before
        fitting; at least 3 distinct points must remain.
    n_points : int, default=50
        Number of output points, at least 3.

    Returns
    -------
    resampled : ndarray of shape (2, n_points)
        ``float64`` contour. Its first point is the input's first distinct
        point, the direction of travel is preserved, and the start point is not
        repeated at the end.

    Raises
    ------
    ValueError
        If the contour is malformed, has fewer than 3 distinct points, or
        ``n_points < 3``.

    See Also
    --------
    vampire.ContourResampler : Applies this function to a contour set.

    Examples
    --------
    >>> import numpy as np
    >>> from vampire.contours import resample_contour
    >>> square = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    >>> resampled = resample_contour(square, n_points=40)
    >>> resampled.shape
    (2, 40)
    >>> steps = np.hypot(*np.diff(resampled, axis=1, append=resampled[:, :1]))
    >>> bool(np.ptp(steps) < 1e-3 * steps.mean())  # equal spacing
    True
    """
    contour = _check_contour(contour)
    if not isinstance(n_points, int | np.integer) or n_points < 3:
        raise ValueError(f"n_points must be an integer >= 3; got {n_points!r}.")

    points = _drop_repeated_points(contour)
    if points.shape[1] < MIN_CONTOUR_POINTS:
        raise ValueError(
            f"contour has only {points.shape[1]} distinct points; at least "
            f"{MIN_CONTOUR_POINTS} are required to fit a closed spline."
        )

    # Periodic spline through the closed polygon, parameterized by chord length.
    closed = np.hstack([points, points[:, :1]])
    t = np.concatenate([[0.0], np.cumsum(_segment_lengths(closed))])
    spline = make_interp_spline(t, closed, k=3, bc_type="periodic", axis=1)

    # Chord length only approximates the spline's arc length; measure the
    # latter on a dense evaluation and invert it to get equidistant samples.
    n_dense = _ARC_LENGTH_OVERSAMPLING * max(n_points, points.shape[1])
    t_dense = np.linspace(0.0, t[-1], n_dense + 1)
    arc_dense = np.concatenate([[0.0], np.cumsum(_segment_lengths(spline(t_dense)))])
    arc_target = np.arange(n_points) * (arc_dense[-1] / n_points)
    t_target = np.interp(arc_target, arc_dense, t_dense)
    return np.ascontiguousarray(spline(t_target))


def register_contour(contour: NDArray) -> NDArray[np.float64]:
    r"""Remove translation, scale, and rotation from a contour.

    Steps:

    1. subtract the centroid (mean of the points);
    2. divide by the characteristic length
       :math:`R = \sqrt{\frac{1}{n}\sum_i (x_i^2 + y_i^2)}` [1]_;
    3. rotate onto the principal axes (SVD) so the major axis lies along x;
    4. reverse the point order if needed so the contour runs counterclockwise
       (positive signed area, see :func:`signed_area`), keeping the same
       points;
    5. cyclically shift so the first point is the one with the smallest
       absolute polar angle, i.e. closest to the positive major axis.

    Parameters
    ----------
    contour : ndarray of shape (2, n_points)
        Contour with at least 3 points, typically from :func:`resample_contour`.

    Returns
    -------
    registered : ndarray of shape (2, n_points)
        ``float64`` contour: centered, unit characteristic length,
        principal-axis-aligned, counterclockwise.

    Raises
    ------
    ValueError
        If the contour is malformed or all its points coincide.

    See Also
    --------
    vampire.ContourRegistrar : Applies this function to a contour set.

    Notes
    -----
    With the centered, scaled contour :math:`\mathbf{B}` (shape ``(2, n)``) and
    its SVD :math:`\mathbf{B} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T`, the
    columns of :math:`\mathbf{U}` are the principal directions and the
    registered contour is :math:`\mathbf{U}^T\mathbf{B}`.

    SVD determines each principal direction only up to sign. To make the
    result deterministic and a pure rotation of the input (never a mirror
    image), the signs are fixed as follows:

    - the major axis :math:`\mathbf{u}_1` points toward the heavier tail of the
      shape, i.e. the third moment :math:`\sum_i (\mathbf{u}_1^T\mathbf{b}_i)^3`
      is non-negative;
    - the minor axis is chosen so :math:`\det\mathbf{U} = +1`.

    Shapes with (near) zero third moment along the major axis -- e.g. ellipses
    -- may still come out rotated by 180 degrees; :func:`align_contour` resolves
    any remaining rotation. Unlike the pre-1.0 implementation, mirror images
    are not conflated.

    References
    ----------
    .. [1] Phillip, J. M., Han, K.-S., Chen, W.-C., Wirtz, D. & Wu, P.-H. A
       robust unsupervised machine-learning method to quantify the
       morphological heterogeneity of cells and nuclei. Nat Protoc 16,
       754–774 (2021). https://doi.org/10.1038/s41596-020-00432-x

    Examples
    --------
    >>> import numpy as np
    >>> from vampire.contours import register_contour
    >>> t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    >>> ellipse = np.vstack([5 - np.sin(t), 2 - 3 * np.cos(t)])  # tall, clockwise
    >>> registered = register_contour(ellipse)
    >>> np.allclose(registered.mean(axis=1), 0)
    True
    >>> float(np.sqrt(np.mean(np.sum(registered**2, axis=0))))
    1.0
    >>> bool(np.ptp(registered[0]) > np.ptp(registered[1]))  # major axis along x
    True
    """
    contour = _check_contour(contour)
    centered = contour - contour.mean(axis=1, keepdims=True)
    length_scale = np.sqrt(np.mean(np.sum(centered**2, axis=0)))  # R in the docs
    if length_scale <= np.finfo(np.float64).eps * (np.max(np.abs(contour)) + 1.0):
        raise ValueError("contour is degenerate: all its points coincide.")
    B = centered / length_scale

    U, _, _ = np.linalg.svd(B, full_matrices=False)
    if np.sum((U[:, 0] @ B) ** 3) < 0:
        U[:, 0] = -U[:, 0]
    if np.linalg.det(U) < 0:
        U[:, 1] = -U[:, 1]
    T = U.T @ B

    if signed_area(T) < 0:
        T = T[:, ::-1]
    start = np.argmin(np.abs(np.arctan2(T[1], T[0])))
    return np.roll(T, -start, axis=1)


def align_contour(contour: NDArray, reference: NDArray) -> NDArray[np.float64]:
    r"""Align a registered contour to a reference contour.

    Tries every cyclic shift of the contour's starting point, finds the optimal
    rotation to ``reference`` for each (Kabsch algorithm), and returns the
    shifted and rotated contour with the smallest sum of squared distances.

    Parameters
    ----------
    contour : ndarray of shape (2, n_points)
        Registered (centered) contour.
    reference : ndarray of shape (2, n_points)
        Reference (typically the mean registered contour).

    Returns
    -------
    aligned : ndarray of shape (2, n_points)
        ``float64`` contour: a cyclic shift of ``contour`` rotated by a proper
        rotation (no reflection).

    Raises
    ------
    ValueError
        If the two contours have different shapes or are malformed.

    See Also
    --------
    vampire.ContourAligner : Applies this function to a contour set.

    Notes
    -----
    For the contour :math:`\mathbf{A}_k` shifted to start at point :math:`k`
    and a rotation by angle :math:`\theta`, the squared distance to the
    reference :math:`\mathbf{B}` is

    .. math::

        \Vert \mathbf{R}_\theta \mathbf{A}_k - \mathbf{B} \Vert_F^2
        = \Vert \mathbf{A}_k \Vert_F^2 + \Vert \mathbf{B} \Vert_F^2
        - 2 (c_k \cos\theta + s_k \sin\theta),

    with :math:`c_k = \sum_i (a_{x,i} b_{x,i} + a_{y,i} b_{y,i})` and
    :math:`s_k = \sum_i (a_{x,i} b_{y,i} - a_{y,i} b_{x,i})`. Its minimum over
    :math:`\theta` is attained at :math:`\sqrt{c_k^2 + s_k^2}`, so the best
    shift maximizes :math:`\sqrt{c_k^2 + s_k^2}`; all shifts are scored at once.
    The rotation for the best shift is then computed with the Kabsch algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from vampire.contours import align_contour
    >>> t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    >>> reference = np.vstack([2 * np.cos(t), np.sin(t)])
    >>> c, s = np.cos(0.3), np.sin(0.3)
    >>> moved = np.roll(np.array([[c, -s], [s, c]]) @ reference, 5, axis=1)
    >>> np.allclose(align_contour(moved, reference), reference)
    True
    """
    A = _check_contour(contour)
    B = _check_contour(reference, name="reference")
    if A.shape != B.shape:
        raise ValueError(
            f"contour and reference must have the same shape; got {A.shape} "
            f"and {B.shape}. Resample both to the same number of points."
        )
    n = A.shape[1]
    # shifted[:, k] is the contour starting at point k, shape (2, n, n).
    shifted = A[:, (np.arange(n)[:, None] + np.arange(n)) % n]
    c = shifted[0] @ B[0] + shifted[1] @ B[1]
    s = shifted[0] @ B[1] - shifted[1] @ B[0]
    A_best = shifted[:, np.argmax(np.hypot(c, s))]
    return kabsch_rotation(A_best, B) @ A_best


def lowpass_contour(contour: NDArray, n_harmonics: int) -> NDArray[np.float64]:
    r"""Remove high-frequency detail from a closed contour (Fourier low-pass).

    The contour is treated as a periodic complex signal
    :math:`z_j = x_j + i\,y_j` and expanded in its discrete Fourier series
    (elliptic Fourier descriptors). Harmonics above ``n_harmonics`` are set to
    zero, and the contour is reconstructed at the same points. Harmonic 0 is
    the centroid, harmonic 1 the best-fitting ellipse; each further harmonic
    adds finer detail, so small values give smoother outlines.

    Points should be roughly equidistant along the contour, as produced by
    :func:`resample_contour` (and therefore by every fitted model attribute
    such as ``Vampire.mean_cluster_contours_``).

    Parameters
    ----------
    contour : ndarray of shape (2, n_points)
        Closed contour.
    n_harmonics : int
        Number of harmonics kept in each rotation direction (>= 1). Values of
        ``n_points // 2`` or more keep every harmonic and return the contour
        unchanged.

    Returns
    -------
    smoothed : ndarray of shape (2, n_points)
        Low-pass filtered contour with the same number of points and the same
        centroid.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    >>> noisy = np.vstack([np.cos(t), np.sin(t)]) * (1 + 0.05 * np.cos(20 * t))
    >>> smooth = lowpass_contour(noisy, n_harmonics=5)
    >>> smooth.shape
    (2, 64)
    >>> bool(np.allclose(np.hypot(*smooth), 1.0))  # the ripple is gone
    True
    """
    contour = _check_contour(contour)
    if (
        isinstance(n_harmonics, bool)
        or not isinstance(n_harmonics, int | np.integer)
        or n_harmonics < 1
    ):
        raise ValueError(f"n_harmonics must be an integer >= 1; got {n_harmonics!r}.")
    n = contour.shape[1]
    coefficients = np.fft.fft(contour[0] + 1j * contour[1])
    frequencies = np.fft.fftfreq(n, d=1.0 / n)  # integer harmonic numbers
    coefficients[np.abs(frequencies) > n_harmonics] = 0
    smoothed = np.fft.ifft(coefficients)
    return np.vstack([smoothed.real, smoothed.imag])
