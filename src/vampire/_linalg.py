"""Private linear-algebra helpers."""

import numpy as np
from numpy.typing import NDArray


def kabsch_rotation(A: NDArray[np.floating], B: NDArray[np.floating]) -> NDArray:
    r"""Return the proper rotation matrix ``R`` minimizing ``||R @ A - B||_F``.

    Parameters
    ----------
    A, B : ndarray of shape (2, n_points)
        Centered point sets with corresponding columns (float).

    Returns
    -------
    R : ndarray of shape (2, 2)
        Rotation matrix (float64) with ``det(R) == +1`` (reflections are
        excluded).

    Raises
    ------
    ValueError
        If ``A`` and ``B`` do not both have shape ``(2, n_points)``.

    Notes
    -----
    Kabsch algorithm [1]_, [2]_. With the cross-covariance matrix
    :math:`\mathbf{C} = \mathbf{A}\mathbf{B}^T` and its SVD
    :math:`\mathbf{C} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T`, the optimal
    orthogonal matrix is :math:`\mathbf{V}\mathbf{U}^T`. It may be a reflection
    (determinant -1); restricting to proper rotations gives

    .. math::

        \mathbf{R} = \mathbf{V}
        \begin{bmatrix} 1 & 0 \\ 0 & d \end{bmatrix}
        \mathbf{U}^T, \qquad
        d = \operatorname{sign}\det(\mathbf{V}\mathbf{U}^T).

    References
    ----------
    .. [1] Kabsch, W. A solution for the best rotation to relate two sets of
       vectors. Acta Cryst. A32, 922–923 (1976).
    .. [2] Dryden, I. L. & Mardia, K. V. Statistical Shape Analysis, with
       Applications in R, 2nd edn., Lemma 4.1 (2016).

    Examples
    --------
    >>> import numpy as np
    >>> from vampire._linalg import kabsch_rotation
    >>> A = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 2.0, -2.0]])
    >>> R_true = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90 degrees
    >>> R = kabsch_rotation(A, R_true @ A)
    >>> np.allclose(R, R_true)
    True
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != 2 or A.shape != B.shape:
        raise ValueError(
            f"A and B must both have shape (2, n_points); got {A.shape} and {B.shape}."
        )
    C = A @ B.T
    U, _, VT = np.linalg.svd(C)
    V = VT.T
    d = 1.0 if np.linalg.det(V @ U.T) >= 0 else -1.0
    R = V @ np.diag([1.0, d]) @ U.T
    return R
