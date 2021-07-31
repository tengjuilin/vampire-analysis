import numpy as np


def mean_center(A):
    r"""
    Mean center the matrix `A` by subtracting its mean.

    Parameters
    ----------
    A : ndarray
        Matrix with columns of features and rows of measurements.

    Returns
    -------
    B : ndarray
        Mean-centered matrix.

    Notes
    -----
    Suppose we have a matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}`
    with :math:`n` columns of features
    :math:`\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n` and :math:`m`
    rows of measurements:

    .. math::

       \mathbf{A} =
       \begin{bmatrix}
       | & | &  & | \\
       \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_n  \\
       | & | &  & | \\
       \end{bmatrix}.

    The means of the features are
    :math:`\bar{x}_1, \bar{x}_2, \dots, \bar{x}_n`, respectively, and they are
    stored in the matrix

    .. math::

       \mathbf{\bar{A}} =
       \begin{bmatrix}
       1 \\ 1 \\ \vdots \\ 1
       \end{bmatrix}
       \begin{bmatrix}
       \bar{x}_1 & \bar{x}_2 & \cdots & \bar{x}_n
       \end{bmatrix}.

    The mean-centered (mean-subtracted) data is then

    .. math::

       \mathbf{B = A - \bar{A}}.

    """
    A_bar = np.mean(A, axis=0)
    B = A - A_bar
    return B


def pca(A, method=None):
    r"""
    Principal component analysis of matrix `A`.

    Returns loadings, principal components, and explained variance.

    Parameters
    ----------
    A : ndarray
        Matrix with shape (m, n), where n features are in columns,
        and m measurements are in rows.
    method : None or str, optional
        Algorithm used to compute PCA:

        ``None``
            If m >= n, use eigen-decomposition algorithm.

            If m < n, use singular value decomposition algorithm.

        ``'eig'``
            Eigen-decomposition algorithm.

        ``'svd'``
            Singular value decomposition algorithm.

    Returns
    -------
    V : ndarray
        Loadings, weights, principal directions, principal axes,
        eigenvector of covariance matrix of mean-subtracted A,
        with shape (n, n).
    T : ndarray
        PC score, principal components, coordinates of mean-subtracted A
        in its principal directions, with shape (m, n).
    d : ndarray
        Explained variance, eigenvalues of covariance matrix of
        mean-subtracted A, with size n.

    See Also
    --------
    _pca_eig : Implementation of eigen-decomposition algorithm.
    _pca_svd : Implementation of singular value decomposition algorithm.
    sklearn.decomposition.PCA : Packaged implementation.

    """
    if method is None:
        m, n = A.shape
        if m >= n:
            return _pca_eig(A)
        else:
            return _pca_svd(A)
    elif method == 'eig':
        return _pca_eig(A)
    elif method == 'svd':
        return _pca_svd(A)
    else:
        raise ValueError(f'Unrecognized method {method}. \n'
                         'Expect method from one of {"svd", "eig"}')


def _pca_eig(A):
    r"""
    Principal component analysis of matrix `A` by eigen decomposition.

    Returns loadings, principal components, and explained variance.

    Parameters
    ----------
    A : ndarray
        Matrix with shape (m, n), where n features are in columns,
        and m measurements are in rows.

    Returns
    -------
    V : ndarray
        Loadings, weights, principal directions, principal axes,
        eigenvector of covariance matrix of mean-subtracted A,
        with shape (n, n).
    T : ndarray
        PC score, principal components, coordinates of mean-subtracted A
        in its principal directions, with shape (m, n).
    d : ndarray
        Explained variance, eigenvalues of covariance matrix of
        mean-subtracted A, with size n.

    See Also
    --------
    numpy.linalg.eigh, numpy.linalg.eig

    Notes
    -----
    Suppose we have a matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` with
    :math:`n` columns of features
    :math:`\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n` and :math:`m`
    rows of measurements:

    .. math::

       \mathbf{A} =
       \begin{bmatrix}
       | & | &  & | \\
       \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_n  \\
       | & | &  & | \\
       \end{bmatrix}.

    We can perform principal component analysis (PCA) [1]_ on the matrix
    using eigen-decomposition.

    .. rubric:: **Mean subtraction**

    We first calculate the mean of the features
    :math:`\bar{x}_1, \bar{x}_2, \dots, \bar{x}_n`, respectively, and stored
    them in the matrix

    .. math::

       \mathbf{\bar{A}} =
       \begin{bmatrix}
       1 \\ 1 \\ \vdots \\ 1
       \end{bmatrix}
       \begin{bmatrix}
       \bar{x}_1 & \bar{x}_2 & \cdots & \bar{x}_n
       \end{bmatrix}.

    We then calculate the mean-subtracted data

    .. math::

       \mathbf{B = A - \bar{A}}

    to make the data zero mean.

    .. rubric:: **Covariance matrix**

    The covariance matrix :math:`\mathbf{C}` of the rows of
    :math:`\mathbf{B}` is

    .. math::

       \mathbf{C} = \dfrac{1}{n-1} \mathbf{B}^T \mathbf{B}.

    The eigenvalue decomposition of the symmetric matrix :math:`\mathbf{C}`
    gives

    .. math::

       \mathbf{C} = \mathbf{V}\mathbf{D}\mathbf{V}^{-1},

    where :math:`\mathbf{V}` is an orthogonal matrix containing the
    eigenvectors, and :math:`\mathbf{D}` is a diagonal matrix containing the
    eigenvalues.

    .. rubric:: **Principal components**

    The principal components :math:`\mathbf{T}` is defined as

    .. math::

       \mathbf{T} \equiv \mathbf{BV},

    where :math:`\mathbf{V}` is called the loadings.

    References
    ----------
    .. [1] Brunton, S., & Kutz, J. (2019). Data-Driven Science and Engineering:
       Machine Learning, Dynamical Systems, and Control. Cambridge: Cambridge
       University Press. doi:10.1017/9781108380690

    """
    # A_bar = np.mean(A, axis=0)
    # B = A - A_bar
    B = mean_center(A)
    C = B.T @ B / (B.shape[0] - 1)
    d, V = np.linalg.eigh(C)  # d is diagonal entries of D
    # sort the eigenvalues and eigenvectors in descending order
    # the convention of `eigh()` gives them in ascending order
    sorting_index = np.arange(len(d))[::-1]
    d = d[sorting_index]
    V = V[:, sorting_index]
    T = B @ V
    return V, T, d


def _pca_svd(A):
    r"""
    Principal component analysis of matrix `A` by singular value decomposition.

    Returns loadings, principal components, and explained variance.

    Parameters
    ----------
    A : ndarray
        Matrix with shape (m, n), where n features are in columns,
        and m measurements are in rows.

    Returns
    -------
    V : ndarray
        Loadings, weights, principal directions, principal axes,
        eigenvector of covariance matrix of mean-subtracted A,
        with shape (n, n).
    T : ndarray
        PC score, principal components, coordinates of mean-subtracted A
        in its principal directions, with shape (m, n).
    d : ndarray
        Explained variance, eigenvalues of covariance matrix of
        mean-subtracted A, with size n.

    See Also
    --------
    numpy.linalg.svd

    Notes
    -----
    Suppose we have a matrix
    :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` with
    :math:`n` columns of features
    :math:`\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n` and :math:`m`
    rows of measurements:

    .. math::

       \mathbf{A} =
       \begin{bmatrix}
       | & | &  & | \\
       \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_n  \\
       | & | &  & | \\
       \end{bmatrix}.

    We can perform principal component analysis (PCA) [1]_ on the matrix
    using singular value decomposition (SVD).

    .. rubric:: **Mean subtraction**

    We first calculate the mean of the features
    :math:`\bar{x}_1, \bar{x}_2, \dots, \bar{x}_n`, respectively, and stored
    them in the matrix

    .. math::

       \mathbf{\bar{A}} =
       \begin{bmatrix}
       1 \\ 1 \\ \vdots \\ 1
       \end{bmatrix}
       \begin{bmatrix}
       \bar{x}_1 & \bar{x}_2 & \cdots & \bar{x}_n
       \end{bmatrix}.

    We then calculate the mean-subtracted data

    .. math::

       \mathbf{B = A - \bar{A}}

    to make the data zero mean.

    .. rubric:: **Singular value decomposition**

    We compute the SVD of :math:`\mathbf{B}`:

    .. math::

        \mathbf{B} = \mathbf{U \Sigma V}^T.

    Multiply :math:`\mathbf{V}` at the right on both sides, we get the
    principal components

    .. math::

        \mathbf{T \equiv BV = U\Sigma},

    where :math:`\mathbf{V}` is the loading. The explained variance matrix
    :math:`\mathbf{D}` is related to :math:`\mathbf{\Sigma}` by

    .. math::

        \mathbf{D} = \dfrac{1}{n-1}\mathbf{\Sigma}^2.

    References
    ----------
    .. [1] Brunton, S., & Kutz, J. (2019). Data-Driven Science and Engineering:
       Machine Learning, Dynamical Systems, and Control. Cambridge: Cambridge
       University Press. doi:10.1017/9781108380690

    """
    n = len(A)
    # A_bar = np.mean(A, axis=0)
    # B = A - A_bar
    B = mean_center(A)
    U, s, VT = np.linalg.svd(B, full_matrices=False)
    V = VT.T
    T = U @ np.diag(s)
    d = s ** 2 / (n - 1)
    return V, T, d


def get_rotation_matrix(A, B):
    r"""
    Returns optimal rotation matrix to align 2D coordinates `A` to `B`.

    Parameters
    ----------
    A : ndarray
        Matrix to be rotated, with shape (2, n).
    B : ndarray
        Matrix to be aligned to, with shape (2, n).

    Returns
    -------
    R : ndarray
        Optimal rotation matrix to be applied to `A`, with shape (2, 2).

    See Also
    --------
    scipy.spatial.transform.Rotation.align_vectors : Aligns 3D coordinates.

    Notes
    -----
    We want to align 2D coordinates of object A to that of object B,
    represented by the matrices

    .. math::

       \mathbf{A} =
       \begin{bmatrix}
       — & \mathbf{x}_A & — \\
       — & \mathbf{y}_A  & — \\
       \end{bmatrix},
       \mathbf{B} =
       \begin{bmatrix}
       — & \mathbf{x}_B & — \\
       — & \mathbf{y}_B  & — \\
       \end{bmatrix},

    respectively, where :math:`\mathbf{x}_A, \mathbf{x}_B` are the
    x-coordinates, and :math:`\mathbf{y}_A, \mathbf{y}_B` are the
    y-coordinates.

    It is equivalent to finding the optimal rotation matrix
    :math:`\mathbf{R}` such that :math:`\mathbf{A}` after rotation has
    the minimum sum of squared distance loss :math:`L(\mathbf{R})`
    with :math:`\mathbf{B}`:

    .. math::
       L(\mathbf{R})  = \Vert \mathbf{RA} - \mathbf{B} \Vert_2^2.

    .. rubric:: **Kabsch algorithm**

    The optimal rotation matrix can be found using the Kabsch algorithm [1]_,
    [2]_, [3]_:

    1. Compute the covariance matrix

    .. math::
       \mathbf{C = AB}^T

    2. Compute the SVD of the covariance matrix

    .. math::
       \mathbf{C = U\Sigma V}^T

    3. The optimal rotation matrix is

    .. math::
       \mathbf{R = VU}^T

    References
    ----------

    .. [1] Lydia E. Kavraki, Molecular Distance Measures. OpenStax CNX. (2007)
       http://cnx.org/contents/1d5f91b1-dc0b-44ff-8b4d-8809313588f2@23

       *  The section "Optimal Alignment for lRMSD Using Rotation Matrices"
          describes and proves the Kabsch algorithm.

    .. [2] Dryden, I. L. & Mardia, K. V. Statistical Shape Analysis, with
       Applications in R 2nd edn. https://doi.org/10.1002/9781119072492
       (2016).

       *  Section 4.1.1 "Procrustes distances" describes and proves the
          Kabsch algorithm with Lemma 4.1.

    .. [3] Wu, PH., Phillip, J., Khatau, S. et al. Evolution of cellular
       morpho-phenotypes in cancer metastasis. Sci Rep 5, 18437 (2016).
       https://doi.org/10.1038/srep18437

       *  Supplemental information section "Decomposition of 2-dimensional
          shape and identification of shape mode" describes the implementation
          of the Kabsch algorithm in the context of VAMPIRE methodology.

    """
    C = A @ B.T
    U, s, VT = np.linalg.svd(C)
    V = VT.T
    R = V @ U.T
    return R
