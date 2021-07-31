import numpy as np
from scipy import interpolate

from vampire import amath


def sample_contour(contour, num_sample_points):
    """
    Returns sample points of contour using B-spline.

    Interpolate given coordinates of an object contour using B-spline,
    then fit `num_sample_points` equidistant points to the B-spline.

    Parameters
    ----------
    contour : ndarray
        x and y coordinates of object contour, with shape (2, n).
    num_sample_points : int
        Number of sample points after resample.

    Returns
    -------
    sampled_contour : ndarray
        `num_sample_points` equidistant contour sample points,
        with shape (2, num_sample_points).

    """
    x, y = contour
    # check if object shape is closed
    if np.all(contour[:, 0] != contour[:, -1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    distance = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    # pad arbitrary number to give same length as x and y
    # can be arbitrary since cumulative sum is taken, identical result
    distance = np.append([1], distance)
    cum_distance = np.cumsum(distance)
    # TODO: testing, use truncated points, why??
    sample_points = np.linspace(cum_distance[0], cum_distance[-1], num_sample_points+1)[:-1]
    # sample_points = np.linspace(cum_distance[0], cum_distance[-1], num_sample_points)  # my soln
    # interpolate the data points using b-spline
    x_spliner = interpolate.splrep(cum_distance, x, s=0)
    y_spliner = interpolate.splrep(cum_distance, y, s=0)
    # fit points to the b-spline
    x_samples = interpolate.splev(sample_points, x_spliner)
    y_samples = interpolate.splev(sample_points, y_spliner)
    sampled_contour = np.vstack([x_samples, y_samples])
    return sampled_contour


def sample_contours(contours, num_points=50):
    """
    Returns sampled contours using B-spline.

    Parameters
    ----------
    contours : list[ndarray]
        List of contour coordinates, list with length num_contours,
        ndarray with shape (2, num_points).
    num_points : int, optional
        Number of sample points of object contour. Defaults to 50.

    Returns
    -------
    sampled_contours : list[ndarray]
        Sampled contours, list with length num_contours, ndarray with
        shape (2, num_points).

    See Also
    --------
    sample_contour

    """
    num_contours = len(contours)
    sampled_contours = []
    for i in range(num_contours):
        sampled_contour = sample_contour(contours[i], num_points)
        sampled_contours.append(sampled_contour)
    return sampled_contours


# noinspection PyPep8Naming
def register_contour(contour):
    r"""
    Returns registered contour to its principal component.

    Register given `contour` by mean-subtraction (moving center to origin),
    normalization by characteristic length scale, and making the contour
    positively oriented.

    Parameters
    ----------
    contour : ndarray
        x and y coordinates of object contour, with shape (2, n).

    Returns
    -------
    registered_contour : ndarray
        x and y coordinates of registered contour, with shape (2, n).

    Notes
    -----
    Suppose we have :math:`N` pairs of :math:`x` and :math:`y` coordinates
    of an object contour stored in column vectors :math:`\mathbf{x}`
    and :math:`\mathbf{y}`. We define the coordinate matrix to be

    .. math::

        \mathbf{A} =
        \begin{bmatrix}
        | & | \\
        \mathbf{x} & \mathbf{y} \\
        | & | \\
        \end{bmatrix}.

    .. rubric:: **Mean subtraction**

    We first calculate the mean of :math:`x` and :math:`y` coordinates
    :math:`\bar{x}` and :math:`\bar{y}`, respectively, and stored them
    in the matrix

    .. math::

        \mathbf{\bar{A}} =
        \begin{bmatrix}
        1 \\ 1 \\ \vdots \\ 1
        \end{bmatrix}
        \begin{bmatrix}
        \bar{x} & \bar{y}
        \end{bmatrix}.

    We then calculate the mean-subtracted data

    .. math::

        \mathbf{B = A - \bar{A}}

    to shift the center of the coordinates to :math:`(0, 0)`.

    .. rubric:: **Normalization**

    We then normalize the mean-subtracted data by the characteristic
    length scale [1]_ defined by

    .. math::

        R = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(x_i^2 + y_i^2)},

    getting the normalized contour

    .. math::

        \mathbf{B' = B} / R,

    where the division is element-wize division.

    .. rubric:: **Singular value decomposition (SVD)**

    SVD [2]_ is used to rotate the contours to eliminate rotational variations.
    The SVD of the mean-centered normalized contour is given by

    .. math::

        \mathbf{B' = U \Sigma V^T}.

    Multiply :math:`\mathbf{V}` at the right, we get the principal components

    .. math::

        \mathbf{T \equiv B'V = U\Sigma},

    where :math:`\mathbf{V}` contains the principal directions and acts as
    a rotation matrix. The principal components has maximum variance across
    the x-axis, eliminating rotational variability.

    .. rubric:: **Re-orientate the data points**

    Although the contour data points are ordered so that a closed shape can
    be drawn by connecting each point, the starting data point could start
    at random location of the shape and proceed in either clockwise or
    counterclockwise directions. Here, we reorient the data points so that
    the contour has *positive orientation*. Meaning, the data points starts
    at the point that makes the smallest angle with the major axis at the
    right side of the shape, and the data points goes in counterclockwise
    direction.

    We first reorder the data points by finding the appropriate starting
    point. The major axis in this case is :math:`x = 0` since the data is
    mean-subtracted. The starting data point that makes the smallest angle
    with the major axis at the right side will have minimum absolute angle
    defined by polar coordinates. The angles of points with respect to the
    origin is given by

    .. math::

        \theta_i = \arctan\left(\dfrac{y_i}{x_i}\right),

    and the index of point with smallest angle is

    .. math::

        \mathrm{argmin}_{i} \vert\theta_i\vert.

    We then check if the contour goes in the counterclockwise direction.
    Assuming the object shape is locally convex, counterclockwise contours
    satisfies

    .. math::

        \theta_0 < \theta_1,

    where :math:`\theta_0`$` is the starting data point, and :math:`\theta_1`
    is the next data point in sequence.

    References
    ----------
    .. [1] Phillip, J.M., Han, KS., Chen, WC. et al. A robust unsupervised
       machine-learning method to quantify the morphological heterogeneity of
       cells and nuclei. Nat Protoc 16, 754–774 (2021).
       https://doi.org/10.1038/s41596-020-00432-x

    .. [2] Brunton, S., & Kutz, J. (2019). Data-Driven Science and Engineering:
       Machine Learning, Dynamical Systems, and Control. Cambridge: Cambridge
       University Press. doi:10.1017/9781108380690

    """
    # mean-subtracted contour coordinates
    A = contour.T
    B = amath.mean_center(A)
    # normalization by characteristic length scale
    N = B.shape[0]
    R = np.sqrt(np.sum(B**2) / N)
    B_prime = B / R
    # principal component analysis
    V, T, d = amath.pca(B_prime, method='svd')
    # let data point start at the right close to the major axis
    theta = np.arctan2(T[:, 1], T[:, 0])
    starting_index = np.argmin(np.abs(theta))
    reorder_index = np.hstack([np.arange(starting_index, N), np.arange(starting_index)])
    T = T[reorder_index, :]
    theta = theta[reorder_index]
    # make contour positively oriented
    if theta[0] > theta[4]:  # assume locally convex, 4 is arbitrary choice
        T = np.flip(T, axis=0)
    registered_contour = T.T
    return registered_contour


def register_contours(contours):
    """
    Returns registered contours to their principal components.

    Parameters
    ----------
    contours : list[ndarray]
        List of contour coordinates, list with length num_contours,
        ndarray with shape (2, num_points).

    Returns
    -------
    registered_contours : list[ndarray]
        Registered contours, list with length num_contours, ndarray with
        shape (2, num_points).
    mean_contour : ndarray
        Mean of registered contours, with shape (2, num_points).

    See Also
    --------
    register_contour

    """
    num_contours = len(contours)
    registered_contours = []
    for i in range(num_contours):
        registered_contour = register_contour(contours[i])
        registered_contours.append(registered_contour)
    registered_contours_flat = np.asarray(registered_contours).reshape(num_contours, -1)
    mean_contour = np.mean(registered_contours_flat, axis=0).reshape(2, -1)
    return registered_contours, mean_contour


# noinspection PyPep8Naming
def align_contour(contour, mean_contour):
    r"""
    Aligns one contour to the mean of the set of contours.

    Parameters
    ----------
    contour : ndarray
         x and y coordinates of object contour, with shape (2, n).
    mean_contour : ndarray
        Mean contour coordinates of the set of contours, with shape (2, n).

    Returns
    -------
    aligned_contour : ndarray
        Aligned contour closest to the mean contour, with shape (2, n).

    See Also
    --------
    amath.get_rotation_matrix : Find rotation matrix by Kabsch algorithm.

    Notes
    -----
    .. rubric:: **Defining the contours**

    Suppose we have :math:`N` pairs of mean-subtracted :math:`x` and :math:`y`
    coordinates of the :math:`i` th object contour stored in row vectors
    :math:`\mathbf{x}` and :math:`\mathbf{y}`. We define the coordinate matrix

    .. math::

        \mathbf{A}_i =
        \begin{bmatrix}
        — & \mathbf{x}_i & — \\
        — & \mathbf{y}_i  & — \\
        \end{bmatrix}.

    The average object contour have coordinates :math:`\mathbf{\bar{x}}` and
    :math:`\mathbf{\bar{y}}`, obtained from averaging the coordinates of
    corresponding points from each object. We define the average coordinate
    matrix

    .. math::

        \mathbf{\bar{A}} =
        \begin{bmatrix}
        — & \mathbf{\bar{x}} & — \\
        — & \mathbf{\bar{y}}  & — \\
        \end{bmatrix}.

    .. rubric:: **Finding the optimal rotation matrix**

    We can find the optimal rotation matrix :math:`\mathbf{R}` using the
    Kabsch algorithm (implemented in ``get_rotation_matrix``).

    .. rubric:: **Applying the optimal rotation matrix**

    The rotated contour :math:`\mathbf{A}_i'` aligned with
    the average object contour is then

    .. math::

        \mathbf{A}_i' = \mathbf{RA}_i

    """
    A = contour
    A_bar = mean_contour
    N = contour.shape[1]
    SSD_best = np.linalg.norm(A - A_bar) ** 2

    best_index = 0
    for i in range(N):
        starting_index = i
        reorder_index = np.hstack([np.arange(starting_index, N), np.arange(starting_index)])
        A_i = A[:, reorder_index]
        R = amath.get_rotation_matrix(A_i, A_bar)
        A_prime = R @ A_i
        SSD_current = np.linalg.norm(A_prime - A_bar) ** 2
        if SSD_current < SSD_best:
            best_index = i
            SSD_best = SSD_current

    reorder_index = np.hstack([np.arange(best_index, N), np.arange(best_index)])
    A_i = A[:, reorder_index]
    R = amath.get_rotation_matrix(A_i, A_bar)
    A_prime = R @ A_i
    aligned_contour = A_prime
    return aligned_contour


def align_contours(contours, mean_contour):
    """
    Returns aligned contours to their mean.

    Parameters
    ----------
    contours : list[ndarray]
        List of contour coordinates. List with length num_contours;
        ndarray with shape (2, num_points).
    mean_contour : ndarray
        Mean of registered contours, with shape (2, num_points).

    Returns
    -------
    aligned_contours_flat : ndarray
        Flattened aligned contours, with shape (num_contours, 2*num_points).

    See Also
    --------
    align_contour

    """
    num_contours = len(contours)
    aligned_contours_flat = []
    for j in range(num_contours):
        aligned_contour = align_contour(contours[j], mean_contour)
        aligned_contours_flat.append(aligned_contour.reshape(-1))
    aligned_contours_flat = np.asarray(aligned_contours_flat)
    mean_contour_flat = np.mean(aligned_contours_flat, axis=0)
    return aligned_contours_flat, mean_contour_flat


# def process_contours(contours, num_points=50):
#     """
#     Returns processed contours ready for PCA.
#
#     The contours are sampled, registered, and aligned to produce processed
#     contours.
#
#     Parameters
#     ----------
#     contours : list[ndarray]
#         List of contour coordinates.
#     num_points : int, optional
#         Number of sample points of object contour. Defaults to 50.
#
#     Returns
#     -------
#     aligned_contours_flat : ndarray
#         Flattened contour coordinates of all objects.
#         With shape (num_contours, 2*num_points).
#
#     """
#     num_contours = len(contours)
#     processed_contours = []
#     processed_contours_flat = np.zeros([num_contours, 2*num_points])
#
#     # sample and register contour, then calculate mean contour
#     for i in range(num_contours):
#         processed_contour = sample_contour(contours[i], num_points)
#         # plt.plot(*contours[i], '.-', alpha=0.5)
#         # plt.plot(*processed_contour, '.-', alpha=0.5)
#         # plt.axis('equal')
#         processed_contour = register_contour(processed_contour)
#         # plt.plot(*processed_contour, 'k.-', alpha=0.1)
#         # plt.plot(processed_contour[0][0], processed_contour[1][0], 'ro', alpha=0.5)
#         # plt.plot(processed_contour[0][-1], processed_contour[1][-1], 'go', alpha=0.5)
#         processed_contours.append(processed_contour)
#         processed_contours_flat[i] = processed_contour.reshape(-1)
#     mean_contour = np.mean(processed_contours_flat, axis=0).reshape(2, -1)
#     # plt.plot(*mean_contour, 'bo-')
#     # plt.axis('equal')
#
#     # align contours based on mean contour
#     aligned_contours_flat = []
#     for j in range(num_contours):
#         aligned_contour = align_contour(processed_contours[j], mean_contour)
#         aligned_contours_flat.append(aligned_contour.reshape(-1))
#     aligned_contours_flat = np.asarray(aligned_contours_flat)
#
#     # testing visualization
#     # for contour in aligned_contours_flat:
#     #     plt.plot(contour[:50], contour[50:], 'k.-', alpha=0.1)
#     # plt.plot(*mean_contour, 'bo-')
#     # plt.axis('equal')
#     return aligned_contours_flat


# def test():
#     contours = []
#     for i in range(len(B)):
#         contours.append(B[i].T)
#     import mod_processing
#     mod_processing.sample_contour(contours[0], N)
#     mod_processing.register_contour(bdt)
#
#     contours = []
#     for i in range(len(bstack[0])):
#         contours.append(bstack[0][i].T)
#     import mod_processing, mod_analysis, mod_plot
#     contours = mod_processing.process_contours(contours, num_points=50)
#     principal_coord = mod_analysis.pca_contours(contours)
#     contours_df = mod_analysis.cluster_contours(principal_coord, contours,
#                                                 num_clusters=5,
#                                                 num_pc=20, random_state=1)
#     mod_plot.custom_plot_settings()
#     mod_plot.plot_full(contours_df)
#     return
