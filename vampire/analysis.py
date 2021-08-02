import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import preprocessing
from sklearn.cluster import KMeans

from . import amath


def pca_contours(contours):
    """
    Return the principal component of the contours.

    Parameters
    ----------
    contours : ndarray
        Object contours, with shape (num_contour, 2*num_points).

    Returns
    -------
    principal_directions : ndarray
        Loadings, weights, principal directions, principal axes,
        eigenvector of covariance matrix of mean-subtracted contours,
        with shape (2*num_points, 2*num_points).
    principal_components : ndarray
        PC score, principal components, coordinates of mean-subtracted contours
        in their principal directions, with shape (num_contours, 2*num_points).

    See Also
    --------
    amath.pca : Implementation of principal component analysis.

    """
    principal_directions, principal_components, variance = amath.pca(contours, 'eig')
    return principal_directions, principal_components


def pca_transform_contours(contours, mean_contour, principal_directions):
    """
    Transform contour coordinates to principal directions in the PC space.

    Parameters
    ----------
    contours : ndarray
        Object contours, with shape (num_contour, 2*num_points).
    mean_contour : ndarray
        Mean contour used to mean-center object contours.
    principal_directions : ndarray
        Loadings, weights, principal directions, principal axes,
        eigenvector of covariance matrix of mean-subtracted contours,
        with shape (2*num_points, 2*num_points).

    Returns
    -------
    principal_components : ndarray
        PC score, principal components, coordinates of mean-subtracted contours
        in their principal directions, with shape (num_contours, 2*num_points).

    """
    mean_centered_contours = contours - mean_contour
    principal_components = mean_centered_contours @ principal_directions
    return principal_components


def cluster_contours(pc, contours, num_clusters=5, num_pc=20, random_state=None):  # random: None
    """
    K-means clustering of contour principal components.

    Parameters
    ----------
    pc : ndarray
        Principal components of contours.
    contours : ndarray
        Object contours, with shape (num_contour, 2*num_points).
    num_clusters : int, optional
        Number of clusters.
    num_pc : int, optional
        Number of principal components used for approximation.
    random_state : None or int, optional
        Random state for K-means clustering.

    Returns
    -------
    contours_df : DataFrame
        DataFrame of objects' contour coordinates with cluster id.
    centroids : ndarray
        Coordinates of cluster centers of K-means clusters.

    See Also
    --------
    sklearn.cluster.KMeans : Implementation of K-means clustering.

    """
    pc_truncated = pc[:, :num_pc]
    pc_truncated_normalized = preprocessing.normalize(pc_truncated)

    # k-means clustering of normalized principal coordinates
    k_means = KMeans(n_clusters=num_clusters,
                     random_state=random_state,
                     init='k-means++',
                     n_init=3,
                     max_iter=300).fit(pc_truncated_normalized)
    centroids = k_means.cluster_centers_
    # distance = spatial.distance.cdist(pc_truncated_normalized, centroid)  # D, why not this line?
    distance = spatial.distance.cdist(pc_truncated, centroids)
    cluster_id = np.argmin(distance, axis=1)

    # tag each object with cluster id
    contours_df = pd.DataFrame(contours)
    contours_df['cluster_id'] = cluster_id

    return contours_df, centroids


def assign_clusters_id(pc, contours, centroids, num_pc=20):
    """
    Assign the contours with id of the closest centroid.

    Parameters
    ----------
    pc : ndarray
        Principal components of contours.
    contours : ndarray
        Object contours, with shape (num_contour, 2*num_points).
    centroids : ndarray
        Coordinates of cluster centers of K-means clusters.
    num_pc : int, optional
        Number of principal components used for approximation.

    Returns
    -------
    centroids : ndarray
        Coordinates of cluster centers of K-means clusters.

    """
    # find closest centroid and get cluster id
    pc_truncated = pc[:, :num_pc]
    distance = spatial.distance.cdist(pc_truncated, centroids)
    cluster_id = np.argmin(distance, axis=1)

    # tag each object with cluster id
    contours_df = pd.DataFrame(contours)
    contours_df['cluster_id'] = cluster_id
    return contours_df
