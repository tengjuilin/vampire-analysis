import numpy as np
import pandas as pd
from scipy import cluster, spatial
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
    vampire.amath.pca : Implementation of principal component analysis.

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


def cluster_contours(pc, num_clusters=5, num_pc=20, random_state=None):
    """
    K-means clustering of contour principal components.

    Parameters
    ----------
    pc : ndarray
        Principal components of contours.
    num_clusters : int, optional
        Number of clusters.
    num_pc : int, optional
        Number of principal components used for approximation.
    random_state : None or int, optional
        Random state for K-means clustering.

    Returns
    -------
    cluster_id_df : DataFrame
        DataFrame of objects' cluster id and min distance to centroid.
    centroids : ndarray
        Coordinates of cluster centers of K-means clusters.
    inertia : float
        Sum of squared distances of samples to their closest cluster center.

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
    inertia = k_means.inertia_
    distance = spatial.distance.cdist(pc_truncated_normalized, centroids)
    cluster_id = np.argmin(distance, axis=1)
    min_distance = np.min(distance, axis=1)

    # tag each object with cluster id
    cluster_id_df = pd.DataFrame({'cluster_id': cluster_id,
                                  'distance_to_centroid': min_distance})
    return cluster_id_df, centroids, inertia


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
    contours_df : DataFrame
        DataFrame of objects' contour coordinates, cluster id,
        and min distance from centroid.

    """
    # find closest centroid and get cluster id
    pc_truncated = pc[:, :num_pc]
    # Original VAMPIRE GUI software did not normalize when
    # assigning clusters. However, it is logical to keep the
    # input of clustering and classifying consistent, so that
    # the same data used in clustering and assign cluster give
    # the same result.
    pc_truncated = preprocessing.normalize(pc_truncated)

    distance = spatial.distance.cdist(pc_truncated, centroids)
    cluster_id = np.argmin(distance, axis=1)
    min_distance = np.min(distance, axis=1)

    # tag each object with cluster id
    normalized_contours = {'normalized_contour': list(contours)}
    contours_df = pd.DataFrame(normalized_contours)
    contours_df['cluster_id'] = cluster_id
    contours_df['distance_to_centroid'] = min_distance
    return contours_df


def get_labeled_contours_df(contours, cluster_id_df):
    """
    Return contour coordinates, cluster id, and distance to centroid.

    Parameters
    ----------
    contours : ndarray
        Object contours, with shape (num_contour, 2*num_points).
    cluster_id_df : DataFrame
        DataFrame of objects' cluster id and min distance to centroid.

    Returns
    -------
    labeled_contours_df : DaraFrame
        DataFrame of contour coordinates, cluster id, and min
        distance to centroid.

    """
    return pd.DataFrame(contours).join(cluster_id_df)


def get_mean_cluster_contours(labeled_contours_df):
    """
    Return mean contour of each cluster.

    Parameters
    ----------
    labeled_contours_df : DaraFrame
        DataFrame of contour coordinates, cluster id, and min
        distance to centroid.

    Returns
    -------
    mean_cluster_contours : ndarray
        Mean contour of each cluster.

    """
    return labeled_contours_df.drop(['distance_to_centroid'], axis=1) \
        .groupby('cluster_id') \
        .mean().values


def hierarchical_cluster_contour(labeled_contours_df):
    """
    Compute data structure for rendering dendrogram.

    Parameters
    ----------
    labeled_contours_df : DaraFrame
        DataFrame of contour coordinates, cluster id, and min
        distance to centroid.

    Returns
    -------
    pair_distance : ndarray
        Pairwise distance of mean cluster contour.
        Result of `scipy.spatial.distance.pdist`.
    linkage_matrix : ndarray
        Linkage matrix for dendrogram.
        Result of `scipy.cluster.hierarchy.linkage`.
    branches : dict
        A dictionary of data structures computed to render the dendrogram.
        Result of `scipy.cluster.hierarchy.dendrogram`.

    See Also
    --------
    scipy.spatial.distance.pdist
    scipy.cluster.hierarchy.linkage
    scipy.cluster.hierarchy.dendrogram

    """
    mean_cluster_contours = get_mean_cluster_contours(labeled_contours_df)
    pair_distance = spatial.distance.pdist(mean_cluster_contours, 'euclidean')
    linkage_matrix = cluster.hierarchy.linkage(pair_distance, method='complete')
    branches = cluster.hierarchy.dendrogram(linkage_matrix,
                                            p=0,
                                            truncate_mode='lastp',
                                            orientation='bottom',
                                            above_threshold_color='k')
    return pair_distance, linkage_matrix, branches


def get_cluster_order(branches):
    """
    Get the cluster id of contours in order of dendrogram.

    Parameters
    ----------
    branches : dict
        Output of ``scipy.cluster.hierarchy.dendrogram``.

    Returns
    -------
    object_index
        The cluster id of contours in order of dendrogram.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram

    """
    object_index = np.array(branches['ivl'], dtype=int)
    return object_index


def get_distribution(properties_df):
    """
    Return proportion of each cluster.

    Parameters
    ----------
    properties_df : DataFrame
        DataFrame containing column `cluster_id`.

    Returns
    -------
    distribution : ndarray
        Proportion of each cluster.

    """
    cluster_id = properties_df['cluster_id'].values
    unique, counts = np.unique(cluster_id, return_counts=True)
    distribution = counts / np.sum(counts)
    return distribution


def get_shannon_entropy(distribution):
    r"""
    Calculate Shannon entropy from distribution (probability)
    of each shape mode.

    Parameters
    ----------
    distribution : ndarray
        Distribution of shape modes.

    Returns
    -------
    entropy : float
        Shannon entropy.

    See Also
    --------
    vampire.analysis.get_distribution

    Notes
    -----
    Shannon entropy here is defined as

    .. math::

        S = - \sum p_i \ln (p_i)

    where :math:`p_i` is probability of cells in each
    shape mode.

    """
    entropy = -np.sum(distribution * np.log(distribution))
    return entropy


def reorder_clusters(cluster_id, object_index):
    """
    Reorder cluster id according to dendrogram order.

    Parameters
    ----------
    cluster_id : ndarray
        Cluster ids.
    object_index : ndarray
        How original cluster id correspond to new id.

    Returns
    -------
    cluster_id_sorted : ndarray
        Reordered cluster id.

    """
    cluster_id_sorted = np.zeros_like(cluster_id)
    for i in range(len(object_index)):
        cluster_id_sorted[cluster_id == object_index[i]] = i
    return cluster_id_sorted


def reorder_centroids(centroids, object_index):
    """
    Reorder centroids according to dendrogram order.

    Parameters
    ----------
    centroids : ndarray
        Centroids
    object_index : ndarray
        How original cluster id correspond to new id.

    Returns
    -------
    reordered_centroids : ndarray
        Reordered centroids.

    """
    return centroids[object_index, :]
