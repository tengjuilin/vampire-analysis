import numpy as np

from . import analysis, processing, amath


class Vampire:
    """
    Visually Aided Morpho-Phenotyping Image Recognition (VAMPIRE) model.

    Attributes
    ----------
    model_name : str
        Name of the VAMPIRE model.
    num_points : int
        Number of sample points in a contour.
    num_coord : int
        Number of coordinates in a contour.
        num_coord = 2*num_points
    num_clusters : int
        Number of clusters for K-Means clustering.
    num_pc : int
        Number of principal components kept for analysis.
    random_state : int
        Random state of random processes.
    mean_registered_contour : ndarray
        Mean registered contour.
    mean_aligned_contour : ndarray
        Mean aligned contour.
    contours : ndarray
        Raw contours.
    principal_directions : ndarray
        Principal directions of PCA.
    cluster_id_df : DataFrame
        Contour's cluster id and min distance to centroid.
    labeled_contours_df : DataFrame
        Contour coordinates, cluster id, and min distance to centroid.
    centroids : ndarray
        Coordinate of centroids.
    mean_cluster_contours : ndarray
        Mean contours of each contour cluster.
    pair_distance : ndarray
        Pair distance between each cluster
    linkage_matrix : ndarray
        Linkage matrix for cluster dendrogram.
    branches : dict
        A dictionary of data structures computed to render the dendrogram.

    """
    def __init__(self,
                 model_name,
                 num_points=50,
                 num_clusters=5,
                 num_pc=None,
                 random_state=None):
        """
        Initialize VAMPIRE model wih hyperparameters.

        Parameters
        ----------
        model_name : str
            Name of the VAMPIRE model.
        num_points : int, optional
            Number of sample points for contour.
        num_clusters : int, optional
            Number of cluster for K-Means clustering.
        num_pc : int, optional
            Number of principal components to keep in analysis.
        random_state : int or None, optional
            Determines random number generation for K-Means clustering.
            Use an int to make the randomness deterministic.

        """
        # model hyperparameters
        self.model_name = model_name
        self.num_points = num_points
        self.num_coord = num_points * 2
        self.num_clusters = num_clusters
        self.num_pc = num_pc
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.default_rng().integers(100000)
        # contour info
        self.mean_registered_contour = None
        self.mean_aligned_contour = None
        self.contours = None
        # pca analysis info
        self.principal_directions = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.cum_explained_variance_ratio = None
        # k-means clustering info
        self.cluster_id_df = None
        self.labeled_contours_df = None
        self.centroids = None
        self.inertia = None
        self.mean_cluster_contours = None
        # hierarchical clustering info
        self.pair_distance = None
        self.linkage_matrix = None
        self.branches = None

    def build(self, properties_df):
        """
        Builds VAMPIRE model from dataset.

        Samples, registers, and aligns contour coordinates.
        PCA contour coordinates, then K-Means analysis.
        Hierarchical clustering determines cluster order.

        Parameters
        ----------
        properties_df : DataFrame
            DataFrame containing contour properties and raw contours.

        Returns
        -------
        self : vampire.model.Vampire
            Built VAMPIRE model.

        """
        # process contours
        contours = list(properties_df['raw_contour'])
        contours = processing.sample_contours(contours, num_points=self.num_points)
        contours = processing.register_contours(contours)
        self.mean_registered_contour = processing.get_mean_registered_contour(contours)
        self.contours = processing.align_contours(contours, self.mean_registered_contour)
        self.mean_aligned_contour = processing.get_mean_aligned_contour(self.contours)

        # pca contours
        (self.principal_directions,
            principal_components,
            self.explained_variance) = amath.pca(self.contours, 'eig')
        self.explained_variance_ratio = analysis.get_explained_variance_ratio(self.explained_variance)
        self.cum_explained_variance_ratio = analysis.get_cum_explained_variance_ratio(self.explained_variance_ratio)
        if self.num_pc is None:
            self.num_pc = analysis.get_optimal_num_pc(self.cum_explained_variance_ratio)

        # cluster contours
        (self.cluster_id_df,
            centroids,
            self.inertia) = analysis.cluster_contours(principal_components,
                                                      num_clusters=self.num_clusters,
                                                      num_pc=self.num_pc,
                                                      random_state=self.random_state)
        self.labeled_contours_df = analysis.get_labeled_contours_df(self.contours, self.cluster_id_df)
        (self.pair_distance,
            self.linkage_matrix,
            self.branches) = analysis.hierarchical_cluster_contour(self.labeled_contours_df)

        # reorder clusters and centroid according to dendrogram
        # to be consistent with dendrogram visualization
        object_index = analysis.get_cluster_order(self.branches)
        cluster_id_sorted = analysis.reorder_clusters(self.cluster_id_df['cluster_id'], object_index)
        self.cluster_id_df['cluster_id'] = cluster_id_sorted
        self.labeled_contours_df['cluster_id'] = cluster_id_sorted
        self.mean_cluster_contours = analysis.get_mean_cluster_contours(self.labeled_contours_df)
        self.centroids = analysis.reorder_centroids(centroids, object_index)
        return self

    def apply(self, properties_df):
        """
        Applies built VAMPIRE model on dataset.

        Parameters
        ----------
        properties_df : DataFrame
            DataFrame containing contour properties and raw contours.

        Returns
        -------
        properties_df : DataFrame
            DataFrame containing contour properties, raw contours,
            normalized contours, cluster id, and min distance from
            centroid.

        """
        contours = list(properties_df['raw_contour'])
        contours = processing.sample_contours(contours,
                                              num_points=self.num_points)
        contours = processing.register_contours(contours)
        contours = processing.align_contours(contours, self.mean_registered_contour)

        principal_components = analysis.pca_transform_contours(contours,
                                                               self.mean_aligned_contour,
                                                               self.principal_directions)
        apply_contours_df = analysis.assign_clusters_id(principal_components,
                                                        contours,
                                                        self.centroids,
                                                        num_pc=self.num_pc)
        properties_df = properties_df.join(apply_contours_df)
        return properties_df

    def __eq__(self, other):
        """
        Test equality with another Vampire object.

        Parameters
        ----------
        other : vampire.model.Vampire
            Vampire object to be compared to `self`.

        Returns
        -------
        equal : bool

        """
        equal = (
            # model hyperparameters
            self.model_name == other.model_name
            and self.num_points == other.num_points
            and self.num_coord == other.num_coord
            and self.num_clusters == other.num_clusters
            and self.num_pc == other.num_pc
            and self.random_state == other.random_state
            # contour info
            and np.allclose(self.mean_registered_contour, other.mean_registered_contour)
            and np.allclose(self.mean_aligned_contour, other.mean_aligned_contour)
            and np.allclose(self.contours, other.contours)
            # pca analysis info
            and np.allclose(self.principal_directions, other.principal_directions)
            and np.allclose(self.explained_variance, other.explained_variance)
            and np.allclose(self.explained_variance_ratio, other.explained_variance_ratio)
            and np.allclose(self.cum_explained_variance_ratio, other.cum_explained_variance_ratio)
            # k-means clustering info
            and self.cluster_id_df.equals(other.cluster_id_df)
            and self.labeled_contours_df.equals(other.labeled_contours_df)
            and np.allclose(self.centroids, other.centroids)
            and self.inertia == other.inertia
            and np.allclose(self.mean_cluster_contours, other.mean_cluster_contours)
            # hierarchical clustering info
            and np.allclose(self.pair_distance, other.pair_distance)
            and np.allclose(self.linkage_matrix, other.linkage_matrix)
            and self.branches['icoord'] == other.branches['icoord']
            and self.branches['dcoord'] == other.branches['dcoord']
            and self.branches['ivl'] == other.branches['ivl']
            and self.branches['leaves'] == other.branches['leaves']
        )
        return equal
