"""The VAMPIRE shape-mode model."""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from ._validation import ContourSet


class Vampire(ClusterMixin, TransformerMixin, BaseEstimator):
    """Cluster object contours into shape modes (VAMPIRE).

    Pipeline fitted by :meth:`fit`::

        ContourResampler(n_points) -> ContourRegistrar() -> ContourAligner()
        -> PCA(n_components) -> Normalizer() -> KMeans(n_clusters)

    After K-means, the mean contour of each cluster is computed, clusters are
    hierarchically linked (complete linkage on Euclidean distance between mean
    cluster contours), and cluster ids are renumbered to follow the dendrogram
    leaf order. Labels from :meth:`predict` therefore run 0..n_clusters-1 from
    left to right in the dendrogram, so similar shape modes get adjacent ids.

    Parameters
    ----------
    n_points : int, default=50
        Number of points each contour is resampled to.
    n_clusters : int, default=5
        Number of shape modes.
    n_components : int or float, default=0.95
        Number of principal components kept. A float in (0, 1) keeps the
        fewest components explaining at least that fraction of variance.
    n_init : int, default=3
        Number of K-means initializations.
    max_iter : int, default=300
        Maximum K-means iterations per initialization.
    random_state : int, RandomState instance or None, default=None
        Seed for K-means (and PCA solvers). Pass an int for reproducibility.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Fitted pipeline (steps ``resample``, ``register``, ``align``, ``pca``,
        ``normalize``, ``kmeans``). Its KMeans centers are already in
        dendrogram order.
    mean_contour_ : ndarray of shape (2, n_points)
        Mean of the aligned training contours.
    n_components_ : int
        Number of principal components kept.
    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Variance fraction explained by each kept component.
    cluster_centers_ : ndarray of shape (n_clusters, n_components_)
        K-means centers in normalized PC space, dendrogram-ordered.
    labels_ : ndarray of shape (n_samples,)
        Shape-mode label of each training contour.
    inertia_ : float
        K-means inertia on the training data.
    mean_cluster_contours_ : ndarray of shape (n_clusters, 2, n_points)
        Mean aligned contour of each shape mode (the shape-mode "picture").
    linkage_matrix_ : ndarray of shape (n_clusters - 1, 4)
        SciPy linkage matrix over shape modes (in the new label numbering).
    dendrogram_ : dict
        Output of :func:`scipy.cluster.hierarchy.dendrogram` (``no_plot=True``)
        for ``linkage_matrix_``; ``dendrogram_["leaves"] == list(range(n_clusters))``.
    n_features_in_ : int
        ``2 * n_points``.

    Examples
    --------
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(n_samples=60, random_state=0)
    >>> model = vp.Vampire(n_clusters=2, random_state=0).fit(X)
    >>> model.predict(X).shape
    (60,)
    """

    def __init__(
        self,
        n_points: int = 50,
        n_clusters: int = 5,
        n_components: int | float = 0.95,
        n_init: int = 3,
        max_iter: int = 300,
        random_state=None,
    ):
        self.n_points = n_points
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: ContourSet, y=None) -> "Vampire":
        """Fit the model to a contour set and return self."""
        raise NotImplementedError

    def predict(self, X: ContourSet) -> NDArray[np.intp]:
        """Return the shape-mode label of each contour, shape ``(n_samples,)``."""
        raise NotImplementedError

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Return distances to each shape-mode center, ``(n_samples, n_clusters)``."""
        raise NotImplementedError

    def normalize(self, X: ContourSet) -> NDArray[np.float64]:
        """Return normalized contours, shape ``(n_samples, 2, n_points)``."""
        raise NotImplementedError

    def score(self, X: ContourSet, y=None) -> float:
        """Return the negative K-means inertia of ``X`` (higher is better)."""
        raise NotImplementedError
