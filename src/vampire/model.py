"""The VAMPIRE shape-mode model."""

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import check_is_fitted

from ._validation import ContourSet
from .preprocessing import ContourAligner, ContourRegistrar, ContourResampler

#: Number of pipeline steps that normalize contours (resample, register, align).
_N_CONTOUR_STEPS = 3


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
        Number of shape modes, at least 2.
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
        ``normalize``, ``kmeans``). Its KMeans centers (and ``labels_``) are
        already in dendrogram order, so ``pipeline_.predict`` returns shape-mode
        labels.
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
        A mode with no training contours (possible only with heavily
        duplicated data) gets ``mean_contour_``.
    linkage_matrix_ : ndarray of shape (n_clusters - 1, 4)
        SciPy linkage matrix over shape modes (in the new label numbering).
    dendrogram_ : dict
        Output of :func:`scipy.cluster.hierarchy.dendrogram` (``no_plot=True``)
        for ``linkage_matrix_``; ``dendrogram_["leaves"] == list(range(n_clusters))``.
    n_features_in_ : int
        ``2 * n_points``.

    See Also
    --------
    vampire.ContourResampler, vampire.ContourRegistrar, vampire.ContourAligner

    Notes
    -----
    Renumbering. Let ``order = dendrogram(linkage(...))["leaves"]`` for the
    K-means clusters in their original numbering; original cluster
    ``order[j]`` becomes shape mode ``j``. The fitted KMeans step's
    ``cluster_centers_`` and ``labels_`` are permuted in place. The linkage
    matrix is renumbered by mapping its leaf ids (entries ``< n_clusters`` in
    the first two columns) through the same permutation while keeping
    merge order and left/right positions, so the dendrogram drawn from
    ``linkage_matrix_`` is the same tree with leaves labelled 0, 1, ... from
    left to right. Barring exact distance ties it equals
    ``linkage(pdist(mean_cluster_contours_.reshape(n_clusters, -1)),
    "complete")``.

    References
    ----------
    .. [1] Phillip, J. M., Han, K.-S., Chen, W.-C., Wirtz, D. & Wu, P.-H. A
       robust unsupervised machine-learning method to quantify the
       morphological heterogeneity of cells and nuclei. Nat Protoc 16,
       754–774 (2021). https://doi.org/10.1038/s41596-020-00432-x

    Examples
    --------
    >>> import vampire as vp
    >>> X = vp.datasets.make_shapes(n_samples=60, random_state=0)
    >>> model = vp.Vampire(n_clusters=2, random_state=0).fit(X)
    >>> model.predict(X).shape
    (60,)
    >>> model.mean_cluster_contours_.shape
    (2, 2, 50)
    >>> model.dendrogram_["leaves"]
    [0, 1]
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

    def __sklearn_tags__(self):
        """Declare that input is a contour set rather than a 2-D array."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags

    def _build_pipeline(self) -> Pipeline:
        if (
            not isinstance(self.n_clusters, int | np.integer)
            or isinstance(self.n_clusters, bool)
            or self.n_clusters < 2
        ):
            raise ValueError(
                "n_clusters must be an integer >= 2 (shape modes are ordered by a "
                f"dendrogram); got {self.n_clusters!r}."
            )
        return Pipeline(
            [
                ("resample", ContourResampler(n_points=self.n_points)),
                ("register", ContourRegistrar()),
                ("align", ContourAligner(flatten=True)),
                (
                    "pca",
                    PCA(n_components=self.n_components, random_state=self.random_state),
                ),
                ("normalize", Normalizer()),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=self.n_clusters,
                        init="k-means++",
                        n_init=self.n_init,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _fit(self, X: ContourSet) -> NDArray[np.float64]:
        """Fit the model; return the K-means input features of the training data."""
        pipeline = self._build_pipeline()
        contour_steps = pipeline[:_N_CONTOUR_STEPS]
        feature_steps = pipeline[_N_CONTOUR_STEPS:-1]
        kmeans = pipeline[-1]

        # Fit step by step so contours are aligned only once.
        aligned = contour_steps.fit_transform(X)  # (n_samples, 2 * n_points)
        features = feature_steps.fit_transform(aligned)
        kmeans.fit(features)

        n_clusters = self.n_clusters
        aligned_contours = aligned.reshape(len(aligned), 2, -1)
        mean_contour = pipeline.named_steps["align"].mean_contour_
        mean_cluster_contours = np.stack(
            [
                aligned_contours[kmeans.labels_ == k].mean(axis=0)
                if np.any(kmeans.labels_ == k)
                else mean_contour
                for k in range(n_clusters)
            ]
        )

        # Order clusters by the dendrogram of their mean contours.
        Z = linkage(pdist(mean_cluster_contours.reshape(n_clusters, -1)), "complete")
        order = np.asarray(dendrogram(Z, no_plot=True)["leaves"])
        new_label = np.argsort(order)  # new_label[old] = new
        kmeans.cluster_centers_ = kmeans.cluster_centers_[order]
        kmeans.labels_ = new_label[kmeans.labels_].astype(kmeans.labels_.dtype)
        Z = Z.copy()
        leaf = Z[:, :2] < n_clusters
        Z[:, :2][leaf] = new_label[Z[:, :2][leaf].astype(int)]

        self.pipeline_ = pipeline
        self.mean_contour_ = mean_contour
        pca = pipeline.named_steps["pca"]
        self.n_components_ = int(pca.n_components_)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_
        self.inertia_ = float(kmeans.inertia_)
        self.mean_cluster_contours_ = mean_cluster_contours[order]
        self.linkage_matrix_ = Z
        self.dendrogram_ = dendrogram(Z, no_plot=True)
        self.n_features_in_ = 2 * pipeline.named_steps["align"].n_points_
        return features

    def fit(self, X: ContourSet, y=None) -> "Vampire":
        """Fit the model to a contour set and return self.

        Parameters
        ----------
        X : contour set
            List of ``(2, n_i)`` arrays (lengths may differ), ndarray of shape
            ``(n_samples, 2, n)``, Series of arrays, or DataFrame with a
            ``"contour"`` column. Needs at least ``n_clusters`` contours.
        y : None
            Ignored.

        Returns
        -------
        self : Vampire
        """
        self._fit(X)
        return self

    def fit_transform(self, X: ContourSet, y=None) -> NDArray[np.float64]:
        """Fit the model and return distances to each shape-mode center.

        Equivalent to ``fit(X).transform(X)`` but normalizes contours only once.

        Parameters
        ----------
        X : contour set
        y : None
            Ignored.

        Returns
        -------
        distances : ndarray of shape (n_samples, n_clusters)
        """
        features = self._fit(X)
        return self.pipeline_.named_steps["kmeans"].transform(features)

    def predict(self, X: ContourSet) -> NDArray[np.intp]:
        """Return the shape-mode label of each contour.

        Parameters
        ----------
        X : contour set

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Integers in ``[0, n_clusters)`` (dendrogram order).
        """
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def transform(self, X: ContourSet) -> NDArray[np.float64]:
        """Return distances to each shape-mode center.

        Distances are Euclidean, in normalized principal-component space.

        Parameters
        ----------
        X : contour set

        Returns
        -------
        distances : ndarray of shape (n_samples, n_clusters)
            Column ``k`` is the distance to shape mode ``k``.
        """
        check_is_fitted(self)
        return self.pipeline_.transform(X)

    def normalize(self, X: ContourSet) -> NDArray[np.float64]:
        """Return contours resampled, registered, and aligned as in training.

        Parameters
        ----------
        X : contour set

        Returns
        -------
        normalized : ndarray of shape (n_samples, 2, n_points)
            Contours comparable to ``mean_contour_`` and
            ``mean_cluster_contours_``.
        """
        check_is_fitted(self)
        aligned = self.pipeline_[:_N_CONTOUR_STEPS].transform(X)
        return aligned.reshape(len(aligned), 2, -1)

    def score(self, X: ContourSet, y=None) -> float:
        """Return the negative K-means inertia of ``X`` (higher is better).

        Parameters
        ----------
        X : contour set
        y : None
            Ignored.

        Returns
        -------
        score : float
            Minus the sum of squared distances of ``X`` to their closest
            shape-mode center in normalized PC space.
        """
        check_is_fitted(self)
        return float(self.pipeline_.score(X))
