========
Analysis
========

.. currentmodule:: vampire.analysis


PCA
---

.. autosummary::
   :toctree: api/

   get_explained_variance_ratio
   get_cum_explained_variance_ratio
   get_optimal_n_pcs
   pca_transform_contours

Hierarchical Clustering
-----------------------

.. autosummary::
   :toctree: api/

   cluster_contours
   assign_clusters_id
   get_labeled_contours_df
   get_mean_cluster_contours
   hierarchical_cluster_contour
   get_cluster_order
   get_distribution
   get_shannon_entropy
   reorder_clusters
   reorder_centroids