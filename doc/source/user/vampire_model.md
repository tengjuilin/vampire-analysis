(the_vampire_model)=

# The VAMPIRE Model

```{seealso}
:func:`vampire.model.initialize_model`
```

The VAMPIRE model is contained the dict ``model``, which is stored in a ``.pickle`` file named by the model name. The following can properties can be accessed as attributes or keys.

model_name : str
    Name of the model (build image set).
num_point : int
    Number of points used to describe the contour.
num_clusters : int
    Number of cluster centers for K-means clustering.
num_pc : int
    Number of principal components used after truncation.
mean_registered_contour : ndarray
    Mean registered contour.
mean_aligned_contour : ndarray
    Mean aligned contour.
principal_directions : ndarray
    Principal direction of PCA.
centroids : ndarray
    Cluster center in K-means clustering.
build_contours_df : DataFrame
    DataFrame of objects' contour coordinates with cluster id.