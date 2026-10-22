---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Custom pipelines

{class}`vampire.Vampire` is a scikit-learn estimator built from scikit-learn
parts. This page shows how to take it apart, swap components, and get labeled
pandas output. Choosing its hyperparameters (`n_components`, `n_clusters`) is
covered in {doc}`n_shape_modes`.

```{code-cell} ipython3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import vampire as vp

kinds = ("circle", "ellipse", "triangle", "star")
X, kind = vp.datasets.make_shapes(
    n_samples=200, kinds=kinds, random_state=0, return_labels=True
)
```

## Inside `Vampire`

The fitted model keeps its pipeline in `pipeline_`:

```{code-cell} ipython3
model = vp.Vampire(n_clusters=4, random_state=0).fit(X)
model.pipeline_
```

Slicing it gives intermediate representations:

```{code-cell} ipython3
aligned = model.pipeline_[:3].transform(X)  # resample, register, align
features = model.pipeline_[:-1].transform(X)  # ... + PCA + Normalizer
print("aligned contours, flattened:", aligned.shape)  # (n_samples, 2 * n_points)
print("K-means input features:", features.shape)  # (n_samples, n_components_)
print("same as model.normalize:", np.allclose(aligned.reshape(-1, 2, 50), model.normalize(X)))
```

The last step is the scikit-learn K-means, whose cluster ids are 0-based and
already in dendrogram order; `Vampire.predict` adds 1:

```{code-cell} ipython3
print("pipeline_ K-means + 1 == predict:", (model.pipeline_.predict(X) + 1 == model.predict(X)).all())
```

## The same pipeline by hand

The three contour transformers turn a contour set into a feature matrix of
shape `(n_samples, 2 * n_points)` that any scikit-learn estimator accepts:

```{code-cell} ipython3
manual = make_pipeline(
    vp.ContourResampler(n_points=50),
    vp.ContourRegistrar(),
    vp.ContourAligner(),  # flatten=True by default
    PCA(n_components=0.95, random_state=0),
    Normalizer(),
    KMeans(n_clusters=4, n_init=3, random_state=0),
)
manual_labels = manual.fit_predict(X)
print("agreement with Vampire (ARI):", adjusted_rand_score(model.labels_, manual_labels))
```

The labels describe the same partition, but their *numbers* differ: the
hand-built K-means numbers clusters `0 ... K - 1` arbitrarily, while only
{class}`~vampire.Vampire` renumbers them `1 ... K` in dendrogram order and computes
`mean_cluster_contours_`, `linkage_matrix_`, and the other attributes that
{mod}`vampire.plotting` needs. Use a hand-built pipeline for experiments, and
`Vampire` for the published method and its figures.

## Swapping components

Because every step is a scikit-learn estimator, you can replace any of them.
For example, a Gaussian mixture on standardized PC scores instead of K-means
on normalized scores:

```{code-cell} ipython3
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

gmm = make_pipeline(
    vp.ContourResampler(),
    vp.ContourRegistrar(),
    vp.ContourAligner(),
    PCA(n_components=5, random_state=0),
    StandardScaler(),
    GaussianMixture(n_components=4, random_state=0),
)
gmm_labels = gmm.fit(X).predict(X)
print("GMM vs. true kinds (ARI):", adjusted_rand_score(kind, gmm_labels))
```

Or use the normalized contours as features for a *supervised* classifier,
when you have labels (e.g. predict the treatment group from shape):

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

classifier = make_pipeline(
    vp.ContourResampler(),
    vp.ContourRegistrar(),
    vp.ContourAligner(),
    PCA(n_components=10, random_state=0),
    LogisticRegression(max_iter=1000),
)
scores = cross_val_score(classifier, X, kind, cv=5)
print("cross-validated accuracy: %.3f +- %.3f" % (scores.mean(), scores.std()))
```

Contour sets work with scikit-learn's cross-validation utilities as long as
they are a list, an ndarray, or a pandas object (all support indexing by
sample).

## Pandas output

`Vampire` and `ContourAligner` (with `flatten=True`) implement
`get_feature_names_out`, so scikit-learn's `set_output` API returns labeled
DataFrames from `transform`:

```{code-cell} ipython3
distances = vp.Vampire(n_clusters=4, random_state=0).set_output(transform="pandas")
distances.fit_transform(X).head(3)  # columns mode1..mode4: distance to each mode
```

```{code-cell} ipython3
contour_features = make_pipeline(
    vp.ContourResampler(n_points=20), vp.ContourRegistrar(), vp.ContourAligner()
).set_output(transform="pandas")
contour_features.fit_transform(X).iloc[:2, [0, 1, 20, 21]]  # columns x0..x19, y0..y19
```

`ContourResampler` and `ContourRegistrar` accept `set_output(transform="pandas")`
(or `"polars"`) as a no-op so that `Pipeline.set_output` works, but their
output stays a 3-D array of shape `(n_samples, 2, n_points)`. With pandas
output, `Vampire.transform` keeps the index of a DataFrame input, so the
distances can be joined back onto the property table.

## Parameters and cloning

`Vampire` follows the scikit-learn estimator API, so `get_params`,
`set_params`, and {func}`sklearn.base.clone` work:

```{code-cell} ipython3
from sklearn.base import clone

template = vp.Vampire(n_points=60, n_clusters=4, random_state=0)
variant = clone(template).set_params(n_clusters=6)
print(template.get_params())
print(variant.n_clusters, variant.n_points)
```
