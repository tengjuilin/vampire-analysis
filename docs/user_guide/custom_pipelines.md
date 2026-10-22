---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Custom pipelines and model selection

{class}`vampire.Vampire` is a scikit-learn estimator built from scikit-learn
parts. This page shows how to take it apart, swap components, and choose its
hyperparameters (`n_components`, `n_clusters`) with standard scikit-learn
tools.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
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

The labels describe the same partition, but their *numbers* differ: only
{class}`~vampire.Vampire` renumbers clusters in dendrogram order and computes
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

## Choosing `n_components`

The default `n_components=0.95` keeps the fewest principal components that
explain 95% of the variance of the aligned contours. Inspect the result:

```{code-cell} ipython3
print("components kept:", model.n_components_)
print("explained variance ratio:", model.explained_variance_ratio_.round(3))
```

A scree plot over all components helps to pick an integer instead:

```{code-cell} ipython3
pca_full = PCA().fit(model.pipeline_[:3].transform(X))
ratio = pca_full.explained_variance_ratio_[:15]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(np.arange(1, 16), ratio, color="0.6", label="per component")
ax.plot(np.arange(1, 16), np.cumsum(ratio), "o-", color="C3", label="cumulative")
ax.axhline(0.95, ls=":", color="k", lw=1)
ax.set(xlabel="principal component", ylabel="explained variance ratio")
ax.legend();
```

````{warning}
**The one-component pitfall.** On very simple data the first component alone
can explain 95% of the variance. After the row normalization step every
one-dimensional score becomes exactly `+1` or `-1`, so K-means sees only two
distinct points, whatever `n_clusters` is:

```python
X2 = vp.datasets.make_shapes(n_samples=100, kinds=("circle", "ellipse"), random_state=0)
vp.Vampire(n_clusters=3).fit(X2).n_components_  # 1 -> at most 2 non-empty modes
```

If `model.n_components_ == 1`, pass an integer `n_components >= 2`. Real cell
outlines are far more variable (the bundled microglia images need 20
components for 95% of the variance), so this mainly affects synthetic or very
homogeneous data.
````

The pitfall, demonstrated (note the scikit-learn warning and the empty mode):

```{code-cell} ipython3
X2 = vp.datasets.make_shapes(n_samples=100, kinds=("circle", "ellipse"), random_state=0)
trivial = vp.Vampire(n_clusters=3, random_state=0).fit(X2)
print("n_components_:", trivial.n_components_)
print("distinct feature values:", np.unique(trivial.pipeline_[:-1].transform(X2)))
print("objects per mode:", np.bincount(trivial.labels_, minlength=3))

fixed = vp.Vampire(n_clusters=3, n_components=5, random_state=0).fit(X2)
print("with n_components=5, objects per mode:", np.bincount(fixed.labels_, minlength=3))
```

## Choosing `n_clusters`

`n_clusters` (the number of shape modes) is a modeling choice; values between
2 and 10 are typical. VAMPIRE is designed to *quantify* heterogeneity, so a
number that gives interpretable, reproducible modes is often more useful than
a statistically "optimal" one. Still, scikit-learn's tools help.

### Do not grid-search on `score` alone

`Vampire.score` returns minus the K-means inertia. Inertia always decreases
when clusters are added, so a grid search on the default score simply picks
the **largest** `n_clusters` in the grid:

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV

grid = {"n_clusters": [2, 3, 4, 5, 6, 7]}
naive = GridSearchCV(vp.Vampire(n_components=5, random_state=0), grid, cv=3).fit(X)
print("best by score:", naive.best_params_)
print("mean test score:", naive.cv_results_["mean_test_score"].round(3))
```

### Use a scorer that penalizes extra clusters

A custom scorer receives the fitted estimator and the held-out contours. The
silhouette coefficient on the K-means input features works well:

```{code-cell} ipython3
def silhouette(estimator, X, y=None):
    """Silhouette coefficient of the shape modes in the model's feature space."""
    features = estimator.pipeline_[:-1].transform(X)
    return silhouette_score(features, estimator.predict(X))


search = GridSearchCV(
    vp.Vampire(n_components=5, random_state=0), grid, scoring=silhouette, cv=3
).fit(X)
print("best by silhouette:", search.best_params_)
pd.DataFrame(search.cv_results_)[["param_n_clusters", "mean_test_score", "std_test_score"]]
```

The data has four shape kinds, and the silhouette search finds `n_clusters=4`.
`search.best_estimator_` is a `Vampire` refitted on all of `X`.

### Elbow plot

Plotting inertia against `n_clusters` and looking for a bend ("elbow") is the
classic visual check:

```{code-cell} ipython3
ks = range(2, 9)
inertia = [vp.Vampire(n_clusters=k, n_components=5, random_state=0).fit(X).inertia_ for k in ks]
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(list(ks), inertia, "o-")
ax.set(xlabel="n_clusters", ylabel="inertia (training)");
```

On real images there is rarely a sharp elbow or a clear silhouette maximum;
check that the shape modes you pick are stable across `random_state` values
and subsamples (e.g. compare labels with {func}`~sklearn.metrics.adjusted_rand_score`).

## Pandas output

`Vampire` and `ContourAligner` (with `flatten=True`) implement
`get_feature_names_out`, so scikit-learn's `set_output` API returns labeled
DataFrames from `transform`:

```{code-cell} ipython3
distances = vp.Vampire(n_clusters=4, random_state=0).set_output(transform="pandas")
distances.fit_transform(X).head(3)  # columns vampire0..vampire3: distance to each mode
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
