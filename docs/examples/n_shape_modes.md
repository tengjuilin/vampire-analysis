---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Choosing the number of shape modes

Two hyperparameters of {class}`vampire.Vampire` decide how many shape modes
you get and how well they separate: `n_components`, the number of principal
components the contours are compressed to, and `n_clusters`, the number of
shape modes. This page inspects the explained variance, shows a pitfall with a
single component, and compares grid search, silhouette, and elbow checks for
`n_clusters`. The example data are 200 synthetic contours of four shape kinds.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import vampire as vp

kinds = ("circle", "ellipse", "triangle", "star")
X, kind = vp.datasets.make_shapes(
    n_samples=200, kinds=kinds, random_state=0, return_labels=True
)
model = vp.Vampire(n_clusters=4, random_state=0).fit(X)
```

## Number of principal components

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

`Vampire` keeps **at least 2** components. After the row normalization step
a single component would leave only the values `+1` and `-1`, so K-means could
find at most two shape modes whatever `n_clusters` is. On very simple data the
first component alone explains 95% of the variance; a second one is then kept
anyway:

```{code-cell} ipython3
X2 = vp.datasets.make_shapes(n_samples=100, kinds=("circle", "ellipse"), random_state=0)
simple = vp.Vampire(n_clusters=3, random_state=0).fit(X2)
print("first component explains:", PCA().fit(simple.pipeline_[:3].transform(X2)).explained_variance_ratio_[0].round(3))
print("n_components_:", simple.n_components_)
print("objects per mode 1..3:", np.bincount(simple.labels_ - 1, minlength=3))
```

Real cell outlines are far more variable (the bundled microglia images need
about 20 components for 95% of the variance). An integer `n_components` below
2 raises `ValueError`, and so does a fit in which K-means leaves a shape mode
empty because the data have fewer distinct shapes than `n_clusters`.

## Number of shape modes (`n_clusters`)

`n_clusters` (the number of shape modes) is a modeling choice; values between
2 and 10 are typical. VAMPIRE is designed to *quantify* heterogeneity, so a
number that gives interpretable, reproducible modes is often more useful than
a statistically "optimal" one. Still, scikit-learn's tools help.

### Do not grid-search `n_clusters` on `score` alone

`Vampire.score` is minus the mean squared distance between each normalized
contour and its shape mode's mean contour. It does not depend on
`n_components`, so it can compare models that differ only there. Like any
within-cluster error, though, it improves when shape modes are added, so a
grid search over `n_clusters` on the default score tends to pick the
**largest** value in the grid:

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

See {doc}`custom_pipelines` for using the fitted pipeline's feature space in
other scikit-learn estimators.
