---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# How VAMPIRE works

VAMPIRE {doc}`[1] <../references>` describes every object by its outline only,
removes everything that is not shape (position, size, rotation, where the
outline "starts"), and clusters the normalized outlines. {class}`vampire.Vampire`
runs the steps below in this order; each step is also available on its own so
you can inspect or replace it ({doc}`custom_pipelines`).

| Step | What it removes / does | Implemented by |
|-|-|-|
| 1. Resample | different numbers of points, uneven spacing | {class}`~vampire.ContourResampler`, {func}`~vampire.contours.resample_contour` |
| 2. Register | translation, scale, rotation, direction | {class}`~vampire.ContourRegistrar`, {func}`~vampire.contours.register_contour` |
| 3. Align | residual rotation and starting point, relative to the mean shape | {class}`~vampire.ContourAligner`, {func}`~vampire.contours.align_contour` |
| 4. PCA | redundancy between coordinates | {class}`sklearn.decomposition.PCA` |
| 5. Normalize | overall magnitude in PC space | {class}`sklearn.preprocessing.Normalizer` |
| 6. K-means | -- groups contours into shape modes | {class}`sklearn.cluster.KMeans` |
| 7. Order | arbitrary K-means numbering | dendrogram of the mean mode contours |

Steps 1--3 are fitted only once (on the training data) and then applied
unchanged to new data in `predict` / `transform`, so training and new contours
end up in the same coordinate frame.

Throughout, $N$ is the number of points per contour (`n_points`, default 50),
a contour is a $2 \times N$ matrix with x in row 0 and y in row 1, and
$n$ is the number of objects.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import vampire as vp
from vampire.contours import align_contour, register_contour, resample_contour


def draw(ax, contour, start=True, **kwargs):
    """Draw a closed contour of shape (2, n); mark its first point."""
    x, y = np.append(contour[0], contour[0, 0]), np.append(contour[1], contour[1, 0])
    (line,) = ax.plot(x, y, **kwargs)
    if start:
        ax.plot(contour[0, 0], contour[1, 0], "o", color=line.get_color(), ms=5)
    ax.set_aspect("equal")
    return line


kinds = ("circle", "ellipse", "triangle", "star")
X, kind = vp.datasets.make_shapes(
    n_samples=200, kinds=kinds, random_state=0, return_labels=True
)
print(len(X), X[0].shape)
```

`X` is a list of 200 synthetic contours of four kinds with random size,
rotation, position, starting point, and direction. A correct pipeline should
recover the four kinds.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5))
for c, k in zip(X[:16], kind[:16]):
    draw(ax, c, color=f"C{k}")
ax.set_title("Raw contours (dot = first point)")
ax.invert_yaxis()  # image convention: y grows downward
```

## 1. Resampling

Contours traced from pixels have different numbers of points, unevenly spaced
along the outline (the bundled microglia images give 40 to 430 points per
object). {func}`~vampire.contours.resample_contour` fits a periodic cubic
B-spline $\mathbf{s}(t)$ through the points, parameterized by cumulative chord
length, and evaluates it at $N$ points equally spaced in arc length:

$$
\ell(t) = \int_0^t \lVert \mathbf{s}'(\tau) \rVert \, d\tau, \qquad
\mathbf{p}_j = \mathbf{s}\!\left(\ell^{-1}\!\left(\tfrac{j L}{N}\right)\right),
\quad j = 0, \dots, N - 1,
$$

where $L = \ell(t_\text{end})$ is the perimeter of the spline. The arc length is
measured numerically on a dense evaluation of the spline. Because the contour
is closed, the last point $\mathbf{p}_{N-1}$ is one step *before* the start
point, which is not repeated.

```{code-cell} ipython3
images, filenames = vp.datasets.load_example_images()
props = vp.extract_properties(images, filenames)
raw = props.loc[5, "contour"]
resampled = resample_contour(raw, n_points=50)
print("raw:", raw.shape, " resampled:", resampled.shape)

steps = np.hypot(*np.diff(resampled, axis=1, append=resampled[:, :1]))
print("spacing: min %.3f, max %.3f px" % (steps.min(), steps.max()))

fig, ax = plt.subplots(figsize=(5, 5))
draw(ax, raw, start=False, color="0.7", lw=4, label=f"raw ({raw.shape[1]} points)")
ax.plot(*resampled, "o-", ms=3, color="C3", label="resampled (50 points)")
ax.invert_yaxis()
ax.legend(loc="upper right", fontsize=8);
```

Raw pixel contours are made of horizontal, vertical, and diagonal steps; the
spline smooths these and the resampled points are equidistant.

## 2. Registration

{func}`~vampire.contours.register_contour` removes translation, scale, and
rotation from each contour $\mathbf{A}$ ($2 \times N$) independently:

1. **Center:** $\mathbf{B} = \mathbf{A} - \bar{\mathbf{a}}\mathbf{1}^T$, with
   $\bar{\mathbf{a}}$ the mean of the points.
2. **Scale** by the characteristic length (root-mean-square distance to the
   centroid):
   $R = \sqrt{\tfrac{1}{N}\sum_{i=1}^N (x_i^2 + y_i^2)}$, so
   $\mathbf{B} \leftarrow \mathbf{B} / R$.
3. **Rotate** onto the principal axes: with the SVD
   $\mathbf{B} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, the registered
   contour is $\mathbf{T} = \mathbf{U}^T\mathbf{B}$, so the major axis lies
   along x. SVD fixes each axis only up to sign; the major axis is flipped so
   the third moment $\sum_i (\mathbf{u}_1^T\mathbf{b}_i)^3 \ge 0$ (it points to
   the heavier side of the shape) and the minor axis is chosen so that
   $\det \mathbf{U} = +1$. $\mathbf{U}$ is therefore always a **proper rotation**:
   a shape and its mirror image are not made identical.
4. **Direction:** if the signed area
   $\tfrac12 \sum_i (x_i y_{i+1} - x_{i+1} y_i)$ is negative (clockwise), the
   point order is reversed so every contour runs counterclockwise.
5. **Start point:** the points are cyclically shifted so the first point has
   the smallest absolute polar angle $|\operatorname{atan2}(y_i, x_i)|$, i.e. it
   lies closest to the positive major axis.

```{code-cell} ipython3
resampler = vp.ContourResampler(n_points=50)
registrar = vp.ContourRegistrar()
resampled_set = resampler.fit_transform(X)  # (200, 2, 50)
registered = registrar.fit_transform(resampled_set)  # (200, 2, 50)
print(resampled_set.shape, registered.shape)
print("centered:", np.allclose(registered.mean(axis=2), 0))
print("unit scale:", np.allclose(np.sqrt((registered**2).sum(axis=1).mean(axis=1)), 1))
print("counterclockwise:", all(vp.contours.signed_area(c) > 0 for c in registered))

fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharex=True, sharey=True)
for k, ax in enumerate(axes):
    for c in registered[kind == k][:8]:
        draw(ax, c, color=f"C{k}", lw=1, alpha=0.8)
    ax.set_title(kinds[k])
fig.suptitle("Registered contours (dot = first point)");
```

Circles and stars have no well-defined major axis, so their rotation and start
point after registration are still essentially random; alignment fixes that.

## 3. Alignment

Registration treats every contour on its own. Alignment makes them comparable
point by point. {class}`~vampire.ContourAligner` computes the **reference**
contour $\mathbf{M}$ = the mean of the registered training contours
(`reference_contour_`). Each registered contour $\mathbf{A}$ is then rotated
and cyclically shifted to best match $\mathbf{M}$:

$$
\min_{k \in \{0,\dots,N-1\}} \; \min_{\mathbf{R} \in SO(2)}
\lVert \mathbf{R}\mathbf{A}_k - \mathbf{M} \rVert_F^2 ,
$$

where $\mathbf{A}_k$ is $\mathbf{A}$ re-started at point $k$. For a given
shift, the optimal rotation is found with the **Kabsch algorithm**: with
$\mathbf{C} = \mathbf{A}_k \mathbf{M}^T = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$,

$$
\mathbf{R} = \mathbf{V}
\begin{bmatrix} 1 & 0 \\ 0 & d \end{bmatrix}
\mathbf{U}^T, \qquad d = \operatorname{sign}\det(\mathbf{V}\mathbf{U}^T),
$$

which excludes reflections. In 2-D the best rotation angle for each shift has
a closed form, so all $N$ shifts are scored at once: with
$c_k = \sum_i (a_{x,i} m_{x,i} + a_{y,i} m_{y,i})$ and
$s_k = \sum_i (a_{x,i} m_{y,i} - a_{y,i} m_{x,i})$, the residual is
$\lVert \mathbf{A} \rVert_F^2 + \lVert \mathbf{M} \rVert_F^2 - 2\sqrt{c_k^2 + s_k^2}$,
minimized by the shift with the largest $\sqrt{c_k^2 + s_k^2}$.

```{code-cell} ipython3
aligner = vp.ContourAligner(flatten=False)
aligned = aligner.fit_transform(registered)  # (200, 2, 50)
print(aligned.shape, aligner.reference_contour_.shape)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharex=True, sharey=True)
for k, ax in enumerate(axes):
    for c in aligned[kind == k][:8]:
        draw(ax, c, color=f"C{k}", lw=1, alpha=0.8)
    draw(ax, aligner.reference_contour_, start=False, color="k", ls="--", lw=1.5)
    ax.set_title(kinds[k])
fig.suptitle("Aligned contours; dashed = reference (mean registered contour)");
```

After alignment, the first points coincide and point $i$ of every contour sits
at roughly the same place on the outline, so the coordinates can be compared
as feature vectors. With `flatten=True` (the default, used by
{class}`~vampire.Vampire`) the aligner returns these as a matrix of shape
`(n_samples, 2 * N)`: all x coordinates, then all y coordinates.

## 4. Principal component analysis

The $2N$ coordinates are strongly correlated (neighboring points move
together). PCA finds the few directions of largest variation. Let
$\mathbf{X} \in \mathbb{R}^{n \times 2N}$ hold the flattened aligned contours,
$\boldsymbol{\mu}$ their mean, and
$\mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T = \mathbf{U}\boldsymbol{\Sigma}\mathbf{W}^T$.
The scores on the first $q$ principal components are

$$
\mathbf{Z} = (\mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T)\,\mathbf{W}_{:, 1:q}
\in \mathbb{R}^{n \times q}.
$$

`n_components` sets $q$: an integer keeps that many components; a float in
$(0, 1)$ (default `0.95`) keeps the fewest components that explain at least
that fraction of the variance. The fitted value is `model.n_components_`.

```{code-cell} ipython3
from sklearn.decomposition import PCA

flat = aligned.reshape(len(aligned), -1)  # (200, 100)
pca = PCA().fit(flat)
ratio = pca.explained_variance_ratio_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
ax1.bar(np.arange(1, 11), ratio[:10], color="0.5")
ax1.plot(np.arange(1, 11), np.cumsum(ratio[:10]), "o-", color="C3", label="cumulative")
ax1.axhline(0.95, color="k", ls=":", lw=1)
ax1.set(xlabel="principal component", ylabel="explained variance ratio")
ax1.legend()

mean = pca.mean_.reshape(2, -1)
for j, color in [(0, "C0"), (1, "C1")]:
    mode = pca.components_[j].reshape(2, -1) * np.sqrt(pca.explained_variance_[j])
    for t, ls in [(-2, "--"), (2, "-")]:
        draw(ax2, mean + t * mode, start=False, color=color, ls=ls, lw=1,
             label=f"PC{j + 1} {t:+d} sd")
draw(ax2, mean, start=False, color="k", lw=2, label="mean")
ax2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
ax2.set_title("Shape variation along PC1 and PC2")
print("components for 95% variance:", int(np.searchsorted(np.cumsum(ratio), 0.95) + 1))
```

## 5. Normalization

Each row of $\mathbf{Z}$ is scaled to unit Euclidean length
(scikit-learn's {class}`~sklearn.preprocessing.Normalizer`):

$$
\hat{\mathbf{z}}_i = \mathbf{z}_i / \lVert \mathbf{z}_i \rVert_2 .
$$

K-means on $\hat{\mathbf{z}}$ therefore groups contours by the *direction* of
their deviation from the mean shape rather than its size (similar to cosine
distance). The same normalization is applied in `fit` and in
`predict`/`transform`.

```{code-cell} ipython3
from sklearn.preprocessing import Normalizer

Z = PCA(n_components=0.95).fit_transform(flat)
Z_hat = Normalizer().fit_transform(Z)
print("Z:", Z.shape, " row norms after normalization:", np.unique(np.linalg.norm(Z_hat, axis=1).round(6)))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, data, title in [(axes[0], Z, "PC scores"), (axes[1], Z_hat, "normalized")]:
    ax.scatter(data[:, 0], data[:, 1], c=[f"C{k}" for k in kind], s=10)
    ax.set(xlabel="PC1", ylabel="PC2", title=title, aspect="equal")
```

```{warning}
If only **one** component is kept, every normalized score is exactly $+1$ or
$-1$, so K-means can find at most two distinct groups. This happens on very
simple data (for example two shape kinds only). Check `model.n_components_`
and set `n_components` to an integer of at least 2 if needed; see
{doc}`custom_pipelines`.
```

## 6. K-means clustering

{class}`~sklearn.cluster.KMeans` partitions the normalized scores into $K$
clusters (`n_clusters`, the shape modes) by minimizing the inertia

$$
\sum_{i=1}^n \lVert \hat{\mathbf{z}}_i - \mathbf{c}_{\ell_i} \rVert^2 ,
$$

using k-means++ initialization, `n_init` restarts (default 3), and at most
`max_iter` iterations (default 300). Pass `random_state` for reproducible
results. `model.score(X)` returns minus this inertia for `X`.

## 7. Ordering shape modes by a dendrogram

K-means numbers its clusters arbitrarily. VAMPIRE computes the **mean aligned
contour of each cluster** $\mathbf{m}_k$ (`mean_cluster_contours_`, shape
`(K, 2, N)`), links them by hierarchical clustering with complete linkage on
Euclidean distance,

$$
d(\mathcal{A}, \mathcal{B}) = \max_{a \in \mathcal{A},\, b \in \mathcal{B}}
\lVert \mathbf{m}_a - \mathbf{m}_b \rVert ,
$$

and renumbers the clusters `0, 1, ..., K - 1` in the dendrogram's left-to-right
leaf order. Similar modes therefore get adjacent numbers, and the labels from
`predict`, `labels_`, `cluster_centers_`, and `mean_cluster_contours_` all use
this numbering.

```{code-cell} ipython3
model = vp.Vampire(n_clusters=4, random_state=0).fit(X)
print("n_components_:", model.n_components_)
print("leaf order:", model.dendrogram_["leaves"])
fig, axes = vp.plotting.plot_summary(model)
```

Cross-tabulating the shape modes against the true kinds shows that every mode
contains exactly one kind:

```{code-cell} ipython3
import pandas as pd

pd.crosstab(pd.Series(model.labels_, name="shape mode"),
            pd.Series(np.array(kinds)[kind], name="true kind"))
```

## 8. Prediction on new data

`model.predict(X_new)` resamples and registers the new contours, aligns them to
the **training** reference contour, projects them with the **training** PCA,
normalizes them, and assigns each to the nearest training cluster center. The
model is not refitted, so labels of different data sets are directly
comparable:

```{code-cell} ipython3
X_new, kind_new = vp.datasets.make_shapes(
    n_samples=40, kinds=kinds, random_state=1, return_labels=True
)
labels_new = model.predict(X_new)
print(labels_new.shape, labels_new.dtype)
print("distances to each mode:", model.transform(X_new).shape)
```
