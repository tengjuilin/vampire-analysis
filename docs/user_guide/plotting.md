---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Plotting gallery

{mod}`vampire.plotting` draws fitted models with Matplotlib. All functions:

- take an optional `ax` (single-axes functions) or create their own figure;
- return the `Axes`, or `(Figure, ndarray of Axes)` for multi-panel figures;
- never call `plt.show()`, never save files, and never change global style --
  call `fig.savefig(...)` yourself (for an `Axes`, `ax.figure.savefig(...)`);
- use one categorical palette (`tab10`, or `tab20` above 10 modes), so shape
  mode `i` has the same color in every figure;
- place mode `i` at `x = 5 + 10 * i`, so distributions, contours, and
  dendrograms line up when stacked with a shared x axis.

| Function | Shows | Returns |
|-|-|-|
| {func}`~vampire.plotting.plot_summary` | distribution + mean contours + dendrogram | `(fig, axes)`, `axes.shape == (3,)` |
| {func}`~vampire.plotting.plot_distribution` | % of objects per mode | `Axes` |
| {func}`~vampire.plotting.plot_shape_modes` | mean contour of each mode | `Axes` |
| {func}`~vampire.plotting.plot_dendrogram` | tree linking the modes | `Axes` |
| {func}`~vampire.plotting.plot_representatives` | most typical objects of each mode | `(fig, axes)`, `axes.shape == (n_clusters,)` |
| {func}`~vampire.plotting.plot_labeled_image` | an image with objects colored by mode | `Axes` |

All examples use the bundled microglia images:

```{code-cell} ipython3
import matplotlib.pyplot as plt

import vampire as vp

images, filenames = vp.datasets.load_example_images()
props = vp.extract_properties(images, filenames)
model = vp.Vampire(n_clusters=6, random_state=0).fit(props)
props["shape_mode"] = model.predict(props)
```

## Summary figure

```{code-cell} ipython3
fig, axes = vp.plotting.plot_summary(model, props["shape_mode"])
```

`labels` defaults to the training labels (`model.labels_`). Pass the labels of
one group to show that group's distribution against the shared modes. For
many modes, increase the figure width and shrink the contour row:

```{code-cell} ipython3
fig, axes = vp.plotting.plot_summary(
    model,
    props.loc[props["filename"] == "img2", "shape_mode"],
    figsize=(7, 4),
    height_ratios=(3, 1, 1),
)
axes[0].set_title("img2 only");
```

## Distribution

```{code-cell} ipython3
ax = vp.plotting.plot_distribution(props["shape_mode"], n_clusters=model.n_clusters)
```

Always pass `n_clusters=model.n_clusters`, so that modes without objects still
get a (zero) bar and colors stay consistent. Drawing into existing axes makes
side-by-side comparisons easy:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, (name, df) in zip(axes, props.groupby("filename")):
    vp.plotting.plot_distribution(df["shape_mode"], n_clusters=model.n_clusters, ax=ax)
    ax.set_title(name)
```

## Shape modes

```{code-cell} ipython3
ax = vp.plotting.plot_shape_modes(model)
```

`colors` accepts a colormap name, a `Colormap`, one color, or a list of
`n_clusters` colors:

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(6, 3))
vp.plotting.plot_shape_modes(model, ax=axes[0], colors="viridis")
vp.plotting.plot_shape_modes(model, ax=axes[1], colors="black", linewidth=1)
```

## Dendrogram

```{code-cell} ipython3
ax = vp.plotting.plot_dendrogram(model)
```

Extra keyword arguments go to {func}`scipy.cluster.hierarchy.dendrogram`:

```{code-cell} ipython3
ax = vp.plotting.plot_dendrogram(model, orientation="left")
```

## Representative contours

For each mode, the objects closest to the mode's center (in the model's
feature space), normalized and overlaid, with the mode's mean contour dashed:

```{code-cell} ipython3
fig, axes = vp.plotting.plot_representatives(model, props, n_per_mode=8)
```

`X` can be any contour set, e.g. one group only; `labels` defaults to
`model.predict(X)`.

## Labeled image

Color the objects of one image by shape mode. Pass the rows of the property
table that belong to that image (they carry the object `label`) and their
shape modes:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, image_id in zip(axes, [0, 1]):
    rows = props[props["image_id"] == image_id]
    vp.plotting.plot_labeled_image(
        images[image_id], rows, rows["shape_mode"], ax=ax, n_clusters=model.n_clusters
    )
    ax.set_title(filenames[image_id])
```

`background_color` sets the color of background pixels and of objects that are
not listed in `objects` (for example objects dropped as too small):

```{code-cell} ipython3
rows = props[(props["image_id"] == 0) & (props["shape_mode"] == 0)]
ax = vp.plotting.plot_labeled_image(
    images[0], rows, rows["shape_mode"], n_clusters=model.n_clusters,
    background_color="0.9",
)
ax.set_title("mode 0 only");
```

## Combining and saving

Every function that takes `ax` composes into larger figures, and all return
values are ordinary Matplotlib objects:

```{code-cell} ipython3
fig = plt.figure(figsize=(10, 4), layout="constrained")
left, right = fig.subfigures(1, 2, width_ratios=[1.3, 1])
top, bottom = left.subplots(2, 1, sharex=True, height_ratios=[3, 1])
vp.plotting.plot_distribution(props["shape_mode"], n_clusters=model.n_clusters, ax=top)
vp.plotting.plot_dendrogram(model, ax=bottom)
ax = right.subplots()
rows = props[props["image_id"] == 0]
vp.plotting.plot_labeled_image(images[0], rows, rows["shape_mode"], ax=ax,
                               n_clusters=model.n_clusters)
# fig.savefig("figure_2.pdf")  # vector output for publications
```
