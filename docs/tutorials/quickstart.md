---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quickstart

This page goes from segmented images to shape modes in five steps, using two
segmented microglia images that ship with the package. Every code cell below
was run when this page was built, so you can copy the cells in order into a
script or notebook and get the same result.

**Input:** segmented images, i.e. 2-D integer arrays in which background is 0
and each object (cell or nucleus) is either 1/255 (binary mask) or has its own
positive label.
**Output:** a shape-mode label for every object, plus figures.

## 1. Load images

```{code-cell} ipython3
import numpy as np

import vampire as vp

images, filenames = vp.datasets.load_example_images()
print(filenames)
print(type(images[0]), images[0].shape, images[0].dtype)
print("objects per image:", [len(np.unique(img)) - 1 for img in images])
```

`images` is a `list` of `(512, 512)` `uint16` label images; `filenames` is a
`list` of `str`. For your own data use
`images, filenames = vp.read_images("path/to/folder")`, which returns the same
two lists (see {doc}`your_images`).

## 2. Extract contours and properties

{func}`vampire.extract_properties` measures every object and traces its
outline:

```{code-cell} ipython3
props = vp.extract_properties(images, filenames)
print(props.shape)
props.head()
```

`props` is a {class}`pandas.DataFrame` with one row per object. The
`contour` column holds each object's outline as a float array of shape
`(2, n)` (row 0 = x, row 1 = y, in pixels); `n` differs between objects:

```{code-cell} ipython3
contour = props.loc[0, "contour"]
print(contour.shape, contour.dtype)
print("points per contour: min", props["contour"].map(lambda c: c.shape[1]).min(),
      "max", props["contour"].map(lambda c: c.shape[1]).max())
```

The other columns (area, perimeter, circularity, ...) are standard
morphological measurements; see {doc}`/user_guide/data_conventions`.

## 3. Fit the model and predict shape modes

{class}`vampire.Vampire` accepts the property table directly (it reads the
`contour` column):

```{code-cell} ipython3
model = vp.Vampire(n_clusters=5, random_state=0)
model.fit(props)
props["shape_mode"] = model.predict(props)
props["shape_mode"].value_counts().sort_index()
```

Shape modes are integers `1 ... n_clusters`, numbered left to right in the
dendrogram that links similar modes (so modes 1 and 2 are more alike than
modes 1 and 5, roughly). Pass `random_state` to make K-means reproducible.

A few fitted attributes (all names ending in `_`):

```{code-cell} ipython3
print("principal components kept:", model.n_components_)
print("mean contour shape:", model.mean_contour_.shape)
print("mean contour of each mode:", model.mean_cluster_contours_.shape)
```

## 4. Summarize

```{code-cell} ipython3
distribution = vp.shape_mode_distribution(props["shape_mode"], model.n_clusters)
print("fraction per mode:", distribution.round(3))
print("Shannon entropy (bits):", round(vp.shannon_entropy(distribution, base=2), 3))
```

Per-image (or per-group) distributions use an ordinary pandas `groupby`:

```{code-cell} ipython3
props.groupby("filename")["shape_mode"].value_counts(normalize=True).unstack().round(3)
```

## 5. Plot

{func}`vampire.plotting.plot_summary` stacks the shape-mode distribution
(with the mode numbers underneath), the mean contour of each mode, and the
dendrogram, whose branches point up to the modes they join:

```{code-cell} ipython3
fig, axes = vp.plotting.plot_summary(model, props["shape_mode"])
```

{func}`vampire.plotting.plot_labeled_image` colors each object of an image by
its shape mode, with the same colors:

```{code-cell} ipython3
first = props[props["filename"] == "img1"]
ax = vp.plotting.plot_labeled_image(
    images[0], first, first["shape_mode"], n_clusters=model.n_clusters
)
```

Save any figure with `fig.savefig("summary.png", dpi=300)` (for an `Axes`,
use `ax.figure.savefig(...)`).

## Next steps

- {doc}`your_images` -- your own folders, experimental groups,
  saving results.
- {doc}`/user_guide/method` -- what each step of the model does.
- {doc}`/examples/plotting` -- all figures.
- {doc}`/examples/new_condition` -- label a new experimental condition with
  an existing model.
- {doc}`/api/index` -- full reference.
