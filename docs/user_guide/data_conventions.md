---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Data conventions

This page defines the data structures every function in `vampire` accepts and
returns. They are invariants of the package: code written against them keeps
working across 1.x releases.

```{code-cell} ipython3
import numpy as np
import pandas as pd

import vampire as vp
```

## Images

Input images are 2-D arrays of shape `(height, width)`, i.e. **already
segmented**. `vampire` does not segment raw micrographs; use your usual tool
(Fiji/ImageJ thresholding, CellProfiler, Cellpose, scikit-image, ...) and save
the masks.

{func}`~vampire.extract_properties` decides how to turn each image into
objects with its `binary` keyword:

| `binary=` | Image is treated as | Objects |
|-|-|-|
| `None` (default) | auto-detected: a `bool` image, or one with a single non-zero value (e.g. `{0, 1}` or `{0, 255}`), is **binary**; an integer-valued image with several labels is a **label image**; non-integer floats raise `ValueError` | |
| `True` | binary mask | 8-connected components of `image > 0` ({func}`skimage.measure.label`) |
| `False` | label image | pixels with value `l > 0` form object `l`, used as is |

Touching objects in a *binary* mask merge into one object; save a label image
if your segmentation separates touching cells, and pass `binary=False` if a
label image happens to contain only one object (otherwise it is auto-detected
as binary, which gives the same objects but relabels them to 1).
{func}`~vampire.plotting.plot_labeled_image` takes the same `binary` keyword
and must interpret an image the same way as `extract_properties` did.

```{code-cell} ipython3
touching = np.zeros((6, 10), dtype=np.uint8)
touching[1:5, 1:5] = 1  # object 1
touching[1:5, 5:9] = 2  # object 2, touching object 1
print("label image:", len(vp.extract_properties([touching])), "objects")
print("binary=True:", len(vp.extract_properties([touching], binary=True)), "object")
```

## Contour

A **contour** is a `float64` array of shape `(2, n)`:

- row 0 = x = column index, row 1 = y = row index (image pixel coordinates,
  **y grows downward**). Contours are never flipped to a y-up frame; the
  shape-mode plots in {mod}`vampire.plotting` invert their y axis so shapes look
  as they do in the image. When you plot contours yourself with Matplotlib, call
  `ax.invert_yaxis()` (or `ax.imshow` the image first) to match;
- the `n` points are ordered along the closed outline; the first point is
  **not** repeated at the end;
- `n >= 3`. Every non-empty object yields a contour with at least 4 points
  (even a single pixel), so {func}`~vampire.extract_properties` keeps all
  objects; the single-contour functions in {mod}`vampire.contours` need at
  least 3 distinct points.

```{code-cell} ipython3
mask = np.zeros((7, 7), dtype=bool)
mask[2:5, 2:5] = True  # a 3 x 3 pixel square, pixel centers at 2, 3, 4
contour = vp.extract_contour(mask)
print(contour.shape)
print(contour)
```

Contours are traced with {func}`skimage.measure.find_contours` at level 0.5,
so they run **between pixel centers**, half a pixel outside the centers of the
object's edge pixels: the square above spans x = 1.5 ... 4.5, not 2 ... 4.
Diagonal steps appear at corners. Pixels of one object that touch only
diagonally stay in one contour (`fully_connected="high"`). The orientation of a traced contour
(clockwise or not) is whatever `find_contours` returns; registration makes all
contours counterclockwise later, so you never need to fix it yourself.

## Contour set

A **contour set** `X` is any of the following; every transformer, the model,
and the plotting functions that take contours accept all of them:

| Form | Example | Notes |
|-|-|-|
| `list` of `(2, n_i)` arrays | `[c1, c2, c3]` | lengths `n_i` may differ |
| ndarray `(n_samples, 2, n)` | `np.stack([c1, c2])` | equal lengths |
| `pandas.Series` of `(2, n_i)` arrays | `props["contour"]` | |
| `pandas.DataFrame` with a `"contour"` column | `props` | output of {func}`~vampire.extract_properties`; other columns are ignored |

{class}`~vampire.ContourRegistrar` and {class}`~vampire.ContourAligner` need
equal-length contours (resample first) and also accept the **flattened** form
`(n_samples, 2 * n)` with all x coordinates first, then all y:

```{code-cell} ipython3
X = vp.datasets.make_shapes(n_samples=3, n_points=6, random_state=0)
dense = np.stack(X)  # (3, 2, 6)
flat = dense.reshape(len(dense), -1)  # (3, 12): x0..x5, y0..y5
print(dense.shape, flat.shape)
print(np.array_equal(flat[0, :6], dense[0, 0]), np.array_equal(flat[0, 6:], dense[0, 1]))
```

A single contour is **not** a contour set: passing one `(2, n)` array to a
transformer or the model raises `ValueError`. Wrap it in a list: `[contour]`.

```{code-cell} ipython3
try:
    vp.ContourResampler().fit_transform(contour)
except ValueError as err:
    print("ValueError:", err)
print(vp.ContourResampler(n_points=8).fit_transform([contour]).shape)
```

## Property table

{func}`~vampire.extract_properties` returns a {class}`pandas.DataFrame` with
one row per object and a fresh `RangeIndex`:

```{code-cell} ipython3
images, filenames = vp.datasets.load_example_images()
props = vp.extract_properties(images, filenames)
props.dtypes.to_frame("dtype").T
```

| Column | Meaning |
|-|-|
| `filename` | name passed in `filenames` (only present if given; first column) |
| `image_id` | 0-based position of the image in `images` |
| `label` | object's label value in its (labeled) image |
| `centroid_x`, `centroid_y` | centroid, column / row coordinate in pixels |
| `area`, `bbox_area`, `convex_area`, `filled_area` | areas in pixels² |
| `perimeter` | perimeter in pixels ({func}`skimage.measure.regionprops`) |
| `equivalent_diameter` | diameter of the circle with the same area |
| `major_axis_length`, `minor_axis_length` | axes of the ellipse with the same second moments |
| `orientation` | angle between the row axis and the major axis, radians |
| `euler_number` | 1 minus the number of holes |
| `eccentricity`, `solidity`, `extent` | standard regionprops shape descriptors |
| `circularity` | `4 * pi * area / perimeter**2` |
| `aspect_ratio` | `major_axis_length / minor_axis_length` (`inf` for zero-width objects) |
| `contour` | the object's contour, `(2, n)` array in image pixel coordinates (object dtype column) |

All measurements are in pixels; multiply by your pixel size to get physical
units. You may add columns freely (e.g. `shape_mode`, experimental group); the
model only reads `contour`.

## Shape-mode labels

`Vampire.predict` returns an integer ndarray of shape `(n_samples,)` with values
in `0 ... n_clusters - 1`:

- labels are **0-based** (the pre-1.0 package and the GUI plotted them 1-based);
- labels follow the dendrogram leaf order, so similar modes have adjacent ids;
- mode `k` corresponds to `model.mean_cluster_contours_[k]`,
  `model.cluster_centers_[k]`, column `k` of `model.transform(X)`, and color
  `k` in every {mod}`vampire.plotting` figure.

```{code-cell} ipython3
model = vp.Vampire(n_clusters=5, random_state=0).fit(props)
labels = model.predict(props)
print(labels.dtype, labels.shape, labels.min(), labels.max())
print(model.mean_cluster_contours_.shape, model.transform(props).shape)
```

## Normalized contours

`model.normalize(X)` returns contours after resampling, registration, and
alignment, shape `(n_samples, 2, n_points)`. They are centered at the origin,
have unit root-mean-square radius, have positive signed area (counterclockwise
in a y-up frame), and are dimensionless;
they live in the same frame as `model.mean_contour_` and
`model.mean_cluster_contours_`, *not* in image pixel coordinates.
