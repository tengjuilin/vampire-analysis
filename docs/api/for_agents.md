# For coding agents

Compact, machine-oriented reference for `vampire-analysis` {{ version }}.
Plain-text versions of the whole documentation are served at
[`llms.txt`](https://vampire.readthedocs.io/en/latest/llms.txt) (index) and [`llms-full.txt`](https://vampire.readthedocs.io/en/latest/llms-full.txt) (all pages).

## Install and import

```bash
pip install vampire-analysis   # or: uv add vampire-analysis   (Python >= 3.13)
```

```python
import vampire as vp  # distribution name: vampire-analysis; import name: vampire
```

## Canonical workflow

```python
import vampire as vp

images, filenames = vp.read_images(
    "segmented/", pattern=["cortex", "c1"]
)  # AND of regexes
props = vp.extract_properties(images, filenames)  # DataFrame, one row per object
props = vp.label_by_regex(
    props, {"ctrl": {"group": "control"}, "ogd": {"group": "OGD"}}
)
model = vp.Vampire(n_clusters=5, random_state=0).fit(props)  # fit ONCE on pooled groups
props["shape_mode"] = model.predict(props)  # int shape modes 1..n_clusters
dist = {
    g: vp.shape_mode_distribution(d["shape_mode"], model.n_clusters)
    for g, d in props.groupby("group")
}
entropy = {g: vp.shannon_entropy(p, base=2) for g, p in dist.items()}
fig, axes = vp.plotting.plot_summary(model, props["shape_mode"])
fig.savefig("summary.png", dpi=300)
props.drop(columns="contour").to_csv("properties.csv", index=False)
```

Runnable without your own data: replace the first line with
`images, filenames = vp.datasets.load_example_images()`.

## Data types and shapes

| Name | Type / shape | Notes |
|-|-|-|
| image | ndarray `(H, W)` | segmented: binary mask or integer label image (0 = background) |
| contour | float64 ndarray `(2, n)` | row 0 = x (column), row 1 = y (row, **pointing down**), closed, first point not repeated, `n >= 3` |
| contour set `X` | `list[(2, n_i)]` \| ndarray `(n_samples, 2, n)` \| Series of contours \| DataFrame with `"contour"` column | single `(2, n)` array is **not** a contour set -> `ValueError`; wrap as `[c]` |
| flattened contours | ndarray `(n_samples, 2 * n)` | all x then all y = `arr.reshape(n_samples, -1)`; accepted by `ContourRegistrar`, `ContourAligner` |
| property table | DataFrame `(n_objects, 21)` (20 without `filename`) | columns below; fresh `RangeIndex` |
| labels (shape modes) | int ndarray `(n_samples,)` | **1-based** `1..n_clusters`, dendrogram leaf order left to right; 0 raises `ValueError` in metrics/plotting |

Property-table columns: `filename` (only if `filenames` given), `image_id`,
`label`, `centroid_x`, `centroid_y`, `area`, `bbox_area`, `convex_area`,
`filled_area`, `perimeter`, `equivalent_diameter`, `major_axis_length`,
`minor_axis_length`, `orientation`, `euler_number`, `eccentricity`,
`solidity`, `extent`, `circularity`, `aspect_ratio`, `contour`. Units: pixels.

## Public API

```text
vp.read_images(directory, pattern=None) -> (list[ndarray], list[str])
vp.label_by_regex(properties, annotations, column="filename") -> DataFrame
    annotations: {regex: {col: value}} or DataFrame with "regex" column
vp.extract_properties(images, filenames=None, binary=None) -> DataFrame
    binary: None = auto-detect, True = relabel image > 0, False = label image as is
vp.extract_contour(object_mask) -> ndarray (2, n) | None

vp.Vampire(n_points=50, n_clusters=5, n_components=0.95, n_init=3,
           max_iter=300, random_state=None)
    .fit(X) -> self                 .predict(X) -> (n_samples,) int in 1..n_clusters
    .transform(X) -> (n_samples, n_clusters) distances to mode centers; column m-1 = mode m
    .fit_transform(X)               .fit_predict(X) -> labels_ (1-based)
    .get_feature_names_out() -> ["mode1", ..., "mode{n_clusters}"]
    .normalize(X) -> (n_samples, 2, n_points) aligned contours
    .score(X) -> float (= -inertia; increases with n_clusters!)
    fitted: pipeline_, labels_, cluster_centers_ (n_clusters, n_components_),
            mean_contour_ (2, n_points), mean_cluster_contours_ (n_clusters, 2, n_points),
            n_components_, explained_variance_ratio_, inertia_,
            linkage_matrix_ (n_clusters - 1, 4; SciPy 0-based leaf ids),
            dendrogram_ (dict; dendrogram_["ivl"] == ["1", ..., "K"]), n_features_in_
    labels_ (n_samples,) 1-based; row m-1 of cluster_centers_/mean_cluster_contours_ = mode m
    pipeline_ ends with sklearn KMeans (0-based): pipeline_.predict(X) + 1 == predict(X)

vp.ContourResampler(n_points=50)   X -> (n_samples, 2, n_points); stateless
vp.ContourRegistrar()              (n_samples, 2, n) -> same; stateless
vp.ContourAligner(flatten=True)    -> (n_samples, 2 * n) or (n_samples, 2, n)
    fitted: reference_contour_ (2, n), mean_contour_ (2, n), n_points_
    set_output(transform="pandas"): ContourAligner(flatten=True) -> columns x0..x{n-1}, y0..y{n-1};
    Vampire.transform -> columns mode1..mode{k} (input index kept);
    ContourResampler/ContourRegistrar accept it as a no-op (output stays 3-D)

vp.contours.signed_area(contour) -> float  (> 0 counterclockwise in a y-up frame)
vp.contours.resample_contour(contour, n_points=50) -> (2, n_points)
vp.contours.register_contour(contour) -> (2, n)
vp.contours.align_contour(contour, reference) -> (2, n)

vp.shape_mode_distribution(labels, n_clusters=None) -> (n_clusters,) fractions
    # labels in 1..n_clusters; entry i = fraction of mode i+1; default n_clusters = max(labels);
    # label 0 or > n_clusters -> ValueError; n_clusters: int >= 1
vp.shannon_entropy(distribution, base=None) -> float   # nats by default; base > 0, != 1
    (aliases of vp.metrics.*)

vp.plotting.plot_summary(model, labels=None, figsize=None, height_ratios=(4, 1, 1)) -> (Figure, axes(3,))
vp.plotting.plot_distribution(labels, n_clusters=None, ax=None, colors=None) -> Axes
vp.plotting.plot_shape_modes(model, ax=None, colors=None, ...) -> Axes
vp.plotting.plot_dendrogram(model, ax=None, **dendrogram_kwargs) -> Axes
    # default orientation="bottom" (root at bottom, leaves up toward the modes), no spines;
    # also "top"/"left"/"right"; mode m at 10*m - 5 on the leaf axis
# all plotting functions take 1-based labels and show modes 1..k; mode m at x = 10*m - 5
vp.plotting.plot_representatives(model, X, labels=None, n_per_mode=5, ...) -> (Figure, axes(n_clusters,))
vp.plotting.plot_labeled_image(image_label, objects, labels, ax=None, cmap=None,
                               background_color="white", n_clusters=None,
                               binary=None) -> Axes   # binary: same meaning as extract_properties

vp.datasets.load_example_images() -> (list of 2 uint16 (512, 512) label images, ["img1", "img2"])
vp.datasets.make_shapes(n_samples=100, kinds=("circle", "ellipse"), n_points=120,
                        noise=0.01, random_state=None, return_labels=False)
    kinds from {"circle", "ellipse", "triangle", "star"}
```

Full signatures and docstrings: {doc}`index`.

## Pipeline inside `Vampire`

`ContourResampler(n_points) -> ContourRegistrar() -> ContourAligner() ->
PCA(n_components) -> Normalizer() -> KMeans(n_clusters)`, then clusters are
renumbered by the dendrogram (complete linkage, Euclidean) of their mean
aligned contours (KMeans ids `0..K-1` in leaf order; public labels add 1).
`model.pipeline_[:3]` = contour normalization,
`model.pipeline_[:-1]` = K-means input features. See {doc}`/user_guide/method`.

## Invariants

- `predict(X_train)` equals `labels_` (same normalization in fit and predict).
- `dendrogram_["ivl"] == [str(m) for m in range(1, n_clusters + 1)]`
  (`dendrogram_["leaves"] == list(range(n_clusters))`, SciPy 0-based ids).
- Mode `m` (1-based label) <-> `mean_cluster_contours_[m - 1]` <->
  `cluster_centers_[m - 1]` <-> column `m - 1` of `transform(X)` (pandas
  column `f"mode{m}"`) <-> the same color and position `10 * m - 5` in every plot.
- `shape_mode_distribution(labels, k)[m - 1]` = fraction of mode `m`.
- Registration and alignment use proper rotations only; mirror images are not
  merged.
- Resampled contours have `n_points` equidistant points, start not repeated.
- Plotting functions never call `plt.show()`, never save, never change global
  style; they return Matplotlib objects.
- No function writes files (`read_images` only reads).

## Common pitfalls

1. **Fitting per group.** Fit one model on all groups (optionally a balanced
   subsample), then `predict` each group. Separate models give incomparable
   modes.
2. **Grid search with the default score.** `score` = minus inertia, which
   always improves with more clusters; `GridSearchCV` then picks the largest
   `n_clusters`. Use a custom scorer such as the silhouette of
   `est.pipeline_[:-1].transform(X)` ({doc}`/examples/n_shape_modes`).
3. **One principal component.** If `model.n_components_ == 1`, normalized
   features are all +-1 and K-means finds at most 2 modes. Pass an integer
   `n_components >= 2`. Mostly happens on trivial synthetic data.
4. **Forgetting `n_clusters`** in `shape_mode_distribution`,
   `plot_distribution`, `plot_labeled_image`: without it, the length/colors
   follow `max(labels)` and trailing empty modes disappear.
5. **Single contour input.** `model.predict(c)` with `c.shape == (2, n)`
   raises; use `model.predict([c])`.
6. **Binary vs. label images.** Touching objects in a binary mask merge. Pass a
   label image (and `binary=False` if it may contain a single object: in auto
   mode an image with one non-zero value is treated as a mask and relabeled
   to 1). Use the same `binary` value in `plot_labeled_image`. Non-integer
   float images raise in auto mode.
7. **CSV and contours.** The `contour` column holds arrays; drop it for CSV,
   use `to_pickle` to keep it.
8. **y axis.** Contours are in image coordinates (y down). When plotting
   contours yourself, `ax.invert_yaxis()`.
9. **Labels are 1-based**, but per-mode arrays are 0-based: index
   `mean_cluster_contours_[m - 1]`, not `[m]`. Count modes with
   `np.bincount(labels - 1, minlength=k)` or `shape_mode_distribution`.
   Feeding 0-based ids (e.g. from `pipeline_.predict` or a hand-built KMeans)
   to metrics/plotting raises `ValueError`; add 1. Old pickled models and
   `raw_contour` columns from 0.x are incompatible.
10. **Reproducibility.** Pass `random_state=<int>`; K-means is random otherwise.

## Saving and loading a model

```python
import joblib

joblib.dump(model, "vampire_model.joblib")
model = joblib.load(
    "vampire_model.joblib"
)  # same vampire-analysis + scikit-learn versions
```
