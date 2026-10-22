# Changelog

## 1.0.0

A rewrite with a scikit-learn-style API. It requires **Python 3.13+** and is
**not backward compatible** with 0.x: code, saved models (pickles), and output
files from earlier versions do not work with 1.0. See
{doc}`/user_guide/differences_from_gui` for the changes in the method itself.

### Highlights

- {class}`vampire.Vampire`: a scikit-learn estimator (`fit`, `predict`,
  `transform`, `fit_transform`, `score`, `get_params`/`set_params`) that
  accepts contour sets directly, including the property table from
  {func}`vampire.extract_properties`. Works with
  {class}`~sklearn.model_selection.GridSearchCV`, {func}`~sklearn.base.clone`,
  and {mod}`joblib`.
- Contour transformers {class}`~vampire.ContourResampler`,
  {class}`~vampire.ContourRegistrar`, {class}`~vampire.ContourAligner` for
  building custom scikit-learn pipelines, and the single-contour functions in
  {mod}`vampire.contours`.
- In-memory I/O: {func}`vampire.read_images` (regex AND filters) and
  {func}`vampire.label_by_regex` replace spreadsheet-driven batch functions.
  Nothing is written to disk unless you save it.
- {mod}`vampire.metrics`: {func}`~vampire.metrics.shape_mode_distribution` and
  {func}`~vampire.metrics.shannon_entropy`.
- {mod}`vampire.plotting`: consistent colors across figures, every function
  returns Matplotlib objects and accepts `ax`; new
  {func}`~vampire.plotting.plot_labeled_image`.
  {func}`~vampire.plotting.plot_dendrogram` defaults to
  `orientation="bottom"` (root at the bottom, leaves pointing up to the shape
  modes) without axis spines; {func}`~vampire.plotting.plot_summary` shows the
  mode numbers once, under the distribution bars, with the dendrogram under
  the shape modes. Contours are drawn as smooth periodic splines through
  their points (`smooth=True` in {func}`~vampire.plotting.plot_shape_modes`
  and {func}`~vampire.plotting.plot_representatives`).
- Documentation figures are rendered at 300 dpi.
- {mod}`vampire.datasets`: bundled example images and a synthetic shape
  generator.
- Method fixes: equidistant resampling without a duplicated endpoint, proper
  rotations only (mirror images are no longer merged), exact orientation via
  signed area, identical normalization in fit and predict.
- Shape modes are **1-based** and dendrogram-ordered: `labels_`, `predict`,
  and `fit_predict` return `1 ... n_clusters` (left to right in the
  dendrogram), the same numbers every figure shows. Arrays with one row per
  mode keep 0-based indexing (mode `m` is `mean_cluster_contours_[m - 1]`,
  `cluster_centers_[m - 1]`, column `m - 1` of `transform`);
  `dendrogram_["ivl"]` is `["1", ..., "K"]`; `linkage_matrix_` keeps SciPy's
  0-based leaf ids, and the K-means inside `pipeline_` stays 0-based
  (`pipeline_.predict(X) + 1 == predict(X)`).
  {func}`~vampire.metrics.shape_mode_distribution` takes 1-based labels
  (entry `i` is mode `i + 1`, default length `max(labels)`, a 0 label raises
  `ValueError`), and every plotting function takes them directly.
- {func}`vampire.extract_properties` and
  {func}`~vampire.plotting.plot_labeled_image` take `binary=None` (auto-detect
  binary masks vs. label images), `True` (relabel `image > 0`), or `False`
  (use a label image as is). In auto mode a `bool` image or one with a single
  non-zero value is a mask; non-integer float images raise.
- OpenCV is no longer a dependency; contours come from scikit-image
  ({func}`skimage.measure.find_contours`, level 0.5, `fully_connected="high"`
  so diagonally touching pixels of one object stay in one contour). Contours
  stay in image coordinates (y = row, pointing down); shape-mode plots invert
  the y axis so shapes look as they do in the image.
- scikit-learn `set_output` support: `Vampire.transform` gives columns
  `mode1 ... mode{k}`, `ContourAligner` gives `x0 ... y{n-1}`;
  `ContourResampler`/`ContourRegistrar` accept it as a no-op so
  `Pipeline.set_output` works.
- {func}`vampire.label_by_regex` keeps existing columns and skips missing
  annotation values.
- Documentation rebuilt in five sections (User Guide, Tutorials, Examples,
  API Reference, Releases) with a new vector logo; tutorials and examples are
  executed at build time. Machine-readable summaries at
  [`llms.txt`](https://vampire.readthedocs.io/en/latest/llms.txt) and
  {doc}`/api/for_agents`.

### Migration from 0.x

| 0.x | 1.0 |
|-|-|
| `vp.quickstart.fit_models(build_df)` (spreadsheet of paths, model params, filters) | `images, names = vp.read_images(path, pattern=[...])`; `props = vp.extract_properties(images, names)`; `model = vp.Vampire(n_points=..., n_clusters=..., n_components=...).fit(props)` |
| `vp.quickstart.transform_datasets(apply_df)` / `apply_models` | `props["shape_mode"] = model.predict(props)` on the new property table |
| `vp.extraction.extract_properties(img_set_path, filter_info, write=True)` | `vp.extract_properties(*vp.read_images(img_set_path, pattern=filter_info))`; save with `props.drop(columns="contour").to_csv(...)` |
| column `raw_contour` | column `contour` (same `(2, n)` layout: row 0 = x, row 1 = y) |
| `vp.model.Vampire(model_name, n_points=50, n_clusters=5, n_pcs=None, random_state=None)` | `vp.Vampire(n_points=50, n_clusters=5, n_components=0.95, random_state=None)` -- no name; `n_pcs=None` corresponds to `n_components=0.95`, an int `n_pcs` to the same int |
| `model.fit(properties_df)` | `model.fit(props)` (returns the model) |
| `model.transform(df)` returning a DataFrame with `cluster_id`, `distance_to_centroid` | `model.predict(props)` for labels; `model.transform(props)` for `(n, n_clusters)` distances (column `m - 1` = mode `m`; `.min(axis=1)` = distance to own center); `model.normalize(props)` for aligned contours |
| `cluster_id` (plotted as `plot_cluster_id = cluster_id + 1`) | 1-based shape mode from `predict` (= `cluster_id + 1`, the number the old figures showed) |
| `model.mean_registered_contour` / `model.mean_aligned_contour` | `model.pipeline_.named_steps["align"].reference_contour_` / `model.mean_contour_` |
| `model.principal_directions`, `explained_variance_ratio` | `model.pipeline_.named_steps["pca"].components_`, `model.explained_variance_ratio_` |
| `model.centroids`, `model.inertia` | `model.cluster_centers_`, `model.inertia_` |
| `model.mean_cluster_contours`, `model.linkage_matrix`, `model.branches` | `model.mean_cluster_contours_`, `model.linkage_matrix_`, `model.dendrogram_` |
| `vp.plot.plot_distribution_contour_dendrogram(model, ...)` | `fig, axes = vp.plotting.plot_summary(model, labels)` |
| `vp.plot.plot_distribution(properties_df)` | `vp.plotting.plot_distribution(labels, n_clusters=model.n_clusters)` |
| `vp.plot.plot_contours(model)` | `vp.plotting.plot_shape_modes(model)` |
| `vp.plot.plot_dendrogram(model)` | `vp.plotting.plot_dendrogram(model)` |
| `vp.plot.plot_representatives(model, ...)` | `vp.plotting.plot_representatives(model, X)` |
| `vp.plot.plot_contour_dendrogram`, `plot_distribution_contour` | compose `plot_shape_modes`, `plot_dendrogram`, `plot_distribution` on shared axes (see {doc}`/examples/plotting`) |
| `vp.plot.save_fig(...)`, `vp.plot.set_plot_style()` | `fig.savefig(...)`; use Matplotlib styles directly |
| `vp.coloring.label_imgs` / `color_img` | `vp.plotting.plot_labeled_image(image, rows, rows["shape_mode"])` |
| `vp.util.read_pickle(path)` / `write_pickle(path, obj)` | `joblib.load(path)` / `joblib.dump(obj, path)` (or {mod}`pickle`) |
| `vp.util.label_df(properties_df, id_df, target_props, search_prop)` | `vp.label_by_regex(props, annotations, column="filename")` with an annotations DataFrame having a `regex` column |
| `vp.util.get_*_path(...)` | build paths with {mod}`pathlib` |
| `vp.processing.sample_contours`, `register_contours`, `align_contours` | `vp.ContourResampler`, `vp.ContourRegistrar`, `vp.ContourAligner` (or `vp.contours.resample_contour` etc. for one contour) |
| `vp.amath.pca`, `vp.analysis.cluster_contours` | {class}`sklearn.decomposition.PCA`, {class}`sklearn.cluster.KMeans` (used inside `Vampire`) |

## 0.0.1

First release of `vampire-analysis`, based on the `vampireanalysis` GUI
{doc}`[1] </user_guide/references>`. The algorithmic operations were separated from the
GUI and grouped into modules to encourage reuse and improve reproducibility.

**Changes**

- Package API instead of a graphical user interface.
- Information used to build and apply models is stored in a `.csv`/`.xlsx`
  file or a DataFrame instead of being entered at GUI prompts
  (`vampire.model.build_models`, `vampire.model.apply_models`).

**New features**

- Option for the random state of K-means clustering and of plotting
  representative contours, for reproducible results.
- AND filtering of image file names with optional columns, more flexible than
  tags.
- New PCA implementation using SVD or eigendecomposition depending on the
  input matrix, faster than the previous one and than scikit-learn.
- Plots of shape-mode distribution, dendrogram, and mean shape modes, as
  separate and combined figures.

**Improvements**

- Default values for `output_path`, `model_name`, `num_points`, and
  `num_clusters`, used when the corresponding cell is blank in the
  spreadsheet or `None`/`NaN` in the DataFrame.
- Performance, for an image set of 221 images with 11,173 segmented cells:

  | | Build model [s] | Apply model [s] |
  |-|-|-|
  | `vampireanalysis` GUI | 517 | 98 |
  | `vampire-analysis` 0.0.1 | 80 | 26 |
  | Improvement | 85 % faster | 73 % faster |

**Known limitations**

- The number of principal components was fixed at 20, as in the GUI, rather
  than chosen from the explained variance (1.0: `n_components`, default 95 %
  of the variance).
- No scree plot of the PCA explained variance (1.0: `explained_variance_ratio_`;
  see {doc}`/examples/n_shape_modes` for a scree plot).
