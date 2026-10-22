# Differences from the GUI and pre-1.0 versions

`vampire-analysis` 1.0 implements the method of the original VAMPIRE GUI
([`vampireanalysis`](https://pypi.org/project/vampireanalysis/),
[source](https://github.com/kukionfr/VAMPIRE_open)) {doc}`[1] <references>`
and of the pre-1.0 releases of this package, but fixes several details. The
overall result -- shape modes, their distributions, and the figures -- is
comparable, but **labels and distributions are not numerically identical** to
those from older versions. Refit models and recompute results with 1.0 rather
than mixing versions within one study.

| Topic | GUI / pre-1.0 | 1.0 |
|-|-|-|
| Interface | GUI; then functions driven by an Excel/CSV sheet of paths (`fit_models`, `transform_datasets`) writing files | scikit-learn estimator (`Vampire().fit(...)`, `predict`) operating on in-memory data; you decide what to save |
| Contour tracing | OpenCV `findContours` on pixel edges | scikit-image `find_contours` at level 0.5 (between pixel centers), diagonal pixels kept in one contour; no OpenCV dependency |
| Resampling | B-spline through the points, `n_points` samples spaced evenly in cumulative chord length (inclusive of both ends), so the last sample duplicates the first | periodic cubic spline; `n_points` samples equidistant in true arc length; start point **not** duplicated |
| Registration rotation | SVD principal axes with arbitrary signs, which can **mirror** a contour | sign-fixed axes forming a proper rotation (det = +1); mirror images stay distinct |
| Direction | guessed from two points assuming local convexity | reversed if the signed (shoelace) area is negative -- exact for any simple contour |
| Alignment | single pass to the mean registered contour; loop over shifts, Kabsch rotation for each | iterated to the mean aligned contour (generalized Procrustes; `ContourAligner(max_iter=1)` for a single pass); all cyclic shifts scored in closed form, then one Kabsch rotation |
| Number of PCs | GUI and 0.0.1: fixed 20; later pre-1.0 code: 95 % variance by default | `n_components` (int or variance fraction, default 0.95), via scikit-learn PCA |
| Normalization | GUI: row normalization when clustering but **not** when assigning new data | same {class}`~sklearn.preprocessing.Normalizer` step in fit and predict, so `predict(X_train)` reproduces `labels_` |
| Labels | 0-based cluster ids reordered by dendrogram, plotted 1-based (`cluster_id + 1`) | dendrogram-ordered, **1-based** everywhere (`1 ... n_clusters`), in labels and figures alike |
| Randomness | random seed drawn when none was given | `random_state=None` follows scikit-learn; pass an int for reproducibility |

## Why these changes matter

**Mirror images.** A contour and its mirror image (for example a cell bending
left vs. right) are different shapes. The pre-1.0 registration could flip a
contour while rotating it onto its principal axes, merging such shapes into
one mode. 1.0 only applies proper rotations, during both registration and
alignment.

**Duplicated endpoint.** Sampling a closed curve at `n_points` positions from
start to end *inclusive* returns the start point twice, giving that point
double weight and leaving one fewer distinct point. 1.0 samples `n_points`
distinct, evenly spaced points.

**Consistent normalization.** If clustering uses normalized PC scores but new
contours are assigned with unnormalized scores, an object can receive a
different label at prediction time than during training. In 1.0 the full
fitted pipeline is reused, so this cannot happen.

**1-based labels.** Shape modes are numbered `1 ... n_clusters` from left to
right in the dendrogram, the same numbers the figures show, so a label in a
table always matches the mode number in a plot. Arrays with one row per mode
keep NumPy indexing: mode `m` is `mean_cluster_contours_[m - 1]`,
`cluster_centers_[m - 1]`, and column `m - 1` of `transform` (named
`mode{m}` with pandas output). Old 0-based `cluster_id` values correspond to
`cluster_id + 1`.

## Migrating code

See the migration table in the {doc}`/releases/changelog`.
