# AGENTS.md — guide for coding agents (and humans)

`vampire-analysis` (import `vampire`, alias `vp`) clusters object contours from
segmented microscopy images into **shape modes** (VAMPIRE; Phillip et al.,
*Nat Protoc* 16, 754–774, 2021). The API follows scikit-learn conventions.

## Commands

```bash
uv sync                                   # create .venv with all dev groups
uv run pytest                             # unit tests + doctests (src/ and tests/)
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Python 3.13+. Dependencies live in `pyproject.toml`; never edit `uv.lock` by hand
(`uv lock` regenerates it).

## Layout

| Path | Purpose |
|-|-|
| `src/vampire/contours.py` | Pure functions on one contour: `resample_contour`, `register_contour`, `align_contour` |
| `src/vampire/preprocessing.py` | sklearn transformers: `ContourResampler`, `ContourRegistrar`, `ContourAligner` |
| `src/vampire/model.py` | `Vampire` estimator (resample → register → align → PCA → Normalizer → KMeans → dendrogram ordering) |
| `src/vampire/extraction.py` | `extract_contour`, `extract_properties` (scikit-image regionprops) |
| `src/vampire/io.py` | `read_images`, `label_by_regex` |
| `src/vampire/metrics.py` | `shape_mode_distribution`, `shannon_entropy` |
| `src/vampire/plotting.py` | Matplotlib figures of fitted models |
| `src/vampire/datasets.py` | `load_example_images` (bundled in `src/vampire/_data/`), `make_shapes` (synthetic) |
| `src/vampire/_validation.py` | Shared input checks (`check_contours`, `check_dense_contours`) |
| `src/vampire/_linalg.py` | `kabsch_rotation` |
| `tests/` | pytest suite, one `test_<module>.py` per module, `tests/data/` for fixtures |
| `docs/` | Sphinx (MyST markdown, executed tutorials via myst-nb) |

## Data conventions (invariants — do not break)

- A **contour** is a float array of shape `(2, n)`: row 0 = x, row 1 = y, points
  ordered along a closed boundary, first point **not** repeated at the end.
- A **contour set** `X` is any of: a list of `(2, n_i)` arrays (ragged allowed
  before resampling), an ndarray `(n_samples, 2, n)`, a pandas Series of arrays,
  or a DataFrame with a `"contour"` column (the output of `extract_properties`).
  Always normalize input via `vampire._validation.check_contours` /
  `check_dense_contours`.
- Flattened contours are `(n_samples, 2 * n_points)` with all x first, then all y
  (i.e. `contours.reshape(n_samples, -1)` of a `(n_samples, 2, n_points)` array).
- Shape-mode labels are 0-based ints ordered by dendrogram leaf order.

## Code conventions

- scikit-learn estimator rules: `__init__` only stores parameters (no
  validation, no logic); learned state goes in attributes with a trailing
  underscore set in `fit`; `fit` returns `self`; use
  `sklearn.utils.validation.check_is_fitted` before using fitted state.
- Full type hints. numpydoc docstrings that state array **shapes and dtypes**;
  every public function/class has an `Examples` section that runs as a doctest.
- No I/O side effects outside `io.py`; no `plt.show()`; no global matplotlib
  style changes; no printing.
- Raise `ValueError` with an actionable message for bad input.
- Uppercase single-letter matrix names (`A`, `R`, `U`) are allowed in
  `contours.py` and `_linalg.py` to mirror the published math.
- Tests: deterministic (`random_state`), fast (whole suite < ~1 min), prefer
  property checks on synthetic data (`datasets.make_shapes`) over golden files.

## Reference implementation

`_legacy/` holds the pre-1.0 package (OpenCV, pickled fixtures) for reference
while porting. It is not importable as `vampire` and will be deleted before
release; do not import from it.
