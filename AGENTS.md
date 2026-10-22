# AGENTS.md — guide for coding agents (and humans)

`vampire-analysis` (import `vampire`, alias `vp`) clusters object contours from
segmented microscopy images into **shape modes** (VAMPIRE; Phillip et al.,
*Nat Protoc* 16, 754–774, 2021). The API follows scikit-learn conventions.

## Commands

```bash
uv sync                                   # create .venv with all dev groups
uv run pytest                             # unit tests + doctests (src/ and tests/)
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -W --keep-going -b html docs docs/_build/html   # zero warnings required
NB_EXECUTION_MODE=cache uv run sphinx-build -b html docs docs/_build/html  # faster local rebuilds
```

Python 3.13+. Dependencies live in `pyproject.toml`; never edit `uv.lock` by hand
(`uv lock` regenerates it).

## Layout

| Path | Purpose |
|-|-|
| `src/vampire/contours.py` | Pure functions on one contour: `signed_area`, `resample_contour`, `register_contour`, `align_contour` |
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
| `docs/` | Sphinx (MyST markdown, executed tutorials via myst-nb); see "Documentation" below |
| `docs/api/index.md` | API reference (autosummary; stubs generated into `docs/api/generated/`, gitignored) |
| `docs/api/for_agents.md` | Compact API/invariants/pitfalls summary; keep in sync with the public API |
| `tools/make_logo.py` | Generates the logos in `docs/_static/` (`vampire-logo.svg`, `vampire-logo-dark.svg`, `vampire-icon.svg`/`.png`) |

## Data conventions (invariants — do not break)

- A **contour** is a float array of shape `(2, n)`: row 0 = x, row 1 = y, points
  ordered along a closed boundary, first point **not** repeated at the end.
- A **contour set** `X` is any of: a list of `(2, n_i)` arrays (ragged allowed
  before resampling), an ndarray `(n_samples, 2, n)`, a pandas Series of arrays,
  or a DataFrame with a `"contour"` column (the output of `extract_properties`).
  Always normalize input via `vampire._validation.check_contours` /
  `check_dense_contours`.
- Contours stay in image pixel coordinates: x = column, y = row (y points
  down), traced by `find_contours` at level 0.5 (between pixel centers).
  Plotting code inverts the y axis; never flip coordinates in the data.
- A single `(2, n)` array is not a contour set; transformers and the model
  raise on it (wrap as `[contour]`).
- Flattened contours are `(n_samples, 2 * n_points)` with all x first, then all y
  (i.e. `contours.reshape(n_samples, -1)` of a `(n_samples, 2, n_points)` array).
- Shape modes are **1-based** ints `1..n_clusters` in dendrogram leaf order
  (left to right): `labels_`, `predict`, `fit_predict`, and every metrics and
  plotting function use them; plots show `1..k`, mode `m` at x = `10 * m - 5`.
  Arrays with one row/column per mode stay 0-based: mode `m` is
  `mean_cluster_contours_[m - 1]`, `cluster_centers_[m - 1]`,
  `transform(X)[:, m - 1]` (pandas column `mode{m}`). `linkage_matrix_` keeps
  SciPy's 0-based leaf ids (`dendrogram_["ivl"] == ["1", ..., "k"]`), and the
  KMeans in `pipeline_` is 0-based (`pipeline_.predict(X) + 1 == predict(X)`).
  `shape_mode_distribution(labels)[i]` is the fraction of mode `i + 1`; label 0
  raises `ValueError`.

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

## Documentation

- Sources in `docs/`, MyST markdown only. Pages with `file_format: mystnb`
  front matter are executed at build time (`nb_execution_raise_on_error`), so
  every code cell must run top to bottom with only `vampire` and its
  dependencies, use bundled/synthetic data (`datasets.load_example_images`,
  `datasets.make_shapes`), write only to a temp directory, and stay fast.
- Structure: five top-level sections, which are the navbar entries; each has
  an `index.md` (short intro + toctree) and the root `index.md` is a landing
  page whose hidden toctree lists exactly these five indexes:
  - `user_guide/`: `installation`, `method`, `data_conventions`,
    `differences_from_gui`, `references`, `license`
  - `tutorials/`: `quickstart`, `your_images`
  - `examples/`: `plotting`, `new_condition`, `n_shape_modes`, `custom_pipelines`
  - `api/`: `index` (autosummary) and `for_agents`
  - `releases/`: `changelog`
  Do not add top-level sections; put new pages into one of these. Cross-link
  with absolute `{doc}` targets (e.g. ``{doc}`/user_guide/method` ``).
- New public function/class: add it to the autosummary list in
  `docs/api/index.md` and to `docs/api/for_agents.md`; add a changelog entry
  in `docs/releases/changelog.md`.
- Theme (pydata-sphinx-theme) settings are in `docs/conf.py`: vector logos
  `_static/vampire-logo.svg` / `vampire-logo-dark.svg` (the lockup includes
  the name, so no text title), favicon `_static/vampire-icon.svg` (PNG
  fallback in `_templates/layout.html`), search as a navbar icon only
  (Ctrl/Cmd+K). Navbar height and colors are in `_static/vampire.css`.
- Docstrings are rendered by numpydoc: avoid bare `name_` (trailing
  underscore) in type fields -- RST reads it as a link; write ``name_``.
- `sphinx-llms-txt` emits `llms.txt` and `llms-full.txt` at the HTML root.
- The build must pass with `-W` (CI: `.github/workflows/docs.yml`, Read the
  Docs: `.readthedocs.yaml`). Intersphinx needs network access.
