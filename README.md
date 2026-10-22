# vampire-analysis

[![Documentation](https://readthedocs.org/projects/vampire/badge/?version=latest)](https://vampire.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/vampire-analysis)](https://pypi.org/project/vampire-analysis/)
[![Python](https://img.shields.io/pypi/pyversions/vampire-analysis)](https://pypi.org/project/vampire-analysis/)
[![License](https://img.shields.io/github/license/tengjuilin/vampire-analysis)](LICENSE)
[![Tests](https://github.com/tengjuilin/vampire-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/tengjuilin/vampire-analysis/actions/workflows/tests.yml)
[![Lint](https://github.com/tengjuilin/vampire-analysis/actions/workflows/lint.yml/badge.svg)](https://github.com/tengjuilin/vampire-analysis/actions/workflows/lint.yml)
[![Docs](https://github.com/tengjuilin/vampire-analysis/actions/workflows/docs.yml/badge.svg)](https://github.com/tengjuilin/vampire-analysis/actions/workflows/docs.yml)

**VAMPIRE** (Visually Aided Morpho-Phenotyping Image Recognition) quantifies and
visualizes the heterogeneity of cell and nucleus shapes in segmented microscopy
images [1]. It traces the outline of every object, removes position, size, and
rotation, and clusters the outlines into a few **shape modes**, whose
frequencies can then be compared between experimental groups. VAMPIRE has been
used, for example, to study microglial shape changes after oxygen-glucose
deprivation [2] and cell morphology in cancer metastasis [3].

`vampire-analysis` implements the method of the
[`vampireanalysis`](https://pypi.org/project/vampireanalysis/) GUI [1] as a
Python package with a scikit-learn-style API (`fit`, `predict`, pipelines,
grid search), fully documented array shapes, and tutorials that are executed
as part of the documentation build.

**Documentation:** <https://vampire.readthedocs.io>

## Installation

Requires Python 3.13 or newer.

```bash
pip install vampire-analysis
# or, in a uv project:
uv add vampire-analysis
```

## Quickstart

```python
import vampire as vp

# Segmented images: binary masks or label images (background 0).
# Use vp.read_images("path/to/folder") for your own images.
images, filenames = vp.datasets.load_example_images()

props = vp.extract_properties(images, filenames)  # pandas DataFrame, one row per object
model = vp.Vampire(n_clusters=5, random_state=0).fit(props)
props["shape_mode"] = model.predict(props)  # 0-based shape-mode labels

distribution = vp.shape_mode_distribution(props["shape_mode"], model.n_clusters)
print(distribution, vp.shannon_entropy(distribution, base=2))

fig, axes = vp.plotting.plot_summary(model, props["shape_mode"])
fig.savefig("shape_modes.png", dpi=300)
```

To compare experimental groups, label rows from their file names, fit one
model on the pooled data, and compare the per-group distributions:

```python
props = vp.label_by_regex(
    props, {"img1": {"group": "control"}, "img2": {"group": "OGD"}}
)
props.groupby("group")["shape_mode"].value_counts(normalize=True).unstack()
```

See the [quickstart](https://vampire.readthedocs.io/en/latest/quickstart.html),
[analyzing your own images](https://vampire.readthedocs.io/en/latest/user_guide/your_images.html),
and [how VAMPIRE works](https://vampire.readthedocs.io/en/latest/user_guide/method.html).
Upgrading from 0.x? See the
[changelog and migration table](https://vampire.readthedocs.io/en/latest/changelog.html).

## Development

```bash
git clone https://github.com/tengjuilin/vampire-analysis.git
cd vampire-analysis
uv sync
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Contributor and coding-agent guidelines are in [AGENTS.md](AGENTS.md).

## Citation

If you use this package, please cite the VAMPIRE method [1] and the package
version you used.

## References

[1] Phillip, J. M., Han, K.-S., Chen, W.-C., Wirtz, D. & Wu, P.-H. A robust
unsupervised machine-learning method to quantify the morphological
heterogeneity of cells and nuclei. *Nat Protoc* **16**, 754–774 (2021).
<https://doi.org/10.1038/s41596-020-00432-x>

[2] Joseph, A., Liao, R., Zhang, M., et al. Nanoparticle-microglial interaction
in the ischemic brain is modulated by injury duration and treatment.
*Bioeng Transl Med* **5**, e10175 (2020). <https://doi.org/10.1002/btm2.10175>

[3] Wu, P.-H., Phillip, J. M., Khatau, S. B., et al. Evolution of cellular
morpho-phenotypes in cancer metastasis. *Sci Rep* **5**, 18437 (2016).
<https://doi.org/10.1038/srep18437>
