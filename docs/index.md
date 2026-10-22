# vampire-analysis

**VAMPIRE** (Visually Aided Morpho-Phenotyping Image Recognition) quantifies the
heterogeneity of cell and nucleus shapes in segmented microscopy images
([Phillip et al., *Nat Protoc* 2021](https://doi.org/10.1038/s41596-020-00432-x);
see {doc}`references`). It traces the outline of every
object, normalizes away position, size, and rotation, and groups the outlines
into a small number of **shape modes**. Comparing how often each shape mode
occurs between experimental groups (for example control vs. oxygen-glucose
deprivation) turns "the cells look different" into numbers.

`vampire-analysis` is a Python package with a
[scikit-learn](https://scikit-learn.org)-style API: `fit`, `predict`,
`transform`, pipelines, and grid search all work as you would expect.

## Install

```bash
pip install vampire-analysis    # or: uv add vampire-analysis
```

Requires Python 3.13 or newer. See {doc}`installation` for details.

## Ten-line quickstart

```python
import vampire as vp

images, filenames = vp.datasets.load_example_images()  # two 512x512 label images
props = vp.extract_properties(images, filenames)  # one row per object
model = vp.Vampire(n_clusters=5, random_state=0).fit(props)
props["shape_mode"] = model.predict(props)  # 0-based shape-mode labels

fig, axes = vp.plotting.plot_summary(model, props["shape_mode"])
fig.savefig("shape_modes.png", dpi=200)
```

Replace `load_example_images()` with
`vp.read_images("path/to/segmented_images")` to analyze your own data. The
{doc}`quickstart` runs this code and shows its output.

<div class="vp-cards">
  <div class="vp-card">
    <h3><a href="quickstart.html">Quickstart</a></h3>
    <p>From segmented images to shape modes and figures in five steps.</p>
  </div>
  <div class="vp-card">
    <h3><a href="user_guide/your_images.html">Analyze your images</a></h3>
    <p>Read a folder, label experimental groups, compare shape-mode distributions, save results.</p>
  </div>
  <div class="vp-card">
    <h3><a href="user_guide/method.html">How VAMPIRE works</a></h3>
    <p>Resampling, registration, alignment, PCA, and K-means, with the math and figures.</p>
  </div>
  <div class="vp-card">
    <h3><a href="api/index.html">API reference</a></h3>
    <p>Every public function and class, with array shapes and runnable examples.</p>
  </div>
  <div class="vp-card">
    <h3><a href="for_agents.html">For coding agents</a></h3>
    <p>Compact summary of signatures, shapes, invariants, and pitfalls. Also served as <a href="llms.txt">llms.txt</a>.</p>
  </div>
  <div class="vp-card">
    <h3><a href="changelog.html">What's new in 1.0</a></h3>
    <p>Release notes and a migration table from the pre-1.0 API.</p>
  </div>
</div>

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

user_guide/method
user_guide/data_conventions
user_guide/your_images
user_guide/custom_pipelines
user_guide/plotting
user_guide/differences_from_gui
```

```{toctree}
:hidden:
:caption: Reference

api/index
for_agents
changelog
references
license
```
