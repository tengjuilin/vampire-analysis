# vampire-analysis

**VAMPIRE** (Visually Aided Morpho-Phenotyping Image Recognition) quantifies the
heterogeneity of cell and nucleus shapes in segmented microscopy images
([Phillip et al., *Nat Protoc* 2021](https://doi.org/10.1038/s41596-020-00432-x);
see {doc}`/user_guide/references`). It traces the outline of every
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

Requires Python 3.13 or newer. See {doc}`/user_guide/installation` for details.

## Ten-line quickstart

```python
import vampire as vp

images, filenames = vp.datasets.load_example_images()  # two 512x512 label images
props = vp.extract_properties(images, filenames)  # one row per object
model = vp.Vampire(n_clusters=5, random_state=0).fit(props)
props["shape_mode"] = model.predict(props)  # shape modes 1..5

fig, axes = vp.plotting.plot_summary(model, props["shape_mode"])
fig.savefig("shape_modes.png", dpi=200)
```

Replace `load_example_images()` with
`vp.read_images("path/to/segmented_images")` to analyze your own data. The
{doc}`/tutorials/quickstart` runs this code and shows its output.

<div class="vp-cards">
  <div class="vp-card">
    <h3><a href="user_guide/index.html">User Guide</a></h3>
    <p>Installation, how VAMPIRE works, data conventions, differences from the GUI, and how to cite.</p>
  </div>
  <div class="vp-card">
    <h3><a href="tutorials/index.html">Tutorials</a></h3>
    <p>From segmented images to shape modes and figures, then the same for your own folders and groups.</p>
  </div>
  <div class="vp-card">
    <h3><a href="examples/index.html">Examples</a></h3>
    <p>Plotting gallery, applying a model to a new condition, choosing the number of shape modes, custom pipelines.</p>
  </div>
  <div class="vp-card">
    <h3><a href="api/index.html">API Reference</a></h3>
    <p>Every public function and class with array shapes and runnable examples, plus a <a href="api/for_agents.html">summary for coding agents</a> (also as <a href="llms.txt">llms.txt</a>).</p>
  </div>
  <div class="vp-card">
    <h3><a href="releases/index.html">Releases</a></h3>
    <p>What's new in 1.0 and a migration table from the pre-1.0 API.</p>
  </div>
</div>

```{toctree}
:hidden:

user_guide/index
tutorials/index
examples/index
api/index
releases/index
```
