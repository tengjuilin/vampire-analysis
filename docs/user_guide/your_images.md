---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Analyzing your own images

A typical experiment has segmented images from several conditions (control vs.
treatment, brain regions, time points) in one or more folders. The recommended
workflow:

1. read the images (optionally filtering file names with regular expressions);
2. extract contours and properties;
3. annotate each object with its experimental group from the file name;
4. **fit one model on the pooled data** of all groups, so every group is
   described by the *same* shape modes;
5. compare shape-mode distributions between groups;
6. save the table, the figures, and the model.

To make this page runnable, we first create a small folder of example files.
Each of the two bundled 512 x 512 images is cut into four 256 x 256 tiles and
saved as TIFF; we *pretend* the first image is a control slice and the second
an OGD (oxygen-glucose deprivation) slice. Your own folder would simply
contain your segmented images.

```{code-cell} ipython3
import tempfile
from pathlib import Path

import numpy as np
from skimage.io import imsave

import vampire as vp

workdir = Path(tempfile.mkdtemp())  # stand-in for e.g. Path("D:/experiment-42")
image_dir = workdir / "segmented"
image_dir.mkdir()

images, _ = vp.datasets.load_example_images()
for group, slice_id, image in [("control", 1, images[0]), ("ogd", 2, images[1])]:
    for tile, (r, c) in enumerate([(0, 0), (0, 256), (256, 0), (256, 256)], start=1):
        name = f"{group}_slice{slice_id}_cortex_c1_{tile}.tif"
        imsave(image_dir / name, image[r : r + 256, c : c + 256], check_contrast=False)

sorted(p.name for p in image_dir.iterdir())
```

## 1. Read a directory

{func}`vampire.read_images` reads every image file (`.tif`, `.tiff`, `.png`,
`.jpg`, `.jpeg`, `.bmp`, `.gif`; not recursive) in sorted file-name order and
returns two lists: the images and their file names.

```{code-cell} ipython3
images, filenames = vp.read_images(image_dir)
print(len(images), images[0].shape, images[0].dtype)
print(filenames[:3])
```

`pattern` keeps only files whose name matches a regular expression
({func}`re.search`). With a list, a file must match **all** patterns (AND):

```{code-cell} ipython3
_, ogd_files = vp.read_images(image_dir, pattern="ogd")
_, tiles_1_2 = vp.read_images(image_dir, pattern=["cortex", r"_[12]\.tif$"])
print(ogd_files)
print(tiles_1_2)
```

For "either A or B" use regex alternation, e.g. `pattern="control|ogd"`.
Channel filters such as `pattern="c1"` are useful when one folder holds
several channels.

## 2. Extract properties

```{code-cell} ipython3
props = vp.extract_properties(images, filenames)
print(props.shape)
props[["filename", "image_id", "label", "area", "circularity"]].head()
```

If your images are spread over several folders, read each folder and
concatenate the lists before calling `extract_properties` (or concatenate
property tables with {func}`pandas.concat` using `ignore_index=True`).

## 3. Label experimental groups

{func}`vampire.label_by_regex` adds metadata columns by matching each file
name against regular expressions. Pass a mapping
`{regex: {column: value, ...}}`:

```{code-cell} ipython3
props = vp.label_by_regex(
    props,
    {
        "control": {"group": "control"},
        "ogd": {"group": "OGD"},
    },
)
# Values captured from the file name itself are easiest with pandas:
props["slice"] = props["filename"].str.extract(r"slice(\d+)", expand=False).astype(int)
props.groupby(["group", "slice"]).size()
```

or a DataFrame with a `regex` column, e.g. read from a spreadsheet you keep
next to the images (`pd.read_csv("groups.csv")` or `pd.read_excel(...)`):

```{code-cell} ipython3
import pandas as pd

annotations = pd.DataFrame(
    {
        "regex": ["control", "ogd"],
        "group": ["control", "OGD"],
        "treatment_min": [0, 30],
    }
)
props = vp.label_by_regex(props, annotations)
props[["filename", "group", "treatment_min"]].drop_duplicates()
```

Rules are applied in order, so when several regexes match a row, later ones
win. New columns are `NaN` for rows that match no regex; columns that already
exist (like `group` above) keep their values in rows no rule sets. Missing
values (`NaN`/`None`) in an annotation are skipped, so a blank spreadsheet cell
never overwrites an earlier match.

## 4. Fit on pooled data, predict per object

Fit **one** model on objects from all groups. Fitting separate models per group
would produce different, incomparable shape modes. If groups have very
different numbers of objects, consider fitting on a balanced subsample so the
largest group does not dominate the modes, and then predict on everything:

```{code-cell} ipython3
n_per_group = props["group"].value_counts().min()
balanced = props.groupby("group").sample(n=n_per_group, random_state=0)

model = vp.Vampire(n_clusters=5, random_state=0).fit(balanced)
props["shape_mode"] = model.predict(props)
props["distance_to_mode"] = model.transform(props).min(axis=1)
print("fitted on", len(balanced), "objects; labeled", len(props))
```

`model.transform(props)` returns an `(n_objects, n_clusters)` array of
distances to each shape-mode center; its row minimum measures how typical an
object is of its mode.

## 5. Compare groups

Shape-mode distribution and Shannon entropy per group. Always pass
`n_clusters` so that empty modes are counted and all vectors have the same
length:

```{code-cell} ipython3
k = model.n_clusters
summary = pd.DataFrame(
    {
        group: vp.shape_mode_distribution(df["shape_mode"], n_clusters=k)
        for group, df in props.groupby("group")
    }
).T
summary.columns = [f"mode {i}" for i in range(k)]
summary["entropy_bits"] = [
    vp.shannon_entropy(row, base=2) for row in summary.iloc[:, :k].to_numpy()
]
summary["n_objects"] = props.groupby("group").size()
summary.round(3)
```

Entropy ranges from 0 (all objects in one mode) to $\log_2 K$ bits (objects
spread evenly over all $K$ modes) and is a single-number summary of shape
heterogeneity.

Side-by-side figures with identical mode colors:

```{code-cell} ipython3
import matplotlib.pyplot as plt

groups = ["control", "OGD"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, group in zip(axes, groups):
    labels = props.loc[props["group"] == group, "shape_mode"]
    vp.plotting.plot_distribution(labels, n_clusters=k, ax=ax)
    ax.set_title(f"{group} (n = {len(labels)})")
```

and one summary figure per group:

```{code-cell} ipython3
fig, axes = vp.plotting.plot_summary(
    model, props.loc[props["group"] == "OGD", "shape_mode"]
)
axes[0].set_title("OGD")
fig.savefig(workdir / "shape_modes_ogd.png", dpi=300)
```

For statistics, treat **images or animals, not cells**, as replicates: compute
the distribution per image (`groupby("filename")`) and compare those between
groups, e.g. with a permutation test or a mixed model.

```{code-cell} ipython3
per_image = (
    props.groupby(["group", "filename"])["shape_mode"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=range(k), fill_value=0)
)
per_image.round(2)
```

## 6. Save results

The `contour` column holds arrays and does not round-trip through CSV, so drop
it for spreadsheets and keep a pickle when you need the contours again:

```{code-cell} ipython3
props.drop(columns="contour").to_csv(workdir / "properties.csv", index=False)
props.to_pickle(workdir / "properties.pkl")  # keeps contours; read with pd.read_pickle
summary.to_csv(workdir / "summary_by_group.csv")
```

A fitted model is an ordinary scikit-learn estimator and can be saved with
{mod}`joblib` (installed with scikit-learn) or {mod}`pickle`, then reused to
label new images with the same shape modes:

```{code-cell} ipython3
import joblib

joblib.dump(model, workdir / "vampire_model.joblib")

loaded = joblib.load(workdir / "vampire_model.joblib")
assert (loaded.predict(props) == props["shape_mode"]).all()
sorted(p.name for p in workdir.iterdir())
```

```{note}
Pickled models should be loaded with the same versions of `vampire-analysis`
and scikit-learn that saved them. Keep the property table (or the images) so
you can refit if you upgrade.
```

```{code-cell} ipython3
:tags: [remove-cell]

import shutil

shutil.rmtree(workdir)
```
