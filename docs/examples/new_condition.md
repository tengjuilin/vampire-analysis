---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Applying a model to a new condition

A fitted {class}`vampire.Vampire` defines a fixed set of shape modes. Calling
`predict` on new contours assigns them to those modes without refitting, so
objects from a new condition, a later experiment, or another lab are
described in the same terms as the data the model was fitted on.

There are two common designs:

- **Pooled model:** fit on all groups of one study, then predict each group
  (the default workflow of {doc}`/tutorials/your_images`).
- **Reference model:** fit on a reference group (for example control
  animals, or an earlier experiment), save the model, and apply it to new
  conditions as they come in. This page shows this design.

Here we pretend that the first bundled image is a control slice and the second
a treated one.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vampire as vp

images, filenames = vp.datasets.load_example_images()
props = vp.extract_properties(images, filenames)
props = vp.label_by_regex(
    props, {"img1": {"group": "control"}, "img2": {"group": "treated"}}
)
props["group"].value_counts()
```

## Fit on the reference group

```{code-cell} ipython3
control = props[props["group"] == "control"]
model = vp.Vampire(n_clusters=5, random_state=0).fit(control)
print("fitted on", len(control), "control objects")
```

## Predict the new condition

`predict` normalizes the new contours with the *fitted* reference contour,
PCA, and cluster centers, and returns shape modes `1 ... n_clusters`. For the
reference group it reproduces `model.labels_`:

```{code-cell} ipython3
props["shape_mode"] = model.predict(props)
assert (props.loc[control.index, "shape_mode"] == model.labels_).all()
props.groupby("group")["shape_mode"].value_counts().unstack(fill_value=0)
```

## Compare distributions and entropy

Pass `n_clusters` so every group gets a vector of the same length, with zeros
for empty modes:

```{code-cell} ipython3
k = model.n_clusters
summary = pd.DataFrame(
    {
        group: vp.shape_mode_distribution(df["shape_mode"], n_clusters=k)
        for group, df in props.groupby("group")
    },
    index=[f"mode {m}" for m in range(1, k + 1)],
).T
summary["entropy_bits"] = [
    vp.shannon_entropy(row, base=2) for row in summary.iloc[:, :k].to_numpy()
]
summary.round(3)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, (group, df) in zip(axes, props.groupby("group")):
    vp.plotting.plot_distribution(df["shape_mode"], n_clusters=k, ax=ax)
    ax.set_title(f"{group} (n = {len(df)})")
```

The summary figure of the new condition shows its distribution over the
reference shape modes:

```{code-cell} ipython3
fig, axes = vp.plotting.plot_summary(
    model, props.loc[props["group"] == "treated", "shape_mode"]
)
axes[0].set_title("treated, reference model fitted on control");
```

## Check that the reference modes still fit

A reference model can only describe shapes it has seen. `model.transform`
gives the distance of every object to each mode center (column `m - 1` is
mode `m`); the row minimum is the distance to its own mode. If many objects of
the new condition are farther from every mode than nearly all reference
objects, the new condition contains shapes the reference modes do not
capture, and a pooled model is the better choice.

```{code-cell} ipython3
props["distance_to_mode"] = model.transform(props).min(axis=1)
cutoff = props.loc[control.index, "distance_to_mode"].quantile(0.95)
far = props["distance_to_mode"] > cutoff
print(f"95th percentile of control distances: {cutoff:.3f}")
far.groupby(props["group"]).mean().round(3).rename("fraction beyond cutoff")
```

About 5 % of the reference objects lie beyond the cutoff by construction; a
much larger fraction in the new condition is a warning sign.

## Keep the model

Save the fitted reference model with {mod}`joblib` and load it whenever a new
condition is imaged; see {doc}`/tutorials/your_images` (section "Save
results") for saving models and property tables. Load models with the same
versions of `vampire-analysis` and scikit-learn that saved them.
