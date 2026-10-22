# API Reference

```python
import vampire as vp
```

This page documents every public function and class. For a compact summary of
signatures, array shapes, invariants, and common pitfalls (also useful for
coding agents and LLMs), see {doc}`for_agents`.

```{toctree}
:hidden:

for_agents
```

Everything below is public and stable for 1.x. Names in the top-level
namespace (`vp.read_images`, `vp.Vampire`, ...) are the recommended spelling;
`vp.contours`, `vp.metrics`, `vp.plotting`, and `vp.datasets` are submodules.
`vp.shape_mode_distribution` and `vp.shannon_entropy` are aliases of the
functions in {mod}`vampire.metrics`.

Shapes use NumPy notation. A **contour** is an array of shape `(2, n)`
(row 0 = x, row 1 = y); a **contour set** is a list of contours, an array of
shape `(n_samples, 2, n)`, a Series of contours, or a DataFrame with a
`"contour"` column. See {doc}`/user_guide/data_conventions`.

```{eval-rst}
.. automodule:: vampire
   :no-members:
```

## Reading images

```{eval-rst}
.. automodule:: vampire.io
   :no-members:

.. currentmodule:: vampire

.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_images
   label_by_regex

.. autodata:: vampire.io.IMAGE_EXTENSIONS
   :no-value:

   File extensions recognized by :func:`vampire.read_images`:
   ``(".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")``.
```

## Extracting contours and properties

```{eval-rst}
.. automodule:: vampire.extraction
   :no-members:

.. currentmodule:: vampire

.. autosummary::
   :toctree: generated/
   :nosignatures:

   extract_properties
   extract_contour
```

## The VAMPIRE model

```{eval-rst}
.. automodule:: vampire.model
   :no-members:

.. currentmodule:: vampire

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Vampire
```

## Contour transformers

```{eval-rst}
.. automodule:: vampire.preprocessing
   :no-members:

.. currentmodule:: vampire

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ContourResampler
   ContourRegistrar
   ContourAligner
```

## Single-contour functions

```{eval-rst}
.. automodule:: vampire.contours
   :no-members:

.. currentmodule:: vampire.contours

.. autosummary::
   :toctree: generated/
   :nosignatures:

   signed_area
   resample_contour
   register_contour
   align_contour
   lowpass_contour
```

## Metrics

```{eval-rst}
.. automodule:: vampire.metrics
   :no-members:

.. currentmodule:: vampire.metrics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   shape_mode_distribution
   shannon_entropy
```

## Plotting

```{eval-rst}
.. automodule:: vampire.plotting
   :no-members:

.. currentmodule:: vampire.plotting

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_summary
   plot_distribution
   plot_shape_modes
   plot_dendrogram
   plot_representatives
   plot_labeled_image
```

## Datasets

```{eval-rst}
.. automodule:: vampire.datasets
   :no-members:

.. currentmodule:: vampire.datasets

.. autosummary::
   :toctree: generated/
   :nosignatures:

   load_example_images
   make_shapes
```
