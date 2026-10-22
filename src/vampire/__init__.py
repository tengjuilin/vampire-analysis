"""VAMPIRE: Visually Aided Morpho-Phenotyping Image Recognition.

Quantify and visualize the heterogeneity of cell and nucleus shapes with a
scikit-learn-style API. The typical workflow is::

    import vampire as vp

    images, filenames = vp.read_images("path/to/segmented", pattern="cortex")
    props = vp.extract_properties(images, filenames)
    model = vp.Vampire(n_clusters=5, random_state=0).fit(props)
    props["shape_mode"] = model.predict(props)
    vp.plotting.plot_summary(model, props["shape_mode"])

See https://vampire.readthedocs.io for the full documentation.
"""

from importlib.metadata import version

from . import contours, datasets, metrics, plotting
from .extraction import extract_contour, extract_properties
from .io import label_by_regex, read_images
from .metrics import shannon_entropy, shape_mode_distribution
from .model import Vampire
from .preprocessing import ContourAligner, ContourRegistrar, ContourResampler

__version__ = version("vampire-analysis")

__all__ = [
    "ContourAligner",
    "ContourRegistrar",
    "ContourResampler",
    "Vampire",
    "__version__",
    "contours",
    "datasets",
    "extract_contour",
    "extract_properties",
    "label_by_regex",
    "metrics",
    "plotting",
    "read_images",
    "shannon_entropy",
    "shape_mode_distribution",
]
