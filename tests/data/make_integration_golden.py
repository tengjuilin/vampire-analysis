"""Regenerate ``integration_golden.npz`` for ``tests/test_integration.py``.

Run from the repository root::

    uv run python tests/data/make_integration_golden.py

The golden file pins the end-to-end output of
``vampire.datasets.load_example_images()`` (two 512x512 label images, 238
objects) -> ``extract_properties`` -> ``Vampire(n_clusters=5,
random_state=0).fit``: per-object properties, ``labels_``, ``n_components_``,
``explained_variance_ratio_``, ``mean_contour_``, ``mean_cluster_contours_``
and ``inertia_``.

Regenerate ONLY after an intentional change to the algorithm or its defaults
(e.g. contour extraction, resampling, registration/alignment, PCA/K-means
settings, dendrogram ordering) or to the bundled example images, and say so
in the commit message. Never regenerate just to make a failing test pass
without understanding why the numbers moved; a change caused by a dependency
upgrade (scikit-image, scikit-learn, NumPy/BLAS) should be investigated first.

This module is import-safe (pytest's ``--doctest-modules`` imports it), and
the test imports :func:`run_pipeline` from it so both share one pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import vampire as vp

GOLDEN_PATH = Path(__file__).parent / "integration_golden.npz"
N_CLUSTERS = 5
RANDOM_STATE = 0
PROPERTY_COLUMNS = ("area", "perimeter", "circularity")


def run_pipeline() -> tuple[pd.DataFrame, vp.Vampire]:
    """Extract properties of the example images and fit the reference model."""
    images, filenames = vp.datasets.load_example_images()
    props = vp.extract_properties(images, filenames)
    model = vp.Vampire(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit(props)
    return props, model


def compute_golden() -> dict[str, np.ndarray]:
    """Run the pipeline and return the arrays stored in the golden file."""
    props, model = run_pipeline()
    golden = {
        "n_objects": np.array(len(props)),
        "image_id": props["image_id"].to_numpy(np.int16),
        "label": props["label"].to_numpy(np.int32),
    }
    for col in PROPERTY_COLUMNS:
        golden[col] = props[col].to_numpy(np.float64)
    golden |= {
        "labels": model.labels_.astype(np.int8),
        "n_components": np.array(model.n_components_),
        "explained_variance_ratio": model.explained_variance_ratio_,
        "mean_contour": model.mean_contour_,
        "mean_cluster_contours": model.mean_cluster_contours_,
        "inertia": np.array(model.inertia_),
    }
    return golden


def main() -> None:
    """Write the golden file."""
    np.savez_compressed(GOLDEN_PATH, **compute_golden())
    print(f"Wrote {GOLDEN_PATH} ({GOLDEN_PATH.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
