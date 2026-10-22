import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

import vampire as vp

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture
def shapes():
    """Synthetic circles and ellipses with ground-truth kind labels."""
    return vp.datasets.make_shapes(n_samples=80, random_state=0, return_labels=True)
