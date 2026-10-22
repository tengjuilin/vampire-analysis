"""Summary statistics of shape-mode assignments."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def shape_mode_distribution(
    labels: ArrayLike, n_clusters: int | None = None
) -> NDArray[np.float64]:
    """Return the fraction of objects in each shape mode.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Integer shape-mode labels in ``[0, n_clusters)``, e.g. from
        :meth:`vampire.Vampire.predict`.
    n_clusters : int, optional
        Total number of shape modes. Defaults to ``max(labels) + 1``. Modes with
        no objects get a fraction of 0.

    Returns
    -------
    distribution : ndarray of shape (n_clusters,)
        Fractions summing to 1.
    """
    raise NotImplementedError


def shannon_entropy(distribution: ArrayLike, base: float | None = None) -> float:
    """Return the Shannon entropy ``-sum(p * log(p))`` of a distribution.

    Zero-probability modes contribute 0.

    Parameters
    ----------
    distribution : array-like of shape (n_clusters,)
        Probabilities, e.g. from :func:`shape_mode_distribution`.
    base : float, optional
        Logarithm base. Defaults to natural log (nats).

    Returns
    -------
    entropy : float
    """
    raise NotImplementedError
