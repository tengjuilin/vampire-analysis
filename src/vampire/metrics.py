"""Summary statistics of shape-mode assignments."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_labels


def shape_mode_distribution(
    labels: ArrayLike, n_clusters: int | None = None
) -> NDArray[np.float64]:
    """Return the fraction of objects in each shape mode.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Integer shape-mode labels in ``1..n_clusters`` (shape modes are
        1-based), e.g. from :meth:`vampire.Vampire.predict`.
    n_clusters : int, optional
        Total number of shape modes. Defaults to ``max(labels)``. Modes with
        no objects get a fraction of 0. Pass ``model.n_clusters`` so that
        distributions of different groups have the same length.

    Returns
    -------
    distribution : ndarray of shape (n_clusters,)
        ``float64`` fractions summing to 1. Entry ``i`` is the fraction of
        objects in shape mode ``i + 1`` (``distribution[m - 1]`` is mode
        ``m``).

    Raises
    ------
    ValueError
        If ``labels`` is empty, not 1-D, not integer-valued (whole-number
        floats are accepted), below 1 (0 is rejected because shape modes are
        1-based), or above ``n_clusters``; or if ``n_clusters`` is not an
        integer >= 1.

    Examples
    --------
    >>> from vampire.metrics import shape_mode_distribution
    >>> shape_mode_distribution([1, 1, 2, 4])
    array([0.5 , 0.25, 0.  , 0.25])
    >>> shape_mode_distribution([1, 2], n_clusters=4)
    array([0.5, 0.5, 0. , 0. ])
    """
    values, n_clusters = check_labels(labels, n_clusters)
    counts = np.bincount(values - 1, minlength=n_clusters)
    return counts / counts.sum()


def shannon_entropy(distribution: ArrayLike, base: float | None = None) -> float:
    r"""Return the Shannon entropy ``-sum(p * log(p))`` of a distribution.

    Zero-probability modes contribute 0 (the limit of :math:`p \log p` as
    :math:`p \to 0`).

    Parameters
    ----------
    distribution : array-like of shape (n_clusters,)
        Probabilities summing to 1, e.g. from :func:`shape_mode_distribution`.
    base : float, optional
        Logarithm base, positive and not 1. Defaults to natural log (nats);
        use 2 for bits.

    Returns
    -------
    entropy : float
        Between 0 (all objects in one mode) and ``log(n_clusters)`` (uniform).

    Raises
    ------
    ValueError
        If ``distribution`` is empty, not 1-D, has negative or non-finite
        entries, or does not sum to 1; or if ``base`` is not a positive
        number other than 1.

    Examples
    --------
    >>> from vampire.metrics import shannon_entropy
    >>> shannon_entropy([0.5, 0.5, 0.0], base=2)
    1.0
    >>> shannon_entropy([1.0, 0.0])
    0.0
    """
    if base is not None and not (
        isinstance(base, int | float | np.integer | np.floating)
        and not isinstance(base, bool)
        and np.isfinite(base)
        and base > 0
        and base != 1
    ):
        raise ValueError(f"base must be a positive number other than 1; got {base!r}.")
    p = np.asarray(distribution, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(
            f"distribution must be a non-empty 1-D array; got shape {p.shape}."
        )
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("distribution must contain finite, non-negative values.")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError(
            f"distribution must sum to 1; got {p.sum()}. Normalize counts with "
            "vampire.shape_mode_distribution."
        )
    nonzero = p[p > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    if base is not None:
        entropy /= np.log(base)
    return float(entropy) + 0.0  # + 0.0 turns -0.0 into 0.0
