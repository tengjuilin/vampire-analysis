"""Extract object contours and morphological properties from images."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def extract_contour(object_mask: NDArray) -> NDArray[np.float64] | None:
    """Return the outer boundary of a single object as a ``(2, n)`` contour.

    Uses :func:`skimage.measure.find_contours` on a zero-padded copy of the
    mask and keeps the longest contour. Coordinates are in the mask's pixel
    frame (row 0 = x = column index, row 1 = y = row index). The closing point
    duplicated by ``find_contours`` is removed.

    Parameters
    ----------
    object_mask : ndarray of shape (height, width)
        Binary (or boolean) image containing one object.

    Returns
    -------
    contour : ndarray of shape (2, n) or None
        ``None`` if the object has fewer than 4 boundary points.
    """
    raise NotImplementedError


def extract_properties(
    images: Sequence[NDArray] | NDArray,
    filenames: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Measure every object in a set of segmented images.

    Each image is labeled with :func:`skimage.measure.label` if it is binary
    (already-labeled integer images are used as is), then measured with
    :func:`skimage.measure.regionprops_table`.

    Parameters
    ----------
    images : sequence of ndarray of shape (height, width), or a single 2-D ndarray
        Segmented images (binary masks or label images).
    filenames : sequence of str, optional
        One name per image; stored in the ``"filename"`` column.

    Returns
    -------
    properties : DataFrame of shape (n_objects, n_columns)
        One row per object (objects with < 4 boundary points are dropped), with
        columns ``filename`` (if given), ``image_id``, ``label``,
        ``centroid_x``, ``centroid_y``, ``area``, ``bbox_area``,
        ``convex_area``, ``filled_area``, ``perimeter``,
        ``equivalent_diameter``, ``major_axis_length``, ``minor_axis_length``,
        ``orientation``, ``euler_number``, ``eccentricity``, ``solidity``,
        ``extent``, ``circularity``, ``aspect_ratio``, and ``contour`` (object
        dtype; each a ``(2, n)`` array in image pixel coordinates). The index
        is a fresh ``RangeIndex``.
    """
    raise NotImplementedError
