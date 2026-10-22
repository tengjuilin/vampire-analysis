"""Extract object contours and morphological properties from images."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.measure import find_contours, label, regionprops, regionprops_table

#: Mapping from output column name to :func:`skimage.measure.regionprops` name.
_PROPERTY_NAMES = {
    "area": "area",
    "bbox_area": "area_bbox",
    "convex_area": "area_convex",
    "filled_area": "area_filled",
    "perimeter": "perimeter",
    "equivalent_diameter": "equivalent_diameter_area",
    "major_axis_length": "axis_major_length",
    "minor_axis_length": "axis_minor_length",
    "orientation": "orientation",
    "euler_number": "euler_number",
    "eccentricity": "eccentricity",
    "solidity": "solidity",
    "extent": "extent",
}

_COLUMNS = (
    "image_id",
    "label",
    "centroid_x",
    "centroid_y",
    *_PROPERTY_NAMES,
    "circularity",
    "aspect_ratio",
    "contour",
)


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

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((7, 7), dtype=bool)
    >>> mask[2:5, 2:5] = True
    >>> contour = extract_contour(mask)
    >>> contour.shape
    (2, 12)
    >>> float(contour[0].min()), float(contour[0].max())
    (1.5, 4.5)
    """
    mask = np.asarray(object_mask)
    if mask.ndim != 2:
        raise ValueError(f"object_mask must be 2-D, got {mask.ndim}-D.")
    padded = np.pad(mask.astype(np.float64) != 0, 1).astype(np.float64)
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    longest = max(contours, key=len)
    if len(longest) > 1 and np.array_equal(longest[0], longest[-1]):
        longest = longest[:-1]
    if len(longest) < 4:
        return None
    longest = longest - 1.0  # remove padding offset
    return np.ascontiguousarray(longest[:, ::-1].T)  # (row, col) -> (x, y)


def _to_label_image(image: NDArray) -> NDArray:
    """Return an integer label image for a binary mask or label image."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Each image must be 2-D, got shape {arr.shape}.")
    if arr.dtype == bool:
        return label(arr)
    if np.issubdtype(arr.dtype, np.integer):
        values = set(np.unique(arr).tolist())
        if values <= {0, 1} or values <= {0, 255}:
            return label(arr != 0)
        return arr
    # Floating point: anything non-zero is foreground.
    return label(arr != 0)


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
        is a fresh ``RangeIndex``. ``filename`` is placed first when given.

    Raises
    ------
    ValueError
        If ``filenames`` and ``images`` differ in length, or an image is not 2-D.

    Notes
    -----
    Integer images with more than two distinct values (or values other than
    ``{0, 1}`` / ``{0, 255}``) are treated as label images and used as is;
    boolean, 0/1 and 0/255 images are binary and labeled with 8-connectivity.
    ``centroid_x`` / ``centroid_y`` are column / row coordinates. ``circularity``
    is ``4 * pi * area / perimeter**2``; ``aspect_ratio`` is
    ``major_axis_length / minor_axis_length`` and is ``inf`` for a
    zero-width object (``NaN`` if both are zero). ``image_id`` is the 0-based
    position of the image in ``images``.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((20, 20), dtype=bool)
    >>> mask[2:8, 2:8] = True
    >>> mask[10:18, 10:16] = True
    >>> props = extract_properties([mask], filenames=["a.tif"])
    >>> props[["filename", "image_id", "label", "area"]]
      filename  image_id  label  area
    0    a.tif         0      1  36.0
    1    a.tif         0      2  48.0
    >>> props.loc[0, "contour"].shape[0]
    2
    """
    if isinstance(images, np.ndarray) and images.ndim == 2:
        images = [images]
    images = list(images)
    if filenames is not None:
        filenames = list(filenames)
        if len(filenames) != len(images):
            raise ValueError(
                f"filenames has {len(filenames)} entries but there are "
                f"{len(images)} images."
            )

    frames = []
    for image_id, image in enumerate(images):
        labels = _to_label_image(image)
        if labels.max(initial=0) == 0:
            continue
        table = regionprops_table(
            labels,
            properties=["label", "centroid", *_PROPERTY_NAMES.values()],
        )
        df = pd.DataFrame(table).rename(
            columns={
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
                **{v: k for k, v in _PROPERTY_NAMES.items()},
            }
        )
        regions = regionprops(labels)
        contours: list[NDArray | None] = []
        for region in regions:
            contour = extract_contour(region.image)
            if contour is not None:
                min_row, min_col = region.bbox[0], region.bbox[1]
                contour = contour + np.array([[min_col], [min_row]], dtype=float)
            contours.append(contour)
        df["contour"] = pd.Series(contours, dtype=object)
        df["image_id"] = image_id
        if filenames is not None:
            df["filename"] = filenames[image_id]
        frames.append(df[df["contour"].notna()])

    lead = ["filename"] if filenames is not None else []
    columns = [*lead, *_COLUMNS]
    if not frames:
        return pd.DataFrame({c: [] for c in columns})
    out = pd.concat(frames, ignore_index=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["circularity"] = 4 * np.pi * out["area"] / out["perimeter"] ** 2
        out["aspect_ratio"] = out["major_axis_length"] / out["minor_axis_length"]
    out["contour"] = out["contour"].astype(object)
    return out[columns].reset_index(drop=True)
