"""Extract object contours and morphological properties from images."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.measure import find_contours, label, regionprops
from skimage.segmentation import clear_border

from ._validation import MIN_CONTOUR_POINTS

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

    Uses :func:`skimage.measure.find_contours` (with
    ``fully_connected="high"``, i.e. 8-connected foreground, matching the
    8-connectivity of :func:`skimage.measure.label`) on a zero-padded copy of
    the mask and keeps the longest contour. Coordinates are in the mask's pixel
    frame (row 0 = x = column index, row 1 = y = row index). The closing point
    duplicated by ``find_contours`` is removed.

    Parameters
    ----------
    object_mask : ndarray of shape (height, width)
        Binary (or boolean) image containing one object; nonzero pixels are
        foreground. Pixels touching only diagonally belong to the same object.

    Returns
    -------
    contour : ndarray of shape (2, n) or None
        ``None`` if the mask is empty (any nonempty mask yields at least 4
        boundary points).

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
    padded = np.pad(mask != 0, 1).astype(np.float64)
    contours = find_contours(padded, 0.5, fully_connected="high")
    if not contours:
        return None
    longest = max(contours, key=len)
    if len(longest) > 1 and np.array_equal(longest[0], longest[-1]):
        longest = longest[:-1]
    if len(longest) < MIN_CONTOUR_POINTS:
        return None
    longest = longest - 1.0  # remove padding offset
    return np.ascontiguousarray(longest[:, ::-1].T)  # (row, col) -> (x, y)


def _is_integer_valued(arr: NDArray) -> bool:
    if np.issubdtype(arr.dtype, np.integer):
        return True
    return bool(
        np.issubdtype(arr.dtype, np.floating)
        and np.all(np.isfinite(arr))
        and np.all(arr == np.round(arr))
    )


def _to_label_image(image: NDArray, binary: bool | None = None) -> NDArray:
    """Return an integer label image for a binary mask or label image.

    See the ``binary`` parameter of :func:`extract_properties`.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Each image must be 2-D, got shape {arr.shape}.")
    if not (
        arr.dtype == bool
        or np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError(f"Images must be boolean or numeric; got dtype {arr.dtype}.")
    if binary is None:
        if arr.dtype == bool:
            binary = True
        else:
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Image contains NaN or infinite values; pass binary=True "
                    "(foreground = pixels > 0) or clean the image."
                )
            nonzero = np.unique(arr[arr != 0])
            if nonzero.size <= 1:
                binary = True
            elif _is_integer_valued(nonzero):
                binary = False
            else:
                raise ValueError(
                    "Image has non-integer values, so it is neither a label image "
                    "nor a binary mask; pass binary=True to treat pixels > 0 as "
                    "foreground, or threshold the image first."
                )
    if binary:
        return label(arr > 0)  # 8-connectivity (connectivity=ndim)
    if arr.dtype == bool or not _is_integer_valued(arr):
        raise ValueError(
            "binary=False requires an integer-valued label image (0 = background); "
            f"got dtype {arr.dtype} with non-integer values. Pass binary=True for "
            "a mask."
        )
    if arr.size and arr.min() < 0:
        raise ValueError("Label images must not contain negative values.")
    return arr if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.int64)


def extract_properties(
    images: Sequence[NDArray] | NDArray,
    filenames: Sequence[str] | None = None,
    *,
    binary: bool | None = None,
    min_area: float = 0,
    exclude_border: bool = False,
) -> pd.DataFrame:
    """Measure every object in a set of segmented images.

    Each image is turned into a label image (see ``binary``) and every object
    is measured with :func:`skimage.measure.regionprops`.

    Parameters
    ----------
    images : sequence of ndarray of shape (height, width), or a single 2-D ndarray
        Segmented images (binary masks or label images).
    filenames : sequence of str, optional
        One name per image; stored in the ``"filename"`` column.
    binary : bool or None, default=None
        How pixel values are interpreted:

        - ``True``: the image is a mask; pixels ``> 0`` are foreground and are
          labeled with :func:`skimage.measure.label` (8-connectivity), so each
          connected blob becomes one object with labels ``1, 2, ...``.
        - ``False``: the image is a label image (0 = background, ``l`` =
          object ``l``) and is used as is. It must be integer-valued (integer
          dtype, or floats that are whole numbers) and non-negative. Touching
          objects stay separate and label values are kept.
        - ``None`` (auto): boolean images, and images with exactly one
          nonzero value (e.g. 0/1, 0/255, 0/65535, 0.0/1.0), are masks;
          integer-valued images with more than one nonzero value are label
          images; images with non-integer values raise ``ValueError`` (pass
          ``binary=True`` or threshold them). A label image containing a
          single object is therefore relabeled to 1; pass ``binary=False`` to
          keep its label value.
    min_area : float, default=0
        Drop objects with fewer than ``min_area`` pixels (``area`` column).
        Objects of a few pixels have pixel-staircase outlines (a single pixel
        becomes a diamond) that carry no shape information; a threshold well
        above the noise level of the segmentation is recommended.
    exclude_border : bool, default=False
        Drop objects that touch the image edge
        (:func:`skimage.segmentation.clear_border`); their outlines are cut
        off by the edge, not by the object.

    Returns
    -------
    properties : DataFrame of shape (n_objects, n_columns)
        One row per object (objects without a boundary are dropped), with
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
        If ``filenames`` and ``images`` differ in length, an image is not 2-D,
        an image cannot be interpreted as described under ``binary``, or
        ``min_area`` is negative.

    Notes
    -----
    ``centroid_x`` / ``centroid_y`` are column / row coordinates (y grows
    downward, as in the image). The contour is the outer boundary of the
    object (8-connected, holes ignored), so its enclosed area is close to
    ``filled_area``. If a label in a label image consists of several
    disconnected pieces, its contour is that of the piece with the longest
    boundary, while the other properties cover all pieces. ``circularity``
    is ``4 * pi * area / perimeter**2`` (``NaN`` when the perimeter is 0, as
    for objects one pixel wide; above 1 for some small objects, whose
    pixel perimeter underestimates the true one); ``aspect_ratio`` is
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

    A label image whose objects touch keeps them apart:

    >>> labels = np.zeros((10, 10), dtype=np.int32)
    >>> labels[2:5, 2:5], labels[5:8, 2:5] = 3, 4
    >>> extract_properties(labels, binary=False)["label"].tolist()
    [3, 4]

    Drop small objects and objects cut off by the image edge:

    >>> mask = np.zeros((20, 20), dtype=bool)
    >>> mask[0:5, 0:5] = True  # touches the edge
    >>> mask[8:14, 8:14] = True
    >>> mask[17, 10] = True  # one pixel
    >>> props = extract_properties(mask, min_area=4, exclude_border=True)
    >>> props["area"].tolist()
    [36.0]
    """
    if not (isinstance(min_area, int | float | np.number) and min_area >= 0):
        raise ValueError(f"min_area must be a number >= 0; got {min_area!r}.")
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

    rows = []
    for image_id, image in enumerate(images):
        label_image = _to_label_image(image, binary=binary)
        if exclude_border:
            label_image = clear_border(label_image)
        for region in regionprops(label_image):
            if region.area < min_area:
                continue
            contour = extract_contour(region.image)
            if contour is None:
                continue
            min_row, min_col = region.bbox[0], region.bbox[1]
            row = {
                "image_id": image_id,
                "label": region.label,
                "centroid_x": region.centroid[1],
                "centroid_y": region.centroid[0],
            }
            row |= {col: region[name] for col, name in _PROPERTY_NAMES.items()}
            row["contour"] = contour + np.array([[min_col], [min_row]], dtype=float)
            if filenames is not None:
                row["filename"] = filenames[image_id]
            rows.append(row)

    lead = ["filename"] if filenames is not None else []
    columns = [*lead, *_COLUMNS]
    if not rows:
        return pd.DataFrame({c: [] for c in columns})
    out = pd.DataFrame(rows)
    for col in ("area", "bbox_area", "convex_area", "filled_area"):
        out[col] = out[col].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        perimeter = out["perimeter"].where(out["perimeter"] > 0)  # 0 -> NaN
        out["circularity"] = 4 * np.pi * out["area"] / perimeter**2
        out["aspect_ratio"] = out["major_axis_length"] / out["minor_axis_length"]
    out["contour"] = out["contour"].astype(object)
    return out[columns]
