"""Read image sets from disk and annotate property tables."""

from collections.abc import Mapping, Sequence
from os import PathLike

import pandas as pd
from numpy.typing import NDArray

#: File extensions recognized by :func:`read_images` (case-insensitive).
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")


def read_images(
    directory: str | PathLike,
    pattern: str | Sequence[str] | None = None,
) -> tuple[list[NDArray], list[str]]:
    """Read all images in a directory, optionally filtered by regex.

    Parameters
    ----------
    directory : str or path-like
        Directory containing segmented images (non-recursive).
    pattern : str or sequence of str, optional
        Regular expression(s) matched with :func:`re.search` against each file
        name; a file is kept only if it matches **all** patterns.

    Returns
    -------
    images : list of ndarray
        Images in sorted file-name order.
    filenames : list of str
        Corresponding file names (not full paths).
    """
    raise NotImplementedError


def label_by_regex(
    properties: pd.DataFrame,
    annotations: pd.DataFrame | Mapping[str, Mapping[str, object]],
    column: str = "filename",
) -> pd.DataFrame:
    """Add metadata columns to a property table by regex-matching a column.

    Parameters
    ----------
    properties : DataFrame
        Table with a column (default ``"filename"``) to search.
    annotations : DataFrame or mapping
        Either a DataFrame with a ``"regex"`` column plus one column per
        metadata field, or a mapping ``{regex: {field: value, ...}}``.
    column : str, default="filename"
        Column of ``properties`` searched with each regex.

    Returns
    -------
    labeled : DataFrame
        Copy of ``properties`` with the metadata columns added. Rows matching no
        regex get NaN; later regexes override earlier ones on overlap.

    Examples
    --------
    >>> import pandas as pd
    >>> props = pd.DataFrame({"filename": ["ctrl_1.tif", "ogd_1.tif"]})
    >>> label_by_regex(props, {"ctrl": {"group": "control"}, "ogd": {"group": "OGD"}})
         filename    group
    0  ctrl_1.tif  control
    1   ogd_1.tif      OGD
    """
    raise NotImplementedError
