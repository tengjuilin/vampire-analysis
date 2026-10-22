"""Read image sets from disk and annotate property tables."""

import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import pandas as pd
from numpy.typing import NDArray
from skimage.io import imread

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

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist.
    NotADirectoryError
        If ``directory`` is not a directory.

    Notes
    -----
    Only files whose extension is in :data:`IMAGE_EXTENSIONS` (compared
    case-insensitively) are considered.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from skimage.io import imsave
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     mask = np.zeros((4, 4), dtype=np.uint8)
    ...     for name in ["ctrl_1.png", "ogd_1.png"]:
    ...         imsave(Path(tmp) / name, mask, check_contrast=False)
    ...     images, names = read_images(tmp, pattern="ctrl")
    >>> names
    ['ctrl_1.png']
    >>> images[0].shape
    (4, 4)
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    patterns = [pattern] if isinstance(pattern, str) else list(pattern or [])
    compiled = [re.compile(p) for p in patterns]
    files = sorted(
        f.name
        for f in path.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
        and all(rx.search(f.name) for rx in compiled)
    )
    return [imread(path / name) for name in files], files


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
    >>> annotations = pd.DataFrame(
    ...     {"regex": ["ctrl", "ogd"], "group": ["control", "OGD"]}
    ... )
    >>> label_by_regex(props, annotations)["group"].tolist()
    ['control', 'OGD']
    """
    if isinstance(annotations, pd.DataFrame):
        if "regex" not in annotations.columns:
            raise ValueError("annotations DataFrame must have a 'regex' column.")
        rules = [
            (row["regex"], row.drop("regex").to_dict())
            for _, row in annotations.iterrows()
        ]
    else:
        rules = [(rx, dict(fields)) for rx, fields in annotations.items()]
    if column not in properties.columns:
        raise ValueError(f"Column {column!r} not found in properties.")

    labeled = properties.copy()
    fields = list(dict.fromkeys(f for _, d in rules for f in d))
    for field in fields:
        labeled[field] = pd.Series(float("nan"), index=labeled.index, dtype=object)
    text = labeled[column].astype(str)
    for regex, values in rules:
        mask = text.str.contains(regex, regex=True)
        for field, value in values.items():
            labeled.loc[mask, field] = value
    for field in fields:
        labeled[field] = labeled[field].infer_objects()
    return labeled
