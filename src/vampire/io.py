"""Read image sets from disk and annotate property tables."""

import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
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
        Column of ``properties`` searched with each regex (with
        :func:`re.search`; missing values never match).

    Returns
    -------
    labeled : DataFrame
        Copy of ``properties`` with the metadata columns set. Rules are applied
        in order, so later regexes override earlier ones on overlap. Missing
        values (NaN/None) in an annotation are skipped and never override an
        earlier match. Fields that are not yet columns of ``properties`` are
        created and are NaN for rows matching no regex; existing columns keep
        their values in rows no rule sets.

    Raises
    ------
    ValueError
        If ``column`` is missing from ``properties`` or an annotations
        DataFrame has no ``"regex"`` column.

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
        rules = list(
            zip(
                annotations["regex"],
                annotations.drop(columns="regex").to_dict("records"),
                strict=True,
            )
        )
    else:
        rules = [(rx, dict(fields)) for rx, fields in annotations.items()]
    if column not in properties.columns:
        raise ValueError(f"Column {column!r} not found in properties.")

    labeled = properties.copy()
    fields = list(dict.fromkeys(f for _, d in rules for f in d))
    for field in fields:
        if field in labeled.columns:
            labeled[field] = labeled[field].astype(object)
        else:
            labeled[field] = pd.Series(float("nan"), index=labeled.index, dtype=object)
    text = labeled[column].tolist()
    for regex, values in rules:
        compiled = re.compile(regex)
        mask = np.array(
            [not _is_missing(v) and compiled.search(str(v)) is not None for v in text],
            dtype=bool,
        )
        if not mask.any():
            continue
        for field, value in values.items():
            if not _is_missing(value):
                labeled.loc[mask, field] = value
    for field in fields:
        labeled[field] = labeled[field].infer_objects()
    return labeled


def _is_missing(value: object) -> bool:
    """Return True for scalar missing values (None, NaN, NA, NaT)."""
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))
