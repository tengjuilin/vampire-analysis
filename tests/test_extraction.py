import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from skimage.draw import disk
from skimage.io import imread
from skimage.segmentation import find_boundaries

from vampire.extraction import extract_contour, extract_properties

COLUMNS = [
    "image_id", "label", "centroid_x", "centroid_y", "area", "bbox_area",
    "convex_area", "filled_area", "perimeter", "equivalent_diameter",
    "major_axis_length", "minor_axis_length", "orientation", "euler_number",
    "eccentricity", "solidity", "extent", "circularity", "aspect_ratio",
    "contour",
]  # fmt: skip


def rect_mask(shape=(30, 40), r0=5, r1=15, c0=8, c1=30):
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def disk_mask(shape=(60, 60), center=(30, 25), radius=10):
    mask = np.zeros(shape, dtype=bool)
    mask[disk(center, radius, shape=shape)] = True
    return mask


def test_extract_contour_shape_and_open():
    contour = extract_contour(rect_mask())
    assert contour.shape[0] == 2 and contour.dtype == np.float64
    assert not np.allclose(contour[:, 0], contour[:, -1])
    assert contour.shape[1] >= 4


@pytest.mark.parametrize("mask", [rect_mask(), disk_mask()])
def test_extract_contour_lies_on_boundary(mask):
    contour = extract_contour(mask)
    edge = np.argwhere(find_boundaries(mask, mode="inner"))
    tree = cKDTree(edge[:, ::-1].astype(float))  # (x, y)
    dist, _ = tree.query(contour.T)
    assert dist.max() <= 1.0
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    assert contour[0].min() == pytest.approx(cols.min() - 0.5, abs=0.5)
    assert contour[1].max() == pytest.approx(rows.max() + 0.5, abs=0.5)


def test_extract_contour_xy_order_and_touching_border():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:3, 0:6] = True  # touches the image border
    contour = extract_contour(mask)
    assert contour[0].max() > contour[1].max()  # wider than tall
    assert contour[0].min() < 0 and contour[1].min() < 0  # padding handled


def test_extract_contour_longest_of_multiple():
    mask = rect_mask((40, 40), 2, 6, 2, 6) | rect_mask((40, 40), 15, 35, 15, 35)
    contour = extract_contour(mask)
    assert contour[0].min() > 10


def test_extract_contour_too_small_or_empty():
    assert extract_contour(np.zeros((5, 5), dtype=bool)) is None
    assert extract_contour(np.ones((1, 1), dtype=bool)) is not None  # 4-point diamond


def test_extract_contour_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        extract_contour(np.zeros((2, 3, 4)))


def test_properties_rectangle_values():
    props = extract_properties(rect_mask())
    assert list(props.columns) == COLUMNS
    row = props.iloc[0]
    assert len(props) == 1
    assert row["area"] == 10 * 22
    assert row["centroid_x"] == pytest.approx(8 + 21 / 2)
    assert row["centroid_y"] == pytest.approx(5 + 9 / 2)
    assert row["aspect_ratio"] > 1
    assert 0 < row["circularity"] < 1.5
    assert row["contour"].shape[0] == 2
    # contour is in image frame
    assert row["contour"][0].min() == pytest.approx(7.5)
    assert row["contour"][1].min() == pytest.approx(4.5)


def test_properties_disk_circularity():
    props = extract_properties([disk_mask(radius=15, shape=(80, 80), center=(40, 40))])
    assert props.loc[0, "aspect_ratio"] == pytest.approx(1, abs=0.05)
    assert props.loc[0, "circularity"] == pytest.approx(1, abs=0.15)
    assert props.loc[0, "eccentricity"] < 0.2


def test_properties_multi_object_multi_image():
    a = rect_mask() | disk_mask(center=(22, 20), radius=5, shape=(30, 40))
    b = rect_mask()
    props = extract_properties([a, b], filenames=["a.tif", "b.tif"])
    assert list(props.columns) == ["filename", *COLUMNS]
    assert list(props.columns[:2]) == ["filename", "image_id"]
    assert isinstance(props.index, pd.RangeIndex)
    assert (props["image_id"] == 0).sum() == 2
    assert set(props["filename"]) == {"a.tif", "b.tif"}
    assert (props["image_id"] == 1).sum() == 1


def test_properties_binary_vs_label_input():
    mask = rect_mask((30, 60), 2, 10, 2, 10) | rect_mask((30, 60), 15, 25, 30, 50)
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[2:10, 2:10] = 1
    labels[15:25, 30:50] = 2
    from_bool = extract_properties(mask)
    from_u8 = extract_properties(mask.astype(np.uint8) * 255)
    from_01 = extract_properties(mask.astype(np.uint8))
    from_labels = extract_properties(labels)
    for other in (from_u8, from_01, from_labels):
        assert other["area"].tolist() == from_bool["area"].tolist()
        assert other["label"].tolist() == from_bool["label"].tolist()
    # non-contiguous label values are preserved
    labels[labels == 2] = 7
    assert extract_properties(labels)["label"].tolist() == [1, 7]


def test_properties_label_image_keeps_touching_objects_separate():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:10, 2:10] = 1
    labels[10:18, 2:10] = 2  # touches object 1
    props = extract_properties(labels)
    assert len(props) == 2


def test_properties_filenames_mismatch():
    with pytest.raises(ValueError, match="filenames"):
        extract_properties([rect_mask()], filenames=["a", "b"])


def test_properties_empty_image():
    props = extract_properties([np.zeros((5, 5), dtype=bool)], filenames=["x"])
    assert len(props) == 0
    assert list(props.columns) == ["filename", *COLUMNS]


def test_properties_rejects_3d():
    with pytest.raises(ValueError, match="2-D"):
        extract_properties([np.zeros((2, 3, 4))])


def test_aspect_ratio_degenerate_line():
    mask = np.zeros((10, 20), dtype=bool)
    mask[5, 2:15] = True  # one pixel thick: minor axis 0 -> contour has points
    props = extract_properties(mask)
    assert len(props) == 1
    assert props.loc[0, "aspect_ratio"] > 1 or np.isinf(props.loc[0, "aspect_ratio"])


def test_properties_real_images():
    img = imread("tests/data/img1.tif")
    props = extract_properties(img, filenames=["img1.tif"])
    assert len(props) > 0
    assert all(c.shape[0] == 2 and c.shape[1] >= 4 for c in props["contour"])
