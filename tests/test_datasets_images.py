import numpy as np

from vampire.datasets import load_example_images
from vampire.extraction import extract_properties


def test_load_example_images():
    images, names = load_example_images()
    assert names == ["img1", "img2"]
    for im in images:
        assert im.shape == (512, 512)
        assert im.dtype == np.uint16
        # Real segmentations: mostly background, objects are contiguous blobs.
        assert (im == 0).mean() > 0.5
        assert 50 < len(np.unique(im)) < 500


def test_example_images_have_realistic_objects():
    images, names = load_example_images()
    props = extract_properties(images, filenames=names)
    assert set(props["image_id"]) == {0, 1}
    assert props["area"].median() > 20
    assert (props["perimeter"] > 0).all()
    for col in ["filename", "area", "centroid_x", "circularity", "contour"]:
        assert col in props.columns
