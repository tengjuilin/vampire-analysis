from vampire.datasets import load_example_images
from vampire.extraction import extract_properties


def test_load_example_images():
    images, names = load_example_images()
    assert len(images) == len(names) == 4
    assert names == sorted(names)
    assert all(im.ndim == 2 for im in images)
    assert all(n.endswith(".tif") for n in names)


def test_example_images_have_objects():
    images, names = load_example_images()
    props = extract_properties(images, filenames=names)
    assert len(props) > 0
    assert set(props["image_id"]) == {0, 1, 2, 3}
    for col in ["filename", "area", "centroid_x", "circularity", "contour"]:
        assert col in props.columns
