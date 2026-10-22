import numpy as np
import pandas as pd
import pytest
from skimage.io import imsave

from vampire.io import label_by_regex, read_images


@pytest.fixture
def image_dir(tmp_path):
    rng = np.random.default_rng(0)
    for name in ["b_ctrl_2.png", "a_ctrl_1.png", "c_ogd_1.PNG", "d_ogd_2.png"]:
        imsave(tmp_path / name, rng.integers(0, 255, (6, 5), dtype=np.uint8),
               check_contrast=False)  # fmt: skip
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "sub.png").mkdir()
    return tmp_path


def test_read_images_sorted_and_filters_extensions(image_dir):
    images, names = read_images(image_dir)
    assert names == sorted(names)
    assert names == ["a_ctrl_1.png", "b_ctrl_2.png", "c_ogd_1.PNG", "d_ogd_2.png"]
    assert all(im.shape == (6, 5) for im in images)


def test_read_images_pattern_all_must_match(image_dir):
    _, names = read_images(image_dir, "ctrl")
    assert names == ["a_ctrl_1.png", "b_ctrl_2.png"]
    _, names = read_images(image_dir, ["ogd", r"_2\."])
    assert names == ["d_ogd_2.png"]
    _, names = read_images(image_dir, ["ctrl", "ogd"])
    assert names == []


def test_read_images_str_path(image_dir):
    assert len(read_images(str(image_dir))[0]) == 4


def test_read_images_skips_hidden_files(image_dir):
    # macOS AppleDouble sidecar: an image extension but not an image.
    (image_dir / "._a_ctrl_1.png").write_bytes(b"\x00\x05\x16\x07junk")
    (image_dir / ".hidden.png").write_bytes(b"junk")
    _, names = read_images(image_dir)
    assert names == ["a_ctrl_1.png", "b_ctrl_2.png", "c_ogd_1.PNG", "d_ogd_2.png"]


def test_read_images_names_unreadable_file(image_dir):
    (image_dir / "e_broken.png").write_bytes(b"not a png")
    with pytest.raises(OSError, match=r"e_broken\.png"):
        read_images(image_dir)


def test_read_images_errors(tmp_path, image_dir):
    with pytest.raises(FileNotFoundError):
        read_images(tmp_path / "missing")
    with pytest.raises(NotADirectoryError):
        read_images(image_dir / "notes.txt")


@pytest.fixture
def props():
    return pd.DataFrame({"filename": ["ctrl_1.tif", "ogd_1.tif", "other.tif"], "x": 1})


def test_label_by_regex_mapping(props):
    out = label_by_regex(props, {"ctrl": {"group": "control"}, "ogd": {"group": "OGD"}})
    assert out["group"].iloc[:2].tolist() == ["control", "OGD"]
    assert pd.isna(out["group"].iloc[2])


def test_label_by_regex_dataframe(props):
    ann = pd.DataFrame(
        {"regex": ["ctrl", "ogd"], "group": ["control", "OGD"], "dose": [0, 5]}
    )
    out = label_by_regex(props, ann)
    assert out["group"].iloc[:2].tolist() == ["control", "OGD"]
    assert out["dose"].iloc[:2].tolist() == [0, 5]
    assert pd.isna(out.loc[2, "dose"]) and pd.isna(out.loc[2, "group"])


def test_label_by_regex_later_overrides(props):
    out = label_by_regex(props, {"_1": {"g": "a"}, "ogd": {"g": "b"}})
    assert out["g"].iloc[:2].tolist() == ["a", "b"]


def test_label_by_regex_does_not_mutate(props):
    before = props.copy()
    label_by_regex(props, {"ctrl": {"group": "control"}})
    pd.testing.assert_frame_equal(props, before)


def test_label_by_regex_custom_column_and_errors(props):
    p = props.rename(columns={"filename": "name"})
    out = label_by_regex(p, {"ctrl": {"g": 1}}, column="name")
    assert out["g"].iloc[0] == 1
    with pytest.raises(ValueError, match="not found"):
        label_by_regex(props, {"a": {"g": 1}}, column="nope")
    with pytest.raises(ValueError, match="regex"):
        label_by_regex(props, pd.DataFrame({"g": [1]}))


def test_label_by_regex_skips_missing_annotation_values(props):
    ann = pd.DataFrame({"regex": ["_1", "ogd"], "group": ["one", None], "dose": [1, 5]})
    out = label_by_regex(props, ann)
    assert out["group"].iloc[:2].tolist() == ["one", "one"]  # None does not clobber
    assert out["dose"].iloc[:2].tolist() == [1, 5]
    out = label_by_regex(props, {"_1": {"g": "a"}, "ogd": {"g": float("nan")}})
    assert out["g"].iloc[:2].tolist() == ["a", "a"]


def test_label_by_regex_keeps_existing_columns(props):
    existing = props.assign(group=["old0", "old1", "old2"], dose=[1.5, 2.5, 3.5])
    out = label_by_regex(existing, {"ctrl": {"group": "control", "dose": 0.0}})
    assert out["group"].tolist() == ["control", "old1", "old2"]
    assert out["dose"].tolist() == [0.0, 2.5, 3.5]
    assert out["dose"].dtype == np.float64
    assert out["x"].tolist() == [1, 1, 1]


def test_label_by_regex_capture_groups_do_not_warn(props, recwarn):
    out = label_by_regex(props, {r"(ctrl|ogd)_1": {"g": "x"}})
    assert out["g"].iloc[:2].tolist() == ["x", "x"]
    assert pd.isna(out.loc[2, "g"])
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_label_by_regex_missing_and_categorical_text():
    props = pd.DataFrame({"filename": ["nan_1.tif", None]})
    out = label_by_regex(props, {"nan|None": {"g": 1}})
    assert out["g"].iloc[0] == 1 and pd.isna(out["g"].iloc[1])
    cat = pd.DataFrame({"filename": pd.Categorical(["ctrl_1", "ogd_1"])})
    assert label_by_regex(cat, {"ctrl": {"g": 1}})["g"].iloc[0] == 1


def test_label_by_regex_preserves_int_dtype(props):
    out = label_by_regex(props.iloc[:2], {"ctrl": {"dose": 0}, "ogd": {"dose": 5}})
    assert out["dose"].dtype == np.int64
