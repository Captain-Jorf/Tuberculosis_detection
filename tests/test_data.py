"""Data pipeline tests on a tiny synthetic dataset."""

import numpy as np
import pytest
from PIL import Image

from tb_detection.config import Config
from tb_detection.data import build_dataset, load_or_create_split, make_splits, scan_dataset


@pytest.fixture()
def tiny_dataset(tmp_path):
    """8 normal + 8 tb, distinct means so a real split is possible."""
    rng = np.random.default_rng(0)
    for cls, mean in [("Normal", 40), ("Tuberculosis", 200)]:
        d = tmp_path / "raw" / cls
        d.mkdir(parents=True)
        for i in range(8):
            arr = rng.normal(mean, 10, (48, 48)).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(d / f"{cls}_{i}.png")
    return tmp_path / "raw"


def test_scan_finds_everything(tmp_path, tiny_dataset):
    df = scan_dataset(tiny_dataset)
    assert len(df) == 16
    assert set(df["label"]) == {0, 1}


def test_splits_are_stratified_and_disjoint(tmp_path, tiny_dataset):
    cfg = Config(data_dir=tiny_dataset, split_file=tmp_path / "splits.csv", test_size=0.25, val_size=0.25)
    df = make_splits(scan_dataset(tiny_dataset), cfg)
    tr = df[df.split == "train"]
    assert tr["label"].mean() == pytest.approx(0.5)


def test_split_file_is_frozen(tmp_path, tiny_dataset):
    cfg = Config(data_dir=tiny_dataset, split_file=tmp_path / "splits.csv", test_size=0.25, val_size=0.25)
    s1 = load_or_create_split(cfg)
    s2 = load_or_create_split(cfg)
    assert s1.train == s2.train  # second call must re-read the CSV, not reshuffle


def test_dataset_batch_shapes(tmp_path, tiny_dataset):
    import tensorflow as tf  # noqa: F401 (import only when tf exists)

    cfg = Config(data_dir=tiny_dataset, split_file=tmp_path / "s.csv", batch_size=4)
    items = [(str(p), 0) for p in tiny_dataset.glob("Normal/*.png")]
    ds = build_dataset(items, cfg)
    xb, yb = next(iter(ds))
    assert xb.shape == (4, 160, 160, 1)
    assert yb.shape == (4,)
