"""Dataset scanning, stratified splitting and the tf.data pipeline.

The raw folder layout is the classic one:

    data/raw/
        Normal/          *.png
        Tuberculosis/    *.png

Nothing fancy, and I want to keep it that way. Splits are written to a
CSV the first time they're made, so every later run evaluates on the
exact same test images (learned that the hard way on an earlier project
where my "test set" quietly changed between runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import Config

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
LABEL_MAP: Dict[str, int] = {"Normal": 0, "Tuberculosis": 1}


@dataclass
class Split:
    """Three lists of (path, label) pairs."""

    train: List[Tuple[str, int]]
    val: List[Tuple[str, int]]
    test: List[Tuple[str, int]]

    def class_weight(self) -> Dict[int, float]:
        """Inverse-frequency weights, normalised so the mean weight is 1."""
        counts = np.bincount([y for _, y in self.train], minlength=2).astype(float)
        n = counts.sum()
        w = n / (2.0 * np.maximum(counts, 1.0))  # don't divide by a zero if a class is empty
        return {0: float(w[0]), 1: float(w[1])}


def scan_dataset(data_dir: Path) -> pd.DataFrame:
    rows = []
    for cls_name, label in LABEL_MAP.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"missing folder {cls_dir} - run scripts/download_data.py first"
            )
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTENSIONS:
                rows.append((str(p), cls_name, label))
    if not rows:
        raise RuntimeError(f"no images found under {data_dir}")
    return pd.DataFrame(rows, columns=["path", "class_name", "label"])


def make_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Stratified 70/15/15 split (or whatever cfg says). Deterministic."""
    holdout = cfg.test_size + cfg.val_size
    train, rest = train_test_split(
        df, test_size=holdout, stratify=df["label"], random_state=cfg.seed
    )
    val_ratio = cfg.val_size / holdout  # slice `rest` into val/test
    val, test = train_test_split(
        rest, test_size=1.0 - val_ratio, stratify=rest["label"], random_state=cfg.seed
    )
    df = df.copy()
    df["split"] = "unused"
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def load_or_create_split(cfg: Config) -> Split:
    """Re-use the frozen CSV split if it exists; otherwise make it once."""
    if cfg.split_file.exists():
        df = pd.read_csv(cfg.split_file)
    else:
        df = make_splits(scan_dataset(cfg.data_dir), cfg)
        cfg.split_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.split_file, index=False)

    def _take(name: str) -> List[Tuple[str, int]]:
        sub = df[df["split"] == name]
        return list(zip(sub["path"].tolist(), sub["label"].astype(int).tolist()))

    return Split(train=_take("train"), val=_take("val"), test=_take("test"))


def _load_image(path: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int]):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)  # CXRs are grayscale
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)  # NOTE: kept 0-255 here, model rescales internally
    return img, label


def build_dataset(
    items: List[Tuple[str, int]],
    cfg: Config,
    batch: bool = True,
    shuffle: bool = False,
) -> tf.data.Dataset:
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(items), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _load_image(p, y, cfg.img_size), num_parallel_calls=tf.data.AUTOTUNE)
    if batch:
        ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
