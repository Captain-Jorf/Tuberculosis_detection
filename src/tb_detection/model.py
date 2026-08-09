"""Model definition.

No pretrained backbone here: the dataset is small (~4k films), grayscale,
and fairly different from ImageNet photos, so a compact ConvNet trained
from scratch does the job and actually fits on a laptop CPU. I tried a
fine-tuning route before; the gain wasn't worth the memory bill.

The augmentation layers live *inside* the model so we can never get
train/inference skew, and rescaling happens inside too (data pipeline
hands over raw 0-255 pixels).
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

# _conv_block appends "_conv" to the block name, so the Grad-CAM target
# is "<block>_conv", not the block name itself (bit me once already).
LAST_CONV_BLOCK = "conv_block3b"
LAYERNAMES_LAST_CONV = f"{LAST_CONV_BLOCK}_conv"


def _conv_block(x, filters: int, name: str):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(
        x
    )
    # GroupNorm, not BatchNorm. BN's moving stats poisoned early epochs and
    # quietly collapsed the model at inference (train metrics looked fine,
    # of course). Spent an evening on that; GN normalises per-sample so the
    # train/inference mismatch can't happen. Not going back.
    x = tf.keras.layers.GroupNormalization(groups=8, name=f"{name}_gn")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu")(x)


def build_model(img_size: Tuple[int, int] = (160, 160)) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(*img_size, 1), name="xray")

    # train-time augmentation only; off at inference by definition
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inp)
    x = tf.keras.layers.RandomFlip("horizontal", name="aug_flip")(x)
    x = tf.keras.layers.RandomRotation(0.05, name="aug_rot")(x)
    x = tf.keras.layers.RandomZoom(0.10, name="aug_zoom")(x)
    x = tf.keras.layers.RandomContrast(0.10, name="aug_contrast")(x)

    x = _conv_block(x, 32, "conv1a")
    x = _conv_block(x, 32, "conv1b")
    x = tf.keras.layers.MaxPooling2D(name="pool1")(x)
    x = tf.keras.layers.SpatialDropout2D(0.10, name="sdrop1")(x)

    x = _conv_block(x, 64, "conv2a")
    x = _conv_block(x, 64, "conv2b")
    x = tf.keras.layers.MaxPooling2D(name="pool2")(x)
    x = tf.keras.layers.SpatialDropout2D(0.10, name="sdrop2")(x)

    x = _conv_block(x, 128, "conv3a")
    x = _conv_block(x, 128, LAST_CONV_BLOCK)  # Grad-CAM hooks onto this one
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = tf.keras.layers.Dropout(0.40, name="drop")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop2")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="tb_prob")(x)

    model = tf.keras.Model(inp, out, name="tb_compact_cnn")
    return model


def compile_model(model: tf.keras.Model, lr: float = 3e-4) -> tf.keras.Model:
    model.compile(
        # clipnorm stays as cheap insurance against early-epoch blowups
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
