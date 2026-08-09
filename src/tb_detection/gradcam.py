"""Grad-CAM on the last conv block.

Doctors (rightly) don't trust a bare probability, so every evaluation run
dumps a few heatmaps showing *where* the model is looking. If those start
lighting up outside the lung field, something's wrong with the data,
not with the lungs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

from .model import LAYERNAMES_LAST_CONV


def gradcam_heatmap(model: tf.keras.Model, img: np.ndarray) -> np.ndarray:
    """Return an HxW float heatmap in [0, 1] for a single pre-scaled image.

    `img` is (H, W, 1), float32, range 0-255 (model rescales internally).
    """
    last_conv = model.get_layer(LAYERNAMES_LAST_CONV)
    grad_model = tf.keras.Model(model.input, [last_conv.output, model.output])

    x = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("no gradient reached the last conv layer")

    weights = tf.reduce_mean(grads, axis=(1, 2))  # GAP over spatial dims
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam -= tf.reduce_min(cam)
    cam /= tf.reduce_max(cam) + 1e-8
    return cam.numpy().astype(np.float32)


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Cam over the grayscale CXR, RGB uint8 out."""
    base = img[..., 0] if img.ndim == 3 else img
    base = np.clip(base, 0, 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    heat = cv2.resize(cam, (base_rgb.shape[1], base_rgb.shape[0]))
    heat_rgb = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1 - alpha, heat_rgb, alpha, 0)


def save_gradcam(
    model: tf.keras.Model,
    img: np.ndarray,
    path: Path,
    caption: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam = gradcam_heatmap(model, img)
    over = overlay_heatmap(img, cam)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
    ax[0].imshow(img[..., 0] if img.ndim == 3 else img, cmap="gray")
    ax[0].set_title("film")
    ax[1].imshow(over)
    ax[1].set_title(caption or "grad-cam")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
