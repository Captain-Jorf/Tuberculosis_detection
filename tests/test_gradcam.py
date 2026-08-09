import numpy as np
import tensorflow as tf

from tb_detection.gradcam import gradcam_heatmap
from tb_detection.model import build_model


def test_gradcam_returns_normalized_heatmap():
    model = build_model((64, 64))
    img = tf.random.uniform((64, 64, 1), maxval=255).numpy().astype(np.float32)
    cam = gradcam_heatmap(model, img)
    assert cam.shape == (16, 16)  # after 2 poolings on 64px input
    assert 0.0 <= cam.min() and cam.max() <= 1.0
