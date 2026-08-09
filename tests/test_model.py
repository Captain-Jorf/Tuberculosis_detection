import numpy as np
import tensorflow as tf

from tb_detection.model import LAYERNAMES_LAST_CONV, build_model, compile_model


def test_output_shape_and_sigmoid_range():
    model = build_model((64, 64))
    x = tf.random.uniform((2, 64, 64, 1), maxval=255)
    y = model(x, training=False)
    assert y.shape == (2, 1)
    assert 0.0 <= float(y.numpy().min()) and float(y.numpy().max()) <= 1.0


def test_last_conv_layer_exists():
    model = build_model((64, 64))
    model.get_layer(LAYERNAMES_LAST_CONV)  # raises if the Grad-CAM target went missing


def test_one_training_step_is_finite():
    model = compile_model(build_model((64, 64)))
    x = tf.random.uniform((4, 64, 64, 1), maxval=255)
    y = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
    loss = model.train_on_batch(x, y)
    assert np.isfinite(loss[0])


def test_model_stays_small():
    # the whole point of this architecture; don't let it creep up
    assert build_model().count_params() < 900_000
