"""Score a single chest X-ray.

    tb-predict path/to/cxr.png
    tb-predict path/to/cxr.png --cam out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import Config
from .gradcam import save_gradcam

tf.get_logger().setLevel("ERROR")

THRESHOLD = 0.5


def load_image(path: Path, img_size) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", target_size=img_size)
    return tf.keras.preprocessing.image.img_to_array(img)


def predict_one(model: tf.keras.Model, arr: np.ndarray) -> float:
    return float(model.predict(arr[None], verbose=0)[0, 0])


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", type=Path, default=cfg.model_path)
    ap.add_argument("--cam", type=Path, default=None, help="also save a Grad-CAM overlay here")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"no model at {args.model} - train first (tb-train)")

    model = tf.keras.models.load_model(args.model)
    arr = load_image(args.image, cfg.img_size)
    prob = predict_one(model, arr)

    label = "Tuberculosis" if prob >= THRESHOLD else "Normal"
    print(f"P(TB) = {prob:.3f} -> {label}")
    print("(screening aid only, not a diagnosis)")

    if args.cam is not None:
        save_gradcam(model, arr, args.cam, caption=f"P(TB)={prob:.2f}")
        print(f"grad-cam saved to {args.cam}")


if __name__ == "__main__":
    main()
