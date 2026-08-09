"""Training entry point.

    tb-train --epochs 30

Trains the compact CNN, evaluates on the frozen test split, and writes
metrics + figures + Grad-CAM samples under outputs/results/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import Config
from .data import build_dataset, load_or_create_split
from .evaluate import (
    compute_metrics,
    dump_metrics,
    plot_confusion,
    plot_history,
    plot_roc_pr,
)
from .gradcam import save_gradcam
from .model import build_model, compile_model

tf.get_logger().setLevel("ERROR")


def run(cfg: Config) -> dict:
    t0 = time.time()
    tf.keras.utils.set_random_seed(cfg.seed)

    split = load_or_create_split(cfg)
    print(f"train/val/test: {len(split.train)}/{len(split.val)}/{len(split.test)}")

    train_ds = build_dataset(split.train, cfg, shuffle=True)
    val_ds = build_dataset(split.val, cfg)

    model = compile_model(build_model(cfg.img_size), lr=cfg.learning_rate)
    model.summary(print_fn=lambda s: print(" ", s))

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            cfg.model_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=cfg.early_stop_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(cfg.log_dir / "history.csv"),
    ]

    cw = split.class_weight()
    print(f"class weights: {cw}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=cw,
        callbacks=cbs,
        verbose=1,
    )

    # ---- held-out test evaluation ----
    model = tf.keras.models.load_model(cfg.model_path)  # best checkpoint
    probs, trues = [], []
    for xb, yb in build_dataset(split.test, cfg):
        probs.append(model.predict(xb, verbose=0).ravel())
        trues.append(yb.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)

    metrics = compute_metrics(y_true, y_prob)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["n_train"] = len(split.train)
    metrics["n_val"] = len(split.val)
    metrics["epochs_ran"] = len(history.history["loss"])

    rdir = cfg.results_dir
    rdir.mkdir(parents=True, exist_ok=True)
    dump_metrics(metrics, rdir / "metrics.json")
    cfg.save(rdir / "config_used.json")
    plot_history(history, rdir / "training_history.png")
    plot_confusion(metrics, rdir / "confusion_matrix.png")
    plot_roc_pr(y_true, y_prob, rdir / "roc_curve.png", rdir / "pr_curve.png")

    # a handful of Grad-CAMs, both classes, so the repo shows what the
    # model attends to instead of just claiming it works
    cam_items = [it for it in split.test if it[1] == 1][:4] + [
        it for it in split.test if it[1] == 0
    ][:2]
    cam_dir = rdir / "gradcam"
    for i, (p, y) in enumerate(cam_items):
        arr = tf.keras.preprocessing.image.load_img(
            p, color_mode="grayscale", target_size=cfg.img_size
        )
        arr = tf.keras.preprocessing.image.img_to_array(arr)
        prob = float(model.predict(arr[None], verbose=0)[0, 0])
        cls = "TB" if y == 1 else "Normal"
        save_gradcam(model, arr, cam_dir / f"cam_{i}_{cls}.png", caption=f"P(TB)={prob:.2f}")

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--lr", type=float, default=Config.learning_rate)
    ap.add_argument("--data", type=Path, default=Config.data_dir)
    args = ap.parse_args()

    cfg = Config(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, data_dir=args.data
    )
    run(cfg)


if __name__ == "__main__":
    main()
