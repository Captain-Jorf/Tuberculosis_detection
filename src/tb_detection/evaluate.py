"""Metrics + figures for the held-out test set.

Accuracy alone on 2-3:1 class counts is a feel-good number, so I track
sensitivity (recall on the TB class), specificity, AUC and the PR curve.
For a screening tool, sensitivity is the one you can't afford to fudge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sns.set_style("whitegrid")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n_test": int(len(y_true)),
        "n_tb": int((y_true == 1).sum()),
        "n_normal": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_tb": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tb_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_normal": float(tn / max(tn + fp, 1)),
        "f1_tb": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_confusion(metrics: Dict, out: Path) -> None:
    c = metrics["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "TB"],
        yticklabels=["Normal", "TB"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("test set confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_roc: Path, out_pr: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("sensitivity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_roc, dpi=140)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}", lw=2, color="darkred")
    ax.set_xlabel("sensitivity (recall)")
    ax.set_ylabel("precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_pr, dpi=140)
    plt.close(fig)


def plot_history(history, out: Path) -> None:
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(h["auc"], label="train")
    axes[1].plot(h["val_auc"], label="val")
    axes[1].set_title("AUC")
    axes[1].legend()
    for a in axes:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def dump_metrics(metrics: Dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
