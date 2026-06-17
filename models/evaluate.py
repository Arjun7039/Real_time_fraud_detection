"""Unified evaluation module — precision, recall, AUC, F1.

Shared evaluation function used by all training scripts.
Generates precision-recall curves, AUC-ROC, and F1 scores
at the optimal threshold.

Usage:
    from models.evaluate import evaluate_model
    metrics = evaluate_model(y_true, y_pred_proba, model_name="xgboost")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the threshold that maximises F1 score.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        float: Optimal classification threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Compute F1 at each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_dir: str = "reports",
) -> dict:
    """Compute all evaluation metrics and save PR curve plot.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Name of the model (used in filenames and titles).
        output_dir: Directory to save plots and reports.

    Returns:
        dict: Dictionary of metric name → value pairs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Optimal threshold
    threshold = find_optimal_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    # Core metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "f1_score": round(f1, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "optimal_threshold": round(threshold, 5),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # ---- Precision-Recall Curve ----
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recalls, precisions, linewidth=2, color="#E63946")
    ax.fill_between(recalls, precisions, alpha=0.15, color="#E63946")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"{model_name} — Precision-Recall Curve (PR-AUC={pr_auc:.4f})", fontsize=14)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.6, label="95% Precision")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f"{model_name}_pr_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # ---- Print summary ----
    print(f"\n{'='*50}")
    print(f"  {model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  ROC-AUC          : {metrics['roc_auc']}")
    print(f"  PR-AUC           : {metrics['pr_auc']}")
    print(f"  F1 Score         : {metrics['f1_score']}")
    print(f"  Precision        : {metrics['precision']}")
    print(f"  Recall           : {metrics['recall']}")
    print(f"  Optimal Threshold: {metrics['optimal_threshold']}")
    print(f"  Confusion Matrix : TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"  PR Curve saved   : {plot_path}")
    print(f"{'='*50}\n")

    return metrics
