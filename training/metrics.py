"""
Evaluation Metrics Module

This module provides comprehensive metrics for seizure prediction:
- Standard classification metrics (accuracy, sensitivity, specificity, F1)
- ROC-AUC and Precision-Recall curves
- False Prediction Rate (FPR) per hour - critical for clinical applications
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1) - if None, computed from y_prob
        y_prob: Predicted probabilities (0-1)
        threshold: Classification threshold

    Returns:
        Dictionary of metric names to values
    """
    # Convert probabilities to predictions if needed
    if y_pred is None and y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Recall / TPR
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # TNR
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    # AUC if probabilities available
    if y_prob is not None:
        y_prob = np.asarray(y_prob).flatten()
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            metrics["auc_pr"] = average_precision_score(y_true, y_prob)
        except ValueError:
            # If only one class present
            metrics["auc_roc"] = 0.0
            metrics["auc_pr"] = 0.0

    return metrics


def compute_false_prediction_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_duration: float = 10.0,  # seconds
    total_duration_hours: Optional[float] = None,
) -> float:
    """
    Compute False Prediction Rate per hour.

    This is a critical clinical metric that measures how many false alarms
    occur per hour of recording. Lower is better.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        window_duration: Duration of each window in seconds
        total_duration_hours: Total recording duration in hours (if None, computed from data)

    Returns:
        False predictions per hour
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Count false positives
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # Calculate total duration
    if total_duration_hours is None:
        # Estimate from number of windows
        total_seconds = len(y_true) * window_duration
        total_duration_hours = total_seconds / 3600.0

    # Avoid division by zero
    if total_duration_hours <= 0:
        return float("inf")

    return fp / total_duration_hours


def compute_seizure_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    window_duration: float = 10.0,
    prediction_horizon: int = 30,  # seconds
) -> Dict[str, float]:
    """
    Compute seizure prediction specific metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        window_duration: Window duration in seconds
        prediction_horizon: Prediction horizon in seconds

    Returns:
        Dictionary of metrics
    """
    # Get standard metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Add false prediction rate
    metrics["false_prediction_rate_per_hour"] = compute_false_prediction_rate(
        y_true, y_pred, window_duration
    )

    # Compute detection delay (if seizures present)
    # This is simplified - in practice would need seizure event timestamps

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    min_sensitivity: float = 0.8,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'sensitivity', 'specificity', 'youden')
        min_sensitivity: Minimum required sensitivity constraint

    Returns:
        Tuple of (optimal threshold, metrics at that threshold)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = -1
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        # Check sensitivity constraint
        if metrics["sensitivity"] < min_sensitivity:
            continue

        # Score based on metric
        if metric == "f1":
            score = metrics["f1_score"]
        elif metric == "sensitivity":
            score = metrics["sensitivity"]
        elif metric == "specificity":
            score = metrics["specificity"]
        elif metric == "youden":
            score = metrics["sensitivity"] + metrics["specificity"] - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = metrics

    return best_threshold, best_metrics


def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Get ROC curve data for plotting.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (fpr, tpr, thresholds, auc)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return fpr, tpr, thresholds, auc


def get_precision_recall_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Get Precision-Recall curve data for plotting.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (precision, recall, thresholds, avg_precision)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    return precision, recall, thresholds, avg_precision


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Tuple[str, str] = ("Interictal", "Preictal"),
) -> str:
    """
    Print a detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        target_names: Names for the classes

    Returns:
        Report string
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    # Standard sklearn report
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    # Additional metrics
    metrics = compute_seizure_prediction_metrics(y_true, y_pred, y_prob)

    print("-" * 60)
    print("Seizure Prediction Specific Metrics:")
    print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(
        f"  False Prediction Rate: {metrics['false_prediction_rate_per_hour']:.4f}/hour"
    )

    if "auc_roc" in metrics:
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")

    print("=" * 60)

    return report


def compare_models(
    model_results: Dict[str, Dict[str, float]],
) -> str:
    """
    Compare results from multiple models.

    Args:
        model_results: Dict mapping model name to metrics dict

    Returns:
        Comparison table string
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Key metrics to compare
    key_metrics = [
        "accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "auc_roc",
        "false_prediction_rate_per_hour",
    ]

    # Header
    header = f"{'Model':<25}" + "".join(f"{m:<15}" for m in key_metrics)
    print(header)
    print("-" * 80)

    # Rows
    for model_name, metrics in model_results.items():
        row = f"{model_name:<25}"
        for metric in key_metrics:
            value = metrics.get(metric, "N/A")
            if isinstance(value, float):
                row += f"{value:<15.4f}"
            else:
                row += f"{value:<15}"
        print(row)

    print("=" * 80)

    # Find best model
    best_model = max(model_results.items(), key=lambda x: x[1].get("sensitivity", 0))[0]
    print(f"\nBest model (by sensitivity): {best_model}")

    return ""


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics Module...")
    print("=" * 50)

    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 500

    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.clip(y_true + np.random.randn(n_samples) * 0.3, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)

    # Test basic metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\nBasic Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    # Test false prediction rate
    fpr = compute_false_prediction_rate(y_true, y_pred)
    print(f"\nFalse Prediction Rate: {fpr:.2f}/hour")

    # Test threshold optimization
    best_thresh, best_metrics = find_optimal_threshold(y_true, y_prob, metric="f1")
    print(f"\nOptimal threshold (F1): {best_thresh:.2f}")
    print(f"F1 at optimal: {best_metrics['f1_score']:.4f}")

    # Test classification report
    print_classification_report(y_true, y_pred, y_prob)

    print("\n✓ Metrics test passed!")
