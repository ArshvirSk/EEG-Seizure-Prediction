"""
Visualization Module for EEG Seizure Prediction

This module provides visualization functions for:
- Training history (loss, accuracy curves)
- ROC and PR curves
- Confusion matrices
- EEG signal plots
- Attention weight heatmaps for model interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot training history showing loss and metrics over epochs.

    Args:
        history: Training history dictionary from model.fit()
        save_path: Path to save the figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history.get("loss", [])) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history.get("loss", []), "b-", label="Training Loss", linewidth=2)
    ax.plot(
        epochs, history.get("val_loss", []), "r--", label="Validation Loss", linewidth=2
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(
        epochs,
        history.get("accuracy", []),
        "b-",
        label="Training Accuracy",
        linewidth=2,
    )
    ax.plot(
        epochs,
        history.get("val_accuracy", []),
        "r--",
        label="Validation Accuracy",
        linewidth=2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[2]
    if "auc" in history:
        ax.plot(epochs, history.get("auc", []), "b-", label="Training AUC", linewidth=2)
        ax.plot(
            epochs,
            history.get("val_auc", []),
            "r--",
            label="Validation AUC",
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Training and Validation AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under ROC curve
        model_name: Name of the model for legend
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"{model_name} (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multiple_roc_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot multiple ROC curves for model comparison.

    Args:
        curves: Dict mapping model name to (fpr, tpr, auc)
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(curves)))

    for (name, (fpr, tpr, auc)), color in zip(curves.items(), colors):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, str] = ("Interictal", "Preictal"),
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for the classes
        normalize: Whether to normalize the matrix
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        "Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_eeg_sample(
    eeg_data: np.ndarray,
    sampling_rate: int = 256,
    channel_names: Optional[List[str]] = None,
    title: str = "EEG Signal",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot EEG signals across channels.

    Args:
        eeg_data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        channel_names: Names for each channel
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_channels, n_samples = eeg_data.shape
    time = np.arange(n_samples) / sampling_rate

    if channel_names is None:
        channel_names = [f"Ch {i+1}" for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    if n_channels == 1:
        axes = [axes]

    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(time, eeg_data[i], "b-", linewidth=0.5)
        ax.set_ylabel(ch_name, fontsize=8)
        ax.set_xlim([time[0], time[-1]])
        ax.grid(True, alpha=0.3)

        # Remove x-axis labels except for last plot
        if i < n_channels - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    layer_idx: int = 0,
    head_idx: int = 0,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot attention weight heatmap for interpretability.

    Args:
        attention_weights: Attention weights from transformer
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # attention_weights shape: (batch, heads, seq_len, seq_len)
    if len(attention_weights.shape) == 4:
        attn = attention_weights[0, head_idx, :, :]  # First batch item
    else:
        attn = attention_weights

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)
    ax.set_title(f"{title} (Layer {layer_idx}, Head {head_idx})", fontsize=14)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_heads(
    attention_weights: np.ndarray,
    n_heads_to_show: int = 4,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot attention weights for multiple heads.

    Args:
        attention_weights: Attention weights (batch, heads, seq_len, seq_len)
        n_heads_to_show: Number of heads to display
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_heads = min(n_heads_to_show, attention_weights.shape[1])
    n_cols = 2
    n_rows = (n_heads + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_heads):
        attn = attention_weights[0, i, :, :]
        im = axes[i].imshow(attn, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Head {i}")
        axes[i].set_xlabel("Key")
        axes[i].set_ylabel("Query")
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    # Hide empty subplots
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Multi-Head Attention Weights", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = [
        "accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "auc_roc",
    ],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot bar chart comparing multiple models across metrics.

    Args:
        results: Dict mapping model name to metrics dict
        metrics: List of metrics to compare
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        values = [results[model_name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=color)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Test visualizations
    print("Testing Visualization Module...")
    print("=" * 50)

    # Create dummy data
    np.random.seed(42)

    # Test training history plot
    history = {
        "loss": np.exp(-np.linspace(0, 2, 20)) + np.random.randn(20) * 0.05,
        "val_loss": np.exp(-np.linspace(0, 1.5, 20)) + np.random.randn(20) * 0.08,
        "accuracy": 1
        - np.exp(-np.linspace(0, 2, 20)) * 0.5
        + np.random.randn(20) * 0.02,
        "val_accuracy": 1
        - np.exp(-np.linspace(0, 1.5, 20)) * 0.6
        + np.random.randn(20) * 0.03,
        "auc": 1 - np.exp(-np.linspace(0, 2, 20)) * 0.3 + np.random.randn(20) * 0.01,
        "val_auc": 1
        - np.exp(-np.linspace(0, 1.5, 20)) * 0.4
        + np.random.randn(20) * 0.02,
    }

    fig = plot_training_history(history)
    plt.close(fig)

    # Test ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)
    fig = plot_roc_curve(fpr, tpr, 0.85, "Test Model")
    plt.close(fig)

    # Test confusion matrix
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    fig = plot_confusion_matrix(y_true, y_pred)
    plt.close(fig)

    # Test EEG plot
    eeg = np.random.randn(8, 512) * 50
    fig = plot_eeg_sample(eeg[:8], title="Sample EEG")
    plt.close(fig)

    # Test attention weights
    attn = np.random.rand(1, 8, 20, 20)
    fig = plot_attention_weights(attn)
    plt.close(fig)

    print("✓ Visualization test passed!")
