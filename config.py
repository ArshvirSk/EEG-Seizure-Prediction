"""
EEG-Based Pediatric Seizure Prediction System
Configuration Settings

This module contains all configuration parameters for:
- Data processing (sampling rate, filtering, windowing)
- Model architecture (CNN, Transformer, Classification head)
- Training parameters (optimizer, learning rate, epochs)
"""

import os
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# PATH CONFIGURATIONS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# EEG DATA CONFIGURATIONS
# ============================================================================
@dataclass
class EEGConfig:
    """EEG signal processing configuration."""

    # Sampling parameters
    original_sampling_rate: int = 256  # CHB-MIT dataset sampling rate (Hz)
    target_sampling_rate: int = 256  # Target sampling rate after resampling

    # Channel configuration
    n_channels: int = 22  # Number of EEG channels to use

    # Filtering parameters
    bandpass_low: float = 0.5  # Low cutoff frequency (Hz)
    bandpass_high: float = 40.0  # High cutoff frequency (Hz)
    notch_freq: float = (
        60.0  # Powerline noise frequency (Hz) - 60Hz for US, 50Hz for EU
    )
    notch_width: float = 2.0  # Notch filter width (Hz)

    # Windowing parameters
    window_duration: float = 10.0  # Window duration in seconds
    window_overlap: float = 0.5  # Overlap ratio (50%)

    # Seizure labeling
    preictal_duration: int = 300  # Preictal period duration (5 minutes before seizure)
    seizure_prediction_horizon: int = 30  # Predict seizure within next 30 seconds

    @property
    def window_samples(self) -> int:
        """Number of samples per window."""
        return int(self.window_duration * self.target_sampling_rate)

    @property
    def window_stride(self) -> int:
        """Number of samples between consecutive windows."""
        return int(self.window_samples * (1 - self.window_overlap))


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
@dataclass
class CNNConfig:
    """CNN encoder configuration."""

    # Convolutional layers
    conv_filters: Tuple[int, ...] = (64, 128, 256)
    conv_kernel_sizes: Tuple[int, ...] = (7, 5, 3)
    pool_sizes: Tuple[int, ...] = (2, 2, 2)

    # Activation and regularization
    activation: str = "relu"
    dropout_rate: float = 0.3
    use_batch_norm: bool = True


@dataclass
class TransformerConfig:
    """Transformer encoder configuration."""

    # Transformer architecture
    d_model: int = 256  # Model dimension (should match CNN output)
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 4  # Number of transformer encoder layers
    d_ff: int = 512  # Feed-forward network hidden dimension

    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1

    # Positional encoding
    max_seq_length: int = 500  # Maximum sequence length for positional encoding


@dataclass
class ClassifierConfig:
    """Classification head configuration."""

    # Dense layers
    hidden_units: Tuple[int, ...] = (128, 64)

    # Regularization
    dropout_rate: float = 0.5

    # Output
    n_classes: int = 1  # Binary classification (sigmoid output)


# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================
@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer
    optimizer: str = "adam"

    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-7

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Model checkpointing
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

    # Class balancing
    use_class_weights: bool = True

    # Random seed for reproducibility
    random_seed: int = 42


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================
eeg_config = EEGConfig()
cnn_config = CNNConfig()
transformer_config = TransformerConfig()
classifier_config = ClassifierConfig()
training_config = TrainingConfig()


def print_config():
    """Print all configuration settings."""
    print("=" * 60)
    print("EEG SEIZURE PREDICTION SYSTEM - CONFIGURATION")
    print("=" * 60)

    configs = [
        ("EEG Processing", eeg_config),
        ("CNN Encoder", cnn_config),
        ("Transformer Encoder", transformer_config),
        ("Classifier Head", classifier_config),
        ("Training", training_config),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        print("-" * 40)
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config()
