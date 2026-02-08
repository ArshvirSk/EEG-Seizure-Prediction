"""
Training Module for Seizure Prediction Models

This module provides:
- Trainer class for model training with callbacks
- Early stopping and model checkpointing
- Learning rate scheduling
- Class weighting for imbalanced data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
from datetime import datetime


class EarlyStoppingCallback(callbacks.Callback):
    """
    Custom early stopping with model restoration.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        mode: str = "auto",
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

        self.best_weights = None
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nEarly stopping triggered at epoch {self.stopped_epoch + 1}")


class Trainer:
    """
    Training manager for seizure prediction models.

    Handles:
    - Model compilation with appropriate loss and optimizer
    - Training with callbacks (early stopping, checkpointing, LR scheduling)
    - Class weight computation for imbalanced data
    - Training history tracking
    """

    def __init__(
        self,
        model: keras.Model,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "saved_models",
        use_mixed_precision: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model: Keras model to train
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            checkpoint_dir: Directory for model checkpoints
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.history = None
        self._compile_model()

    def _compile_model(self):
        """Compile the model with optimizer and loss."""
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=(
                self.weight_decay
                if hasattr(keras.optimizers.Adam, "weight_decay")
                else None
            ),
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Args:
            y: Label array

        Returns:
            Dictionary mapping class index to weight
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_classes = len(unique)

        weights = {}
        for cls, count in zip(unique, counts):
            weights[int(cls)] = total / (n_classes * count)

        return weights

    def get_callbacks(
        self,
        early_stopping_patience: int = 15,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        lr_min: float = 1e-7,
        checkpoint_monitor: str = "val_loss",
    ) -> List[callbacks.Callback]:
        """
        Create training callbacks.

        Args:
            early_stopping_patience: Epochs to wait before stopping
            lr_patience: Epochs to wait before reducing LR
            lr_factor: Factor to reduce LR by
            lr_min: Minimum learning rate
            checkpoint_monitor: Metric to monitor for checkpointing

        Returns:
            List of callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        callback_list = [
            # Early stopping
            EarlyStoppingCallback(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=lr_min,
                verbose=1,
            ),
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.checkpoint_dir,
                    f"model_{timestamp}_epoch{{epoch:02d}}_val{{val_loss:.4f}}.keras",
                ),
                monitor=checkpoint_monitor,
                save_best_only=True,
                verbose=1,
            ),
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=os.path.join(self.checkpoint_dir, "logs", timestamp),
                histogram_freq=1,
            ),
        ]

        return callback_list

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        use_class_weights: bool = True,
        callbacks: Optional[List] = None,
        verbose: int = 1,
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            use_class_weights: Whether to use class weights
            callbacks: Custom callbacks (if None, uses default)
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("TRAINING SEIZURE PREDICTION MODEL")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")

        # Compute class weights
        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(y_train)
            print(f"Class weights: {class_weights}")

        # Get callbacks
        if callbacks is None:
            callbacks = self.get_callbacks()

        print("=" * 60 + "\n")

        # Train
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose,
        )

        return self.history.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size

        Returns:
            Dictionary of metric names to values
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)

        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

        # Create results dict
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
            print(f"  {name}: {value:.4f}")

        print("=" * 60)
        return metrics

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Input features
            batch_size: Batch size
            threshold: Classification threshold

        Returns:
            Tuple of (probabilities, binary predictions)
        """
        probabilities = self.model.predict(X, batch_size=batch_size)
        predictions = (probabilities >= threshold).astype(int)
        return probabilities, predictions

    def save_model(self, filepath: str):
        """Save the model."""
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")


def train_and_evaluate(
    model: keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "saved_models",
) -> Tuple[Dict, Dict]:
    """
    Convenience function to train and evaluate a model.

    Args:
        model: Model to train
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        test_data: (X_test, y_test)
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints

    Returns:
        Tuple of (training history, test metrics)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    trainer = Trainer(
        model=model, learning_rate=learning_rate, checkpoint_dir=checkpoint_dir
    )

    history = trainer.train(
        X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size
    )

    test_metrics = trainer.evaluate(X_test, y_test)

    return history, test_metrics


if __name__ == "__main__":
    # Test trainer with synthetic data
    print("Testing Trainer...")
    print("=" * 50)

    # Create a simple model for testing
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(22, 2560)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 22, 2560).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 22, 2560).astype(np.float32)
    y_val = np.random.randint(0, 2, 20)

    # Test trainer
    trainer = Trainer(model, learning_rate=0.001, checkpoint_dir="test_checkpoints")

    # Test class weights
    weights = trainer.compute_class_weights(y_train)
    print(f"Class weights: {weights}")

    # Test training (just 2 epochs)
    history = trainer.train(
        X_train, y_train, X_val, y_val, epochs=2, batch_size=16, verbose=1
    )

    print("\n✓ Trainer test passed!")
