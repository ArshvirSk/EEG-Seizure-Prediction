"""
Inference Script for Seizure Prediction

This script provides a deployment-ready interface for making seizure predictions
on new EEG data using a trained CNN+Transformer model.

Usage:
    python inference.py --model-path saved_models/cnn_transformer_final.keras --input data.npy
    python inference.py --interactive  # Interactive mode for testing
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import eeg_config, MODEL_DIR
from data.preprocessing import EEGPreprocessor


class SeizurePredictor:
    """
    Deployment-ready seizure prediction interface.

    Handles model loading, preprocessing, and inference.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        preprocessor: Optional[EEGPreprocessor] = None,
    ):
        """
        Initialize the seizure predictor.

        Args:
            model_path: Path to saved Keras model
            threshold: Classification threshold (default 0.5)
            preprocessor: EEGPreprocessor instance (creates default if None)
        """
        self.model_path = model_path
        self.threshold = threshold

        # Load model with custom objects
        print(f"Loading model from: {model_path}")
        try:
            # Try loading with custom objects for models with custom layers
            from models.cnn_encoder import CNNEncoder
            from models.transformer_encoder import TransformerEncoder
            from models.seizure_predictor import (
                SeizurePredictorCNNTransformer,
                ClassificationHead,
            )

            custom_objects = {
                "CNNEncoder": CNNEncoder,
                "TransformerEncoder": TransformerEncoder,
                "SeizurePredictorCNNTransformer": SeizurePredictorCNNTransformer,
                "ClassificationHead": ClassificationHead,
            }
            self.model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects
            )
        except Exception as e:
            # Fallback to regular loading for baseline models
            print(f"Loading with custom objects failed, trying standard load: {e}")
            self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = EEGPreprocessor(
                sampling_rate=eeg_config.original_sampling_rate,
                bandpass_low=eeg_config.bandpass_low,
                bandpass_high=eeg_config.bandpass_high,
                notch_freq=eeg_config.notch_freq,
                window_duration=eeg_config.window_duration,
                window_overlap=eeg_config.window_overlap,
            )
        else:
            self.preprocessor = preprocessor

    def preprocess(self, raw_eeg: np.ndarray) -> np.ndarray:
        """
        Preprocess raw EEG data.

        Args:
            raw_eeg: Raw EEG array of shape (n_channels, n_samples)

        Returns:
            Preprocessed and segmented windows of shape (n_windows, n_channels, window_samples)
        """
        # Preprocess
        processed = self.preprocessor.preprocess(raw_eeg)

        # Segment into windows
        windows, _ = self.preprocessor.segment(processed)

        return windows

    def predict(
        self,
        eeg_data: np.ndarray,
        preprocess: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Make seizure predictions on EEG data.

        Args:
            eeg_data: EEG data - either raw (n_channels, n_samples)
                     or preprocessed windows (n_windows, n_channels, window_samples)
            preprocess: Whether to apply preprocessing

        Returns:
            Tuple of:
                - probabilities: Predicted preictal probabilities
                - predictions: Binary predictions (0=interictal, 1=preictal)
                - info: Dictionary with additional information
        """
        # Preprocess if needed
        if preprocess and eeg_data.ndim == 2:
            windows = self.preprocess(eeg_data)
        else:
            windows = eeg_data

        if len(windows) == 0:
            return np.array([]), np.array([]), {"n_windows": 0, "alert": False}

        # Get predictions
        probabilities = self.model.predict(windows, verbose=0)
        probabilities = probabilities.flatten()

        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)

        # Compute summary info
        info = {
            "n_windows": len(windows),
            "mean_probability": float(np.mean(probabilities)),
            "max_probability": float(np.max(probabilities)),
            "n_preictal": int(np.sum(predictions)),
            "alert": bool(np.any(predictions)),
            "confidence": float(np.max(probabilities)) if np.any(predictions) else 0.0,
        }

        return probabilities, predictions, info

    def predict_continuous(
        self,
        eeg_stream: np.ndarray,
        alert_threshold: int = 3,
    ) -> Dict:
        """
        Predict on continuous EEG stream with alert logic.

        Args:
            eeg_stream: Continuous EEG data (n_channels, n_samples)
            alert_threshold: Number of consecutive preictal predictions to trigger alert

        Returns:
            Dictionary with prediction results and alert status
        """
        probs, preds, info = self.predict(eeg_stream, preprocess=True)

        if len(preds) == 0:
            return {"status": "no_data", "alert": False}

        # Check for consecutive preictal predictions
        consecutive = 0
        max_consecutive = 0

        for p in preds:
            if p == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        alert = max_consecutive >= alert_threshold

        return {
            "status": "alert" if alert else "normal",
            "alert": alert,
            "max_consecutive_preictal": max_consecutive,
            "mean_probability": info["mean_probability"],
            "n_preictal_windows": info["n_preictal"],
            "total_windows": info["n_windows"],
        }


def interactive_demo():
    """Run an interactive demonstration with synthetic data."""
    print("\n" + "=" * 60)
    print("SEIZURE PREDICTION - INTERACTIVE DEMO")
    print("=" * 60)

    # Check for trained model
    model_path = os.path.join(MODEL_DIR, "cnn_transformer_final.keras")

    if not os.path.exists(model_path):
        print("\nNo trained model found. Creating demo with synthetic model...")

        # Create a simple demo model
        from models.seizure_predictor import create_model

        model = create_model(n_channels=22, n_timesteps=2560)

        # Save it temporarily
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(model_path)
        print(f"Demo model saved to: {model_path}")

    # Initialize predictor
    predictor = SeizurePredictor(model_path)

    print("\n" + "-" * 60)
    print("Generating synthetic EEG data for demonstration...")
    print("-" * 60)

    # Create synthetic EEG data (60 seconds)
    np.random.seed(42)
    duration = 60  # seconds
    sampling_rate = 256
    n_channels = 22
    n_samples = duration * sampling_rate

    t = np.linspace(0, duration, n_samples)
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Normal EEG patterns
        eeg_data[ch] = (
            np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) * 50  # Alpha
            + np.sin(2 * np.pi * 22 * t + np.random.rand() * 2 * np.pi) * 20  # Beta
            + np.random.randn(n_samples) * 10  # Noise
        )

    print(f"Generated {duration} seconds of synthetic EEG data")
    print(f"Shape: {eeg_data.shape} (channels × samples)")

    # Make predictions
    print("\n" + "-" * 60)
    print("Running seizure prediction...")
    print("-" * 60)

    probs, preds, info = predictor.predict(eeg_data)

    print(f"\nResults:")
    print(f"  Windows analyzed: {info['n_windows']}")
    print(f"  Mean probability: {info['mean_probability']:.4f}")
    print(f"  Max probability: {info['max_probability']:.4f}")
    print(f"  Preictal windows: {info['n_preictal']}")
    print(f"  Alert triggered: {'YES' if info['alert'] else 'NO'}")

    # Continuous prediction demo
    print("\n" + "-" * 60)
    print("Continuous monitoring simulation...")
    print("-" * 60)

    result = predictor.predict_continuous(eeg_data)
    print(f"  Status: {result['status']}")
    print(f"  Max consecutive preictal: {result['max_consecutive_preictal']}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seizure Prediction Inference")

    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to saved model"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Path to input EEG data (.npy file)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive demo"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    elif args.model_path and args.input:
        # Load and predict on provided data
        predictor = SeizurePredictor(args.model_path, threshold=args.threshold)

        eeg_data = np.load(args.input)
        probs, preds, info = predictor.predict(eeg_data)

        print(f"\nResults:")
        print(f"  Windows: {info['n_windows']}")
        print(f"  Preictal detected: {info['n_preictal']}")
        print(f"  Alert: {info['alert']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
