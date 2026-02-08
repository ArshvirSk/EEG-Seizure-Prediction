"""
EEG Dataset Module

This module provides:
- EEGDataset class for loading and managing preprocessed EEG data
- Data loading utilities for training/validation/test splits
- Class balancing for handling imbalanced preictal/interictal data

Compatible with TensorFlow/Keras training pipeline.
"""

import numpy as np
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE not installed. Using basic EDF reading fallback.")


@dataclass
class EEGSample:
    """Container for a single EEG sample."""

    data: np.ndarray  # Shape: (n_channels, n_samples)
    label: int  # 0=interictal, 1=preictal
    patient_id: str
    file_name: str
    window_idx: int


class EEGDataset:
    """
    Dataset class for EEG seizure prediction.

    Handles:
    - Loading EDF files using MNE
    - Preprocessing and windowing
    - Label assignment based on seizure annotations
    - Train/validation/test splitting
    """

    def __init__(
        self,
        data_dir: str,
        preprocessor=None,
        preictal_duration: int = 300,
        exclude_ictal: bool = True,
    ):
        """
        Initialize the EEG dataset.

        Args:
            data_dir: Directory containing patient data folders
            preprocessor: EEGPreprocessor instance (or None to create default)
            preictal_duration: Duration of preictal period in seconds
            exclude_ictal: Whether to exclude ictal (during seizure) segments
        """
        self.data_dir = data_dir
        self.preictal_duration = preictal_duration
        self.exclude_ictal = exclude_ictal

        # Initialize preprocessor
        if preprocessor is None:
            from .preprocessing import EEGPreprocessor

            self.preprocessor = EEGPreprocessor()
        else:
            self.preprocessor = preprocessor

        # Data storage
        self.samples: List[EEGSample] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        # Expected shape parameters (from config)
        from config import eeg_config

        self.n_channels = eeg_config.n_channels  # 22
        self.window_samples = int(
            eeg_config.window_duration * eeg_config.original_sampling_rate
        )  # 2560

    def load_edf_file(
        self, file_path: str, target_channels: List[str] = None
    ) -> Tuple[np.ndarray, int, List[str]]:
        """
        Load an EDF file and return EEG data.

        Args:
            file_path: Path to EDF file
            target_channels: List of channel names to load (None for all)

        Returns:
            Tuple of (data, sampling_rate, channel_names)
        """
        if MNE_AVAILABLE:
            return self._load_with_mne(file_path, target_channels)
        else:
            return self._load_with_fallback(file_path)

    def _load_with_mne(
        self, file_path: str, target_channels: List[str] = None
    ) -> Tuple[np.ndarray, int, List[str]]:
        """Load EDF using MNE library."""
        # Read raw EDF
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Get sampling rate
        sampling_rate = int(raw.info["sfreq"])

        # Get channel names
        all_channels = raw.ch_names

        # Select channels
        if target_channels is not None:
            # Find matching channels
            available = [ch for ch in target_channels if ch in all_channels]
            if len(available) < len(target_channels):
                missing = set(target_channels) - set(available)
                warnings.warn(f"Missing channels: {missing}")
            raw.pick_channels(available)

        # Get data
        data = raw.get_data()
        channel_names = raw.ch_names

        return data, sampling_rate, channel_names

    def _load_with_fallback(self, file_path: str) -> Tuple[np.ndarray, int, List[str]]:
        """Basic EDF reading fallback when MNE is not available."""
        raise NotImplementedError(
            "MNE is required for EDF reading. Install with: pip install mne"
        )

    def load_patient_data(
        self,
        patient_id: str,
        seizure_info: List[Dict],
        max_interictal_ratio: float = 3.0,
    ) -> int:
        """
        Load and preprocess data for a single patient.

        Args:
            patient_id: Patient identifier (e.g., "chb01")
            seizure_info: List of dicts with 'file_name', 'start_time', 'end_time'
            max_interictal_ratio: Maximum ratio of interictal to preictal samples

        Returns:
            Number of samples loaded
        """
        patient_dir = os.path.join(self.data_dir, patient_id)

        if not os.path.exists(patient_dir):
            warnings.warn(f"Patient directory not found: {patient_dir}")
            return 0

        # Group seizures by file
        seizures_by_file = {}
        for s in seizure_info:
            # Handle both SeizureInfo dataclass and dict formats
            if hasattr(s, "file_name"):
                # It's a SeizureInfo dataclass
                fname = s.file_name
                start = s.start_time
                end = s.end_time
            else:
                # It's a dictionary
                fname = s.get("file_name", s.get("filename"))
                start = s["start_time"]
                end = s["end_time"]

            if fname not in seizures_by_file:
                seizures_by_file[fname] = []
            seizures_by_file[fname].append((start, end))

        samples_loaded = 0

        # Process each EDF file
        for file_name in os.listdir(patient_dir):
            if not file_name.endswith(".edf"):
                continue

            file_path = os.path.join(patient_dir, file_name)

            try:
                # Load EDF
                data, sr, channels = self.load_edf_file(file_path)

                # Get seizure times for this file
                file_seizures = seizures_by_file.get(file_name, [])

                # Create sample-level labels
                n_samples = data.shape[1]
                labels = self._create_labels(n_samples, sr, file_seizures)

                # Preprocess and segment
                windows, window_labels = self.preprocessor.preprocess_and_segment(
                    data, labels
                )

                if windows.shape[0] == 0:
                    continue

                # Filter out ictal segments
                if self.exclude_ictal:
                    valid_mask = window_labels >= 0
                    windows = windows[valid_mask]
                    window_labels = window_labels[valid_mask]

                # Add samples
                for i, (w, l) in enumerate(zip(windows, window_labels)):
                    self.samples.append(
                        EEGSample(
                            data=w,
                            label=int(l),
                            patient_id=patient_id,
                            file_name=file_name,
                            window_idx=i,
                        )
                    )

                samples_loaded += len(windows)

            except Exception as e:
                warnings.warn(f"Error processing {file_path}: {e}")
                continue

        return samples_loaded

    def _create_labels(
        self, n_samples: int, sampling_rate: int, seizure_times: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Create sample-level labels."""
        labels = np.zeros(n_samples)  # 0 = interictal

        for start, end in seizure_times:
            start_sample = int(start * sampling_rate)
            end_sample = int(end * sampling_rate)

            # Mark ictal period as -1 (to be excluded)
            start_sample = min(start_sample, n_samples)
            end_sample = min(end_sample, n_samples)
            labels[start_sample:end_sample] = -1

            # Mark preictal period as 1
            preictal_start = max(
                0, start_sample - self.preictal_duration * sampling_rate
            )
            labels[preictal_start:start_sample] = 1

        return labels

    def build_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build numpy arrays from loaded samples.

        Returns:
            Tuple of (X, y) arrays
        """
        if len(self.samples) == 0:
            raise ValueError("No samples loaded. Call load_patient_data first.")

        # Get the target shape from the expected config
        target_channels = self.n_channels
        target_timesteps = self.window_samples

        # Filter samples with consistent shapes
        valid_samples = []
        for s in self.samples:
            if s.data.shape == (target_channels, target_timesteps):
                valid_samples.append(s)
            elif (
                s.data.shape[0] >= target_channels
                and s.data.shape[1] == target_timesteps
            ):
                # Take first n_channels if we have more channels
                s.data = s.data[:target_channels, :]
                valid_samples.append(s)

        if len(valid_samples) == 0:
            # Fallback: find most common shape
            shapes = [s.data.shape for s in self.samples]
            from collections import Counter

            shape_counts = Counter(shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]
            print(f"  Using most common shape: {most_common_shape}")
            valid_samples = [
                s for s in self.samples if s.data.shape == most_common_shape
            ]

        if len(valid_samples) < len(self.samples):
            print(
                f"  Filtered {len(self.samples) - len(valid_samples)} samples with inconsistent shapes"
            )
            print(
                f"  Keeping {len(valid_samples)} samples with shape ({target_channels}, {target_timesteps})"
            )

        # Limit samples to prevent memory issues (max ~10GB with float32)
        max_samples = 20000  # ~4.5 GB with float32
        if len(valid_samples) > max_samples:
            print(f"  Limiting to {max_samples} samples to prevent memory issues")
            # Stratified sampling to maintain class balance
            preictal_samples = [s for s in valid_samples if s.label == 1]
            interictal_samples = [s for s in valid_samples if s.label == 0]

            # Keep all preictal (seizure) samples, undersample interictal
            np.random.shuffle(interictal_samples)
            n_interictal = min(
                len(interictal_samples), max_samples - len(preictal_samples)
            )
            valid_samples = preictal_samples + interictal_samples[:n_interictal]
            np.random.shuffle(valid_samples)
            print(
                f"  Final: {len(preictal_samples)} preictal, {n_interictal} interictal"
            )

        self.samples = valid_samples
        # Use float32 to halve memory usage (float64 -> float32)
        self.X = np.array([s.data for s in self.samples], dtype=np.float32)
        self.y = np.array([s.label for s in self.samples], dtype=np.float32)

        return self.X, self.y

    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes."""
        if self.y is None:
            self.build_arrays()

        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))

    def balance_classes(
        self, method: str = "undersample"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset classes.

        Args:
            method: 'undersample' or 'oversample'

        Returns:
            Balanced (X, y) arrays
        """
        if self.X is None:
            self.build_arrays()

        # Get indices for each class
        idx_0 = np.where(self.y == 0)[0]
        idx_1 = np.where(self.y == 1)[0]

        n_0, n_1 = len(idx_0), len(idx_1)

        if method == "undersample":
            # Undersample majority class
            if n_0 > n_1:
                idx_0 = np.random.choice(idx_0, n_1, replace=False)
            else:
                idx_1 = np.random.choice(idx_1, n_0, replace=False)

        elif method == "oversample":
            # Oversample minority class
            if n_0 < n_1:
                idx_0 = np.random.choice(idx_0, n_1, replace=True)
            else:
                idx_1 = np.random.choice(idx_1, n_0, replace=True)

        # Combine and shuffle
        indices = np.concatenate([idx_0, idx_1])
        np.random.shuffle(indices)

        return self.X[indices], self.y[indices]


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    random_seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Create train/validation/test splits.

    Args:
        X: Feature array of shape (n_samples, n_channels, n_timesteps)
        y: Label array of shape (n_samples,)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size (for info only in this implementation)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    np.random.seed(random_seed)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Create splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_channels: int = 22,
    n_timesteps: int = 2560,
    preictal_ratio: float = 0.3,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic EEG dataset for testing.

    Args:
        n_samples: Number of samples
        n_channels: Number of EEG channels
        n_timesteps: Number of time steps per sample
        preictal_ratio: Ratio of preictal samples
        random_seed: Random seed

    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(random_seed)

    n_preictal = int(n_samples * preictal_ratio)
    n_interictal = n_samples - n_preictal

    # Create synthetic EEG patterns
    t = np.linspace(0, 10, n_timesteps)

    X = np.zeros((n_samples, n_channels, n_timesteps))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        for ch in range(n_channels):
            # Base EEG: alpha + beta + noise
            base = (
                np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) * 20
                + np.sin(2 * np.pi * 22 * t + np.random.rand() * 2 * np.pi) * 10
                + np.random.randn(n_timesteps) * 5
            )

            if i < n_preictal:
                # Preictal: add higher frequency activity
                base += np.sin(2 * np.pi * 35 * t) * 15
                base += np.random.randn(n_timesteps) * 8
                y[i] = 1

            X[i, ch] = base

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing EEG Dataset...")
    print("=" * 50)

    # Create synthetic dataset
    X, y = create_synthetic_dataset(n_samples=500)
    print(f"Synthetic dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Test data splitting
    train, val, test = create_data_loaders(X, y)
    print("\n✓ Dataset test passed!")
