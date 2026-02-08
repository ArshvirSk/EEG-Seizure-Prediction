"""
EEG Signal Preprocessing Module

This module provides comprehensive EEG signal preprocessing including:
- Bandpass filtering (0.5-40 Hz)
- Notch filtering for powerline noise removal (50/60 Hz)
- Z-score normalization
- Windowing and segmentation with overlap

Uses MNE-Python and SciPy for signal processing.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Tuple, List, Optional
import warnings


class EEGPreprocessor:
    """
    Comprehensive EEG signal preprocessor.

    Implements the preprocessing pipeline:
    1. Bandpass filtering (0.5-40 Hz)
    2. Notch filtering (50/60 Hz powerline noise)
    3. Z-score normalization
    4. Windowing with overlap
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        bandpass_low: float = 0.5,
        bandpass_high: float = 40.0,
        notch_freq: float = 60.0,
        notch_width: float = 2.0,
        window_duration: float = 10.0,
        window_overlap: float = 0.5,
    ):
        """
        Initialize the EEG preprocessor.

        Args:
            sampling_rate: Sampling frequency in Hz
            bandpass_low: Low cutoff frequency for bandpass filter
            bandpass_high: High cutoff frequency for bandpass filter
            notch_freq: Notch filter center frequency (powerline noise)
            notch_width: Width of notch filter in Hz
            window_duration: Duration of each window in seconds
            window_overlap: Overlap ratio between consecutive windows (0-1)
        """
        self.sampling_rate = sampling_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.notch_width = notch_width
        self.window_duration = window_duration
        self.window_overlap = window_overlap

        # Pre-compute filter coefficients
        self._init_filters()

    def _init_filters(self):
        """Initialize filter coefficients."""
        nyquist = self.sampling_rate / 2

        # Bandpass filter (Butterworth, order 4)
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist

        # Clamp values to valid range
        low = max(0.001, min(low, 0.99))
        high = max(0.001, min(high, 0.99))

        if low >= high:
            low = 0.001
            high = 0.99

        self.bandpass_b, self.bandpass_a = signal.butter(4, [low, high], btype="band")

        # Notch filter (IIR notch)
        notch_normalized = self.notch_freq / nyquist
        if 0 < notch_normalized < 1:
            quality_factor = self.notch_freq / self.notch_width
            self.notch_b, self.notch_a = signal.iirnotch(
                notch_normalized, quality_factor
            )
        else:
            self.notch_b = None
            self.notch_a = None

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.

        Args:
            data: EEG data of shape (n_channels, n_samples) or (n_samples,)

        Returns:
            Filtered data with same shape as input
        """
        if data.ndim == 1:
            return signal.filtfilt(self.bandpass_b, self.bandpass_a, data)
        else:
            return np.array(
                [signal.filtfilt(self.bandpass_b, self.bandpass_a, ch) for ch in data]
            )

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove powerline noise.

        Args:
            data: EEG data of shape (n_channels, n_samples) or (n_samples,)

        Returns:
            Filtered data with same shape as input
        """
        if self.notch_b is None:
            return data

        if data.ndim == 1:
            return signal.filtfilt(self.notch_b, self.notch_a, data)
        else:
            return np.array(
                [signal.filtfilt(self.notch_b, self.notch_a, ch) for ch in data]
            )

    def normalize(self, data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize EEG data.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            Normalized data with same shape as input
        """
        if method == "zscore":
            # Z-score normalization per channel
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            return (data - mean) / std

        elif method == "minmax":
            # Min-max normalization per channel
            min_val = np.min(data, axis=-1, keepdims=True)
            max_val = np.max(data, axis=-1, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return (data - min_val) / range_val

        elif method == "robust":
            # Robust normalization using median and IQR
            median = np.median(data, axis=-1, keepdims=True)
            q75, q25 = np.percentile(data, [75, 25], axis=-1, keepdims=True)
            iqr = q75 - q25
            iqr[iqr == 0] = 1
            return (data - median) / iqr

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def segment(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segment EEG data into overlapping windows.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            labels: Optional sample-level labels of shape (n_samples,)

        Returns:
            Tuple of:
                - Segmented data of shape (n_windows, n_channels, window_samples)
                - Window labels of shape (n_windows,) if labels provided, else None
        """
        n_channels, n_samples = data.shape
        window_samples = int(self.window_duration * self.sampling_rate)
        stride = int(window_samples * (1 - self.window_overlap))

        # Calculate number of windows
        n_windows = (n_samples - window_samples) // stride + 1

        if n_windows <= 0:
            warnings.warn(
                f"Data too short for windowing. Need at least {window_samples} samples."
            )
            return np.array([]).reshape(0, n_channels, window_samples), None

        # Create windows
        windows = np.zeros((n_windows, n_channels, window_samples))
        window_labels = np.zeros(n_windows) if labels is not None else None

        for i in range(n_windows):
            start = i * stride
            end = start + window_samples
            windows[i] = data[:, start:end]

            if labels is not None:
                # Use majority vote for window label
                window_labels[i] = np.round(np.mean(labels[start:end]))

        return windows, window_labels

    def preprocess(
        self,
        data: np.ndarray,
        apply_bandpass: bool = True,
        apply_notch: bool = True,
        apply_normalization: bool = True,
        normalization_method: str = "zscore",
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to EEG data.

        Args:
            data: Raw EEG data of shape (n_channels, n_samples)
            apply_bandpass: Whether to apply bandpass filter
            apply_notch: Whether to apply notch filter
            apply_normalization: Whether to normalize
            normalization_method: Method for normalization

        Returns:
            Preprocessed data with same shape as input
        """
        processed = data.copy()

        # Apply filters
        if apply_bandpass:
            processed = self.bandpass_filter(processed)

        if apply_notch:
            processed = self.notch_filter(processed)

        # Apply normalization
        if apply_normalization:
            processed = self.normalize(processed, method=normalization_method)

        return processed

    def preprocess_and_segment(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None, **preprocess_kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply preprocessing and segmentation in one step.

        Args:
            data: Raw EEG data of shape (n_channels, n_samples)
            labels: Optional sample-level labels of shape (n_samples,)
            **preprocess_kwargs: Arguments for preprocess()

        Returns:
            Tuple of segmented windows and labels
        """
        processed = self.preprocess(data, **preprocess_kwargs)
        return self.segment(processed, labels)


def create_sample_labels(
    n_samples: int,
    sampling_rate: int,
    seizure_times: List[Tuple[int, int]],
    preictal_duration: int = 300,
) -> np.ndarray:
    """
    Create sample-level labels for EEG data.

    Args:
        n_samples: Total number of samples
        sampling_rate: Sampling rate in Hz
        seizure_times: List of (start, end) times in seconds
        preictal_duration: Duration of preictal period before seizure (seconds)

    Returns:
        Array of labels: 0=interictal, 1=preictal
        (Ictal periods are excluded/marked as -1)
    """
    labels = np.zeros(n_samples)

    for start, end in seizure_times:
        start_sample = start * sampling_rate
        end_sample = end * sampling_rate

        # Mark ictal period (to be excluded)
        labels[start_sample:end_sample] = -1

        # Mark preictal period
        preictal_start = max(0, start_sample - preictal_duration * sampling_rate)
        labels[preictal_start:start_sample] = 1

    return labels


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("Testing EEG Preprocessor...")
    print("=" * 50)

    # Create synthetic EEG data
    np.random.seed(42)
    sampling_rate = 256
    duration = 60  # 60 seconds
    n_channels = 22
    n_samples = duration * sampling_rate

    # Simulate EEG: mix of frequencies + noise
    t = np.linspace(0, duration, n_samples)
    eeg = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Add alpha (8-12 Hz), beta (13-30 Hz), and noise
        eeg[ch] = (
            np.sin(2 * np.pi * 10 * t) * 50  # Alpha
            + np.sin(2 * np.pi * 20 * t) * 20  # Beta
            + np.sin(2 * np.pi * 60 * t) * 30  # Powerline noise
            + np.random.randn(n_samples) * 10  # Random noise
        )

    print(f"Input shape: {eeg.shape}")

    # Initialize preprocessor
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate, window_duration=10.0, window_overlap=0.5
    )

    # Preprocess
    processed = preprocessor.preprocess(eeg)
    print(f"Preprocessed shape: {processed.shape}")

    # Segment
    windows, _ = preprocessor.segment(processed)
    print(f"Windows shape: {windows.shape}")
    print(f"Expected windows: {(n_samples - 2560) // 1280 + 1}")

    print("\n✓ Preprocessing test passed!")
