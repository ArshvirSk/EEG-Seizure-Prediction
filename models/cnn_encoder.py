"""
CNN Encoder Module for EEG Feature Extraction

This module implements a 1D Convolutional Neural Network encoder for
extracting spatial and local temporal features from EEG signals.

Architecture:
- Multiple 1D Conv layers with increasing filters
- Batch Normalization for training stability
- ReLU activation
- MaxPooling for downsampling
- Dropout for regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional


class ConvBlock(layers.Layer):
    """
    Single convolutional block: Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )
        self.batch_norm = layers.BatchNormalization() if use_batch_norm else None
        self.activation = layers.ReLU()
        self.pool = layers.MaxPooling1D(pool_size=pool_size, padding="same")
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        return x


class CNNEncoder(Model):
    """
    CNN Encoder for EEG feature extraction.

    Takes raw EEG windows and extracts local spatial-temporal features
    using stacked 1D convolutional layers.

    Input shape: (batch_size, n_channels, n_timesteps) or (batch_size, n_timesteps, n_channels)
    Output shape: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        conv_filters: Tuple[int, ...] = (64, 128, 256),
        conv_kernel_sizes: Tuple[int, ...] = (7, 5, 3),
        pool_sizes: Tuple[int, ...] = (2, 2, 2),
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        input_channels_last: bool = False,
        **kwargs,
    ):
        """
        Initialize CNN Encoder.

        Args:
            conv_filters: Number of filters for each conv block
            conv_kernel_sizes: Kernel sizes for each conv block
            pool_sizes: Pool sizes for each conv block
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            input_channels_last: If True, input is (batch, time, channels)
                               If False, input is (batch, channels, time)
        """
        super().__init__(**kwargs)

        self.input_channels_last = input_channels_last

        # Validate inputs
        n_blocks = len(conv_filters)
        if len(conv_kernel_sizes) != n_blocks or len(pool_sizes) != n_blocks:
            raise ValueError(
                "conv_filters, conv_kernel_sizes, and pool_sizes must have same length"
            )

        # Build convolutional blocks
        self.conv_blocks = []
        for i, (filters, kernel, pool) in enumerate(
            zip(conv_filters, conv_kernel_sizes, pool_sizes)
        ):
            block = ConvBlock(
                filters=filters,
                kernel_size=kernel,
                pool_size=pool,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                name=f"conv_block_{i}",
            )
            self.conv_blocks.append(block)

        # Store output dimension
        self.d_model = conv_filters[-1]

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: EEG tensor of shape (batch, channels, time) or (batch, time, channels)
            training: Whether in training mode

        Returns:
            Feature tensor of shape (batch, seq_len, d_model)
        """
        # Ensure input is (batch, time, channels) for Conv1D
        x = inputs

        if not self.input_channels_last:
            # Convert from (batch, channels, time) to (batch, time, channels)
            x = tf.transpose(x, perm=[0, 2, 1])

        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x, training=training)

        return x

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate output shape given input shape.

        Args:
            input_shape: (batch, channels, time) or (batch, time, channels)

        Returns:
            Output shape tuple
        """
        if self.input_channels_last:
            time_dim = input_shape[1]
        else:
            time_dim = input_shape[2]

        # Calculate time dimension after pooling
        for pool_size in [2, 2, 2]:  # Default pool sizes
            time_dim = (time_dim + 1) // pool_size

        return (input_shape[0], time_dim, self.d_model)


def create_cnn_encoder(
    n_channels: int = 22,
    n_timesteps: int = 2560,
    conv_filters: Tuple[int, ...] = (64, 128, 256),
    **kwargs,
) -> CNNEncoder:
    """
    Factory function to create a CNN encoder.

    Args:
        n_channels: Number of EEG channels
        n_timesteps: Number of time steps per window
        conv_filters: Filter sizes for conv layers
        **kwargs: Additional arguments for CNNEncoder

    Returns:
        Configured CNNEncoder instance
    """
    encoder = CNNEncoder(conv_filters=conv_filters, **kwargs)

    # Build the model with dummy input
    dummy_input = tf.zeros((1, n_channels, n_timesteps))
    _ = encoder(dummy_input)

    return encoder


if __name__ == "__main__":
    # Test CNN Encoder
    print("Testing CNN Encoder...")
    print("=" * 50)

    # Create encoder
    encoder = CNNEncoder(
        conv_filters=(64, 128, 256),
        conv_kernel_sizes=(7, 5, 3),
        pool_sizes=(2, 2, 2),
        dropout_rate=0.3,
    )

    # Test with dummy input
    batch_size = 4
    n_channels = 22
    n_timesteps = 2560

    dummy_input = tf.random.normal((batch_size, n_channels, n_timesteps))
    output = encoder(dummy_input, training=False)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output d_model: {encoder.d_model}")

    # Count parameters
    encoder.summary()

    print("\n✓ CNN Encoder test passed!")
