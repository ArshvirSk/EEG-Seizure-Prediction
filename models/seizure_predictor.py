"""
Main Seizure Predictor Model: CNN + Transformer Architecture

This module implements the complete seizure prediction model that combines:
1. CNN Encoder for local spatial-temporal feature extraction
2. Transformer Encoder for long-range temporal dependency modeling
3. Classification Head for binary prediction (preictal vs interictal)

Architecture:
    Input (channels × time)
        → CNN Encoder
        → Transformer Encoder
        → Global Pooling
        → Dense Layers
        → Sigmoid Output
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional

try:
    from .cnn_encoder import CNNEncoder
    from .transformer_encoder import TransformerEncoder
except ImportError:
    from cnn_encoder import CNNEncoder
    from transformer_encoder import TransformerEncoder


class ClassificationHead(layers.Layer):
    """
    Classification head for binary seizure prediction.

    Applies global pooling followed by dense layers with dropout.
    """

    def __init__(
        self,
        hidden_units: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.5,
        n_classes: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling1D()

        # Dense layers
        self.dense_layers = []
        self.dropout_layers = []

        for i, units in enumerate(hidden_units):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation="relu",
                    kernel_initializer="he_normal",
                    name=f"dense_{i}",
                )
            )
            self.dropout_layers.append(
                layers.Dropout(
                    dropout_rate * (0.6 if i > 0 else 1.0)
                )  # Less dropout in later layers
            )

        # Output layer
        self.output_layer = layers.Dense(
            n_classes,
            activation="sigmoid" if n_classes == 1 else "softmax",
            name="output",
        )

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, seq_len, d_model)
            training: Whether in training mode

        Returns:
            Prediction tensor of shape (batch, n_classes)
        """
        # Global pooling
        x = self.global_pool(inputs)

        # Dense layers with dropout
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)

        # Output
        return self.output_layer(x)


class SeizurePredictorCNNTransformer(Model):
    """
    Complete CNN + Transformer model for EEG seizure prediction.

    This model combines:
    1. CNN Encoder: Extracts local spatial-temporal patterns from EEG
    2. Transformer Encoder: Captures long-range temporal dependencies
    3. Classification Head: Outputs seizure probability

    Input: EEG window of shape (batch, n_channels, n_timesteps)
    Output: Preictal probability of shape (batch, 1)
    """

    def __init__(
        self,
        # CNN parameters
        conv_filters: Tuple[int, ...] = (64, 128, 256),
        conv_kernel_sizes: Tuple[int, ...] = (7, 5, 3),
        pool_sizes: Tuple[int, ...] = (2, 2, 2),
        cnn_dropout: float = 0.3,
        # Transformer parameters
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        d_ff: int = 512,
        transformer_dropout: float = 0.1,
        max_seq_length: int = 500,
        # Classifier parameters
        classifier_hidden: Tuple[int, ...] = (128, 64),
        classifier_dropout: float = 0.5,
        # Input parameters
        n_channels: int = 22,
        n_timesteps: int = 2560,
        **kwargs,
    ):
        """
        Initialize the CNN-Transformer seizure predictor.

        Args:
            conv_filters: Number of filters for each CNN layer
            conv_kernel_sizes: Kernel sizes for each CNN layer
            pool_sizes: Pooling sizes for each CNN layer
            cnn_dropout: Dropout rate for CNN
            n_heads: Number of attention heads
            n_transformer_layers: Number of transformer layers
            d_ff: Feed-forward dimension in transformer
            transformer_dropout: Dropout rate for transformer
            max_seq_length: Maximum sequence length for positional encoding
            classifier_hidden: Hidden units for classification head
            classifier_dropout: Dropout rate for classifier
            n_channels: Number of EEG channels
            n_timesteps: Number of time steps per window
        """
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # CNN Encoder
        self.cnn_encoder = CNNEncoder(
            conv_filters=conv_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            pool_sizes=pool_sizes,
            dropout_rate=cnn_dropout,
            use_batch_norm=True,
            input_channels_last=False,
        )

        # Get d_model from CNN output
        d_model = conv_filters[-1]

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            d_ff=d_ff,
            dropout_rate=transformer_dropout,
            max_seq_length=max_seq_length,
        )

        # Classification Head
        self.classifier = ClassificationHead(
            hidden_units=classifier_hidden, dropout_rate=classifier_dropout, n_classes=1
        )

    def call(self, inputs, training=None, return_attention=False):
        """
        Forward pass.

        Args:
            inputs: EEG tensor of shape (batch, n_channels, n_timesteps)
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Prediction tensor of shape (batch, 1)
            Optionally, attention weights
        """
        # CNN feature extraction
        cnn_features = self.cnn_encoder(inputs, training=training)

        # Transformer temporal modeling
        if return_attention:
            transformer_output, attention_weights = self.transformer_encoder(
                cnn_features, training=training, return_attention=True
            )
        else:
            transformer_output = self.transformer_encoder(
                cnn_features, training=training
            )

        # Classification
        predictions = self.classifier(transformer_output, training=training)

        if return_attention:
            return predictions, attention_weights
        return predictions

    def get_attention_maps(self, inputs) -> list:
        """
        Get attention weights for interpretability.

        Args:
            inputs: EEG tensor

        Returns:
            List of attention weight tensors
        """
        _, attention_weights = self.call(inputs, training=False, return_attention=True)
        return attention_weights

    def build_model(self):
        """Build the model by running a forward pass with dummy data."""
        dummy_input = tf.zeros((1, self.n_channels, self.n_timesteps))
        _ = self.call(dummy_input, training=False)
        return self

    def summary_custom(self):
        """Print a custom summary of the model architecture."""
        print("\n" + "=" * 60)
        print("SeizurePredictorCNNTransformer Architecture")
        print("=" * 60)

        print(
            f"\nInput Shape: (batch, {self.n_channels} channels, {self.n_timesteps} timesteps)"
        )

        print("\n--- CNN Encoder ---")
        print(f"  Output dimension: {self.cnn_encoder.d_model}")

        print("\n--- Transformer Encoder ---")
        print(f"  d_model: {self.transformer_encoder.d_model}")
        print(f"  Layers: {self.transformer_encoder.n_layers}")

        print("\n--- Classification Head ---")
        print(f"  Hidden units: {self.classifier.hidden_units}")
        print(f"  Output: 1 (sigmoid)")

        # Get output shape
        dummy_input = tf.zeros((1, self.n_channels, self.n_timesteps))
        output = self.call(dummy_input, training=False)
        print(f"\nOutput Shape: {output.shape}")

        # Count parameters
        self.summary()


def create_model(
    n_channels: int = 22, n_timesteps: int = 2560, **kwargs
) -> SeizurePredictorCNNTransformer:
    """
    Factory function to create and build the model.

    Args:
        n_channels: Number of EEG channels
        n_timesteps: Number of time steps per window
        **kwargs: Additional model parameters

    Returns:
        Built SeizurePredictorCNNTransformer model
    """
    model = SeizurePredictorCNNTransformer(
        n_channels=n_channels, n_timesteps=n_timesteps, **kwargs
    )
    model.build_model()
    return model


if __name__ == "__main__":
    # Test the complete model
    print("Testing SeizurePredictorCNNTransformer...")
    print("=" * 60)

    # Create model
    model = create_model(
        n_channels=22,
        n_timesteps=2560,
        conv_filters=(64, 128, 256),
        n_heads=8,
        n_transformer_layers=4,
    )

    # Test forward pass
    batch_size = 4
    dummy_input = tf.random.normal((batch_size, 22, 2560))

    output = model(dummy_input, training=False)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.numpy().flatten()}")

    # Test with attention
    output, attention = model(dummy_input, training=False, return_attention=True)
    print(f"\nAttention layers: {len(attention)}")
    print(f"Attention shape: {attention[0].shape}")

    # Print model summary
    model.summary_custom()

    print("\n✓ SeizurePredictorCNNTransformer test passed!")
