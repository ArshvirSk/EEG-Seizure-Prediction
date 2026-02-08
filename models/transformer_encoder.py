"""
Transformer Encoder Module for Long-Range Temporal Modeling

This module implements a Transformer encoder for capturing long-range
temporal dependencies in EEG signals using multi-head self-attention.

Key components:
- Positional Encoding (sinusoidal or learnable)
- Multi-Head Self-Attention
- Feed-Forward Networks
- Layer Normalization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Optional


class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer using sinusoidal functions.

    Adds position information to input embeddings to help the model
    understand the sequential nature of the data.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 500,
        dropout_rate: float = 0.1,
        learnable: bool = False,
        **kwargs,
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate
            learnable: If True, use learnable embeddings; else sinusoidal
        """
        super().__init__(**kwargs)

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.learnable = learnable
        self.dropout = layers.Dropout(dropout_rate)

        if learnable:
            self.pos_embedding = self.add_weight(
                name="pos_embedding",
                shape=(1, max_seq_length, d_model),
                initializer="random_normal",
                trainable=True,
            )
        else:
            # Create sinusoidal positional encoding
            self.pos_encoding = self._create_sinusoidal_encoding(
                max_seq_length, d_model
            )

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> tf.Tensor:
        """Create sinusoidal positional encoding."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        return tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs, training=None):
        """
        Add positional encoding to inputs.

        Args:
            inputs: Input tensor of shape (batch, seq_len, d_model)
            training: Whether in training mode

        Returns:
            Tensor with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]

        if self.learnable:
            x = inputs + self.pos_embedding[:, :seq_len, :]
        else:
            x = inputs + self.pos_encoding[:, :seq_len, :]

        return self.dropout(x, training=training)


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention layer.

    Computes attention weights to capture relationships between
    different positions in the sequence.
    """

    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=self.d_k, dropout=dropout_rate
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None, return_attention=False):
        """
        Apply multi-head self-attention.

        Args:
            inputs: Input tensor of shape (batch, seq_len, d_model)
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention
        if return_attention:
            attn_output, attn_weights = self.mha(
                inputs, inputs, inputs, training=training, return_attention_scores=True
            )
        else:
            attn_output = self.mha(inputs, inputs, inputs, training=training)
            attn_weights = None

        # Residual connection and layer norm
        attn_output = self.dropout(attn_output, training=training)
        output = self.layernorm(inputs + attn_output)

        if return_attention:
            return output, attn_weights
        return output


class FeedForwardNetwork(layers.Layer):
    """
    Position-wise Feed-Forward Network.

    Two dense layers with ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.dense1 = layers.Dense(
            d_ff, activation="relu", kernel_initializer="he_normal"
        )
        self.dense2 = layers.Dense(d_model, kernel_initializer="he_normal")
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        """Apply feed-forward network with residual connection."""
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return self.layernorm(inputs + x)


class TransformerEncoderLayer(layers.Layer):
    """
    Single Transformer encoder layer.

    Combines multi-head self-attention and feed-forward network.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate
        )
        self.ffn = FeedForwardNetwork(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate
        )

    def call(self, inputs, training=None, return_attention=False):
        """Apply transformer encoder layer."""
        if return_attention:
            attn_output, attn_weights = self.self_attention(
                inputs, training=training, return_attention=True
            )
        else:
            attn_output = self.self_attention(inputs, training=training)
            attn_weights = None

        output = self.ffn(attn_output, training=training)

        if return_attention:
            return output, attn_weights
        return output


class TransformerEncoder(Model):
    """
    Complete Transformer Encoder.

    Stacks multiple transformer encoder layers with positional encoding.

    Input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout_rate: float = 0.1,
        max_seq_length: int = 500,
        learnable_pos_encoding: bool = False,
        **kwargs,
    ):
        """
        Initialize Transformer Encoder.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward network hidden dimension
            dropout_rate: Dropout rate
            max_seq_length: Maximum sequence length for positional encoding
            learnable_pos_encoding: Use learnable positional encoding
        """
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_layers = n_layers

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout_rate=dropout_rate,
            learnable=learnable_pos_encoding,
        )

        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                name=f"encoder_layer_{i}",
            )
            for i in range(n_layers)
        ]

    def call(self, inputs, training=None, return_attention=False):
        """
        Forward pass through transformer encoder.

        Args:
            inputs: Input tensor of shape (batch, seq_len, d_model)
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model)
            Optionally, attention weights from all layers
        """
        # Add positional encoding
        x = self.pos_encoding(inputs, training=training)

        attention_weights = []

        # Apply transformer layers
        for layer in self.encoder_layers:
            if return_attention:
                x, attn = layer(x, training=training, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, training=training)

        if return_attention:
            return x, attention_weights
        return x

    def get_attention_weights(self, inputs) -> list:
        """
        Get attention weights for visualization.

        Args:
            inputs: Input tensor

        Returns:
            List of attention weight tensors from each layer
        """
        _, attention_weights = self.call(inputs, training=False, return_attention=True)
        return attention_weights


if __name__ == "__main__":
    # Test Transformer Encoder
    print("Testing Transformer Encoder...")
    print("=" * 50)

    # Create encoder
    encoder = TransformerEncoder(
        d_model=256, n_heads=8, n_layers=4, d_ff=512, dropout_rate=0.1
    )

    # Test with dummy input
    batch_size = 4
    seq_len = 40
    d_model = 256

    dummy_input = tf.random.normal((batch_size, seq_len, d_model))
    output = encoder(dummy_input, training=False)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test attention weights
    output, attn_weights = encoder(dummy_input, training=False, return_attention=True)
    print(f"Number of attention layers: {len(attn_weights)}")
    print(f"Attention shape (per layer): {attn_weights[0].shape}")

    # Count parameters
    encoder.summary()

    print("\n✓ Transformer Encoder test passed!")
