"""
Baseline Models for Benchmark Comparison

This module implements baseline models for comparison:
1. CNNBaseline: Pure CNN model
2. LSTMBaseline: Pure LSTM model
3. CNNLSTMHybrid: CNN feature extraction + LSTM temporal modeling

These baselines help demonstrate the superiority of the CNN + Transformer approach.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple


class CNNBaseline(Model):
    """
    Pure CNN baseline for seizure prediction.

    Uses only convolutional layers without any recurrent or attention components.
    """

    def __init__(
        self,
        conv_filters: Tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: Tuple[int, ...] = (7, 5, 5, 3),
        pool_sizes: Tuple[int, ...] = (2, 2, 2, 2),
        dense_units: Tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.5,
        n_channels: int = 22,
        n_timesteps: int = 2560,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # Convolutional layers
        self.conv_layers = []
        self.bn_layers = []
        self.pool_layers = []

        for i, (filters, ks, ps) in enumerate(
            zip(conv_filters, kernel_sizes, pool_sizes)
        ):
            self.conv_layers.append(
                layers.Conv1D(
                    filters, ks, padding="same", activation="relu", name=f"conv_{i}"
                )
            )
            self.bn_layers.append(layers.BatchNormalization(name=f"bn_{i}"))
            self.pool_layers.append(
                layers.MaxPooling1D(ps, padding="same", name=f"pool_{i}")
            )

        self.dropout = layers.Dropout(dropout_rate)
        self.flatten = layers.GlobalAveragePooling1D()

        # Dense layers
        self.dense_layers = [
            layers.Dense(units, activation="relu", name=f"dense_{i}")
            for i, units in enumerate(dense_units)
        ]

        # Output
        self.output_layer = layers.Dense(1, activation="sigmoid", name="output")

    def call(self, inputs, training=None):
        # Transpose to (batch, time, channels)
        x = tf.transpose(inputs, perm=[0, 2, 1])

        # Convolutional blocks
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x, training=training)
            x = pool(x)

        x = self.dropout(x, training=training)
        x = self.flatten(x)

        # Dense layers
        for dense in self.dense_layers:
            x = dense(x)
            x = self.dropout(x, training=training)

        return self.output_layer(x)

    def build_model(self):
        dummy = tf.zeros((1, self.n_channels, self.n_timesteps))
        _ = self.call(dummy)
        return self


class LSTMBaseline(Model):
    """
    Pure LSTM baseline for seizure prediction.

    Uses bidirectional LSTM layers for sequence modeling.
    """

    def __init__(
        self,
        lstm_units: Tuple[int, ...] = (128, 64),
        dense_units: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.5,
        recurrent_dropout: float = 0.2,
        bidirectional: bool = True,
        n_channels: int = 22,
        n_timesteps: int = 2560,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            lstm = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{i}",
            )
            if bidirectional:
                lstm = layers.Bidirectional(lstm, name=f"bilstm_{i}")
            self.lstm_layers.append(lstm)

        self.dropout = layers.Dropout(dropout_rate)

        # Dense layers
        self.dense_layers = [
            layers.Dense(units, activation="relu", name=f"dense_{i}")
            for i, units in enumerate(dense_units)
        ]

        # Output
        self.output_layer = layers.Dense(1, activation="sigmoid", name="output")

    def call(self, inputs, training=None):
        # Transpose to (batch, time, channels)
        x = tf.transpose(inputs, perm=[0, 2, 1])

        # LSTM layers
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)

        x = self.dropout(x, training=training)

        # Dense layers
        for dense in self.dense_layers:
            x = dense(x)
            x = self.dropout(x, training=training)

        return self.output_layer(x)

    def build_model(self):
        dummy = tf.zeros((1, self.n_channels, self.n_timesteps))
        _ = self.call(dummy)
        return self


class CNNLSTMHybrid(Model):
    """
    CNN-LSTM Hybrid baseline for seizure prediction.

    Uses CNN for feature extraction followed by LSTM for temporal modeling.
    """

    def __init__(
        self,
        # CNN parameters
        conv_filters: Tuple[int, ...] = (64, 128),
        kernel_sizes: Tuple[int, ...] = (7, 5),
        pool_sizes: Tuple[int, ...] = (2, 2),
        # LSTM parameters
        lstm_units: Tuple[int, ...] = (64, 32),
        # Dense parameters
        dense_units: Tuple[int, ...] = (64,),
        dropout_rate: float = 0.5,
        bidirectional: bool = True,
        n_channels: int = 22,
        n_timesteps: int = 2560,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # CNN layers
        self.conv_layers = []
        self.bn_layers = []
        self.pool_layers = []

        for i, (filters, ks, ps) in enumerate(
            zip(conv_filters, kernel_sizes, pool_sizes)
        ):
            self.conv_layers.append(
                layers.Conv1D(
                    filters, ks, padding="same", activation="relu", name=f"conv_{i}"
                )
            )
            self.bn_layers.append(layers.BatchNormalization(name=f"bn_{i}"))
            self.pool_layers.append(
                layers.MaxPooling1D(ps, padding="same", name=f"pool_{i}")
            )

        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            lstm = layers.LSTM(
                units, return_sequences=return_sequences, name=f"lstm_{i}"
            )
            if bidirectional:
                lstm = layers.Bidirectional(lstm, name=f"bilstm_{i}")
            self.lstm_layers.append(lstm)

        self.dropout = layers.Dropout(dropout_rate)

        # Dense layers
        self.dense_layers = [
            layers.Dense(units, activation="relu", name=f"dense_{i}")
            for i, units in enumerate(dense_units)
        ]

        # Output
        self.output_layer = layers.Dense(1, activation="sigmoid", name="output")

    def call(self, inputs, training=None):
        # Transpose to (batch, time, channels)
        x = tf.transpose(inputs, perm=[0, 2, 1])

        # CNN feature extraction
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x, training=training)
            x = pool(x)

        # LSTM temporal modeling
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)

        x = self.dropout(x, training=training)

        # Dense layers
        for dense in self.dense_layers:
            x = dense(x)
            x = self.dropout(x, training=training)

        return self.output_layer(x)

    def build_model(self):
        dummy = tf.zeros((1, self.n_channels, self.n_timesteps))
        _ = self.call(dummy)
        return self


def create_baseline(model_type: str, **kwargs):
    """
    Factory function to create baseline models.

    Args:
        model_type: 'cnn', 'lstm', or 'cnn_lstm'
        **kwargs: Model parameters

    Returns:
        Built baseline model
    """
    models = {"cnn": CNNBaseline, "lstm": LSTMBaseline, "cnn_lstm": CNNLSTMHybrid}

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )

    model = models[model_type](**kwargs)
    model.build_model()
    return model


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...")
    print("=" * 60)

    batch_size = 4
    n_channels = 22
    n_timesteps = 2560

    dummy_input = tf.random.normal((batch_size, n_channels, n_timesteps))

    # Test CNN Baseline
    print("\n--- CNN Baseline ---")
    cnn_model = create_baseline("cnn", n_channels=n_channels, n_timesteps=n_timesteps)
    cnn_output = cnn_model(dummy_input, training=False)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {cnn_output.shape}")
    print(f"Parameters: {cnn_model.count_params():,}")

    # Test LSTM Baseline
    print("\n--- LSTM Baseline ---")
    lstm_model = create_baseline("lstm", n_channels=n_channels, n_timesteps=n_timesteps)
    lstm_output = lstm_model(dummy_input, training=False)
    print(f"Output shape: {lstm_output.shape}")
    print(f"Parameters: {lstm_model.count_params():,}")

    # Test CNN-LSTM Hybrid
    print("\n--- CNN-LSTM Hybrid ---")
    hybrid_model = create_baseline(
        "cnn_lstm", n_channels=n_channels, n_timesteps=n_timesteps
    )
    hybrid_output = hybrid_model(dummy_input, training=False)
    print(f"Output shape: {hybrid_output.shape}")
    print(f"Parameters: {hybrid_model.count_params():,}")

    print("\n✓ All baseline tests passed!")
