"""
Models package for EEG seizure prediction.
Contains CNN encoder, Transformer encoder, and complete predictor models.
"""

try:
    from .cnn_encoder import CNNEncoder
    from .transformer_encoder import TransformerEncoder, PositionalEncoding
    from .seizure_predictor import SeizurePredictorCNNTransformer
    from .baselines import CNNBaseline, LSTMBaseline, CNNLSTMHybrid
except ImportError:
    # Handle case when running as script
    pass

__all__ = [
    "CNNEncoder",
    "TransformerEncoder",
    "PositionalEncoding",
    "SeizurePredictorCNNTransformer",
    "CNNBaseline",
    "LSTMBaseline",
    "CNNLSTMHybrid",
]
